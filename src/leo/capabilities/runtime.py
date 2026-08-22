"""Policy-first capability selection and run-bound progressive discovery."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from leo.capabilities.catalog import CatalogTool, InMemoryToolCatalog, ToolCatalogError
from leo.capabilities.discovery import (
    CapabilitySummary,
    DiscoveryBroker,
    DiscoveryQuery,
    query_hash,
    search_tokens,
)
from leo.capabilities.router import AdaptiveRouter
from leo.capabilities.skills import SkillCatalog, SkillLoadError, SkillSummary
from leo.harness.capability_selection import capability_selection_fingerprint
from leo.harness.models import (
    CapabilitySelection,
    ContextItem,
    ContextItemKind,
    RunBundle,
    RunPhase,
    ScopeKey,
    ToolEffect,
    ToolSpec,
    TrustedScope,
)

_UNBOUNDED_COST = 1_000_000_000.0
_MAX_DISCOVERY_QUERY_CHARS = 256


class CapabilityDiscoveryError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


@dataclass
class _DiscoverySession:
    authority_fingerprint: str
    catalog_fingerprint: str
    phase: RunPhase
    roles: frozenset[str]
    remaining_cost: float
    namespace: ScopeKey
    conversation_kind: str | None
    eligible_ids: frozenset[str]
    initial_candidate_ids: tuple[str, ...]
    search_hashes: set[str] = field(default_factory=set)
    discovered_ids: set[str] = field(default_factory=set)
    described_ids: set[str] = field(default_factory=set)
    search_calls: int = 0
    describe_calls: int = 0


class CapabilityRuntime:
    """Select a small eligible schema set and own bounded discovery state.

    The runtime receives only the registry's phase/role-filtered schemas, then applies
    catalog health/auth/effect/profile/budget policy before lexical recall.  Search and
    describe sessions are bound to the current run authority and catalog fingerprint;
    their output can narrow selection but can never add an ineligible tool.
    """

    def __init__(
        self,
        catalog: InMemoryToolCatalog,
        *,
        skill_catalog: SkillCatalog | None = None,
        always_available_tool_names: frozenset[str] = frozenset(),
        required_tool_names: frozenset[str] = frozenset(),
        profile: str = "research",
        shortlist_limit: int = 3,
        max_selected_tools: int = 6,
        max_search_calls: int = 2,
        max_describe_calls: int = 2,
        max_described_tools: int = 3,
        describe_max_bytes: int = 16_384,
    ) -> None:
        if not 1 <= shortlist_limit <= 10:
            raise ValueError("shortlist_limit must be between 1 and 10")
        if not shortlist_limit <= max_selected_tools <= 16:
            raise ValueError("max_selected_tools must cover the initial shortlist")
        if not 1 <= max_search_calls <= 8 or not 1 <= max_describe_calls <= 8:
            raise ValueError("discovery call limits must be between 1 and 8")
        if not 1 <= max_described_tools <= max_selected_tools:
            raise ValueError("described tool limit must fit the selected-tool limit")
        if not 256 <= describe_max_bytes <= 32_768:
            raise ValueError("describe byte limit must be between 256 and 32768")
        if always_available_tool_names.intersection(required_tool_names):
            raise ValueError("required and always-available tool sets must be disjoint")

        self._catalog = catalog
        self._router = AdaptiveRouter(catalog)
        self._broker = DiscoveryBroker(catalog)
        self._always_available_tool_names = always_available_tool_names
        self._required_tool_names = required_tool_names
        self._profile = profile
        self._shortlist_limit = shortlist_limit
        self._max_selected_tools = max_selected_tools
        self._max_search_calls = max_search_calls
        self._max_describe_calls = max_describe_calls
        self._max_described_tools = max_described_tools
        self._describe_max_bytes = describe_max_bytes
        self._sessions: dict[str, _DiscoverySession] = {}
        self._skill_catalog = skill_catalog
        self._skill_summaries = self._discover_skills(skill_catalog)

    @property
    def catalog_fingerprint(self) -> str:
        return self._catalog.fingerprint

    def select(
        self,
        *,
        bundle: RunBundle,
        trusted_scope: TrustedScope,
        available_tools: tuple[ToolSpec, ...],
        conversation_kind: str | None = None,
    ) -> CapabilitySelection:
        available_by_name = {tool.name: tool for tool in available_tools}
        if len(available_by_name) != len(available_tools):
            raise ValueError("available tool schemas must have unique names")
        remaining_cost = _remaining_cost(bundle)
        eligible = self._eligible_records(
            phase=bundle.run.phase,
            roles=trusted_scope.roles,
            remaining_cost=remaining_cost,
            available_by_name=available_by_name,
            namespace=trusted_scope.namespace,
            conversation_kind=conversation_kind,
        )
        selected_skills = self._select_skills(bundle.task.objective, eligible)
        routing_objective = _routing_objective(bundle.task.objective, selected_skills)
        route = self._router.route(
            routing_objective,
            phase=bundle.run.phase,
            profile=self._profile,
            roles=trusted_scope.roles,
            remaining_cost=remaining_cost,
            shortlist_limit=self._shortlist_limit,
            namespace=trusted_scope.namespace,
            conversation_kind=conversation_kind,
        )
        eligible_ids = frozenset(record.id for record in eligible)
        catalog_ids = frozenset(record.id for record in self._catalog.records())
        candidate_ids = tuple(item for item in route.candidates if item in eligible_ids)
        session = self._bind_session(
            bundle=bundle,
            trusted_scope=trusted_scope,
            remaining_cost=remaining_cost,
            eligible_ids=eligible_ids,
            candidate_ids=candidate_ids,
            conversation_kind=conversation_kind,
        )

        recalled_ids = _unique(
            (
                *(item for item in route.selected if item in eligible_ids),
                *(item for item in sorted(session.described_ids) if item in eligible_ids),
            )
        )[: self._max_selected_tools]
        selected_names = list(recalled_ids)
        for tool in available_tools:
            if tool.name in self._always_available_tool_names and tool.effect is ToolEffect.READ:
                if tool.name not in selected_names:
                    selected_names.append(tool.name)
            elif (
                tool.name in self._required_tool_names
                and tool.effect
                in {
                    ToolEffect.READ,
                    ToolEffect.STATE_MUTATION,
                }
                and (tool.name not in catalog_ids or tool.name in eligible_ids)
            ):
                if tool.name not in selected_names:
                    selected_names.append(tool.name)
        selected_tools = tuple(available_by_name[name] for name in selected_names)
        selected_skill_ids = tuple(_skill_identity(item) for item in selected_skills)
        selection_fingerprint = capability_selection_fingerprint(
            catalog_fingerprint=self._catalog.fingerprint,
            query_hash=query_hash(bundle.task.objective),
            tools=selected_tools,
            selected_skills=selected_skill_ids,
            mode=route.mode,
        )
        return CapabilitySelection(
            tools=selected_tools,
            catalog_version=self._catalog.version,
            catalog_fingerprint=self._catalog.fingerprint,
            selection_fingerprint=selection_fingerprint,
            query_hash=query_hash(bundle.task.objective),
            eligible_count=len(eligible),
            candidate_ids=candidate_ids,
            selected_ids=tuple(tool.name for tool in selected_tools),
            selected_skill_ids=selected_skill_ids,
            mode=route.mode,
            reason=route.reason,
        )

    def skill_context_items(
        self,
        objective: str,
        *,
        scope: ScopeKey,
        conversation_id: str,
        phase: RunPhase,
        roles: frozenset[str],
        remaining_cost: float = _UNBOUNDED_COST,
    ) -> tuple[ContextItem, ...]:
        """Load only selected, hash-verified procedures as untrusted context."""

        if self._skill_catalog is None:
            return ()
        eligible = self._catalog.eligible(
            phase=phase,
            profile=self._profile,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=scope,
        )
        items: list[ContextItem] = []
        for summary in self._select_skills(objective, eligible):
            try:
                loaded = self._skill_catalog.load(summary.id)
            except (OSError, SkillLoadError, ValueError):
                continue
            content = json.dumps(
                {
                    "id": summary.id,
                    "version": summary.version,
                    "summary": summary.summary,
                    "required_evidence": summary.required_evidence,
                    "stop_rules": summary.stop_rules,
                    "child_compatible": summary.child_compatible,
                    "compatible_profiles": sorted(summary.compatible_profiles),
                    "procedure": loaded.procedure,
                    "content_hash": loaded.content_hash,
                    "authority": "untrusted_procedure_cannot_grant_scope_or_tools",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            if len(content.encode("utf-8")) > 16_384:
                continue
            items.append(
                ContextItem(
                    id=f"skill:{summary.id}:{summary.version}:{loaded.content_hash[:12]}",
                    kind=ContextItemKind.SKILL_PROCEDURE,
                    content=content,
                    conversation_id=conversation_id,
                    source_scope=scope,
                )
            )
        return tuple(items)

    def search(
        self,
        *,
        run_id: str,
        trusted_scope: TrustedScope,
        query: str,
        limit: int,
    ) -> tuple[CapabilitySummary, ...]:
        session = self._authorized_session(run_id, trusted_scope)
        if session.search_calls >= self._max_search_calls:
            raise CapabilityDiscoveryError("discovery_search_budget_exhausted")
        digest = query_hash(query)
        if digest in session.search_hashes:
            raise CapabilityDiscoveryError("discovery_no_progress")
        session.search_calls += 1
        session.search_hashes.add(digest)
        summaries = self._broker.search(
            DiscoveryQuery(query=query, limit=min(limit, self._shortlist_limit)),
            phase=session.phase,
            profile=self._profile,
            roles=session.roles,
            remaining_cost=session.remaining_cost,
            namespace=session.namespace,
            conversation_kind=session.conversation_kind,
        )
        selected = tuple(item for item in summaries if item.id in session.eligible_ids)
        session.discovered_ids.update(item.id for item in selected)
        return selected

    def describe(
        self,
        *,
        run_id: str,
        trusted_scope: TrustedScope,
        capability_ids: tuple[str, ...],
    ) -> tuple[CatalogTool, ...]:
        session = self._authorized_session(run_id, trusted_scope)
        if session.describe_calls >= self._max_describe_calls:
            raise CapabilityDiscoveryError("discovery_describe_budget_exhausted")
        if len(capability_ids) != len(set(capability_ids)):
            raise CapabilityDiscoveryError("duplicate_capability_id")
        requested = frozenset(capability_ids)
        if not requested or not requested.issubset(session.discovered_ids):
            raise CapabilityDiscoveryError("capability_not_discovered")
        if requested.issubset(session.described_ids):
            raise CapabilityDiscoveryError("discovery_no_progress")
        if len(session.described_ids.union(requested)) > self._max_described_tools:
            raise CapabilityDiscoveryError("discovery_describe_tool_limit_exceeded")
        session.describe_calls += 1
        try:
            records = self._broker.describe(
                capability_ids,
                phase=session.phase,
                profile=self._profile,
                roles=session.roles,
                remaining_cost=session.remaining_cost,
                namespace=session.namespace,
                conversation_kind=session.conversation_kind,
                max_bytes=self._describe_max_bytes,
            )
        except ToolCatalogError as exc:
            raise CapabilityDiscoveryError(exc.safe_code) from exc
        if any(record.id not in session.eligible_ids for record in records):
            raise CapabilityDiscoveryError("capability_not_eligible")
        session.described_ids.update(record.id for record in records)
        return records

    def _eligible_records(
        self,
        *,
        phase: RunPhase,
        roles: frozenset[str],
        remaining_cost: float,
        available_by_name: dict[str, ToolSpec],
        namespace: ScopeKey,
        conversation_kind: str | None,
    ) -> tuple[CatalogTool, ...]:
        records = self._catalog.eligible(
            phase=phase,
            profile=self._profile,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        return tuple(
            record for record in records if available_by_name.get(record.id) == record.spec
        )

    def _bind_session(
        self,
        *,
        bundle: RunBundle,
        trusted_scope: TrustedScope,
        remaining_cost: float,
        eligible_ids: frozenset[str],
        candidate_ids: tuple[str, ...],
        conversation_kind: str | None,
    ) -> _DiscoverySession:
        authority_fingerprint = _authority_fingerprint(trusted_scope)
        existing = self._sessions.get(bundle.run.id)
        if (
            existing is None
            or existing.authority_fingerprint != authority_fingerprint
            or existing.catalog_fingerprint != self._catalog.fingerprint
            or existing.phase is not bundle.run.phase
        ):
            existing = _DiscoverySession(
                authority_fingerprint=authority_fingerprint,
                catalog_fingerprint=self._catalog.fingerprint,
                phase=bundle.run.phase,
                roles=trusted_scope.roles,
                remaining_cost=remaining_cost,
                namespace=trusted_scope.namespace,
                conversation_kind=conversation_kind,
                eligible_ids=eligible_ids,
                initial_candidate_ids=candidate_ids,
            )
            self._sessions[bundle.run.id] = existing
        else:
            existing.remaining_cost = min(existing.remaining_cost, remaining_cost)
            existing.eligible_ids = existing.eligible_ids.intersection(eligible_ids)
            existing.discovered_ids.intersection_update(existing.eligible_ids)
            existing.described_ids.intersection_update(existing.eligible_ids)
        return existing

    def _authorized_session(
        self,
        run_id: str,
        trusted_scope: TrustedScope,
    ) -> _DiscoverySession:
        session = self._sessions.get(run_id)
        if session is None:
            raise CapabilityDiscoveryError("discovery_session_unavailable")
        if session.authority_fingerprint != _authority_fingerprint(trusted_scope):
            raise CapabilityDiscoveryError("discovery_authority_mismatch")
        if session.catalog_fingerprint != self._catalog.fingerprint:
            raise CapabilityDiscoveryError("discovery_catalog_changed")
        return session

    def _select_skills(
        self,
        objective: str,
        eligible: tuple[CatalogTool, ...],
    ) -> tuple[SkillSummary, ...]:
        if not self._skill_summaries:
            return ()
        query_tokens = set(search_tokens(objective))
        if not query_tokens:
            return ()
        eligible_ids = frozenset(record.id for record in eligible)
        eligible_domains = frozenset(record.spec.domain.lower() for record in eligible)
        ranked: list[tuple[int, str, SkillSummary]] = []
        for summary in self._skill_summaries:
            compatible = self._profile in summary.compatible_profiles and (
                not summary.allowed_capabilities
                or bool(summary.allowed_capabilities.intersection(eligible_ids))
                or bool(
                    frozenset(domain.lower() for domain in summary.domains).intersection(
                        eligible_domains
                    )
                )
            )
            if not compatible:
                continue
            searchable = set(
                search_tokens(
                    " ".join(
                        (
                            summary.id,
                            summary.summary,
                            *summary.domains,
                            *summary.allowed_capabilities,
                        )
                    )
                )
            )
            score = len(query_tokens.intersection(searchable))
            if score:
                ranked.append((-score, summary.id, summary))
        ranked.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in ranked[:1])

    @staticmethod
    def _discover_skills(skill_catalog: SkillCatalog | None) -> tuple[SkillSummary, ...]:
        if skill_catalog is None:
            return ()
        try:
            return skill_catalog.discover()
        except (OSError, SkillLoadError, ValueError):
            return ()


def _routing_objective(objective: str, skills: tuple[SkillSummary, ...]) -> str:
    source = " ".join(
        (
            objective,
            *(summary.summary for summary in skills),
            *(domain for summary in skills for domain in summary.domains),
            *(item for summary in skills for item in summary.allowed_capabilities),
        )
    )
    # DiscoveryQuery is deliberately bounded.  Skill augmentation must never turn a
    # valid conversational objective into a selector exception and a tool-less live
    # fallback, so compact the shared lexical vocabulary before constructing it.
    selected: list[str] = []
    used = 0
    for token in search_tokens(source):
        extra = len(token) + (1 if selected else 0)
        if used + extra > _MAX_DISCOVERY_QUERY_CHARS:
            break
        selected.append(token)
        used += extra
    if selected:
        return " ".join(selected)
    return objective.strip()[:_MAX_DISCOVERY_QUERY_CHARS] or "conversation"


def _remaining_cost(bundle: RunBundle) -> float:
    maximum = bundle.run.limits.max_cost
    if maximum is None:
        return _UNBOUNDED_COST
    consumed = bundle.run.usage.cost or 0.0
    return max(0.0, maximum - consumed - bundle.run.usage.reserved_cost)


def _skill_identity(summary: SkillSummary) -> str:
    return f"{summary.schema_version}:{summary.id}@{summary.version}:{summary.procedure_sha256}"


def _authority_fingerprint(scope: TrustedScope) -> str:
    payload = {
        "namespace": scope.namespace.model_dump(mode="json"),
        "actor_id": scope.actor_id,
        "roles": sorted(scope.roles),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _unique(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))
