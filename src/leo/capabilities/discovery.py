"""Bounded lexical capability discovery over the already eligible catalog."""

from __future__ import annotations

import re
from hashlib import sha256

from pydantic import Field

from leo.capabilities.catalog import CatalogTool, InMemoryToolCatalog, ToolCatalogError
from leo.harness.models import ContractModel, NonEmptyStr, RunPhase, ScopeKey

_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "can",
        "could",
        "for",
        "from",
        "get",
        "give",
        "help",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "one",
        "please",
        "read",
        "show",
        "the",
        "this",
        "to",
        "tool",
        "use",
        "what",
        "with",
        "would",
        "you",
    }
)
_ALIASES = (
    frozenset({"quote", "price", "pricing"}),
    frozenset(
        {
            "coin",
            "crypto",
            "cryptocurrency",
            "market-data",
            "token",
        }
    ),
    frozenset({"bitcoin", "btc"}),
    frozenset({"ethereum", "eth", "ether"}),
    frozenset(
        {
            "agreement",
            "compare",
            "corroborate",
            "corroboration",
            "cross-check",
            "divergence",
            "redundancy",
        }
    ),
    frozenset({"current", "latest", "recent", "now"}),
    frozenset({"filing", "filings", "disclosure", "disclosures", "10-k", "10-q"}),
    frozenset({"company", "profile", "identity", "listing", "exchange", "industry"}),
    frozenset(
        {
            "changed",
            "development",
            "developments",
            "happened",
            "headline",
            "headlines",
            "news",
            "story",
            "stories",
            "update",
            "updates",
        }
    ),
    frozenset({"earnings", "eps", "surprise", "surprises", "estimate", "actual"}),
    frozenset(
        {"beta", "financial", "financials", "fundamental", "fundamentals", "metric", "metrics"}
    ),
    frozenset({"web", "internet", "online", "url", "page", "website"}),
    frozenset({"delegate", "delegated", "subagent", "subagents"}),
    frozenset({"plan", "compare", "comparison", "multi-source", "synthesize"}),
)


class DiscoveryQuery(ContractModel):
    query: NonEmptyStr = Field(max_length=256)
    limit: int = Field(default=5, ge=1, le=20)
    max_bytes: int = Field(default=4096, ge=256, le=32_768)


class CapabilitySummary(ContractModel):
    id: NonEmptyStr
    version: NonEmptyStr
    short_description: NonEmptyStr
    tags: frozenset[NonEmptyStr]
    schema_fingerprint: NonEmptyStr
    score: float = Field(ge=0)


class DiscoveryBroker:
    def __init__(self, catalog: InMemoryToolCatalog) -> None:
        self._catalog = catalog

    def search(
        self,
        request: DiscoveryQuery,
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
    ) -> tuple[CapabilitySummary, ...]:
        tokens = _tokens(request.query)
        if not tokens:
            return ()
        eligible = self._catalog.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        ranked: list[CapabilitySummary] = []
        for record in eligible:
            searchable = _tokens(
                " ".join(
                    (
                        record.id,
                        record.spec.domain,
                        record.short_description,
                        *record.tags,
                    )
                )
            )
            matched = len(set(tokens).intersection(searchable))
            if matched == 0:
                continue
            ranked.append(
                CapabilitySummary(
                    id=record.id,
                    version=record.semantic_version,
                    short_description=record.short_description,
                    tags=record.tags,
                    schema_fingerprint=record.schema_fingerprint,
                    score=matched / len(tokens),
                )
            )
        ranked.sort(key=lambda item: (-item.score, item.id, item.version))
        return tuple(ranked[: request.limit])

    def describe(
        self,
        capability_ids: tuple[str, ...],
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
        max_bytes: int = 16_384,
    ) -> tuple[CatalogTool, ...]:
        eligible = self._catalog.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        by_id = {record.id: record for record in eligible}
        records: list[CatalogTool] = []
        used_bytes = 0
        for capability_id in capability_ids:
            record = by_id.get(capability_id)
            if record is None:
                raise ToolCatalogError("capability_not_eligible")
            size = len(record.model_dump_json().encode("utf-8"))
            if used_bytes + size > max_bytes:
                raise ToolCatalogError("describe_budget_exceeded")
            used_bytes += size
            records.append(record)
        return tuple(records)


def _tokens(value: str) -> tuple[str, ...]:
    raw = tuple(
        token
        for token in re.findall(r"[a-z0-9_.-]{2,64}", value.lower())
        if token not in _STOP_TOKENS
    )
    expanded: list[str] = list(raw)
    for token in raw:
        expanded.extend(
            part
            for part in re.split(r"[._-]+", token)
            if len(part) >= 2 and part not in _STOP_TOKENS
        )
    for token in tuple(dict.fromkeys(expanded)):
        singular = token[:-1] if token.endswith("s") and len(token) > 3 else token
        if singular != token:
            expanded.append(singular)
        for group in _ALIASES:
            if token in group or singular in group:
                expanded.extend(group)
    return tuple(dict.fromkeys(expanded))


def query_hash(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()


def search_tokens(value: str) -> tuple[str, ...]:
    """Return the deterministic lexical vocabulary shared by routing and skills."""

    return _tokens(value)
