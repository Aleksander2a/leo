"""Deterministic scope-first lexical retrieval baseline before vector search."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import datetime

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.models import MemoryRevision, MemoryStatus, MemoryVisibility


class ScopedMemoryCandidate(ContractModel):
    scope: ScopeKey
    revision: MemoryRevision
    record_status: MemoryStatus = MemoryStatus.ACTIVE
    current_revision: int | None = Field(default=None, ge=1)


class AuthorizedMemoryNamespace(ContractModel):
    """One exact visibility/namespace pair authorized by trusted access state."""

    visibility: MemoryVisibility
    namespace_id: NonEmptyStr

    @model_validator(mode="after")
    def conversation_native_authority_only(self) -> AuthorizedMemoryNamespace:
        if self.visibility in {
            MemoryVisibility.STRATEGY_SHARED,
            MemoryVisibility.ORGANIZATION_SHARED,
        }:
            raise ValueError("retrieval authority must be conversation-native")
        return self


class MemorySearchRequest(ContractModel):
    scope: ScopeKey
    query: NonEmptyStr
    authorized_namespaces: frozenset[AuthorizedMemoryNamespace] = Field(min_length=1)
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    max_sensitivity: float = Field(default=1, ge=0, le=1)
    as_of: datetime
    limit: int = Field(default=10, ge=1, le=100)
    per_namespace_limit: int | None = Field(default=None, ge=1, le=50)
    # Deliberately excluded from the retrieval cache key (see RetrievalCacheKey):
    # it is a deterministic function of `query`, so a text-based cache hit already
    # implies the same vector. Absent entirely, hybrid search degrades to lexical-only.
    query_embedding: tuple[float, ...] | None = None


class MemorySearchHit(ContractModel):
    record_id: NonEmptyStr
    revision: int = Field(ge=1)
    content: NonEmptyStr
    score: float = Field(ge=0)
    match_reason: NonEmptyStr
    recorded_at: datetime
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    lifecycle_status: MemoryStatus
    conflict_group_id: str | None = None


class MemorySearchTrace(ContractModel):
    policy_version: NonEmptyStr = "fts-scope-first-v2"
    candidate_count: int = Field(ge=0)
    authorized_current_count: int = Field(ge=0)
    lexical_match_count: int = Field(ge=0)
    selected_count: int = Field(ge=0)
    selected_ids: tuple[NonEmptyStr, ...] = ()
    query_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class MemorySearchResult(ContractModel):
    hits: tuple[MemorySearchHit, ...]
    trace: MemorySearchTrace


class MemoryRetrievalError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def search_memory(
    candidates: tuple[ScopedMemoryCandidate, ...], request: MemorySearchRequest
) -> tuple[MemorySearchHit, ...]:
    """Apply every trusted filter before scoring, ordering, or limiting candidates."""

    return search_memory_with_trace(candidates, request).hits


def search_memory_with_trace(
    candidates: tuple[ScopedMemoryCandidate, ...], request: MemorySearchRequest
) -> MemorySearchResult:
    """Return authorized current hits plus a content-free, replayable selection trace."""

    tokens = tuple(normalize_memory_query(request.query).split())
    authorized_pairs = {
        (item.visibility, item.namespace_id) for item in request.authorized_namespaces
    }
    current_candidates: dict[str, list[ScopedMemoryCandidate]] = defaultdict(list)
    for candidate in candidates:
        revision = candidate.revision
        # strategy_id is optional domain provenance under D-054, never a disclosure gate.
        # The workspace plus exact authorized visibility/namespace pair is authoritative.
        if candidate.scope.organization_id != request.scope.organization_id:
            continue
        if candidate.record_status not in {MemoryStatus.ACTIVE, MemoryStatus.CONTESTED}:
            continue
        if revision.status not in {MemoryStatus.ACTIVE, MemoryStatus.CONTESTED}:
            continue
        if candidate.current_revision is not None and revision.number != candidate.current_revision:
            continue
        if (revision.visibility, revision.namespace_id) not in authorized_pairs:
            continue
        if revision.sensitivity > request.max_sensitivity:
            continue
        if revision.valid_from > request.as_of:
            continue
        if revision.valid_until is not None and request.as_of >= revision.valid_until:
            continue
        if revision.expires_at is not None and request.as_of >= revision.expires_at:
            continue
        current_candidates[revision.record_id].append(candidate)

    current: list[ScopedMemoryCandidate] = []
    for record_candidates in current_candidates.values():
        # A missing current-revision hint is tolerated only for deterministic fixtures. If
        # multiple revisions are present, the highest append-only revision is current.
        current.append(
            max(record_candidates, key=lambda item: (item.revision.number, item.revision.id))
        )

    eligible: list[MemorySearchHit] = []
    for candidate in current:
        revision = candidate.revision
        searchable_tokens = frozenset(_query_tokens(f"{revision.content} {revision.reason}"))
        if not set(tokens).issubset(searchable_tokens):
            continue
        eligible.append(
            MemorySearchHit(
                record_id=revision.record_id,
                revision=revision.number,
                content=revision.content,
                score=1,
                match_reason=f"lexical_all:{len(tokens)}/{len(tokens)}",
                recorded_at=revision.recorded_at,
                visibility=revision.visibility,
                namespace_id=revision.namespace_id,
                source_ids=revision.source_ids,
                lifecycle_status=(
                    MemoryStatus.CONTESTED
                    if MemoryStatus.CONTESTED in {candidate.record_status, revision.status}
                    else MemoryStatus.ACTIVE
                ),
                conflict_group_id=conflict_group_id(revision),
            )
        )
    eligible.sort(key=lambda hit: (-hit.score, -hit.recorded_at.timestamp(), hit.record_id))
    selected = _select_with_source_and_conflict_budgets(eligible, request)
    return MemorySearchResult(
        hits=selected,
        trace=MemorySearchTrace(
            candidate_count=len(candidates),
            authorized_current_count=len(current),
            lexical_match_count=len(eligible),
            selected_count=len(selected),
            selected_ids=tuple(hit.record_id for hit in selected),
            query_hash=normalized_query_hash(request.query),
            access_hash=request.access_hash,
            membership_hash=request.membership_hash,
        ),
    )


def select_bounded_memory_hits(
    hits: Iterable[MemorySearchHit], request: MemorySearchRequest
) -> tuple[MemorySearchHit, ...]:
    """Apply the same deterministic source/conflict budget to an authorized SQL pool."""

    ordered = sorted(
        hits,
        key=lambda hit: (-hit.score, -hit.recorded_at.timestamp(), hit.record_id),
    )
    return _select_with_source_and_conflict_budgets(ordered, request)


def _select_with_source_and_conflict_budgets(
    ordered: Iterable[MemorySearchHit], request: MemorySearchRequest
) -> tuple[MemorySearchHit, ...]:
    values = tuple(ordered)
    group_keys = tuple(
        dict.fromkeys(
            (hit.visibility, hit.namespace_id, hit.conflict_group_id)
            for hit in values
            if hit.conflict_group_id is not None
        )
    )
    conflict_groups = {
        key: tuple(
            hit
            for hit in values
            if (hit.visibility, hit.namespace_id, hit.conflict_group_id) == key
        )
        for key in group_keys
    }
    visited_groups: set[tuple[MemoryVisibility, str, str | None]] = set()
    selected: list[MemorySearchHit] = []
    source_counts: dict[tuple[MemoryVisibility, str], int] = defaultdict(int)
    for hit in values:
        unit: tuple[MemorySearchHit, ...]
        if hit.conflict_group_id is None:
            unit = (hit,)
        else:
            group_key = (hit.visibility, hit.namespace_id, hit.conflict_group_id)
            if group_key in visited_groups:
                continue
            visited_groups.add(group_key)
            unit = conflict_groups[group_key]
        if len(selected) + len(unit) > request.limit:
            if len(unit) > 1:
                raise MemoryRetrievalError("conflict_set_exceeds_result_budget")
            continue
        unit_counts: dict[tuple[MemoryVisibility, str], int] = defaultdict(int)
        for item in unit:
            unit_counts[(item.visibility, item.namespace_id)] += 1
        if request.per_namespace_limit is not None and any(
            source_counts[key] + count > request.per_namespace_limit
            for key, count in unit_counts.items()
        ):
            if len(unit) > 1:
                raise MemoryRetrievalError("conflict_set_exceeds_source_budget")
            continue
        selected.extend(unit)
        for key, count in unit_counts.items():
            source_counts[key] += count
    selected.sort(key=lambda hit: (-hit.score, -hit.recorded_at.timestamp(), hit.record_id))
    return tuple(selected)


def conflict_group_id(revision: MemoryRevision) -> str | None:
    match = re.match(r"^conflict:([A-Za-z0-9._-]{1,64})(?:\s|$)", revision.reason)
    if match is None:
        return None
    return match.group(1)


_MEMORY_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "are",
        "channel",
        "conversation",
        "did",
        "do",
        "does",
        "earlier",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "memory",
        "memories",
        "me",
        "of",
        "on",
        "our",
        "please",
        "recall",
        "remember",
        "remembered",
        "tell",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "which",
        "you",
    }
)


def _query_tokens(query: str) -> tuple[str, ...]:
    raw = tuple(
        match.group(0).lower() for match in re.finditer(r"[\w]{2,64}(?:[.-][\w]{1,64})*", query)
    )
    relevant = tuple(token for token in raw if token not in _MEMORY_QUERY_STOP_WORDS)
    # A query made entirely of conversational scaffolding is not useful retrieval
    # authority. Preserve the empty result so the caller fails closed.
    return tuple(dict.fromkeys(relevant))


def normalize_memory_query(query: str) -> str:
    """Return the deterministic lexical representation shared by retrieval and caching."""

    tokens = _query_tokens(query)
    if not tokens:
        raise MemoryRetrievalError("empty_search_query")
    return " ".join(tokens)


def normalized_query_hash(query: str) -> str:
    """Hash the normalized lexical query used by retrieval, not caller formatting."""

    return hashlib.sha256(normalize_memory_query(query).encode("utf-8")).hexdigest()


def channel_authorized_namespaces(
    conversation_id: str,
    *,
    thread_namespace_id: str | None = None,
) -> frozenset[AuthorizedMemoryNamespace]:
    """Authorize one shared destination and, optionally, its exact current thread."""

    namespaces = {
        AuthorizedMemoryNamespace(visibility=visibility, namespace_id=conversation_id)
        for visibility in (
            MemoryVisibility.CONVERSATION_LOCAL,
            MemoryVisibility.CHANNEL_LOCAL,
        )
    }
    if thread_namespace_id is not None:
        namespaces.add(
            AuthorizedMemoryNamespace(
                visibility=MemoryVisibility.THREAD_LOCAL,
                namespace_id=thread_namespace_id,
            )
        )
    return frozenset(namespaces)


def dm_authorized_namespaces(
    conversation_ids: Iterable[str],
    *,
    actor_id: str,
    thread_namespace_id: str | None = None,
) -> frozenset[AuthorizedMemoryNamespace]:
    """Build the exact current-membership union usable only for a trusted 1:1 DM."""

    normalized_ids = frozenset(conversation_ids)
    if not normalized_ids:
        raise ValueError("DM authorized conversation set cannot be empty")
    namespaces = {
        AuthorizedMemoryNamespace(visibility=visibility, namespace_id=conversation_id)
        for conversation_id in normalized_ids
        for visibility in (
            MemoryVisibility.CONVERSATION_LOCAL,
            MemoryVisibility.CHANNEL_LOCAL,
        )
    }
    namespaces.add(
        AuthorizedMemoryNamespace(
            visibility=MemoryVisibility.ACTOR_PRIVATE,
            namespace_id=actor_id,
        )
    )
    if thread_namespace_id is not None:
        namespaces.add(
            AuthorizedMemoryNamespace(
                visibility=MemoryVisibility.THREAD_LOCAL,
                namespace_id=thread_namespace_id,
            )
        )
    return frozenset(namespaces)
