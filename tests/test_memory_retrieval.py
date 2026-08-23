from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.harness.models import ScopeKey
from leo.memory.cache import RetrievalCache, RetrievalCacheEntry, RetrievalCacheKey
from leo.memory.models import MemoryRevision, MemoryStatus, MemoryVisibility
from leo.memory.retrieval import (
    AuthorizedMemoryNamespace,
    MemorySearchRequest,
    ScopedMemoryCandidate,
    channel_authorized_namespaces,
    dm_authorized_namespaces,
    normalize_memory_query,
    normalized_query_hash,
    search_memory,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
ACCESS_HASH = "a" * 64
MEMBERSHIP_HASH = "b" * 64


def _candidate(
    record_id: str,
    content: str,
    *,
    scope: ScopeKey = SCOPE,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    visibility: MemoryVisibility = MemoryVisibility.CONVERSATION_LOCAL,
    namespace_id: str = "conv-a",
    sensitivity: float = 0.2,
    valid_from: datetime = NOW,
) -> ScopedMemoryCandidate:
    revision = MemoryRevision.from_content(
        id=f"revision-{record_id}",
        record_id=record_id,
        number=1,
        content=content,
        source_ids=(f"source-{record_id}",),
        visibility=visibility,
        namespace_id=namespace_id,
        sensitivity=sensitivity,
        valid_from=valid_from,
        recorded_at=NOW,
        actor_id="actor",
        reason="synthetic note",
        status=status,
        supersedes_revision=1 if status is MemoryStatus.RETRACTED else None,
    )
    return ScopedMemoryCandidate(scope=scope, revision=revision)


def _request(**updates: object) -> MemorySearchRequest:
    payload: dict[str, object] = {
        "scope": SCOPE,
        "query": "NVDA demand",
        "authorized_namespaces": frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="conv-a",
                )
            }
        ),
        "access_hash": ACCESS_HASH,
        "membership_hash": MEMBERSHIP_HASH,
        "as_of": NOW,
    }
    payload.update(updates)
    return MemorySearchRequest(**payload)


def test_normalize_memory_query_returns_empty_string_instead_of_raising() -> None:
    """A pure-stop-word recall phrasing (the exact shape live.py's direct_recall
    regex forces memory.search for) must normalize to "" deterministically,
    never raise -- normalized_query_hash must stay a pure function of that
    empty string too, since it feeds both the SQL cache key and the trace.
    """
    assert normalize_memory_query("What do you remember about our conversation?") == ""
    assert normalize_memory_query("! @ #") == ""
    assert normalize_memory_query("") == ""
    assert normalized_query_hash("What do you remember about our conversation?") == (
        normalized_query_hash("! @ #")
    )

    # Real lexical content still normalizes exactly as before.
    assert normalize_memory_query("NVDA demand") == "nvda demand"
    assert normalize_memory_query("  NVDA   DEMAND ") == "nvda demand"


def test_retrieval_filters_scope_status_sensitivity_and_time_before_rank() -> None:
    candidates = (
        _candidate("allowed", "NVDA demand remains constructive."),
        _candidate(
            "foreign",
            "NVDA demand is foreign.",
            scope=ScopeKey(organization_id="other", strategy_id="strategy"),
        ),
        _candidate("retracted", "NVDA demand is retracted.", status=MemoryStatus.RETRACTED),
        _candidate("sensitive", "NVDA demand is sensitive.", sensitivity=0.9),
        _candidate("future", "NVDA demand is future.", valid_from=NOW + timedelta(days=1)),
    )
    hits = search_memory(candidates, _request(max_sensitivity=0.5))
    assert [hit.record_id for hit in hits] == ["allowed"]


def test_retrieval_is_bounded_deterministic_and_rejects_injected_query_safely() -> None:
    candidates = tuple(_candidate(f"memory-{index}", "NVDA demand") for index in range(4))
    request = _request(limit=2)
    assert search_memory(candidates, request) == search_memory(candidates, request)
    assert len(search_memory(candidates, request)) == 2
    assert search_memory(candidates, _request(query="' OR 1=1 --")) == ()


def test_query_with_no_lexical_content_falls_back_to_recent_authorized_browse() -> None:
    """A query that normalizes away entirely (stop words/punctuation only) must
    never raise -- it degrades to a bounded browse of the caller's own
    already-authorized, non-retracted records rather than failing the turn.
    """
    candidates = tuple(_candidate(f"memory-{index}", "NVDA demand") for index in range(4))

    all_stop_words = search_memory(
        candidates, _request(query="What do you remember about our conversation?")
    )
    assert {hit.record_id for hit in all_stop_words} == {c.revision.record_id for c in candidates}

    punctuation_only = search_memory(candidates, _request(query="! @ #"))
    assert {hit.record_id for hit in punctuation_only} == {c.revision.record_id for c in candidates}

    bounded = search_memory(candidates, _request(query="! @ #", limit=2))
    assert len(bounded) == 2


def test_channel_authority_reads_only_the_exact_channel_namespace() -> None:
    candidates = (
        _candidate(
            "channel-a",
            "NVDA demand from A",
            visibility=MemoryVisibility.CHANNEL_LOCAL,
            namespace_id="A",
        ),
        _candidate(
            "channel-b",
            "NVDA demand from B",
            visibility=MemoryVisibility.CHANNEL_LOCAL,
            namespace_id="B",
        ),
    )
    request = _request(
        authorized_namespaces=channel_authorized_namespaces("A"),
    )
    assert [hit.record_id for hit in search_memory(candidates, request)] == ["channel-a"]


def test_dm_authority_can_union_exact_current_conversations() -> None:
    candidates = tuple(
        _candidate(
            f"channel-{namespace.lower()}",
            f"NVDA demand from {namespace}",
            scope=ScopeKey(
                organization_id=SCOPE.organization_id,
                strategy_id=f"optional-domain-{namespace.lower()}",
            ),
            visibility=MemoryVisibility.CHANNEL_LOCAL,
            namespace_id=namespace,
        )
        for namespace in ("A", "B", "C")
    )
    request = _request(
        authorized_namespaces=dm_authorized_namespaces(
            ("A", "B"),
            actor_id="actor-1",
        ),
    )
    assert {hit.record_id for hit in search_memory(candidates, request)} == {
        "channel-a",
        "channel-b",
    }


def test_retrieval_rejects_legacy_strategy_or_organization_shared_authority() -> None:
    for visibility in (
        MemoryVisibility.STRATEGY_SHARED,
        MemoryVisibility.ORGANIZATION_SHARED,
    ):
        with pytest.raises(ValueError, match="conversation-native"):
            _request(
                authorized_namespaces=frozenset(
                    {
                        AuthorizedMemoryNamespace(
                            visibility=visibility,
                            namespace_id="legacy-global",
                        )
                    }
                )
            )


def test_visibility_and_namespace_widening_cannot_form_an_unauthorized_pair() -> None:
    mismatched = _candidate(
        "mismatched",
        "NVDA demand should stay hidden",
        visibility=MemoryVisibility.ACTOR_PRIVATE,
        namespace_id="A",
    )
    request = _request(
        authorized_namespaces=frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CHANNEL_LOCAL,
                    namespace_id="A",
                ),
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.ACTOR_PRIVATE,
                    namespace_id="actor-1",
                ),
            }
        )
    )
    assert search_memory((mismatched,), request) == ()


def test_cache_key_binds_normalized_query_and_access_snapshots() -> None:
    request = _request(query="  NVDA   DEMAND ")
    key = RetrievalCacheKey.from_request(
        request,
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    equivalent = RetrievalCacheKey.from_request(
        request.model_copy(update={"query": "nvda demand"}),
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    assert key.query_hash == equivalent.query_hash

    cache = RetrievalCache()
    cache.put(RetrievalCacheEntry(key=key, record_ids=("allowed",)))
    revoked_access = RetrievalCacheKey.from_request(
        request.model_copy(update={"access_hash": "c" * 64}),
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    changed_membership = RetrievalCacheKey.from_request(
        request.model_copy(update={"membership_hash": "d" * 64}),
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    changed_budget = RetrievalCacheKey.from_request(
        request.model_copy(update={"limit": request.limit + 1}),
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    changed_clock = RetrievalCacheKey.from_request(
        request.model_copy(update={"as_of": request.as_of + timedelta(seconds=1)}),
        generation=1,
        policy_version="retrieval-v1",
        content_digest="content-v1",
    )
    assert cache.get(key) is not None
    assert cache.get(revoked_access) is None
    assert cache.get(changed_membership) is None
    assert cache.get(changed_budget) is None
    assert cache.get(changed_clock) is None
