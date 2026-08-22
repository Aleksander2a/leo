from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import postgresql

from leo.harness.models import ScopeKey
from leo.memory.models import MemoryRevision, MemoryStatus, MemoryVisibility
from leo.memory.retrieval import (
    AuthorizedMemoryNamespace,
    MemoryRetrievalError,
    MemorySearchRequest,
    ScopedMemoryCandidate,
    search_memory,
    search_memory_with_trace,
)
from leo.persistence.memory_retrieval import build_memory_search_statement

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="workspace-demo", strategy_id="demo")


def _candidate(
    record_id: str,
    content: str,
    *,
    revision: int = 1,
    current_revision: int | None = 1,
    namespace_id: str = "conv-a",
    revision_status: MemoryStatus = MemoryStatus.ACTIVE,
    record_status: MemoryStatus = MemoryStatus.ACTIVE,
    reason: str = "synthetic",
    recorded_at: datetime = NOW,
    scope: ScopeKey = SCOPE,
) -> ScopedMemoryCandidate:
    value = MemoryRevision.from_content(
        id=f"revision-{record_id}-{revision}",
        record_id=record_id,
        number=revision,
        content=content,
        source_ids=(f"source-{record_id}-{revision}",),
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id=namespace_id,
        sensitivity=0.2,
        valid_from=NOW - timedelta(days=1),
        recorded_at=recorded_at,
        actor_id="actor",
        reason=reason,
        status=revision_status,
        supersedes_revision=(
            max(1, revision - 1) if revision_status is MemoryStatus.SUPERSEDED else None
        ),
    )
    return ScopedMemoryCandidate(
        scope=scope,
        revision=value,
        current_revision=current_revision,
        record_status=record_status,
    )


def _request(**updates: object) -> MemorySearchRequest:
    payload: dict[str, object] = {
        "scope": SCOPE,
        "query": "Atlas delivery target",
        "authorized_namespaces": frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="conv-a",
                )
            }
        ),
        "access_hash": "a" * 64,
        "membership_hash": "b" * 64,
        "as_of": NOW,
        "limit": 5,
        "per_namespace_limit": 3,
    }
    payload.update(updates)
    return MemorySearchRequest(**payload)


def test_current_correction_wins_and_trace_contains_no_memory_content() -> None:
    candidates = (
        _candidate(
            "atlas",
            "Atlas delivery target June outdated.",
            revision=1,
            current_revision=2,
            revision_status=MemoryStatus.SUPERSEDED,
        ),
        _candidate(
            "atlas",
            "Atlas delivery target October corrected.",
            revision=2,
            current_revision=2,
        ),
    )

    result = search_memory_with_trace(candidates, _request())

    assert [(hit.record_id, hit.revision) for hit in result.hits] == [("atlas", 2)]
    assert result.trace.authorized_current_count == 1
    assert "October" not in result.trace.model_dump_json()


def test_conflict_sides_are_atomic_under_result_and_per_source_budgets() -> None:
    candidates = (
        _candidate(
            "helios-up",
            "Helios demand outlook expansion.",
            revision_status=MemoryStatus.CONTESTED,
            record_status=MemoryStatus.CONTESTED,
            reason="conflict:helios-demand bullish",
        ),
        _candidate(
            "helios-down",
            "Helios demand outlook contraction.",
            revision_status=MemoryStatus.CONTESTED,
            record_status=MemoryStatus.CONTESTED,
            reason="conflict:helios-demand bearish",
        ),
    )
    request = _request(query="Helios demand outlook", limit=4, per_namespace_limit=3)

    assert {hit.record_id for hit in search_memory(candidates, request)} == {
        "helios-up",
        "helios-down",
    }
    with pytest.raises(MemoryRetrievalError, match="conflict_set_exceeds_result_budget"):
        search_memory(candidates, request.model_copy(update={"limit": 1}))
    with pytest.raises(MemoryRetrievalError, match="conflict_set_exceeds_source_budget"):
        search_memory(candidates, request.model_copy(update={"per_namespace_limit": 1}))


def test_per_namespace_budget_preserves_dm_source_coverage_deterministically() -> None:
    candidates = (
        _candidate("a-primary", "Zenith margin bridge", recorded_at=NOW),
        _candidate(
            "a-secondary",
            "Zenith margin bridge",
            recorded_at=NOW - timedelta(minutes=1),
        ),
        _candidate(
            "b-primary",
            "Zenith margin bridge",
            namespace_id="conv-b",
            recorded_at=NOW - timedelta(minutes=2),
            scope=ScopeKey(
                organization_id=SCOPE.organization_id,
                strategy_id="optional-domain-b",
            ),
        ),
    )
    request = _request(
        query="Zenith margin bridge",
        authorized_namespaces=frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="conv-a",
                ),
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="conv-b",
                ),
            }
        ),
        limit=2,
        per_namespace_limit=1,
    )

    assert [hit.record_id for hit in search_memory(candidates, request)] == [
        "a-primary",
        "b-primary",
    ]


def test_postgres_fts_statement_filters_exact_authority_and_time_in_inner_cte() -> None:
    request = _request(
        authorized_namespaces=frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="conv-a",
                ),
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.ACTOR_PRIVATE,
                    namespace_id="user-1",
                ),
            }
        ),
        per_namespace_limit=2,
    )
    statement = build_memory_search_statement(request)
    compilation = statement.compile(dialect=postgresql.dialect())
    compiled = str(compilation).lower()
    parameters = tuple(compilation.params.values())

    assert "with authorized_current_ranked_memory as" in compiled
    assert "memory_records.current_revision = memory_revisions.number" in compiled
    assert "memory_revisions.search_vector @@ plainto_tsquery" in compiled
    assert "to_tsvector" not in compiled
    assert "memory_revisions.organization_id =" in compiled
    assert "memory_revisions.strategy_id =" not in compiled
    assert "memory_records.strategy_id =" not in compiled
    assert "memory_revisions.visibility =" in compiled
    assert "memory_revisions.namespace_id =" in compiled
    assert all(value in parameters for value in ("workspace-demo", "conversation_local", "conv-a"))
    assert all(value in parameters for value in ("actor_private", "user-1"))
    assert "memory_revisions.valid_from <=" in compiled
    assert "memory_revisions.valid_until is null" in compiled
    assert "memory_revisions.expires_at is null" in compiled
    assert "namespace_rank" not in compiled
    assert 1001 in parameters
