from __future__ import annotations

from datetime import UTC, datetime
from typing import cast

import pytest

from leo.domain.conversation import ConversationKind
from leo.harness.models import (
    RunPhase,
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import FixedClock
from leo.memory.models import MemoryStatus, MemoryVisibility
from leo.memory.navigation import (
    AuthorizedMemoryDocument,
    MemoryNavigationAuthority,
    MemoryNavigationError,
    MemoryResultKind,
    ProgressiveMemoryItem,
    ProgressiveMemoryOpenResult,
    ProgressiveMemorySearchResult,
    deterministic_memory_chunks,
    membership_snapshot_hash,
    project_open_window,
)
from leo.memory.navigation_tools import build_memory_navigation_tools
from leo.persistence.memory_navigation import PostgresProgressiveMemoryService

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="optional-domain")


def _authority(**updates: object) -> MemoryNavigationAuthority:
    values: dict[str, object] = {
        "scope": SCOPE,
        "team_id": "T1",
        "destination_id": "D1",
        "destination_kind": ConversationKind.DM,
        "actor_id": "U1",
        "task_id": "task-1",
        "run_id": "run-1",
        "allowed_conversation_ids": ("C1", "C2", "D1"),
        "access_hash": "a" * 64,
        "membership_hash": membership_snapshot_hash(("C1", "C2", "D1")),
        "current_thread_namespace_id": "slack:T1:D1:100.1",
    }
    values.update(updates)
    return MemoryNavigationAuthority.model_validate(values)


def _document(content: str) -> AuthorizedMemoryDocument:
    return AuthorizedMemoryDocument(
        record_id="internal-record-id",
        revision=2,
        content=content,
        content_hash=__import__("hashlib").sha256(content.encode()).hexdigest(),
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C1",
        status=MemoryStatus.ACTIVE,
        handle="mh_opaque-handle-value",
        reference="mem_safe-reference",
    )


def test_navigation_authority_is_exact_and_group_destinations_cannot_union() -> None:
    authority = _authority()
    assert {
        item.namespace_id
        for item in authority.authorized_namespaces
        if item.visibility is MemoryVisibility.CONVERSATION_LOCAL
    } == {"C1", "C2", "D1"}
    with pytest.raises(ValueError, match="exact destination"):
        _authority(
            destination_id="G1",
            destination_kind=ConversationKind.GROUP_DM,
            allowed_conversation_ids=("C1", "G1"),
            membership_hash=membership_snapshot_hash(("C1", "G1")),
        )


def test_long_document_chunks_are_bounded_and_model_projection_has_no_internal_id() -> None:
    content = " ".join(f"Sentence {index} carries synthetic evidence." for index in range(180))
    chunks = deterministic_memory_chunks(content)
    assert len(chunks) > 4
    assert all(len(chunk) <= 1_000 for chunk in chunks)
    first = project_open_window(_document(content), max_chunks=3)
    assert tuple(chunk.ordinal for chunk in first.chunks) == (0, 1, 2)
    assert first.next_ordinal == 3
    selected = project_open_window(
        _document(content),
        query="Sentence 170",
        max_chunks=2,
    )
    assert any("Sentence 170" in chunk.text for chunk in selected.chunks)
    item = ProgressiveMemoryItem(
        kind=MemoryResultKind.CARD,
        reference="mem_safe-reference",
        excerpt="Synthetic excerpt",
        handle="mh_opaque-handle-value",
        chunk_count=len(chunks),
        source_conversation="C1",
        lifecycle_status=MemoryStatus.ACTIVE,
    )
    payload = item.model_dump(mode="json")
    assert "record_id" not in payload
    assert "internal-record-id" not in str(payload)


class _FakeNavigationService:
    async def search(self, *args: object, **kwargs: object) -> ProgressiveMemorySearchResult:
        del args, kwargs
        return ProgressiveMemorySearchResult(
            items=(
                ProgressiveMemoryItem(
                    kind=MemoryResultKind.INLINE,
                    reference="mem_safe-reference",
                    content="Synthetic authorized memory.",
                    source_conversation="C1",
                    lifecycle_status=MemoryStatus.ACTIVE,
                ),
            ),
            query_hash="b" * 64,
            selected_count=1,
            cache_status="hit",
        )

    async def open(self, *args: object, **kwargs: object) -> ProgressiveMemoryOpenResult:
        del args, kwargs
        return project_open_window(_document("Synthetic authorized memory."))

    async def search_within(self, *args: object, **kwargs: object) -> ProgressiveMemoryOpenResult:
        del args, kwargs
        return project_open_window(_document("Synthetic authorized memory."), query="authorized")


class _ExplodingNavigationService(_FakeNavigationService):
    async def search(self, *args: object, **kwargs: object) -> ProgressiveMemorySearchResult:
        del args, kwargs
        raise RuntimeError("do-not-disclose-memory-content")


def _context(*, run_id: str = "run-1", actor_id: str = "U1") -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=SCOPE,
            actor_id=actor_id,
            roles=frozenset({"researcher"}),
        ),
        run_id=run_id,
        tool_call_id="call-1",
    )


@pytest.mark.asyncio
async def test_navigation_tools_bind_authority_and_surface_typed_denials() -> None:
    service = cast(PostgresProgressiveMemoryService, _FakeNavigationService())
    search, open_tool, within = build_memory_navigation_tools(
        service=service,
        authority=_authority(),
        clock=FixedClock(NOW),
    )
    searched = await search.execute({"query": "synthetic"}, _context())
    assert isinstance(searched, ToolSuccess)
    assert searched.data["cache_status"] == "hit"
    opened = await open_tool.execute({"handle": "mh_opaque-handle-value"}, _context())
    assert isinstance(opened, ToolSuccess)
    matched = await within.execute(
        {"handle": "mh_opaque-handle-value", "query": "authorized"},
        _context(),
    )
    assert isinstance(matched, ToolSuccess)
    denied = await search.execute({"query": "synthetic"}, _context(run_id="other"))
    assert isinstance(denied, ToolFailure)
    assert denied.code == "MEMORY_AUTHORITY_MISMATCH"


@pytest.mark.asyncio
async def test_unexpected_memory_read_failure_is_content_free_typed_and_retryable_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    service = cast(PostgresProgressiveMemoryService, _ExplodingNavigationService())
    search = build_memory_navigation_tools(
        service=service,
        authority=_authority(),
        clock=FixedClock(NOW),
    )[0]
    registry = ToolRegistry((search,))

    result = await registry.execute(
        ToolRequest(id="call-1", name="memory.search", arguments={"query": "Project Borealis"}),
        _context(),
        RunPhase.RESEARCH,
    )

    assert isinstance(result, ToolFailure)
    assert result.code == "MEMORY_SEARCH_UNAVAILABLE"
    assert result.retryable is True
    assert search.spec.retry.max_attempts == 2
    assert "RuntimeError" in caplog.text
    assert "do-not-disclose-memory-content" not in caplog.text


@pytest.mark.parametrize("value", [(), ("D1", "D1"), ("C2", "C1")])
def test_membership_snapshot_hash_rejects_ambiguous_source_sets(value: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        membership_snapshot_hash(value)


def test_open_window_rejects_out_of_range_ordinal() -> None:
    with pytest.raises(MemoryNavigationError, match="out_of_range"):
        project_open_window(_document("Short memory."), start_ordinal=2)
