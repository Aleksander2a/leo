from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.harness.models import ScopeKey
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.projection import ProjectionRequest, render_memory_projection_page
from leo.memory.retrieval import AuthorizedMemoryNamespace

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="workspace-demo", strategy_id="demo")
AUTH = frozenset(
    {
        AuthorizedMemoryNamespace(
            visibility=MemoryVisibility.CONVERSATION_LOCAL,
            namespace_id="conv-a",
        )
    }
)


def _pair(
    record_id: str,
    content: str,
    *,
    scope: ScopeKey = SCOPE,
    namespace_id: str = "conv-a",
    current_revision: int = 1,
    revision_number: int = 1,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> tuple[MemoryRecord, MemoryRevision]:
    record = MemoryRecord(
        id=record_id,
        scope=scope,
        kind=MemoryKind.NOTE,
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id=namespace_id,
        current_revision=current_revision,
        status=status,
        created_at=NOW,
    )
    revision = MemoryRevision.from_content(
        id=f"revision-{record_id}-{revision_number}",
        record_id=record_id,
        number=revision_number,
        content=content,
        source_ids=(f"source-{record_id}",),
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id=namespace_id,
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="actor",
        reason="synthetic",
        status=status,
    )
    return record, revision


def _request(**updates: object) -> ProjectionRequest:
    payload: dict[str, object] = {
        "scope": SCOPE,
        "authorized_namespaces": AUTH,
        "generated_at": "2026-08-21T00:00:00Z",
        "policy_version": "projection-v1",
        "page_size": 1,
    }
    payload.update(updates)
    return ProjectionRequest(**payload)


def test_projection_is_exact_current_escaped_and_keyset_paginated() -> None:
    records = (
        _pair("record-a", "<script>@here [click](x) *bold*</script>"),
        _pair(
            "record-b",
            "Second authorized memory.",
            scope=ScopeKey(
                organization_id=SCOPE.organization_id,
                strategy_id="optional-domain-b",
            ),
        ),
        _pair("record-channel-b", "Hidden channel B.", namespace_id="conv-b"),
        _pair(
            "record-foreign",
            "Hidden workspace.",
            scope=ScopeKey(organization_id="other", strategy_id="demo"),
        ),
        _pair("record-old", "Old revision.", current_revision=2, revision_number=1),
        _pair("record-retracted", "Retracted.", status=MemoryStatus.RETRACTED),
    )

    first = render_memory_projection_page(records, _request())
    assert first.item_count == 1
    assert first.source_revisions == (("record-a", 1),)
    assert first.next_cursor is not None
    assert "&lt;script&gt;" in first.markdown
    assert "&#64;here" in first.markdown
    assert "\\[click\\]\\(x\\)" in first.markdown
    assert "record-channel-b" not in first.markdown
    assert "record-foreign" not in first.markdown

    second = render_memory_projection_page(
        records,
        _request(after=first.next_cursor),
    )
    assert second.source_revisions == (("record-b", 1),)
    assert second.next_cursor is None
    assert second.digest != first.digest
    assert "optional\\-domain\\-b" in second.markdown
    assert "not disclosure authority" in second.markdown


def test_projection_cursor_is_bound_to_scope_and_rejects_malformed_input() -> None:
    first = render_memory_projection_page(
        (_pair("record-a", "Authorized."), _pair("record-b", "Also authorized.")),
        _request(),
    )
    assert first.next_cursor is not None

    with pytest.raises(ValueError, match="invalid for this scope"):
        render_memory_projection_page(
            (),
            _request(
                scope=ScopeKey(organization_id="other", strategy_id="demo"),
                after=first.next_cursor,
            ),
        )
    with pytest.raises(ValueError, match="invalid for this scope"):
        render_memory_projection_page((), _request(after="not-base64***"))
