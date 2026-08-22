from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.thread_context import (
    select_context_with_thread_compaction,
    thread_context_source_digest,
)
from leo.harness.thread_context_tools import (
    ThreadContextAuthority,
    ThreadContextOpenTool,
    ThreadContextSnapshotService,
)
from leo.integrations.fake import FixedClock

SCOPE = ScopeKey(organization_id="org-thread", strategy_id="metadata-only")
NOW = datetime(2026, 8, 22, tzinfo=UTC)


def _authority(**updates: object) -> ThreadContextAuthority:
    values: dict[str, object] = {
        "scope": SCOPE,
        "team_id": "T1",
        "destination_id": "C1",
        "actor_id": "U1",
        "task_id": "task-1",
        "run_id": "run-1",
        "thread_root_ts": "100.000",
        "current_message_ts": "200.000",
        "allowed_conversation_ids": ("C1",),
        "access_hash": "a" * 64,
        "membership_hash": "b" * 64,
    }
    values.update(updates)
    return ThreadContextAuthority.model_validate(values)


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


def _compacted_selection():
    root = ContextItem(
        id="thread-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Root objective",
        conversation_id="C1",
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    omitted_detail = ContextItem(
        id="thread-omitted-detail",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=("supporting context " * 700) + " exact-reopen-needle",
        conversation_id="C1",
        budget_priority=1,
    )
    question = ContextItem(
        id="thread-question",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Which dependency remains unresolved?",
        conversation_id="C1",
        retention=ContextItemRetention.UNRESOLVED_QUESTION,
        budget_priority=98,
    )
    outcome = ContextItem(
        id="thread-outcome",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Leo: The prior verified tool workflow completed.",
        conversation_id="C1",
        retention=ContextItemRetention.PRIOR_OUTCOME,
        budget_priority=99,
    )
    items = (root, omitted_detail, question, outcome)
    return select_context_with_thread_compaction(
        items,
        thread_item_ids=frozenset(item.id for item in items),
        conversation_id="C1",
        summary_id_namespace="slack-thread:T1:C1:100.000",
        max_tokens=1_200,
    )


@pytest.mark.asyncio
async def test_compacted_detail_can_be_reopened_without_model_supplied_scope() -> None:
    selection = _compacted_selection()
    assert selection.compacted_item_ids == ("thread-omitted-detail",)
    assert len(selection.reopen_ranges) == 1
    source_range = selection.reopen_ranges[0]
    summary = next(item for item in selection.items if item.kind is ContextItemKind.THREAD_SUMMARY)
    assert source_range.handle in summary.content
    assert "exact-reopen-needle" not in summary.content
    assert any(
        item.retention is ContextItemRetention.UNRESOLVED_QUESTION for item in selection.items
    )
    assert any(item.retention is ContextItemRetention.PRIOR_OUTCOME for item in selection.items)

    authority = _authority()
    service = ThreadContextSnapshotService(
        authority=authority,
        ranges=selection.reopen_ranges,
    )
    tool = ThreadContextOpenTool(
        service=service,
        authority=authority,
        clock=FixedClock(NOW),
    )
    assert set(tool.spec.input_schema["properties"]) == {
        "handle",
        "max_chunks",
        "start_ordinal",
    }
    with pytest.raises(ValidationError):
        tool.validate({"handle": source_range.handle, "conversation_id": "C-forged"})

    start = 0
    opened_text = ""
    while True:
        outcome = await tool.execute(
            {
                "handle": source_range.handle,
                "start_ordinal": start,
                "max_chunks": 8,
            },
            _context(),
        )
        assert isinstance(outcome, ToolSuccess)
        opened_text += str(outcome.data)
        next_ordinal = outcome.data["next_ordinal"]
        if next_ordinal is None:
            break
        assert isinstance(next_ordinal, int)
        start = next_ordinal
    assert "exact-reopen-needle" in opened_text


@pytest.mark.asyncio
async def test_thread_handle_fails_closed_for_wrong_run_handle_and_budgets() -> None:
    selection = _compacted_selection()
    authority = _authority()
    source_range = selection.reopen_ranges[0]
    service = ThreadContextSnapshotService(
        authority=authority,
        ranges=selection.reopen_ranges,
        max_calls=2,
        max_returned_bytes=1_200,
    )
    tool = ThreadContextOpenTool(
        service=service,
        authority=authority,
        clock=FixedClock(NOW),
    )

    wrong_run = await tool.execute({"handle": source_range.handle}, _context(run_id="run-2"))
    assert isinstance(wrong_run, ToolFailure)
    assert wrong_run.code == "THREAD_CONTEXT_AUTHORITY_MISMATCH"

    unknown = await tool.execute({"handle": "thr_" + ("f" * 32)}, _context())
    assert isinstance(unknown, ToolFailure)
    assert unknown.code == "THREAD_CONTEXT_OPEN_DENIED"

    first = await tool.execute(
        {"handle": source_range.handle, "max_chunks": 1},
        _context(),
    )
    assert isinstance(first, ToolSuccess)
    exhausted = await tool.execute(
        {"handle": source_range.handle, "max_chunks": 1},
        _context(),
    )
    assert isinstance(exhausted, ToolFailure)
    assert exhausted.code == "THREAD_CONTEXT_OPEN_DENIED"


def test_snapshot_rejects_cross_conversation_source_even_with_a_valid_handle() -> None:
    selection = _compacted_selection()
    source_range = selection.reopen_ranges[0]
    forged_item = source_range.items[0].model_copy(update={"conversation_id": "C-other"})
    forged_range = type(source_range)(
        handle=source_range.handle,
        digest=thread_context_source_digest((forged_item,)),
        items=(forged_item,),
    )
    with pytest.raises(ValueError, match="exact destination"):
        ThreadContextSnapshotService(
            authority=_authority(),
            ranges=(forged_range,),
        )


def test_snapshot_rejects_a_forged_source_digest() -> None:
    source_range = _compacted_selection().reopen_ranges[0]
    forged_range = type(source_range)(
        handle=source_range.handle,
        digest="f" * 64,
        items=source_range.items,
    )
    with pytest.raises(ValueError, match="digest"):
        ThreadContextSnapshotService(
            authority=_authority(),
            ranges=(forged_range,),
        )


def test_compaction_handle_is_scoped_to_the_admitted_event_namespace() -> None:
    first = _compacted_selection()
    source_items = tuple(
        item
        for item in (*first.items, *first.reopen_ranges[0].items)
        if item.kind is not ContextItemKind.THREAD_SUMMARY
    )
    first_again = select_context_with_thread_compaction(
        source_items,
        thread_item_ids=frozenset(item.id for item in source_items),
        conversation_id="C1",
        summary_id_namespace="slack-thread:T1:C1:100.000:Ev-first",
        max_tokens=1_200,
    )
    second = select_context_with_thread_compaction(
        source_items,
        thread_item_ids=frozenset(item.id for item in source_items),
        conversation_id="C1",
        summary_id_namespace="slack-thread:T1:C1:100.000:Ev-second",
        max_tokens=1_200,
    )

    assert first_again.compaction_digest == second.compaction_digest
    assert first_again.reopen_ranges[0].handle != second.reopen_ranges[0].handle
