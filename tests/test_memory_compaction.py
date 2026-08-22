from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.harness.models import ScopeKey
from leo.memory.compaction import (
    CompactionPolicy,
    SummaryProposal,
    compaction_result,
    make_summary,
    select_compaction_window,
)
from leo.memory.planes import SanitizedMessage

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="domain")


def _messages(count: int) -> tuple[SanitizedMessage, ...]:
    values: list[SanitizedMessage] = []
    for index in range(count):
        marker = (
            "The accepted correction is October."
            if index == 15
            else "The unresolved question is launch timing."
            if index == 31
            else "Synthetic discussion detail that may be compacted safely."
        )
        values.append(
            SanitizedMessage.from_text(
                id=f"message-{index:03d}",
                scope=SCOPE,
                destination_id="C1",
                external_event_id=f"event-{index:03d}",
                text=f"{marker} " * 5,
                recorded_at=NOW + timedelta(seconds=index),
                conversation_id="conversation-1",
                harness_thread_id="thread-1",
                actor_id="U1",
                provider_message_ts=f"100.{index}",
                context_access_hash="a" * 64,
            )
        )
    return tuple(values)


def test_compaction_retains_recent_window_and_labeled_facts_with_token_reduction() -> None:
    messages = _messages(100)
    window = select_compaction_window(
        messages,
        CompactionPolicy(trigger_messages=50, recent_window_messages=12),
    )
    assert window.should_compact
    assert len(window.compactable_message_ids) == 88
    assert window.recent_message_ids == tuple(message.id for message in messages[-12:])
    proposal = SummaryProposal(
        objective="Ship the synthetic conversational demo",
        corrections=("The accepted correction is October.",),
        decisions=("Use exact conversation isolation.",),
        commitments=("Verify the demo in Slack.",),
        unresolved_questions=("The unresolved question is launch timing.",),
        evidence_ids=("evidence-1",),
        covered_message_ids=window.compactable_message_ids,
    )
    summary = make_summary(
        "thread-1",
        SCOPE,
        1,
        proposal,
        available_source_ids=frozenset((*window.compactable_message_ids, "evidence-1")),
    )
    result = compaction_result(summary, messages, window)
    assert result.token_reduction_ratio > 0.5
    assert result.summary.proposal.corrections == proposal.corrections
    assert result.summary.proposal.unresolved_questions == proposal.unresolved_questions


def test_compaction_does_not_trigger_below_threshold_or_drop_incremental_facts() -> None:
    messages = _messages(49)
    window = select_compaction_window(messages, CompactionPolicy())
    assert not window.should_compact
    first = make_summary(
        "thread-1",
        SCOPE,
        1,
        SummaryProposal(
            objective="Synthetic objective",
            corrections=("October is current.",),
            covered_message_ids=(messages[0].id,),
        ),
        available_source_ids=frozenset({messages[0].id}),
    )
    with pytest.raises(ValueError, match="dropped prior corrections"):
        make_summary(
            "thread-1",
            SCOPE,
            2,
            SummaryProposal(
                objective="Synthetic objective",
                covered_message_ids=(messages[0].id, messages[1].id),
            ),
            available_source_ids=frozenset({messages[0].id, messages[1].id}),
            previous=first,
        )
