from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.harness.context_budget import (
    BudgetSegment,
    ContextBudget,
    ContextBudgetError,
    assemble_budgeted_context,
)
from leo.harness.models import ScopeKey
from leo.memory.cards import HandleStore, MemoryCard
from leo.memory.compaction import SummaryProposal, make_summary
from leo.memory.corpus import CorpusEntry, freeze_corpus
from leo.memory.maintenance import make_purge_plan, validate_confirmation
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemoryVisibility,
)
from leo.memory.policy import PromotionStatus, assess_candidate
from leo.memory.projection import render_memory_projection
from leo.memory.service import MemoryCandidate

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _revision(content: str = "Synthetic note") -> MemoryRevision:
    return MemoryRevision.from_content(
        id="revision-1",
        record_id="record-1",
        number=1,
        content=content,
        source_ids=("source-1",),
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="actor",
        reason="synthetic",
    )


def _candidate(content: str) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.NOTE,
        content=content,
        source_ids=("source-1",),
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
        sensitivity=0.2,
        valid_from=NOW,
        reason="explicit",
    )


def test_promotion_requires_confirmation_and_marks_conflicts() -> None:
    assert (
        assess_candidate(SCOPE, _candidate("New"), (), confirmed=False).status
        is PromotionStatus.CONFIRMATION_REQUIRED
    )
    decision = assess_candidate(
        SCOPE,
        _candidate("Different"),
        (("record-1", _revision()),),
        confirmed=True,
    )
    assert decision.status is PromotionStatus.CONTESTED
    duplicate = assess_candidate(
        SCOPE,
        _candidate("Synthetic note"),
        (("record-1", _revision()),),
        confirmed=True,
    )
    assert duplicate.status is PromotionStatus.DUPLICATE


def test_frozen_corpus_cards_handles_and_cache_boundary() -> None:
    corpus = freeze_corpus(
        (
            CorpusEntry(
                id="entry-1",
                scope=SCOPE,
                content="NVDA synthetic",
                visibility=MemoryVisibility.STRATEGY_SHARED,
                namespace_id="strategy:strategy",
                recorded_at=NOW,
            ),
        )
    )
    assert len(corpus.digest) == 64
    card = MemoryCard(
        record_id="record-1",
        revision=1,
        scope=SCOPE,
        title="Synthetic note",
        excerpt="Synthetic excerpt",
        source_ids=("source-1",),
        created_at=NOW,
    )
    handles = HandleStore()
    handle = handles.issue(
        run_id="run-1", scope=SCOPE, card=card, expires_at=NOW + timedelta(hours=1)
    )
    assert handles.open(handle.handle, run_id="run-1", scope=SCOPE, now=NOW).record_id == "record-1"
    with pytest.raises(KeyError, match="not_authorized"):
        handles.open(handle.handle, run_id="run-2", scope=SCOPE, now=NOW)


def test_budget_eviction_keeps_pinned_segments() -> None:
    result = assemble_budgeted_context(
        (
            BudgetSegment(name="protocol", text="pinned", priority=100, pinned=True),
            BudgetSegment(name="low", text="x" * 20, priority=1),
            BudgetSegment(name="high", text="y" * 20, priority=10),
        ),
        ContextBudget(max_tokens=4, max_bytes=1000),
    )
    assert "protocol" in {segment.name for segment in result.segments}
    assert result.evicted_names == ("low", "high")
    with pytest.raises(ContextBudgetError, match="pinned_context"):
        assemble_budgeted_context(
            (BudgetSegment(name="protocol", text="x" * 100, priority=100, pinned=True),),
            ContextBudget(max_tokens=1, max_bytes=1000),
        )


def test_summary_projection_and_manual_purge_are_scoped_and_repeatable() -> None:
    summary = make_summary(
        "thread-1",
        SCOPE,
        1,
        SummaryProposal(
            objective="Track synthetic thesis",
            corrections=("Corrected price",),
            evidence_ids=("source-1",),
            covered_message_ids=("message-1",),
        ),
        available_source_ids=frozenset({"source-1", "message-1"}),
    )
    assert summary.source_ids == ("message-1", "source-1")
    record = MemoryRecord(
        id="record-1",
        scope=SCOPE,
        kind=MemoryKind.NOTE,
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
        created_at=NOW,
    )
    text, digest = render_memory_projection(
        ((record, _revision()),), generated_at="now", policy_version="p1"
    )
    assert digest and "Derived/read-only" in text
    plan = make_purge_plan(SCOPE, ("record-1",))
    validate_confirmation(plan, plan.confirmation_token, scope=SCOPE)
    with pytest.raises(ValueError, match="unauthorized"):
        validate_confirmation(
            plan,
            plan.confirmation_token,
            scope=ScopeKey(organization_id="other", strategy_id="strategy"),
        )
