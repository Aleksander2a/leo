from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.harness.models import ScopeKey
from leo.memory.cache import RetrievalCache, RetrievalCacheEntry, RetrievalCacheKey
from leo.memory.compaction import SummaryProposal, make_summary
from leo.memory.maintenance import PurgeTarget, make_purge_plan, validate_confirmation

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="workspace-demo", strategy_id="demo")


def _cache_key(
    *,
    access_hash: str = "a" * 64,
    membership_hash: str = "b" * 64,
    generation: int = 1,
) -> RetrievalCacheKey:
    return RetrievalCacheKey(
        scope=SCOPE,
        query_hash="c" * 64,
        access_hash=access_hash,
        membership_hash=membership_hash,
        as_of=NOW,
        max_sensitivity=1,
        limit=10,
        generation=generation,
        policy_version="fts-v2",
        content_digest="content-v1",
    )


def test_incremental_summary_is_versioned_and_cannot_drop_sources_or_prior_facts() -> None:
    first = make_summary(
        "thread-1",
        SCOPE,
        1,
        SummaryProposal(
            objective="Track the synthetic project",
            corrections=("The current target is October.",),
            decisions=("Use the revised schedule.",),
            commitments=("Publish a follow-up.",),
            unresolved_questions=("Is qualification complete?",),
            evidence_ids=("evidence-1",),
            covered_message_ids=("message-1",),
        ),
        available_source_ids=frozenset({"message-1", "evidence-1"}),
    )
    second_proposal = first.proposal.model_copy(
        update={"covered_message_ids": ("message-1", "message-2")}
    )
    second = make_summary(
        "thread-1",
        SCOPE,
        2,
        second_proposal,
        available_source_ids=frozenset({"message-1", "message-2", "evidence-1"}),
        previous=first,
    )

    assert second.version == 2
    assert second.source_ids == ("message-1", "message-2", "evidence-1")
    assert second.digest != first.digest

    with pytest.raises(ValueError, match="dropped covered source messages"):
        make_summary(
            "thread-1",
            SCOPE,
            3,
            second.proposal.model_copy(update={"covered_message_ids": ("message-2",)}),
            available_source_ids=frozenset({"message-2", "evidence-1"}),
            previous=second,
        )
    with pytest.raises(ValueError, match="append exactly one"):
        make_summary(
            "thread-1",
            SCOPE,
            4,
            second.proposal,
            available_source_ids=frozenset({"message-1", "message-2", "evidence-1"}),
            previous=second,
        )


def test_cache_expiry_and_authority_and_generation_invalidation_fail_closed() -> None:
    current = _cache_key()
    old_authority = _cache_key(access_hash="d" * 64)
    old_membership = _cache_key(membership_hash="e" * 64)
    old_generation = _cache_key(generation=1)
    new_generation = _cache_key(generation=2)
    cache = RetrievalCache()
    for key in {current, old_authority, old_membership, old_generation, new_generation}:
        cache.put(
            RetrievalCacheEntry(
                key=key,
                record_ids=("record-1",),
                expires_at=NOW + timedelta(minutes=5),
            )
        )

    assert cache.get(current, now=NOW) is not None
    assert cache.get(current, now=NOW + timedelta(minutes=6)) is None
    cache.invalidate_authority(
        SCOPE,
        access_hash=current.access_hash,
        membership_hash=current.membership_hash,
    )
    assert cache.get(old_authority) is None
    assert cache.get(old_membership) is None
    cache.invalidate_generation(SCOPE, current_generation=2)
    assert cache.get(old_generation) is None
    assert cache.get(new_generation) is not None


def test_manual_purge_confirmation_binds_exact_versioned_snapshot() -> None:
    target = PurgeTarget(record_id="record-1", generation=2, current_revision=3)
    plan = make_purge_plan(SCOPE, ("record-1",), targets=(target,))

    validate_confirmation(plan, plan.confirmation_token, scope=SCOPE)
    changed = make_purge_plan(
        SCOPE,
        ("record-1",),
        targets=(target.model_copy(update={"generation": 3}),),
    )
    assert changed.confirmation_token != plan.confirmation_token
    with pytest.raises(ValueError, match="stale or unauthorized"):
        validate_confirmation(changed, plan.confirmation_token, scope=SCOPE)
    with pytest.raises(ValueError, match="non-wildcard"):
        make_purge_plan(SCOPE, ("*",))
