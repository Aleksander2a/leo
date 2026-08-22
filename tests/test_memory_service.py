from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.harness.models import ScopeKey
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.memory.models import MemoryKind, MemorySource, MemoryVisibility
from leo.memory.service import (
    ExplicitMemoryService,
    MemoryCandidate,
    MemoryCommandRejected,
)
from leo.memory.store import InMemoryMemoryStore

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _candidate(
    content: str = "NVDA is a synthetic preference.",
    *,
    source_id: str = "source-1",
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.NOTE,
        content=content,
        source_ids=(source_id,),
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
        sensitivity=0.2,
        valid_from=NOW,
        reason="explicit demo remember",
    )


def _source(source_id: str = "source-1") -> MemorySource:
    return MemorySource(
        id=source_id,
        scope=SCOPE,
        source_kind="slack_message",
        reference=source_id,
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
    )


@pytest.mark.asyncio
async def test_remember_correct_forget_are_explicit_and_scope_bound() -> None:
    store = InMemoryMemoryStore()
    service = ExplicitMemoryService(store, FixedClock(), SequentialIdGenerator())
    with pytest.raises(MemoryCommandRejected, match="explicit_confirmation_required"):
        await service.remember(
            SCOPE, _candidate(), actor_id="actor", sources=(_source(),), confirmed=False
        )
    record = await service.remember(
        SCOPE, _candidate(), actor_id="actor", sources=(_source(),), confirmed=True
    )
    corrected = await service.correct(
        SCOPE,
        record.id,
        _candidate("NVDA is a corrected synthetic preference.", source_id="source-2"),
        actor_id="actor",
        sources=(_source("source-2"),),
        confirmed=True,
    )
    assert corrected.current_revision == 2
    forgotten = await service.forget(
        SCOPE,
        record.id,
        actor_id="actor",
        visibility=MemoryVisibility.STRATEGY_SHARED,
        namespace_id="strategy:strategy",
        sources=(_source("source-3"),),
        confirmed=True,
        reason="user requested forgetting",
    )
    assert forgotten.status.value == "retracted"
    assert await store.current(SCOPE, record.id) is None
