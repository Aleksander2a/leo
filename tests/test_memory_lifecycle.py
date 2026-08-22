from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.harness.models import ScopeKey
from leo.harness.store_errors import NotFoundError, StoreError
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.store import InMemoryMemoryStore


def _fixture() -> tuple[MemoryRecord, MemoryRevision, MemorySource, ScopeKey]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scope = ScopeKey(organization_id="memory-org", strategy_id="technology")
    source = MemorySource(
        id="source-thread-1",
        scope=scope,
        source_kind="slack_thread",
        reference="thread-synthetic-1",
        visibility=MemoryVisibility.CHANNEL_LOCAL,
        namespace_id="channel-demo",
    )
    record = MemoryRecord(
        id="memory-1",
        scope=scope,
        kind=MemoryKind.NOTE,
        visibility=MemoryVisibility.CHANNEL_LOCAL,
        namespace_id="channel-demo",
        current_revision=1,
        created_at=now,
    )
    revision = MemoryRevision.from_content(
        id="memory-1-revision-1",
        record_id=record.id,
        number=1,
        content="NVDA is a synthetic research note.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.1,
        valid_from=now,
        recorded_at=now,
        actor_id="actor-1",
        reason="explicit_remember",
    )
    return record, revision, source, scope


@pytest.mark.asyncio
async def test_memory_store_appends_correction_and_forgets_immediately() -> None:
    record, revision, source, scope = _fixture()
    store = InMemoryMemoryStore()
    await store.create(record, revision, (source,))
    corrected = MemoryRevision.from_content(
        id="memory-1-revision-2",
        record_id=record.id,
        number=2,
        content="NVDA is a corrected synthetic research note.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.1,
        valid_from=revision.valid_from,
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
        actor_id="actor-1",
        reason="explicit_correct",
        supersedes_revision=1,
    )
    updated = await store.append_revision(scope, record.id, 1, corrected)
    assert updated.current_revision == 2
    assert (await store.current(scope, record.id)).content == corrected.content  # type: ignore[union-attr]
    forgotten = await store.forget(scope, record.id, "explicit_forget")
    assert forgotten.status is MemoryStatus.RETRACTED
    assert await store.current(scope, record.id) is None


@pytest.mark.asyncio
async def test_memory_store_rejects_cross_scope_sources_and_reads() -> None:
    record, revision, source, _scope = _fixture()
    store = InMemoryMemoryStore()
    wrong_source = source.model_copy(
        update={"id": "source-wrong", "scope": ScopeKey(organization_id="other", strategy_id="x")}
    )
    with pytest.raises(StoreError):
        await store.create(
            record,
            revision.model_copy(update={"source_ids": (wrong_source.id,)}),
            (wrong_source,),
        )
    with pytest.raises(NotFoundError):
        await store.current(ScopeKey(organization_id="other", strategy_id="x"), record.id)


def test_memory_hash_and_temporal_contracts_are_harness_owned() -> None:
    record, revision, source, _scope = _fixture()
    with pytest.raises(ValidationError):
        invalid = revision.model_dump()
        invalid["content_hash"] = "0" * 64
        MemoryRevision.model_validate(invalid)
    assert record.visibility is source.visibility
