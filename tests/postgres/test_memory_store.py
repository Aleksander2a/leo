from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import delete

from leo.config import Settings
from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryStatus,
    MemoryVisibility,
)
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.schema import (
    MemoryCapabilityHandleRow,
    MemoryRecordRow,
    MemoryRetrievalCacheRow,
    MemoryRevisionRow,
    MemorySourceRow,
)


@pytest_asyncio.fixture
async def memory_store() -> AsyncIterator[PostgresMemoryStore]:
    database_url = Settings().database_url
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(database_url.get_secret_value())
    sessions = create_session_factory(engine)
    try:
        yield PostgresMemoryStore(sessions)
    finally:
        async with sessions() as session, session.begin():
            for row_type in (
                MemoryCapabilityHandleRow,
                MemoryRetrievalCacheRow,
                MemoryRevisionRow,
                MemorySourceRow,
                MemoryRecordRow,
            ):
                await session.execute(
                    delete(row_type).where(row_type.organization_id == "memory-org")
                )
        await engine.dispose()


def _memory() -> tuple[ScopeKey, MemoryRecord, MemoryRevision, tuple[MemorySource, ...]]:
    scope = ScopeKey(organization_id="memory-org", strategy_id="memory-strategy")
    source = MemorySource(
        id="source-1",
        scope=scope,
        source_kind="slack_thread",
        reference="synthetic:thread-1",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-memory-store-test",
    )
    record = MemoryRecord(
        id="memory-1",
        scope=scope,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        created_at=datetime(2026, 8, 21, tzinfo=UTC),
    )
    revision = MemoryRevision.from_content(
        id="memory-1-revision-1",
        record_id=record.id,
        number=1,
        content="Synthetic thesis context.",
        source_ids=(source.id,),
        visibility=record.visibility,
        namespace_id=record.namespace_id,
        sensitivity=0.2,
        valid_from=record.created_at,
        recorded_at=record.created_at,
        actor_id="actor-1",
        reason="initial note",
    )
    return scope, record, revision, (source,)


@pytest.mark.asyncio
async def test_postgres_memory_round_trip_and_scope_denial(
    memory_store: PostgresMemoryStore,
) -> None:
    scope, record, revision, sources = _memory()
    await memory_store.create(record, revision, sources)

    current = await memory_store.current(scope, record.id)
    assert current is not None
    assert current.content == revision.content
    with pytest.raises(NotFoundError):
        await memory_store.current(
            ScopeKey(organization_id="other-org", strategy_id=scope.strategy_id), record.id
        )


@pytest.mark.asyncio
async def test_postgres_memory_correction_and_forget_are_append_only(
    memory_store: PostgresMemoryStore,
) -> None:
    scope, record, revision, sources = _memory()
    await memory_store.create(record, revision, sources)
    corrected = MemoryRevision.from_content(
        id="memory-1-revision-2",
        record_id=record.id,
        number=2,
        content="Corrected synthetic thesis context.",
        source_ids=revision.source_ids,
        visibility=record.visibility,
        namespace_id=record.namespace_id,
        sensitivity=0.2,
        valid_from=revision.recorded_at,
        recorded_at=revision.recorded_at,
        actor_id="actor-1",
        reason="corrected note",
        supersedes_revision=1,
    )
    updated = await memory_store.append_revision(scope, record.id, 1, corrected)
    assert updated.current_revision == 2
    assert (await memory_store.current(scope, record.id)).content == corrected.content  # type: ignore[union-attr]

    forgotten = await memory_store.forget(scope, record.id, "synthetic forget request")
    assert forgotten.status is MemoryStatus.RETRACTED
    assert await memory_store.current(scope, record.id) is None


@pytest.mark.asyncio
async def test_postgres_memory_append_is_serialized_by_record_lock(
    memory_store: PostgresMemoryStore,
) -> None:
    scope, record, revision, sources = _memory()
    await memory_store.create(record, revision, sources)

    async def append(revision_id: str) -> object:
        candidate = MemoryRevision.from_content(
            id=revision_id,
            record_id=record.id,
            number=2,
            content=f"Concurrent candidate {revision_id}.",
            source_ids=revision.source_ids,
            visibility=record.visibility,
            namespace_id=record.namespace_id,
            sensitivity=0.2,
            valid_from=revision.recorded_at,
            recorded_at=revision.recorded_at,
            actor_id="actor-1",
            reason="concurrent correction",
            supersedes_revision=1,
        )
        try:
            return await memory_store.append_revision(scope, record.id, 1, candidate)
        except Exception as exc:  # pragma: no cover - asserted below
            return exc

    results = await asyncio.gather(append("memory-1-race-a"), append("memory-1-race-b"))
    assert sum(isinstance(result, ConcurrencyError) for result in results) == 1
    assert sum(not isinstance(result, Exception) for result in results) == 1
