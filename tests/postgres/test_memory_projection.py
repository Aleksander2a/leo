from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.harness.models import ScopeKey
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)
from leo.memory.projection import ProjectionRequest
from leo.memory.retrieval import AuthorizedMemoryNamespace
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.memory_projection import PostgresMemoryProjectionService
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.schema import MemoryRecordRow, MemoryRevisionRow, MemorySourceRow

NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="projection-org", strategy_id="destination-domain")
SOURCE_SCOPE = ScopeKey(organization_id=SCOPE.organization_id, strategy_id="source-domain")


@pytest_asyncio.fixture
async def projection_harness(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[PostgresMemoryStore, PostgresMemoryProjectionService]]:
    yield (
        PostgresMemoryStore(preserved_postgres_sessions),
        PostgresMemoryProjectionService(preserved_postgres_sessions),
    )


@pytest_asyncio.fixture
async def projection_race_harness() -> AsyncIterator[
    tuple[PostgresMemoryStore, PostgresMemoryProjectionService]
]:
    """Use real independent transactions and clean only this synthetic org."""

    configured = Settings().database_url
    if configured is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(configured.get_secret_value())
    sessions = create_session_factory(engine)
    try:
        yield PostgresMemoryStore(sessions), PostgresMemoryProjectionService(sessions)
    finally:
        async with sessions() as session, session.begin():
            await session.execute(
                delete(MemoryRevisionRow).where(
                    MemoryRevisionRow.organization_id == SCOPE.organization_id
                )
            )
            await session.execute(
                delete(MemorySourceRow).where(
                    MemorySourceRow.organization_id == SCOPE.organization_id
                )
            )
            await session.execute(
                delete(MemoryRecordRow).where(
                    MemoryRecordRow.organization_id == SCOPE.organization_id
                )
            )
        await engine.dispose()


def _request(*, after: str | None = None) -> ProjectionRequest:
    return ProjectionRequest(
        scope=SCOPE,
        authorized_namespaces=frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id="C-projection",
                )
            }
        ),
        generated_at=NOW.isoformat(),
        policy_version="projection-v1",
        page_size=10,
        after=after,
    )


async def _seed_record(store: PostgresMemoryStore) -> tuple[MemoryRecord, MemoryRevision]:
    source = MemorySource(
        id="projection-source",
        scope=SOURCE_SCOPE,
        source_kind="synthetic",
        reference="fixture:projection",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-projection",
    )
    record = MemoryRecord(
        id="projection-record",
        scope=SOURCE_SCOPE,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        created_at=NOW,
    )
    revision = MemoryRevision.from_content(
        id="projection-revision-1",
        record_id=record.id,
        number=1,
        content="First synthetic projection value.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="U-projection",
        reason="synthetic projection fixture",
    )
    await store.create(record, revision, (source,))
    return record, revision


@pytest.mark.asyncio
async def test_projection_revision_race_is_one_current_atomic_snapshot(
    projection_race_harness: tuple[PostgresMemoryStore, PostgresMemoryProjectionService],
) -> None:
    store, projection = projection_race_harness
    record, first = await _seed_record(store)
    second = MemoryRevision.from_content(
        id="projection-revision-2",
        record_id=record.id,
        number=2,
        content="Second corrected synthetic projection value.",
        source_ids=first.source_ids,
        visibility=first.visibility,
        namespace_id=first.namespace_id,
        sensitivity=first.sensitivity,
        valid_from=NOW,
        recorded_at=NOW + timedelta(seconds=1),
        actor_id="U-projection",
        reason="confirmed correction",
        supersedes_revision=1,
    )

    raced_page, _updated = await asyncio.gather(
        projection.render_page(_request(), as_of=NOW + timedelta(seconds=2)),
        store.append_revision(SOURCE_SCOPE, record.id, 1, second),
    )
    assert raced_page.item_count == 1
    assert raced_page.source_revisions in {
        ((record.id, 1),),
        ((record.id, 2),),
    }
    assert not (
        "First synthetic" in raced_page.markdown and "Second corrected" in raced_page.markdown
    )

    current_page = await projection.render_page(
        _request(),
        as_of=NOW + timedelta(seconds=2),
    )
    assert current_page.source_revisions == ((record.id, 2),)
    assert "Second corrected" in current_page.markdown
    assert "First synthetic" not in current_page.markdown


@pytest.mark.asyncio
async def test_projection_excludes_wrong_namespace_before_render(
    projection_harness: tuple[PostgresMemoryStore, PostgresMemoryProjectionService],
) -> None:
    store, projection = projection_harness
    await _seed_record(store)
    unauthorized = _request().model_copy(
        update={
            "authorized_namespaces": frozenset(
                {
                    AuthorizedMemoryNamespace(
                        visibility=MemoryVisibility.CONVERSATION_LOCAL,
                        namespace_id="C-other",
                    )
                }
            )
        }
    )
    page = await projection.render_page(unauthorized, as_of=NOW + timedelta(seconds=1))
    assert page.item_count == 0
    assert "projection-record" not in page.markdown
