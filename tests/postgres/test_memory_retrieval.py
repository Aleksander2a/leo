from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)
from leo.memory.retrieval import AuthorizedMemoryNamespace, MemorySearchRequest
from leo.persistence.memory_retrieval import PostgresMemoryRetriever
from leo.persistence.memory_store import PostgresMemoryStore


@pytest.fixture
def memory_retriever(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> tuple[PostgresMemoryStore, PostgresMemoryRetriever]:
    return (
        PostgresMemoryStore(preserved_postgres_sessions),
        PostgresMemoryRetriever(preserved_postgres_sessions),
    )


@pytest.mark.asyncio
async def test_postgres_fts_retrieval_filters_scope_and_current_status(
    memory_retriever: tuple[PostgresMemoryStore, PostgresMemoryRetriever],
) -> None:
    store, retriever = memory_retriever
    source_scope = ScopeKey(organization_id="fts-org", strategy_id="optional-source-domain")
    destination_scope = ScopeKey(
        organization_id="fts-org",
        strategy_id="optional-destination-domain",
    )
    source = MemorySource(
        id="fts-source",
        scope=source_scope,
        source_kind="synthetic",
        reference="fixture:fts",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="conv-fts",
    )
    record = MemoryRecord(
        id="fts-memory",
        scope=source_scope,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    revision = MemoryRevision.from_content(
        id="fts-revision",
        record_id=record.id,
        number=1,
        content="NVDA synthetic demand remains constructive.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.2,
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
        recorded_at=datetime(2026, 1, 1, tzinfo=UTC),
        actor_id="actor",
        reason="fixture",
    )
    await store.create(record, revision, (source,))
    request = MemorySearchRequest(
        scope=destination_scope,
        query="NVDA demand",
        authorized_namespaces=frozenset(
            {
                AuthorizedMemoryNamespace(
                    visibility=MemoryVisibility.CONVERSATION_LOCAL,
                    namespace_id=source.namespace_id,
                )
            }
        ),
        access_hash="a" * 64,
        membership_hash="b" * 64,
        as_of=datetime(2026, 1, 2, tzinfo=UTC),
    )
    hits = await retriever.search(request)
    assert [hit.record_id for hit in hits] == [record.id]
    foreign = request.model_copy(
        update={"scope": ScopeKey(organization_id="other", strategy_id="other")}
    )
    assert await retriever.search(foreign) == ()
