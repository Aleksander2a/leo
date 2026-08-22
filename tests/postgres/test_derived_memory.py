from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest
import pytest_asyncio
from sqlalchemy import delete

from leo.config import Settings
from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, StoreError
from leo.memory.cache import RetrievalCacheEntry, RetrievalCacheKey
from leo.memory.compaction import SummaryProposal
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)
from leo.memory.planes import DataPlane, EmbeddingJob, SanitizedMessage
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.derived_memory import (
    PostgresDerivedMemoryRepository,
    PostgresMemoryMaintenance,
)
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.schema import (
    ConversationRow,
    MemoryCapabilityHandleRow,
    MemoryEmbeddingJobRow,
    MemoryRecordRow,
    MemoryRetrievalCacheRow,
    MemoryRevisionRow,
    MemorySourceRow,
    SanitizedMessageRow,
    ThreadRow,
    ThreadSummaryRevisionRow,
)

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="derived-org", strategy_id="derived-strategy")


@dataclass(frozen=True)
class DerivedHarness:
    repository: PostgresDerivedMemoryRepository
    memory: PostgresMemoryStore
    maintenance: PostgresMemoryMaintenance


@pytest_asyncio.fixture
async def derived_harness() -> AsyncIterator[DerivedHarness]:
    configured = Settings().database_url
    if configured is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(configured.get_secret_value())
    sessions = create_session_factory(engine)
    async with sessions() as session, session.begin():
        session.add(
            ConversationRow(
                id="derived-conversation",
                provider="slack",
                team_id="derived-team",
                external_id="C-derived",
                kind="channel",
                actor_id=None,
            )
        )
        await session.flush()
        session.add(
            ThreadRow(
                id="derived-thread",
                organization_id=SCOPE.organization_id,
                strategy_id=SCOPE.strategy_id,
                origin_provider="slack",
                external_thread_id="C-derived:100.1",
                conversation_id="derived-conversation",
            )
        )
    try:
        yield DerivedHarness(
            repository=PostgresDerivedMemoryRepository(sessions),
            memory=PostgresMemoryStore(sessions),
            maintenance=PostgresMemoryMaintenance(sessions),
        )
    finally:
        async with sessions() as session, session.begin():
            for row_type in (
                MemoryCapabilityHandleRow,
                MemoryEmbeddingJobRow,
                MemoryRetrievalCacheRow,
                ThreadSummaryRevisionRow,
                SanitizedMessageRow,
                MemoryRevisionRow,
                MemorySourceRow,
                MemoryRecordRow,
            ):
                await session.execute(
                    delete(row_type).where(row_type.organization_id == SCOPE.organization_id)
                )
            await session.execute(delete(ThreadRow).where(ThreadRow.id == "derived-thread"))
            await session.execute(
                delete(ConversationRow).where(ConversationRow.id == "derived-conversation")
            )
        await engine.dispose()


@pytest.mark.asyncio
async def test_derived_planes_rebuild_resume_invalidate_and_confirm_purge(
    derived_harness: DerivedHarness,
) -> None:
    repository = derived_harness.repository
    first_message = SanitizedMessage.from_text(
        id="derived-message-1",
        scope=SCOPE,
        destination_id="C-derived",
        external_event_id="event-1",
        text="Remember the corrected synthetic schedule. token=redacted-value",
        recorded_at=NOW,
        conversation_id="derived-conversation",
        harness_thread_id="derived-thread",
        actor_id="U-derived",
        provider_message_ts="100.1",
        context_access_hash="a" * 64,
    )
    second_message = SanitizedMessage.from_text(
        id="derived-message-2",
        scope=SCOPE,
        destination_id="C-derived",
        external_event_id="event-2",
        text="The current target is October.",
        recorded_at=NOW + timedelta(seconds=1),
        conversation_id="derived-conversation",
        harness_thread_id="derived-thread",
        actor_id="U-derived",
        provider_message_ts="100.2",
        context_access_hash="a" * 64,
    )
    await repository.put_message(first_message)
    assert await repository.put_message(first_message) == first_message
    with pytest.raises(StoreError, match="conversation authority"):
        await repository.put_message(
            first_message.model_copy(
                update={
                    "id": "derived-message-wrong",
                    "external_event_id": "event-wrong",
                    "destination_id": "C-other",
                }
            )
        )
    await repository.put_message(second_message)
    messages = await repository.list_messages(
        SCOPE,
        conversation_id="derived-conversation",
        harness_thread_id="derived-thread",
    )
    assert messages == (first_message, second_message)
    assert "redacted-value" not in messages[0].text
    first_page = await repository.list_messages(
        SCOPE,
        conversation_id="derived-conversation",
        harness_thread_id="derived-thread",
        limit=1,
    )
    second_page = await repository.list_messages(
        SCOPE,
        conversation_id="derived-conversation",
        harness_thread_id="derived-thread",
        after=(first_page[0].recorded_at, first_page[0].id),
        limit=1,
    )
    assert first_page + second_page == messages

    first_summary = await repository.rebuild_summary(
        SCOPE,
        thread_id="derived-thread",
        proposal=SummaryProposal(
            objective="Track the synthetic schedule",
            corrections=("The target is October.",),
            covered_message_ids=(first_message.id,),
        ),
    )
    second_summary = await repository.rebuild_summary(
        SCOPE,
        thread_id="derived-thread",
        proposal=first_summary.proposal.model_copy(
            update={"covered_message_ids": (first_message.id, second_message.id)}
        ),
    )
    assert second_summary.version == 2
    assert await repository.latest_summary(SCOPE, thread_id="derived-thread") == second_summary
    assert await repository.invalidate_summaries_for_messages(SCOPE, (second_message.id,)) == 2
    assert await repository.latest_summary(SCOPE, thread_id="derived-thread") is None

    key = RetrievalCacheKey(
        scope=SCOPE,
        query_hash="b" * 64,
        access_hash="a" * 64,
        membership_hash="c" * 64,
        as_of=NOW,
        max_sensitivity=1,
        limit=10,
        generation=1,
        policy_version="fts-v2",
        content_digest="content-v1",
    )
    entry = RetrievalCacheEntry(
        key=key,
        record_ids=("derived-memory",),
        expires_at=NOW + timedelta(hours=1),
    )
    await repository.put_cache(entry)
    assert await repository.get_cache(key, now=NOW) == entry
    other_domain_key = key.model_copy(
        update={
            "scope": ScopeKey(
                organization_id=SCOPE.organization_id,
                strategy_id="other-optional-domain",
            )
        }
    )
    await repository.put_cache(entry.model_copy(update={"key": other_domain_key}))
    assert await repository.invalidate_cache_for_authority_change(SCOPE) == 2
    assert await repository.get_cache(key, now=NOW) is None
    assert await repository.get_cache(other_domain_key, now=NOW) is None

    embedding = EmbeddingJob(
        id="derived-embedding",
        scope=SCOPE,
        source_plane=DataPlane.MEMORY,
        source_id="derived-memory",
        content_hash="d" * 64,
        model="synthetic-embedding-v1",
        dimensions=8,
        status="queued",
    )
    await repository.enqueue_embedding(embedding)
    claimed = await repository.claim_embedding(SCOPE, now=NOW)
    assert claimed is not None and claimed.attempts == 1
    assert await repository.claim_embedding(SCOPE, now=NOW + timedelta(minutes=1)) is None
    reclaimed = await repository.claim_embedding(SCOPE, now=NOW + timedelta(minutes=6))
    assert reclaimed is not None and reclaimed.attempts == 2
    with pytest.raises(ConcurrencyError, match="attempt is stale"):
        await repository.finish_embedding(
            SCOPE,
            job_id=embedding.id,
            expected_attempt=1,
            status="succeeded",
            now=NOW + timedelta(minutes=6),
        )
    finished = await repository.finish_embedding(
        SCOPE,
        job_id=embedding.id,
        expected_attempt=2,
        status="succeeded",
        now=NOW + timedelta(minutes=6),
    )
    assert finished.status == "succeeded"

    source = MemorySource(
        id="derived-source",
        scope=SCOPE,
        source_kind="synthetic",
        reference="fixture:derived",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="derived-conversation",
    )
    record = MemoryRecord(
        id="derived-memory",
        scope=SCOPE,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        created_at=NOW,
    )
    revision = MemoryRevision.from_content(
        id="derived-memory-revision",
        record_id=record.id,
        number=1,
        content="Synthetic memory ready for explicit purge.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="U-derived",
        reason="synthetic",
    )
    await derived_harness.memory.create(record, revision, (source,))
    await derived_harness.memory.forget(SCOPE, record.id, "explicit demo purge")
    await repository.put_cache(entry)
    plan = await derived_harness.maintenance.prepare_purge(SCOPE, (record.id,))
    first, second = await asyncio.gather(
        derived_harness.maintenance.execute_purge(
            plan,
            scope=SCOPE,
            confirmation_token=plan.confirmation_token,
        ),
        derived_harness.maintenance.execute_purge(
            plan,
            scope=SCOPE,
            confirmation_token=plan.confirmation_token,
        ),
    )
    winner = next(result for result in (first, second) if result.purged_record_ids)
    resumed = next(result for result in (first, second) if result.already_absent_record_ids)
    assert winner.purged_record_ids == (record.id,)
    assert winner.deleted_revision_count == 2
    assert winner.invalidated_cache_count == 1
    assert resumed.already_absent_record_ids == (record.id,)
