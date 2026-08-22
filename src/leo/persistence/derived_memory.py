"""Durable repositories for sanitized messages and regenerable memory artifacts."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.memory.cache import RetrievalCacheEntry, RetrievalCacheKey
from leo.memory.compaction import (
    CompactionPolicy,
    CompactionResult,
    SummaryProposal,
    SummaryRevision,
    compaction_result,
    make_summary,
    render_summary_content,
    select_compaction_window,
)
from leo.memory.maintenance import (
    MaintenanceHealth,
    PurgePlan,
    PurgeResult,
    PurgeTarget,
    make_purge_plan,
    validate_confirmation,
)
from leo.memory.models import MemoryStatus
from leo.memory.planes import DataPlane, EmbeddingJob, MessageRole, SanitizedMessage
from leo.persistence.schema import (
    ConversationRow,
    MemoryEmbeddingJobRow,
    MemoryRecordRow,
    MemoryRetrievalCacheRow,
    MemoryRevisionRow,
    MemorySourceRow,
    SanitizedMessageRow,
    ThreadRow,
    ThreadSummaryRevisionRow,
)


class PostgresDerivedMemoryRepository:
    """Persist only derived/sanitized planes; source messages remain immutable."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def put_message(self, message: SanitizedMessage) -> SanitizedMessage:
        if message.conversation_id is None:
            raise StoreError("durable sanitized messages require canonical conversation identity")
        values = _message_values(message)
        async with self._sessions() as session, session.begin():
            conversation_external_id = await session.scalar(
                select(ConversationRow.external_id).where(
                    ConversationRow.id == message.conversation_id
                )
            )
            if conversation_external_id != message.destination_id:
                raise StoreError(
                    "sanitized message conversation authority does not match destination"
                )
            if message.harness_thread_id is not None:
                thread_id = await session.scalar(
                    select(ThreadRow.id).where(
                        ThreadRow.id == message.harness_thread_id,
                        ThreadRow.organization_id == message.scope.organization_id,
                        ThreadRow.conversation_id == message.conversation_id,
                    )
                )
                if thread_id is None:
                    raise StoreError("sanitized message thread is outside conversation authority")
            inserted = await session.scalar(
                pg_insert(SanitizedMessageRow)
                .values(**values)
                .on_conflict_do_nothing()
                .returning(SanitizedMessageRow.id)
            )
            if inserted is not None:
                return message
            existing = await session.scalar(
                select(SanitizedMessageRow).where(
                    SanitizedMessageRow.conversation_id == message.conversation_id,
                    SanitizedMessageRow.external_event_id == message.external_event_id,
                    SanitizedMessageRow.role == message.role.value,
                )
            )
            if existing is None:
                existing = await session.scalar(
                    select(SanitizedMessageRow).where(SanitizedMessageRow.id == message.id)
                )
            if existing is None or _message_model(existing) != message:
                raise ConcurrencyError("sanitized message identity is immutable")
            return message

    async def list_messages(
        self,
        scope: ScopeKey,
        *,
        conversation_id: str,
        harness_thread_id: str | None = None,
        after: tuple[datetime, str] | None = None,
        limit: int = 100,
    ) -> tuple[SanitizedMessage, ...]:
        if limit < 1 or limit > 500:
            raise ValueError("message page limit must be between 1 and 500")
        statement = select(SanitizedMessageRow).where(
            SanitizedMessageRow.organization_id == scope.organization_id,
            SanitizedMessageRow.strategy_id == scope.strategy_id,
            SanitizedMessageRow.conversation_id == conversation_id,
        )
        if harness_thread_id is not None:
            statement = statement.where(SanitizedMessageRow.harness_thread_id == harness_thread_id)
        if after is not None:
            recorded_at, message_id = after
            statement = statement.where(
                (SanitizedMessageRow.recorded_at > recorded_at)
                | (
                    (SanitizedMessageRow.recorded_at == recorded_at)
                    & (SanitizedMessageRow.id > message_id)
                )
            )
        statement = statement.order_by(
            SanitizedMessageRow.recorded_at,
            SanitizedMessageRow.id,
        ).limit(limit)
        async with self._sessions() as session:
            rows = tuple(await session.scalars(statement))
        return tuple(_message_model(row) for row in rows)

    async def rebuild_summary(
        self,
        scope: ScopeKey,
        *,
        thread_id: str,
        proposal: SummaryProposal,
        available_evidence_ids: frozenset[str] = frozenset(),
    ) -> SummaryRevision:
        async with self._sessions() as session, session.begin():
            previous = await _latest_summary(session, scope, thread_id, lock=True)
            message_ids = tuple(
                await session.scalars(
                    select(SanitizedMessageRow.id).where(
                        SanitizedMessageRow.organization_id == scope.organization_id,
                        SanitizedMessageRow.strategy_id == scope.strategy_id,
                        SanitizedMessageRow.harness_thread_id == thread_id,
                    )
                )
            )
            available = frozenset((*message_ids, *available_evidence_ids))
            summary = make_summary(
                thread_id,
                scope,
                1 if previous is None else previous.version + 1,
                proposal,
                available_source_ids=available,
                previous=previous,
            )
            await _insert_summary(session, summary, frozenset(message_ids))
            return summary

    async def compact_thread_if_needed(
        self,
        scope: ScopeKey,
        *,
        thread_id: str,
        proposal: SummaryProposal,
        policy: CompactionPolicy | None = None,
        available_evidence_ids: frozenset[str] = frozenset(),
    ) -> CompactionResult | None:
        """Append one summary revision for a stable prefix and retain the recent window."""

        async with self._sessions() as session, session.begin():
            previous = await _latest_summary(session, scope, thread_id, lock=True)
            rows = tuple(
                await session.scalars(
                    select(SanitizedMessageRow)
                    .where(
                        SanitizedMessageRow.organization_id == scope.organization_id,
                        SanitizedMessageRow.strategy_id == scope.strategy_id,
                        SanitizedMessageRow.harness_thread_id == thread_id,
                    )
                    .order_by(
                        SanitizedMessageRow.recorded_at,
                        SanitizedMessageRow.id,
                    )
                )
            )
            messages = tuple(_message_model(row) for row in rows)
            window = select_compaction_window(messages, policy or CompactionPolicy())
            if not window.should_compact:
                return None
            if not set(window.compactable_message_ids).issubset(proposal.covered_message_ids):
                raise StoreError("summary proposal omitted the compactable source prefix")
            message_ids = frozenset(message.id for message in messages)
            summary = make_summary(
                thread_id,
                scope,
                1 if previous is None else previous.version + 1,
                proposal,
                available_source_ids=frozenset((*message_ids, *available_evidence_ids)),
                previous=previous,
            )
            await _insert_summary(session, summary, message_ids)
            return compaction_result(summary, messages, window)

    async def latest_summary(self, scope: ScopeKey, *, thread_id: str) -> SummaryRevision | None:
        async with self._sessions() as session:
            return await _latest_summary(session, scope, thread_id, lock=False)

    async def invalidate_summaries_for_messages(
        self, scope: ScopeKey, message_ids: tuple[str, ...]
    ) -> int:
        if not message_ids:
            return 0
        async with self._sessions() as session, session.begin():
            thread_ids = tuple(
                item
                for item in await session.scalars(
                    select(SanitizedMessageRow.harness_thread_id).where(
                        SanitizedMessageRow.organization_id == scope.organization_id,
                        SanitizedMessageRow.strategy_id == scope.strategy_id,
                        SanitizedMessageRow.id.in_(message_ids),
                    )
                )
                if item is not None
            )
            if not thread_ids:
                return 0
            result = await session.execute(
                delete(ThreadSummaryRevisionRow).where(
                    ThreadSummaryRevisionRow.organization_id == scope.organization_id,
                    ThreadSummaryRevisionRow.strategy_id == scope.strategy_id,
                    ThreadSummaryRevisionRow.thread_id.in_(thread_ids),
                )
            )
            return _result_rowcount(result)

    async def put_cache(self, entry: RetrievalCacheEntry) -> RetrievalCacheEntry:
        key_hash = entry.key.digest()
        row_id = f"cache-{key_hash[:58]}"
        async with self._sessions() as session, session.begin():
            inserted = await session.scalar(
                pg_insert(MemoryRetrievalCacheRow)
                .values(
                    id=row_id,
                    organization_id=entry.key.scope.organization_id,
                    strategy_id=entry.key.scope.strategy_id,
                    key_hash=key_hash,
                    generation=entry.key.generation,
                    result_ids=list(entry.record_ids),
                    expires_at=entry.expires_at,
                )
                .on_conflict_do_nothing()
                .returning(MemoryRetrievalCacheRow.id)
            )
            if inserted is not None:
                return entry
            existing = await session.scalar(
                select(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == entry.key.scope.organization_id,
                    MemoryRetrievalCacheRow.strategy_id == entry.key.scope.strategy_id,
                    MemoryRetrievalCacheRow.key_hash == key_hash,
                    MemoryRetrievalCacheRow.generation == entry.key.generation,
                )
            )
            if existing is None or tuple(existing.result_ids) != entry.record_ids:
                raise ConcurrencyError("retrieval cache key produced divergent results")
            return entry

    async def get_cache(
        self,
        key: RetrievalCacheKey,
        *,
        now: datetime,
    ) -> RetrievalCacheEntry | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == key.scope.organization_id,
                    MemoryRetrievalCacheRow.strategy_id == key.scope.strategy_id,
                    MemoryRetrievalCacheRow.key_hash == key.digest(),
                    MemoryRetrievalCacheRow.generation == key.generation,
                    or_(
                        MemoryRetrievalCacheRow.expires_at.is_(None),
                        MemoryRetrievalCacheRow.expires_at > now,
                    ),
                )
            )
        if row is None:
            return None
        return RetrievalCacheEntry(
            key=key,
            record_ids=tuple(row.result_ids),
            expires_at=row.expires_at,
        )

    async def invalidate_cache_for_authority_change(self, scope: ScopeKey) -> int:
        """Access/membership changes safely over-invalidate the affected scope."""

        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == scope.organization_id,
                )
            )
            return _result_rowcount(result)

    async def invalidate_cache_before_generation(
        self, scope: ScopeKey, *, current_generation: int
    ) -> int:
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                delete(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == scope.organization_id,
                    MemoryRetrievalCacheRow.generation < current_generation,
                )
            )
            return _result_rowcount(result)

    async def enqueue_embedding(self, job: EmbeddingJob) -> EmbeddingJob:
        async with self._sessions() as session, session.begin():
            await session.execute(
                pg_insert(MemoryEmbeddingJobRow)
                .values(
                    id=job.id,
                    organization_id=job.scope.organization_id,
                    strategy_id=job.scope.strategy_id,
                    source_plane=job.source_plane.value,
                    source_id=job.source_id,
                    content_hash=job.content_hash,
                    model=job.model,
                    dimensions=job.dimensions,
                    status=job.status,
                    attempts=job.attempts,
                )
                .on_conflict_do_nothing()
            )
            row = await session.scalar(
                select(MemoryEmbeddingJobRow).where(
                    MemoryEmbeddingJobRow.source_id == job.source_id,
                    MemoryEmbeddingJobRow.content_hash == job.content_hash,
                    MemoryEmbeddingJobRow.model == job.model,
                )
            )
            if row is None:
                raise StoreError("embedding job enqueue did not produce a durable row")
            existing = _embedding_model(row)
            immutable = (
                existing.scope,
                existing.source_plane,
                existing.source_id,
                existing.content_hash,
                existing.model,
                existing.dimensions,
            )
            proposed = (
                job.scope,
                job.source_plane,
                job.source_id,
                job.content_hash,
                job.model,
                job.dimensions,
            )
            if immutable != proposed:
                raise ConcurrencyError("embedding work identity is immutable")
            return existing

    async def claim_embedding(
        self,
        scope: ScopeKey,
        *,
        now: datetime,
        reclaim_after: timedelta = timedelta(minutes=5),
        max_attempts: int = 3,
    ) -> EmbeddingJob | None:
        if max_attempts < 1:
            raise ValueError("embedding max attempts must be positive")
        if reclaim_after <= timedelta(0):
            raise ValueError("embedding reclaim interval must be positive")
        if now.utcoffset() is None:
            raise ValueError("embedding claim clock must be timezone-aware")
        reclaim_before = now - reclaim_after
        reclaimable = or_(
            MemoryEmbeddingJobRow.status == "queued",
            and_(
                MemoryEmbeddingJobRow.status == "retry",
                MemoryEmbeddingJobRow.updated_at <= reclaim_before,
            ),
        )
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(MemoryEmbeddingJobRow)
                .where(
                    MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                    MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                    reclaimable,
                    MemoryEmbeddingJobRow.attempts >= max_attempts,
                )
                .values(status="dead", updated_at=now)
            )
            row = await session.scalar(
                select(MemoryEmbeddingJobRow)
                .where(
                    MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                    MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                    reclaimable,
                    MemoryEmbeddingJobRow.attempts < max_attempts,
                )
                .order_by(MemoryEmbeddingJobRow.created_at, MemoryEmbeddingJobRow.id)
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            if row is None:
                return None
            row.attempts += 1
            row.status = "retry"
            row.updated_at = now
            await session.flush()
            return _embedding_model(row)

    async def finish_embedding(
        self,
        scope: ScopeKey,
        *,
        job_id: str,
        expected_attempt: int,
        status: str,
        now: datetime,
    ) -> EmbeddingJob:
        if status not in {"retry", "succeeded", "dead"}:
            raise ValueError("embedding completion status is invalid")
        if now.utcoffset() is None:
            raise ValueError("embedding completion clock must be timezone-aware")
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(MemoryEmbeddingJobRow)
                .where(
                    MemoryEmbeddingJobRow.id == job_id,
                    MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                    MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                )
                .with_for_update()
            )
            if row is None:
                raise NotFoundError("embedding job not found")
            if row.attempts != expected_attempt or row.status != "retry":
                raise ConcurrencyError("embedding job attempt is stale")
            row.status = status
            row.updated_at = now
            await session.flush()
            return _embedding_model(row)


class PostgresMemoryMaintenance:
    """Inspect and physically purge only explicitly confirmed retracted demo records."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def prepare_purge(self, scope: ScopeKey, record_ids: tuple[str, ...]) -> PurgePlan:
        # Validate wildcard/duplicate/batch constraints before touching the database.
        make_purge_plan(scope, record_ids)
        async with self._sessions() as session:
            rows = tuple(
                await session.scalars(
                    select(MemoryRecordRow).where(
                        MemoryRecordRow.organization_id == scope.organization_id,
                        MemoryRecordRow.strategy_id == scope.strategy_id,
                        MemoryRecordRow.id.in_(record_ids),
                    )
                )
            )
        by_id = {row.id: row for row in rows}
        if set(by_id) != set(record_ids):
            raise NotFoundError("one or more purge records were not found in scope")
        if any(row.status != MemoryStatus.RETRACTED.value for row in rows):
            raise StoreError("physical purge requires logically retracted records")
        targets = tuple(
            PurgeTarget(
                record_id=record_id,
                generation=by_id[record_id].generation,
                current_revision=by_id[record_id].current_revision,
            )
            for record_id in record_ids
        )
        return make_purge_plan(scope, record_ids, targets=targets)

    async def execute_purge(
        self,
        plan: PurgePlan,
        *,
        scope: ScopeKey,
        confirmation_token: str,
    ) -> PurgeResult:
        validate_confirmation(plan, confirmation_token, scope=scope)
        if not plan.targets:
            raise StoreError("physical purge requires a versioned dry-run snapshot")
        targets = {item.record_id: item for item in plan.targets}
        async with self._sessions() as session, session.begin():
            rows = tuple(
                await session.scalars(
                    select(MemoryRecordRow)
                    .where(
                        MemoryRecordRow.organization_id == scope.organization_id,
                        MemoryRecordRow.strategy_id == scope.strategy_id,
                        MemoryRecordRow.id.in_(plan.record_ids),
                    )
                    .with_for_update()
                )
            )
            by_id = {row.id: row for row in rows}
            absent = tuple(record_id for record_id in plan.record_ids if record_id not in by_id)
            for record_id, row in by_id.items():
                target = targets[record_id]
                if (
                    row.status != MemoryStatus.RETRACTED.value
                    or row.generation != target.generation
                    or row.current_revision != target.current_revision
                ):
                    raise ConcurrencyError("purge snapshot is stale")
            purged = tuple(record_id for record_id in plan.record_ids if record_id in by_id)
            revision_rows = tuple(
                await session.scalars(
                    select(MemoryRevisionRow).where(
                        MemoryRevisionRow.organization_id == scope.organization_id,
                        MemoryRevisionRow.strategy_id == scope.strategy_id,
                        MemoryRevisionRow.record_id.in_(purged),
                    )
                )
            )
            revision_ids = tuple(row.id for row in revision_rows)
            source_ids = tuple(
                dict.fromkeys(source_id for row in revision_rows for source_id in row.source_ids)
            )
            embedding_result = await session.execute(
                delete(MemoryEmbeddingJobRow).where(
                    MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                    MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                    MemoryEmbeddingJobRow.source_id.in_((*purged, *revision_ids)),
                )
            )
            cache_result = await session.execute(
                delete(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == scope.organization_id,
                )
            )
            revision_result = await session.execute(
                delete(MemoryRevisionRow).where(
                    MemoryRevisionRow.organization_id == scope.organization_id,
                    MemoryRevisionRow.strategy_id == scope.strategy_id,
                    MemoryRevisionRow.record_id.in_(purged),
                )
            )
            await session.execute(
                delete(MemoryRecordRow).where(
                    MemoryRecordRow.organization_id == scope.organization_id,
                    MemoryRecordRow.strategy_id == scope.strategy_id,
                    MemoryRecordRow.id.in_(purged),
                )
            )
            await session.flush()
            deleted_sources = 0
            deleted_source_embedding_jobs = 0
            for source_id in source_ids:
                still_used = await session.scalar(
                    select(MemoryRevisionRow.id)
                    .where(MemoryRevisionRow.source_ids.contains([source_id]))
                    .limit(1)
                )
                if still_used is None:
                    source_embedding_result = await session.execute(
                        delete(MemoryEmbeddingJobRow).where(
                            MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                            MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                            MemoryEmbeddingJobRow.source_id == source_id,
                        )
                    )
                    deleted_source_embedding_jobs += _result_rowcount(source_embedding_result)
                    source_result = await session.execute(
                        delete(MemorySourceRow).where(
                            MemorySourceRow.id == source_id,
                            MemorySourceRow.organization_id == scope.organization_id,
                            MemorySourceRow.strategy_id == scope.strategy_id,
                        )
                    )
                    deleted_sources += _result_rowcount(source_result)
            return PurgeResult(
                scope=scope,
                manifest_hash=plan.manifest_hash,
                purged_record_ids=purged,
                already_absent_record_ids=absent,
                deleted_revision_count=_result_rowcount(revision_result),
                deleted_source_count=deleted_sources,
                invalidated_cache_count=_result_rowcount(cache_result),
                deleted_embedding_job_count=(
                    _result_rowcount(embedding_result) + deleted_source_embedding_jobs
                ),
            )

    async def health(self, scope: ScopeKey, *, now: datetime) -> MaintenanceHealth:
        async with self._sessions() as session:
            expired_active = await session.scalar(
                select(func.count())
                .select_from(MemoryRevisionRow)
                .join(
                    MemoryRecordRow,
                    (MemoryRecordRow.id == MemoryRevisionRow.record_id)
                    & (MemoryRecordRow.current_revision == MemoryRevisionRow.number),
                )
                .where(
                    MemoryRecordRow.organization_id == scope.organization_id,
                    MemoryRecordRow.strategy_id == scope.strategy_id,
                    MemoryRecordRow.status.in_(("active", "contested")),
                    or_(
                        MemoryRevisionRow.expires_at <= now,
                        MemoryRevisionRow.valid_until <= now,
                    ),
                )
            )
            job_counts = dict(
                (
                    str(status),
                    int(count),
                )
                for status, count in (
                    await session.execute(
                        select(MemoryEmbeddingJobRow.status, func.count())
                        .where(
                            MemoryEmbeddingJobRow.organization_id == scope.organization_id,
                            MemoryEmbeddingJobRow.strategy_id == scope.strategy_id,
                        )
                        .group_by(MemoryEmbeddingJobRow.status)
                    )
                ).all()
            )
            cache_count = await session.scalar(
                select(func.count()).where(
                    MemoryRetrievalCacheRow.organization_id == scope.organization_id,
                    MemoryRetrievalCacheRow.strategy_id == scope.strategy_id,
                )
            )
        return MaintenanceHealth(
            scope=scope,
            expired_active_records=int(expired_active or 0),
            queued_embedding_jobs=job_counts.get("queued", 0),
            retry_embedding_jobs=job_counts.get("retry", 0),
            dead_embedding_jobs=job_counts.get("dead", 0),
            retrieval_cache_entries=int(cache_count or 0),
        )


async def _latest_summary(
    session: AsyncSession,
    scope: ScopeKey,
    thread_id: str,
    *,
    lock: bool,
) -> SummaryRevision | None:
    statement = (
        select(ThreadSummaryRevisionRow)
        .where(
            ThreadSummaryRevisionRow.organization_id == scope.organization_id,
            ThreadSummaryRevisionRow.strategy_id == scope.strategy_id,
            ThreadSummaryRevisionRow.thread_id == thread_id,
        )
        .order_by(ThreadSummaryRevisionRow.revision.desc())
        .limit(1)
    )
    if lock:
        statement = statement.with_for_update()
    row = await session.scalar(statement)
    if row is None:
        return None
    try:
        payload = json.loads(row.content)
        proposal = SummaryProposal.model_validate(payload["proposal"])
        digest = str(payload["summary_digest"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise StoreError("durable summary payload is malformed") from exc
    return SummaryRevision(
        thread_id=row.thread_id,
        scope=scope,
        version=row.revision,
        proposal=proposal,
        source_ids=tuple(row.source_message_ids),
        digest=digest,
    )


async def _insert_summary(
    session: AsyncSession,
    summary: SummaryRevision,
    available_message_ids: frozenset[str],
) -> None:
    if not set(summary.proposal.covered_message_ids).issubset(available_message_ids):
        raise StoreError("summary references a message outside the exact thread")
    payload = json.dumps(
        {
            "proposal": json.loads(render_summary_content(summary)),
            "summary_digest": summary.digest,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    session.add(
        ThreadSummaryRevisionRow(
            id=f"summary-{summary.digest[:56]}",
            organization_id=summary.scope.organization_id,
            strategy_id=summary.scope.strategy_id,
            thread_id=summary.thread_id,
            source_message_ids=list(summary.source_ids),
            revision=summary.version,
            content=payload,
            content_hash=hashlib.sha256(payload.encode()).hexdigest(),
        )
    )
    try:
        await session.flush()
    except Exception as exc:
        raise ConcurrencyError("summary revision append conflicted") from exc


def _message_values(message: SanitizedMessage) -> dict[str, object]:
    return {
        "id": message.id,
        "organization_id": message.scope.organization_id,
        "strategy_id": message.scope.strategy_id,
        "destination_id": message.destination_id,
        "external_event_id": message.external_event_id,
        "text": message.text,
        "content_hash": message.content_hash,
        "recorded_at": message.recorded_at,
        "conversation_id": message.conversation_id,
        "harness_thread_id": message.harness_thread_id,
        "actor_id": message.actor_id,
        "role": message.role.value,
        "provider_message_ts": message.provider_message_ts,
        "context_access_hash": message.context_access_hash,
    }


def _message_model(row: SanitizedMessageRow) -> SanitizedMessage:
    return SanitizedMessage(
        id=row.id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        destination_id=row.destination_id,
        external_event_id=row.external_event_id,
        text=row.text,
        content_hash=row.content_hash,
        recorded_at=row.recorded_at,
        conversation_id=row.conversation_id,
        harness_thread_id=row.harness_thread_id,
        actor_id=row.actor_id,
        role=MessageRole(row.role),
        provider_message_ts=row.provider_message_ts,
        context_access_hash=row.context_access_hash,
    )


def _embedding_model(row: MemoryEmbeddingJobRow) -> EmbeddingJob:
    return EmbeddingJob(
        id=row.id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        source_plane=DataPlane(row.source_plane),
        source_id=row.source_id,
        content_hash=row.content_hash,
        model=row.model,
        dimensions=row.dimensions,
        status=row.status,
        attempts=row.attempts,
    )


def _rowcount(value: int | None) -> int:
    return 0 if value is None or value < 0 else value


def _result_rowcount(result: object) -> int:
    value = getattr(result, "rowcount", None)
    return _rowcount(value if isinstance(value, int) else None)
