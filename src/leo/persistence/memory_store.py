"""Scope-first Postgres repository for append-only memory lifecycle state."""

from __future__ import annotations

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.memory.lifecycle import next_record, validate_append_revision, validate_initial_revision
from leo.memory.models import MemoryKind, MemoryRecord, MemoryRevision, MemorySource, MemoryStatus
from leo.memory.ports import MemoryStore
from leo.persistence.schema import (
    MemoryCapabilityHandleRow,
    MemoryRecordRow,
    MemoryRetrievalCacheRow,
    MemoryRevisionRow,
    MemorySourceRow,
)


class PostgresMemoryStore(MemoryStore):
    """Persist memory records and revisions with database-serialized appends."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def create(
        self,
        record: MemoryRecord,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...],
    ) -> MemoryRecord:
        validate_initial_revision(record, revision)
        _validate_sources(record, revision, sources)
        try:
            async with self._sessions() as session, session.begin():
                session.add(_record_row(record))
                for source in sources:
                    session.add(_source_row(source))
                await session.flush()
                session.add(_revision_row(record.scope, revision))
                await session.flush()
                await _invalidate_scope_cache(session, record.scope)
        except IntegrityError as exc:
            raise ConcurrencyError("memory record or source already exists") from exc
        return record

    async def append_revision(
        self,
        scope: ScopeKey,
        record_id: str,
        expected_revision: int,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...] = (),
    ) -> MemoryRecord:
        if revision.record_id != record_id or revision.number != expected_revision + 1:
            raise StoreError("memory revision does not match the expected revision")
        try:
            async with self._sessions() as session, session.begin():
                row = await session.scalar(
                    select(MemoryRecordRow)
                    .where(
                        MemoryRecordRow.id == record_id,
                        MemoryRecordRow.organization_id == scope.organization_id,
                        MemoryRecordRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise NotFoundError("memory record not found")
                record = _record_model(row)
                if record.current_revision != expected_revision:
                    raise ConcurrencyError("stale memory revision")
                validate_append_revision(record, expected_revision, revision)
                if len({source.id for source in sources}) != len(sources):
                    raise StoreError("duplicate memory source")
                if any(
                    source.scope != record.scope
                    or source.visibility != revision.visibility
                    or source.namespace_id != revision.namespace_id
                    for source in sources
                ):
                    raise StoreError("memory source is outside the revision scope")
                current_row = await session.scalar(
                    select(MemoryRevisionRow).where(
                        MemoryRevisionRow.record_id == record.id,
                        MemoryRevisionRow.number == record.current_revision,
                        MemoryRevisionRow.organization_id == scope.organization_id,
                        MemoryRevisionRow.strategy_id == scope.strategy_id,
                    )
                )
                if current_row is None:
                    raise StoreError("current memory revision is missing")
                new_source_ids = {source.id for source in sources}
                expected_source_ids = set(current_row.source_ids) | new_source_ids
                if set(revision.source_ids) != expected_source_ids:
                    raise StoreError("memory revision must retain prior and current provenance")
                for source in sources:
                    session.add(_source_row(source))
                if sources:
                    await session.flush()
                source_rows = await session.scalars(
                    select(MemorySourceRow).where(
                        MemorySourceRow.id.in_(revision.source_ids),
                        MemorySourceRow.organization_id == scope.organization_id,
                        MemorySourceRow.strategy_id == scope.strategy_id,
                    )
                )
                _validate_source_rows(record, revision, tuple(source_rows))
                session.add(_revision_row(scope, revision))
                updated = next_record(record, revision)
                row.current_revision = updated.current_revision
                row.generation = updated.generation
                row.status = updated.status.value
                await session.flush()
                await _invalidate_scope_cache(session, scope)
                return updated
        except IntegrityError as exc:
            raise ConcurrencyError("memory revision already exists") from exc

    async def current(self, scope: ScopeKey, record_id: str) -> MemoryRevision | None:
        async with self._sessions() as session:
            record = await session.scalar(
                select(MemoryRecordRow).where(
                    MemoryRecordRow.id == record_id,
                    MemoryRecordRow.organization_id == scope.organization_id,
                    MemoryRecordRow.strategy_id == scope.strategy_id,
                )
            )
            if record is None:
                raise NotFoundError("memory record not found")
            if record.status == MemoryStatus.RETRACTED.value:
                return None
            revision = await session.scalar(
                select(MemoryRevisionRow).where(
                    MemoryRevisionRow.record_id == record.id,
                    MemoryRevisionRow.number == record.current_revision,
                    MemoryRevisionRow.organization_id == scope.organization_id,
                    MemoryRevisionRow.strategy_id == scope.strategy_id,
                )
            )
            if revision is None:
                raise StoreError("current memory revision is missing")
            return _revision_model(revision)

    async def forget(self, scope: ScopeKey, record_id: str, reason: str) -> MemoryRecord:
        if not reason.strip():
            raise StoreError("forget reason must be non-empty")
        try:
            async with self._sessions() as session, session.begin():
                row = await session.scalar(
                    select(MemoryRecordRow)
                    .where(
                        MemoryRecordRow.id == record_id,
                        MemoryRecordRow.organization_id == scope.organization_id,
                        MemoryRecordRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update()
                )
                if row is None:
                    raise NotFoundError("memory record not found")
                record = _record_model(row)
                if record.status is MemoryStatus.RETRACTED:
                    return record
                current_row = await session.scalar(
                    select(MemoryRevisionRow).where(
                        MemoryRevisionRow.record_id == record.id,
                        MemoryRevisionRow.number == record.current_revision,
                    )
                )
                if current_row is None:
                    raise StoreError("current memory revision is missing")
                current = _revision_model(current_row)
                revision = current.model_copy(
                    update={
                        "id": f"{current.id}:forget:{record.generation + 1}",
                        "number": current.number + 1,
                        "status": MemoryStatus.RETRACTED,
                        "reason": reason,
                        "supersedes_revision": current.number,
                    }
                )
                session.add(_revision_row(scope, revision))
                updated = next_record(record, revision)
                row.current_revision = updated.current_revision
                row.generation = updated.generation
                row.status = updated.status.value
                await session.flush()
                await _invalidate_scope_cache(session, scope)
                return updated
        except IntegrityError as exc:
            raise ConcurrencyError("memory forget conflicted with another revision") from exc


def _validate_sources(
    record: MemoryRecord,
    revision: MemoryRevision,
    sources: tuple[MemorySource, ...],
) -> None:
    if len({source.id for source in sources}) != len(sources):
        raise StoreError("duplicate memory source")
    if set(revision.source_ids) != {source.id for source in sources}:
        raise StoreError("memory revision source provenance is incomplete")
    if any(
        source.scope != record.scope
        or source.visibility != revision.visibility
        or source.namespace_id != revision.namespace_id
        for source in sources
    ):
        raise StoreError("memory source is outside the revision scope")


def _validate_source_rows(
    record: MemoryRecord,
    revision: MemoryRevision,
    sources: tuple[MemorySourceRow, ...],
) -> None:
    if len(sources) != len(set(revision.source_ids)):
        raise StoreError("memory revision references an unknown source")
    models = tuple(_source_model(source) for source in sources)
    _validate_sources(record, revision, models)


def _record_row(item: MemoryRecord) -> MemoryRecordRow:
    return MemoryRecordRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        kind=item.kind.value,
        visibility=item.visibility.value,
        namespace_id=item.namespace_id,
        current_revision=item.current_revision,
        generation=item.generation,
        status=item.status.value,
        created_at=item.created_at,
    )


def _source_row(item: MemorySource) -> MemorySourceRow:
    return MemorySourceRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        source_kind=item.source_kind,
        reference=item.reference,
        visibility=item.visibility.value,
        namespace_id=item.namespace_id,
    )


def _revision_row(scope: ScopeKey, item: MemoryRevision) -> MemoryRevisionRow:
    return MemoryRevisionRow(
        id=item.id,
        record_id=item.record_id,
        organization_id=scope.organization_id,
        strategy_id=scope.strategy_id,
        number=item.number,
        content=item.content,
        content_hash=item.content_hash,
        source_ids=list(item.source_ids),
        visibility=item.visibility.value,
        namespace_id=item.namespace_id,
        sensitivity=item.sensitivity,
        valid_from=item.valid_from,
        valid_until=item.valid_until,
        recorded_at=item.recorded_at,
        expires_at=item.expires_at,
        status=item.status.value,
        actor_id=item.actor_id,
        reason=item.reason,
        supersedes_revision=item.supersedes_revision,
    )


def _record_model(row: MemoryRecordRow) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        scope=ScopeKey(organization_id=row.organization_id, strategy_id=row.strategy_id),
        kind=MemoryKind(row.kind),
        visibility=row.visibility,
        namespace_id=row.namespace_id,
        current_revision=row.current_revision,
        generation=row.generation,
        status=MemoryStatus(row.status),
        created_at=row.created_at,
    )


def _source_model(row: MemorySourceRow) -> MemorySource:
    return MemorySource(
        id=row.id,
        scope=ScopeKey(organization_id=row.organization_id, strategy_id=row.strategy_id),
        source_kind=row.source_kind,
        reference=row.reference,
        visibility=row.visibility,
        namespace_id=row.namespace_id,
    )


def _revision_model(row: MemoryRevisionRow) -> MemoryRevision:
    return MemoryRevision(
        id=row.id,
        record_id=row.record_id,
        number=row.number,
        content=row.content,
        content_hash=row.content_hash,
        source_ids=tuple(str(source_id) for source_id in row.source_ids),
        visibility=row.visibility,
        namespace_id=row.namespace_id,
        sensitivity=row.sensitivity,
        valid_from=row.valid_from,
        valid_until=row.valid_until,
        recorded_at=row.recorded_at,
        expires_at=row.expires_at,
        status=MemoryStatus(row.status),
        actor_id=row.actor_id,
        reason=row.reason,
        supersedes_revision=row.supersedes_revision,
    )


async def _invalidate_scope_cache(session: AsyncSession, scope: ScopeKey) -> None:
    # One conversation-local revision can feed a 1:1 DM run under another optional domain
    # strategy. D-054 therefore requires workspace-wide cache invalidation on generation change.
    await session.execute(
        delete(MemoryRetrievalCacheRow).where(
            MemoryRetrievalCacheRow.organization_id == scope.organization_id,
        )
    )
    await session.execute(
        update(MemoryCapabilityHandleRow)
        .where(
            MemoryCapabilityHandleRow.organization_id == scope.organization_id,
            MemoryCapabilityHandleRow.invalidated_at.is_(None),
        )
        .values(
            invalidated_at=func.now(),
            invalidation_reason="memory_generation_changed",
            updated_at=func.now(),
        )
    )
