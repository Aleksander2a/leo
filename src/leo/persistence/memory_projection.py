"""Scope-first Postgres adapter for read-only memory projections."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.projection import (
    MemoryProjectionPage,
    ProjectionRequest,
    render_memory_projection_page,
)
from leo.persistence.schema import MemoryRecordRow, MemoryRevisionRow


class PostgresMemoryProjectionService:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def render_page(
        self,
        request: ProjectionRequest,
        *,
        as_of: datetime,
    ) -> MemoryProjectionPage:
        if as_of.utcoffset() is None:
            raise ValueError("memory projection clock must be timezone-aware")
        authorized = tuple(
            and_(
                MemoryRecordRow.visibility == item.visibility.value,
                MemoryRecordRow.namespace_id == item.namespace_id,
                MemoryRevisionRow.visibility == item.visibility.value,
                MemoryRevisionRow.namespace_id == item.namespace_id,
            )
            for item in request.authorized_namespaces
        )
        statement = (
            select(MemoryRecordRow, MemoryRevisionRow)
            .join(
                MemoryRevisionRow,
                (MemoryRevisionRow.record_id == MemoryRecordRow.id)
                & (MemoryRevisionRow.number == MemoryRecordRow.current_revision),
            )
            .where(
                MemoryRecordRow.organization_id == request.scope.organization_id,
                MemoryRecordRow.status.in_(("active", "contested")),
                MemoryRevisionRow.status.in_(("active", "contested")),
                MemoryRevisionRow.valid_from <= as_of,
                or_(
                    MemoryRevisionRow.valid_until.is_(None),
                    MemoryRevisionRow.valid_until > as_of,
                ),
                or_(
                    MemoryRevisionRow.expires_at.is_(None),
                    MemoryRevisionRow.expires_at > as_of,
                ),
                or_(*authorized),
            )
            .order_by(MemoryRecordRow.id, MemoryRevisionRow.number)
            .limit(10_001)
        )
        async with self._sessions() as session, session.begin():
            rows = (await session.execute(statement)).all()
        if len(rows) > 10_000:
            raise RuntimeError("memory projection authorized set exceeds the demo bound")
        records = tuple(
            (_record_model(record), _revision_model(revision)) for record, revision in rows
        )
        return render_memory_projection_page(records, request)


def _record_model(row: MemoryRecordRow) -> MemoryRecord:
    return MemoryRecord(
        id=row.id,
        scope={
            "organization_id": row.organization_id,
            "strategy_id": row.strategy_id,
        },
        kind=MemoryKind(row.kind),
        visibility=MemoryVisibility(row.visibility),
        namespace_id=row.namespace_id,
        current_revision=row.current_revision,
        generation=row.generation,
        status=MemoryStatus(row.status),
        created_at=row.created_at,
    )


def _revision_model(row: MemoryRevisionRow) -> MemoryRevision:
    return MemoryRevision(
        id=row.id,
        record_id=row.record_id,
        number=row.number,
        content=row.content,
        content_hash=row.content_hash,
        source_ids=tuple(row.source_ids),
        visibility=MemoryVisibility(row.visibility),
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
