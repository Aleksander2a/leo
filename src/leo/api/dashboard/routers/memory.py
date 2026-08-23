"""Memory record browser and per-record append-only write history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import PageParams, get_session
from leo.persistence.schema import MemoryRecordRow, MemoryRevisionRow, MemorySourceRow

router = APIRouter()

_PREVIEW_LENGTH = 280


@router.get("/memory/records")
async def list_memory_records(
    kind: str | None = None,
    visibility: str | None = None,
    namespace_id: str | None = None,
    status: str | None = None,
    page: PageParams = Depends(),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    filters: list[ColumnElement[bool]] = []
    if kind:
        filters.append(MemoryRecordRow.kind == kind)
    if visibility:
        filters.append(MemoryRecordRow.visibility == visibility)
    if namespace_id:
        filters.append(MemoryRecordRow.namespace_id == namespace_id)
    if status:
        filters.append(MemoryRecordRow.status == status)

    count_stmt = select(func.count()).select_from(MemoryRecordRow)
    list_stmt = select(MemoryRecordRow)
    if filters:
        count_stmt = count_stmt.where(*filters)
        list_stmt = list_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    records = (
        (
            await session.execute(
                list_stmt.order_by(MemoryRecordRow.created_at.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )

    items = [await _record_summary(session, record) for record in records]
    return {"items": items, "total": total or 0, "limit": page.limit, "offset": page.offset}


@router.get("/memory/records/{record_id}")
async def get_memory_record(
    record_id: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    record = await session.get(MemoryRecordRow, record_id)
    if record is None:
        raise HTTPException(status_code=404, detail="memory record not found")

    revisions = (
        (
            await session.execute(
                select(MemoryRevisionRow)
                .where(MemoryRevisionRow.record_id == record_id)
                .order_by(MemoryRevisionRow.number)
            )
        )
        .scalars()
        .all()
    )

    source_ids = sorted(
        {source_id for revision in revisions for source_id in revision.source_ids or ()}
    )
    sources = (
        (await session.execute(select(MemorySourceRow).where(MemorySourceRow.id.in_(source_ids))))
        .scalars()
        .all()
        if source_ids
        else []
    )

    return {
        "record": {
            "id": record.id,
            "kind": record.kind,
            "visibility": record.visibility,
            "namespace_id": record.namespace_id,
            "current_revision": record.current_revision,
            "generation": record.generation,
            "status": record.status,
            "created_at": record.created_at,
            "scope_label": _scope_label(record.visibility, record.namespace_id),
        },
        "sources": [
            {
                "id": source.id,
                "source_kind": source.source_kind,
                "reference": source.reference,
                "visibility": source.visibility,
                "namespace_id": source.namespace_id,
            }
            for source in sources
        ],
        "revisions": [
            {
                "number": revision.number,
                "content": revision.content,
                "content_hash": revision.content_hash,
                "source_ids": revision.source_ids,
                "visibility": revision.visibility,
                "sensitivity": revision.sensitivity,
                "valid_from": revision.valid_from,
                "valid_until": revision.valid_until,
                "recorded_at": revision.recorded_at,
                "expires_at": revision.expires_at,
                "status": revision.status,
                "actor_id": revision.actor_id,
                "reason": revision.reason,
                "supersedes_revision": revision.supersedes_revision,
                "source_type": revision.source_type,
            }
            for revision in revisions
        ],
    }


def _scope_label(visibility: str, namespace_id: str) -> str:
    """Human-readable isolation-boundary label for the memory inspector.

    Mirrors ``leo.memory.navigation.source_conversation_label``'s two special
    cases; the dashboard additionally distinguishes conversation/channel scope
    by name since an operator reading this has no other context for it.
    """

    if visibility == "actor_private":
        return "Private to one actor"
    if visibility == "thread_local":
        return f"Thread {namespace_id}"
    if visibility in {"conversation_local", "channel_local"}:
        return f"Conversation/DM {namespace_id}"
    if visibility == "strategy_shared":
        return "Shared across the strategy"
    if visibility == "organization_shared":
        return "Shared across the organization"
    return namespace_id


async def _record_summary(session: AsyncSession, record: MemoryRecordRow) -> dict[str, Any]:
    current = (
        await session.execute(
            select(
                MemoryRevisionRow.content,
                MemoryRevisionRow.recorded_at,
                MemoryRevisionRow.source_type,
            ).where(
                MemoryRevisionRow.record_id == record.id,
                MemoryRevisionRow.number == record.current_revision,
            )
        )
    ).first()
    content_preview = current[0][:_PREVIEW_LENGTH] if current is not None else None
    last_recorded_at = current[1] if current is not None else None
    source_type = current[2] if current is not None else None
    return {
        "id": record.id,
        "kind": record.kind,
        "visibility": record.visibility,
        "namespace_id": record.namespace_id,
        "current_revision": record.current_revision,
        "generation": record.generation,
        "status": record.status,
        "created_at": record.created_at,
        "content_preview": content_preview,
        "last_recorded_at": last_recorded_at,
        "source_type": source_type,
        "scope_label": _scope_label(record.visibility, record.namespace_id),
    }
