"""Non-successful runs for failure-mode investigation."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import PageParams, get_session
from leo.persistence.schema import RunRow, TaskRow

router = APIRouter()

_NON_FAILURE_STATUSES = ("completed", "queued", "running", "requires_action")


@router.get("/failures")
async def list_failures(
    terminal_reason: str | None = None,
    status: str | None = None,
    page: PageParams = Depends(),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    filters: list[ColumnElement[bool]] = [RunRow.status.notin_(_NON_FAILURE_STATUSES)]
    if terminal_reason:
        filters.append(RunRow.terminal_reason == terminal_reason)
    if status:
        filters.append(RunRow.status == status)

    count_stmt = (
        select(func.count(RunRow.id))
        .select_from(RunRow)
        .join(TaskRow, TaskRow.id == RunRow.task_id)
        .where(*filters)
    )
    total = await session.scalar(count_stmt)

    list_stmt = (
        select(RunRow, TaskRow.objective, TaskRow.last_error, TaskRow.attempt_count)
        .join(TaskRow, TaskRow.id == RunRow.task_id)
        .where(*filters)
        .order_by(RunRow.updated_at.desc())
        .limit(page.limit)
        .offset(page.offset)
    )
    rows = (await session.execute(list_stmt)).all()

    items = [
        {
            "run_id": run.id,
            "task_id": run.task_id,
            "status": run.status,
            "phase": run.phase,
            "terminal_reason": run.terminal_reason,
            "task_objective": objective,
            "task_last_error": last_error,
            "attempt_count": attempt_count,
            "updated_at": run.updated_at,
        }
        for run, objective, last_error, attempt_count in rows
    ]
    return {"items": items, "total": total or 0, "limit": page.limit, "offset": page.offset}
