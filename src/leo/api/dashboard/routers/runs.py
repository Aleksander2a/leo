"""List and detail endpoints for individual runs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import ColumnElement, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import PageParams, get_session
from leo.persistence.schema import (
    ClaimRow,
    DeliveryOutboxRow,
    ObservationRow,
    RunEventRow,
    RunRow,
    TaskRow,
    ThreadRow,
)

router = APIRouter()


@router.get("/runs")
async def list_runs(
    status: str | None = None,
    phase: str | None = None,
    task_status: str | None = None,
    page: PageParams = Depends(),
    session: AsyncSession = Depends(get_session),
) -> dict[str, Any]:
    filters: list[ColumnElement[bool]] = []
    if status:
        filters.append(RunRow.status == status)
    if phase:
        filters.append(RunRow.phase == phase)
    if task_status:
        filters.append(TaskRow.status == task_status)

    count_stmt = (
        select(func.count(RunRow.id))
        .select_from(RunRow)
        .join(TaskRow, TaskRow.id == RunRow.task_id)
    )
    list_stmt = select(RunRow, TaskRow.objective, TaskRow.status).join(
        TaskRow, TaskRow.id == RunRow.task_id
    )
    if filters:
        count_stmt = count_stmt.where(*filters)
        list_stmt = list_stmt.where(*filters)

    total = await session.scalar(count_stmt)
    rows = (
        await session.execute(
            list_stmt.order_by(RunRow.created_at.desc()).limit(page.limit).offset(page.offset)
        )
    ).all()

    items = [
        _run_summary(run, objective, task_status_value)
        for run, objective, task_status_value in rows
    ]
    return {"items": items, "total": total or 0, "limit": page.limit, "offset": page.offset}


@router.get("/runs/{run_id}")
async def get_run_detail(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    run = await session.get(RunRow, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    task = await session.get(TaskRow, run.task_id)
    thread = await session.get(ThreadRow, task.thread_id) if task is not None else None

    observations = (
        (
            await session.execute(
                select(ObservationRow)
                .where(ObservationRow.run_id == run_id)
                .order_by(ObservationRow.observed_at)
            )
        )
        .scalars()
        .all()
    )
    claims = (
        (await session.execute(select(ClaimRow).where(ClaimRow.run_id == run_id))).scalars().all()
    )
    deliveries = (
        (
            await session.execute(
                select(DeliveryOutboxRow)
                .where(DeliveryOutboxRow.run_id == run_id)
                .order_by(DeliveryOutboxRow.created_at)
            )
        )
        .scalars()
        .all()
    )
    event_count = await session.scalar(
        select(func.count()).select_from(RunEventRow).where(RunEventRow.run_id == run_id)
    )

    return {
        "run": _run_fields(run),
        "task": _task_fields(task) if task is not None else None,
        "thread": _thread_fields(thread) if thread is not None else None,
        "observations": [_observation_summary(observation) for observation in observations],
        "claims": [
            {
                "id": claim.id,
                "kind": claim.kind,
                "statement": claim.statement,
                "observation_ids": claim.observation_ids,
            }
            for claim in claims
        ],
        "deliveries": [_delivery_summary(delivery) for delivery in deliveries],
        "event_count": event_count or 0,
    }


def _run_summary(run: RunRow, task_objective: str, task_status: str) -> dict[str, Any]:
    usage = run.usage or {}
    return {
        "id": run.id,
        "task_id": run.task_id,
        "status": run.status,
        "phase": run.phase,
        "iteration": run.iteration,
        "task_objective": task_objective,
        "task_status": task_status,
        "started_at": run.started_at,
        "terminal_reason": run.terminal_reason,
        "total_tokens": usage.get("total_tokens"),
        "cost": usage.get("cost"),
        "created_at": run.created_at,
    }


def _run_fields(run: RunRow) -> dict[str, Any]:
    return {
        "id": run.id,
        "task_id": run.task_id,
        "organization_id": run.organization_id,
        "strategy_id": run.strategy_id,
        "status": run.status,
        "phase": run.phase,
        "iteration": run.iteration,
        "limits": run.limits,
        "usage": run.usage,
        "started_at": run.started_at,
        "deadline_at": run.deadline_at,
        "final_output": run.final_output,
        "terminal_reason": run.terminal_reason,
        "created_at": run.created_at,
        "updated_at": run.updated_at,
    }


def _task_fields(task: TaskRow) -> dict[str, Any]:
    return {
        "id": task.id,
        "thread_id": task.thread_id,
        "objective": task.objective,
        "parent_task_id": task.parent_task_id,
        "continuation_kind": task.continuation_kind,
        "status": task.status,
        "final_output": task.final_output,
        "verifier_feedback": task.verifier_feedback,
        "attempt_count": task.attempt_count,
        "last_error": task.last_error,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
    }


def _thread_fields(thread: ThreadRow) -> dict[str, Any]:
    return {
        "id": thread.id,
        "origin_provider": thread.origin_provider,
        "external_thread_id": thread.external_thread_id,
        "external_channel_id": thread.external_channel_id,
        "conversation_id": thread.conversation_id,
        "created_at": thread.created_at,
    }


def _delivery_summary(delivery: DeliveryOutboxRow) -> dict[str, Any]:
    return {
        "id": delivery.id,
        "kind": delivery.kind,
        "state": delivery.state,
        "destination_channel_id": delivery.destination_channel_id,
        "destination_thread_ts": delivery.destination_thread_ts,
        "attempt_count": delivery.attempt_count,
        "receipt_message_ts": delivery.receipt_message_ts,
        "last_error": delivery.last_error,
        "created_at": delivery.created_at,
        "updated_at": delivery.updated_at,
    }


def _observation_summary(observation: ObservationRow) -> dict[str, Any]:
    return {
        "id": observation.id,
        "tool_call_id": observation.tool_call_id,
        "kind": observation.kind,
        "data": observation.data,
        "source": observation.source or {},
        "status": observation.status,
        "quality": observation.quality,
        "observed_at": observation.observed_at,
        "expires_at": observation.expires_at,
        "rejection_code": observation.rejection_code,
    }
