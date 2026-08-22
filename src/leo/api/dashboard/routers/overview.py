"""Aggregate KPIs across all runs, tasks, tools, memory, and delivery."""

from __future__ import annotations

from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from leo.api.dashboard.deps import get_session
from leo.persistence.schema import (
    DeliveryOutboxRow,
    MemoryRevisionRow,
    RunEventRow,
    RunRow,
    TaskRow,
)

router = APIRouter()

_TOOL_EVENT_TYPES = ("tool_started", "tool_completed", "tool_failed")
_TERMINAL_NON_SUCCESS_STATUSES = ("failed", "cancelled", "timed_out", "budget_exhausted")
_LATENCY_ELIGIBLE_STATUSES = ("completed", "failed", "cancelled", "timed_out", "budget_exhausted")


@router.get("/overview")
async def get_overview(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    run_status_counts = await _status_counts(session, RunRow.status)
    task_status_counts = await _status_counts(session, TaskRow.status)
    delivery_state_counts = await _status_counts(session, DeliveryOutboxRow.state)

    tool_event_types = (
        (
            await session.execute(
                select(RunEventRow.type).where(RunEventRow.type.in_(_TOOL_EVENT_TYPES))
            )
        )
        .scalars()
        .all()
    )
    tool_event_counts = Counter(tool_event_types)
    started = tool_event_counts.get("tool_started", 0)
    completed = tool_event_counts.get("tool_completed", 0)
    failed = tool_event_counts.get("tool_failed", 0)
    finished = completed + failed
    tool_call_success_rate = (completed / finished) if finished else None

    usage_rows = (await session.execute(select(RunRow.usage))).scalars().all()
    total_cost = 0.0
    total_tokens = 0
    total_model_calls = 0
    total_tool_calls = 0
    have_cost = False
    have_tokens = False
    for raw_usage in usage_rows:
        usage = raw_usage or {}
        cost = usage.get("cost")
        if isinstance(cost, int | float):
            total_cost += float(cost)
            have_cost = True
        tokens = usage.get("total_tokens")
        if isinstance(tokens, int | float):
            total_tokens += int(tokens)
            have_tokens = True
        model_calls = usage.get("model_calls")
        if isinstance(model_calls, int | float):
            total_model_calls += int(model_calls)
        tool_calls = usage.get("tool_calls")
        if isinstance(tool_calls, int | float):
            total_tool_calls += int(tool_calls)

    memory_writes_total = await session.scalar(select(func.count()).select_from(MemoryRevisionRow))

    context_payloads = (
        (
            await session.execute(
                select(RunEventRow.payload).where(RunEventRow.type == "context_built")
            )
        )
        .scalars()
        .all()
    )
    memory_pages: set[str] = set()
    for raw_payload in context_payloads:
        manifest = (raw_payload or {}).get("source_manifest")
        if not isinstance(manifest, dict):
            continue
        for key in ("included_source_ids", "excluded_source_ids"):
            for source_id in manifest.get(key) or ():
                if isinstance(source_id, str) and source_id.startswith("memory-projection:"):
                    memory_pages.add(source_id)

    failure_reasons_raw = (
        (
            await session.execute(
                select(RunRow.terminal_reason).where(
                    RunRow.status.in_(_TERMINAL_NON_SUCCESS_STATUSES)
                )
            )
        )
        .scalars()
        .all()
    )
    failure_reasons = Counter(reason or "unspecified" for reason in failure_reasons_raw)

    latency_rows = (
        await session.execute(
            select(RunRow.started_at, RunRow.updated_at).where(
                RunRow.started_at.is_not(None),
                RunRow.status.in_(_LATENCY_ELIGIBLE_STATUSES),
            )
        )
    ).all()
    durations = [
        (updated_at - started_at).total_seconds()
        for started_at, updated_at in latency_rows
        if started_at is not None and updated_at is not None
    ]
    avg_run_latency_seconds = (sum(durations) / len(durations)) if durations else None

    return {
        "run_status_counts": run_status_counts,
        "task_status_counts": task_status_counts,
        "tool_calls": {"started": started, "completed": completed, "failed": failed},
        "tool_call_success_rate": tool_call_success_rate,
        "total_cost": total_cost if have_cost else None,
        "total_tokens": total_tokens if have_tokens else None,
        "total_model_calls": total_model_calls,
        "total_tool_calls": total_tool_calls,
        "memory_writes_total": memory_writes_total or 0,
        "memory_pages_referenced_total": len(memory_pages),
        "delivery_state_counts": delivery_state_counts,
        "failure_reasons": [
            {"key": key, "count": count} for key, count in failure_reasons.most_common(10)
        ],
        "avg_run_latency_seconds": avg_run_latency_seconds,
    }


async def _status_counts(
    session: AsyncSession, column: InstrumentedAttribute[str]
) -> dict[str, int]:
    rows = await session.execute(select(column, func.count()).group_by(column))
    return {key: value for key, value in rows.all()}
