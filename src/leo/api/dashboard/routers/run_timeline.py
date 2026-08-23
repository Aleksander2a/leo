"""Full, normalized+raw event trace for one run."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import get_session
from leo.api.dashboard.events import build_timeline
from leo.harness.models import ScopeKey
from leo.persistence.schema import (
    ModelCallTranscriptRow,
    RunEventRow,
    RunRow,
    TaskRow,
)

router = APIRouter()


@router.get("/runs/{run_id}/reasoning")
async def get_run_reasoning(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    """Return the run's plan/act/observe trace: what Leo tried, and why.

    The event timeline records what the harness did. This records what the model
    was *trying to do* at each step, which is the thing you actually need when a
    run goes sideways -- an operator seeing "three searches, then a weak answer"
    can now read the intent behind each one instead of inferring it.
    """

    run = await session.get(RunRow, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    task = await session.get(TaskRow, run.task_id)
    steps = list(task.scratchpad or ()) if task is not None else []
    return {
        "run_id": run_id,
        "task_id": run.task_id,
        "objective": task.objective if task is not None else None,
        "steps": steps,
        "step_count": len(steps),
    }


@router.get("/runs/{run_id}/timeline")
async def get_run_timeline(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> list[dict[str, Any]]:
    run = await session.get(RunRow, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")

    rows = (
        (
            await session.execute(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id)
                .order_by(RunEventRow.sequence)
            )
        )
        .scalars()
        .all()
    )

    transcript_rows = (
        await session.execute(
            select(
                ModelCallTranscriptRow.request_id,
                ModelCallTranscriptRow.raw_request,
                ModelCallTranscriptRow.raw_response,
            ).where(ModelCallTranscriptRow.run_id == run_id)
        )
    ).all()
    transcripts_by_request_id = {
        request_id: {"request": raw_request, "response": raw_response}
        for request_id, raw_request, raw_response in transcript_rows
    }

    scope = ScopeKey(organization_id=run.organization_id, strategy_id=run.strategy_id)
    return build_timeline(list(rows), scope, transcripts_by_request_id=transcripts_by_request_id)
