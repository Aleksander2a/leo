"""Full, normalized+raw event trace for one run."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import get_session
from leo.api.dashboard.events import build_timeline
from leo.harness.models import ScopeKey
from leo.persistence.schema import RunEventRow, RunRow

router = APIRouter()


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

    scope = ScopeKey(organization_id=run.organization_id, strategy_id=run.strategy_id)
    return build_timeline(list(rows), scope)
