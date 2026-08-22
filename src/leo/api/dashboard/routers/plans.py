"""Recursive plan / delegation tree for a run's child research work."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import get_session
from leo.persistence.schema import DelegationRow, PlanNodeRow, PlanRevisionRow, PlanRow, RunRow

router = APIRouter()

_MAX_DEPTH = 10


@router.get("/runs/{run_id}/plan-tree")
async def get_plan_tree(
    run_id: str, session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    run = await session.get(RunRow, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    plans = await _plans_for_run(session, run_id, depth=0)
    return {"run_id": run_id, "plans": plans}


async def _plans_for_run(session: AsyncSession, run_id: str, depth: int) -> list[dict[str, Any]]:
    if depth >= _MAX_DEPTH:
        return []
    plan_rows = (
        (await session.execute(select(PlanRow).where(PlanRow.parent_run_id == run_id)))
        .scalars()
        .all()
    )
    return [await _plan_node(session, plan, depth) for plan in plan_rows]


async def _plan_node(session: AsyncSession, plan: PlanRow, depth: int) -> dict[str, Any]:
    revisions = (
        (
            await session.execute(
                select(PlanRevisionRow)
                .where(PlanRevisionRow.plan_id == plan.id)
                .order_by(PlanRevisionRow.number)
            )
        )
        .scalars()
        .all()
    )
    current_revision = next(
        (revision for revision in revisions if revision.number == plan.current_revision),
        revisions[-1] if revisions else None,
    )

    nodes: list[dict[str, Any]] = []
    if current_revision is not None:
        node_rows = (
            (
                await session.execute(
                    select(PlanNodeRow)
                    .where(PlanNodeRow.revision_id == current_revision.id)
                    .order_by(PlanNodeRow.node_key)
                )
            )
            .scalars()
            .all()
        )
        for node in node_rows:
            nodes.append(await _node_with_delegations(session, node, depth))

    return {
        "id": plan.id,
        "status": plan.status,
        "current_revision": plan.current_revision,
        "max_revisions": plan.max_revisions,
        "output": plan.output,
        "error": plan.error,
        "revisions": [
            {
                "id": revision.id,
                "number": revision.number,
                "goal": revision.goal,
                "reason": revision.reason,
                "digest": revision.digest,
                "created_at": revision.created_at,
            }
            for revision in revisions
        ],
        "nodes": nodes,
        "created_at": plan.created_at,
        "updated_at": plan.updated_at,
    }


async def _node_with_delegations(
    session: AsyncSession, node: PlanNodeRow, depth: int
) -> dict[str, Any]:
    delegation_rows = (
        (
            await session.execute(
                select(DelegationRow)
                .where(DelegationRow.node_id == node.id)
                .order_by(DelegationRow.attempt)
            )
        )
        .scalars()
        .all()
    )

    delegations: list[dict[str, Any]] = []
    for delegation in delegation_rows:
        child_run: dict[str, Any] | None = None
        child_plans: list[dict[str, Any]] = []
        if delegation.child_run_id is not None:
            child_run_row = await session.get(RunRow, delegation.child_run_id)
            if child_run_row is not None:
                child_run = {
                    "id": child_run_row.id,
                    "status": child_run_row.status,
                    "phase": child_run_row.phase,
                    "terminal_reason": child_run_row.terminal_reason,
                }
                child_plans = await _plans_for_run(session, delegation.child_run_id, depth + 1)
        delegations.append(
            {
                "id": delegation.id,
                "attempt": delegation.attempt,
                "status": delegation.status,
                "output": delegation.output,
                "error": delegation.error,
                "child_task_id": delegation.child_task_id,
                "child_run": child_run,
                "child_plans": child_plans,
                "created_at": delegation.created_at,
                "finished_at": delegation.finished_at,
            }
        )

    return {
        "id": node.id,
        "node_key": node.node_key,
        "objective": node.objective,
        "depends_on": node.depends_on,
        "status": node.status,
        "attempt": node.attempt,
        "max_attempts": node.max_attempts,
        "output": node.output,
        "error": node.error,
        "delegations": delegations,
    }
