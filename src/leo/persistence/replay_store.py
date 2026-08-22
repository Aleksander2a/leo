"""Exact-scope durable parent/plan/child replay composition."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import RunBundle, ScopeKey
from leo.harness.plan_models import PlanSnapshot
from leo.harness.ports import Clock, IdGenerator
from leo.harness.store_errors import NotFoundError, StoreError
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import PlanRow, RunRow, TaskRow
from leo.replay import MAX_REPLAY_ENTRIES, NormalizedReplay, normalize_replay


class DurableReplayConflictError(StoreError):
    """Durable parent state cannot be projected into one unambiguous replay."""


class PostgresReplayStore:
    """Load one parent and its exact durable plan/children as a normalized replay."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._runs = PostgresRunStore(sessions, clock, ids)
        self._plans = PostgresPlanStore(sessions, clock, ids)

    async def load(
        self,
        *,
        scope: ScopeKey,
        run_id: str,
        max_entries: int = MAX_REPLAY_ENTRIES,
    ) -> NormalizedReplay:
        async with self._sessions() as session, session.begin():
            # Discover identity without granting authority, then lock in the same
            # plan -> parent -> child order used by terminal propagation. Every lane
            # below is loaded from this one read transaction.
            parent_task_id = await session.scalar(
                select(RunRow.task_id).where(
                    RunRow.id == run_id,
                    RunRow.organization_id == scope.organization_id,
                    RunRow.strategy_id == scope.strategy_id,
                )
            )
            if parent_task_id is None:
                raise NotFoundError("run not found")
            plan_rows = tuple(
                (
                    await session.scalars(
                        select(PlanRow)
                        .where(
                            PlanRow.organization_id == scope.organization_id,
                            PlanRow.parent_task_id == parent_task_id,
                            PlanRow.parent_run_id == run_id,
                        )
                        .order_by(PlanRow.created_at, PlanRow.id)
                        .with_for_update(read=True)
                    )
                ).all()
            )
            if len(plan_rows) > 1:
                raise DurableReplayConflictError("parent run has multiple durable plan aggregates")
            parent_task_row = await session.scalar(
                select(TaskRow)
                .where(
                    TaskRow.id == parent_task_id,
                    TaskRow.organization_id == scope.organization_id,
                    TaskRow.strategy_id == scope.strategy_id,
                )
                .with_for_update(read=True)
            )
            parent_run_row = await session.scalar(
                select(RunRow)
                .where(
                    RunRow.id == run_id,
                    RunRow.task_id == parent_task_id,
                    RunRow.organization_id == scope.organization_id,
                    RunRow.strategy_id == scope.strategy_id,
                )
                .with_for_update(read=True)
            )
            if parent_task_row is None or parent_run_row is None:
                raise NotFoundError("run not found")
            parent = await self._runs._load_bundle(
                session,
                parent_task_id,
                run_id,
                scope,
            )
            plan = None if not plan_rows else await self._plans._snapshot(session, plan_rows[0])
            child_run_ids = _child_run_ids(plan)
            children: list[RunBundle] = []
            for child_run_id in child_run_ids:
                child_run_row = await session.scalar(
                    select(RunRow)
                    .where(
                        RunRow.id == child_run_id,
                        RunRow.organization_id == scope.organization_id,
                        RunRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update(read=True)
                )
                if child_run_row is None:
                    raise DurableReplayConflictError(
                        "durable child run is outside the replay authority"
                    )
                child_task_row = await session.scalar(
                    select(TaskRow)
                    .where(
                        TaskRow.id == child_run_row.task_id,
                        TaskRow.organization_id == scope.organization_id,
                        TaskRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update(read=True)
                )
                if child_task_row is None:
                    raise DurableReplayConflictError("durable child task is missing")
                children.append(
                    await self._runs._load_bundle(
                        session,
                        child_task_row.id,
                        child_run_id,
                        scope,
                    )
                )
            if plan is not None:
                linked_child_task_ids = {
                    child_task_id
                    for child_task_id in (
                        *(node.child_task_id for node in plan.nodes),
                        *(delegation.child_task_id for delegation in plan.delegations),
                    )
                    if child_task_id is not None
                }
                if any(child.task.id not in linked_child_task_ids for child in children):
                    raise DurableReplayConflictError(
                        "durable child run does not match its plan task identity"
                    )
            return normalize_replay(
                parent,
                plan=plan,
                children=tuple(children),
                max_entries=max_entries,
            )


def _child_run_ids(plan: PlanSnapshot | None) -> tuple[str, ...]:
    if plan is None:
        return ()
    return tuple(
        sorted(
            {
                child_run_id
                for child_run_id in (
                    *(node.child_run_id for node in plan.nodes),
                    *(delegation.child_run_id for delegation in plan.delegations),
                )
                if child_run_id is not None
            }
        )
    )
