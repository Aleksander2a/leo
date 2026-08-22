from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import (
    CompletionProposal,
    ModelRequest,
    ModelTurnResult,
    OriginRef,
    Run,
    ScopeKey,
    Task,
    Thread,
    ToolExecutionContext,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.plan_models import PlanStatus
from leo.harness.subagents import SubagentPlanTool
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import FixedClock
from leo.integrations.system import UuidIdGenerator
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import TaskRow


class _CompletingModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        return ModelTurnResult(
            decision=CompletionProposal(answer=f"Completed: {request.objective}", claims=()),
            provider="fixture",
            model="fixture-model",
        )


@dataclass(frozen=True)
class DurableSubagentHarness:
    tool: SubagentPlanTool
    plans: PostgresPlanStore
    sessions: async_sessionmaker[AsyncSession]
    model: _CompletingModel
    scope: ScopeKey
    parent_task: Task
    parent_run: Run


@pytest_asyncio.fixture
async def durable_subagents(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[DurableSubagentHarness]:
    suffix = uuid4().hex
    sessions = preserved_postgres_sessions
    clock = FixedClock()
    ids = UuidIdGenerator()
    scope = ScopeKey(
        organization_id=f"org-subagent-{suffix}",
        strategy_id=f"strategy-{suffix}",
    )
    thread = Thread(
        id=f"thread-subagent-parent-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="test",
            external_thread_id=f"subagent-parent-{suffix}",
        ),
    )
    task = Task(
        id=f"task-subagent-parent-{suffix}",
        thread_id=thread.id,
        scope=scope,
        objective="Coordinate durable children",
    )
    run = Run(id=f"run-subagent-parent-{suffix}", task_id=task.id, scope=scope)
    run_store = PostgresRunStore(sessions, clock, ids)
    await run_store.seed(thread, task, run)
    plan_store = PostgresPlanStore(sessions, clock, ids)
    model = _CompletingModel()
    yield DurableSubagentHarness(
        tool=SubagentPlanTool(
            model=model,
            tools=ToolRegistry(()),
            context_items=(),
            clock=clock,
            ids=ids,
            run_store=run_store,
            plan_store=plan_store,
            parent_task_id=task.id,
            parent_run_id=run.id,
        ),
        plans=plan_store,
        sessions=sessions,
        model=model,
        scope=scope,
        parent_task=task,
        parent_run=run,
    )


@pytest.mark.asyncio
async def test_postgres_plan_executes_attached_children_and_replays_without_duplication(
    durable_subagents: DurableSubagentHarness,
) -> None:
    harness = durable_subagents
    arguments = harness.tool.validate(
        {
            "goal": "Research and synthesize durably.",
            "max_concurrency": 2,
            "nodes": [
                {"id": "research", "objective": "Research evidence."},
                {
                    "id": "synthesis",
                    "objective": "Synthesize evidence.",
                    "depends_on": ["research"],
                },
            ],
        }
    )
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=harness.scope, actor_id="actor"),
        run_id=harness.parent_run.id,
        tool_call_id="durable-postgres-plan",
    )

    first = await harness.tool.execute(arguments, context)
    calls_after_first = harness.model.calls
    second = await harness.tool.execute(arguments, context)

    assert isinstance(first, ToolSuccess)
    assert second == first
    assert calls_after_first == harness.model.calls == 2
    plan_id = str(first.data["plan_id"])
    snapshot = await harness.plans.replay(scope=harness.scope, plan_id=plan_id)
    assert snapshot.plan.status is PlanStatus.COMPLETED
    assert all(node.child_task_id and node.child_run_id for node in snapshot.current_nodes)
    assert all(item.child_task_id and item.child_run_id for item in snapshot.delegations)

    async with harness.sessions() as session:
        child_count = await session.scalar(
            select(func.count())
            .select_from(TaskRow)
            .where(TaskRow.parent_task_id == harness.parent_task.id)
        )
    assert child_count == 2
