from __future__ import annotations

import asyncio

import pytest
from pydantic import JsonValue

from leo.demo import run_conversation_smoke, run_quote_smoke
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    BudgetLimits,
    CompletionProposal,
    EventDraft,
    EventType,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    OriginRef,
    Run,
    RunPhase,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolEffect,
    ToolExecutionContext,
    ToolRequest,
    ToolRequests,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.transitions import cancel_task_and_run
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import (
    AlwaysFailTool,
    FabricatingModel,
    FakeQuoteTool,
    FixedClock,
    SequentialIdGenerator,
    SlowModel,
    TwoToolBatchModel,
)


class _CountingCompletionModel:
    def __init__(self, *, cost: float | None) -> None:
        self.calls = 0
        self._cost = cost

    async def decide(self, request: object) -> ModelTurnResult:
        del request
        self.calls += 1
        return ModelTurnResult(
            decision=CompletionProposal(answer="Bounded direct answer.", claims=()),
            provider="fixture",
            model="bounded-completion-v1",
            finish_reason="stop",
            usage=ModelUsage(cost=self._cost),
        )


class _BlockingPlanModel:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        self.started.set()
        await self.release.wait()
        return ModelTurnResult(
            decision=ToolRequests(
                calls=(
                    ToolRequest(
                        id="stale-plan-call",
                        name="agent.execute_research_plan",
                        arguments={"goal": "Must never start after cancellation."},
                    ),
                )
            ),
            provider="fixture",
            model="blocking-plan-model",
            finish_reason="tool_calls",
        )


class _NeverPlanTool:
    def __init__(self, clock: FixedClock) -> None:
        self.clock = clock
        self.calls = 0
        self._spec = ToolSpec(
            name="agent.execute_research_plan",
            description="A plan fixture that must remain fenced after cancellation.",
            domain="HARNESS",
            input_schema={
                "type": "object",
                "properties": {"goal": {"type": "string"}},
                "required": ["goal"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return arguments

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del arguments, context
        self.calls += 1
        return ToolSuccess(
            data={"status": "completed"},
            source=SourceRef(provider="fixture-plan", reference="forbidden"),
            observed_at=self.clock.now(),
        )


@pytest.mark.asyncio
async def test_durable_cancellation_while_model_is_inflight_fences_stale_plan_launch() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    scope = ScopeKey(organization_id="cancel-race-org", strategy_id="default-domain")
    trusted_scope = TrustedScope(namespace=scope, actor_id="slack-user")
    thread = Thread(
        id="cancel-race-thread",
        scope=scope,
        origin=OriginRef(provider="slack", external_thread_id="slack:T1:C1:cancel-race"),
    )
    task = Task(
        id="cancel-race-task",
        thread_id=thread.id,
        scope=scope,
        objective="Run a delegated research plan.",
    )
    run = Run(id="cancel-race-run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, ids)
    await store.seed(thread, task, run)
    model = _BlockingPlanModel()
    plan = _NeverPlanTool(clock)
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=ToolRegistry((plan,)),
        context=DefaultContextAssembler(),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    running = asyncio.create_task(
        coordinator.run(
            task_id=task.id,
            run_id=run.id,
            trusted_scope=trusted_scope,
        )
    )
    await asyncio.wait_for(model.started.wait(), timeout=1)
    reserved = await store.load(task.id, run.id, scope)
    cancelled_task, cancelled_run = cancel_task_and_run(
        reserved.task,
        reserved.run,
        "slack_user_cancelled",
        usage=reserved.run.usage,
    )
    await store.commit(
        expected_task_version=reserved.task.version,
        expected_run_version=reserved.run.version,
        task=cancelled_task,
        run=cancelled_run,
        events=(
            EventDraft(
                type=EventType.RUN_CANCELLED,
                iteration=cancelled_run.iteration,
                payload={"reason": "slack_user_cancelled"},
            ),
        ),
    )
    model.release.set()

    result = await asyncio.wait_for(running, timeout=1)

    assert result.run.status is RunStatus.CANCELLED
    assert result.run.terminal_reason == "slack_user_cancelled"
    assert plan.calls == 0
    assert EventType.TOOL_STARTED not in {event.type for event in result.events}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("limits", "reason"),
    [
        (
            BudgetLimits(
                max_iterations=1,
                max_model_calls=4,
                max_tool_calls=1,
            ),
            "iteration_budget_exhausted",
        ),
        (
            BudgetLimits(
                max_iterations=4,
                max_model_calls=1,
                max_tool_calls=1,
            ),
            "model_call_budget_exhausted",
        ),
    ],
)
async def test_iteration_and_model_caps_stop_without_an_n_plus_one_call(
    limits: BudgetLimits,
    reason: str,
) -> None:
    result = await run_quote_smoke(model=FabricatingModel(), limits=limits)

    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.terminal_reason == reason
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0


@pytest.mark.asyncio
async def test_tool_batch_larger_than_remaining_budget_starts_no_tool() -> None:
    clock = FixedClock()
    result = await run_quote_smoke(
        model=TwoToolBatchModel(),
        tool_registry=ToolRegistry((FakeQuoteTool(clock), AlwaysFailTool())),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=1),
    )

    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.terminal_reason == "tool_call_budget_exhausted"
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
    assert result.observations == ()


@pytest.mark.asyncio
async def test_exact_estimated_cost_boundary_allows_one_verified_completion() -> None:
    model = _CountingCompletionModel(cost=0.0)
    result = await run_conversation_smoke(
        model=model,
        objective="Give a bounded direct answer.",
        limits=BudgetLimits(
            max_iterations=1,
            max_model_calls=1,
            max_tool_calls=0,
            estimated_model_cost=0.1,
            max_cost=0.1,
        ),
    )

    assert model.calls == 1
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.model_calls == 1
    assert result.run.usage.cost == 0.0
    assert result.run.usage.reservation_id is None
    assert result.run.usage.reserved_cost == 0.0


@pytest.mark.asyncio
async def test_estimated_cost_overflow_and_unknown_actual_cost_fail_closed() -> None:
    pre_call_model = _CountingCompletionModel(cost=0.0)
    pre_call = await run_conversation_smoke(
        model=pre_call_model,
        objective="Do not start an unaffordable call.",
        limits=BudgetLimits(
            max_iterations=2,
            max_model_calls=2,
            max_tool_calls=0,
            estimated_model_cost=0.2,
            max_cost=0.1,
        ),
    )
    unknown_cost_model = _CountingCompletionModel(cost=None)
    unknown_cost = await run_conversation_smoke(
        model=unknown_cost_model,
        objective="Fail closed when metering is missing.",
        limits=BudgetLimits(
            max_iterations=2,
            max_model_calls=2,
            max_tool_calls=0,
            estimated_model_cost=0.01,
            max_cost=0.1,
        ),
    )

    assert pre_call_model.calls == 0
    assert pre_call.run.terminal_reason == "estimated_model_cost_budget_exhausted"
    assert unknown_cost_model.calls == 1
    assert unknown_cost.run.terminal_reason == "model_cost_unknown"
    assert unknown_cost.run.status is RunStatus.BUDGET_EXHAUSTED


@pytest.mark.asyncio
async def test_elapsed_budget_cancels_inflight_model_once_without_retry() -> None:
    result = await run_quote_smoke(
        model=SlowModel(),
        limits=BudgetLimits(
            max_iterations=2,
            max_model_calls=2,
            max_tool_calls=1,
            max_elapsed_seconds=0.01,
        ),
    )

    assert result.run.status is RunStatus.TIMED_OUT
    assert result.run.terminal_reason == "model_call_exceeded_run_deadline"
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
