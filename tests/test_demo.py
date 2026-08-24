from __future__ import annotations

import asyncio

import pytest

from leo.demo import run_quote_smoke
from leo.harness.models import (
    BudgetLimits,
    BudgetUsage,
    EventType,
    ModelUsage,
    OriginRef,
    RunStatus,
    ScopeKey,
    TaskStatus,
)
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import (
    AlwaysFailTool,
    EndlessQuoteModel,
    FabricatingModel,
    FakeQuoteTool,
    FixedClock,
    FlakyQuoteTool,
    MisstatingQuoteModel,
    ScriptedQuoteModel,
    SlowModel,
    TwoToolBatchModel,
)
from leo.replay import ReplaySourceManifest


class _MeteredQuoteModel:
    def __init__(self) -> None:
        self._delegate = ScriptedQuoteModel()

    async def decide(self, request):
        result = await self._delegate.decide(request)
        return result.model_copy(
            update={
                "usage": ModelUsage(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                    cost=0.001,
                )
            }
        )


class _CostLimitedEndlessModel:
    def __init__(self) -> None:
        self._delegate = EndlessQuoteModel()
        self.calls = 0

    async def decide(self, request):
        self.calls += 1
        result = await self._delegate.decide(request)
        return result.model_copy(update={"usage": ModelUsage(cost=0.002)})


class _ParallelProbe:
    def __init__(self) -> None:
        self.started = 0
        self.all_started = asyncio.Event()

    async def arrive(self) -> None:
        self.started += 1
        if self.started == 2:
            self.all_started.set()
        await asyncio.wait_for(self.all_started.wait(), timeout=0.2)


class _BarrierQuoteTool(FakeQuoteTool):
    def __init__(self, clock: FixedClock, probe: _ParallelProbe) -> None:
        super().__init__(clock)
        self._probe = probe

    async def execute(self, arguments, context):
        await self._probe.arrive()
        return await super().execute(arguments, context)


class _BarrierFailTool(AlwaysFailTool):
    def __init__(self, probe: _ParallelProbe) -> None:
        super().__init__()
        self._probe = probe

    async def execute(self, arguments, context):
        await self._probe.arrive()
        return await super().execute(arguments, context)


@pytest.mark.asyncio
async def test_quote_smoke_completes_only_after_tool_and_verifier() -> None:
    result = await run_quote_smoke()

    assert result.task.status is TaskStatus.COMPLETED
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 1
    assert result.run.usage.prompt_tokens is None
    assert result.run.usage.cost is None
    assert len(result.observations) == 1
    assert len(result.claims) == 1
    assert result.claims[0].observation_ids == (result.observations[0].id,)
    assert result.run.final_output == "NVDA is quoted at 181.25 USD."
    assert result.task.final_output == result.run.final_output
    assert [event.sequence for event in result.events] == list(range(1, len(result.events) + 1))
    event_types = [event.type for event in result.events]
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    source_manifest = ReplaySourceManifest.model_validate(context_event.payload["source_manifest"])
    assert source_manifest.manifest_digest != "0" * 64
    assert source_manifest.included_estimated_tokens > 0
    assert EventType.OBSERVATION_CREATED in event_types
    assert EventType.VERIFICATION_PASSED in event_types
    assert event_types[-1] is EventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_model_turn_usage_and_metadata_are_accumulated_and_recorded() -> None:
    result = await run_quote_smoke(model=_MeteredQuoteModel())

    assert result.run.usage.model_calls == 2
    assert result.run.usage.prompt_tokens == 20
    assert result.run.usage.completion_tokens == 4
    assert result.run.usage.total_tokens == 24
    assert result.run.usage.cost == 0.002
    model_events = [event for event in result.events if event.type is EventType.MODEL_CALLED]
    assert [event.payload["provider"] for event in model_events] == ["fixture", "fixture"]
    assert [event.payload["finish_reason"] for event in model_events] == ["tool_calls", "stop"]
    assert [event.payload["prompt_tokens"] for event in model_events] == [10, 10]


@pytest.mark.asyncio
async def test_cost_budget_stops_before_a_second_provider_call() -> None:
    model = _CostLimitedEndlessModel()
    result = await run_quote_smoke(
        model=model,
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=4, max_cost=0.001),
    )

    assert model.calls == 1
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.terminal_reason == "model_cost_budget_exhausted"
    assert result.run.usage.cost == 0.002


@pytest.mark.asyncio
async def test_unreconciled_model_reservation_fails_without_an_extra_provider_call() -> None:
    model = _CostLimitedEndlessModel()
    result = await run_quote_smoke(
        model=model,
        initial_usage=BudgetUsage(reservation_id="model-reservation-stale"),
    )

    assert model.calls == 0
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.terminal_reason == "model_budget_reservation_unreconciled"


@pytest.mark.asyncio
async def test_fabricated_observation_never_completes() -> None:
    result = await run_quote_smoke(
        model=FabricatingModel(),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=1),
    )

    assert result.task.status is TaskStatus.FAILED
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.observations == ()
    assert result.run.final_output is None
    assert EventType.RUN_COMPLETED not in {event.type for event in result.events}
    assert sum(event.type is EventType.VERIFICATION_FAILED for event in result.events) == 2


@pytest.mark.asyncio
async def test_tool_budget_stops_endless_model_without_extra_execution() -> None:
    result = await run_quote_smoke(
        model=EndlessQuoteModel(),
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=1),
    )

    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 1
    assert len(result.observations) == 1


@pytest.mark.asyncio
async def test_real_observation_cannot_support_a_misstated_quote() -> None:
    result = await run_quote_smoke(
        model=MisstatingQuoteModel(),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=1),
    )

    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.claims == ()
    assert EventType.RUN_COMPLETED not in {event.type for event in result.events}
    assert any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
    assert any(
        "symbol NVDA and exact current price 181.25" in item
        for item in result.task.verifier_feedback
    )
    assert any("source claim price" in item for item in result.task.verifier_feedback)
    assert any("answer price" in item for item in result.task.verifier_feedback)


@pytest.mark.asyncio
async def test_quote_value_must_match_as_a_numeric_token_not_a_substring() -> None:
    result = await run_quote_smoke(
        model=MisstatingQuoteModel("1181.25"),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=1),
    )

    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.claims == ()
    assert EventType.RUN_COMPLETED not in {event.type for event in result.events}


@pytest.mark.asyncio
async def test_smoke_carries_trusted_scope_actor_and_origin() -> None:
    scope = ScopeKey(organization_id="org-1", strategy_id="strategy-1")
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T1:C1:1.0",
        external_event_id="Ev1",
        external_channel_id="C1",
    )
    result = await run_quote_smoke(scope=scope, actor_id="U1", origin=origin)

    assert result.thread.scope == scope
    assert result.thread.origin == origin
    assert result.task.scope == scope


@pytest.mark.asyncio
async def test_run_deadline_cancels_slow_model_call() -> None:
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
    assert result.run.usage.model_calls == 1
    assert result.events[-1].type is EventType.RUN_TIMED_OUT


@pytest.mark.asyncio
async def test_retryable_read_tool_failure_gets_bounded_corrective_turn() -> None:
    result = await run_quote_smoke(
        model=ScriptedQuoteModel(),
        tool_registry=ToolRegistry((FlakyQuoteTool(FixedClock()),)),
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=2),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.tool_calls == 2
    assert EventType.TOOL_FAILED in {event.type for event in result.events}


@pytest.mark.asyncio
async def test_success_before_later_batch_failure_remains_persisted() -> None:
    result = await run_quote_smoke(
        model=TwoToolBatchModel(),
        tool_registry=ToolRegistry((FakeQuoteTool(FixedClock()), AlwaysFailTool())),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=2),
    )

    # The batch's failing sibling no longer ends the run on the spot: the tool is
    # withdrawn and the model gets its remaining turns to route around it. This
    # model only ever calls the same dead tool, so the run terminates on its
    # budget instead -- what matters here is that the successful observation from
    # the earlier call in the same batch survived.
    assert result.run.status is not RunStatus.COMPLETED
    assert result.run.terminal_reason == "iteration_budget_exhausted"
    assert len(result.observations) == 1
    created_ids = {
        str(event.payload["observation_id"])
        for event in result.events
        if event.type is EventType.OBSERVATION_CREATED
    }
    assert created_ids == {result.observations[0].id}


@pytest.mark.asyncio
async def test_independent_read_tool_batch_executes_concurrently() -> None:
    probe = _ParallelProbe()
    clock = FixedClock()
    result = await run_quote_smoke(
        model=TwoToolBatchModel(),
        tool_registry=ToolRegistry(
            (
                _BarrierQuoteTool(clock, probe),
                _BarrierFailTool(probe),
            )
        ),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=2),
    )

    assert probe.started == 2
    # See test_success_before_later_batch_failure_remains_persisted: a failing
    # sibling withdraws its tool rather than ending the run.
    assert result.run.status is not RunStatus.COMPLETED
    assert len(result.observations) == 1
