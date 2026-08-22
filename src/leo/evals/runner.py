"""Executable deterministic scenarios backed by Leo's real coordinator."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime

from leo.evals.closure_scenarios import (
    CLOSURE_VARIANTS,
    ClosureScenarioUnsupported,
    execute_closure_scenario,
)
from leo.evals.control import BaselineExecution, NoCorrectionVerifier
from leo.evals.milestone5 import (
    MILESTONE5_VARIANTS,
    Milestone5UnsupportedScenario,
    execute_milestone5_scenario,
    execute_milestone5_trace,
)
from leo.evals.models import ProviderMode, Scenario, ScenarioResult, ScenarioStatus
from leo.evals.revised_m5_scenarios import (
    REVISED_M5_VARIANTS,
    RevisedM5UnsupportedScenario,
    execute_revised_m5_scenario,
)
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    BudgetLimits,
    CoordinatorResult,
    EventType,
    ModelRequest,
    ModelTurnResult,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
    TrustedScope,
)
from leo.harness.ports import ModelGateway
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import (
    FabricatingModel,
    FakeQuoteTool,
    FixedClock,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)


class _UnsupportedScenario(RuntimeError):
    pass


ControlUnsupportedScenario = _UnsupportedScenario


class _CountingModelGateway:
    def __init__(self, delegate: ModelGateway) -> None:
        self._delegate = delegate
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        return await self._delegate.decide(request)


@dataclass(frozen=True)
class _ExecutionEvidence:
    result: CoordinatorResult
    provider_calls: int
    tool_adapter_calls: int


@dataclass(frozen=True)
class _ObservedOutcome:
    invariants: frozenset[str]
    metrics: dict[str, float | int | str]
    hard_failures: tuple[str, ...] = ()


_Executor = Callable[[Scenario], Awaitable[_ExecutionEvidence]]


def _parse_clock(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise _UnsupportedScenario("fixed_clock_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _UnsupportedScenario("fixed_clock_requires_timezone")
    return parsed


def _limits(scenario: Scenario) -> BudgetLimits:
    if scenario.budget.max_model_calls < 1:
        raise _UnsupportedScenario("coordinator_scenario_requires_model_budget")
    return BudgetLimits(
        max_iterations=scenario.budget.max_model_calls,
        max_model_calls=scenario.budget.max_model_calls,
        max_tool_calls=scenario.budget.max_tool_calls,
        max_elapsed_seconds=scenario.budget.max_elapsed_seconds,
    )


async def _run_composition(
    scenario: Scenario,
    *,
    model: ModelGateway,
    objective: str,
    correction_retries: bool = True,
) -> _ExecutionEvidence:
    clock = FixedClock(_parse_clock(scenario.fixed_clock))
    ids = SequentialIdGenerator()
    prefix = scenario.deterministic_id_prefix
    scope = ScopeKey(organization_id="eval-org", strategy_id=f"{prefix}-scope")
    thread = Thread(
        id=f"{prefix}-thread",
        scope=scope,
        origin=OriginRef(provider="fixture", external_thread_id=f"{prefix}-external-thread"),
    )
    task = Task(
        id=f"{prefix}-task",
        thread_id=thread.id,
        scope=scope,
        objective=objective,
    )
    run = Run(
        id=f"{prefix}-run",
        task_id=task.id,
        scope=scope,
        limits=_limits(scenario),
    )
    store = InMemoryRunStore(clock, ids)
    await store.seed(thread, task, run)
    quote_tool = FakeQuoteTool(clock)
    counting_model = _CountingModelGateway(model)
    deterministic_verifier = DeterministicCompletionVerifier(
        ids,
        clock,
        required_observation_kinds=frozenset({"market.get_quote"}),
    )
    verifier = (
        deterministic_verifier
        if correction_retries
        else NoCorrectionVerifier(deterministic_verifier)
    )
    coordinator = RunCoordinator(
        store=store,
        model=counting_model,
        tools=ToolRegistry((quote_tool,)),
        context=DefaultContextAssembler(),
        verifier=verifier,
        clock=clock,
        ids=ids,
    )
    result = await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=TrustedScope(
            namespace=scope,
            actor_id="eval-user",
            roles=frozenset({"researcher"}),
        ),
    )
    return _ExecutionEvidence(
        result=result,
        provider_calls=counting_model.calls,
        tool_adapter_calls=quote_tool.calls,
    )


async def _execute_quote_control(scenario: Scenario) -> _ExecutionEvidence:
    symbol = scenario.inputs.get("symbol")
    if symbol != "NVDA":
        raise _UnsupportedScenario("quote_control_only_supports_nvda")
    return await _run_composition(
        scenario,
        model=ScriptedQuoteModel(),
        objective="Report the current NVDA quote from an allowed market tool.",
    )


async def _execute_safe_failure(scenario: Scenario) -> _ExecutionEvidence:
    prompt = scenario.inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise _UnsupportedScenario("safe_failure_requires_prompt")
    return await _run_composition(
        scenario,
        model=FabricatingModel(),
        objective=prompt,
    )


_EXECUTORS: dict[str, _Executor] = {
    "quote_control": _execute_quote_control,
    "safe_failure": _execute_safe_failure,
}


async def execute_scenario_trace(scenario: Scenario) -> CoordinatorResult:
    """Execute a supported offline scenario and return its real coordinator state."""

    if scenario.provider_mode is not ProviderMode.OFFLINE:
        raise _UnsupportedScenario(f"provider_mode_not_executable:{scenario.provider_mode.value}")
    if scenario.execution_variant in CLOSURE_VARIANTS:
        raise _UnsupportedScenario(f"coordinator_trace_not_available:{scenario.execution_variant}")
    elif scenario.execution_variant in REVISED_M5_VARIANTS:
        raise _UnsupportedScenario(f"coordinator_trace_not_available:{scenario.execution_variant}")
    elif scenario.execution_variant in MILESTONE5_VARIANTS:
        try:
            return await execute_milestone5_trace(scenario)
        except Milestone5UnsupportedScenario as exc:
            raise _UnsupportedScenario(str(exc)) from exc
    executor = _EXECUTORS.get(scenario.execution_variant)
    if executor is None:
        raise _UnsupportedScenario(f"execution_variant_not_supported:{scenario.execution_variant}")
    return (await executor(scenario)).result


def _common_observations(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> tuple[set[str], dict[str, float | int | str], list[str]]:
    result = evidence.result
    event_types = [event.type for event in result.events]
    model_events = event_types.count(EventType.MODEL_CALLED)
    tool_events = event_types.count(EventType.TOOL_COMPLETED)
    invariants: set[str] = set()
    hard_failures: list[str] = []

    scoped_objects = (
        result.thread.scope,
        result.task.scope,
        result.run.scope,
        *(item.scope for item in result.observations),
        *(item.scope for item in result.claims),
    )
    if scoped_objects and all(scope == result.thread.scope for scope in scoped_objects):
        invariants.add("scope_is_pinned")
    if (
        evidence.provider_calls == result.run.usage.model_calls == model_events
        and evidence.tool_adapter_calls == result.run.usage.tool_calls == tool_events
    ):
        invariants.add("provider_calls_accounted")
    if (
        result.run.usage.model_calls <= scenario.budget.max_model_calls
        and result.run.usage.tool_calls <= scenario.budget.max_tool_calls
    ):
        invariants.add("budget_is_respected")
    else:
        hard_failures.append("budget_counter_exceeded")

    metrics: dict[str, float | int | str] = {
        "turns": result.run.usage.model_calls,
        "model_calls": result.run.usage.model_calls,
        "provider_calls": evidence.provider_calls,
        "tool_calls": result.run.usage.tool_calls,
        "observation_count": len(result.observations),
        "event_count": len(result.events),
        "task_success_count": int(
            result.run.status is RunStatus.COMPLETED
            and result.run.terminal_reason == "verified_completion"
        ),
        "false_success_count": int(
            result.run.status is RunStatus.COMPLETED
            and result.run.terminal_reason != "verified_completion"
        ),
        "terminal_reason_count": int(result.run.terminal_reason is not None),
    }
    if result.run.usage.total_tokens is not None:
        metrics["total_tokens"] = result.run.usage.total_tokens
    if result.run.usage.cost is not None:
        metrics["provider_cost"] = result.run.usage.cost
    return invariants, metrics, hard_failures


def _observe_quote_control(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> _ObservedOutcome:
    result = evidence.result
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    observation_ids = {observation.id for observation in result.observations}
    quote_observations = [
        observation
        for observation in result.observations
        if observation.kind == "market.get_quote"
        and observation.data.get("symbol") == scenario.inputs.get("symbol")
        and observation.source.provider == "fixture"
    ]
    claims_are_linked = bool(result.claims) and all(
        claim.observation_ids and set(claim.observation_ids).issubset(observation_ids)
        for claim in result.claims
    )
    if len(quote_observations) == 1 and claims_are_linked:
        invariants.add("quote_is_grounded")
    event_types = [event.type for event in result.events]
    if (
        result.task.status is TaskStatus.COMPLETED
        and result.run.status is RunStatus.COMPLETED
        and result.task.final_output == result.run.final_output
        and result.run.terminal_reason == "verified_completion"
        and EventType.VERIFICATION_PASSED in event_types
        and event_types[-1:] == [EventType.RUN_COMPLETED]
    ):
        invariants.add("terminal_is_verified")
    return _ObservedOutcome(
        invariants=frozenset(invariants),
        metrics=metrics,
        hard_failures=tuple(hard_failures),
    )


def _observe_safe_failure(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> _ObservedOutcome:
    result = evidence.result
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    event_types = [event.type for event in result.events]
    if (
        result.task.status is TaskStatus.FAILED
        and result.run.status is not RunStatus.COMPLETED
        and result.task.final_output is None
        and result.run.final_output is None
        and not result.claims
        and EventType.RUN_COMPLETED not in event_types
    ):
        invariants.add("no_false_success")
    if result.task.verifier_feedback and EventType.VERIFICATION_FAILED in event_types:
        invariants.add("rejection_is_recorded")
    if (
        result.run.status is RunStatus.BUDGET_EXHAUSTED
        and result.run.terminal_reason == "iteration_budget_exhausted"
        and event_types[-1:] == [EventType.BUDGET_EXHAUSTED]
    ):
        invariants.add("terminal_failure_is_recorded")
    return _ObservedOutcome(
        invariants=frozenset(invariants),
        metrics=metrics,
        hard_failures=tuple(hard_failures),
    )


_OBSERVERS: dict[str, Callable[[Scenario, _ExecutionEvidence], _ObservedOutcome]] = {
    "quote_control": _observe_quote_control,
    "safe_failure": _observe_safe_failure,
}


async def execute_control_baseline_scenario(scenario: Scenario) -> BaselineExecution:
    """Execute the frozen simple baseline for quote/safe-failure control fixtures."""

    if scenario.execution_variant == "quote_control":
        symbol = scenario.inputs.get("symbol")
        if symbol != "NVDA":
            raise _UnsupportedScenario("quote_control_only_supports_nvda")
        model: ModelGateway = ScriptedQuoteModel()
        objective = "Report the current NVDA quote from an allowed market tool."
        model_fixture = "ScriptedQuoteModel"
    elif scenario.execution_variant == "safe_failure":
        prompt = scenario.inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise _UnsupportedScenario("safe_failure_requires_prompt")
        model = FabricatingModel()
        objective = prompt
        model_fixture = "FabricatingModel"
    else:
        raise _UnsupportedScenario(f"execution_variant_not_supported:{scenario.execution_variant}")

    evidence = await _run_composition(
        scenario,
        model=model,
        objective=objective,
        correction_retries=False,
    )
    result = evidence.result
    event_types = tuple(event.type for event in result.events)
    scopes = (
        result.thread.scope,
        result.task.scope,
        result.run.scope,
        *(item.scope for item in result.observations),
        *(item.scope for item in result.claims),
    )
    invariants: set[str] = set()
    hard_failures: list[str] = []
    if all(scope == result.thread.scope for scope in scopes):
        invariants.add("scope_is_pinned")
    if evidence.provider_calls == result.run.usage.model_calls == event_types.count(
        EventType.MODEL_CALLED
    ) and result.run.usage.tool_calls == event_types.count(EventType.TOOL_STARTED):
        invariants.add("provider_calls_accounted")
    if (
        result.run.usage.model_calls <= scenario.budget.max_model_calls
        and result.run.usage.tool_calls <= scenario.budget.max_tool_calls
    ):
        invariants.add("budget_is_respected")
    else:
        hard_failures.append("baseline_budget_counter_exceeded")
    completed = result.run.status is RunStatus.COMPLETED
    verified = (
        completed
        and result.run.terminal_reason == "verified_completion"
        and EventType.VERIFICATION_PASSED in event_types
        and event_types[-1:] == (EventType.RUN_COMPLETED,)
    )
    false_success = int(
        (completed and not verified)
        or (
            not completed
            and (
                result.run.final_output is not None
                or result.task.final_output is not None
                or EventType.RUN_COMPLETED in event_types
            )
        )
    )
    if false_success == 0:
        invariants.add("no_false_success")
    else:
        hard_failures.append("baseline_false_success")
    if {
        "scope_is_pinned",
        "provider_calls_accounted",
        "budget_is_respected",
        "no_false_success",
    } <= invariants:
        invariants.add("baseline_hard_safety_preserved")

    verification_failures = tuple(
        event for event in result.events if event.type is EventType.VERIFICATION_FAILED
    )
    correction_calls = sum(
        event.type is EventType.MODEL_CALLED
        and any(event.sequence > failure.sequence for failure in verification_failures)
        for event in result.events
    )
    return BaselineExecution(
        invariants=frozenset(invariants),
        metrics={
            "task_success_count": int(verified),
            "false_success_count": false_success,
            "model_calls": result.run.usage.model_calls,
            "provider_calls": evidence.provider_calls,
            "tool_calls": result.run.usage.tool_calls,
            "tool_adapter_calls": evidence.tool_adapter_calls,
            "observation_count": len(result.observations),
            "event_count": len(result.events),
            "correction_retry_count": correction_calls,
            "context_items_seen": 0,
            "plan_nodes_completed": 0,
        },
        hard_failures=tuple(hard_failures),
        eligible_schema_count=1,
        admitted_destination=f"{scenario.deterministic_id_prefix}-external-thread",
        model_fixture=model_fixture,
        matched_tool_catalog=("market.get_quote",),
        exposed_tool_catalog=("market.get_quote",),
    )


def _result(
    scenario: Scenario,
    *,
    status: ScenarioStatus,
    reason: str,
    invariant_failures: tuple[str, ...] = (),
    metrics: dict[str, float | int | str] | None = None,
    raw_counts: dict[str, float | int] | None = None,
) -> ScenarioResult:
    return ScenarioResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        status=status,
        provider_mode=scenario.provider_mode,
        fixture_digest=scenario.fixture_digest,
        invariant_failures=invariant_failures,
        metrics=metrics or {},
        raw_counts=raw_counts or {},
        replay_pointer=(
            f"scenario:{scenario.id}:{scenario.version}:{scenario.deterministic_id_prefix}"
        ),
        reason=reason,
    )


async def run_scenario_async(scenario: Scenario) -> ScenarioResult:
    if scenario.provider_mode is not ProviderMode.OFFLINE:
        return _result(
            scenario,
            status=ScenarioStatus.UNSUPPORTED,
            reason=f"provider_mode_not_executable:{scenario.provider_mode.value}",
        )
    if scenario.execution_variant in CLOSURE_VARIANTS:
        try:
            closure_observed = await execute_closure_scenario(scenario)
        except ClosureScenarioUnsupported as exc:
            return _result(
                scenario,
                status=ScenarioStatus.UNSUPPORTED,
                reason=str(exc),
            )
        observed = _ObservedOutcome(
            invariants=closure_observed.invariants,
            metrics=closure_observed.metrics,
            hard_failures=closure_observed.hard_failures,
        )
    elif scenario.execution_variant in MILESTONE5_VARIANTS:
        try:
            milestone5_observed = await execute_milestone5_scenario(scenario)
        except Milestone5UnsupportedScenario as exc:
            return _result(
                scenario,
                status=ScenarioStatus.UNSUPPORTED,
                reason=str(exc),
            )
        observed = _ObservedOutcome(
            invariants=milestone5_observed.invariants,
            metrics=milestone5_observed.metrics,
            hard_failures=milestone5_observed.hard_failures,
        )
    elif scenario.execution_variant in REVISED_M5_VARIANTS:
        try:
            revised_observed = await execute_revised_m5_scenario(scenario)
        except RevisedM5UnsupportedScenario as exc:
            return _result(
                scenario,
                status=ScenarioStatus.UNSUPPORTED,
                reason=str(exc),
            )
        observed = _ObservedOutcome(
            invariants=revised_observed.invariants,
            metrics=revised_observed.metrics,
            hard_failures=revised_observed.hard_failures,
        )
    else:
        executor = _EXECUTORS.get(scenario.execution_variant)
        observer = _OBSERVERS.get(scenario.execution_variant)
        if executor is None or observer is None:
            return _result(
                scenario,
                status=ScenarioStatus.UNSUPPORTED,
                reason=f"execution_variant_not_supported:{scenario.execution_variant}",
            )
        try:
            evidence = await executor(scenario)
        except _UnsupportedScenario as exc:
            return _result(
                scenario,
                status=ScenarioStatus.UNSUPPORTED,
                reason=str(exc),
            )
        observed = observer(scenario, evidence)
    missing = scenario.expected_hard_invariants - observed.invariants
    failures = tuple(sorted(set(observed.hard_failures) | set(missing)))
    selected_metrics = {
        name: observed.metrics[name]
        for name in scenario.expected_quality_metrics
        if name in observed.metrics
    }
    raw_counts = {
        name: value
        for name, value in observed.metrics.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }
    if failures:
        return _result(
            scenario,
            status=ScenarioStatus.FAILED,
            reason="hard_invariant_failure",
            invariant_failures=failures,
            metrics=selected_metrics,
            raw_counts=raw_counts,
        )
    return _result(
        scenario,
        status=ScenarioStatus.PASSED,
        reason="observed_invariants_passed",
        metrics=selected_metrics,
        raw_counts=raw_counts,
    )


async def run_scenarios_async(scenarios: Iterable[Scenario]) -> tuple[ScenarioResult, ...]:
    return tuple([await run_scenario_async(scenario) for scenario in scenarios])


def run_scenario(scenario: Scenario) -> ScenarioResult:
    """Synchronous compatibility boundary used by the existing CLI."""

    return asyncio.run(run_scenario_async(scenario))


def run_scenarios(scenarios: Iterable[Scenario]) -> tuple[ScenarioResult, ...]:
    """Run executable scenarios without changing the ``leo eval`` call contract."""

    return asyncio.run(run_scenarios_async(scenarios))
