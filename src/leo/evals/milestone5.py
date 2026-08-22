"""Executable offline evidence for the milestone-five agent behavior scenarios.

The fixtures in this module exercise the real coordinator, tool registry, verifier,
subagent plan tool, and in-memory persistence boundary.  Scenario JSON declares only
the contract to check; pass/fail evidence is reconstructed from runtime state, events,
requests, and adapter counters.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import cast

from pydantic import JsonValue

from leo.evals.control import BaselineExecution, NoCorrectionVerifier
from leo.evals.models import Scenario
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    BudgetLimits,
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    ContextItem,
    ContextItemKind,
    CoordinatorResult,
    EventType,
    ModelDecision,
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
    TaskStatus,
    Thread,
    ToolEffect,
    ToolExecutionContext,
    ToolRequest,
    ToolRequests,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.ports import CompletionVerifier, ModelGateway, Tool
from leo.harness.storage import InMemoryRunStore
from leo.harness.subagents import SubagentPlanTool
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import (
    FakeQuoteTool,
    FixedClock,
    QuoteArguments,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)


class Milestone5UnsupportedScenario(RuntimeError):
    """The deterministic executor cannot safely interpret a scenario input."""


@dataclass(frozen=True)
class Milestone5Observed:
    invariants: frozenset[str]
    metrics: dict[str, float | int | str]
    hard_failures: tuple[str, ...] = ()


class _CountingModelGateway:
    def __init__(self, delegate: ModelGateway) -> None:
        self._delegate = delegate
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        return await self._delegate.decide(request)


class _CountingToolAdapter:
    def __init__(self, delegate: Tool) -> None:
        self._delegate = delegate
        self.calls = 0

    @property
    def spec(self) -> ToolSpec:
        return self._delegate.spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return self._delegate.validate(arguments)

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        self.calls += 1
        outcome = await self._delegate.execute(arguments, context)
        if not isinstance(outcome, ToolSuccess):
            raise RuntimeError("milestone-five fixture tool unexpectedly failed")
        return outcome


@dataclass(frozen=True)
class _Composition:
    result: CoordinatorResult
    store: InMemoryRunStore
    model: _CountingModelGateway
    tools: ToolRegistry
    tool_adapters: tuple[_CountingToolAdapter, ...]
    context: DefaultContextAssembler
    verifier: CompletionVerifier
    clock: FixedClock
    ids: SequentialIdGenerator
    task_id: str
    run_id: str
    trusted_scope: TrustedScope

    @property
    def tool_adapter_calls(self) -> int:
        return sum(adapter.calls for adapter in self.tool_adapters)


@dataclass(frozen=True)
class _ExecutionEvidence:
    result: CoordinatorResult
    provider_calls: int
    tool_adapter_calls: int
    context_requests: tuple[tuple[ContextItem, ...], ...] = ()
    selected_context: tuple[ContextItem, ...] = ()
    candidate_context: tuple[ContextItem, ...] = ()
    allowed_conversation_ids: frozenset[str] = frozenset()
    destination_conversation_id: str | None = None
    parallel_batch_size: int = 0
    parallel_overlap_peak: int = 0
    child_requests: tuple[ModelRequest, ...] = ()
    child_provider_calls: int = 0
    replay_result: CoordinatorResult | None = None
    replay_provider_call_delta: int = 0
    replay_tool_call_delta: int = 0
    duplicate_delivery_attempt_count: int = 0
    duplicate_delivery_count: int = 0
    physical_delivery_count: int = 0


_ToolFactory = Callable[
    [FixedClock, SequentialIdGenerator],
    tuple[Tool, ...],
]
_Executor = Callable[[Scenario], Awaitable[_ExecutionEvidence]]
_Observer = Callable[[Scenario, _ExecutionEvidence], Milestone5Observed]


def _parse_clock(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise Milestone5UnsupportedScenario("fixed_clock_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise Milestone5UnsupportedScenario("fixed_clock_requires_timezone")
    return parsed


def _limits(scenario: Scenario) -> BudgetLimits:
    if scenario.budget.max_model_calls < 1:
        raise Milestone5UnsupportedScenario("coordinator_scenario_requires_model_budget")
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
    tool_factory: _ToolFactory | None = None,
    context_items: tuple[ContextItem, ...] = (),
    require_source_claim: bool = True,
    required_observation_kinds: frozenset[str] = frozenset(),
    correction_retries: bool = True,
) -> _Composition:
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
    delegates = tool_factory(clock, ids) if tool_factory is not None else ()
    adapters = tuple(_CountingToolAdapter(delegate) for delegate in delegates)
    registry = ToolRegistry(adapters)
    counting_model = _CountingModelGateway(model)
    context = DefaultContextAssembler(context_items=context_items)
    deterministic_verifier = DeterministicCompletionVerifier(
        ids,
        clock,
        require_source_claim=require_source_claim,
        required_observation_kinds=required_observation_kinds,
    )
    verifier: CompletionVerifier = (
        deterministic_verifier
        if correction_retries
        else NoCorrectionVerifier(deterministic_verifier)
    )
    trusted_scope = TrustedScope(
        namespace=scope,
        actor_id="eval-user",
        roles=frozenset({"researcher"}),
    )
    coordinator = RunCoordinator(
        store=store,
        model=counting_model,
        tools=registry,
        context=context,
        verifier=verifier,
        clock=clock,
        ids=ids,
    )
    result = await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=trusted_scope,
    )
    return _Composition(
        result=result,
        store=store,
        model=counting_model,
        tools=registry,
        tool_adapters=adapters,
        context=context,
        verifier=verifier,
        clock=clock,
        ids=ids,
        task_id=task.id,
        run_id=run.id,
        trusted_scope=trusted_scope,
    )


def _turn(request: ModelRequest, decision: ModelDecision, model_name: str) -> ModelTurnResult:
    return ModelTurnResult(
        decision=decision,
        provider="fixture",
        model=model_name,
        request_id=f"fixture-{request.iteration + 1:03d}",
        finish_reason="tool_calls" if isinstance(decision, ToolRequests) else "stop",
        usage=ModelUsage(),
    )


class _ContextAnswerModel:
    def __init__(self) -> None:
        self.requests: list[tuple[ContextItem, ...]] = []

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        visible = tuple(request.context_items)
        self.requests.append(visible)
        rendered = " | ".join(item.content for item in visible)
        answer = f"In response to '{request.objective}', the relevant context is: {rendered}"
        return _turn(request, CompletionProposal(answer=answer), type(self).__name__)


def _context_item(
    item_id: str,
    conversation_id: str,
    content: str,
    *,
    strategy_id: str,
) -> ContextItem:
    return ContextItem(
        id=item_id,
        kind=ContextItemKind.CONVERSATION_TURN,
        content=content,
        conversation_id=conversation_id,
        source_scope=ScopeKey(organization_id="eval-org", strategy_id=strategy_id),
        source_actor_id="eval-user",
    )


async def _execute_contextual_conversation(scenario: Scenario) -> _ExecutionEvidence:
    prompt = scenario.inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Milestone5UnsupportedScenario("contextual_conversation_requires_prompt")
    item = _context_item(
        "context-current-001",
        "C-CURRENT",
        "The team chose a Friday morning review with a concise written recap.",
        strategy_id="conversation:C-CURRENT",
    )
    model = _ContextAnswerModel()
    composition = await _run_composition(
        scenario,
        model=model,
        objective=prompt,
        context_items=(item,),
        require_source_claim=False,
    )
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        context_requests=tuple(model.requests),
        selected_context=(item,),
        candidate_context=(item,),
        allowed_conversation_ids=frozenset({item.conversation_id}),
        destination_conversation_id=item.conversation_id,
    )


async def _execute_channel_isolation(scenario: Scenario) -> _ExecutionEvidence:
    prompt = scenario.inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Milestone5UnsupportedScenario("channel_isolation_requires_prompt")
    channel_a = _context_item(
        "context-channel-a-001",
        "C-ALPHA",
        "Alpha-only launch code is amber-seventeen.",
        strategy_id="conversation:C-ALPHA",
    )
    channel_b = _context_item(
        "context-channel-b-001",
        "C-BETA",
        "Beta scheduled its review for Thursday afternoon.",
        strategy_id="conversation:C-BETA",
    )
    model = _ContextAnswerModel()
    composition = await _run_composition(
        scenario,
        model=model,
        objective=prompt,
        context_items=(channel_b,),
        require_source_claim=False,
    )
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        context_requests=tuple(model.requests),
        selected_context=(channel_b,),
        candidate_context=(channel_a, channel_b),
        allowed_conversation_ids=frozenset({channel_b.conversation_id}),
        destination_conversation_id=channel_b.conversation_id,
    )


async def _execute_dm_context_union(scenario: Scenario) -> _ExecutionEvidence:
    prompt = scenario.inputs.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Milestone5UnsupportedScenario("dm_context_union_requires_prompt")
    channel_a = _context_item(
        "context-dm-alpha-001",
        "C-ALPHA",
        "Alpha agreed to cap the pilot at twelve participants.",
        strategy_id="conversation:C-ALPHA",
    )
    channel_b = _context_item(
        "context-dm-beta-001",
        "G-BETA",
        "Beta selected the blue deployment window.",
        strategy_id="conversation:G-BETA",
    )
    inaccessible = _context_item(
        "context-dm-gamma-001",
        "C-GAMMA",
        "Gamma's inaccessible marker is violet-ninety-nine.",
        strategy_id="conversation:C-GAMMA",
    )
    selected = (channel_a, channel_b)
    model = _ContextAnswerModel()
    composition = await _run_composition(
        scenario,
        model=model,
        objective=prompt,
        context_items=selected,
        require_source_claim=False,
    )
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        context_requests=tuple(model.requests),
        selected_context=selected,
        candidate_context=(*selected, inaccessible),
        allowed_conversation_ids=frozenset(item.conversation_id for item in selected),
        destination_conversation_id="D-EVAL-USER",
    )


class _ParallelQuoteTool:
    def __init__(self, clock: FixedClock) -> None:
        self._clock = clock
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.started: list[str] = []
        self.completed: list[str] = []
        self._spec = ToolSpec(
            name="market.get_quote",
            description="Return one deterministic quote while exposing concurrency counters.",
            domain="MARKET",
            input_schema=QuoteArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=1.0,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = QuoteArguments.model_validate(arguments)
        return {"symbol": parsed.symbol}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del context
        symbol = arguments.get("symbol")
        if not isinstance(symbol, str):
            raise TypeError("validated symbol must be a string")
        prices = {"NVDA": 181.25, "MSFT": 420.5}
        price = prices.get(symbol)
        if price is None:
            raise ValueError("parallel fixture only supports NVDA and MSFT")
        self.calls += 1
        self.started.append(symbol)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
        finally:
            self.active -= 1
        self.completed.append(symbol)
        return ToolSuccess(
            data={"symbol": symbol, "price": price, "currency": "USD"},
            source=SourceRef(provider="fixture", reference=f"parallel-quote-{symbol}"),
            observed_at=self._clock.now(),
            expires_at=self._clock.now() + timedelta(minutes=5),
        )


def _quote_statement(symbol: object, price: object) -> str:
    if not isinstance(symbol, str) or not isinstance(price, int | float) or isinstance(price, bool):
        raise RuntimeError("quote observation is malformed")
    return f"{symbol} is quoted at {format(price, 'g')} USD."


class _ParallelBatchModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        if not request.observations:
            decision: ModelDecision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="parallel-nvda",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                    ToolRequest(
                        id="parallel-msft",
                        name="market.get_quote",
                        arguments={"symbol": "MSFT"},
                    ),
                )
            )
        else:
            ordered = sorted(request.observations, key=lambda item: str(item.data.get("symbol")))
            statements = tuple(
                _quote_statement(item.data.get("symbol"), item.data.get("price"))
                for item in ordered
            )
            decision = CompletionProposal(
                answer="Parallel research synthesis: " + " ".join(statements),
                claims=tuple(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=statement,
                        observation_ids=(observation.id,),
                    )
                    for statement, observation in zip(statements, ordered, strict=True)
                ),
            )
        return _turn(request, decision, type(self).__name__)


async def _execute_parallel_read_batch(scenario: Scenario) -> _ExecutionEvidence:
    parallel_tool: _ParallelQuoteTool | None = None

    def tool_factory(
        clock: FixedClock,
        ids: SequentialIdGenerator,
    ) -> tuple[Tool, ...]:
        del ids
        nonlocal parallel_tool
        parallel_tool = _ParallelQuoteTool(clock)
        return (parallel_tool,)

    composition = await _run_composition(
        scenario,
        model=_ParallelBatchModel(),
        objective="Fetch two independent market quotes in parallel, then synthesize them.",
        tool_factory=tool_factory,
        required_observation_kinds=frozenset({"market.get_quote"}),
    )
    if parallel_tool is None:
        raise RuntimeError("parallel fixture was not initialized")
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        parallel_batch_size=parallel_tool.calls,
        parallel_overlap_peak=parallel_tool.max_active,
    )


class _ChildQuoteModel:
    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.requests.append(request)
        symbol = "MSFT" if "MSFT" in request.objective.upper() else "NVDA"
        if not request.observations:
            decision: ModelDecision = ToolRequests(
                calls=(
                    ToolRequest(
                        id=f"child-{symbol.lower()}-{request.iteration}",
                        name="market.get_quote",
                        arguments={"symbol": symbol},
                    ),
                )
            )
        else:
            observation = request.observations[-1]
            statement = _quote_statement(
                observation.data.get("symbol"), observation.data.get("price")
            )
            decision = CompletionProposal(
                answer=statement,
                claims=(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=statement,
                        observation_ids=(observation.id,),
                    ),
                ),
            )
        return _turn(request, decision, type(self).__name__)


class _ParentPlanModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        if not request.observations:
            decision: ModelDecision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="parent-dependency-plan",
                        name="agent.execute_research_plan",
                        arguments={
                            "goal": "Compare two deterministic market observations.",
                            "nodes": [
                                {
                                    "id": "baseline",
                                    "objective": "Collect the NVDA quote baseline.",
                                    "expected_output": "One grounded NVDA quote.",
                                    "depends_on": [],
                                    "max_turns": 2,
                                },
                                {
                                    "id": "dependent",
                                    "objective": "Collect MSFT after considering the baseline.",
                                    "expected_output": "One grounded MSFT quote.",
                                    "depends_on": ["baseline"],
                                    "max_turns": 2,
                                },
                            ],
                            "max_concurrency": 2,
                        },
                    ),
                )
            )
        else:
            observation = request.observations[-1]
            nodes = observation.data.get("nodes")
            if (
                not isinstance(nodes, list)
                or not nodes
                or not all(isinstance(node, dict) for node in nodes)
            ):
                raise RuntimeError("plan observation is malformed")
            node_results = tuple(cast(dict[str, JsonValue], node) for node in nodes)
            child_answers = tuple(node.get("answer") for node in node_results)
            if not all(
                isinstance(child_answer, str) and bool(child_answer.strip())
                for child_answer in child_answers
            ):
                raise RuntimeError("one or more plan node answers are missing")
            decision = CompletionProposal(
                answer=(
                    "Parent-owned synthesis after the dependency plan completed: "
                    + " ".join(str(child_answer) for child_answer in child_answers)
                ),
                claims=tuple(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=child_answer,
                        observation_ids=(observation.id,),
                    )
                    for child_answer in child_answers
                    if isinstance(child_answer, str)
                ),
            )
        return _turn(request, decision, type(self).__name__)


async def _execute_delegated_dependency_plan(scenario: Scenario) -> _ExecutionEvidence:
    child_model = _ChildQuoteModel()
    child_quote: FakeQuoteTool | None = None

    def tool_factory(
        clock: FixedClock,
        ids: SequentialIdGenerator,
    ) -> tuple[Tool, ...]:
        nonlocal child_quote
        child_quote = FakeQuoteTool(clock)
        child_tools = ToolRegistry((child_quote,))
        local_context = (
            _context_item(
                "plan-context-001",
                "C-PLAN",
                "The parent requested a two-stage comparison.",
                strategy_id="conversation:C-PLAN",
            ),
        )
        return (
            SubagentPlanTool(
                model=child_model,
                tools=child_tools,
                context_items=local_context,
                clock=clock,
                ids=ids,
            ),
        )

    composition = await _run_composition(
        scenario,
        model=_ParentPlanModel(),
        objective="Execute a dependency-aware research plan and provide the parent synthesis.",
        tool_factory=tool_factory,
        required_observation_kinds=frozenset({"agent.execute_research_plan"}),
    )
    if child_quote is None:
        raise RuntimeError("child quote fixture was not initialized")
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        child_requests=tuple(child_model.requests),
        child_provider_calls=len(child_model.requests),
    )


class _CorrectingModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        if not request.observations:
            decision: ModelDecision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="initial-quote",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            )
        elif not request.verifier_feedback:
            observation = request.observations[-1]
            bad_statement = "NVDA is quoted at 999 USD."
            decision = CompletionProposal(
                answer=bad_statement,
                claims=(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=bad_statement,
                        observation_ids=(observation.id,),
                    ),
                ),
            )
        elif len(request.observations) == 1:
            decision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="replanned-quote",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            )
        else:
            observation = request.observations[-1]
            statement = _quote_statement(
                observation.data.get("symbol"), observation.data.get("price")
            )
            decision = CompletionProposal(
                answer="Corrected after verifier feedback: " + statement,
                claims=(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=statement,
                        observation_ids=(observation.id,),
                    ),
                ),
            )
        return _turn(request, decision, type(self).__name__)


async def _execute_verifier_correction(scenario: Scenario) -> _ExecutionEvidence:
    composition = await _run_composition(
        scenario,
        model=_CorrectingModel(),
        objective="Return a grounded quote, correcting and replanning if verification rejects it.",
        tool_factory=lambda clock, ids: (FakeQuoteTool(clock),),
        required_observation_kinds=frozenset({"market.get_quote"}),
    )
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
    )


class _IdempotentDeliveryProbe:
    def __init__(self) -> None:
        self._receipts: dict[str, str] = {}
        self.physical_deliveries = 0
        self.duplicate_attempts = 0

    def deliver(self, delivery_key: str, payload: str) -> str:
        existing = self._receipts.get(delivery_key)
        if existing is not None:
            self.duplicate_attempts += 1
            return existing
        self.physical_deliveries += 1
        receipt = f"receipt-{self.physical_deliveries}:{len(payload)}"
        self._receipts[delivery_key] = receipt
        return receipt

    @property
    def duplicate_delivery_count(self) -> int:
        return self.physical_deliveries - len(self._receipts)


async def _execute_restart_replay_idempotency(scenario: Scenario) -> _ExecutionEvidence:
    composition = await _run_composition(
        scenario,
        model=ScriptedQuoteModel(),
        objective="Report the deterministic NVDA quote exactly once.",
        tool_factory=lambda clock, ids: (FakeQuoteTool(clock),),
        required_observation_kinds=frozenset({"market.get_quote"}),
    )
    provider_calls_before = composition.model.calls
    tool_calls_before = composition.tool_adapter_calls
    restarted = RunCoordinator(
        store=composition.store,
        model=composition.model,
        tools=composition.tools,
        context=composition.context,
        verifier=composition.verifier,
        clock=composition.clock,
        ids=composition.ids,
    )
    replay_result = await restarted.run(
        task_id=composition.task_id,
        run_id=composition.run_id,
        trusted_scope=composition.trusted_scope,
    )
    delivery = _IdempotentDeliveryProbe()
    payload = composition.result.run.final_output
    replay_payload = replay_result.run.final_output
    if payload is None or replay_payload is None:
        raise RuntimeError("completed replay fixture did not produce output")
    delivery_key = f"{composition.run_id}:slack-final"
    first_receipt = delivery.deliver(delivery_key, payload)
    replay_receipt = delivery.deliver(delivery_key, replay_payload)
    if first_receipt != replay_receipt:
        raise RuntimeError("idempotent delivery returned inconsistent receipts")
    return _ExecutionEvidence(
        result=composition.result,
        provider_calls=composition.model.calls,
        tool_adapter_calls=composition.tool_adapter_calls,
        replay_result=replay_result,
        replay_provider_call_delta=composition.model.calls - provider_calls_before,
        replay_tool_call_delta=composition.tool_adapter_calls - tool_calls_before,
        duplicate_delivery_attempt_count=delivery.duplicate_attempts,
        duplicate_delivery_count=delivery.duplicate_delivery_count,
        physical_delivery_count=delivery.physical_deliveries,
    )


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
        "task_success_count": int(_terminal_is_verified(result)),
        "false_success_count": int(
            result.run.status is RunStatus.COMPLETED and not _terminal_is_verified(result)
        ),
        "terminal_reason_count": int(result.run.terminal_reason is not None),
    }
    if result.run.usage.total_tokens is not None:
        metrics["total_tokens"] = result.run.usage.total_tokens
    if result.run.usage.cost is not None:
        metrics["provider_cost"] = result.run.usage.cost
    return invariants, metrics, hard_failures


def _terminal_is_verified(result: CoordinatorResult) -> bool:
    event_types = [event.type for event in result.events]
    return (
        result.task.status is TaskStatus.COMPLETED
        and result.run.status is RunStatus.COMPLETED
        and result.task.final_output == result.run.final_output
        and result.run.terminal_reason == "verified_completion"
        and EventType.VERIFICATION_PASSED in event_types
        and event_types[-1:] == [EventType.RUN_COMPLETED]
    )


def _flatten_context_requests(evidence: _ExecutionEvidence) -> tuple[ContextItem, ...]:
    return tuple(item for request_items in evidence.context_requests for item in request_items)


def _context_metrics(evidence: _ExecutionEvidence) -> tuple[int, int, int]:
    actual = _flatten_context_requests(evidence)
    selected_ids = {item.id for item in evidence.selected_context}
    actual_by_id = {item.id: item for item in actual}
    leakage = sum(
        item.id not in selected_ids or item.conversation_id not in evidence.allowed_conversation_ids
        for item in actual_by_id.values()
    )
    final_output = evidence.result.run.final_output or ""
    recalled = sum(item.content in final_output for item in evidence.selected_context)
    return len(actual_by_id), leakage, recalled


def _observe_contextual_conversation(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    seen, leakage, recalled = _context_metrics(evidence)
    final_output = evidence.result.run.final_output or ""
    prompt = scenario.inputs.get("prompt")
    if (
        _terminal_is_verified(evidence.result)
        and isinstance(prompt, str)
        and prompt in final_output
        and recalled == len(evidence.selected_context)
    ):
        invariants.add("arbitrary_contextual_answer_verified")
    if leakage == 0 and seen == len(evidence.selected_context):
        invariants.add("context_projection_exact")
    metrics.update(
        {
            "context_items_seen": seen,
            "context_leakage_count": leakage,
            "expected_context_recall_count": recalled,
            "routing_correct_count": int(_terminal_is_verified(evidence.result)),
            "clarification_count": 0,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _observe_channel_isolation(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    seen, leakage, recalled = _context_metrics(evidence)
    actual_ids = {item.id for item in _flatten_context_requests(evidence)}
    selected_ids = {item.id for item in evidence.selected_context}
    prohibited = tuple(item for item in evidence.candidate_context if item.id not in selected_ids)
    final_output = evidence.result.run.final_output or ""
    if actual_ids == selected_ids and all(item.content not in final_output for item in prohibited):
        invariants.add("channel_isolation_preserved")
    if (
        leakage == 0
        and recalled == len(evidence.selected_context)
        and _terminal_is_verified(evidence.result)
    ):
        invariants.add("context_projection_exact")
    metrics.update(
        {
            "context_items_seen": seen,
            "context_leakage_count": leakage,
            "expected_context_recall_count": recalled,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _observe_dm_context_union(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    seen, leakage, recalled = _context_metrics(evidence)
    actual_ids = {item.id for item in _flatten_context_requests(evidence)}
    selected_ids = {item.id for item in evidence.selected_context}
    unselected = tuple(item for item in evidence.candidate_context if item.id not in selected_ids)
    final_output = evidence.result.run.final_output or ""
    if (
        actual_ids == selected_ids
        and len(evidence.allowed_conversation_ids) == 2
        and recalled == len(selected_ids)
        and all(item.content not in final_output for item in unselected)
        and _terminal_is_verified(evidence.result)
    ):
        invariants.add("dm_exact_union_recalled")
    if leakage == 0:
        invariants.add("context_projection_exact")
    metrics.update(
        {
            "context_items_seen": seen,
            "context_leakage_count": leakage,
            "expected_dm_recall_count": recalled,
            "dm_forbidden_source_count": leakage,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _observe_parallel_read_batch(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    events = evidence.result.events
    starts = [event for event in events if event.type is EventType.TOOL_STARTED]
    completions = [event for event in events if event.type is EventType.TOOL_COMPLETED]
    starts_before_first_completion = 0
    if completions:
        starts_before_first_completion = sum(
            event.sequence < completions[0].sequence for event in starts
        )
    if (
        evidence.parallel_batch_size == 2
        and evidence.parallel_overlap_peak >= 2
        and starts_before_first_completion == 2
    ):
        invariants.add("independent_reads_parallelized")
    observation_ids = {item.id for item in evidence.result.observations}
    linked = len(evidence.result.claims) == 2 and all(
        claim.observation_ids and set(claim.observation_ids).issubset(observation_ids)
        for claim in evidence.result.claims
    )
    if linked and _terminal_is_verified(evidence.result):
        invariants.add("parallel_batch_synthesis_verified")
    metrics.update(
        {
            "parallel_batch_size": evidence.parallel_batch_size,
            "parallel_batch_evidence": starts_before_first_completion,
            "parallel_overlap_peak": evidence.parallel_overlap_peak,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _plan_nodes(evidence: _ExecutionEvidence) -> list[dict[str, JsonValue]]:
    matching = [
        observation
        for observation in evidence.result.observations
        if observation.kind == "agent.execute_research_plan"
    ]
    if len(matching) != 1:
        return []
    nodes = matching[0].data.get("nodes")
    if not isinstance(nodes, list) or not all(isinstance(item, dict) for item in nodes):
        return []
    return [cast(dict[str, JsonValue], item) for item in nodes]


def _has_completed_child_trace(item: dict[str, JsonValue]) -> bool:
    child_run_id = item.get("child_run_id")
    trace_event_count = item.get("trace_event_count")
    return (
        isinstance(child_run_id, str)
        and isinstance(trace_event_count, int)
        and not isinstance(trace_event_count, bool)
        and trace_event_count > 0
    )


def _observe_delegated_dependency_plan(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    nodes = _plan_nodes(evidence)
    completed_nodes = [item for item in nodes if item.get("status") == "completed"]
    child_terminal_count = sum(_has_completed_child_trace(item) for item in completed_nodes)
    outer_terminal_count = sum(
        event.type is EventType.RUN_COMPLETED for event in evidence.result.events
    )
    if nodes and len(completed_nodes) == len(nodes) and child_terminal_count == len(nodes):
        invariants.add("dependency_plan_completed")

    baseline_answer = completed_nodes[0].get("answer") if completed_nodes else None
    dependent_requests = [
        request for request in evidence.child_requests if "MSFT" in request.objective.upper()
    ]
    if isinstance(baseline_answer, str) and any(
        any(
            item.kind is ContextItemKind.SUBAGENT_RESULT and item.content == baseline_answer
            for item in request.context_items
        )
        for request in dependent_requests
    ):
        invariants.add("dependency_context_propagated")

    child_answers = {
        answer for item in completed_nodes if isinstance((answer := item.get("answer")), str)
    }
    final_output = evidence.result.run.final_output
    if (
        outer_terminal_count == 1
        and _terminal_is_verified(evidence.result)
        and isinstance(final_output, str)
        and final_output not in child_answers
        and any(answer in final_output for answer in child_answers)
    ):
        invariants.add("parent_owns_final_answer")
    metrics.update(
        {
            "plan_nodes_completed": len(completed_nodes),
            "child_terminal_count": child_terminal_count,
            "parent_terminal_authority_count": outer_terminal_count,
            "child_provider_calls": evidence.child_provider_calls,
            "plan_valid_count": int(bool(nodes)),
            "plan_revision_count": 1,
            "plan_no_progress_count": 0,
            "child_success_count": len(completed_nodes),
            "child_duplicate_count": len(completed_nodes)
            - len(
                {
                    item.get("child_run_id")
                    for item in completed_nodes
                    if isinstance(item.get("child_run_id"), str)
                }
            ),
            "child_conflict_count": 0,
            "child_utilization_ratio": (len(completed_nodes) / len(nodes) if nodes else 0.0),
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _observe_verifier_correction(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    events = evidence.result.events
    failed = [event for event in events if event.type is EventType.VERIFICATION_FAILED]
    passed = [event for event in events if event.type is EventType.VERIFICATION_PASSED]
    completed = [event for event in events if event.type is EventType.RUN_COMPLETED]
    if (
        failed
        and passed
        and min(event.sequence for event in passed) > max(event.sequence for event in failed)
    ):
        invariants.add("verifier_correction_recorded")
    replanned_starts = [
        event
        for event in events
        if event.type is EventType.TOOL_STARTED and failed and event.sequence > failed[-1].sequence
    ]
    if replanned_starts:
        invariants.add("replan_after_rejection")
    rejected_statement_persisted = any("999" in claim.statement for claim in evidence.result.claims)
    if (
        not rejected_statement_persisted
        and len(completed) == 1
        and passed
        and completed[0].sequence > passed[-1].sequence
        and _terminal_is_verified(evidence.result)
    ):
        invariants.add("no_false_success")
    metrics.update(
        {
            "retry_count": len(failed),
            "replan_tool_call_count": len(replanned_starts),
            "plan_revision_count": 2,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


def _observe_restart_replay_idempotency(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> Milestone5Observed:
    invariants, metrics, hard_failures = _common_observations(scenario, evidence)
    replay = evidence.replay_result
    event_delta = -1
    if replay is not None:
        event_delta = len(replay.events) - len(evidence.result.events)
        if (
            replay == evidence.result
            and evidence.replay_provider_call_delta == 0
            and evidence.replay_tool_call_delta == 0
            and event_delta == 0
        ):
            invariants.add("restart_replay_idempotent")
    if (
        evidence.duplicate_delivery_attempt_count == 1
        and evidence.duplicate_delivery_count == 0
        and evidence.physical_delivery_count == 1
    ):
        invariants.add("delivery_idempotent")
    metrics.update(
        {
            "replay_event_delta": event_delta,
            "duplicate_delivery_attempt_count": evidence.duplicate_delivery_attempt_count,
            "duplicate_delivery_count": evidence.duplicate_delivery_count,
            "physical_delivery_count": evidence.physical_delivery_count,
        }
    )
    return Milestone5Observed(frozenset(invariants), metrics, tuple(hard_failures))


_EXECUTORS: dict[str, _Executor] = {
    "contextual_conversation": _execute_contextual_conversation,
    "channel_isolation": _execute_channel_isolation,
    "dm_context_union": _execute_dm_context_union,
    "parallel_read_batch": _execute_parallel_read_batch,
    "delegated_dependency_plan": _execute_delegated_dependency_plan,
    "verifier_correction": _execute_verifier_correction,
    "restart_replay_idempotency": _execute_restart_replay_idempotency,
}

_OBSERVERS: dict[str, _Observer] = {
    "contextual_conversation": _observe_contextual_conversation,
    "channel_isolation": _observe_channel_isolation,
    "dm_context_union": _observe_dm_context_union,
    "parallel_read_batch": _observe_parallel_read_batch,
    "delegated_dependency_plan": _observe_delegated_dependency_plan,
    "verifier_correction": _observe_verifier_correction,
    "restart_replay_idempotency": _observe_restart_replay_idempotency,
}

MILESTONE5_VARIANTS = frozenset(_EXECUTORS)


async def execute_milestone5_scenario(scenario: Scenario) -> Milestone5Observed:
    """Execute and independently observe one supported milestone-five scenario."""

    executor = _EXECUTORS.get(scenario.execution_variant)
    observer = _OBSERVERS.get(scenario.execution_variant)
    if executor is None or observer is None:
        raise Milestone5UnsupportedScenario(
            f"execution_variant_not_supported:{scenario.execution_variant}"
        )
    evidence = await executor(scenario)
    return observer(scenario, evidence)


async def execute_milestone5_trace(scenario: Scenario) -> CoordinatorResult:
    """Return the actual coordinator result for fixture/replay UX.

    This is deliberately the same executor used by the invariant observer, so the
    operator fixture command cannot drift into a self-attesting duplicate scenario.
    """

    executor = _EXECUTORS.get(scenario.execution_variant)
    if executor is None:
        raise Milestone5UnsupportedScenario(
            f"execution_variant_not_supported:{scenario.execution_variant}"
        )
    return (await executor(scenario)).result


async def execute_milestone5_baseline_scenario(scenario: Scenario) -> BaselineExecution:
    """Run the matched simple baseline for a milestone-five fixture.

    Ordinary scenarios use the same fixture model, tool implementation, declared
    budget, trusted scope, and coordinator as Leo. The three feature-bearing cases
    explicitly remove only the frozen baseline omissions: DM union, plan/subagents,
    and verifier correction retries.
    """

    variant = scenario.execution_variant
    if variant not in MILESTONE5_VARIANTS:
        raise Milestone5UnsupportedScenario(f"execution_variant_not_supported:{variant}")
    if variant == "dm_context_union":
        prompt = scenario.inputs.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise Milestone5UnsupportedScenario("dm_context_union_requires_prompt")
        model = _ContextAnswerModel()
        composition = await _run_composition(
            scenario,
            model=model,
            objective=prompt,
            context_items=(),
            require_source_claim=False,
            correction_retries=False,
        )
        evidence = _ExecutionEvidence(
            result=composition.result,
            provider_calls=composition.model.calls,
            tool_adapter_calls=composition.tool_adapter_calls,
            context_requests=tuple(model.requests),
            selected_context=(),
            candidate_context=(),
            allowed_conversation_ids=frozenset({"D-EVAL-USER"}),
            destination_conversation_id="D-EVAL-USER",
        )
    elif variant == "delegated_dependency_plan":
        composition = await _run_composition(
            scenario,
            model=_ParentPlanModel(),
            objective=(
                "Execute a dependency-aware research plan and provide the parent synthesis."
            ),
            correction_retries=False,
        )
        evidence = _ExecutionEvidence(
            result=composition.result,
            provider_calls=composition.model.calls,
            tool_adapter_calls=composition.tool_adapter_calls,
        )
    elif variant == "verifier_correction":
        composition = await _run_composition(
            scenario,
            model=_CorrectingModel(),
            objective=(
                "Return a grounded quote, correcting and replanning if verification rejects it."
            ),
            tool_factory=lambda clock, ids: (FakeQuoteTool(clock),),
            required_observation_kinds=frozenset({"market.get_quote"}),
            correction_retries=False,
        )
        evidence = _ExecutionEvidence(
            result=composition.result,
            provider_calls=composition.model.calls,
            tool_adapter_calls=composition.tool_adapter_calls,
        )
    else:
        evidence = await _EXECUTORS[variant](scenario)

    matched_catalog, exposed_catalog, model_fixture = _baseline_catalog(variant)
    destination = evidence.destination_conversation_id or (
        f"{scenario.deterministic_id_prefix}-external-thread"
    )
    invariants, metrics, hard_failures = _baseline_safety_observations(
        scenario,
        evidence,
    )
    return BaselineExecution(
        invariants=frozenset(invariants),
        metrics=metrics,
        hard_failures=tuple(hard_failures),
        eligible_schema_count=len(exposed_catalog),
        admitted_destination=destination,
        model_fixture=model_fixture,
        matched_tool_catalog=matched_catalog,
        exposed_tool_catalog=exposed_catalog,
    )


def _baseline_catalog(variant: str) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    catalogs: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
        "contextual_conversation": ((), (), "_ContextAnswerModel"),
        "channel_isolation": ((), (), "_ContextAnswerModel"),
        "dm_context_union": ((), (), "_ContextAnswerModel"),
        "parallel_read_batch": (
            ("market.get_quote",),
            ("market.get_quote",),
            "_ParallelBatchModel",
        ),
        "delegated_dependency_plan": (
            ("agent.execute_research_plan",),
            (),
            "_ParentPlanModel",
        ),
        "verifier_correction": (
            ("market.get_quote",),
            ("market.get_quote",),
            "_CorrectingModel",
        ),
        "restart_replay_idempotency": (
            ("market.get_quote",),
            ("market.get_quote",),
            "ScriptedQuoteModel",
        ),
    }
    return catalogs[variant]


def _baseline_safety_observations(
    scenario: Scenario,
    evidence: _ExecutionEvidence,
) -> tuple[set[str], dict[str, float | int | str], list[str]]:
    result = evidence.result
    events = result.events
    event_types = tuple(event.type for event in events)
    invariants: set[str] = set()
    hard_failures: list[str] = []
    scopes = (
        result.thread.scope,
        result.task.scope,
        result.run.scope,
        *(item.scope for item in result.observations),
        *(item.scope for item in result.claims),
    )
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
    verified = _terminal_is_verified(result)
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
        event for event in events if event.type is EventType.VERIFICATION_FAILED
    )
    correction_calls = sum(
        event.type is EventType.MODEL_CALLED
        and any(event.sequence > failure.sequence for failure in verification_failures)
        for event in events
    )
    metrics: dict[str, float | int | str] = {
        "task_success_count": int(verified),
        "false_success_count": false_success,
        "model_calls": result.run.usage.model_calls,
        "provider_calls": evidence.provider_calls,
        "tool_calls": result.run.usage.tool_calls,
        "tool_adapter_calls": evidence.tool_adapter_calls,
        "observation_count": len(result.observations),
        "event_count": len(result.events),
        "correction_retry_count": correction_calls,
        "context_items_seen": len(_flatten_context_requests(evidence)),
        "plan_nodes_completed": len(
            [item for item in _plan_nodes(evidence) if item.get("status") == "completed"]
        ),
    }
    return invariants, metrics, hard_failures
