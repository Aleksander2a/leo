"""Composition root for deterministic harness demonstrations."""

from __future__ import annotations

from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    BudgetLimits,
    BudgetUsage,
    CoordinatorResult,
    OriginRef,
    Run,
    ScopeKey,
    Task,
    Thread,
    TrustedScope,
)
from leo.harness.ports import ModelGateway
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import (
    FakeQuoteTool,
    FixedClock,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)


async def run_quote_smoke(
    *,
    model: ModelGateway | None = None,
    limits: BudgetLimits | None = None,
    objective: str = "Report the current NVDA quote from an allowed market tool.",
    scope: ScopeKey | None = None,
    actor_id: str = "cli-user",
    origin: OriginRef | None = None,
    tool_registry: ToolRegistry | None = None,
    initial_usage: BudgetUsage | None = None,
) -> CoordinatorResult:
    clock = FixedClock()
    return await _run_smoke(
        model=model or ScriptedQuoteModel(),
        limits=limits,
        objective=objective,
        scope=scope,
        actor_id=actor_id,
        origin=origin,
        tool_registry=(
            tool_registry if tool_registry is not None else ToolRegistry((FakeQuoteTool(clock),))
        ),
        initial_usage=initial_usage,
        clock=clock,
        require_source_claim=True,
        required_observation_kinds=frozenset({"market.get_quote"}),
    )


async def run_conversation_smoke(
    *,
    model: ModelGateway,
    objective: str,
    limits: BudgetLimits | None = None,
    scope: ScopeKey | None = None,
    actor_id: str = "cli-user",
    origin: OriginRef | None = None,
    tool_registry: ToolRegistry | None = None,
    initial_usage: BudgetUsage | None = None,
) -> CoordinatorResult:
    """Run an arbitrary deterministic fixture through the same parent coordinator."""

    return await _run_smoke(
        model=model,
        limits=limits,
        objective=objective,
        scope=scope,
        actor_id=actor_id,
        origin=origin,
        tool_registry=tool_registry or ToolRegistry(()),
        initial_usage=initial_usage,
        clock=FixedClock(),
        require_source_claim=False,
        required_observation_kinds=frozenset(),
    )


async def _run_smoke(
    *,
    model: ModelGateway,
    limits: BudgetLimits | None,
    objective: str,
    scope: ScopeKey | None,
    actor_id: str,
    origin: OriginRef | None,
    tool_registry: ToolRegistry,
    initial_usage: BudgetUsage | None,
    clock: FixedClock,
    require_source_claim: bool,
    required_observation_kinds: frozenset[str],
) -> CoordinatorResult:
    ids = SequentialIdGenerator()
    scope = scope or ScopeKey(organization_id="demo-org", strategy_id="technology-ls")
    trusted_scope = TrustedScope(
        namespace=scope, actor_id=actor_id, roles=frozenset({"researcher"})
    )
    thread = Thread(
        id="thread-001",
        scope=scope,
        origin=origin or OriginRef(provider="fixture", external_thread_id="fixture-thread-001"),
    )
    task = Task(
        id="task-001",
        thread_id=thread.id,
        scope=scope,
        objective=objective,
    )
    run = Run(
        id="run-001",
        task_id=task.id,
        scope=scope,
        limits=limits or BudgetLimits(),
        usage=initial_usage or BudgetUsage(),
    )
    store = InMemoryRunStore(clock, ids)
    await store.seed(thread, task, run)
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=tool_registry,
        context=DefaultContextAssembler(),
        verifier=DeterministicCompletionVerifier(
            ids,
            clock,
            require_source_claim=require_source_claim,
            required_observation_kinds=required_observation_kinds,
        ),
        clock=clock,
        ids=ids,
    )
    return await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=trusted_scope,
    )
