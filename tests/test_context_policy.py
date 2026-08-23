from __future__ import annotations

from datetime import timedelta

import pytest

from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    CandidateClaim,
    CardinalityBounds,
    ClaimKind,
    CompletionContract,
    CompletionProposal,
    ContextManifest,
    ContextSegment,
    EventType,
    EvidenceToolRequirement,
    ModelRequest,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolArgumentConstraint,
    ToolChoiceMode,
    ToolChoicePolicy,
    ToolEffect,
    ToolRequest,
    ToolRequests,
    ToolSpec,
    TrustedScope,
    VerifierStatus,
)
from leo.harness.ports import ContextAssemblyError
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import (
    FakeQuoteTool,
    FakeWriteTool,
    FixedClock,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)

QUOTE_REQUIREMENT = EvidenceToolRequirement(
    observation_kind="market.get_quote",
    tool_name="market.get_quote",
    required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
)


def _bundle() -> RunBundle:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread",
        scope=scope,
        origin=OriginRef(provider="test", external_thread_id="external-thread"),
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="Get NVDA quote")
    run = Run(id="run", task_id=task.id, scope=scope)
    return RunBundle(thread=thread, task=task, run=run)


def test_requirement_rejects_kind_that_tool_cannot_produce() -> None:
    with pytest.raises(ValueError, match="observation kind must match"):
        EvidenceToolRequirement(
            observation_kind="market.quote",
            tool_name="market.get_quote",
            required_arguments=QUOTE_REQUIREMENT.required_arguments,
        )


def test_required_evidence_policy_changes_to_auto_then_none_after_observation() -> None:
    clock = FixedClock()
    bundle = _bundle()
    quote_tool = FakeQuoteTool(clock).spec
    assembler = DefaultContextAssembler(
        evidence_requirements=(QUOTE_REQUIREMENT,),
        clock=clock,
    )

    required_request = assembler.assemble(bundle, (quote_tool,))

    assert required_request.tool_choice == ToolChoicePolicy(
        mode=ToolChoiceMode.REQUIRED,
        required_tool_name="market.get_quote",
        required_arguments=QUOTE_REQUIREMENT.required_arguments,
    )
    assert "completion_contract" in {segment.name for segment in required_request.manifest.segments}
    observation = Observation(
        id="obs",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call",
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="test", reference="quote"),
        observed_at=clock.now(),
        raw_hash="hash",
    )
    observed_bundle = RunBundle(
        thread=bundle.thread,
        task=bundle.task,
        run=bundle.run,
        observations=(observation,),
    )

    assert (
        assembler.assemble(observed_bundle, (quote_tool,)).tool_choice.mode is ToolChoiceMode.AUTO
    )
    assert assembler.assemble(observed_bundle, ()).tool_choice.mode is ToolChoiceMode.NONE


def test_confirmed_state_mutation_is_required_until_receipt_observation_exists() -> None:
    clock = FixedClock()
    bundle = _bundle()
    tool = ToolSpec(
        name="memory.remember",
        description="Commit the sealed explicit memory command.",
        domain="memory",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
        effect=ToolEffect.STATE_MUTATION,
    )
    assembler = DefaultContextAssembler(
        clock=clock,
        required_state_mutation_tool=tool.name,
    )

    required = assembler.assemble(bundle, (tool,))

    assert required.tool_choice == ToolChoicePolicy(
        mode=ToolChoiceMode.REQUIRED,
        required_tool_name="memory.remember",
    )
    receipt = Observation(
        id="obs-memory",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-memory",
        kind="memory.remember",
        data={"operation": "remember", "record_id": "memory-1", "revision": 1},
        source=SourceRef(provider="leo_memory", reference="memory-1"),
        observed_at=clock.now(),
        raw_hash="hash-memory",
    )
    observed = bundle.model_copy(update={"observations": (receipt,)})

    assert assembler.assemble(observed, (tool,)).tool_choice.mode is ToolChoiceMode.AUTO


def test_explicit_read_workflow_is_required_until_its_observation_exists() -> None:
    clock = FixedClock()
    bundle = _bundle()
    tool = ToolSpec(
        name="agent.execute_research_plan",
        description="Execute a bounded read-only research plan.",
        domain="harness",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.READ,
    )
    assembler = DefaultContextAssembler(
        clock=clock,
        required_read_tool=tool.name,
    )

    required = assembler.assemble(bundle, (tool,))

    assert required.tool_choice == ToolChoicePolicy(
        mode=ToolChoiceMode.REQUIRED,
        required_tool_name=tool.name,
    )
    result = Observation(
        id="obs-plan",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-plan",
        kind=tool.name,
        data={"status": "completed"},
        source=SourceRef(provider="leo-plan", reference="plan-1"),
        observed_at=clock.now(),
        raw_hash="hash-plan",
    )

    observed = bundle.model_copy(update={"observations": (result,)})
    assert assembler.assemble(observed, (tool,)).tool_choice.mode is ToolChoiceMode.AUTO


@pytest.mark.asyncio
async def test_unconstrained_required_read_tool_accepts_schema_valid_arguments() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    tool = FakeQuoteTool(clock)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    coordinator = RunCoordinator(
        store=store,
        model=ScriptedQuoteModel(),
        tools=ToolRegistry((tool,)),
        context=DefaultContextAssembler(
            clock=clock,
            required_read_tool=tool.spec.name,
        ),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.tool_calls == 1


def test_context_assembler_rejects_write_tool_in_research() -> None:
    with pytest.raises(ContextAssemblyError, match="Write tools are unavailable"):
        DefaultContextAssembler().assemble(_bundle(), (FakeWriteTool(FixedClock()).spec,))


def test_wrong_symbol_and_expired_observations_do_not_satisfy_evidence_policy() -> None:
    clock = FixedClock()
    bundle = _bundle()
    quote_tool = FakeQuoteTool(clock).spec
    assembler = DefaultContextAssembler(
        evidence_requirements=(QUOTE_REQUIREMENT,),
        clock=clock,
    )
    wrong_symbol = Observation(
        id="obs-wrong",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-wrong",
        kind="market.get_quote",
        data={"symbol": "AAPL", "price": 200.0},
        source=SourceRef(provider="test", reference="quote-aapl"),
        observed_at=clock.now(),
        expires_at=clock.now() + timedelta(minutes=1),
        raw_hash="hash-wrong",
    )
    expired_symbol = Observation(
        id="obs-expired",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-expired",
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="test", reference="quote-nvda-expired"),
        observed_at=clock.now() - timedelta(minutes=2),
        expires_at=clock.now() - timedelta(minutes=1),
        raw_hash="hash-expired",
    )
    mismatched_bundle = RunBundle(
        thread=bundle.thread,
        task=bundle.task,
        run=bundle.run,
        observations=(wrong_symbol, expired_symbol),
    )

    request = assembler.assemble(mismatched_bundle, (quote_tool,))

    assert request.tool_choice.mode is ToolChoiceMode.REQUIRED
    assert request.tool_choice.required_arguments == QUOTE_REQUIREMENT.required_arguments


def test_verifier_rejects_wrong_symbol_observation_for_trusted_requirement() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    wrong_symbol = Observation(
        id="obs-aapl",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-aapl",
        kind="market.get_quote",
        data={"symbol": "AAPL", "price": 200.0},
        source=SourceRef(provider="test", reference="quote-aapl"),
        observed_at=clock.now(),
        expires_at=clock.now() + timedelta(minutes=1),
        raw_hash="hash-aapl",
    )
    observed_bundle = RunBundle(
        thread=bundle.thread,
        task=bundle.task,
        run=bundle.run,
        observations=(wrong_symbol,),
    )
    statement = "AAPL is quoted at 200."
    proposal = CompletionProposal(
        answer=statement,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(wrong_symbol.id,),
            ),
        ),
    )
    verifier = DeterministicCompletionVerifier(
        ids,
        clock,
        evidence_requirements=(QUOTE_REQUIREMENT,),
    )

    outcome = verifier.verify(proposal, observed_bundle)

    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None
    required_checks = tuple(
        check for check in outcome.result.checks if check.name.startswith("required_evidence_")
    )
    assert required_checks
    assert all(not check.passed for check in required_checks)


def test_verifier_rejects_unknown_observation_kind_without_grounding_rule() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    observation = Observation(
        id="obs-news",
        scope=bundle.run.scope,
        run_id=bundle.run.id,
        tool_call_id="call-news",
        kind="market.news",
        data={"headline": "NVDA moved"},
        source=SourceRef(provider="test", reference="news"),
        observed_at=clock.now(),
        raw_hash="hash-news",
    )
    observed_bundle = RunBundle(
        thread=bundle.thread,
        task=bundle.task,
        run=bundle.run,
        observations=(observation,),
    )
    proposal = CompletionProposal(
        answer="NVDA moved.",
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement="NVDA moved.",
                observation_ids=(observation.id,),
            ),
        ),
    )

    outcome = DeterministicCompletionVerifier(ids, clock).verify(proposal, observed_bundle)

    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None
    support_check = next(
        check for check in outcome.result.checks if check.name.endswith("_supported")
    )
    assert support_check.passed is False
    assert "No registered grounding rule" in support_check.detail


def test_model_request_rejects_unadvertised_required_tool() -> None:
    with pytest.raises(ValueError, match="required tool must appear exactly once"):
        ModelRequest(
            objective="Get quote",
            iteration=0,
            observations=(),
            verifier_feedback=(),
            tools=(),
            tool_choice=ToolChoicePolicy(
                mode=ToolChoiceMode.REQUIRED,
                required_tool_name="market.get_quote",
                required_arguments=QUOTE_REQUIREMENT.required_arguments,
            ),
            manifest=ContextManifest(
                segments=(ContextSegment(name="objective", priority=100, pinned=True),)
            ),
        )


class _CompletionOnlyModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, request: ModelRequest) -> CompletionProposal:
        del request
        self.calls += 1
        return CompletionProposal(
            answer="No evidence was collected.",
            claims=(
                CandidateClaim(
                    kind=ClaimKind.INFERENCE,
                    statement="No evidence was collected.",
                ),
            ),
        )


class _WriteRequestModel:
    async def decide(self, request: ModelRequest) -> ToolRequests:
        del request
        return ToolRequests(
            calls=(
                ToolRequest(
                    id="write-call",
                    name="fixture.write",
                    arguments={},
                ),
            )
        )


class _WrongSymbolModel:
    async def decide(self, request: ModelRequest) -> ToolRequests:
        del request
        return ToolRequests(
            calls=(
                ToolRequest(
                    id="call-aapl",
                    name="market.get_quote",
                    arguments={"symbol": "AAPL"},
                ),
            )
        )


class _TooManySourceIdsModel:
    async def decide(self, request: ModelRequest) -> CompletionProposal:
        del request
        return CompletionProposal(
            answer="Unsupported shape.",
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement="Unsupported shape.",
                    observation_ids=("obs-one", "obs-two"),
                ),
            ),
        )


class _TooManySourceClaimsModel:
    async def decide(self, request: ModelRequest) -> CompletionProposal:
        del request
        claim = CandidateClaim(
            kind=ClaimKind.SOURCE_CLAIM,
            statement="Unsupported duplicate shape.",
            observation_ids=("obs-one",),
        )
        return CompletionProposal(
            answer="Unsupported duplicate shape.",
            claims=(claim, claim),
        )


@pytest.mark.asyncio
async def test_coordinator_fails_closed_before_model_when_required_tool_is_unavailable() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    model = _CompletionOnlyModel()
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(
            evidence_requirements=(QUOTE_REQUIREMENT,),
            clock=clock,
        ),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    assert result.run.status is RunStatus.FAILED
    assert result.run.usage.model_calls == 0
    assert model.calls == 0
    assert result.run.terminal_reason == (
        "context_assembly_error:required_evidence_tool_unavailable"
    )


@pytest.mark.asyncio
async def test_coordinator_rejects_completion_when_required_tool_was_not_requested() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    model = _CompletionOnlyModel()
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=ToolRegistry((FakeQuoteTool(clock),)),
        context=DefaultContextAssembler(
            evidence_requirements=(QUOTE_REQUIREMENT,),
            clock=clock,
        ),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    # A required-tool violation is a one-turn correctable mistake, not a reason to
    # kill the run outright: the coordinator retries with corrective feedback
    # instead of failing on the model's first (wrong) decision. This fixture model
    # never calls the tool, so the bounded retry loop eventually exhausts its
    # budget -- but it never hits the old instant, zero-attempt terminal failure.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert result.run.usage.tool_calls == 0
    assert any(
        "market.get_quote" in feedback and "required" in feedback.lower()
        for feedback in result.task.verifier_feedback
    )
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["tool_choice"] == "required"
    assert context_event.payload["required_tool"] == "market.get_quote"


@pytest.mark.asyncio
async def test_research_write_proposal_is_rejected_before_tool_execution() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    write_tool = FakeWriteTool(clock)
    coordinator = RunCoordinator(
        store=store,
        model=_WriteRequestModel(),
        tools=ToolRegistry((write_tool,)),
        context=DefaultContextAssembler(),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    # Requesting a disabled tool is retried with corrective feedback instead of
    # instantly failing the run; this fixture model always asks for the write
    # tool, so the bounded retry loop exhausts its budget without ever executing it.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert result.run.usage.tool_calls == 0
    assert any("disabled" in feedback.lower() for feedback in result.task.verifier_feedback)
    assert write_tool.calls == 0


@pytest.mark.asyncio
async def test_coordinator_rejects_wrong_required_arguments_before_tool_execution() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    coordinator = RunCoordinator(
        store=store,
        model=_WrongSymbolModel(),
        tools=ToolRegistry((FakeQuoteTool(clock),)),
        context=DefaultContextAssembler(
            evidence_requirements=(QUOTE_REQUIREMENT,),
            clock=clock,
        ),
        verifier=DeterministicCompletionVerifier(
            ids,
            clock,
            evidence_requirements=(QUOTE_REQUIREMENT,),
        ),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    # Wrong required arguments are retried with corrective feedback instead of
    # instantly failing; this fixture model always requests the wrong symbol, so
    # the bounded retry loop exhausts its budget without ever executing the tool.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert result.run.usage.tool_calls == 0
    assert result.observations == ()
    assert any(
        "market.get_quote" in feedback and "argument" in feedback.lower()
        for feedback in result.task.verifier_feedback
    )


@pytest.mark.asyncio
async def test_coordinator_enforces_completion_contract_before_verification() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    contract = CompletionContract(
        source_claim_count=CardinalityBounds(minimum=1, maximum=1),
        source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
        inference_count=CardinalityBounds(minimum=0, maximum=0),
        guidance="Return one narrowly grounded source claim.",
    )
    coordinator = RunCoordinator(
        store=store,
        model=_TooManySourceIdsModel(),
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(completion_contract=contract),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    # A completion-contract cardinality violation is retried with corrective
    # feedback instead of instantly failing the run; this fixture model always
    # returns the same over-cited claim, so the bounded retry loop exhausts its
    # budget, and verification is still never reached for this malformed shape.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert any(
        "observation id" in feedback.lower() for feedback in result.task.verifier_feedback
    )
    assert EventType.VERIFICATION_STARTED not in {event.type for event in result.events}


@pytest.mark.asyncio
async def test_coordinator_rejects_extra_source_claim_before_verification() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    contract = CompletionContract(
        source_claim_count=CardinalityBounds(minimum=1, maximum=1),
        source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
        inference_count=CardinalityBounds(minimum=0, maximum=0),
        guidance="Return exactly one source claim.",
    )
    coordinator = RunCoordinator(
        store=store,
        model=_TooManySourceClaimsModel(),
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(completion_contract=contract),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=bundle.run.scope, actor_id="actor"),
    )

    # Too many source claims is retried with corrective feedback instead of
    # instantly failing the run; this fixture model always returns the same
    # duplicated claim, so the bounded retry loop exhausts its budget, and
    # verification is still never reached for this malformed shape.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert any(
        "source-backed claim" in feedback.lower() for feedback in result.task.verifier_feedback
    )
    assert EventType.VERIFICATION_STARTED not in {event.type for event in result.events}
