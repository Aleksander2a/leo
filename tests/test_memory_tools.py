from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.domain.conversation import ConversationKind
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    CompletionProposal,
    ModelRequest,
    OriginRef,
    Run,
    RunPhase,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolRequests,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.memory.models import MemoryStatus, MemoryVisibility
from leo.memory.service import ExplicitMemoryService
from leo.memory.store import InMemoryMemoryStore
from leo.memory.tools import (
    MemoryMutationCommand,
    bind_memory_mutation_authority,
    build_explicit_memory_tools,
    parse_explicit_memory_intent,
)

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
NOW = datetime(2026, 8, 21, tzinfo=UTC)


def _authority(
    objective: str,
    *,
    kind: ConversationKind = ConversationKind.CHANNEL,
    conversation_id: str = "C1",
    actor_id: str = "U1",
    event_id: str = "Ev1",
    task_id: str = "task-1",
    run_id: str = "run-1",
):
    authority = bind_memory_mutation_authority(
        scope=SCOPE,
        team_id="T1",
        conversation_id=conversation_id,
        conversation_kind=kind,
        actor_id=actor_id,
        event_id=event_id,
        task_id=task_id,
        run_id=run_id,
        message_reference="1710000000.001",
        objective=objective,
    )
    assert authority is not None
    return authority


def _context(*, actor_id: str = "U1", run_id: str = "run-1") -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=SCOPE,
            actor_id=actor_id,
            roles=frozenset({"researcher"}),
        ),
        run_id=run_id,
        tool_call_id="call-1",
    )


@pytest.mark.parametrize(
    ("text", "command"),
    (
        ("remember that the demo is called Helios", MemoryMutationCommand.REMEMBER),
        (
            "correct memory memory-001 to the demo is called Atlas",
            MemoryMutationCommand.CORRECT,
        ),
        (
            "forget memory memory-001 because the user withdrew it",
            MemoryMutationCommand.FORGET,
        ),
        ("forget memory memory-001", MemoryMutationCommand.FORGET),
    ),
)
def test_parser_accepts_only_deterministic_explicit_commands(
    text: str,
    command: MemoryMutationCommand,
) -> None:
    intent = parse_explicit_memory_intent(text)

    assert intent is not None
    assert intent.command is command


@pytest.mark.parametrize(
    ("text", "content"),
    (
        (
            "Please remember for this conversation that Project Borealis's display "
            "preference is amber hexagons.",
            "Project Borealis's display preference is amber hexagons.",
        ),
        (
            "Can you remember that the demo is called Helios?",
            "the demo is called Helios",
        ),
        (
            "Could you please remember for this conversation that the launch is in October?",
            "the launch is in October",
        ),
        (
            "Would you remember in this channel: the preferred shape is a hexagon?",
            "the preferred shape is a hexagon",
        ),
        (
            "Can you remember the demo preference is amber?",
            "the demo preference is amber",
        ),
    ),
)
def test_parser_accepts_conversational_explicit_remember_requests(
    text: str,
    content: str,
) -> None:
    intent = parse_explicit_memory_intent(text)

    assert intent is not None
    assert intent.command is MemoryMutationCommand.REMEMBER
    assert intent.content == content


@pytest.mark.parametrize(
    "text",
    (
        "I remember that the demo is called Helios.",
        "Can you remember what we decided earlier?",
        "Could you remember our prior decision?",
        "Do you remember that the demo is called Helios?",
        "remember",
        "correct this memory somehow",
        "forget what I said earlier",
        "What do you remember?",
    ),
)
def test_parser_fails_closed_for_ambiguous_or_non_command_language(text: str) -> None:
    assert parse_explicit_memory_intent(text) is None


@pytest.mark.parametrize(
    "kind",
    (
        ConversationKind.CHANNEL,
        ConversationKind.GROUP_DM,
        ConversationKind.SHARED,
        ConversationKind.EXTERNAL,
    ),
)
def test_non_dm_authority_is_exact_conversation_local(kind: ConversationKind) -> None:
    authority = _authority("remember this exact conversation fact", kind=kind)

    assert authority.visibility is MemoryVisibility.CONVERSATION_LOCAL
    assert authority.namespace_id == "C1"
    assert {source.namespace_id for source in authority.sources()} == {"C1"}
    assert {source.visibility for source in authority.sources()} == {
        MemoryVisibility.CONVERSATION_LOCAL
    }


def test_dm_authority_is_actor_private() -> None:
    authority = _authority("remember my private preference", kind=ConversationKind.DM)

    assert authority.visibility is MemoryVisibility.ACTOR_PRIVATE
    assert authority.namespace_id == "U1"
    assert {source.namespace_id for source in authority.sources()} == {"U1"}


@pytest.mark.asyncio
async def test_memory_tool_has_internal_mutation_effect_and_no_model_authority_arguments() -> None:
    authority = _authority("remember that the demo is called Helios")
    store = InMemoryMemoryStore()
    tool = build_explicit_memory_tools(
        service=ExplicitMemoryService(store, FixedClock(NOW), SequentialIdGenerator()),
        authority=authority,
        clock=FixedClock(NOW),
    )[0]
    registry = ToolRegistry((tool,))

    assert tool.spec.effect is ToolEffect.STATE_MUTATION
    assert tool.spec.retry.max_attempts == 1
    assert tool.spec.input_schema["properties"] == {}
    assert (
        registry.requests_are_parallel_safe(
            (
                ToolRequest(id="call-a", name=tool.spec.name, arguments={}),
                ToolRequest(id="call-b", name=tool.spec.name, arguments={}),
            ),
            RunPhase.RESEARCH,
        )
        is False
    )
    request = ToolRequest(
        id="call-1",
        name=tool.spec.name,
        arguments={"namespace_id": "C-ATTACKER", "actor_id": "U-ATTACKER"},
    )
    outcome = await registry.execute(request, _context(), RunPhase.RESEARCH)
    direct_outcome = await tool.execute({"record_id": "memory-attacker"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "INVALID_TOOL_ARGUMENTS"
    assert isinstance(direct_outcome, ToolFailure)
    assert direct_outcome.code == "MEMORY_ARGUMENTS_NOT_ALLOWED"
    assert store._records == {}


class _CommitThenAnswerModel:
    async def decide(self, request: ModelRequest) -> ToolRequests | CompletionProposal:
        if not request.observations:
            return ToolRequests(
                calls=(ToolRequest(id="call-memory", name="memory.remember", arguments={}),)
            )
        return CompletionProposal(answer="I remembered that in this conversation.", claims=())


@pytest.mark.asyncio
async def test_coordinator_requires_memory_commit_before_verified_completion() -> None:
    clock = FixedClock(NOW)
    ids = SequentialIdGenerator()
    memory_store = InMemoryMemoryStore()
    authority = _authority("remember that the demo is called Helios")
    tool = build_explicit_memory_tools(
        service=ExplicitMemoryService(memory_store, clock, ids),
        authority=authority,
        clock=clock,
    )[0]
    thread = Thread(
        id="thread-1",
        scope=SCOPE,
        origin=OriginRef(provider="slack", external_thread_id="slack:T1:C1:1"),
    )
    task = Task(
        id="task-1",
        thread_id=thread.id,
        scope=SCOPE,
        objective="remember that the demo is called Helios",
    )
    run = Run(id="run-1", task_id=task.id, scope=SCOPE)
    run_store = InMemoryRunStore(clock, ids)
    await run_store.seed(thread, task, run)
    coordinator = RunCoordinator(
        store=run_store,
        model=_CommitThenAnswerModel(),
        tools=ToolRegistry((tool,)),
        context=DefaultContextAssembler(
            clock=clock,
            required_state_mutation_tool=tool.spec.name,
        ),
        verifier=DeterministicCompletionVerifier(
            ids,
            clock,
            require_source_claim=False,
            required_observation_kinds=frozenset({tool.spec.name}),
        ),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=TrustedScope(
            namespace=SCOPE,
            actor_id="U1",
            roles=frozenset({"researcher"}),
        ),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.tool_calls == 1
    assert [observation.kind for observation in result.observations] == ["memory.remember"]
    assert len(memory_store._records) == 1


@pytest.mark.asyncio
async def test_remember_correct_forget_are_bound_and_provenance_linked() -> None:
    store = InMemoryMemoryStore()
    service = ExplicitMemoryService(store, FixedClock(NOW), SequentialIdGenerator())

    remember = build_explicit_memory_tools(
        service=service,
        authority=_authority("remember that the demo is called Helios"),
        clock=FixedClock(NOW),
    )[0]
    remembered = await remember.execute({}, _context())
    assert isinstance(remembered, ToolSuccess)
    record_id = str(remembered.data["record_id"])
    first = await store.current(SCOPE, record_id)
    assert first is not None
    assert len(first.source_ids) == 3
    first_sources = tuple(store._sources[source_id] for source_id in first.source_ids)
    assert {source.source_kind for source in first_sources} == {
        "slack_event",
        "leo_task",
        "slack_message",
    }

    correct = build_explicit_memory_tools(
        service=service,
        authority=_authority(
            f"correct memory {record_id} to the demo is called Atlas",
            event_id="Ev2",
            task_id="task-2",
            run_id="run-2",
        ),
        clock=FixedClock(NOW),
    )[0]
    corrected = await correct.execute({}, _context(run_id="run-2"))
    assert isinstance(corrected, ToolSuccess)
    current = await store.current(SCOPE, record_id)
    assert current is not None
    assert current.number == 2
    assert current.content == "the demo is called Atlas"
    assert len(current.source_ids) == 6

    forget = build_explicit_memory_tools(
        service=service,
        authority=_authority(
            f"forget memory {record_id} because the user withdrew it",
            event_id="Ev3",
            task_id="task-3",
            run_id="run-3",
        ),
        clock=FixedClock(NOW),
    )[0]
    forgotten = await forget.execute({}, _context(run_id="run-3"))
    assert isinstance(forgotten, ToolSuccess)
    assert forgotten.data["status"] == MemoryStatus.RETRACTED.value
    retracted = store._revisions[(record_id, 3)]
    assert len(retracted.source_ids) == 9
    assert await store.current(SCOPE, record_id) is None


@pytest.mark.asyncio
async def test_cross_destination_and_runtime_authority_mismatches_fail_closed() -> None:
    store = InMemoryMemoryStore()
    service = ExplicitMemoryService(store, FixedClock(NOW), SequentialIdGenerator())
    remember = build_explicit_memory_tools(
        service=service,
        authority=_authority("remember that this belongs only to C1"),
        clock=FixedClock(NOW),
    )[0]
    remembered = await remember.execute({}, _context())
    assert isinstance(remembered, ToolSuccess)
    record_id = str(remembered.data["record_id"])

    cross_destination = build_explicit_memory_tools(
        service=service,
        authority=_authority(
            f"correct memory {record_id} to leak this into another channel",
            conversation_id="C2",
            event_id="Ev2",
            task_id="task-2",
            run_id="run-2",
        ),
        clock=FixedClock(NOW),
    )[0]
    rejected = await cross_destination.execute({}, _context(run_id="run-2"))
    assert isinstance(rejected, ToolFailure)
    assert rejected.code == "MEMORY_COMMAND_REJECTED"

    actor_mismatch = await remember.execute({}, _context(actor_id="U-OTHER"))
    run_mismatch = await remember.execute({}, _context(run_id="run-other"))
    assert isinstance(actor_mismatch, ToolFailure)
    assert actor_mismatch.code == "MEMORY_AUTHORITY_MISMATCH"
    assert isinstance(run_mismatch, ToolFailure)
    assert run_mismatch.code == "MEMORY_AUTHORITY_MISMATCH"
