from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.harness.context import DefaultContextAssembler
from leo.harness.models import (
    CompletionProposal,
    ContextItem,
    ContextItemKind,
    EvidenceToolRequirement,
    ModelRequest,
    ModelTurnResult,
    OriginRef,
    Run,
    RunBundle,
    ScopeKey,
    Task,
    Thread,
    ToolArgumentConstraint,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.subagents import SubagentPlanTool, SubagentResearchTool
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import (
    FakeQuoteTool,
    FixedClock,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)
from leo.persistence.context_loader import ConversationContextRequest

SCOPE = ScopeKey(organization_id="demo-org", strategy_id="default-domain")


def _bundle() -> RunBundle:
    thread = Thread(
        id="thread-one",
        scope=SCOPE,
        origin=OriginRef(provider="slack", external_thread_id="slack:T1:C1:1.0"),
    )
    task = Task(id="task-one", thread_id=thread.id, scope=SCOPE, objective="What did we decide?")
    run = Run(id="run-one", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run)


def test_channel_context_projection_cannot_include_another_conversation() -> None:
    with pytest.raises(ValidationError, match="exact destination"):
        ConversationContextRequest(
            team_id="T1",
            destination_id="C1",
            destination_kind="channel",
            actor_id="U1",
            objective="hello",
            current_task_id="task-one",
            current_event_id="event-one",
            current_message_ts="2.0",
            thread_root_ts="1.0",
            allowed_conversation_ids=("C1", "C2"),
            access_hash="a" * 64,
            current_thread_namespace_id="slack:T1:C1:1.0",
        )


def test_dm_context_projection_accepts_only_a_canonical_exact_union() -> None:
    request = ConversationContextRequest(
        team_id="T1",
        destination_id="D1",
        destination_kind="dm",
        actor_id="U1",
        objective="compare our channel discussions",
        current_task_id="task-one",
        current_event_id="event-one",
        current_message_ts="2.0",
        thread_root_ts="1.0",
        allowed_conversation_ids=("C1", "C2", "D1", "G1"),
        access_hash="a" * 64,
        current_thread_namespace_id="slack:T1:D1:1.0",
    )
    assert request.allowed_conversation_ids == ("C1", "C2", "D1", "G1")
    with pytest.raises(ValidationError, match="sorted and unique"):
        ConversationContextRequest(
            team_id="T1",
            destination_id="D1",
            destination_kind="dm",
            actor_id="U1",
            objective="compare",
            current_task_id="task-one",
            current_event_id="event-one",
            current_message_ts="2.0",
            thread_root_ts="1.0",
            allowed_conversation_ids=("D1", "C1"),
            access_hash="a" * 64,
            current_thread_namespace_id="slack:T1:D1:1.0",
        )


def test_selected_context_is_present_in_the_model_request_and_manifest() -> None:
    context_item = ContextItem(
        id="turn:prior",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: Prefer concise answers.\nLeo: Understood.",
        conversation_id="C1",
        source_scope=SCOPE,
    )
    request = DefaultContextAssembler(
        context_items=(context_item,),
        authority_snapshot_ids=("access:" + "a" * 64,),
    ).assemble(_bundle(), ())

    assert isinstance(request, ModelRequest)
    assert request.context_items == (context_item,)
    segment = next(item for item in request.manifest.segments if item.name == "scoped_context")
    assert segment.source_ids == (context_item.id,)
    authority = next(
        item for item in request.manifest.segments if item.name == "context_authority_snapshot"
    )
    assert authority.source_ids == ("access:" + "a" * 64,)


class _CompletingModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        assert any(item.content.startswith("User: Prior") for item in request.context_items)
        return ModelTurnResult(
            decision=CompletionProposal(answer="The delegated finding is complete.", claims=()),
            provider="fixture",
            model="fixture-model",
        )


@pytest.mark.asyncio
async def test_subagent_inherits_scope_and_returns_a_typed_parent_observation() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    item = ContextItem(
        id="turn:one",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: Prior scoped context.",
        conversation_id="C1",
        source_scope=SCOPE,
    )
    tool = SubagentResearchTool(
        model=_CompletingModel(),
        tools=ToolRegistry(()),
        context_items=(item,),
        clock=clock,
        ids=ids,
    )

    outcome = await tool.execute(
        tool.validate({"objective": "Resolve one bounded subquestion.", "max_turns": 2}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
            run_id="parent-run",
            tool_call_id="delegate-one",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["answer"] == "The delegated finding is complete."
    assert str(outcome.data["child_run_id"]).startswith("subrun-")
    assert outcome.data["schema_version"] == "child-evidence-v1"
    assert outcome.data["verified_source_claims"] == []


def _quote_requirement(_: str) -> tuple[EvidenceToolRequirement, ...]:
    return (
        EvidenceToolRequirement(
            observation_kind="market.get_quote",
            tool_name="market.get_quote",
            required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
        ),
    )


@pytest.mark.asyncio
async def test_subagent_requirement_selector_forces_and_exports_verified_evidence() -> None:
    clock = FixedClock()
    tool = SubagentResearchTool(
        model=ScriptedQuoteModel(),
        tools=ToolRegistry((FakeQuoteTool(clock),)),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
        requirement_selector=_quote_requirement,
    )

    outcome = await tool.execute(
        tool.validate({"objective": "Obtain the current NVDA market quote."}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
            run_id="parent-run",
            tool_call_id="delegate-sourced",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["observation_count"] == 1
    assert len(outcome.data["verified_source_claims"]) == 1


@pytest.mark.asyncio
async def test_subagent_rejects_cross_organization_context_before_model_call() -> None:
    clock = FixedClock()
    foreign = ContextItem(
        id="turn:foreign",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Foreign context",
        conversation_id="C9",
        source_scope=ScopeKey(organization_id="other-org", strategy_id="strategy"),
    )
    tool = SubagentResearchTool(
        model=_CompletingModel(),
        tools=ToolRegistry(()),
        context_items=(foreign,),
        clock=clock,
        ids=SequentialIdGenerator(),
    )

    outcome = await tool.execute(
        tool.validate({"objective": "Try to broaden scope."}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
            run_id="parent-run",
            tool_call_id="delegate-two",
        ),
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "SUBAGENT_CONTEXT_SCOPE_MISMATCH"


class _PlanCompletingModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        if request.objective == "Synthesize the findings.":
            assert any(
                item.kind is ContextItemKind.SUBAGENT_RESULT for item in request.context_items
            )
        return ModelTurnResult(
            decision=CompletionProposal(answer=f"Completed: {request.objective}", claims=()),
            provider="fixture",
            model="fixture-model",
        )


@pytest.mark.asyncio
async def test_subagent_plan_executes_dependencies_and_returns_parent_owned_results() -> None:
    clock = FixedClock()
    tool = SubagentPlanTool(
        model=_PlanCompletingModel(),
        tools=ToolRegistry(()),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
    )
    arguments = tool.validate(
        {
            "goal": "Research and synthesize.",
            "max_concurrency": 2,
            "nodes": [
                {"id": "market", "objective": "Research the market."},
                {"id": "company", "objective": "Research the company."},
                {
                    "id": "synthesis",
                    "objective": "Synthesize the findings.",
                    "depends_on": ["market", "company"],
                },
            ],
        }
    )

    outcome = await tool.execute(
        arguments,
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
            run_id="parent-run",
            tool_call_id="plan-one",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["status"] == "completed"
    assert outcome.data["completed_count"] == 3
    assert [node["id"] for node in outcome.data["nodes"]] == [
        "market",
        "company",
        "synthesis",
    ]
    assert all(
        node["child_evidence"]["verified_source_claims"] == [] for node in outcome.data["nodes"]
    )


@pytest.mark.asyncio
async def test_subagent_plan_propagates_child_evidence_requirements() -> None:
    clock = FixedClock()
    tool = SubagentPlanTool(
        model=ScriptedQuoteModel(),
        tools=ToolRegistry((FakeQuoteTool(clock),)),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
        requirement_selector=_quote_requirement,
    )
    arguments = tool.validate(
        {
            "goal": "Gather verified market evidence.",
            "nodes": [{"id": "quote", "objective": "Obtain the NVDA market quote."}],
        }
    )

    outcome = await tool.execute(
        arguments,
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
            run_id="parent-run",
            tool_call_id="plan-sourced",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    claims = outcome.data["nodes"][0]["child_evidence"]["verified_source_claims"]
    assert len(claims) == 1


def test_subagent_plan_rejects_a_dependency_cycle() -> None:
    tool = SubagentPlanTool(
        model=_PlanCompletingModel(),
        tools=ToolRegistry(()),
        context_items=(),
        clock=FixedClock(),
        ids=SequentialIdGenerator(),
    )

    with pytest.raises(ValidationError, match="acyclic"):
        tool.validate(
            {
                "goal": "Invalid plan.",
                "nodes": [
                    {"id": "one", "objective": "One.", "depends_on": ["two"]},
                    {"id": "two", "objective": "Two.", "depends_on": ["one"]},
                ],
            }
        )
