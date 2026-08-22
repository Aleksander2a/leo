from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from leo.domain.conversation import ConversationKind, ConversationRef
from leo.harness.child_evidence import ChildEvidenceEnvelope, ChildResult
from leo.harness.models import (
    LEGAL_TASK_RUN_PAIRS,
    Budget,
    BudgetLimits,
    BudgetUsage,
    CandidateClaim,
    CompletionProposal,
    ContextManifest,
    EventType,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunEvent,
    RunStatus,
    ScopeKey,
    SourceManifest,
    SourceRef,
    Task,
    TaskState,
    TaskStatus,
    Thread,
    ToolCall,
    ToolRequest,
)
from leo.harness.plan_models import Delegation, Plan, PlanNode, PlanRevision

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _base() -> tuple[Thread, Task, Run]:
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation-a"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective="Do the work")
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return thread, task, run


def _pair(task_status: TaskStatus, run_status: RunStatus) -> tuple[Task, Run]:
    _, task, run = _base()
    task_values: dict[str, object] = {"status": task_status}
    run_values: dict[str, object] = {"status": run_status}
    if task_status is TaskStatus.COMPLETED:
        task_values["final_output"] = "verified answer"
    if run_status not in {RunStatus.QUEUED, RunStatus.CANCELLED}:
        run_values["started_at"] = NOW
    if run_status is RunStatus.COMPLETED:
        run_values.update(
            final_output="verified answer",
            terminal_reason="verified_completion",
        )
    elif run_status not in {RunStatus.QUEUED, RunStatus.RUNNING}:
        run_values["terminal_reason"] = f"safe_{run_status.value}"
    return Task.model_validate({**task.model_dump(), **task_values}), Run.model_validate(
        {**run.model_dump(), **run_values}
    )


@pytest.mark.parametrize("task_status,run_status", sorted(LEGAL_TASK_RUN_PAIRS))
def test_every_declared_task_run_pair_round_trips(
    task_status: TaskStatus,
    run_status: RunStatus,
) -> None:
    thread, _, _ = _base()
    task, run = _pair(task_status, run_status)

    bundle = RunBundle(thread=thread, task=task, run=run)

    assert RunBundle.model_validate_json(bundle.model_dump_json()) == bundle


@pytest.mark.parametrize(
    "task_status,run_status",
    tuple(
        (task_status, run_status)
        for task_status in TaskStatus
        for run_status in RunStatus
        if (task_status, run_status) not in LEGAL_TASK_RUN_PAIRS
    ),
)
def test_every_undeclared_task_run_pair_fails_closed(
    task_status: TaskStatus,
    run_status: RunStatus,
) -> None:
    thread, _, _ = _base()
    task, run = _pair(task_status, run_status)

    with pytest.raises(ValidationError, match="invalid task/run lifecycle pair"):
        RunBundle(thread=thread, task=task, run=run)


@pytest.mark.parametrize(
    "updates,error",
    (
        (
            {
                "status": RunStatus.COMPLETED,
                "started_at": NOW,
                "final_output": "answer",
            },
            "verified completion reason",
        ),
        (
            {"status": RunStatus.FAILED, "started_at": NOW},
            "requires a reason",
        ),
        (
            {"status": RunStatus.RUNNING, "started_at": NOW, "terminal_reason": "forged"},
            "cannot carry a terminal reason",
        ),
        (
            {
                "status": RunStatus.RUNNING,
                "started_at": NOW,
                "deadline_at": NOW - timedelta(seconds=1),
            },
            "deadline must be after",
        ),
    ),
)
def test_run_terminal_field_matrix_rejects_partial_or_forged_state(
    updates: dict[str, object],
    error: str,
) -> None:
    _, _, run = _base()

    with pytest.raises(ValidationError, match=error):
        Run.model_validate({**run.model_dump(), **updates})


def test_usage_contract_rejects_inconsistent_tokens_and_unfenced_reservation() -> None:
    with pytest.raises(ValidationError, match="total tokens"):
        BudgetUsage(prompt_tokens=3, completion_tokens=4, total_tokens=8)
    with pytest.raises(ValidationError, match="reservation ID"):
        BudgetUsage(reserved_cost=0.01)


def test_whitespace_and_extra_authority_fields_fail_at_contract_parse() -> None:
    with pytest.raises(ValidationError):
        ScopeKey(organization_id="   ", strategy_id="strategy")
    with pytest.raises(ValidationError, match="extra"):
        ToolRequest.model_validate(
            {
                "id": "call",
                "name": "research.read",
                "arguments": {},
                "conversation_id": "forged",
            }
        )
    with pytest.raises(ValidationError, match="extra"):
        CompletionProposal.model_validate(
            {
                "answer": "answer",
                "claims": [],
                "complete_parent": True,
            }
        )


def test_conversation_and_named_manifest_contracts_reconcile_without_duplicates() -> None:
    conversation = ConversationRef(
        provider="slack",
        team_id="team",
        external_id="conversation-a",
        kind=ConversationKind.CHANNEL,
    )
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(
            provider=conversation.provider,
            external_thread_id=conversation.external_id,
        ),
    )

    assert thread.origin.provider == conversation.provider
    assert thread.origin.external_thread_id == conversation.external_id
    assert SourceManifest is ContextManifest
    assert ChildResult is ChildEvidenceEnvelope
    assert TaskState is TaskStatus
    assert ToolCall is ToolRequest
    assert Budget is BudgetLimits


def test_model_and_child_return_contracts_have_no_runtime_authority_fields() -> None:
    forbidden = {
        "approval",
        "approved",
        "complete_parent",
        "conversation_id",
        "deliver_to_slack",
        "memory_write",
        "organization_id",
        "roles",
        "scope",
        "strategy_id",
    }

    assert forbidden.isdisjoint(CompletionProposal.model_fields)
    assert forbidden.isdisjoint(CandidateClaim.model_fields)
    assert forbidden.isdisjoint(ChildResult.model_fields)
    assert forbidden.isdisjoint(Delegation.model_fields)
    assert {"scope", "parent_task_id", "parent_run_id"}.issubset(Plan.model_fields)
    assert {"plan_id", "revision_id", "definition", "status"}.issubset(PlanNode.model_fields)
    assert {"digest", "parent_digest", "nodes"}.issubset(PlanRevision.model_fields)


def test_run_bundle_rejects_cross_scope_duplicate_and_noncontiguous_evidence() -> None:
    thread, task, run = _base()
    observation = Observation(
        id="observation-1",
        scope=SCOPE,
        run_id=run.id,
        tool_call_id="call-1",
        kind="fixture.read",
        data={"value": "safe"},
        source=SourceRef(provider="fixture", reference="source-1"),
        observed_at=NOW,
        raw_hash="hash-1",
    )
    wrong_scope = observation.model_copy(
        update={"scope": ScopeKey(organization_id="other", strategy_id="strategy")}
    )
    with pytest.raises(ValidationError, match="outside the run scope"):
        RunBundle(thread=thread, task=task, run=run, observations=(wrong_scope,))
    with pytest.raises(ValidationError, match="duplicate observations"):
        RunBundle(
            thread=thread,
            task=task,
            run=run,
            observations=(observation, observation),
        )

    event = RunEvent(
        id="event-2",
        run_id=run.id,
        task_id=task.id,
        sequence=2,
        type=EventType.TASK_STARTED,
        occurred_at=NOW,
        iteration=0,
    )
    with pytest.raises(ValidationError, match="contiguous and ordered"):
        RunBundle(thread=thread, task=task, run=run, events=(event,))
