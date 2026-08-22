from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.harness.models import (
    Budget,
    EventDraft,
    EventType,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    TaskState,
    TaskStatus,
    Thread,
    ToolCall,
)
from leo.harness.persistence_rules import validate_commit
from leo.harness.store_errors import StoreError


def _queued_bundle() -> tuple[Thread, Task, Run]:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread",
        scope=scope,
        origin=OriginRef(provider="test", external_thread_id="thread"),
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="objective")
    return thread, task, Run(id="run", task_id=task.id, scope=scope)


def test_plan_contract_names_are_the_existing_strict_contracts() -> None:
    assert TaskState is TaskStatus
    assert ToolCall.__name__ == "ToolRequest"
    assert Budget.__name__ == "BudgetLimits"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("id", ""),
        ("thread_id", ""),
        ("objective", ""),
        ("version", -1),
    ],
)
def test_task_contract_rejects_invalid_identity_and_version(field: str, value: object) -> None:
    _, task, _ = _queued_bundle()
    with pytest.raises(ValidationError):
        Task.model_validate({**task.model_dump(), field: value})


@pytest.mark.parametrize(
    ("task_status", "run_status"),
    [
        (TaskStatus.COMPLETED, RunStatus.QUEUED),
        (TaskStatus.ACTIVE, RunStatus.COMPLETED),
        (TaskStatus.CANCELLED, RunStatus.RUNNING),
    ],
)
def test_task_and_run_terminal_fields_cannot_forge_a_legal_pair(
    task_status: TaskStatus,
    run_status: RunStatus,
) -> None:
    _, task, run = _queued_bundle()
    task_update: dict[str, object] = {"status": task_status}
    run_update: dict[str, object] = {"status": run_status}
    if task_status is TaskStatus.COMPLETED:
        task_update["final_output"] = "answer"
    if run_status not in {RunStatus.QUEUED, RunStatus.CANCELLED}:
        run_update["started_at"] = "2026-01-01T00:00:00Z"
    if run_status is RunStatus.COMPLETED:
        run_update["final_output"] = "answer"
        run_update["terminal_reason"] = "verified_completion"
    candidate_task = Task.model_validate({**task.model_dump(), **task_update})
    candidate_run = Run.model_validate({**run.model_dump(), **run_update})
    with pytest.raises(StoreError, match="invalid task/run lifecycle pair"):
        validate_commit(
            task,
            run,
            candidate_task,
            candidate_run,
            (EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={}),),
        )
