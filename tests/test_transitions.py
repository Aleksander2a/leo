from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.harness.models import (
    BudgetUsage,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
)
from leo.harness.transitions import (
    TransitionError,
    cancel_task_and_run,
    new_retry_run,
    requeue_task_and_run,
    require_action_task_and_run,
    resume_task_and_run,
    start_task_and_run,
)
from leo.integrations.fake import FixedClock


def _queued_pair() -> tuple[Task, Run]:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread",
        scope=scope,
        origin=OriginRef(provider="test", external_thread_id="thread"),
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    return task, Run(id="run", task_id=task.id, scope=scope)


def _active_pair() -> tuple[Task, Run, FixedClock]:
    clock = FixedClock()
    task, run = _queued_pair()
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    return active_task, active_run, clock


def test_requires_action_can_resume_or_requeue() -> None:
    task, run, clock = _active_pair()
    paused_task, paused_run = require_action_task_and_run(
        task,
        run,
        "needs_user_input",
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )

    assert paused_task.status is TaskStatus.REQUIRES_ACTION
    assert paused_run.status is RunStatus.REQUIRES_ACTION
    assert paused_run.terminal_reason == "needs_user_input"
    assert paused_run.started_at == run.started_at
    assert paused_task.version == task.version + 1
    assert paused_run.version == run.version + 1

    resumed_task, resumed_run = resume_task_and_run(paused_task, paused_run)
    assert resumed_task.status is TaskStatus.ACTIVE
    assert resumed_run.status is RunStatus.RUNNING
    assert resumed_run.started_at == run.started_at
    assert resumed_run.terminal_reason is None

    queued_task, queued_run = requeue_task_and_run(paused_task, paused_run)
    assert queued_task.status is TaskStatus.QUEUED
    assert queued_run.status is RunStatus.QUEUED
    assert queued_run.started_at is None
    assert queued_run.terminal_reason is None

    clock.advance(seconds=1)
    restarted_task, restarted_run = start_task_and_run(
        queued_task,
        queued_run,
        started_at=clock.now(),
    )
    assert restarted_task.status is TaskStatus.ACTIVE
    assert restarted_run.status is RunStatus.RUNNING
    assert restarted_run.started_at == clock.now()


def test_cancelled_queued_and_active_pairs_cannot_resume_or_start() -> None:
    queued_task, queued_run = _queued_pair()
    cancelled_task, cancelled_run = cancel_task_and_run(
        queued_task,
        queued_run,
        "operator_cancelled",
    )
    assert cancelled_task.status is TaskStatus.CANCELLED
    assert cancelled_run.status is RunStatus.CANCELLED
    assert cancelled_run.started_at is None
    assert cancelled_run.terminal_reason == "operator_cancelled"

    with pytest.raises(TransitionError, match="queued tasks and runs"):
        start_task_and_run(cancelled_task, cancelled_run, started_at=FixedClock().now())

    active_task, active_run, _ = _active_pair()
    cancelled_active_task, cancelled_active_run = cancel_task_and_run(
        active_task,
        active_run,
        "operator_cancelled",
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )
    assert cancelled_active_task.status is TaskStatus.CANCELLED
    assert cancelled_active_run.status is RunStatus.CANCELLED
    assert cancelled_active_run.started_at == active_run.started_at

    with pytest.raises(TransitionError, match="requires-action"):
        resume_task_and_run(cancelled_active_task, cancelled_active_run)


def test_retry_creates_a_fresh_queued_run_without_reopening_previous_run() -> None:
    _, failed_run, _ = _active_pair()
    failed_run = failed_run.model_copy(
        update={
            "status": RunStatus.FAILED,
            "terminal_reason": "provider_failure",
            "version": failed_run.version + 1,
        }
    )

    retry = new_retry_run(failed_run, run_id="run-retry")

    assert failed_run.status is RunStatus.FAILED
    assert retry.id == "run-retry"
    assert retry.task_id == failed_run.task_id
    assert retry.scope == failed_run.scope
    assert retry.limits == failed_run.limits
    assert retry.status is RunStatus.QUEUED
    assert retry.version == 0
    assert retry.started_at is None
    assert retry.terminal_reason is None

    for status in (RunStatus.COMPLETED, RunStatus.CANCELLED, RunStatus.REQUIRES_ACTION):
        with pytest.raises(TransitionError, match="retryable terminal"):
            new_retry_run(
                failed_run.model_copy(
                    update={
                        "status": status,
                        "terminal_reason": "state",
                    }
                ),
                run_id="run-retry",
            )


def test_unreachable_transition_and_invalid_queued_run_are_rejected() -> None:
    task, run, _ = _active_pair()
    with pytest.raises(TransitionError, match="active/running"):
        require_action_task_and_run(
            task.model_copy(update={"status": TaskStatus.QUEUED}),
            run,
            "needs_user_input",
            usage=run.usage,
        )

    with pytest.raises(ValidationError, match="queued run cannot have started_at"):
        Run(
            id="queued-with-start",
            task_id=task.id,
            scope=task.scope,
            started_at=run.started_at,
        )

    with pytest.raises(TransitionError, match="non-empty"):
        cancel_task_and_run(task, run, "")
