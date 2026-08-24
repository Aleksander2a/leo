"""Intention-specific lifecycle transitions; terminal states are immutable."""

from __future__ import annotations

from datetime import datetime, timedelta

from leo.harness.models import (
    BudgetUsage,
    PlannedStep,
    ReasoningStep,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    VerifiedCompletion,
)


class TransitionError(RuntimeError):
    pass


def start_task_and_run(task: Task, run: Run, *, started_at: datetime) -> tuple[Task, Run]:
    if task.status is not TaskStatus.QUEUED or run.status is not RunStatus.QUEUED:
        raise TransitionError("only queued tasks and runs can start")
    return (
        task.model_copy(update={"status": TaskStatus.ACTIVE, "version": task.version + 1}),
        run.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "started_at": started_at,
                "deadline_at": started_at + timedelta(seconds=run.limits.max_elapsed_seconds),
                "version": run.version + 1,
            }
        ),
    )


def require_action_task_and_run(
    task: Task,
    run: Run,
    reason: str,
    *,
    usage: BudgetUsage,
    advance_iteration: bool = False,
) -> tuple[Task, Run]:
    _require_active(task, run)
    _require_reason(reason)
    _require_usage_not_decreased(run, usage)
    return (
        task.model_copy(
            update={
                "status": TaskStatus.REQUIRES_ACTION,
                "version": task.version + 1,
            }
        ),
        run.model_copy(
            update={
                "status": RunStatus.REQUIRES_ACTION,
                "iteration": run.iteration + int(advance_iteration),
                "usage": usage,
                "terminal_reason": reason,
                "version": run.version + 1,
            }
        ),
    )


def resume_task_and_run(task: Task, run: Run) -> tuple[Task, Run]:
    if task.status is not TaskStatus.REQUIRES_ACTION or run.status is not RunStatus.REQUIRES_ACTION:
        raise TransitionError("resume requires requires-action task/run")
    return (
        task.model_copy(update={"status": TaskStatus.ACTIVE, "version": task.version + 1}),
        run.model_copy(
            update={
                "status": RunStatus.RUNNING,
                "terminal_reason": None,
                "version": run.version + 1,
            }
        ),
    )


def requeue_task_and_run(task: Task, run: Run) -> tuple[Task, Run]:
    if task.status is not TaskStatus.REQUIRES_ACTION or run.status is not RunStatus.REQUIRES_ACTION:
        raise TransitionError("requeue requires requires-action task/run")
    return (
        task.model_copy(update={"status": TaskStatus.QUEUED, "version": task.version + 1}),
        run.model_copy(
            update={
                "status": RunStatus.QUEUED,
                "started_at": None,
                "terminal_reason": None,
                "version": run.version + 1,
            }
        ),
    )


def cancel_task_and_run(
    task: Task,
    run: Run,
    reason: str,
    *,
    usage: BudgetUsage | None = None,
    advance_iteration: bool = False,
) -> tuple[Task, Run]:
    _require_reason(reason)
    if (task.status, run.status) == (TaskStatus.QUEUED, RunStatus.QUEUED):
        next_usage = run.usage if usage is None else usage
        _require_usage_not_decreased(run, next_usage)
        return (
            task.model_copy(update={"status": TaskStatus.CANCELLED, "version": task.version + 1}),
            run.model_copy(
                update={
                    "status": RunStatus.CANCELLED,
                    "usage": next_usage,
                    "terminal_reason": reason,
                    "version": run.version + 1,
                }
            ),
        )
    if (task.status, run.status) not in {
        (TaskStatus.ACTIVE, RunStatus.RUNNING),
        (TaskStatus.REQUIRES_ACTION, RunStatus.REQUIRES_ACTION),
    }:
        raise TransitionError("cancel requires queued/queued, active/running, or requires-action")
    next_usage = run.usage if usage is None else usage
    _require_usage_not_decreased(run, next_usage)
    return (
        task.model_copy(update={"status": TaskStatus.CANCELLED, "version": task.version + 1}),
        run.model_copy(
            update={
                "status": RunStatus.CANCELLED,
                "iteration": run.iteration + int(advance_iteration),
                "usage": next_usage,
                "terminal_reason": reason,
                "version": run.version + 1,
            }
        ),
    )


def new_retry_run(run: Run, *, run_id: str) -> Run:
    if run.status not in {
        RunStatus.FAILED,
        RunStatus.TIMED_OUT,
        RunStatus.BUDGET_EXHAUSTED,
    }:
        raise TransitionError("only retryable terminal runs can create a new run")
    if run_id == run.id:
        raise TransitionError("retry run must have a distinct ID")
    return Run(
        id=run_id,
        task_id=run.task_id,
        scope=run.scope,
        phase=run.phase,
        limits=run.limits,
    )


def advance_step(
    task: Task,
    run: Run,
    *,
    usage: BudgetUsage,
    observation_ids: tuple[str, ...] | None = None,
    verifier_feedback: tuple[str, ...] | None = None,
    reasoning_step: ReasoningStep | None = None,
    step_plan: tuple[PlannedStep, ...] | None = None,
) -> tuple[Task, Run]:
    _require_active(task, run)
    task_update: dict[str, object] = {"version": task.version + 1}
    if step_plan is not None:
        task_update["step_plan"] = step_plan
    if observation_ids is not None:
        task_update["observation_ids"] = observation_ids
    if verifier_feedback is not None:
        task_update["verifier_feedback"] = verifier_feedback
    if reasoning_step is not None:
        # Bounded: the oldest steps fall off so a long run cannot grow the task
        # row without limit, while the recent trace the model needs stays intact.
        task_update["scratchpad"] = (*task.scratchpad, reasoning_step)[-32:]
    return (
        task.model_copy(update=task_update),
        run.model_copy(
            update={
                "iteration": run.iteration + 1,
                "usage": usage,
                "version": run.version + 1,
            }
        ),
    )


def complete_task_and_run(
    task: Task,
    run: Run,
    completion: VerifiedCompletion,
    *,
    usage: BudgetUsage,
) -> tuple[Task, Run]:
    _require_active(task, run)
    return (
        task.model_copy(
            update={
                "status": TaskStatus.COMPLETED,
                "final_output": completion.answer,
                "version": task.version + 1,
            }
        ),
        run.model_copy(
            update={
                "status": RunStatus.COMPLETED,
                "iteration": run.iteration + 1,
                "usage": usage,
                "final_output": completion.answer,
                "terminal_reason": "verified_completion",
                "version": run.version + 1,
            }
        ),
    )


def fail_task_and_run(
    task: Task,
    run: Run,
    reason: str,
    *,
    usage: BudgetUsage,
    observation_ids: tuple[str, ...] | None = None,
) -> tuple[Task, Run]:
    _require_active(task, run)
    return (
        task.model_copy(
            update={
                "status": TaskStatus.FAILED,
                "observation_ids": (
                    task.observation_ids if observation_ids is None else observation_ids
                ),
                "version": task.version + 1,
            }
        ),
        run.model_copy(
            update={
                "status": RunStatus.FAILED,
                "iteration": run.iteration + 1,
                "usage": usage,
                "terminal_reason": reason,
                "version": run.version + 1,
            }
        ),
    )


def exhaust_task_and_run(
    task: Task,
    run: Run,
    reason: str,
    *,
    usage: BudgetUsage,
    advance_iteration: bool = False,
) -> tuple[Task, Run]:
    _require_active(task, run)
    return (
        task.model_copy(update={"status": TaskStatus.FAILED, "version": task.version + 1}),
        run.model_copy(
            update={
                "status": RunStatus.BUDGET_EXHAUSTED,
                "iteration": run.iteration + int(advance_iteration),
                "usage": usage,
                "terminal_reason": reason,
                "version": run.version + 1,
            }
        ),
    )


def time_out_task_and_run(
    task: Task,
    run: Run,
    reason: str,
    *,
    usage: BudgetUsage,
    observation_ids: tuple[str, ...] | None = None,
    advance_iteration: bool = False,
) -> tuple[Task, Run]:
    _require_active(task, run)
    return (
        task.model_copy(
            update={
                "status": TaskStatus.FAILED,
                "observation_ids": (
                    task.observation_ids if observation_ids is None else observation_ids
                ),
                "version": task.version + 1,
            }
        ),
        run.model_copy(
            update={
                "status": RunStatus.TIMED_OUT,
                "iteration": run.iteration + int(advance_iteration),
                "usage": usage,
                "terminal_reason": reason,
                "version": run.version + 1,
            }
        ),
    )


def _require_active(task: Task, run: Run) -> None:
    if task.status is not TaskStatus.ACTIVE or run.status is not RunStatus.RUNNING:
        raise TransitionError("task and run must be active/running")


def _require_reason(reason: str) -> None:
    if not reason.strip():
        raise TransitionError("transition reason must be non-empty")


def _require_usage_not_decreased(run: Run, usage: BudgetUsage) -> None:
    if usage.model_calls < run.usage.model_calls or usage.tool_calls < run.usage.tool_calls:
        raise TransitionError("run usage cannot decrease")
