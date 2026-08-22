"""Intention-specific terminal transitions used by durable worker adapters."""

from __future__ import annotations

from leo.harness.models import EventDraft, EventType, RunBundle, RunStatus, ScopeKey, TaskStatus
from leo.harness.ports import Clock, RunStore
from leo.harness.transitions import fail_task_and_run, start_task_and_run

MAX_TASK_ATTEMPTS = 3
RETRY_ATTEMPTS_EXHAUSTED = "retry_attempts_exhausted"


async def persist_safe_failure(
    store: RunStore,
    *,
    task_id: str,
    run_id: str,
    scope: ScopeKey,
    reason: str,
    clock: Clock,
) -> RunBundle:
    """Turn an adapter-level rejection into durable failure before delivery."""

    bundle = await store.load(task_id, run_id, scope)
    if bundle.run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
        RunStatus.BUDGET_EXHAUSTED,
    }:
        return bundle
    if bundle.task.status is TaskStatus.QUEUED and bundle.run.status is RunStatus.QUEUED:
        active_task, active_run = start_task_and_run(
            bundle.task,
            bundle.run,
            started_at=clock.now(),
        )
        bundle = await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=active_task,
            run=active_run,
            events=(EventDraft(type=EventType.TASK_STARTED, iteration=active_run.iteration),),
        )
    if bundle.task.status is not TaskStatus.ACTIVE or bundle.run.status is not RunStatus.RUNNING:
        return bundle
    failed_task, failed_run = fail_task_and_run(
        bundle.task,
        bundle.run,
        reason,
        usage=bundle.run.usage,
    )
    return await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=failed_task,
        run=failed_run,
        events=(
            EventDraft(
                type=EventType.RUN_FAILED,
                iteration=failed_run.iteration,
                payload={"reason": reason},
            ),
        ),
    )
