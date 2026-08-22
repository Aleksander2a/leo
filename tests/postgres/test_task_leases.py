from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import (
    EventDraft,
    EventType,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
)
from leo.harness.ports import IdGenerator, RunStore
from leo.harness.transitions import fail_task_and_run, start_task_and_run
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.task_leases import PostgresTaskLeaseStore, TaskLeaseConflictError
from leo.worker.terminal import RETRY_ATTEMPTS_EXHAUSTED, persist_safe_failure


class _UniqueIds(IdGenerator):
    def __init__(self) -> None:
        self._suffix = uuid4().hex
        self._counter = 0

    def new(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._suffix[:20]}-{self._counter}"


@dataclass(frozen=True)
class _SeededTask:
    task_id: str
    run_id: str
    scope: ScopeKey


@pytest_asyncio.fixture
async def lease_store(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[PostgresTaskLeaseStore, RunStore]]:
    yield (
        PostgresTaskLeaseStore(preserved_postgres_sessions, _UniqueIds()),
        PostgresRunStore(
            preserved_postgres_sessions,
            FixedClock(),
            SequentialIdGenerator(),
        ),
    )


async def _seed_task(store: RunStore, suffix: str) -> _SeededTask:
    scope = ScopeKey(
        organization_id=f"org-lease-{suffix}",
        strategy_id=f"strategy-lease-{suffix}",
    )
    thread = Thread(
        id=f"thread-{suffix}",
        scope=scope,
        origin=OriginRef(provider="test", external_thread_id=f"lease-{suffix}"),
    )
    task = Task(id=f"task-{suffix}", thread_id=thread.id, scope=scope, objective="lease test")
    run = Run(id=f"run-{suffix}", task_id=task.id, scope=scope)
    await store.seed(thread, task, run)
    return _SeededTask(task_id=task.id, run_id=run.id, scope=scope)


@pytest.mark.asyncio
async def test_two_workers_claim_disjoint_tasks(
    lease_store: tuple[PostgresTaskLeaseStore, RunStore],
) -> None:
    leases, run_store = lease_store
    first_task = await _seed_task(run_store, uuid4().hex)
    second_task = await _seed_task(run_store, uuid4().hex)
    now = datetime(2026, 1, 1, tzinfo=UTC)

    first = await leases.claim_task(first_task.task_id, "worker-a", now=now)
    assert await leases.claim_task(first_task.task_id, "worker-b", now=now) is None
    second = await leases.claim_task(second_task.task_id, "worker-b", now=now)

    assert first is not None
    assert second is not None
    assert {first.task_id, second.task_id} == {first_task.task_id, second_task.task_id}
    assert first.attempt == second.attempt == 1
    assert first.token != second.token


@pytest.mark.asyncio
async def test_heartbeat_and_stale_owner_are_cas_protected(
    lease_store: tuple[PostgresTaskLeaseStore, RunStore],
) -> None:
    leases, run_store = lease_store
    seeded = await _seed_task(run_store, uuid4().hex)
    now = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = await leases.claim_task(seeded.task_id, "worker-a", lease_seconds=30, now=now)
    assert claimed is not None

    renewed = await leases.heartbeat(claimed, lease_seconds=30, now=now + timedelta(seconds=5))
    assert renewed.expires_at > claimed.expires_at

    forged = claimed.__class__(
        task_id=claimed.task_id,
        owner="worker-b",
        token=claimed.token,
        attempt=claimed.attempt,
        expires_at=claimed.expires_at,
    )
    with pytest.raises(TaskLeaseConflictError):
        await leases.heartbeat(forged, now=now + timedelta(seconds=6))


@pytest.mark.asyncio
async def test_expired_lease_is_reclaimed_and_retry_after_is_respected(
    lease_store: tuple[PostgresTaskLeaseStore, RunStore],
) -> None:
    leases, run_store = lease_store
    seeded = await _seed_task(run_store, uuid4().hex)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = await leases.claim_task(seeded.task_id, "worker-a", lease_seconds=10, now=start)
    assert claimed is not None

    await leases.release(
        claimed,
        retry_after=start + timedelta(seconds=20),
        safe_error="temporary_failure",
    )
    assert (
        await leases.claim_task(seeded.task_id, "worker-b", now=start + timedelta(seconds=19))
        is None
    )
    reclaimed = await leases.claim_task(
        seeded.task_id, "worker-b", now=start + timedelta(seconds=20)
    )
    assert reclaimed is not None
    assert reclaimed.task_id == claimed.task_id
    assert reclaimed.attempt == 2

    await leases.release(reclaimed)


@pytest.mark.asyncio
async def test_retry_exhaustion_is_bounded_by_max_attempts(
    lease_store: tuple[PostgresTaskLeaseStore, RunStore],
) -> None:
    leases, run_store = lease_store
    seeded = await _seed_task(run_store, uuid4().hex)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    claimed = await leases.claim_task(seeded.task_id, "worker-a", max_attempts=1, now=start)
    assert claimed is not None
    await leases.release(
        claimed,
        retry_after=start + timedelta(seconds=1),
        safe_error="bounded_failure",
    )
    assert (
        await leases.claim_task(
            seeded.task_id,
            "worker-b",
            max_attempts=1,
            now=start + timedelta(seconds=1),
        )
        is None
    )


@pytest.mark.asyncio
async def test_expired_final_attempt_is_fenced_and_terminalized_without_reexecution(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    clock = FixedClock()
    ids = _UniqueIds()
    run_store = PostgresRunStore(preserved_postgres_sessions, clock, ids)
    leases = PostgresTaskLeaseStore(preserved_postgres_sessions, ids)
    seeded = await _seed_task(run_store, uuid4().hex)
    started_at = clock.now()

    final_attempt = await leases.claim_task(
        seeded.task_id,
        "worker-crashed",
        lease_seconds=10,
        max_attempts=1,
        now=started_at,
    )
    assert final_attempt is not None
    assert final_attempt.attempt == 1
    assert (
        await leases.claim_exhausted_task(
            seeded.task_id,
            "reconciler-early",
            max_attempts=1,
            now=started_at + timedelta(seconds=9),
        )
        is None
    )

    recovery_lease = await leases.claim_exhausted_task(
        seeded.task_id,
        "reconciler",
        max_attempts=1,
        now=started_at + timedelta(seconds=10),
    )
    assert recovery_lease is not None
    assert recovery_lease.attempt == 1
    bundle = await persist_safe_failure(
        LeaseBoundRunStore(run_store, recovery_lease),
        task_id=seeded.task_id,
        run_id=seeded.run_id,
        scope=seeded.scope,
        reason=RETRY_ATTEMPTS_EXHAUSTED,
        clock=clock,
    )

    assert bundle.task.status is TaskStatus.FAILED
    assert bundle.run.status is RunStatus.FAILED
    assert bundle.run.terminal_reason == RETRY_ATTEMPTS_EXHAUSTED
    assert bundle.events[-1].type is EventType.RUN_FAILED
    assert (
        await leases.claim_exhausted_task(
            seeded.task_id,
            "reconciler-again",
            max_attempts=1,
            now=started_at + timedelta(seconds=11),
        )
        is None
    )


@pytest.mark.asyncio
async def test_startup_scan_fences_released_exhausted_task_once(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    clock = FixedClock()
    ids = _UniqueIds()
    run_store = PostgresRunStore(preserved_postgres_sessions, clock, ids)
    leases = PostgresTaskLeaseStore(preserved_postgres_sessions, ids)
    seeded = await _seed_task(run_store, uuid4().hex)
    started_at = clock.now()
    final_attempt = await leases.claim_task(
        seeded.task_id,
        "worker-failed",
        max_attempts=1,
        now=started_at,
    )
    assert final_attempt is not None
    await leases.release(final_attempt, safe_error="bounded_failure")

    recovery_lease = await leases.claim_exhausted_task(
        seeded.task_id,
        "startup-reconciler",
        max_attempts=1,
        now=started_at,
    )
    assert recovery_lease is not None
    assert recovery_lease.task_id == seeded.task_id
    bundle = await persist_safe_failure(
        LeaseBoundRunStore(run_store, recovery_lease),
        task_id=seeded.task_id,
        run_id=seeded.run_id,
        scope=seeded.scope,
        reason=RETRY_ATTEMPTS_EXHAUSTED,
        clock=clock,
    )

    assert bundle.run.status is RunStatus.FAILED
    assert (
        await leases.claim_exhausted_task(
            seeded.task_id,
            "startup-reconciler-again",
            max_attempts=1,
            now=started_at,
        )
        is None
    )


@pytest.mark.asyncio
async def test_expired_active_task_can_resume_and_terminal_commit_clears_lease(
    lease_store: tuple[PostgresTaskLeaseStore, RunStore],
) -> None:
    leases, run_store = lease_store
    seeded = await _seed_task(run_store, uuid4().hex)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = await leases.claim_task(seeded.task_id, "worker-a", lease_seconds=10, now=start)
    assert first is not None
    bundle = await run_store.load(seeded.task_id, seeded.run_id, seeded.scope)
    active_task, active_run = start_task_and_run(bundle.task, bundle.run, started_at=start)
    bound = LeaseBoundRunStore(run_store, first)
    bundle = await bound.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=active_task,
        run=active_run,
        events=(EventDraft(type=EventType.TASK_STARTED, iteration=0),),
    )
    renewed = await leases.heartbeat(first, lease_seconds=10, now=start + timedelta(seconds=5))
    assert renewed.expires_at > first.expires_at

    assert (
        await leases.claim_task(
            seeded.task_id,
            "worker-b",
            lease_seconds=10,
            now=start + timedelta(seconds=10),
        )
        is None
    )
    reclaimed = await leases.claim_task(
        seeded.task_id,
        "worker-b",
        lease_seconds=10,
        now=start + timedelta(seconds=15),
    )
    assert reclaimed is not None
    assert reclaimed.task_id == seeded.task_id
    assert reclaimed.attempt == 2

    failed_task, failed_run = fail_task_and_run(
        bundle.task,
        bundle.run,
        "worker_test_failure",
        usage=bundle.run.usage,
    )
    rebound = LeaseBoundRunStore(run_store, reclaimed)
    await rebound.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=failed_task,
        run=failed_run,
        events=(
            EventDraft(
                type=EventType.RUN_FAILED,
                iteration=failed_run.iteration,
                payload={"reason": "worker_test_failure", "detail": "safe test failure"},
            ),
        ),
    )
    assert (
        await leases.claim_task(seeded.task_id, "worker-c", now=start + timedelta(seconds=16))
        is None
    )
