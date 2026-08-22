from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from leo.persistence.task_leases import TaskLease
from leo.worker.runtime import DurableTaskWorker


class FakeLeaseOperations:
    def __init__(self) -> None:
        self.lease = TaskLease(
            task_id="task-worker",
            owner="worker-test",
            token="lease-token",
            attempt=1,
            expires_at=datetime.now(UTC) + timedelta(seconds=30),
        )
        self.claimed = False
        self.heartbeats = 0
        self.released: list[TaskLease] = []
        self.abandoned: list[tuple[TaskLease, str | None]] = []

    async def claim_next(
        self,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> TaskLease | None:
        del lease_seconds, max_attempts
        if self.claimed:
            return None
        self.claimed = True
        return self.lease.__class__(
            task_id=self.lease.task_id,
            owner=owner,
            token=self.lease.token,
            attempt=self.lease.attempt,
            expires_at=self.lease.expires_at,
        )

    async def heartbeat(self, lease: TaskLease, *, lease_seconds: float = 60.0) -> TaskLease:
        del lease_seconds
        self.heartbeats += 1
        return lease

    async def release(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
    ) -> None:
        del retry_after, safe_error
        self.released.append(lease)

    async def abandon(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
    ) -> None:
        del retry_after
        self.abandoned.append((lease, safe_error))


@pytest.mark.asyncio
async def test_worker_releases_after_success_and_heartbeats() -> None:
    leases = FakeLeaseOperations()
    seen: list[str] = []

    async def handler(lease: TaskLease) -> None:
        seen.append(lease.task_id)
        await asyncio.sleep(0.12)

    worker = DurableTaskWorker(
        leases=leases,
        owner="worker-test",
        handler=handler,
        lease_seconds=0.3,
        idle_wait_seconds=0.1,
    )

    assert await worker.run_once() is True
    assert seen == ["task-worker"]
    assert leases.released and not leases.abandoned
    assert leases.heartbeats >= 1


@pytest.mark.asyncio
async def test_worker_abandons_on_handler_failure() -> None:
    leases = FakeLeaseOperations()

    async def handler(lease: TaskLease) -> None:
        del lease
        raise RuntimeError("provider detail must not become a lease error")

    worker = DurableTaskWorker(
        leases=leases,
        owner="worker-test",
        handler=handler,
        idle_wait_seconds=0.1,
    )

    with pytest.raises(RuntimeError, match="provider detail"):
        await worker.run_once()
    assert leases.released == []
    assert [(lease.task_id, error) for lease, error in leases.abandoned] == [
        ("task-worker", "worker_handler_error")
    ]
