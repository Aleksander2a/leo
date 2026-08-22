"""Durable local worker loop for Postgres-backed Leo Tasks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Protocol

from leo.persistence.task_leases import TaskLease, TaskLeaseConflictError


class TaskLeaseOperations(Protocol):
    async def claim_next(
        self,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
    ) -> TaskLease | None: ...

    async def heartbeat(
        self,
        lease: TaskLease,
        *,
        lease_seconds: float = 60.0,
    ) -> TaskLease: ...

    async def release(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
    ) -> None: ...

    async def abandon(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
    ) -> None: ...


TaskHandler = Callable[[TaskLease], Awaitable[None]]


class LeaseHeartbeat:
    """Keep one claimed Task lease alive while an external coordinator runs."""

    def __init__(
        self,
        leases: TaskLeaseOperations,
        lease: TaskLease,
        lease_seconds: float,
    ) -> None:
        self._leases = leases
        self._lease = lease
        self._lease_seconds = lease_seconds
        self._task: asyncio.Task[None] | None = None

    async def __aenter__(self) -> LeaseHeartbeat:
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._task is None:
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        interval = max(0.1, self._lease_seconds / 3.0)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._leases.heartbeat(
                    self._lease,
                    lease_seconds=self._lease_seconds,
                )
            except TaskLeaseConflictError:
                return


class DurableTaskWorker:
    """Claim, heartbeat, and fence one Task at a time.

    The handler owns terminal state. A successful terminal commit clears the lease;
    a non-terminal return releases it, and an exception expires it for recovery.
    """

    def __init__(
        self,
        *,
        leases: TaskLeaseOperations,
        owner: str,
        handler: TaskHandler,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        idle_wait_seconds: float = 1.0,
    ) -> None:
        if not owner or owner != owner.strip():
            raise ValueError("owner must be a non-empty value")
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if idle_wait_seconds <= 0:
            raise ValueError("idle_wait_seconds must be positive")
        self._leases = leases
        self._owner = owner
        self._handler = handler
        self._lease_seconds = lease_seconds
        self._max_attempts = max_attempts
        self._idle_wait_seconds = idle_wait_seconds

    async def run_once(self) -> bool:
        lease = await self._leases.claim_next(
            self._owner,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
        )
        if lease is None:
            return False

        heartbeat = asyncio.create_task(self._heartbeat_loop(lease))
        try:
            await self._handler(lease)
        except asyncio.CancelledError:
            await self._expire_safely(lease, "worker_stopped")
            raise
        except Exception:
            await self._expire_safely(lease, "worker_handler_error")
            raise
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)

        try:
            await self._leases.release(lease)
        except TaskLeaseConflictError:
            # A terminal coordinator commit clears the lease in the same transaction.
            pass
        return True

    async def run_until_stopped(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            worked = await self.run_once()
            if worked:
                continue
            try:
                await asyncio.wait_for(stop.wait(), timeout=self._idle_wait_seconds)
            except TimeoutError:
                continue

    async def _heartbeat_loop(self, lease: TaskLease) -> None:
        interval = max(0.1, self._lease_seconds / 3.0)
        while True:
            await asyncio.sleep(interval)
            try:
                await self._leases.heartbeat(lease, lease_seconds=self._lease_seconds)
            except TaskLeaseConflictError:
                return

    async def _expire_safely(self, lease: TaskLease, safe_error: str) -> None:
        try:
            await self._leases.abandon(lease, safe_error=safe_error)
        except TaskLeaseConflictError:
            pass
