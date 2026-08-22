"""Postgres-backed Task claims for Leo's local durable worker queue."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from leo.harness.ports import IdGenerator
from leo.persistence.schema import TaskRow


@dataclass(frozen=True, slots=True)
class TaskLease:
    task_id: str
    owner: str
    token: str
    attempt: int
    expires_at: datetime


class TaskLeaseConflictError(RuntimeError):
    """A heartbeat/release was attempted without the current opaque lease."""


class PostgresTaskLeaseStore:
    """Claim queued Task rows without making process-local queues authoritative."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._ids = ids

    async def claim_next(
        self,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> TaskLease | None:
        _validate_owner(owner)
        _validate_lease_seconds(lease_seconds)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        token = self._ids.new("lease")
        current_time = now if now is not None else func.now()
        eligible = _eligible_tasks(current_time, max_attempts)
        candidate = (
            select(TaskRow.id)
            .where(eligible)
            .order_by(TaskRow.created_at, TaskRow.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        expires_at = current_time + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds)
        statement = (
            update(TaskRow)
            .where(TaskRow.id == candidate)
            .values(
                lease_owner=owner,
                lease_token=token,
                lease_expires_at=expires_at,
                heartbeat_at=current_time,
                attempt_count=TaskRow.attempt_count + 1,
                last_error=None,
            )
            .returning(
                TaskRow.id,
                TaskRow.attempt_count,
                TaskRow.lease_expires_at,
            )
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).one_or_none()
        if row is None:
            return None
        if row.lease_expires_at is None:
            raise TaskLeaseConflictError("claimed task has no lease expiry")
        return TaskLease(
            task_id=row.id,
            owner=owner,
            token=token,
            attempt=row.attempt_count,
            expires_at=row.lease_expires_at,
        )

    async def claim_task(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> TaskLease | None:
        """Claim one durable wake-up by identity, preserving SKIP-LOCKED semantics."""

        if not task_id:
            raise ValueError("task_id must be non-empty")
        _validate_owner(owner)
        _validate_lease_seconds(lease_seconds)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        token = self._ids.new("lease")
        current_time = now if now is not None else func.now()
        eligible = _eligible_tasks(current_time, max_attempts)
        expires_at = current_time + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds)
        statement = (
            update(TaskRow)
            .where(TaskRow.id == task_id, eligible)
            .values(
                lease_owner=owner,
                lease_token=token,
                lease_expires_at=expires_at,
                heartbeat_at=current_time,
                attempt_count=TaskRow.attempt_count + 1,
                last_error=None,
            )
            .returning(TaskRow.id, TaskRow.attempt_count, TaskRow.lease_expires_at)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).one_or_none()
        if row is None:
            return None
        if row.lease_expires_at is None:
            raise TaskLeaseConflictError("claimed task has no lease expiry")
        return TaskLease(
            task_id=row.id,
            owner=owner,
            token=token,
            attempt=row.attempt_count,
            expires_at=row.lease_expires_at,
        )

    async def claim_exhausted(
        self,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> TaskLease | None:
        """Fence one expired/unleased Task that can no longer be retried.

        Exhausted work is claimed without incrementing its attempt counter.  The
        caller must use the returned fence only to persist a safe terminal result;
        it must never execute the user request again.
        """

        return await self._claim_exhausted(
            None,
            owner,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            now=now,
        )

    async def claim_exhausted_task(
        self,
        task_id: str,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> TaskLease | None:
        """Fence one exhausted Task by identity for terminal reconciliation."""

        if not task_id:
            raise ValueError("task_id must be non-empty")
        return await self._claim_exhausted(
            task_id,
            owner,
            lease_seconds=lease_seconds,
            max_attempts=max_attempts,
            now=now,
        )

    async def _claim_exhausted(
        self,
        task_id: str | None,
        owner: str,
        *,
        lease_seconds: float,
        max_attempts: int,
        now: datetime | None,
    ) -> TaskLease | None:
        _validate_owner(owner)
        _validate_lease_seconds(lease_seconds)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        token = self._ids.new("lease")
        current_time = now if now is not None else func.now()
        exhausted = _exhausted_tasks(current_time, max_attempts)
        candidate_query = select(TaskRow.id).where(exhausted)
        if task_id is not None:
            candidate_query = candidate_query.where(TaskRow.id == task_id)
        candidate = (
            candidate_query.order_by(TaskRow.created_at, TaskRow.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        expires_at = current_time + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds)
        statement = (
            update(TaskRow)
            .where(TaskRow.id == candidate)
            .values(
                lease_owner=owner,
                lease_token=token,
                lease_expires_at=expires_at,
                heartbeat_at=current_time,
                retry_after=None,
                last_error="retry_attempts_exhausted",
            )
            .returning(TaskRow.id, TaskRow.attempt_count, TaskRow.lease_expires_at)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).one_or_none()
        if row is None:
            return None
        if row.lease_expires_at is None:
            raise TaskLeaseConflictError("claimed exhausted task has no lease expiry")
        return TaskLease(
            task_id=row.id,
            owner=owner,
            token=token,
            attempt=row.attempt_count,
            expires_at=row.lease_expires_at,
        )

    async def heartbeat(
        self,
        lease: TaskLease,
        *,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> TaskLease:
        _validate_lease_seconds(lease_seconds)
        current_time = now if now is not None else func.now()
        statement = (
            update(TaskRow)
            .where(
                TaskRow.id == lease.task_id,
                TaskRow.status.in_(("queued", "active", "requires_action")),
                TaskRow.lease_owner == lease.owner,
                TaskRow.lease_token == lease.token,
                TaskRow.lease_expires_at > current_time,
            )
            .values(
                lease_expires_at=current_time + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds),
                heartbeat_at=current_time,
            )
            .returning(TaskRow.id, TaskRow.attempt_count, TaskRow.lease_expires_at)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).one_or_none()
        if row is None or row.lease_expires_at is None:
            raise TaskLeaseConflictError("task lease is stale or owned by another worker")
        return TaskLease(
            task_id=row.id,
            owner=lease.owner,
            token=lease.token,
            attempt=row.attempt_count,
            expires_at=row.lease_expires_at,
        )

    async def release(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
    ) -> None:
        if safe_error is not None and (not safe_error or len(safe_error) > 255):
            raise ValueError("safe_error must be 1-255 characters when provided")
        statement = (
            update(TaskRow)
            .where(
                TaskRow.id == lease.task_id,
                TaskRow.lease_owner == lease.owner,
                TaskRow.lease_token == lease.token,
            )
            .values(
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                heartbeat_at=None,
                retry_after=retry_after,
                last_error=safe_error,
            )
            .returning(TaskRow.id)
        )
        async with self._sessions() as session, session.begin():
            if (await session.execute(statement)).scalar_one_or_none() is None:
                raise TaskLeaseConflictError("task lease is stale or owned by another worker")

    async def abandon(
        self,
        lease: TaskLease,
        *,
        retry_after: datetime | None = None,
        safe_error: str | None = None,
        now: datetime | None = None,
    ) -> None:
        """Make a current lease immediately reclaimable without dropping its fence."""

        if safe_error is not None and (not safe_error or len(safe_error) > 255):
            raise ValueError("safe_error must be 1-255 characters when provided")
        current_time = now if now is not None else func.now()
        statement = (
            update(TaskRow)
            .where(
                TaskRow.id == lease.task_id,
                TaskRow.lease_owner == lease.owner,
                TaskRow.lease_token == lease.token,
            )
            .values(
                lease_expires_at=current_time,
                heartbeat_at=current_time,
                retry_after=retry_after,
                last_error=safe_error,
            )
            .returning(TaskRow.id)
        )
        async with self._sessions() as session, session.begin():
            if (await session.execute(statement)).scalar_one_or_none() is None:
                raise TaskLeaseConflictError("task lease is stale or owned by another worker")


def _eligible_tasks(current_time: object, max_attempts: int) -> ColumnElement[bool]:
    return and_(
        TaskRow.status.in_(("queued", "active")),
        TaskRow.attempt_count < max_attempts,
        or_(TaskRow.retry_after.is_(None), TaskRow.retry_after <= current_time),
        or_(
            TaskRow.lease_expires_at.is_(None),
            TaskRow.lease_expires_at <= current_time,
        ),
    )


def _exhausted_tasks(current_time: object, max_attempts: int) -> ColumnElement[bool]:
    return and_(
        TaskRow.status.in_(("queued", "active")),
        TaskRow.attempt_count >= max_attempts,
        or_(
            TaskRow.lease_expires_at.is_(None),
            TaskRow.lease_expires_at <= current_time,
        ),
    )


def _validate_owner(owner: str) -> None:
    if not owner or owner != owner.strip() or len(owner) > 128:
        raise ValueError("owner must be a non-empty value of at most 128 characters")


def _validate_lease_seconds(value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > 86_400:
        raise ValueError("lease_seconds must be finite and between 0 and 86400")
