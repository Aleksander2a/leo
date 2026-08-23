"""Durable, idempotent Slack delivery intents and their dispatcher boundary."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Protocol

from slack_sdk.errors import SlackApiError
from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from leo.harness.models import ScopeKey
from leo.harness.ports import IdGenerator
from leo.persistence.conversation_plane import (
    ConversationMessageRole,
    build_conversation_plane_message,
    persist_conversation_plane_message,
)
from leo.persistence.schema import (
    DeliveryOutboxRow,
    RunRow,
    SlackIngressEventRow,
    TaskRow,
)


class DeliveryKind(StrEnum):
    PROGRESS = "progress"
    FINAL = "final"


class DeliveryState(StrEnum):
    PENDING = "pending"
    LEASED = "leased"
    RETRY = "retry"
    DELIVERED = "delivered"
    DEAD = "dead"
    UNKNOWN_EFFECT = "unknown_effect"


@dataclass(frozen=True, slots=True)
class DeliveryIntent:
    id: str
    task_id: str
    run_id: str
    ingress_event_id: str
    organization_id: str
    strategy_id: str
    destination_channel_id: str
    destination_thread_ts: str
    kind: DeliveryKind
    payload_version: int
    payload_hash: str
    payload: str
    state: DeliveryState
    attempt_count: int
    receipt_message_ts: str | None
    lease_owner: str | None
    lease_token: str | None
    lease_expires_at: datetime | None
    retry_after: datetime | None
    last_error: str | None


@dataclass(frozen=True, slots=True)
class DeliveryLease:
    intent_id: str
    owner: str
    token: str
    attempt: int
    expires_at: datetime


class DeliveryOutboxInvariantError(RuntimeError):
    """A delivery intent did not match its trusted durable source."""


class DeliveryPayloadDriftError(RuntimeError):
    """An idempotency key was reused with different immutable payload data."""


class DeliveryLeaseConflictError(RuntimeError):
    """A delivery result was recorded without the current opaque lease."""


class _MissingSlackThreadRoot(RuntimeError):
    """The Slack reply destination no longer exists."""


class SlackPostClient(Protocol):
    async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object: ...

    async def conversations_replies(self, *, channel: str, ts: str, limit: int = 1) -> object: ...


class PostgresDeliveryOutbox:
    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._ids = ids

    async def ensure_intent(
        self,
        *,
        task_id: str,
        run_id: str,
        ingress_event_id: str,
        kind: DeliveryKind,
        payload_version: int,
        payload: str,
    ) -> DeliveryIntent:
        _validate_payload(payload, payload_version)
        payload_hash = _payload_hash(payload)
        intent_id = self._ids.new("delivery")
        async with self._sessions() as session, session.begin():
            task = await session.scalar(select(TaskRow).where(TaskRow.id == task_id))
            run = await session.scalar(
                select(RunRow).where(RunRow.id == run_id, RunRow.task_id == task_id)
            )
            ingress = await session.scalar(
                select(SlackIngressEventRow).where(
                    SlackIngressEventRow.event_id == ingress_event_id
                )
            )
            if task is None or run is None or ingress is None:
                raise DeliveryOutboxInvariantError("delivery source is missing")
            if (
                ingress.task_id != task_id
                or run.organization_id != task.organization_id
                or run.strategy_id != task.strategy_id
                or ingress.organization_id != task.organization_id
                or ingress.strategy_id != task.strategy_id
                or not ingress.channel_id
                or not ingress.thread_root_ts
            ):
                raise DeliveryOutboxInvariantError("delivery source scope or identity mismatch")
            if kind is DeliveryKind.FINAL and run.status not in {
                "completed",
                "failed",
                "cancelled",
                "timed_out",
                "budget_exhausted",
                "requires_action",
            }:
                raise DeliveryOutboxInvariantError("final delivery requires a durable terminal run")

            statement = (
                postgres_insert(DeliveryOutboxRow)
                .values(
                    id=intent_id,
                    task_id=task_id,
                    run_id=run_id,
                    ingress_event_id=ingress_event_id,
                    organization_id=task.organization_id,
                    strategy_id=task.strategy_id,
                    destination_channel_id=ingress.channel_id,
                    destination_thread_ts=ingress.thread_root_ts,
                    kind=kind.value,
                    payload_version=payload_version,
                    payload_hash=payload_hash,
                    payload=payload,
                    state=DeliveryState.PENDING.value,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        DeliveryOutboxRow.task_id,
                        DeliveryOutboxRow.run_id,
                        DeliveryOutboxRow.kind,
                        DeliveryOutboxRow.payload_version,
                    ]
                )
            )
            await session.execute(statement)
            existing = await session.scalar(
                select(DeliveryOutboxRow).where(
                    DeliveryOutboxRow.task_id == task_id,
                    DeliveryOutboxRow.run_id == run_id,
                    DeliveryOutboxRow.kind == kind.value,
                    DeliveryOutboxRow.payload_version == payload_version,
                )
            )
            if existing is None:
                raise DeliveryOutboxInvariantError("delivery intent disappeared after insert")
            _assert_immutable_match(
                existing,
                destination_channel_id=ingress.channel_id,
                destination_thread_ts=ingress.thread_root_ts,
                payload_hash=payload_hash,
                payload=payload,
            )
            return _intent_model(existing)

    async def reconcile_terminal(
        self,
        payload_factory: Callable[[TaskRow, RunRow], str],
        *,
        limit: int = 100,
        payload_version: int = 1000,
        task_id: str | None = None,
        run_id: str | None = None,
        ingress_event_id: str | None = None,
    ) -> tuple[DeliveryIntent, ...]:
        """Repair terminal runs that committed before their final intent."""

        if limit < 1:
            raise ValueError("limit must be positive")
        for value, field in (
            (task_id, "task_id"),
            (run_id, "run_id"),
            (ingress_event_id, "ingress_event_id"),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"{field} must be non-empty when supplied")
        # A renderer upgrade must never re-deliver an already materialized terminal
        # result.  Payload version identifies immutable parts, but it is not a reason
        # to create a second terminal delivery for the same durable ingress/run.
        missing_intent = (
            ~select(DeliveryOutboxRow.id)
            .where(
                DeliveryOutboxRow.task_id == TaskRow.id,
                DeliveryOutboxRow.run_id == RunRow.id,
                DeliveryOutboxRow.ingress_event_id == SlackIngressEventRow.event_id,
                DeliveryOutboxRow.kind == DeliveryKind.FINAL.value,
            )
            .exists()
        )
        source_filters = [
            RunRow.status.in_(
                (
                    "completed",
                    "failed",
                    "cancelled",
                    "timed_out",
                    "budget_exhausted",
                    "requires_action",
                )
            ),
            missing_intent,
        ]
        if task_id is not None:
            source_filters.append(TaskRow.id == task_id)
        if run_id is not None:
            source_filters.append(RunRow.id == run_id)
        if ingress_event_id is not None:
            source_filters.append(SlackIngressEventRow.event_id == ingress_event_id)
        async with self._sessions() as session:
            rows = (
                await session.execute(
                    select(TaskRow, RunRow, SlackIngressEventRow)
                    .join(RunRow, RunRow.task_id == TaskRow.id)
                    .join(SlackIngressEventRow, SlackIngressEventRow.task_id == TaskRow.id)
                    .where(*source_filters)
                    .order_by(RunRow.updated_at, RunRow.id)
                    .limit(limit)
                )
            ).all()
        intents: list[DeliveryIntent] = []
        for task, run, ingress in rows:
            intents.append(
                await self.ensure_intent(
                    task_id=task.id,
                    run_id=run.id,
                    ingress_event_id=ingress.event_id,
                    kind=DeliveryKind.FINAL,
                    payload_version=payload_version,
                    payload=payload_factory(task, run),
                )
            )
        return tuple(intents)

    async def claim_next(
        self,
        owner: str,
        *,
        lease_seconds: float = 60.0,
        max_attempts: int = 5,
        now: datetime | None = None,
        intent_id: str | None = None,
    ) -> tuple[DeliveryLease, DeliveryIntent] | None:
        _validate_owner(owner)
        _validate_duration(lease_seconds)
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        current_time = now if now is not None else func.now()
        token = self._ids.new("delivery-lease")
        eligible = _eligible(current_time, max_attempts)
        candidate_filters = [eligible]
        if intent_id is not None:
            candidate_filters.append(DeliveryOutboxRow.id == intent_id)
        candidate = (
            select(DeliveryOutboxRow.id)
            .where(*candidate_filters)
            .order_by(DeliveryOutboxRow.created_at, DeliveryOutboxRow.id)
            .limit(1)
            .with_for_update(skip_locked=True)
            .scalar_subquery()
        )
        expires_at = current_time + func.make_interval(0, 0, 0, 0, 0, 0, lease_seconds)
        statement = (
            update(DeliveryOutboxRow)
            .where(DeliveryOutboxRow.id == candidate)
            .values(
                state=DeliveryState.LEASED.value,
                lease_owner=owner,
                lease_token=token,
                lease_expires_at=expires_at,
                attempt_count=DeliveryOutboxRow.attempt_count + 1,
                last_error=None,
            )
            .returning(DeliveryOutboxRow)
        )
        async with self._sessions() as session, session.begin():
            row = (await session.execute(statement)).scalar_one_or_none()
        if row is None or row.lease_expires_at is None:
            return None
        intent = _intent_model(row)
        return (
            DeliveryLease(
                intent_id=row.id,
                owner=owner,
                token=token,
                attempt=row.attempt_count,
                expires_at=row.lease_expires_at,
            ),
            intent,
        )

    async def load(self, intent_id: str) -> DeliveryIntent:
        if not intent_id:
            raise ValueError("intent_id must be non-empty")
        async with self._sessions() as session:
            row = await session.scalar(
                select(DeliveryOutboxRow).where(DeliveryOutboxRow.id == intent_id)
            )
        if row is None:
            raise DeliveryOutboxInvariantError("delivery intent not found")
        return _intent_model(row)

    async def mark_delivered(self, lease: DeliveryLease, message_ts: str) -> None:
        if not message_ts or len(message_ts) > 64:
            raise ValueError("message_ts must be a non-empty value of at most 64 characters")
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == lease.intent_id,
                DeliveryOutboxRow.state == DeliveryState.LEASED.value,
                DeliveryOutboxRow.lease_owner == lease.owner,
                DeliveryOutboxRow.lease_token == lease.token,
            )
            .values(
                state=DeliveryState.DELIVERED.value,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                receipt_message_ts=message_ts,
                retry_after=None,
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery lease is stale")

    async def mark_dispatch_started(self, lease: DeliveryLease) -> None:
        """Fence the irreducible Slack-call ambiguity before invoking the API.

        A process death before this transaction commits leaves the expired ``leased``
        intent retryable because no call began.  A death after it commits leaves an
        ``unknown_effect`` intent that is never blindly reclaimed, even when Slack
        accepted the message but its receipt never reached this process.
        """

        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == lease.intent_id,
                DeliveryOutboxRow.state == DeliveryState.LEASED.value,
                DeliveryOutboxRow.lease_owner == lease.owner,
                DeliveryOutboxRow.lease_token == lease.token,
            )
            .values(
                state=DeliveryState.UNKNOWN_EFFECT.value,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                retry_after=None,
                last_error="dispatch_started_receipt_pending",
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery lease is stale")

    async def confirm_dispatched(self, intent_id: str, message_ts: str) -> None:
        """Atomically record a Slack receipt and its sanitized assistant turn."""

        if not intent_id:
            raise ValueError("intent_id must be non-empty")
        if not message_ts or len(message_ts) > 64:
            raise ValueError("message_ts must be a non-empty value of at most 64 characters")
        async with self._sessions() as session, session.begin():
            intent = await session.scalar(
                select(DeliveryOutboxRow).where(DeliveryOutboxRow.id == intent_id).with_for_update()
            )
            if (
                intent is None
                or intent.state != DeliveryState.UNKNOWN_EFFECT.value
                or intent.receipt_message_ts is not None
            ):
                raise DeliveryLeaseConflictError("delivery dispatch is no longer receipt-pending")
            ingress = await session.scalar(
                select(SlackIngressEventRow).where(
                    SlackIngressEventRow.event_id == intent.ingress_event_id
                )
            )
            task = await session.scalar(select(TaskRow).where(TaskRow.id == intent.task_id))
            if (
                ingress is None
                or task is None
                or ingress.conversation_id is None
                or ingress.organization_id != intent.organization_id
                or ingress.strategy_id != intent.strategy_id
                or ingress.channel_id != intent.destination_channel_id
                or ingress.thread_root_ts != intent.destination_thread_ts
                or task.organization_id != intent.organization_id
                or task.strategy_id != intent.strategy_id
            ):
                raise DeliveryOutboxInvariantError(
                    "delivered assistant message authority is missing or mismatched"
                )
            intent.state = DeliveryState.DELIVERED.value
            intent.receipt_message_ts = message_ts
            intent.retry_after = None
            intent.last_error = None
            await persist_conversation_plane_message(
                session,
                build_conversation_plane_message(
                    scope=ScopeKey(
                        organization_id=intent.organization_id,
                        strategy_id=intent.strategy_id,
                    ),
                    conversation_id=ingress.conversation_id,
                    harness_thread_id=task.thread_id,
                    destination_id=intent.destination_channel_id,
                    external_event_id=f"slack-delivery:{intent.id}",
                    actor_id="leo",
                    role=ConversationMessageRole.ASSISTANT,
                    provider_message_ts=message_ts,
                    provider_thread_root_ts=intent.destination_thread_ts,
                    context_access_hash=ingress.context_access_hash,
                    text=intent.payload,
                    recorded_at=datetime.now(UTC),
                ),
            )

    async def reject_dispatched(
        self,
        intent_id: str,
        *,
        retry_after: datetime | None,
        safe_error: str,
        dead: bool,
    ) -> None:
        """Resolve an explicit Slack rejection, which proves the call had no effect."""

        if not intent_id:
            raise ValueError("intent_id must be non-empty")
        _validate_safe_error(safe_error)
        state = DeliveryState.DEAD if dead else DeliveryState.RETRY
        if not dead and retry_after is None:
            raise ValueError("retry_after is required for a retryable rejection")
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == intent_id,
                DeliveryOutboxRow.state == DeliveryState.UNKNOWN_EFFECT.value,
                DeliveryOutboxRow.receipt_message_ts.is_(None),
            )
            .values(
                state=state.value,
                retry_after=None if dead else retry_after,
                last_error=safe_error,
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery dispatch is no longer receipt-pending")

    async def record_unknown_effect(self, intent_id: str, safe_error: str) -> None:
        """Refine an already fenced ambiguous call with an operator-safe reason."""

        if not intent_id:
            raise ValueError("intent_id must be non-empty")
        _validate_safe_error(safe_error)
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == intent_id,
                DeliveryOutboxRow.state == DeliveryState.UNKNOWN_EFFECT.value,
                DeliveryOutboxRow.receipt_message_ts.is_(None),
            )
            .values(last_error=safe_error)
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery dispatch is no longer receipt-pending")

    async def mark_retry(
        self,
        lease: DeliveryLease,
        *,
        retry_after: datetime,
        safe_error: str,
    ) -> None:
        _validate_safe_error(safe_error)
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == lease.intent_id,
                DeliveryOutboxRow.state == DeliveryState.LEASED.value,
                DeliveryOutboxRow.lease_owner == lease.owner,
                DeliveryOutboxRow.lease_token == lease.token,
            )
            .values(
                state=DeliveryState.RETRY.value,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                retry_after=retry_after,
                last_error=safe_error,
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery lease is stale")

    async def mark_dead(self, lease: DeliveryLease, safe_error: str) -> None:
        await self._mark_terminal(lease, DeliveryState.DEAD, safe_error)

    async def mark_unknown_effect(self, lease: DeliveryLease) -> None:
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == lease.intent_id,
                DeliveryOutboxRow.state == DeliveryState.LEASED.value,
                DeliveryOutboxRow.lease_owner == lease.owner,
                DeliveryOutboxRow.lease_token == lease.token,
            )
            .values(
                state=DeliveryState.UNKNOWN_EFFECT.value,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                retry_after=None,
                last_error="success_before_receipt_unknown_effect",
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery lease is stale")

    async def _mark_terminal(
        self,
        lease: DeliveryLease,
        state: DeliveryState,
        safe_error: str,
    ) -> None:
        _validate_safe_error(safe_error)
        statement = (
            update(DeliveryOutboxRow)
            .where(
                DeliveryOutboxRow.id == lease.intent_id,
                DeliveryOutboxRow.state == DeliveryState.LEASED.value,
                DeliveryOutboxRow.lease_owner == lease.owner,
                DeliveryOutboxRow.lease_token == lease.token,
            )
            .values(
                state=state.value,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                retry_after=None,
                last_error=safe_error,
            )
            .returning(DeliveryOutboxRow.id)
        )
        await self._cas_update(statement, "delivery lease is stale")

    async def _cas_update(self, statement: Any, message: str) -> None:
        async with self._sessions() as session, session.begin():
            if (await session.execute(statement)).scalar_one_or_none() is None:
                raise DeliveryLeaseConflictError(message)


class SlackOutboxDispatcher:
    def __init__(
        self,
        outbox: PostgresDeliveryOutbox,
        *,
        owner: str,
        max_attempts: int = 5,
        lease_seconds: float = 60.0,
    ) -> None:
        _validate_owner(owner)
        self._outbox = outbox
        self._owner = owner
        self._max_attempts = max_attempts
        self._lease_seconds = lease_seconds

    async def dispatch_once(
        self,
        client: SlackPostClient,
        *,
        now: datetime | None = None,
        intent_id: str | None = None,
    ) -> DeliveryState | None:
        claimed = await self._outbox.claim_next(
            self._owner,
            lease_seconds=self._lease_seconds,
            max_attempts=self._max_attempts,
            now=now,
            intent_id=intent_id,
        )
        if claimed is None:
            return None
        lease, intent = claimed
        await self._outbox.mark_dispatch_started(lease)
        try:
            await self._verify_thread_root(client, intent)
        except _MissingSlackThreadRoot:
            await self._outbox.reject_dispatched(
                intent.id,
                retry_after=None,
                safe_error="slack_thread_root_missing",
                dead=True,
            )
            return DeliveryState.DEAD
        except Exception:
            # A failed read-only probe has no user-visible side effect. Retry the
            # intent, but never post without proving that Slack still has the root;
            # Slack can otherwise accept an invalid thread_ts as a top-level post.
            exhausted = lease.attempt >= self._max_attempts
            await self._outbox.reject_dispatched(
                intent.id,
                retry_after=None if exhausted else _retry_time(now),
                safe_error=(
                    "slack_thread_probe_retry_exhausted"
                    if exhausted
                    else "slack_thread_probe_failed"
                ),
                dead=exhausted,
            )
            return DeliveryState.DEAD if exhausted else DeliveryState.RETRY
        try:
            response = await client.chat_postMessage(
                channel=intent.destination_channel_id,
                thread_ts=intent.destination_thread_ts,
                text=intent.payload,
            )
        except SlackApiError as exc:
            payload = getattr(exc.response, "data", None)
            error = payload.get("error") if isinstance(payload, Mapping) else None
            permanent = isinstance(error, str) and error in _PERMANENT_SLACK_ERRORS
            exhausted = lease.attempt >= self._max_attempts
            dead = permanent or exhausted
            await self._outbox.reject_dispatched(
                intent.id,
                retry_after=(
                    None
                    if dead
                    else _slack_retry_after(exc.response, now=now)
                    if error == "ratelimited"
                    else _retry_time(now)
                ),
                safe_error=(
                    "slack_error_permanent"
                    if permanent
                    else "slack_error_retry_exhausted"
                    if exhausted
                    else "slack_rate_limited"
                    if error == "ratelimited"
                    else "slack_error_retryable"
                ),
                dead=dead,
            )
            return DeliveryState.DEAD if dead else DeliveryState.RETRY
        except TimeoutError:
            await self._outbox.record_unknown_effect(
                intent.id,
                "slack_timeout_receipt_unknown",
            )
            return DeliveryState.UNKNOWN_EFFECT
        except Exception:
            # Once the API call begins, a transport exception cannot prove that Slack
            # did not accept the message.  Preserve the ambiguity for manual/operator
            # reconciliation instead of risking a duplicate user-visible reply.
            await self._outbox.record_unknown_effect(
                intent.id,
                "slack_transport_receipt_unknown",
            )
            return DeliveryState.UNKNOWN_EFFECT

        payload = getattr(response, "data", response)
        if not isinstance(payload, dict) or payload.get("ok") is not True:
            exhausted = lease.attempt >= self._max_attempts
            await self._outbox.reject_dispatched(
                intent.id,
                retry_after=None if exhausted else _retry_time(now),
                safe_error=(
                    "slack_error_retry_exhausted" if exhausted else "slack_error_retryable"
                ),
                dead=exhausted,
            )
            return DeliveryState.DEAD if exhausted else DeliveryState.RETRY
        message_ts = payload.get("ts")
        if not isinstance(message_ts, str) or not message_ts:
            await self._outbox.record_unknown_effect(
                intent.id,
                "slack_success_receipt_missing",
            )
            return DeliveryState.UNKNOWN_EFFECT
        await self._outbox.confirm_dispatched(intent.id, message_ts)
        return DeliveryState.DELIVERED

    async def _verify_thread_root(
        self,
        client: SlackPostClient,
        intent: DeliveryIntent,
    ) -> None:
        """Prove the destination root still exists before posting a reply.

        Older test doubles may only model ``chat_postMessage``; live Slack clients
        expose ``conversations_replies``. Keeping the capability optional preserves
        compatibility for those doubles while making the real delivery path fail
        closed when a deleted/replayed root would otherwise become a channel post.
        """

        probe = getattr(client, "conversations_replies", None)
        if probe is None:
            return
        try:
            response = await probe(
                channel=intent.destination_channel_id,
                ts=intent.destination_thread_ts,
                limit=1,
            )
        except SlackApiError as exc:
            payload = getattr(exc.response, "data", None)
            error = payload.get("error") if isinstance(payload, Mapping) else None
            if error in _MISSING_SLACK_THREAD_ERRORS:
                raise _MissingSlackThreadRoot from exc
            raise
        payload = getattr(response, "data", response)
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            error = payload.get("error") if isinstance(payload, Mapping) else None
            if error in _MISSING_SLACK_THREAD_ERRORS:
                raise _MissingSlackThreadRoot
            raise RuntimeError("Slack thread-root probe returned an unusable response")
        messages = payload.get("messages")
        if not isinstance(messages, list) or not any(
            isinstance(message, Mapping) and message.get("ts") == intent.destination_thread_ts
            for message in messages
        ):
            raise _MissingSlackThreadRoot

    async def dispatch_available(
        self,
        client: SlackPostClient,
        *,
        limit: int = 100,
        now: datetime | None = None,
    ) -> tuple[DeliveryState, ...]:
        """Drain a bounded durable backlog, including safely expired pre-call leases."""

        if limit < 1:
            raise ValueError("limit must be positive")
        states: list[DeliveryState] = []
        for _ in range(limit):
            state = await self.dispatch_once(client, now=now)
            if state is None:
                break
            states.append(state)
        return tuple(states)


def _eligible(current_time: object, max_attempts: int) -> ColumnElement[bool]:
    return and_(
        DeliveryOutboxRow.state.in_(
            (DeliveryState.PENDING.value, DeliveryState.RETRY.value, DeliveryState.LEASED.value)
        ),
        DeliveryOutboxRow.attempt_count < max_attempts,
        or_(DeliveryOutboxRow.retry_after.is_(None), DeliveryOutboxRow.retry_after <= current_time),
        or_(
            DeliveryOutboxRow.lease_expires_at.is_(None),
            DeliveryOutboxRow.lease_expires_at <= current_time,
        ),
    )


def _intent_model(row: DeliveryOutboxRow) -> DeliveryIntent:
    return DeliveryIntent(
        id=row.id,
        task_id=row.task_id,
        run_id=row.run_id,
        ingress_event_id=row.ingress_event_id,
        organization_id=row.organization_id,
        strategy_id=row.strategy_id,
        destination_channel_id=row.destination_channel_id,
        destination_thread_ts=row.destination_thread_ts,
        kind=DeliveryKind(row.kind),
        payload_version=row.payload_version,
        payload_hash=row.payload_hash,
        payload=row.payload,
        state=DeliveryState(row.state),
        attempt_count=row.attempt_count,
        receipt_message_ts=row.receipt_message_ts,
        lease_owner=row.lease_owner,
        lease_token=row.lease_token,
        lease_expires_at=row.lease_expires_at,
        retry_after=row.retry_after,
        last_error=row.last_error,
    )


def _assert_immutable_match(
    row: DeliveryOutboxRow,
    *,
    destination_channel_id: str,
    destination_thread_ts: str,
    payload_hash: str,
    payload: str,
) -> None:
    if (
        row.destination_channel_id != destination_channel_id
        or row.destination_thread_ts != destination_thread_ts
        or row.payload_hash != payload_hash
        or row.payload != payload
    ):
        raise DeliveryPayloadDriftError("delivery idempotency key payload drift")


def _payload_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_payload(payload: str, payload_version: int) -> None:
    if not payload or len(payload) > 40_000:
        raise ValueError("payload must be 1-40000 characters")
    if payload_version < 1:
        raise ValueError("payload_version must be positive")


def _validate_owner(owner: str) -> None:
    if not owner or owner != owner.strip() or len(owner) > 128:
        raise ValueError("owner must be a non-empty value of at most 128 characters")


def _validate_duration(value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > 86_400:
        raise ValueError("duration must be finite and between 0 and 86400")


def _validate_safe_error(value: str) -> None:
    if not value or len(value) > 255:
        raise ValueError("safe error must be 1-255 characters")


def _retry_time(now: datetime | None) -> datetime:
    return (now or datetime.now(UTC)) + timedelta(seconds=5)


_PERMANENT_SLACK_ERRORS = frozenset(
    {
        "account_inactive",
        "channel_not_found",
        "ekm_access_denied",
        "invalid_auth",
        "is_archived",
        "missing_scope",
        "no_permission",
        "not_authed",
        "not_in_channel",
        "restricted_action",
        "team_access_not_granted",
        "token_expired",
        "token_revoked",
    }
)

_MISSING_SLACK_THREAD_ERRORS = frozenset(
    {
        "message_not_found",
        "thread_not_found",
        "thread_deleted",
    }
)


def _slack_retry_after(response: object, *, now: datetime | None) -> datetime:
    headers = getattr(response, "headers", None)
    raw: object | None = None
    if isinstance(headers, Mapping):
        raw = headers.get("Retry-After") or headers.get("retry-after")
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    try:
        seconds = float(raw) if isinstance(raw, (str, int, float)) else 5.0
    except (TypeError, ValueError):
        seconds = 5.0
    if not math.isfinite(seconds):
        seconds = 5.0
    seconds = min(86_400.0, max(1.0, seconds))
    return (now or datetime.now(UTC)) + timedelta(seconds=seconds)
