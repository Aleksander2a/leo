from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web import SlackResponse

import leo.persistence.outbox as outbox_module
from leo.persistence.outbox import (
    DeliveryIntent,
    DeliveryKind,
    DeliveryLease,
    DeliveryLeaseConflictError,
    DeliveryState,
    PostgresDeliveryOutbox,
    SlackOutboxDispatcher,
)


class _SimulatedProcessDeath(BaseException):
    """Fault injection that bypasses application exception recovery."""


class _MemoryOutbox:
    def __init__(self, *, crash_at: str | None = None) -> None:
        self.intent = DeliveryIntent(
            id="delivery-1",
            task_id="task-1",
            run_id="run-1",
            ingress_event_id="event-1",
            organization_id="org-1",
            strategy_id="strategy-1",
            destination_channel_id="channel-1",
            destination_thread_ts="thread-1",
            kind=DeliveryKind.FINAL,
            payload_version=1000,
            payload_hash="hash",
            payload="safe final",
            state=DeliveryState.PENDING,
            attempt_count=0,
            receipt_message_ts=None,
            lease_owner=None,
            lease_token=None,
            lease_expires_at=None,
            retry_after=None,
            last_error=None,
        )
        self.crash_at = crash_at
        self._token_sequence = 0

    async def claim_next(
        self,
        owner: str,
        *,
        lease_seconds: float,
        max_attempts: int,
        now: datetime | None,
        intent_id: str | None,
    ) -> tuple[DeliveryLease, DeliveryIntent] | None:
        current = now or datetime.now(UTC)
        intent = self.intent
        if intent_id is not None and intent.id != intent_id:
            return None
        if intent.attempt_count >= max_attempts:
            return None
        if intent.state not in {DeliveryState.PENDING, DeliveryState.RETRY, DeliveryState.LEASED}:
            return None
        if intent.retry_after is not None and intent.retry_after > current:
            return None
        if intent.lease_expires_at is not None and intent.lease_expires_at > current:
            return None
        self._token_sequence += 1
        token = f"lease-{self._token_sequence}"
        expires_at = current + timedelta(seconds=lease_seconds)
        self.intent = replace(
            intent,
            state=DeliveryState.LEASED,
            attempt_count=intent.attempt_count + 1,
            lease_owner=owner,
            lease_token=token,
            lease_expires_at=expires_at,
            last_error=None,
        )
        return (
            DeliveryLease(
                intent_id=intent.id,
                owner=owner,
                token=token,
                attempt=self.intent.attempt_count,
                expires_at=expires_at,
            ),
            self.intent,
        )

    async def mark_dispatch_started(self, lease: DeliveryLease) -> None:
        self._assert_lease(lease)
        if self.crash_at == "before_dispatch_fence_commit":
            raise _SimulatedProcessDeath
        self.intent = replace(
            self.intent,
            state=DeliveryState.UNKNOWN_EFFECT,
            lease_owner=None,
            lease_token=None,
            lease_expires_at=None,
            retry_after=None,
            last_error="dispatch_started_receipt_pending",
        )
        if self.crash_at == "after_dispatch_fence_commit":
            raise _SimulatedProcessDeath

    async def confirm_dispatched(self, intent_id: str, message_ts: str) -> None:
        assert intent_id == self.intent.id
        assert self.intent.state is DeliveryState.UNKNOWN_EFFECT
        if self.crash_at == "after_slack_success_before_receipt_commit":
            raise _SimulatedProcessDeath
        self.intent = replace(
            self.intent,
            state=DeliveryState.DELIVERED,
            receipt_message_ts=message_ts,
            last_error=None,
        )

    async def record_unknown_effect(self, intent_id: str, safe_error: str) -> None:
        assert intent_id == self.intent.id
        assert self.intent.state is DeliveryState.UNKNOWN_EFFECT
        self.intent = replace(self.intent, last_error=safe_error)

    async def reject_dispatched(
        self,
        intent_id: str,
        *,
        retry_after: datetime | None,
        safe_error: str,
        dead: bool,
    ) -> None:
        assert intent_id == self.intent.id
        assert self.intent.state is DeliveryState.UNKNOWN_EFFECT
        self.intent = replace(
            self.intent,
            state=DeliveryState.DEAD if dead else DeliveryState.RETRY,
            retry_after=retry_after,
            last_error=safe_error,
        )

    def _assert_lease(self, lease: DeliveryLease) -> None:
        assert self.intent.state is DeliveryState.LEASED
        assert self.intent.lease_owner == lease.owner
        assert self.intent.lease_token == lease.token


class _AcceptedSlackClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
        assert (channel, thread_ts, text) == ("channel-1", "thread-1", "safe final")
        self.calls += 1
        return {"ok": True, "ts": "receipt-1"}


class _ReceiptTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_: object) -> None:
        return None


class _ReceiptSession:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    async def __aenter__(self) -> _ReceiptSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def begin(self) -> _ReceiptTransaction:
        return _ReceiptTransaction()

    async def scalar(self, statement: object) -> object | None:
        del statement
        return self.values.pop(0) if self.values else None


class _ReceiptSessions:
    def __init__(self, session: _ReceiptSession) -> None:
        self.session = session

    def __call__(self) -> _ReceiptSession:
        return self.session


class _UnusedIds:
    def new(self, namespace: str) -> str:
        raise AssertionError(f"unexpected ID allocation: {namespace}")


@pytest.mark.asyncio
async def test_confirmed_outbox_receipt_persists_exact_sanitized_assistant_turn_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = SimpleNamespace(
        id="delivery-confirmed",
        task_id="task-confirmed",
        ingress_event_id="Ev-confirmed",
        organization_id="org-confirmed",
        strategy_id="conversation",
        destination_channel_id="C-confirmed",
        destination_thread_ts="1787393805.333519",
        payload="Leo's confirmed delivered reply.",
        state=DeliveryState.UNKNOWN_EFFECT.value,
        receipt_message_ts=None,
        retry_after=None,
        last_error="dispatch_started_receipt_pending",
    )
    ingress = SimpleNamespace(
        conversation_id="slack-conversation-confirmed",
        organization_id="org-confirmed",
        strategy_id="conversation",
        channel_id="C-confirmed",
        thread_root_ts="1787393805.333519",
        context_access_hash="a" * 64,
    )
    task = SimpleNamespace(
        id="task-confirmed",
        thread_id="thread-confirmed",
        organization_id="org-confirmed",
        strategy_id="conversation",
    )
    session = _ReceiptSession([intent, ingress, task])
    recorded: list[object] = []

    async def record_message(_: object, message: object) -> None:
        recorded.append(message)

    monkeypatch.setattr(outbox_module, "persist_conversation_plane_message", record_message)
    outbox = PostgresDeliveryOutbox(  # type: ignore[arg-type]
        _ReceiptSessions(session),
        _UnusedIds(),
    )

    await outbox.confirm_dispatched("delivery-confirmed", "1787393828.798329")

    assert intent.state == DeliveryState.DELIVERED.value
    assert intent.receipt_message_ts == "1787393828.798329"
    assert len(recorded) == 1
    message = recorded[0]
    assert message.external_event_id == "slack-delivery:delivery-confirmed"  # type: ignore[attr-defined]
    assert message.actor_id == "leo"  # type: ignore[attr-defined]
    assert message.role.value == "assistant"  # type: ignore[attr-defined]
    assert message.provider_message_ts == "1787393828.798329"  # type: ignore[attr-defined]
    assert message.provider_thread_root_ts == "1787393805.333519"  # type: ignore[attr-defined]
    assert message.context_access_hash == "a" * 64  # type: ignore[attr-defined]
    assert message.text == "Leo's confirmed delivered reply."  # type: ignore[attr-defined]

    session.values[:] = [intent]
    with pytest.raises(
        DeliveryLeaseConflictError,
        match="delivery dispatch is no longer receipt-pending",
    ):
        await outbox.confirm_dispatched("delivery-confirmed", "1787393828.798329")
    assert len(recorded) == 1


@pytest.mark.asyncio
async def test_crash_before_dispatch_fence_commit_reclaims_without_duplicate_call() -> None:
    outbox = _MemoryOutbox(crash_at="before_dispatch_fence_commit")
    client = _AcceptedSlackClient()
    dispatcher = SlackOutboxDispatcher(  # type: ignore[arg-type]
        outbox,
        owner="dispatcher-1",
        lease_seconds=1,
    )
    started = datetime(2026, 1, 1, tzinfo=UTC)

    with pytest.raises(_SimulatedProcessDeath):
        await dispatcher.dispatch_once(client, now=started)
    assert client.calls == 0
    assert outbox.intent.state is DeliveryState.LEASED

    outbox.crash_at = None
    assert (
        await dispatcher.dispatch_once(client, now=started + timedelta(seconds=1))
        is DeliveryState.DELIVERED
    )
    assert client.calls == 1
    assert outbox.intent.attempt_count == 2


@pytest.mark.asyncio
async def test_crash_after_dispatch_fence_commit_never_blindly_calls_slack() -> None:
    outbox = _MemoryOutbox(crash_at="after_dispatch_fence_commit")
    client = _AcceptedSlackClient()
    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-1")  # type: ignore[arg-type]

    with pytest.raises(_SimulatedProcessDeath):
        await dispatcher.dispatch_once(client)
    assert client.calls == 0
    assert outbox.intent.state is DeliveryState.UNKNOWN_EFFECT

    outbox.crash_at = None
    assert await dispatcher.dispatch_once(client) is None
    assert client.calls == 0


@pytest.mark.asyncio
async def test_slack_success_before_receipt_commit_stays_unknown_without_duplicate() -> None:
    outbox = _MemoryOutbox(crash_at="after_slack_success_before_receipt_commit")
    client = _AcceptedSlackClient()
    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-1")  # type: ignore[arg-type]

    with pytest.raises(_SimulatedProcessDeath):
        await dispatcher.dispatch_once(client)
    assert client.calls == 1
    assert outbox.intent.state is DeliveryState.UNKNOWN_EFFECT
    assert outbox.intent.receipt_message_ts is None

    outbox.crash_at = None
    assert await dispatcher.dispatch_once(client) is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_transport_timeout_after_call_start_is_unknown_and_not_retried() -> None:
    outbox = _MemoryOutbox()

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            del channel, thread_ts, text
            self.calls += 1
            raise TimeoutError

    client = Client()
    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-1")  # type: ignore[arg-type]

    assert await dispatcher.dispatch_once(client) is DeliveryState.UNKNOWN_EFFECT
    assert outbox.intent.last_error == "slack_timeout_receipt_unknown"
    assert await dispatcher.dispatch_once(client) is None
    assert client.calls == 1


@pytest.mark.asyncio
async def test_malformed_slack_success_without_explicit_ok_is_retryable_not_delivered() -> None:
    outbox = _MemoryOutbox()

    class Client:
        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            del channel, thread_ts, text
            return {"ts": "untrusted-receipt"}

    state = await SlackOutboxDispatcher(  # type: ignore[arg-type]
        outbox,
        owner="dispatcher-1",
    ).dispatch_once(Client())

    assert state is DeliveryState.RETRY
    assert outbox.intent.state is DeliveryState.RETRY
    assert outbox.intent.receipt_message_ts is None
    assert outbox.intent.last_error == "slack_error_retryable"


def _slack_error(error: str, *, status_code: int, retry_after: str | None = None) -> SlackApiError:
    headers = {} if retry_after is None else {"Retry-After": retry_after}
    response = SlackResponse(
        client=None,  # type: ignore[arg-type]
        http_verb="POST",
        api_url="https://slack.com/api/chat.postMessage",
        req_args={},
        data={"ok": False, "error": error},
        headers=headers,
        status_code=status_code,
    )
    return SlackApiError("safe test rejection", response)


@pytest.mark.asyncio
async def test_known_rate_limit_honors_retry_after_without_unknown_effect() -> None:
    outbox = _MemoryOutbox()
    now = datetime(2026, 1, 1, tzinfo=UTC)

    class Client:
        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            del channel, thread_ts, text
            raise _slack_error("ratelimited", status_code=429, retry_after="30")

    state = await SlackOutboxDispatcher(  # type: ignore[arg-type]
        outbox,
        owner="dispatcher-1",
    ).dispatch_once(Client(), now=now)

    assert state is DeliveryState.RETRY
    assert outbox.intent.state is DeliveryState.RETRY
    assert outbox.intent.retry_after == now + timedelta(seconds=30)
    assert outbox.intent.last_error == "slack_rate_limited"


@pytest.mark.asyncio
async def test_known_permanent_slack_rejection_is_dead_without_retry() -> None:
    outbox = _MemoryOutbox()

    class Client:
        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            del channel, thread_ts, text
            raise _slack_error("not_in_channel", status_code=403)

    state = await SlackOutboxDispatcher(  # type: ignore[arg-type]
        outbox,
        owner="dispatcher-1",
    ).dispatch_once(Client())

    assert state is DeliveryState.DEAD
    assert outbox.intent.state is DeliveryState.DEAD
    assert outbox.intent.retry_after is None
    assert outbox.intent.last_error == "slack_error_permanent"
