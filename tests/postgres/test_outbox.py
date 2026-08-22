from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.evals.revised_live_acceptance import (
    OutboxRecoveryCaseId,
    export_contract,
    make_outbox_recovery_probe,
)
from leo.harness.models import OriginRef, Run, ScopeKey, Task, Thread
from leo.harness.ports import IdGenerator, RunStore
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.slack.events import build_context_access_hash
from leo.persistence.outbox import (
    DeliveryKind,
    DeliveryPayloadDriftError,
    DeliveryState,
    PostgresDeliveryOutbox,
    SlackOutboxDispatcher,
)
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import (
    ConversationRow,
    DeliveryOutboxRow,
    RunRow,
    SlackIngressEventRow,
    TaskRow,
)


def _fixture_suffix() -> str:
    return uuid4().hex[:12]


def _bounded_id(prefix: str, suffix: str, counter: int | None = None) -> str:
    tail = suffix if counter is None else f"{suffix}-{counter:x}"
    prefix_budget = 32 - len(tail) - 1
    return f"{prefix[:prefix_budget]}-{tail}"


class _UniqueIds(IdGenerator):
    def __init__(self) -> None:
        self._suffix = _fixture_suffix()
        self._counter = 0

    def new(self, prefix: str) -> str:
        self._counter += 1
        return _bounded_id(prefix, self._suffix, self._counter)


@pytest_asyncio.fixture
async def outbox_store(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]]]:
    yield (
        PostgresDeliveryOutbox(preserved_postgres_sessions, _UniqueIds()),
        PostgresRunStore(
            preserved_postgres_sessions,
            FixedClock(),
            SequentialIdGenerator(),
        ),
        preserved_postgres_sessions,
    )


async def _seed_source(
    store: RunStore,
    sessions: async_sessionmaker[AsyncSession],
    suffix: str,
) -> tuple[str, str, str, str, str]:
    scope = ScopeKey(
        organization_id=f"org-outbox-{suffix}",
        strategy_id=f"strategy-outbox-{suffix}",
    )
    thread_id = f"thread-{suffix}"
    task_id = f"task-{suffix}"
    run_id = f"run-{suffix}"
    event_id = f"event-{suffix}"
    team_id = f"T{suffix.upper()}"
    channel_id = f"C{suffix[::-1].upper()}"
    user_id = f"U{suffix.upper()}"
    epoch = int(suffix, 16)
    root_ts = f"{epoch}.000000"
    conversation_id = f"conversation-{suffix}"
    thread = Thread(
        id=thread_id,
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=f"slack:{team_id}:{channel_id}:{root_ts}",
            external_event_id=event_id,
            external_channel_id=channel_id,
        ),
    )
    task = Task(
        id=task_id,
        thread_id=thread.id,
        scope=scope,
        objective="outbox test",
    )
    run = Run(id=run_id, task_id=task.id, scope=scope)
    await store.seed(thread, task, run)
    async with sessions() as session, session.begin():
        session.add(
            ConversationRow(
                id=conversation_id,
                provider="slack",
                team_id=team_id,
                external_id=channel_id,
                kind="channel",
                actor_id=None,
                version=1,
            )
        )
        session.add(
            SlackIngressEventRow(
                event_id=event_id,
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
                message_ts=f"{epoch}.000001",
                thread_root_ts=root_ts,
                conversation_key=f"slack:{team_id}:{channel_id}:{root_ts}",
                prompt="quote NVDA",
                conversation_kind="ordinary_internal",
                trigger_kind="app_mention",
                context_conversation_ids=[channel_id],
                context_access_hash=build_context_access_hash(
                    team_id=team_id,
                    user_id=user_id,
                    channel_id=channel_id,
                    context_conversation_ids=(channel_id,),
                ),
                conversation_id=conversation_id,
                organization_id=scope.organization_id,
                strategy_id=scope.strategy_id,
                mapping_version=1,
                status="queued",
                task_id=task.id,
                launch_status="queued",
            )
        )
    return task.id, run.id, event_id, channel_id, root_ts


async def _mark_terminal(
    sessions: async_sessionmaker[AsyncSession], task_id: str, run_id: str
) -> None:
    async with sessions() as session, session.begin():
        await session.execute(
            update(TaskRow)
            .where(TaskRow.id == task_id)
            .values(status="completed", final_output="safe answer", version=1)
        )
        await session.execute(
            update(RunRow)
            .where(RunRow.id == run_id)
            .values(
                status="completed",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                final_output="safe answer",
                terminal_reason="verified_completion",
                version=1,
            )
        )


@pytest.mark.asyncio
async def test_intent_is_idempotent_and_payload_drift_is_rejected(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, _channel_id, _root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    first = await outbox.ensure_intent(
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
        kind=DeliveryKind.FINAL,
        payload_version=1,
        payload="safe final",
    )
    second = await outbox.ensure_intent(
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
        kind=DeliveryKind.FINAL,
        payload_version=1,
        payload="safe final",
    )
    assert second.id == first.id
    assert second.payload_hash == first.payload_hash
    with pytest.raises(DeliveryPayloadDriftError):
        await outbox.ensure_intent(
            task_id=task_id,
            run_id=run_id,
            ingress_event_id=event_id,
            kind=DeliveryKind.FINAL,
            payload_version=1,
            payload="changed final",
        )


@pytest.mark.asyncio
async def test_two_dispatchers_claim_disjoint_intents_and_record_receipts(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, channel_id, root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    intents = []
    for kind in (DeliveryKind.PROGRESS, DeliveryKind.FINAL):
        intents.append(
            await outbox.ensure_intent(
                task_id=task_id,
                run_id=run_id,
                ingress_event_id=event_id,
                kind=kind,
                payload_version=1,
                payload=f"safe {kind.value}",
            )
        )

    class Client:
        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            assert channel == channel_id
            assert thread_ts == root_ts
            assert text.startswith("safe ")
            return {"ok": True, "ts": f"receipt-{text}"}

    now = datetime(2026, 1, 1, tzinfo=UTC)
    first = await SlackOutboxDispatcher(outbox, owner="dispatcher-a").dispatch_once(
        Client(), now=now, intent_id=intents[0].id
    )
    second = await SlackOutboxDispatcher(outbox, owner="dispatcher-b").dispatch_once(
        Client(), now=now, intent_id=intents[1].id
    )
    assert {first, second} == {DeliveryState.DELIVERED}


@pytest.mark.asyncio
async def test_pending_final_intent_is_delivered_once(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, channel_id, root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    intent = await outbox.ensure_intent(
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
        kind=DeliveryKind.FINAL,
        payload_version=1,
        payload="safe pending recovery final",
    )
    before = {
        "outbox_count": 1,
        "intent_id": intent.id,
        "state": intent.state,
        "attempt_count": intent.attempt_count,
        "receipt_present": intent.receipt_message_ts is not None,
    }

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            assert (channel, thread_ts, text) == (
                channel_id,
                root_ts,
                "safe pending recovery final",
            )
            self.calls += 1
            return {"ok": True, "ts": "1787361066.900001"}

    client = Client()
    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-pending-recovery")
    assert await dispatcher.dispatch_once(client, intent_id=intent.id) is DeliveryState.DELIVERED
    assert await dispatcher.dispatch_once(client, intent_id=intent.id) is None
    assert client.calls == 1
    assert (
        await outbox.reconcile_terminal(
            lambda _task, _run: "must not duplicate",
            task_id=task_id,
            run_id=run_id,
            ingress_event_id=event_id,
        )
        == ()
    )
    async with sessions() as session:
        final = await session.scalar(
            select(DeliveryOutboxRow).where(
                DeliveryOutboxRow.id == intent.id,
                DeliveryOutboxRow.kind == DeliveryKind.FINAL.value,
            )
        )
    assert final is not None
    after = {
        "outbox_count": 1,
        "intent_id": final.id,
        "state": final.state,
        "attempt_count": final.attempt_count,
        "receipt_present": final.receipt_message_ts is not None,
        "physical_delivery_count": client.calls,
    }
    probe = make_outbox_recovery_probe(
        case_id=OutboxRecoveryCaseId.PENDING_FINAL,
        initial_final_outbox_count=1,
        repair_created_count=0,
        before=before,
        after=after,
    )
    export_contract(probe, tmp_path / "outbox-recovery-probe.json")


@pytest.mark.asyncio
async def test_timeout_is_unknown_effect_and_not_automatically_retried(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, _channel_id, _root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    intent = await outbox.ensure_intent(
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
        kind=DeliveryKind.FINAL,
        payload_version=1,
        payload="safe timeout case",
    )

    class Client:
        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            raise TimeoutError

    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-timeout")
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert (
        await dispatcher.dispatch_once(Client(), now=now, intent_id=intent.id)
        is DeliveryState.UNKNOWN_EFFECT
    )
    assert (
        await dispatcher.dispatch_once(
            Client(), now=now + timedelta(seconds=1), intent_id=intent.id
        )
        is None
    )


@pytest.mark.asyncio
async def test_terminal_reconciliation_repairs_missing_final_intent(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
    tmp_path: Path,
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, channel_id, root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    before = {
        "outbox_count": 0,
        "task_id": task_id,
        "run_id": run_id,
        "terminal": True,
    }
    repaired = await outbox.reconcile_terminal(
        lambda _task, run: f"{run.final_output}\n\nRun: `{run.id}`",
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
    )
    assert len(repaired) == 1
    assert repaired[0].state is DeliveryState.PENDING

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_postMessage(self, *, channel: str, thread_ts: str, text: str) -> object:
            assert channel == channel_id
            assert thread_ts == root_ts
            assert text.startswith("safe answer")
            self.calls += 1
            return {"ok": True, "ts": "1787361066.900002"}

    client = Client()
    dispatcher = SlackOutboxDispatcher(outbox, owner="dispatcher-missing-recovery")
    assert (
        await dispatcher.dispatch_once(client, intent_id=repaired[0].id) is DeliveryState.DELIVERED
    )
    assert await dispatcher.dispatch_once(client, intent_id=repaired[0].id) is None
    assert client.calls == 1
    assert (
        await outbox.reconcile_terminal(
            lambda _task, run: f"{run.final_output}\n\nRun: `{run.id}`",
            task_id=task_id,
            run_id=run_id,
            ingress_event_id=event_id,
        )
        == ()
    )
    async with sessions() as session:
        final = await session.scalar(
            select(DeliveryOutboxRow).where(
                DeliveryOutboxRow.id == repaired[0].id,
                DeliveryOutboxRow.kind == DeliveryKind.FINAL.value,
            )
        )
    assert final is not None
    after = {
        "outbox_count": 1,
        "intent_id": final.id,
        "state": final.state,
        "attempt_count": final.attempt_count,
        "receipt_present": final.receipt_message_ts is not None,
        "physical_delivery_count": client.calls,
    }
    probe = make_outbox_recovery_probe(
        case_id=OutboxRecoveryCaseId.MISSING_FINAL,
        initial_final_outbox_count=0,
        repair_created_count=len(repaired),
        before=before,
        after=after,
    )
    export_contract(probe, tmp_path / "outbox-recovery-probe.json")


@pytest.mark.asyncio
async def test_terminal_reconciliation_does_not_redeliver_after_renderer_upgrade(
    outbox_store: tuple[PostgresDeliveryOutbox, RunStore, async_sessionmaker[AsyncSession]],
) -> None:
    outbox, store, sessions = outbox_store
    task_id, run_id, event_id, _channel_id, _root_ts = await _seed_source(
        store, sessions, _fixture_suffix()
    )
    await _mark_terminal(sessions, task_id, run_id)
    await outbox.ensure_intent(
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
        kind=DeliveryKind.FINAL,
        payload_version=1000,
        payload="renderer-v1 final",
    )

    repaired = await outbox.reconcile_terminal(
        lambda _task, _run: "renderer-v2 final",
        payload_version=2000,
        task_id=task_id,
        run_id=run_id,
        ingress_event_id=event_id,
    )

    assert repaired == ()
