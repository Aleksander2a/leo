from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

import leo.cli as cli_module
import leo.worker.main as worker_main_module
import leo.worker.slack_conversation as slack_conversation_module
from leo.cli import _LiveSlackHarnessRuntime
from leo.harness.models import (
    EventDraft,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
)
from leo.harness.store_errors import ConcurrencyError, StoreError
from leo.harness.transitions import (
    cancel_task_and_run,
    fail_task_and_run,
    start_task_and_run,
)
from leo.integrations.fake import FixedClock
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.slack.context import SlackThreadContextError
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackBotPresence,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackLaunchRef,
    SlackMentionJob,
    SlackScopeResolution,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.integrations.slack.render import RenderedSlackText
from leo.integrations.slack.socket_mode import SlackJobProcessor
from leo.persistence.outbox import DeliveryKind, DeliveryState
from leo.persistence.task_leases import TaskLease
from leo.worker.slack_conversation import (
    _reconcile_timeout_run_winner,
    reconcile_admitted_slack_terminal_winner,
)

NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org-terminal-race", strategy_id="conversation")


def _queued_bundle() -> RunBundle:
    thread = Thread(
        id="thread-terminal-race",
        scope=SCOPE,
        origin=OriginRef(
            provider="slack",
            external_thread_id="slack:T1:C1:1.0",
            external_event_id="Ev-terminal-race",
            external_channel_id="C1",
        ),
    )
    task = Task(
        id="task-terminal-race",
        thread_id=thread.id,
        scope=SCOPE,
        objective="Continue until a terminal race is resolved.",
    )
    run = Run(id="run-terminal-race", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run)


def _active_bundle() -> RunBundle:
    queued = _queued_bundle()
    task, run = start_task_and_run(queued.task, queued.run, started_at=NOW)
    return RunBundle(thread=queued.thread, task=task, run=run)


def _cancelled_bundle() -> RunBundle:
    active = _active_bundle()
    task, run = cancel_task_and_run(
        active.task,
        active.run,
        "slack_user_cancelled",
        usage=active.run.usage,
    )
    return RunBundle(thread=active.thread, task=task, run=run)


def _failed_bundle() -> RunBundle:
    active = _active_bundle()
    task, run = fail_task_and_run(
        active.task,
        active.run,
        "retry_attempts_exhausted",
        usage=active.run.usage,
    )
    return RunBundle(thread=active.thread, task=task, run=run)


def _admitted() -> AdmittedSlackMention:
    context_ids = ("C1",)
    job = SlackMentionJob(
        event_id="Ev-terminal-race",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="1.2",
        thread_root_ts="1.0",
        conversation_key="slack:T1:C1:1.0",
        prompt="Continue until a terminal race is resolved.",
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=context_ids,
        context_access_hash=build_context_access_hash(
            team_id="T1",
            user_id="U1",
            channel_id="C1",
            context_conversation_ids=context_ids,
        ),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )
    return AdmittedSlackMention(
        job=job,
        resolution=SlackScopeResolution(scope=SCOPE, mapping_version=1, provisioned=False),
        launch=SlackLaunchRef(
            thread_id="thread-terminal-race",
            task_id="task-terminal-race",
            run_id="run-terminal-race",
        ),
    )


class _Rows:
    def __init__(self, values: tuple[str, ...]) -> None:
        self._values = values

    def all(self) -> tuple[str, ...]:
        return self._values


class _PlanSession:
    def __init__(self, owner: _PlanSessions) -> None:
        self._owner = owner

    async def __aenter__(self) -> _PlanSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def scalars(self, statement: object) -> _Rows:
        self._owner.statements.append(statement)
        values = self._owner.responses.pop(0) if self._owner.responses else ()
        return _Rows(values)


class _PlanSessions:
    def __init__(self, *responses: tuple[str, ...]) -> None:
        self.responses = list(responses)
        self.statements: list[object] = []

    def __call__(self) -> _PlanSession:
        return _PlanSession(self)


@dataclass
class _BundleStore:
    bundle: RunBundle

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        assert (task_id, run_id, scope) == (
            self.bundle.task.id,
            self.bundle.run.id,
            self.bundle.run.scope,
        )
        return self.bundle


class _CancellationWinsStore:
    def __init__(self) -> None:
        self.bundle = _active_bundle()
        self.commit_calls = 0

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        assert (task_id, run_id, scope) == (
            self.bundle.task.id,
            self.bundle.run.id,
            self.bundle.run.scope,
        )
        return self.bundle

    async def commit(self, **kwargs: object) -> RunBundle:
        del kwargs
        self.commit_calls += 1
        self.bundle = _cancelled_bundle()
        raise ConcurrencyError("cancellation won the timeout CAS")


@pytest.mark.asyncio
async def test_timeout_cas_reloads_cancellation_winner_instead_of_failing_generically() -> None:
    store = _CancellationWinsStore()

    winner = await _reconcile_timeout_run_winner(
        durable=store,  # type: ignore[arg-type]
        fenced=store,  # type: ignore[arg-type]
        task_id="task-terminal-race",
        run_id="run-terminal-race",
        scope=SCOPE,
        clock=FixedClock(NOW),
        reason="slack_runtime_deadline_exceeded",
    )

    assert winner.run.status is RunStatus.CANCELLED
    assert winner.run.terminal_reason == "slack_user_cancelled"
    assert store.commit_calls == 1


@pytest.mark.asyncio
async def test_terminal_reload_propagates_exact_parent_plan_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    class PlanStore:
        def __init__(self, *_: object) -> None:
            return None

        async def terminate_for_parent(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    monkeypatch.setattr(slack_conversation_module, "PostgresPlanStore", PlanStore)
    sessions = _PlanSessions(("plan-terminal-race",), ())
    admitted = _admitted()
    store = _BundleStore(_cancelled_bundle())

    first = await reconcile_admitted_slack_terminal_winner(
        sessions=sessions,  # type: ignore[arg-type]
        admitted=admitted,
        store=store,  # type: ignore[arg-type]
    )
    second = await reconcile_admitted_slack_terminal_winner(
        sessions=sessions,  # type: ignore[arg-type]
        admitted=admitted,
        store=store,  # type: ignore[arg-type]
    )

    assert first == second == store.bundle
    assert calls == [
        {
            "scope": SCOPE,
            "plan_id": "plan-terminal-race",
            "parent_task_id": "task-terminal-race",
            "parent_run_id": "run-terminal-race",
            "parent_status": RunStatus.CANCELLED,
            "reason": "slack_user_cancelled",
            "child_terminal_reason": "parent_cancelled",
        }
    ]


class _Ingress:
    def __init__(self, admitted: AdmittedSlackMention) -> None:
        self.admitted = admitted
        self.statuses: list[tuple[str, str, str | None]] = []

    async def load_linked_mention(self, task_id: str) -> AdmittedSlackMention:
        assert self.admitted.launch is not None
        assert task_id == self.admitted.launch.task_id
        return self.admitted

    async def mark_linked_status(
        self,
        event_id: str,
        status: str,
        safe_error: str | None,
    ) -> None:
        self.statuses.append((event_id, status, safe_error))


class _Leases:
    def __init__(
        self,
        *,
        claimed: TaskLease | None,
        exhausted: TaskLease | None = None,
    ) -> None:
        self.claimed = claimed
        self.exhausted = exhausted
        self.abandoned = False

    async def claim_task(self, *_: object, **__: object) -> TaskLease | None:
        return self.claimed

    async def claim_exhausted_task(self, *_: object, **__: object) -> TaskLease | None:
        return self.exhausted

    async def heartbeat(self, lease: TaskLease, **_: object) -> TaskLease:
        return lease

    async def abandon(self, *_: object, **__: object) -> None:
        self.abandoned = True


def _lease(attempt: int = 1) -> TaskLease:
    return TaskLease(
        task_id="task-terminal-race",
        owner="runtime-owner",
        token="lease-terminal-race",
        attempt=attempt,
        expires_at=NOW + timedelta(minutes=1),
    )


def _runtime(
    *,
    admitted: AdmittedSlackMention,
    bundle: RunBundle,
    leases: _Leases,
    sessions: _PlanSessions,
) -> tuple[_LiveSlackHarnessRuntime, _Ingress]:
    runtime = object.__new__(_LiveSlackHarnessRuntime)
    ingress = _Ingress(admitted)
    runtime._settings = SimpleNamespace(leo_max_run_seconds=60)  # type: ignore[attr-defined]
    runtime._client = object()  # type: ignore[attr-defined]
    runtime._sessions = sessions  # type: ignore[attr-defined]
    runtime._ingress = ingress  # type: ignore[attr-defined]
    runtime._leases = leases  # type: ignore[attr-defined]
    runtime._owner = "runtime-owner"  # type: ignore[attr-defined]
    runtime._clock = FixedClock(NOW)  # type: ignore[attr-defined]
    runtime._provider_gates = ProviderGateRegistry(runtime._clock)  # type: ignore[attr-defined]
    runtime._run_store = _BundleStore(bundle)  # type: ignore[attr-defined]
    runtime._history = SimpleNamespace(  # type: ignore[attr-defined]
        load=_history_result,
    )
    return runtime, ingress


async def _history_result(_: SlackMentionJob) -> object:
    return SimpleNamespace(
        items=(),
        reopen_ranges=(),
        manifest=SimpleNamespace(selection_digest="a" * 64),
    )


@pytest.mark.asyncio
async def test_live_slack_runtime_reuses_one_provider_registry_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_registries: list[ProviderGateRegistry] = []

    async def failed_turn(**kwargs: object) -> object:
        registry = kwargs["provider_gates"]
        assert isinstance(registry, ProviderGateRegistry)
        captured_registries.append(registry)
        return SimpleNamespace(
            run=SimpleNamespace(
                id="run-terminal-race",
                status=RunStatus.FAILED,
                terminal_reason="provider_demo_failure",
                final_output=None,
            )
        )

    monkeypatch.setattr(cli_module, "run_admitted_slack_conversation", failed_turn)
    monkeypatch.setattr(cli_module, "slack_history_authority_ids", lambda _manifest: ())
    admitted = _admitted()
    runtime, _ingress = _runtime(
        admitted=admitted,
        bundle=_active_bundle(),
        leases=_Leases(claimed=_lease()),
        sessions=_PlanSessions((), ()),
    )

    await runtime.handle(admitted)
    await runtime.handle(admitted)

    assert captured_registries == [runtime._provider_gates, runtime._provider_gates]  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stale_coordinator_store_error_renders_cancelled_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stale_commit(**_: object) -> object:
        raise StoreError("stale coordinator commit")

    monkeypatch.setattr(cli_module, "run_admitted_slack_conversation", stale_commit)
    admitted = _admitted()
    leases = _Leases(claimed=_lease())
    runtime, ingress = _runtime(
        admitted=admitted,
        bundle=_cancelled_bundle(),
        leases=leases,
        sessions=_PlanSessions(()),
    )

    response = await runtime.handle(admitted)

    assert isinstance(response, RenderedSlackText)
    payload = "".join(response.chunks)
    assert "You asked me to stop" in payload
    assert "If you want to resume" in payload
    assert "run-terminal-race" not in payload
    assert "Run:" not in payload
    assert "slack_user_cancelled" not in payload
    assert leases.abandoned is False
    assert ingress.statuses == [("Ev-terminal-race", "run_cancelled", None)]


@pytest.mark.asyncio
async def test_terminal_before_claim_is_rendered_instead_of_already_working() -> None:
    admitted = _admitted()
    runtime, ingress = _runtime(
        admitted=admitted,
        bundle=_cancelled_bundle(),
        leases=_Leases(claimed=None, exhausted=None),
        sessions=_PlanSessions(()),
    )

    response = await runtime.handle(admitted)

    assert isinstance(response, RenderedSlackText)
    assert "You asked me to stop" in "".join(response.chunks)
    assert ingress.statuses == [("Ev-terminal-race", "run_cancelled", None)]


class _FenceAwareStore:
    def __init__(self, bundle: RunBundle | None = None) -> None:
        self.bundle = bundle or _queued_bundle()
        self.commit_calls = 0

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        assert (task_id, run_id, scope) == (
            self.bundle.task.id,
            self.bundle.run.id,
            self.bundle.run.scope,
        )
        return self.bundle

    async def commit(
        self,
        *,
        task: Task,
        run: Run,
        events: tuple[EventDraft, ...],
        lease_owner: str,
        lease_token: str,
        **_: object,
    ) -> RunBundle:
        assert lease_owner == "runtime-owner"
        assert lease_token == "lease-terminal-race"
        assert events
        self.commit_calls += 1
        self.bundle = RunBundle(thread=self.bundle.thread, task=task, run=run)
        return self.bundle


@pytest.mark.asyncio
async def test_intermediate_store_error_terminalizes_before_single_final_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def oversized_context_commit(**_: object) -> object:
        raise StoreError("event payload exceeds the maximum size")

    monkeypatch.setattr(cli_module, "run_admitted_slack_conversation", oversized_context_commit)
    admitted = _admitted()
    store = _FenceAwareStore(_active_bundle())
    leases = _Leases(claimed=_lease())
    runtime, ingress = _runtime(
        admitted=admitted,
        bundle=store.bundle,
        leases=leases,
        sessions=_PlanSessions((), ()),
    )
    runtime._run_store = store  # type: ignore[attr-defined]

    response = await runtime.handle(admitted)

    assert isinstance(response, RenderedSlackText)
    assert store.bundle.task.status is TaskStatus.FAILED
    assert store.bundle.run.status is RunStatus.FAILED
    assert store.bundle.run.terminal_reason == "runtime_error"
    assert store.commit_calls == 1
    assert leases.abandoned is False
    assert ingress.statuses == [("Ev-terminal-race", "run_failed", None)]
    payload = "".join(response.chunks)
    assert "hit an unexpected problem" in payload
    assert "Please try again" in payload
    assert "run-terminal-race" not in payload


@pytest.mark.asyncio
async def test_thread_context_failure_is_durable_and_conversational(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def unavailable_history(_: SlackMentionJob) -> object:
        raise SlackThreadContextError("slack_thread_history_incomplete")

    admitted = _admitted()
    store = _FenceAwareStore(_active_bundle())
    runtime, ingress = _runtime(
        admitted=admitted,
        bundle=store.bundle,
        leases=_Leases(claimed=_lease()),
        sessions=_PlanSessions((), ()),
    )
    runtime._run_store = store  # type: ignore[attr-defined]
    runtime._history = SimpleNamespace(load=unavailable_history)  # type: ignore[attr-defined]

    response = await runtime.handle(admitted)

    assert isinstance(response, RenderedSlackText)
    assert store.bundle.run.status is RunStatus.FAILED
    assert store.bundle.run.terminal_reason == (
        "context_unavailable:slack_thread_history_incomplete"
    )
    assert ingress.statuses == [("Ev-terminal-race", "run_failed", None)]
    payload = "".join(response.chunks)
    assert "conversation context" in payload
    assert "run-terminal-race" not in payload
    assert "SlackThreadContextError" in caplog.text
    assert "slack_thread_history_incomplete" in caplog.text

    class Outbox:
        def __init__(self) -> None:
            self.intents: dict[tuple[object, ...], SimpleNamespace] = {}

        async def ensure_intent(self, **kwargs: object) -> SimpleNamespace:
            assert store.bundle.run.status is RunStatus.FAILED
            key = (
                kwargs["task_id"],
                kwargs["run_id"],
                kwargs["kind"],
                kwargs["payload_version"],
            )
            return self.intents.setdefault(key, SimpleNamespace(id="intent-runtime-failure"))

        async def load(self, intent_id: str) -> SimpleNamespace:
            assert intent_id == "intent-runtime-failure"
            return SimpleNamespace(receipt_message_ts="final-receipt")

    class Dispatcher:
        calls = 0

        async def dispatch_once(self, client: object, *, intent_id: str) -> DeliveryState:
            del client
            assert intent_id == "intent-runtime-failure"
            self.calls += 1
            return DeliveryState.DELIVERED

    outbox = Outbox()
    dispatcher = Dispatcher()
    processor = SlackJobProcessor(
        client=object(),  # type: ignore[arg-type]
        runtime=runtime,
        outbox=outbox,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    first = await processor._deliver(admitted, kind=DeliveryKind.FINAL, text=response)
    second = await processor._deliver(admitted, kind=DeliveryKind.FINAL, text=response)

    assert first == second == "final-receipt"
    assert len(outbox.intents) == 1
    assert dispatcher.calls == 2


@pytest.mark.asyncio
async def test_retry_exhaustion_terminalizes_once_and_reuses_one_final_intent() -> None:
    admitted = _admitted()
    store = _FenceAwareStore()
    runtime, ingress = _runtime(
        admitted=admitted,
        bundle=store.bundle,
        leases=_Leases(claimed=None, exhausted=_lease(attempt=3)),
        sessions=_PlanSessions((), ()),
    )
    runtime._run_store = store  # type: ignore[attr-defined]

    first = await runtime._finalize_exhausted(admitted, _lease(attempt=3))
    second = await runtime._finalize_exhausted(admitted, _lease(attempt=3))

    assert first == second
    assert store.bundle.task.status is TaskStatus.FAILED
    assert store.bundle.run.status is RunStatus.FAILED
    assert store.bundle.run.terminal_reason == "retry_attempts_exhausted"
    assert store.commit_calls == 2
    assert ingress.statuses == [
        ("Ev-terminal-race", "run_failed", None),
        ("Ev-terminal-race", "run_failed", None),
    ]
    payload = "".join(first.chunks)
    assert "kept hitting the same temporary problem" in payload
    assert "Ask me to retry later" in payload
    assert "run-terminal-race" not in payload
    assert "retry_attempts_exhausted" not in payload

    class Outbox:
        def __init__(self) -> None:
            self.intents: dict[tuple[object, ...], SimpleNamespace] = {}

        async def ensure_intent(self, **kwargs: object) -> SimpleNamespace:
            key = (
                kwargs["task_id"],
                kwargs["run_id"],
                kwargs["kind"],
                kwargs["payload_version"],
            )
            return self.intents.setdefault(key, SimpleNamespace(id="intent-final"))

    class Dispatcher:
        async def dispatch_once(self, client: object, *, intent_id: str) -> DeliveryState:
            del client
            assert intent_id == "intent-final"
            return DeliveryState.RETRY

    outbox = Outbox()
    processor = SlackJobProcessor(
        client=object(),  # type: ignore[arg-type]
        runtime=runtime,
        outbox=outbox,  # type: ignore[arg-type]
        dispatcher=Dispatcher(),  # type: ignore[arg-type]
    )
    await processor._deliver(admitted, kind=DeliveryKind.FINAL, text=first)
    await processor._deliver(admitted, kind=DeliveryKind.FINAL, text=second)

    assert len(outbox.intents) == 1
    assert next(iter(outbox.intents))[2] is DeliveryKind.FINAL


def test_cli_and_worker_terminal_repair_payloads_share_safe_copy() -> None:
    task = SimpleNamespace(id="task-repair")
    run = SimpleNamespace(
        id="run-repair",
        status="failed",
        terminal_reason=(
            "model_gateway_error:Bearer abcdefghijklmnop:postgresql://demo:secret@example.com/leo"
        ),
        final_output=None,
    )

    cli_payload = cli_module._terminal_delivery_payload(task, run)  # type: ignore[arg-type]
    worker_payload = worker_main_module._terminal_delivery_payload(  # type: ignore[arg-type]
        task,
        run,
    )

    assert cli_payload == worker_payload
    assert "reasoning service stopped unexpectedly" in cli_payload
    assert "Ask me to retry" in cli_payload
    assert "run-repair" not in cli_payload
    assert "Run:" not in cli_payload
    assert "model_gateway_error" not in cli_payload
    assert "abcdefghijklmnop" not in cli_payload
    assert "secret@example.com" not in cli_payload
