from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from slack_bolt.async_app import AsyncApp
from slack_bolt.authorization.authorize_result import AuthorizeResult
from slack_bolt.request.async_request import AsyncBoltRequest
from slack_sdk.web.async_client import AsyncWebClient

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackBotPresence,
    SlackContextProjectionSource,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackConversationPolicyRejected,
    SlackExternalProvenance,
    SlackLaunchRef,
    SlackMentionJob,
    SlackScopeResolution,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.integrations.slack.render import RenderedSlackText
from leo.integrations.slack.socket_mode import (
    RUNTIME_DEADLINE_CANCEL_MESSAGE,
    InMemorySlackIngressAdmission,
    SlackJobProcessor,
    _handle_app_mention,
    _handle_message_im,
    _handle_passive_message,
    _thread_root_is_missing,
)
from leo.persistence.outbox import DeliveryKind, DeliveryState
from leo.persistence.slack_ingress import SlackFollowupBusyError


@pytest.mark.asyncio
async def test_concurrent_duplicate_slack_events_are_claimed_once() -> None:
    admission = InMemorySlackIngressAdmission()
    default_scope = _scope("strategy-original")
    outcomes = await asyncio.gather(
        *(
            admission.admit(
                _job("Ev123"),
                default_scope,
                eligibility=_eligibility(),
            )
            for _ in range(20)
        )
    )

    assert sum(outcome is not None for outcome in outcomes) == 1
    assert outcomes.count(None) == 19


@pytest.mark.asyncio
async def test_released_event_can_be_claimed_again() -> None:
    admission = InMemorySlackIngressAdmission(max_entries=2)
    default_scope = _scope("strategy-original")
    assert (
        await admission.admit(_job("Ev123"), default_scope, eligibility=_eligibility()) is not None
    )
    await admission.release("Ev123")
    assert (
        await admission.admit(_job("Ev123"), default_scope, eligibility=_eligibility()) is not None
    )


@pytest.mark.asyncio
async def test_in_memory_admission_uses_current_non_gating_domain_default() -> None:
    admission = InMemorySlackIngressAdmission()
    first = await admission.admit(
        _job("Ev1"), _scope("strategy-original"), eligibility=_eligibility()
    )
    second = await admission.admit(
        _job("Ev2"), _scope("strategy-changed"), eligibility=_eligibility()
    )

    assert first is not None
    assert second is not None
    assert first.resolution.scope == _scope("strategy-original")
    assert first.resolution.provisioned is False
    assert second.resolution.scope == _scope("strategy-changed")
    assert second.resolution.provisioned is False


@pytest.mark.asyncio
async def test_only_unknown_conversation_authority_is_rejected_before_event_claim() -> None:
    admission = InMemorySlackIngressAdmission()

    with pytest.raises(SlackConversationPolicyRejected, match="not eligible"):
        await admission.admit(
            _job("Ev-ineligible"),
            _scope("strategy-original"),
            eligibility=SlackConversationEligibility(
                kind=SlackConversationKind.UNKNOWN,
                provenance="unknown",
            ),
        )

    admitted = await admission.admit(
        _job("Ev-ineligible"),
        _scope("strategy-original"),
        eligibility=_eligibility(),
    )
    assert admitted is not None


class _Runtime:
    async def handle(self, admitted: AdmittedSlackMention) -> str:
        return f"handled:{admitted.job.event_id}"


class _RecordingClient:
    def __init__(self) -> None:
        self.fail_next_post = True
        self.posts: list[dict[str, str]] = []
        self.updates: list[dict[str, str]] = []

    async def chat_postMessage(self, **kwargs: str) -> dict[str, str]:
        if self.fail_next_post:
            self.fail_next_post = False
            raise RuntimeError("simulated Slack post failure")
        self.posts.append(kwargs)
        return {"ts": f"reply-{len(self.posts)}"}

    async def chat_update(self, **kwargs: str) -> dict[str, str]:
        self.updates.append(kwargs)
        return {"ts": kwargs["ts"]}


def _job(event_id: str) -> SlackMentionJob:
    return SlackMentionJob(
        event_id=event_id,
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="1.2",
        thread_root_ts="1.0",
        conversation_key="slack:T1:C1:1.0",
        prompt="quote NVDA",
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=("C1",),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
        context_access_hash=build_context_access_hash(
            team_id="T1",
            user_id="U1",
            channel_id="C1",
            context_conversation_ids=("C1",),
        ),
    )


def _scope(strategy_id: str) -> ScopeKey:
    return ScopeKey(organization_id="org-demo", strategy_id=strategy_id)


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _admitted(event_id: str) -> AdmittedSlackMention:
    return AdmittedSlackMention(
        job=_job(event_id),
        resolution=SlackScopeResolution(
            scope=_scope("strategy-original"),
            mapping_version=1,
            provisioned=event_id == "Ev1",
        ),
    )


def test_durable_enqueue_treats_queue_pressure_as_a_recoverable_wake_up_miss() -> None:
    processor = SlackJobProcessor(
        client=_RecordingClient(),
        runtime=_Runtime(),
        queue_size=1,
    )

    assert processor.enqueue(_admitted("Ev-queue-first")) is True
    assert processor.enqueue(_admitted("Ev-queue-full")) is False
    assert processor.queue.qsize() == 1
    assert processor.queue.get_nowait().job.event_id == "Ev-queue-first"


@pytest.mark.asyncio
async def test_slack_worker_survives_post_failure_and_keeps_thread_routing() -> None:
    client = _RecordingClient()
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=_Runtime(),
        runtime_timeout_seconds=1,
        slack_timeout_seconds=1,
    )
    worker = asyncio.create_task(processor.run())
    try:
        await processor.queue.put(_admitted("Ev1"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
        assert not worker.done()

        await processor.queue.put(_admitted("Ev2"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
        assert not worker.done()
        assert all(post["thread_ts"] == "1.0" for post in client.posts)
        assert client.updates[-1]["text"] == "handled:Ev2"
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)


@pytest.mark.asyncio
async def test_processor_preserves_harness_rendered_links_and_chunks() -> None:
    class RenderedRuntime:
        async def handle(self, admitted: AdmittedSlackMention) -> RenderedSlackText:
            del admitted
            return RenderedSlackText(
                version=1,
                chunks=(
                    "Fact\n  Source: <https://example.com/evidence|primary evidence>",
                    "Additional verified context.",
                ),
            )

    client = _RecordingClient()
    client.fail_next_post = False
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=RenderedRuntime(),
        runtime_timeout_seconds=1,
        slack_timeout_seconds=1,
    )
    worker = asyncio.create_task(processor.run())
    try:
        await processor.queue.put(_admitted("Ev-rendered"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert client.updates[-1]["text"].endswith("<https://example.com/evidence|primary evidence>")
    assert client.posts[-1]["text"] == "Additional verified context."


@pytest.mark.asyncio
async def test_durable_multi_part_delivery_materializes_all_v2_parts_before_dispatch() -> None:
    class Outbox:
        def __init__(self) -> None:
            self.parts: list[tuple[int, str]] = []

        async def ensure_intent(self, **kwargs: object) -> SimpleNamespace:
            payload_version = int(kwargs["payload_version"])  # type: ignore[arg-type]
            payload = str(kwargs["payload"])
            self.parts.append((payload_version, payload))
            return SimpleNamespace(id=f"intent-{payload_version}")

    outbox = Outbox()

    class Dispatcher:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def dispatch_once(self, client: object, *, intent_id: str) -> DeliveryState:
            del client
            assert outbox.parts == [(2000, "part one"), (2001, "part two")]
            self.calls.append(intent_id)
            return DeliveryState.RETRY

    dispatcher = Dispatcher()
    processor = SlackJobProcessor(
        client=_RecordingClient(),  # type: ignore[arg-type]
        runtime=_Runtime(),
        outbox=outbox,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
    )
    admitted = replace(
        _admitted("Ev-rendered-durable"),
        launch=SlackLaunchRef(thread_id="thread-1", task_id="task-1", run_id="run-1"),
    )

    await processor._deliver(
        admitted,
        kind=DeliveryKind.FINAL,
        text=RenderedSlackText(version=2, chunks=("part one", "part two")),
    )

    assert dispatcher.calls == ["intent-2000", "intent-2001"]


@pytest.mark.asyncio
async def test_processor_attempts_fifo_recovery_after_each_terminal_turn() -> None:
    class Recoverer:
        def __init__(self) -> None:
            self.calls = 0

        async def recover(self) -> tuple[AdmittedSlackMention, ...]:
            self.calls += 1
            return ()

    client = _RecordingClient()
    client.fail_next_post = False
    recoverer = Recoverer()
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=_Runtime(),
        runtime_timeout_seconds=1,
        slack_timeout_seconds=1,
        launch_recoverer=recoverer,  # type: ignore[arg-type]
    )
    worker = asyncio.create_task(processor.run())
    try:
        await processor.queue.put(_admitted("Ev-fifo"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert recoverer.calls == 1


@pytest.mark.asyncio
async def test_queue_pressure_recovery_preserves_fifo_without_duplicate_wakeups() -> None:
    pending = {
        admitted.job.event_id: admitted
        for admitted in (
            _admitted("Ev-fifo-a"),
            _admitted("Ev-fifo-b"),
            _admitted("Ev-fifo-c"),
        )
    }
    completed = asyncio.Event()

    class Recoverer:
        async def recover(self) -> tuple[AdmittedSlackMention, ...]:
            return tuple(pending.values())

    class Runtime:
        def __init__(self) -> None:
            self.seen: list[str] = []

        async def handle(self, admitted: AdmittedSlackMention) -> str:
            event_id = admitted.job.event_id
            self.seen.append(event_id)
            pending.pop(event_id)
            if len(self.seen) == 3:
                completed.set()
            return f"handled:{event_id}"

    client = _RecordingClient()
    client.fail_next_post = False
    runtime = Runtime()
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=runtime,
        queue_size=2,
        runtime_timeout_seconds=1,
        slack_timeout_seconds=1,
        launch_recoverer=Recoverer(),
    )
    assert processor.enqueue(pending["Ev-fifo-a"]) is True
    assert processor.enqueue(pending["Ev-fifo-b"]) is True
    assert processor.enqueue(pending["Ev-fifo-c"]) is False

    worker = asyncio.create_task(processor.run())
    try:
        await asyncio.wait_for(completed.wait(), timeout=2)
        await asyncio.wait_for(processor.queue.join(), timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert runtime.seen == ["Ev-fifo-a", "Ev-fifo-b", "Ev-fifo-c"]


@pytest.mark.asyncio
async def test_runtime_deadline_response_is_returned_only_after_durable_reconciliation() -> None:
    reconciled = asyncio.Event()

    class Runtime:
        async def handle(self, admitted: AdmittedSlackMention) -> str:
            del admitted
            try:
                await asyncio.Future()
            except asyncio.CancelledError as exc:
                assert exc.args == (RUNTIME_DEADLINE_CANCEL_MESSAGE,)
                reconciled.set()
                return "Leo stopped safely with durable status `timed_out`."

    client = _RecordingClient()
    client.fail_next_post = False
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=Runtime(),
        runtime_timeout_seconds=0.01,
        slack_timeout_seconds=1,
    )
    worker = asyncio.create_task(processor.run())
    try:
        processor.enqueue(_admitted("Ev-timeout-durable"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    assert reconciled.is_set()
    assert client.updates[-1]["text"] == "Leo stopped safely with durable status `timed_out`."


@pytest.mark.asyncio
async def test_processor_runtime_exception_uses_safe_conversational_fallback() -> None:
    class Runtime:
        async def handle(self, admitted: AdmittedSlackMention) -> str:
            del admitted
            raise RuntimeError(
                "model_gateway_error:Bearer abcdefghijklmnop:"
                "postgresql://demo:secret@example.com/leo"
            )

    client = _RecordingClient()
    client.fail_next_post = False
    processor = SlackJobProcessor(
        client=client,  # type: ignore[arg-type]
        runtime=Runtime(),
        runtime_timeout_seconds=1,
        slack_timeout_seconds=1,
    )
    worker = asyncio.create_task(processor.run())
    try:
        processor.enqueue(_admitted("Ev-runtime-failed"))
        await asyncio.wait_for(processor.queue.join(), timeout=2)
    finally:
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)

    payload = client.updates[-1]["text"]
    assert "hit an unexpected problem" in payload
    assert "Please try again" in payload
    assert "Ev-runtime-failed" not in payload
    assert "Run:" not in payload
    assert "model_gateway_error" not in payload
    assert "abcdefghijklmnop" not in payload
    assert "secret@example.com" not in payload


class _SlowMetadataClient(AsyncWebClient):
    def __init__(self) -> None:
        super().__init__()

    async def auth_test(self, **kwargs: str) -> dict[str, str]:
        del kwargs
        return {"ok": "true", "team_id": "T1", "user_id": "UBOT"}

    async def conversations_info(self, **kwargs: str) -> dict[str, object]:
        del kwargs
        await asyncio.sleep(0.1)
        return SimpleNamespace(data={"ok": True, "channel": {"id": "C1", "is_channel": True}})  # type: ignore[return-value]


def _event_body() -> dict[str, object]:
    return {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "Ev-ack",
        "event": {
            "type": "app_mention",
            "user": "U1",
            "text": "<@UBOT> ping",
            "ts": "1.2",
            "channel": "C1",
            "channel_type": "channel",
        },
    }


@pytest.mark.asyncio
async def test_socket_mode_auto_ack_is_under_three_seconds_with_slow_admission() -> None:
    client = _SlowMetadataClient()
    admission = InMemorySlackIngressAdmission()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)
    callback_finished = asyncio.Event()

    async def on_mention(body: dict[str, object]) -> None:
        try:
            await _handle_app_mention(
                body,
                client=client,
                expected_team_id="T1",
                bot_user_id="UBOT",
                default_scope=_scope("strategy-original"),
                admission=admission,
                processor=processor,
                fatal_errors=fatal_errors,
                admission_timeout_seconds=1,
            )
        finally:
            callback_finished.set()

    async def authorize(**kwargs: object) -> AuthorizeResult:
        del kwargs
        return AuthorizeResult(
            enterprise_id=None,
            team_id="T1",
            bot_user_id="UBOT",
            bot_token="xoxb-test",
        )

    app = AsyncApp(
        client=client,
        process_before_response=False,
        request_verification_enabled=False,
        authorize=authorize,
    )
    app.event("app_mention")(on_mention)

    started = time.perf_counter()
    response = await app.async_dispatch(AsyncBoltRequest(body=_event_body(), mode="socket_mode"))
    acknowledgement_seconds = time.perf_counter() - started

    assert response.status == 200
    assert acknowledgement_seconds < 3
    await asyncio.wait_for(callback_finished.wait(), timeout=2)
    assert processor.queue.qsize() == 1


@pytest.mark.asyncio
async def test_stale_thread_root_is_rejected_before_admission() -> None:
    class Client:
        async def conversations_replies(self, **kwargs: str) -> dict[str, object]:
            assert kwargs == {"channel": "C1", "ts": "1.0", "limit": 1}
            return {"ok": True, "messages": []}

    assert await _thread_root_is_missing(Client(), _job("Ev-stale")) is True  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_busy_followup_gets_safe_fifo_acknowledgement_without_parallel_work() -> None:
    class Client:
        def __init__(self) -> None:
            self.posts: list[dict[str, str]] = []

        async def conversations_info(self, **kwargs: str) -> dict[str, object]:
            del kwargs
            return {
                "ok": True,
                "channel": {"id": "C1", "is_channel": True, "is_member": True},
            }

        async def chat_postMessage(self, **kwargs: str) -> dict[str, str]:
            self.posts.append(kwargs)
            return {"ok": "true", "ts": "busy-receipt"}

    class BusyPreparer:
        async def prepare(self, admitted: AdmittedSlackMention) -> AdmittedSlackMention:
            del admitted
            raise SlackFollowupBusyError("thread_task_active")

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    await _handle_app_mention(
        _event_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=asyncio.Queue(maxsize=1),
        admission_timeout_seconds=1,
        launch_preparer=BusyPreparer(),
    )

    assert processor.queue.qsize() == 0
    assert client.posts == []


def _message_im_body(*, event_update: dict[str, object] | None = None) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "message",
        "channel_type": "im",
        "user": "U1",
        "text": "Help me reason through this",
        "ts": "2.0",
        "channel": "D1",
    }
    if event_update:
        event.update(event_update)
    return {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": "Ev-im",
        "event": event,
    }


@pytest.mark.asyncio
async def test_message_im_paginates_and_carries_exact_sorted_membership_projection() -> None:
    class Client:
        def __init__(self) -> None:
            self.requests: list[dict[str, object]] = []

        async def users_conversations(self, **kwargs: object) -> dict[str, object]:
            self.requests.append(kwargs)
            if "cursor" not in kwargs:
                return {
                    "ok": True,
                    "channels": [
                        {"id": "G2", "is_group": True, "is_mpim": True},
                        {"id": "C9", "is_channel": True, "is_archived": True},
                    ],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            return {
                "ok": True,
                "channels": [
                    {"id": "C1", "is_channel": True},
                    {"id": "D-other", "is_im": True},
                    {"id": "C1", "is_channel": True},
                ],
                "response_metadata": {"next_cursor": ""},
            }

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    await _handle_message_im(
        _message_im_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=asyncio.Queue(maxsize=1),
        admission_timeout_seconds=1,
    )

    admitted = processor.queue.get_nowait()
    assert admitted.job.context_conversation_ids == ("C1", "D1", "G2")
    assert (
        admitted.job.context_projection_source
        is SlackContextProjectionSource.DM_MEMBERSHIP_INTERSECTION
    )
    assert admitted.job.context_access_hash == build_context_access_hash(
        team_id="T1",
        user_id="U1",
        channel_id="D1",
        context_conversation_ids=("C1", "D1", "G2"),
    )
    assert admitted.job.conversation_kind is SlackConversationKind.DM
    assert admitted.job.trigger_kind is SlackTriggerKind.MESSAGE_IM
    assert client.requests == [
        {
            "user": "U1",
            "types": "public_channel,private_channel,mpim",
            "exclude_archived": True,
            "limit": 200,
        },
        {
            "user": "U1",
            "types": "public_channel,private_channel,mpim",
            "exclude_archived": True,
            "limit": 200,
            "cursor": "page-2",
        },
    ]


@pytest.mark.asyncio
async def test_dm_membership_lookup_failure_falls_back_to_dm_only_without_rejecting() -> None:
    class Client:
        async def users_conversations(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            raise RuntimeError("Slack unavailable")

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)
    await _handle_message_im(
        _message_im_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=fatal_errors,
        admission_timeout_seconds=1,
    )

    job = processor.queue.get_nowait().job
    assert job.context_conversation_ids == ("D1",)
    assert job.context_projection_source is SlackContextProjectionSource.DM_ONLY_FALLBACK
    assert fatal_errors.empty()


def _passive_body(
    *,
    channel_type: str = "channel",
    event_id: str = "Ev-passive",
    event_update: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "message",
        "channel_type": channel_type,
        "user": "U1",
        "text": "Passive context only",
        "ts": "3.0",
        "thread_ts": "2.0",
        "channel": "G1" if channel_type == "mpim" else "C1",
    }
    if event_update is not None:
        event.update(event_update)
    return {
        "type": "event_callback",
        "team_id": "T1",
        "event_id": event_id,
        "api_app_id": "A-LEO",
        "event": event,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("channel_type", ["channel", "group", "mpim"])
async def test_passive_messages_persist_without_launch_queue_or_slack_reply(
    channel_type: str,
) -> None:
    admission = InMemorySlackIngressAdmission()
    client = _RecordingClient()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)

    await _handle_passive_message(
        _passive_body(channel_type=channel_type),
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        sink=admission,
        fatal_errors=fatal_errors,
        persistence_timeout_seconds=1,
    )

    assert len(admission.passive_messages) == 1
    assert admission.passive_messages[0].thread_root_ts == "2.0"
    assert processor.queue.empty()
    assert client.posts == []
    assert client.updates == []
    assert fatal_errors.empty()


@pytest.mark.asyncio
async def test_passive_leo_bot_message_without_user_persists_without_launch_or_reply() -> None:
    admission = InMemorySlackIngressAdmission()
    client = _RecordingClient()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)

    await _handle_passive_message(
        _passive_body(
            event_id="Ev-passive-leo",
            event_update={
                "user": None,
                "bot_id": "B-LEO",
                "app_id": "A-LEO",
                "subtype": "bot_message",
                "text": "Leo's delivered thread answer.",
            },
        ),
        expected_team_id="T1",
        bot_user_id="UBOT",
        bot_id="B-LEO",
        default_scope=_scope("strategy-original"),
        sink=admission,
        fatal_errors=fatal_errors,
        persistence_timeout_seconds=1,
    )

    assert len(admission.passive_messages) == 1
    assert admission.passive_messages[0].role.value == "assistant"
    assert processor.queue.empty()
    assert client.posts == []
    assert client.updates == []
    assert fatal_errors.empty()


@pytest.mark.asyncio
async def test_duplicate_passive_callback_is_idempotent_and_never_launches() -> None:
    admission = InMemorySlackIngressAdmission()
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)
    body = _passive_body()

    for _ in range(2):
        await _handle_passive_message(
            body,
            expected_team_id="T1",
            bot_user_id="UBOT",
            default_scope=_scope("strategy-original"),
            sink=admission,
            fatal_errors=fatal_errors,
            persistence_timeout_seconds=1,
        )

    assert len(admission.passive_messages) == 1
    assert fatal_errors.empty()


@pytest.mark.asyncio
async def test_passive_persistence_failure_is_fatal_without_a_slack_reply() -> None:
    class FailingSink:
        async def record_passive_message(self, *args: object) -> None:
            del args
            raise RuntimeError("synthetic passive persistence failure")

    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)

    await _handle_passive_message(
        _passive_body(),
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        sink=FailingSink(),  # type: ignore[arg-type]
        fatal_errors=fatal_errors,
        persistence_timeout_seconds=1,
    )

    error = fatal_errors.get_nowait()
    assert isinstance(error, RuntimeError)
    assert str(error) == "synthetic passive persistence failure"


@pytest.mark.asyncio
async def test_app_mention_metadata_failure_uses_event_authority() -> None:
    class Client:
        def __init__(self) -> None:
            self.posts: list[dict[str, str]] = []

        async def conversations_info(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            raise RuntimeError("Slack unavailable")

        async def chat_postMessage(self, **kwargs: str) -> dict[str, str]:
            self.posts.append(kwargs)
            return {"ok": "true", "ts": "unexpected"}

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)
    await _handle_app_mention(
        _event_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=fatal_errors,
        admission_timeout_seconds=1,
    )

    admitted = processor.queue.get_nowait()
    assert admitted.job.conversation_kind is SlackConversationKind.ORDINARY_INTERNAL
    assert admitted.job.context_conversation_ids == ("C1",)
    assert admitted.job.context_projection_source is SlackContextProjectionSource.EXACT_DESTINATION
    assert admitted.job.conversation_authority_source == "slack_event"
    assert admitted.job.bot_presence is SlackBotPresence.PRESENT
    assert admitted.job.external_provenance is SlackExternalProvenance.UNKNOWN
    assert client.posts == []
    assert fatal_errors.empty()


@pytest.mark.asyncio
async def test_app_mention_real_mpim_metadata_overrides_generic_event_channel_type() -> None:
    class Client:
        async def conversations_info(self, **kwargs: object) -> dict[str, object]:
            assert kwargs == {"channel": "C1"}
            return {
                "ok": True,
                "channel": {
                    "id": "C1",
                    "is_channel": True,
                    "is_mpim": True,
                    "is_private": True,
                    "is_member": True,
                },
            }

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)

    await _handle_app_mention(
        _event_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=fatal_errors,
        admission_timeout_seconds=1,
    )

    admitted = processor.queue.get_nowait()
    assert admitted.job.conversation_kind is SlackConversationKind.MPIM
    assert admitted.job.context_conversation_ids == ("C1",)
    assert admitted.job.context_projection_source is SlackContextProjectionSource.EXACT_DESTINATION
    assert admitted.job.conversation_authority_source == "slack_conversations_info"
    assert admitted.job.bot_presence is SlackBotPresence.PRESENT
    assert admitted.job.external_provenance is SlackExternalProvenance.NOT_APPLICABLE
    assert fatal_errors.empty()


@pytest.mark.asyncio
async def test_metadata_failure_never_infers_conversation_kind_from_channel_id_prefix() -> None:
    class Client:
        async def conversations_info(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            raise RuntimeError("Slack unavailable")

    body = _event_body()
    event = body["event"]
    assert isinstance(event, dict)
    event.pop("channel_type")
    event["channel"] = "D-prefix-is-not-authority"
    processor = SlackJobProcessor(client=Client(), runtime=_Runtime())  # type: ignore[arg-type]

    await _handle_app_mention(
        body,
        client=Client(),  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=asyncio.Queue(maxsize=1),
        admission_timeout_seconds=1,
    )

    assert processor.queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata",
    [
        {"is_member": False},
        {"is_member": True, "is_archived": True},
    ],
)
async def test_absent_or_archived_conversation_never_launches_work(
    metadata: dict[str, object],
) -> None:
    class Client:
        async def conversations_info(self, **kwargs: object) -> dict[str, object]:
            del kwargs
            return {
                "ok": True,
                "channel": {"id": "C1", "is_channel": True, **metadata},
            }

    client = Client()
    processor = SlackJobProcessor(client=client, runtime=_Runtime())  # type: ignore[arg-type]

    await _handle_app_mention(
        _event_body(),
        client=client,  # type: ignore[arg-type]
        expected_team_id="T1",
        bot_user_id="UBOT",
        default_scope=_scope("strategy-original"),
        admission=InMemorySlackIngressAdmission(),
        processor=processor,
        fatal_errors=asyncio.Queue(maxsize=1),
        admission_timeout_seconds=1,
    )

    assert processor.queue.empty()
