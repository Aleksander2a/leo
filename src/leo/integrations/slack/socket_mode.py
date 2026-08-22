"""Local Slack Socket Mode transport smoke, isolated from Leo's runtime internals."""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Protocol

from pydantic import ValidationError
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

from leo.config import Settings
from leo.harness.models import RunStatus, ScopeKey
from leo.health import SLACK_SOCKET_READINESS, SlackSocketReadinessRegistry
from leo.integrations.slack.cancellation import SlackCancellationResult
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackAdmissionPolicyRejected,
    SlackBotPresence,
    SlackContextProjectionSource,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackConversationPolicyRejected,
    SlackEventRejected,
    SlackExternalProvenance,
    SlackMentionJob,
    SlackPassiveMessage,
    SlackScopeResolution,
    SlackTriggerKind,
    build_context_access_hash,
    classify_slack_conversation,
    normalize_app_mention,
    normalize_message_im,
    normalize_passive_message,
)
from leo.integrations.slack.render import (
    RenderedSlackText,
    SlackTerminalResult,
    render_slack_text,
    render_terminal_result,
)
from leo.persistence.outbox import (
    DeliveryKind,
    DeliveryState,
    PostgresDeliveryOutbox,
    SlackOutboxDispatcher,
)
from leo.persistence.slack_ingress import SlackFollowupBusyError

LOGGER = logging.getLogger(__name__)
RUNTIME_DEADLINE_CANCEL_MESSAGE = "slack_runtime_deadline_exceeded"


class SlackJobRuntime(Protocol):
    async def handle(self, admitted: AdmittedSlackMention) -> str | RenderedSlackText: ...


class SlackIngressAdmission(Protocol):
    async def preflight(self) -> None: ...

    async def admit(
        self,
        job: SlackMentionJob,
        default_scope: ScopeKey,
        *,
        eligibility: SlackConversationEligibility,
    ) -> AdmittedSlackMention | None: ...

    async def release(self, event_id: str) -> None: ...

    async def record_passive_message(
        self,
        message: SlackPassiveMessage,
        default_scope: ScopeKey,
    ) -> None: ...


class SlackLaunchPreparer(Protocol):
    async def prepare(self, admitted: AdmittedSlackMention) -> AdmittedSlackMention: ...

    async def recover(self) -> tuple[AdmittedSlackMention, ...]: ...


class SlackCancellationHandler(Protocol):
    def accepts(self, prompt: str) -> bool: ...

    async def handle(
        self,
        admitted: AdmittedSlackMention,
        launch_preparer: SlackLaunchPreparer,
    ) -> SlackCancellationResult: ...

    async def recover(
        self,
        launch_preparer: SlackLaunchPreparer,
        *,
        limit: int = 100,
    ) -> tuple[SlackCancellationResult, ...]: ...


class StaticTransportRuntime:
    async def handle(self, admitted: AdmittedSlackMention) -> str:
        job = admitted.job
        return (
            "Leo transport is working. "
            f"Received your mention in conversation `{job.conversation_key}`."
        )


class InMemorySlackIngressAdmission:
    """Atomic process-local admission for transport and deterministic Slack smokes only."""

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._events: OrderedDict[str, SlackMentionJob] = OrderedDict()
        self._passive_messages: OrderedDict[str, SlackPassiveMessage] = OrderedDict()
        self._max_entries = max_entries
        self._lock = asyncio.Lock()

    async def preflight(self) -> None:
        return None

    async def admit(
        self,
        job: SlackMentionJob,
        default_scope: ScopeKey,
        *,
        eligibility: SlackConversationEligibility,
    ) -> AdmittedSlackMention | None:
        async with self._lock:
            if not eligibility.admissible or eligibility.kind is not job.conversation_kind:
                raise SlackConversationPolicyRejected(eligibility)
            existing = self._events.get(job.event_id)
            if existing is not None:
                if existing != job:
                    raise RuntimeError("Slack event ID was reused with a different envelope")
                return None

            resolution = SlackScopeResolution(
                scope=default_scope,
                mapping_version=1,
                provisioned=False,
            )

            self._events[job.event_id] = job
            while len(self._events) > self._max_entries:
                self._events.popitem(last=False)
            return AdmittedSlackMention(job=job, resolution=resolution)

    async def release(self, event_id: str) -> None:
        async with self._lock:
            self._events.pop(event_id, None)

    async def record_passive_message(
        self,
        message: SlackPassiveMessage,
        default_scope: ScopeKey,
    ) -> None:
        del default_scope
        async with self._lock:
            existing = self._passive_messages.get(message.event_id)
            if existing is not None and existing != message:
                raise RuntimeError("Slack passive event ID was reused with a different envelope")
            self._passive_messages[message.event_id] = message
            while len(self._passive_messages) > self._max_entries:
                self._passive_messages.popitem(last=False)

    @property
    def passive_messages(self) -> tuple[SlackPassiveMessage, ...]:
        return tuple(self._passive_messages.values())


class SlackJobProcessor:
    def __init__(
        self,
        *,
        client: AsyncWebClient,
        runtime: SlackJobRuntime,
        queue_size: int = 100,
        runtime_timeout_seconds: float = 90.0,
        slack_timeout_seconds: float = 15.0,
        outbox: PostgresDeliveryOutbox | None = None,
        dispatcher: SlackOutboxDispatcher | None = None,
        launch_recoverer: SlackLaunchPreparer | None = None,
    ) -> None:
        self._client = client
        self._runtime = runtime
        self._runtime_timeout_seconds = runtime_timeout_seconds
        self._slack_timeout_seconds = slack_timeout_seconds
        if (outbox is None) != (dispatcher is None):
            raise ValueError("outbox and dispatcher must be supplied together")
        self._outbox = outbox
        self._dispatcher = dispatcher
        self._launch_recoverer = launch_recoverer
        self.queue: asyncio.Queue[AdmittedSlackMention] = asyncio.Queue(maxsize=queue_size)
        self._scheduled_wakeups: set[str] = set()

    def enqueue(self, admitted: AdmittedSlackMention) -> bool:
        """Signal one committed launch without making the local queue authoritative.

        Callers must invoke this only after the durable admission/launch transaction has
        committed.  A full queue is a recoverable wake-up miss: the persisted launch is
        intentionally left untouched for the startup scanner and durable worker claims.
        """

        wakeup_key = _wakeup_key(admitted)
        if wakeup_key in self._scheduled_wakeups:
            return True
        try:
            self.queue.put_nowait(admitted)
        except asyncio.QueueFull:
            LOGGER.error(
                "Slack job queue is full; durable launch remains queued",
                extra={"event_id": admitted.job.event_id},
            )
            return False
        self._scheduled_wakeups.add(wakeup_key)
        return True

    async def recover_pending(self) -> None:
        """Re-signal durable launches while deduplicating process-local wake-up hints."""

        if self._launch_recoverer is None:
            return
        for admitted in await self._launch_recoverer.recover():
            self.enqueue(admitted)

    async def run(self) -> None:
        while True:
            admitted = await self.queue.get()
            job = admitted.job
            wakeup_key = _wakeup_key(admitted)
            self._scheduled_wakeups.add(wakeup_key)
            try:
                progress_ts = await self._deliver(
                    admitted,
                    kind=DeliveryKind.PROGRESS,
                    text="Leo is working…",
                )
                try:
                    response = await _run_runtime_with_deadline(
                        self._runtime,
                        admitted,
                        timeout_seconds=self._runtime_timeout_seconds,
                    )
                except TimeoutError:
                    LOGGER.error("Slack job runtime timed out", extra={"event_id": job.event_id})
                    response = render_terminal_result(
                        SlackTerminalResult(
                            run_id=_run_reference(admitted),
                            status=RunStatus.TIMED_OUT,
                            terminal_reason=RUNTIME_DEADLINE_CANCEL_MESSAGE,
                        )
                    )
                except Exception:
                    LOGGER.exception("Slack job runtime failed", extra={"event_id": job.event_id})
                    response = render_terminal_result(
                        SlackTerminalResult(
                            run_id=_run_reference(admitted),
                            status=RunStatus.FAILED,
                            terminal_reason="runtime_error",
                        )
                    )
                if self._outbox is None:
                    rendered = (
                        response
                        if isinstance(response, RenderedSlackText)
                        else render_slack_text(response)
                    )
                    try:
                        if progress_ts is not None:
                            await asyncio.wait_for(
                                self._client.chat_update(
                                    channel=job.channel_id,
                                    ts=progress_ts,
                                    text=rendered.chunks[0],
                                ),
                                timeout=self._slack_timeout_seconds,
                            )
                            for chunk in rendered.chunks[1:]:
                                await asyncio.wait_for(
                                    self._client.chat_postMessage(
                                        channel=job.channel_id,
                                        thread_ts=job.thread_root_ts,
                                        text=chunk,
                                    ),
                                    timeout=self._slack_timeout_seconds,
                                )
                        else:
                            for chunk in rendered.chunks:
                                await asyncio.wait_for(
                                    self._client.chat_postMessage(
                                        channel=job.channel_id,
                                        thread_ts=job.thread_root_ts,
                                        text=chunk,
                                    ),
                                    timeout=self._slack_timeout_seconds,
                                )
                    except Exception:
                        LOGGER.exception(
                            "Slack final response failed", extra={"event_id": job.event_id}
                        )
                else:
                    await self._deliver(admitted, kind=DeliveryKind.FINAL, text=response)
            finally:
                self.queue.task_done()
                self._scheduled_wakeups.discard(wakeup_key)
                if self._launch_recoverer is not None:
                    try:
                        await self.recover_pending()
                    except Exception:
                        LOGGER.exception("Slack pending follow-up recovery failed")

    async def _deliver(
        self,
        admitted: AdmittedSlackMention,
        *,
        kind: DeliveryKind,
        text: str | RenderedSlackText,
    ) -> str | None:
        if self._outbox is None or self._dispatcher is None:
            rendered = text if isinstance(text, RenderedSlackText) else render_slack_text(text)
            direct_receipt: str | None = None
            for chunk in rendered.chunks:
                receipt = await self._direct_post(admitted, chunk)
                if direct_receipt is None:
                    direct_receipt = receipt
            return direct_receipt
        if admitted.launch is None:
            LOGGER.error(
                "durable Slack delivery has no launch",
                extra={"event_id": admitted.job.event_id},
            )
            return None
        rendered = text if isinstance(text, RenderedSlackText) else render_slack_text(text)
        first_receipt: str | None = None
        try:
            # Materialize every deterministic part before the first external effect.
            # A crash while Slack is accepting part 1 therefore leaves later parts as
            # durable pending intents for startup dispatch instead of losing them.
            intents = []
            for index, chunk in enumerate(rendered.chunks):
                intents.append(
                    await self._outbox.ensure_intent(
                        task_id=admitted.launch.task_id,
                        run_id=admitted.launch.run_id,
                        ingress_event_id=admitted.job.event_id,
                        kind=kind,
                        payload_version=rendered.version * 1000 + index,
                        payload=chunk,
                    )
                )
            for intent in intents:
                state = await asyncio.wait_for(
                    self._dispatcher.dispatch_once(self._client, intent_id=intent.id),
                    timeout=self._slack_timeout_seconds,
                )
                if state is DeliveryState.DELIVERED:
                    delivered = await self._outbox.load(intent.id)
                    if first_receipt is None:
                        first_receipt = delivered.receipt_message_ts
                elif state is DeliveryState.UNKNOWN_EFFECT:
                    LOGGER.error(
                        "Slack delivery has unknown effect",
                        extra={"event_id": admitted.job.event_id, "intent_id": intent.id},
                    )
        except Exception:
            LOGGER.exception(
                "durable Slack delivery failed",
                extra={"event_id": admitted.job.event_id, "kind": kind.value},
            )
        return first_receipt

    async def deliver_control(self, result: SlackCancellationResult) -> None:
        """Deliver a terminalized transport-control result through the outbox."""

        await self._deliver(
            result.admitted,
            kind=DeliveryKind.FINAL,
            text=result.message,
        )

    async def _direct_post(
        self,
        admitted: AdmittedSlackMention,
        text: str,
    ) -> str | None:
        try:
            response = await asyncio.wait_for(
                self._client.chat_postMessage(
                    channel=admitted.job.channel_id,
                    thread_ts=admitted.job.thread_root_ts,
                    text=text,
                ),
                timeout=self._slack_timeout_seconds,
            )
        except Exception:
            LOGGER.exception(
                "Slack direct delivery failed",
                extra={"event_id": admitted.job.event_id},
            )
            return None
        return str(response.get("ts") or "") or None


async def _handle_app_mention(
    body: dict[str, object],
    *,
    client: AsyncWebClient,
    expected_team_id: str,
    bot_user_id: str,
    default_scope: ScopeKey,
    admission: SlackIngressAdmission,
    processor: SlackJobProcessor,
    fatal_errors: asyncio.Queue[BaseException],
    admission_timeout_seconds: float,
    launch_preparer: SlackLaunchPreparer | None = None,
    cancellation_handler: SlackCancellationHandler | None = None,
) -> None:
    try:
        job = normalize_app_mention(
            body,
            expected_team_id=expected_team_id,
            bot_user_id=bot_user_id,
        )
    except (SlackEventRejected, ValidationError) as exc:
        LOGGER.warning("Slack event rejected: %s", exc)
        return
    if job is None:
        return
    await _admit_and_enqueue(
        job,
        client=client,
        default_scope=default_scope,
        admission=admission,
        processor=processor,
        fatal_errors=fatal_errors,
        admission_timeout_seconds=admission_timeout_seconds,
        launch_preparer=launch_preparer,
        cancellation_handler=cancellation_handler,
    )


async def _handle_message_im(
    body: dict[str, object],
    *,
    client: AsyncWebClient,
    expected_team_id: str,
    bot_user_id: str,
    default_scope: ScopeKey,
    admission: SlackIngressAdmission,
    processor: SlackJobProcessor,
    fatal_errors: asyncio.Queue[BaseException],
    admission_timeout_seconds: float,
    launch_preparer: SlackLaunchPreparer | None = None,
    cancellation_handler: SlackCancellationHandler | None = None,
) -> None:
    try:
        job = normalize_message_im(
            body,
            expected_team_id=expected_team_id,
            bot_user_id=bot_user_id,
        )
    except (SlackEventRejected, ValidationError) as exc:
        LOGGER.warning("Slack DM event rejected: %s", exc)
        return
    if job is None:
        return
    await _admit_and_enqueue(
        job,
        client=client,
        default_scope=default_scope,
        admission=admission,
        processor=processor,
        fatal_errors=fatal_errors,
        admission_timeout_seconds=admission_timeout_seconds,
        launch_preparer=launch_preparer,
        cancellation_handler=cancellation_handler,
    )


async def _handle_passive_message(
    body: dict[str, object],
    *,
    expected_team_id: str,
    bot_user_id: str,
    bot_id: str | None = None,
    default_scope: ScopeKey,
    sink: SlackIngressAdmission,
    fatal_errors: asyncio.Queue[BaseException],
    persistence_timeout_seconds: float,
) -> None:
    """Persist passive context without admitting, launching, enqueuing, or replying."""

    try:
        message = normalize_passive_message(
            body,
            expected_team_id=expected_team_id,
            bot_user_id=bot_user_id,
            bot_id=bot_id,
        )
    except (SlackEventRejected, ValidationError) as exc:
        LOGGER.warning("Passive Slack message rejected: %s", exc)
        return
    if message is None:
        return
    try:
        async with asyncio.timeout(persistence_timeout_seconds):
            await sink.record_passive_message(message, default_scope)
    except Exception as exc:
        LOGGER.exception(
            "Passive Slack message persistence failed",
            extra={"event_id": message.event_id},
        )
        if fatal_errors.empty():
            fatal_errors.put_nowait(exc)


async def _admit_and_enqueue(
    job: SlackMentionJob,
    *,
    client: AsyncWebClient,
    default_scope: ScopeKey,
    admission: SlackIngressAdmission,
    processor: SlackJobProcessor,
    fatal_errors: asyncio.Queue[BaseException],
    admission_timeout_seconds: float,
    launch_preparer: SlackLaunchPreparer | None,
    cancellation_handler: SlackCancellationHandler | None = None,
) -> None:
    eligibility = await _conversation_eligibility(client, job)
    context_conversation_ids: tuple[str, ...] = (job.channel_id,)
    context_projection_source = SlackContextProjectionSource.EXACT_DESTINATION
    if eligibility.kind is SlackConversationKind.DM:
        try:
            context_conversation_ids = await _load_dm_context_conversation_ids(client, job)
            context_projection_source = SlackContextProjectionSource.DM_MEMBERSHIP_INTERSECTION
        except Exception:
            # Availability and confidentiality both fail safe here: keep answering the DM,
            # but do not project any conversation Slack could not currently prove shared.
            LOGGER.exception(
                "Slack DM membership projection failed; using DM-only context",
                extra={"event_id": job.event_id},
            )
            context_projection_source = SlackContextProjectionSource.DM_ONLY_FALLBACK
    job = SlackMentionJob.model_validate(
        {
            **job.model_dump(mode="python"),
            "conversation_kind": eligibility.kind,
            "context_conversation_ids": context_conversation_ids,
            "context_projection_source": context_projection_source,
            "conversation_authority_source": eligibility.provenance,
            "bot_presence": eligibility.bot_presence,
            "conversation_lifecycle": eligibility.lifecycle,
            "external_provenance": eligibility.external_provenance,
            "membership_policy_version": eligibility.membership_policy_version,
            "context_access_hash": build_context_access_hash(
                team_id=job.team_id,
                user_id=job.user_id,
                channel_id=job.channel_id,
                context_conversation_ids=context_conversation_ids,
            ),
        }
    )
    try:
        async with asyncio.timeout(admission_timeout_seconds):
            admitted = await admission.admit(
                job,
                default_scope,
                eligibility=eligibility,
            )
    except SlackAdmissionPolicyRejected as exc:
        LOGGER.warning(
            "Slack event authority rejected event",
            extra={"event_id": job.event_id, "safe_code": exc.safe_code},
        )
        return
    except Exception as exc:
        LOGGER.exception("Slack event admission failed", extra={"event_id": job.event_id})
        if fatal_errors.empty():
            fatal_errors.put_nowait(exc)
        return
    if admitted is None:
        return
    if cancellation_handler is not None and cancellation_handler.accepts(job.prompt):
        if launch_preparer is None:
            raise RuntimeError("Slack cancellation requires durable launch preparation")
        try:
            async with asyncio.timeout(admission_timeout_seconds):
                result = await cancellation_handler.handle(admitted, launch_preparer)
            await processor.deliver_control(result)
        except Exception:
            # Admission is committed and remains eligible for control recovery.  Never
            # reinterpret an exact cancellation command as a conversational prompt.
            LOGGER.exception(
                "Slack cancellation control failed",
                extra={"event_id": job.event_id},
            )
        return
    if launch_preparer is not None:
        try:
            async with asyncio.timeout(admission_timeout_seconds):
                admitted = await launch_preparer.prepare(admitted)
        except SlackFollowupBusyError:
            # The preparer has already persisted this turn as FIFO-eligible.  Do not make
            # an untracked Slack call here: progress/clarification/final effects must flow
            # through an immutable outbox intent once the follow-up becomes runnable.
            LOGGER.info(
                "Slack follow-up remains durably queued behind active thread work",
                extra={"event_id": job.event_id},
            )
            return
        except Exception:
            # The committed admission remains durable and recoverable. Do not enqueue
            # an item whose Task/Run launch was not committed.
            LOGGER.exception(
                "Slack launch materialization failed", extra={"event_id": job.event_id}
            )
            return
    # The preparer commits Thread/Task/Run before this notification point.  If the
    # local wake-up is rejected, the durable launch remains recoverable in Postgres.
    processor.enqueue(admitted)


async def _conversation_eligibility(
    client: AsyncWebClient,
    job: SlackMentionJob,
) -> SlackConversationEligibility:
    if job.trigger_kind is SlackTriggerKind.MESSAGE_IM:
        return SlackConversationEligibility(
            kind=SlackConversationKind.DM,
            provenance="slack_event",
            bot_presence=SlackBotPresence.PRESENT,
            lifecycle=SlackConversationLifecycle.ACTIVE,
            external_provenance=SlackExternalProvenance.NOT_APPLICABLE,
        )
    try:
        conversation_info = await client.conversations_info(channel=job.channel_id)
        conversation_payload = getattr(conversation_info, "data", conversation_info)
        eligibility = classify_slack_conversation(
            conversation_payload,
            expected_channel_id=job.channel_id,
        )
        if eligibility.admissible:
            return eligibility
        if (
            eligibility.kind is not SlackConversationKind.UNKNOWN
            and eligibility.bot_presence is SlackBotPresence.UNKNOWN
            and eligibility.lifecycle is SlackConversationLifecycle.ACTIVE
        ):
            # A verified app_mention envelope proves Leo received the event in this
            # conversation even when conversations.info omits is_member.  Conversation
            # kind and shared/external provenance still come from the metadata response.
            return eligibility.model_copy(update={"bot_presence": SlackBotPresence.PRESENT})
        if (
            eligibility.bot_presence is SlackBotPresence.ABSENT
            or eligibility.lifecycle is not SlackConversationLifecycle.ACTIVE
        ):
            return eligibility
        LOGGER.warning(
            "Slack conversations.info returned unusable metadata; using event authority",
            extra={"event_id": job.event_id},
        )
    except Exception:
        LOGGER.exception(
            "Slack conversation metadata lookup failed; using event authority",
            extra={"event_id": job.event_id},
        )
    return SlackConversationEligibility(
        kind=job.conversation_kind,
        provenance=(
            "slack_event"
            if job.conversation_kind is not SlackConversationKind.UNKNOWN
            else "unknown"
        ),
        bot_presence=(
            SlackBotPresence.PRESENT
            if job.conversation_kind is not SlackConversationKind.UNKNOWN
            else SlackBotPresence.UNKNOWN
        ),
        lifecycle=(
            SlackConversationLifecycle.ACTIVE
            if job.conversation_kind is not SlackConversationKind.UNKNOWN
            else SlackConversationLifecycle.UNKNOWN
        ),
        external_provenance=job.external_provenance,
    )


async def _load_dm_context_conversation_ids(
    client: AsyncWebClient,
    job: SlackMentionJob,
) -> tuple[str, ...]:
    """Return the exact current bot/user membership intersection, fully paginated."""

    conversation_ids = {job.channel_id}
    cursor = ""
    seen_cursors: set[str] = set()
    while True:
        if cursor:
            response = await client.users_conversations(
                user=job.user_id,
                types="public_channel,private_channel,mpim",
                exclude_archived=True,
                limit=200,
                cursor=cursor,
            )
        else:
            response = await client.users_conversations(
                user=job.user_id,
                types="public_channel,private_channel,mpim",
                exclude_archived=True,
                limit=200,
            )
        payload = getattr(response, "data", response)
        if not isinstance(payload, Mapping) or payload.get("ok") is not True:
            raise RuntimeError("Slack users.conversations did not return a successful response")
        channels = payload.get("channels")
        if not isinstance(channels, list):
            raise RuntimeError("Slack users.conversations returned malformed channels")
        for channel in channels:
            if not isinstance(channel, Mapping) or channel.get("is_archived") is True:
                continue
            channel_id = channel.get("id")
            if not isinstance(channel_id, str) or not channel_id:
                continue
            is_supported_kind = (
                channel.get("is_channel") is True
                or channel.get("is_group") is True
                or channel.get("is_mpim") is True
            ) and channel.get("is_im") is not True
            if is_supported_kind:
                conversation_ids.add(channel_id)

        metadata = payload.get("response_metadata")
        next_cursor = metadata.get("next_cursor") if isinstance(metadata, Mapping) else ""
        if not isinstance(next_cursor, str) or not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise RuntimeError("Slack users.conversations repeated a pagination cursor")
        seen_cursors.add(next_cursor)
        cursor = next_cursor

    return tuple(sorted(conversation_ids))


async def run_socket_mode(
    settings: Settings,
    *,
    runtime: SlackJobRuntime | None = None,
    admission: SlackIngressAdmission | None = None,
    admission_timeout_seconds: float = 5.0,
    launch_preparer: SlackLaunchPreparer | None = None,
    outbox: PostgresDeliveryOutbox | None = None,
    dispatcher: SlackOutboxDispatcher | None = None,
    cancellation_handler: SlackCancellationHandler | None = None,
    socket_readiness: SlackSocketReadinessRegistry = SLACK_SOCKET_READINESS,
) -> None:
    missing = settings.missing_for_live_slack()
    if missing:
        raise RuntimeError(f"missing Slack configuration names: {', '.join(missing)}")
    assert settings.slack_bot_token is not None
    assert settings.slack_app_token is not None
    assert settings.leo_slack_team_id is not None
    socket_readiness.record_starting()

    default_scope = ScopeKey(
        organization_id=settings.leo_organization_id,
        strategy_id=settings.leo_strategy_id,
    )
    selected_admission = admission or InMemorySlackIngressAdmission()
    try:
        async with asyncio.timeout(admission_timeout_seconds):
            await selected_admission.preflight()
    except Exception as exc:
        socket_readiness.record_probe_failure()
        raise RuntimeError("Slack admission store preflight failed") from exc

    app = AsyncApp(token=settings.slack_bot_token.get_secret_value())
    auth = await app.client.auth_test()
    bot_user_id = str(auth.get("user_id") or "")
    bot_id = str(auth.get("bot_id") or "") or None
    actual_team_id = str(auth.get("team_id") or "")
    if not bot_user_id:
        socket_readiness.record_probe_failure()
        raise RuntimeError("Slack auth.test returned no bot user ID")
    if actual_team_id != settings.leo_slack_team_id:
        socket_readiness.record_probe_failure()
        raise RuntimeError("Slack auth.test team does not match LEO_SLACK_TEAM_ID")

    selected_runtime = runtime or StaticTransportRuntime()
    if cancellation_handler is not None and (
        launch_preparer is None or outbox is None or dispatcher is None
    ):
        raise ValueError(
            "Slack cancellation requires launch preparation and durable outbox dispatch"
        )
    processor = SlackJobProcessor(
        client=app.client,
        runtime=selected_runtime,
        outbox=outbox,
        dispatcher=dispatcher,
        launch_recoverer=launch_preparer,
        runtime_timeout_seconds=settings.leo_max_run_seconds + 30.0,
    )
    if cancellation_handler is not None and launch_preparer is not None:
        for result in await cancellation_handler.recover(launch_preparer):
            await processor.deliver_control(result)
    if launch_preparer is not None:
        await processor.recover_pending()
    if dispatcher is not None:
        # Repaired/pending intents are durable authority; dispatch them before the
        # process accepts more volatile wake-up hints.
        await dispatcher.dispatch_available(app.client)
    fatal_errors: asyncio.Queue[BaseException] = asyncio.Queue(maxsize=1)

    async def on_mention(body: dict[str, object]) -> None:
        await _handle_app_mention(
            body,
            client=app.client,
            expected_team_id=settings.leo_slack_team_id or "",
            bot_user_id=bot_user_id,
            default_scope=default_scope,
            admission=selected_admission,
            processor=processor,
            fatal_errors=fatal_errors,
            admission_timeout_seconds=admission_timeout_seconds,
            launch_preparer=launch_preparer,
            cancellation_handler=cancellation_handler,
        )

    async def on_message(body: dict[str, object]) -> None:
        event = body.get("event")
        channel_type = event.get("channel_type") if isinstance(event, Mapping) else None
        if channel_type == "im":
            await _handle_message_im(
                body,
                client=app.client,
                expected_team_id=settings.leo_slack_team_id or "",
                bot_user_id=bot_user_id,
                default_scope=default_scope,
                admission=selected_admission,
                processor=processor,
                fatal_errors=fatal_errors,
                admission_timeout_seconds=admission_timeout_seconds,
                launch_preparer=launch_preparer,
                cancellation_handler=cancellation_handler,
            )
            return
        await _handle_passive_message(
            body,
            expected_team_id=settings.leo_slack_team_id or "",
            bot_user_id=bot_user_id,
            bot_id=bot_id,
            default_scope=default_scope,
            sink=selected_admission,
            fatal_errors=fatal_errors,
            persistence_timeout_seconds=admission_timeout_seconds,
        )

    app.event("app_mention")(on_mention)
    app.event("message")(on_message)
    worker = asyncio.create_task(processor.run(), name="leo-slack-smoke-worker")
    handler = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
    socket = asyncio.create_task(
        handler.start_async(),  # type: ignore[no-untyped-call]
        name="leo-slack-socket",
    )
    socket_monitor = asyncio.create_task(
        _monitor_socket_readiness(handler, socket_readiness),
        name="leo-slack-socket-readiness",
    )
    fatal = asyncio.create_task(fatal_errors.get(), name="leo-slack-admission-failure")
    try:
        done, _ = await asyncio.wait({worker, socket, fatal}, return_when=asyncio.FIRST_COMPLETED)
        if fatal in done:
            raise RuntimeError("Slack event admission failed") from fatal.result()
        if worker in done:
            error = worker.exception()
            if error is None:
                raise RuntimeError("Slack job worker stopped unexpectedly")
            raise RuntimeError("Slack job worker crashed") from error
        await socket
    finally:
        worker.cancel()
        socket.cancel()
        socket_monitor.cancel()
        fatal.cancel()
        await asyncio.gather(worker, socket, socket_monitor, fatal, return_exceptions=True)
        await handler.close_async()  # type: ignore[no-untyped-call]
        socket_readiness.record_stopped()


async def _monitor_socket_readiness(
    handler: AsyncSocketModeHandler,
    registry: SlackSocketReadinessRegistry,
    *,
    interval_seconds: float = 1.0,
) -> None:
    """Poll the SDK's real ping/session state without making a Slack API call."""

    last_connected: bool | None = None
    while True:
        try:
            connected = await handler.client.is_connected()
        except Exception:
            registry.record_probe_failure()
        else:
            registry.record_probe(connected)
            if connected != last_connected:
                if connected:
                    LOGGER.info("Slack Socket Mode connected")
                elif last_connected is True:
                    LOGGER.warning("Slack Socket Mode disconnected")
                last_connected = connected
        await asyncio.sleep(interval_seconds)


def _wakeup_key(admitted: AdmittedSlackMention) -> str:
    return (
        f"task:{admitted.launch.task_id}"
        if admitted.launch is not None
        else f"event:{admitted.job.event_id}"
    )


def _run_reference(admitted: AdmittedSlackMention) -> str:
    return admitted.launch.run_id if admitted.launch is not None else admitted.job.event_id


async def _run_runtime_with_deadline(
    runtime: SlackJobRuntime,
    admitted: AdmittedSlackMention,
    *,
    timeout_seconds: float,
) -> str | RenderedSlackText:
    """Cancel only the runtime task with an explicit deadline provenance marker."""

    runtime_task = asyncio.create_task(
        runtime.handle(admitted),
        name=f"leo-slack-runtime-{admitted.job.event_id}",
    )
    try:
        return await asyncio.wait_for(asyncio.shield(runtime_task), timeout=timeout_seconds)
    except TimeoutError:
        runtime_task.cancel(RUNTIME_DEADLINE_CANCEL_MESSAGE)
        try:
            # A durable runtime may reconcile the terminal winner while handling this
            # marked cancellation and return the authoritative response. Simpler
            # runtimes re-raise cancellation and receive the generic timeout UX.
            return await runtime_task
        except asyncio.CancelledError as exc:
            raise TimeoutError from exc
    except asyncio.CancelledError:
        # Process shutdown is not a Task/Run timeout. Propagate a distinct marker and
        # do not let the child runtime survive its processor.
        runtime_task.cancel("slack_processor_shutdown")
        await asyncio.gather(runtime_task, return_exceptions=True)
        raise
