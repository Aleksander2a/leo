"""Slack Socket Mode transport.

Its whole job is to get a question to the agent and get the answer back into
the right conversation. It holds no policy about what Leo may say, and it has
exactly one guarantee it must never break:

    **Every message Leo is asked to answer gets a reply.**

If the loop fails, the reply says what actually failed. If the transport itself
throws, the exception handler still posts. Silence is the one outcome that is
never acceptable, because from Slack it is indistinguishable from the bot being
dead.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.app.async_app import AsyncApp

from leo.agent.contracts import Scope
from leo.agent.runtime import LeoRuntime, TurnRequest, runtime
from leo.config import Settings
from leo.slack.render import blocks_for, chunks, clean_prompt, to_mrkdwn

logger = logging.getLogger(__name__)

THINKING = "_Working on it…_"


@dataclass(frozen=True)
class Incoming:
    """A Slack message Leo has been asked to answer."""

    text: str
    channel: str
    user: str
    team: str
    event_ts: str
    thread_ts: str | None
    is_dm: bool
    client_msg_id: str

    @property
    def scope_key(self) -> str:
        # The channel id is the isolation boundary. A DM channel (D…) is unique
        # to one person, so DM memories are private by construction; a channel
        # (C…/G…) is shared by its members and nothing else.
        return f"slack:{self.team}:{self.channel}"

    @property
    def reply_thread(self) -> str:
        # Reply in the thread when there is one, otherwise start one on the
        # message, so a channel does not fill with Leo's long answers.
        return self.thread_ts or self.event_ts

    @property
    def description(self) -> str:
        return "a private direct message" if self.is_dm else f"the Slack channel <#{self.channel}>"


class SlackService:
    """Wires Slack events to the agent, with bounded concurrency."""

    def __init__(self, *, agent: LeoRuntime, settings: Settings, concurrency: int = 4) -> None:
        self._agent = agent
        self._settings = settings
        self._semaphore = asyncio.Semaphore(concurrency)
        self._seen: set[str] = set()
        self._tasks: set[asyncio.Task[None]] = set()

    def accept(self, event_key: str) -> bool:
        """Reject Slack's redeliveries of an event already in flight."""

        if event_key in self._seen:
            return False
        self._seen.add(event_key)
        if len(self._seen) > 5000:
            self._seen = set(list(self._seen)[-2500:])
        return True

    def spawn(self, incoming: Incoming, client: Any) -> None:
        task = asyncio.create_task(self._guarded(incoming, client))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def drain(self) -> None:
        if self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    async def _guarded(self, incoming: Incoming, client: Any) -> None:
        placeholder: str | None = None
        try:
            async with self._semaphore:
                placeholder = await self._post(client, incoming, THINKING)
                await self._answer(incoming, client, placeholder)
        except Exception:
            logger.exception("slack turn failed for %s", incoming.scope_key)
            await self._fail(
                client,
                incoming,
                placeholder,
                "Something in my own plumbing broke before I could answer. "
                "Ask me again — if it keeps happening the logs will have the detail.",
            )

    async def _answer(self, incoming: Incoming, client: Any, placeholder: str | None) -> None:
        async def progress(names: str) -> None:
            if placeholder:
                await self._update(client, incoming.channel, placeholder, f"_Checking {names}…_")

        result = await self._agent.handle(
            TurnRequest(
                question=incoming.text,
                scope=Scope(key=incoming.scope_key, actor_id=incoming.user),
                thread_key=incoming.reply_thread,
                external_id=incoming.client_msg_id or incoming.event_ts,
                scope_description=incoming.description,
                conversation_kind="dm" if incoming.is_dm else "channel",
                team_id=incoming.team,
                channel_id=incoming.channel,
                on_step=progress,
            )
        )

        if not result.answered:
            await self._fail(
                client,
                incoming,
                placeholder,
                _failure_text(result.error),
            )
            return

        pieces = chunks(to_mrkdwn(result.answer))
        if not pieces:
            await self._fail(client, incoming, placeholder, _failure_text(None))
            return
        if placeholder:
            await self._update(client, incoming.channel, placeholder, pieces[0])
        else:
            await self._post(client, incoming, pieces[0])
        for piece in pieces[1:]:
            await self._post(client, incoming, piece)

    async def _fail(
        self,
        client: Any,
        incoming: Incoming,
        placeholder: str | None,
        text: str,
    ) -> None:
        try:
            if placeholder:
                await self._update(client, incoming.channel, placeholder, text)
            else:
                await self._post(client, incoming, text)
        except Exception:
            logger.exception("could not deliver a failure notice to %s", incoming.channel)

    async def _post(self, client: Any, incoming: Incoming, text: str) -> str | None:
        response = await client.chat_postMessage(
            channel=incoming.channel,
            thread_ts=incoming.reply_thread,
            text=_fallback(text),
            blocks=blocks_for(text),
        )
        return str(response.get("ts")) if response else None

    async def _update(self, client: Any, channel: str, ts: str, text: str) -> None:
        await client.chat_update(
            channel=channel,
            ts=ts,
            text=_fallback(text),
            blocks=blocks_for(text),
        )


def _fallback(text: str) -> str:
    """Notification text: plain, short, and never empty."""

    flat = " ".join(text.split())
    return (flat[:250] + "…") if len(flat) > 250 else (flat or "Leo")


def _failure_text(error: str | None) -> str:
    """Say what actually went wrong, not a canned apology.

    The previous runtime matched substrings in an internal reason code to pick
    one of eight stock sentences, which is how a forced tool loop surfaced as
    "the reasoning service stopped unexpectedly" -- a sentence that was not
    true and left the user with nothing to act on.
    """

    detail = (error or "").strip()
    if not detail:
        return (
            "I couldn't put together an answer for that one. "
            "Try asking again, or narrow it to the part you care about most."
        )
    return (
        f"I couldn't finish that: {detail}\n\n"
        "That's a fault on my side, not your question. Ask again in a moment, "
        "or narrow it and I'll take a different route."
    )


def _incoming(event: dict[str, Any], *, bot_user_id: str, is_dm: bool) -> Incoming | None:
    if event.get("bot_id") or event.get("subtype") in {"bot_message", "message_changed"}:
        return None
    user = str(event.get("user") or "")
    if not user or user == bot_user_id:
        return None
    text = clean_prompt(str(event.get("text") or ""))
    if not text:
        return None
    thread_ts = event.get("thread_ts")
    return Incoming(
        text=text,
        channel=str(event.get("channel") or ""),
        user=user,
        team=str(event.get("team") or event.get("user_team") or "unknown"),
        event_ts=str(event.get("event_ts") or event.get("ts") or ""),
        thread_ts=str(thread_ts) if thread_ts else None,
        is_dm=is_dm,
        client_msg_id=str(event.get("client_msg_id") or ""),
    )


async def serve(settings: Settings) -> None:
    """Run Leo on Slack until the process is stopped."""

    if settings.slack_bot_token is None or settings.slack_app_token is None:
        raise RuntimeError("SLACK_BOT_TOKEN and SLACK_APP_TOKEN are required")

    app = AsyncApp(token=settings.slack_bot_token.get_secret_value())
    identity = await app.client.auth_test()
    bot_user_id = str(identity.get("user_id") or "")
    team_id = str(identity.get("team_id") or settings.leo_slack_team_id or "unknown")
    logger.info("connected to Slack as %s in team %s", bot_user_id, team_id)

    async with runtime(settings) as agent:
        service = SlackService(
            agent=agent,
            settings=settings,
            concurrency=settings.leo_slack_worker_concurrency,
        )
        logger.info("agent ready with %d tools", len(agent.tool_names))

        @app.event("app_mention")
        async def on_mention(body: dict[str, Any], client: Any) -> None:
            event = body.get("event") or {}
            incoming = _incoming(event, bot_user_id=bot_user_id, is_dm=False)
            if incoming is None:
                return
            incoming = _with_team(incoming, team_id)
            if not service.accept(f"mention:{incoming.channel}:{incoming.event_ts}"):
                return
            service.spawn(incoming, client)

        @app.event("message")
        async def on_message(body: dict[str, Any], client: Any) -> None:
            event = body.get("event") or {}
            if event.get("channel_type") != "im":
                return
            incoming = _incoming(event, bot_user_id=bot_user_id, is_dm=True)
            if incoming is None:
                return
            incoming = _with_team(incoming, team_id)
            if not service.accept(f"dm:{incoming.channel}:{incoming.event_ts}"):
                return
            service.spawn(incoming, client)

        handler = AsyncSocketModeHandler(app, settings.slack_app_token.get_secret_value())
        try:
            # slack_bolt ships no annotations for its socket-mode handler.
            await handler.start_async()  # type: ignore[no-untyped-call]
        finally:
            await service.drain()


def _with_team(incoming: Incoming, team_id: str) -> Incoming:
    if incoming.team and incoming.team != "unknown":
        return incoming
    return Incoming(**{**incoming.__dict__, "team": team_id})
