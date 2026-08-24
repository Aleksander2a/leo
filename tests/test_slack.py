"""The Slack transport's one promise: every question it accepts gets a reply."""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from leo.agent.loop import AgentResult
from leo.agent.runtime import TurnRequest
from leo.slack.app import Incoming, SlackService, _failure_text, _incoming


def event(**overrides: Any) -> dict[str, Any]:
    base = {
        "user": "U1",
        "team": "T1",
        "channel": "C1",
        "text": "<@BOT> what is BTC doing?",
        "ts": "1.1",
        "event_ts": "1.1",
        "client_msg_id": "m-1",
    }
    base.update(overrides)
    return base


class FakeClient:
    def __init__(self) -> None:
        self.posted: list[dict[str, Any]] = []
        self.updated: list[dict[str, Any]] = []

    async def chat_postMessage(self, **kwargs: Any) -> dict[str, Any]:
        self.posted.append(kwargs)
        return {"ts": f"post-{len(self.posted)}"}

    async def chat_update(self, **kwargs: Any) -> dict[str, Any]:
        self.updated.append(kwargs)
        return {"ok": True}


class FakeAgent:
    def __init__(self, result: AgentResult | Exception) -> None:
        self._result = result
        self.requests: list[TurnRequest] = []

    async def handle(self, request: TurnRequest) -> AgentResult:
        self.requests.append(request)
        if isinstance(self._result, Exception):
            raise self._result
        if request.on_step is not None:
            await request.on_step("web.search_tavily")
        return self._result


def service(agent: Any) -> SlackService:
    return SlackService(agent=agent, settings=None, concurrency=2)  # type: ignore[arg-type]


def test_the_bot_ignores_itself_and_other_bots() -> None:
    assert _incoming(event(user="BOT"), bot_user_id="BOT", is_dm=False) is None
    assert _incoming(event(bot_id="B1"), bot_user_id="BOT", is_dm=False) is None
    assert _incoming(event(subtype="message_changed"), bot_user_id="BOT", is_dm=False) is None


def test_an_empty_message_is_not_a_question() -> None:
    assert _incoming(event(text="<@BOT>"), bot_user_id="BOT", is_dm=False) is None


def test_the_mention_is_stripped_from_the_question() -> None:
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    assert incoming.text == "what is BTC doing?"


def test_the_scope_key_is_the_channel() -> None:
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    assert incoming.scope_key == "slack:T1:C1"


def test_a_dm_and_a_channel_get_different_scopes() -> None:
    dm = _incoming(event(channel="D9"), bot_user_id="BOT", is_dm=True)
    channel = _incoming(event(channel="C1"), bot_user_id="BOT", is_dm=False)
    assert dm is not None and channel is not None
    assert dm.scope_key != channel.scope_key
    assert dm.description != channel.description


def test_replies_go_into_a_thread() -> None:
    root = _incoming(event(), bot_user_id="BOT", is_dm=False)
    reply = _incoming(event(thread_ts="0.9"), bot_user_id="BOT", is_dm=False)
    assert root is not None and reply is not None
    assert root.reply_thread == "1.1"  # starts a thread on the message
    assert reply.reply_thread == "0.9"  # stays in the existing one


def test_a_redelivered_event_is_accepted_once() -> None:
    svc = service(FakeAgent(AgentResult(answer="x", status="answered")))
    assert svc.accept("mention:C1:1.1") is True
    assert svc.accept("mention:C1:1.1") is False


@pytest.mark.asyncio
async def test_a_successful_answer_replaces_the_placeholder() -> None:
    agent = FakeAgent(AgentResult(answer="**BTC** is up.", status="answered"))
    client = FakeClient()
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    await service(agent)._guarded(incoming, client)

    assert len(client.posted) == 1  # only the placeholder was posted
    assert client.updated[-1]["blocks"][0]["text"]["text"] == "*BTC* is up."
    assert client.updated[-1]["channel"] == "C1"


@pytest.mark.asyncio
async def test_progress_updates_the_placeholder_while_tools_run() -> None:
    agent = FakeAgent(AgentResult(answer="done", status="answered"))
    client = FakeClient()
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    await service(agent)._guarded(incoming, client)
    assert any("web.search_tavily" in call["text"] for call in client.updated)


@pytest.mark.asyncio
async def test_a_long_answer_is_split_across_messages() -> None:
    answer = "\n\n".join("paragraph " + "x" * 900 for _ in range(6))
    agent = FakeAgent(AgentResult(answer=answer, status="answered"))
    client = FakeClient()
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    await service(agent)._guarded(incoming, client)
    assert len(client.updated) >= 1
    assert len(client.posted) >= 2  # placeholder plus continuations


@pytest.mark.asyncio
async def test_a_failed_run_still_replies_and_says_what_broke() -> None:
    agent = FakeAgent(AgentResult(answer="", status="failed", error="http_503: upstream down"))
    client = FakeClient()
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    await service(agent)._guarded(incoming, client)
    delivered = client.updated[-1]["blocks"][0]["text"]["text"]
    assert "http_503" in delivered


@pytest.mark.asyncio
async def test_an_exception_in_the_transport_still_replies() -> None:
    """Silence in Slack is indistinguishable from the bot being dead."""

    agent = FakeAgent(RuntimeError("kaboom"))
    client = FakeClient()
    incoming = _incoming(event(), bot_user_id="BOT", is_dm=False)
    assert incoming is not None
    await service(agent)._guarded(incoming, client)
    assert client.updated, "the user was left with a bare placeholder"
    assert "plumbing" in client.updated[-1]["blocks"][0]["text"]["text"]


@pytest.mark.asyncio
async def test_the_turn_request_carries_the_scope_and_thread() -> None:
    agent = FakeAgent(AgentResult(answer="ok", status="answered"))
    incoming = _incoming(event(thread_ts="0.9"), bot_user_id="BOT", is_dm=True)
    assert incoming is not None
    await service(agent)._guarded(incoming, FakeClient())
    request = agent.requests[0]
    assert request.scope.key == "slack:T1:C1"
    assert request.scope.actor_id == "U1"
    assert request.thread_key == "0.9"
    assert request.external_id == "m-1"
    assert request.conversation_kind == "dm"


def test_failure_text_reports_the_real_error_not_a_stock_sentence() -> None:
    assert "rate_limited" in _failure_text("rate_limited: too many requests")
    # And with nothing to report, it is still actionable rather than mysterious.
    assert "narrow" in _failure_text(None)


def test_incoming_is_hashable_and_frozen() -> None:
    incoming = Incoming(
        text="t",
        channel="C1",
        user="U1",
        team="T1",
        event_ts="1.1",
        thread_ts=None,
        is_dm=False,
        client_msg_id="m",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        incoming.text = "other"  # type: ignore[misc]
