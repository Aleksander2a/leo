from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import replace

import pytest

from leo.harness.models import ContextItemKind, ContextItemRetention
from leo.integrations.slack.context import (
    SlackHistoryContextLoader,
    SlackThreadContextError,
)
from leo.integrations.slack.events import (
    SlackConversationKind,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.persistence.schema import SanitizedMessageRow
from leo.persistence.slack_messages import (
    PersistedSlackThreadMessage,
    PersistedSlackThreadSnapshot,
    SlackThreadCoverageReason,
    SlackThreadCoverageSource,
    _assess_thread_coverage,
)


class _HistoryClient:
    def __init__(
        self,
        pages: dict[tuple[str, str], dict[str, object] | BaseException],
        *,
        reply_pages: dict[tuple[str, str, str], dict[str, object] | BaseException] | None = None,
        delay_seconds: float = 0.0,
        auth_team_id: str = "T1",
        auth_outcomes: list[dict[str, object] | BaseException] | None = None,
    ) -> None:
        self._pages = pages
        self._reply_pages = reply_pages or {}
        self._delay_seconds = delay_seconds
        self._auth_team_id = auth_team_id
        self._auth_outcomes = list(auth_outcomes or [])
        self.calls: list[dict[str, object]] = []
        self.reply_calls: list[dict[str, object]] = []
        self.auth_calls = 0
        self.active = 0
        self.max_active = 0

    async def auth_test(self, **_kwargs: object) -> dict[str, object]:
        self.auth_calls += 1
        if self._auth_outcomes:
            outcome = self._auth_outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        return {"ok": True, "team_id": self._auth_team_id}

    async def conversations_history(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay_seconds:
                await asyncio.sleep(self._delay_seconds)
            key = (str(kwargs["channel"]), str(kwargs.get("cursor") or ""))
            outcome = self._pages[key]
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active -= 1

    async def conversations_replies(self, **kwargs: object) -> dict[str, object]:
        self.reply_calls.append(kwargs)
        key = (
            str(kwargs["channel"]),
            str(kwargs["ts"]),
            str(kwargs.get("cursor") or ""),
        )
        outcome = self._reply_pages[key]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _job(
    *,
    channel_id: str = "C1",
    conversation_kind: SlackConversationKind = SlackConversationKind.ORDINARY_INTERNAL,
    projection: tuple[str, ...] | None = None,
    prompt: str = "Review NVDA downside risk",
    message_ts: str = "999.000",
    thread_root_ts: str | None = None,
) -> SlackMentionJob:
    selected_projection = projection or (channel_id,)
    trigger_kind = (
        SlackTriggerKind.MESSAGE_IM
        if conversation_kind is SlackConversationKind.DM
        else SlackTriggerKind.APP_MENTION
    )
    return SlackMentionJob(
        event_id="Ev-current",
        team_id="T1",
        channel_id=channel_id,
        user_id="U-current",
        message_ts=message_ts,
        thread_root_ts=thread_root_ts or message_ts,
        conversation_key=f"slack:T1:{channel_id}:{thread_root_ts or message_ts}",
        prompt=prompt,
        conversation_kind=conversation_kind,
        trigger_kind=trigger_kind,
        context_conversation_ids=selected_projection,
        context_access_hash=build_context_access_hash(
            team_id="T1",
            user_id="U-current",
            channel_id=channel_id,
            context_conversation_ids=selected_projection,
        ),
    )


def _page(
    messages: list[dict[str, object]],
    *,
    next_cursor: str = "",
    has_more: bool = False,
) -> dict[str, object]:
    return {
        "ok": True,
        "messages": messages,
        "has_more": has_more,
        "response_metadata": {"next_cursor": next_cursor},
    }


def _message(ts: str, text: str, *, user: str = "U-source", **extra: object) -> dict[str, object]:
    return {"type": "message", "ts": ts, "user": user, "text": text, **extra}


class _ThreadFallback:
    def __init__(self, snapshot: PersistedSlackThreadSnapshot) -> None:
        self.snapshot = snapshot
        self.record_calls: list[dict[str, object]] = []
        self.load_calls: list[dict[str, object]] = []

    async def record_root_coverage(self, **kwargs: object) -> bool:
        self.record_calls.append(kwargs)
        return True

    async def load_complete_thread(self, **kwargs: object) -> PersistedSlackThreadSnapshot:
        self.load_calls.append(kwargs)
        return self.snapshot


def _persisted_snapshot(
    *,
    complete: bool = True,
    team_id: str = "T1",
    channel_id: str = "C1",
    reason: SlackThreadCoverageReason = SlackThreadCoverageReason.COMPLETE,
    messages: tuple[PersistedSlackThreadMessage, ...] | None = None,
) -> PersistedSlackThreadSnapshot:
    selected = messages or (
        PersistedSlackThreadMessage(
            id="message-root",
            actor_id="U1",
            role="user",
            text="Remember amber hexagons.",
            message_ts="100.000",
        ),
        PersistedSlackThreadMessage(
            id="message-leo",
            actor_id="leo",
            role="assistant",
            text="I will remember amber hexagons in this conversation.",
            message_ts="110.000",
        ),
    )
    return PersistedSlackThreadSnapshot(
        team_id=team_id,
        channel_id=channel_id,
        thread_root_ts="100.000",
        current_message_ts="120.000",
        conversation_id="conversation-internal",
        messages=selected,
        complete=complete,
        coverage_reason=reason,
        authoritative_reply_count=len(selected),
        authoritative_latest_reply_ts="120.000",
        coverage_source=SlackThreadCoverageSource.USER_HISTORY,
        coverage_snapshot_hash="a" * 64,
        complete_through_ts="120.000" if complete else None,
        coverage_digest="b" * 64,
        persisted_message_count=len(selected) + 1,
        boundary_attested=complete,
        boundary_actor_id="U-current" if complete else None,
        boundary_event_id="Ev-current" if complete else None,
    )


def _persisted_row(
    message_id: str,
    message_ts: str,
    text: str,
    *,
    actor_id: str,
    role: str,
    thread_root_ts: str,
    event_id: str | None = None,
) -> SanitizedMessageRow:
    return SanitizedMessageRow(
        id=message_id,
        organization_id="org-demo",
        strategy_id="strategy-default",
        destination_id="C1",
        external_event_id=event_id or f"event-{message_id}",
        text=text,
        content_hash="c" * 64,
        conversation_id="conversation-internal",
        actor_id=actor_id,
        role=role,
        provider_message_ts=message_ts,
        provider_thread_root_ts=thread_root_ts,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "conversation_kind",
    [
        SlackConversationKind.ORDINARY_INTERNAL,
        SlackConversationKind.MPIM,
        SlackConversationKind.SHARED,
        SlackConversationKind.EXTERNAL,
    ],
)
async def test_fresh_non_dm_root_cannot_inherit_an_unrelated_slack_thread(
    conversation_kind: SlackConversationKind,
) -> None:
    client = _HistoryClient(
        {
            ("C1", ""): _page(
                [
                    _message(
                        "100.000",
                        "I want only safe high-dividend stocks in an unrelated old thread.",
                    )
                ]
            ),
        }
    )
    result = await SlackHistoryContextLoader(client).load(
        _job(
            conversation_kind=conversation_kind,
            prompt="What are some interesting investing opportunities right now?",
        )
    )

    assert client.calls == []
    assert result.items == ()
    assert result.manifest.requested_conversation_ids == ()
    assert result.manifest.loaded_conversation_ids == ()
    assert result.manifest.history_requests == 0
    assert result.manifest.raw_messages_scanned == 0
    assert result.manifest.eligible_messages_ranked == 0


@pytest.mark.asyncio
async def test_dm_uses_exact_membership_union_and_returns_access_hash() -> None:
    projection = ("C1", "D1", "G1")
    job = _job(
        channel_id="D1",
        conversation_kind=SlackConversationKind.DM,
        projection=projection,
    )
    client = _HistoryClient(
        {
            ("C1", ""): _page([_message("100.000", "NVDA channel context")]),
            ("D1", ""): _page([_message("101.000", "private portfolio preference")]),
            ("G1", ""): _page([_message("102.000", "NVDA group risk discussion")]),
        }
    )
    result = await SlackHistoryContextLoader(client, user_history_client=client).load(job)

    assert {str(call["channel"]) for call in client.calls} == set(projection)
    assert {item.conversation_id for item in result.items} == set(projection)
    assert result.manifest.requested_conversation_ids == projection
    assert result.manifest.context_access_hash == job.context_access_hash
    assert all("Slack DM continuity (background only" in item.content for item in result.items)


@pytest.mark.asyncio
async def test_relevance_across_dm_union_precedes_token_budgeting() -> None:
    projection = ("C1", "D1")
    job = _job(
        channel_id="D1",
        conversation_kind=SlackConversationKind.DM,
        projection=projection,
        prompt="NVDA downside risk",
    )
    client = _HistoryClient(
        {
            ("C1", ""): _page([_message("100.000", "NVDA downside risk is rising")]),
            ("D1", ""): _page([_message("200.000", "unrelated " + ("x" * 1_000))]),
        }
    )
    loader = SlackHistoryContextLoader(client, max_context_tokens=60)
    result = await loader.load(job)

    assert len(result.items) == 1
    assert result.items[0].conversation_id == "C1"
    assert "NVDA downside risk" in result.items[0].content
    assert result.manifest.truncated is True


@pytest.mark.asyncio
async def test_current_and_system_events_are_excluded_but_conversational_apps_are_retained() -> (
    None
):
    messages: list[dict[str, object]] = [
        _message("999.000", "current message"),
        _message("100.001", "current event duplicate", event_id="Ev-current"),
        _message("100.002", "current client event", client_msg_id="Ev-current"),
        _message("100.003", "bot", bot_id="B1"),
        _message("100.0031", "bot profile", bot_profile={"id": "B1"}),
        _message("100.004", "application", app_id="A1"),
        _message("100.005", "system subtype", subtype="channel_join"),
        _message("100.006", "hidden", hidden=True),
        {"type": "message", "ts": "100.007", "text": "missing actor"},
        _message("100.008", "   "),
        _message("100.0081", "Slack system", user="USLACKBOT"),
        _message("100.009", "human context to keep", user="U-human"),
    ]
    client = _HistoryClient({("C1", ""): _page(messages)})
    result = await SlackHistoryContextLoader(client).load(
        _job(conversation_kind=SlackConversationKind.DM)
    )

    assert len(result.items) == 4
    assert any("human context to keep" in item.content for item in result.items)
    assert {item.source_actor_id for item in result.items} == {
        "U-human",
        "app:A1",
        "bot:B1",
    }
    assert any("author_kind=bot" in item.content for item in result.items)
    assert any("author_kind=app" in item.content for item in result.items)
    assert result.manifest.raw_messages_scanned == len(messages)


@pytest.mark.asyncio
async def test_partial_permission_failure_keeps_available_projected_context() -> None:
    projection = ("C1", "D1", "G1")
    client = _HistoryClient(
        {
            ("C1", ""): PermissionError("missing_scope"),
            ("D1", ""): _page([_message("100.000", "available DM context")]),
            ("G1", ""): _page([_message("101.000", "available group context")]),
        }
    )
    result = await SlackHistoryContextLoader(client, user_history_client=client).load(
        _job(
            channel_id="D1",
            conversation_kind=SlackConversationKind.DM,
            projection=projection,
        )
    )

    assert {item.conversation_id for item in result.items} == {"D1", "G1"}
    assert result.manifest.failed_conversation_ids == ("C1",)
    assert result.manifest.loaded_conversation_ids == ("D1", "G1")
    assert result.manifest.requested_conversation_ids == projection


@pytest.mark.asyncio
async def test_global_page_message_and_concurrency_caps_are_hard() -> None:
    projection = ("C1", "C2", "C3", "D1")
    pages: dict[tuple[str, str], dict[str, object] | BaseException] = {}
    for index, conversation_id in enumerate(projection):
        pages[(conversation_id, "")] = _page(
            [_message(f"{100 + index}.000", f"page one {conversation_id}")],
            next_cursor="page-2",
            has_more=True,
        )
        pages[(conversation_id, "page-2")] = _page(
            [_message(f"{200 + index}.000", f"page two {conversation_id}")],
            next_cursor="page-3",
            has_more=True,
        )
    client = _HistoryClient(pages, delay_seconds=0.01)
    loader = SlackHistoryContextLoader(
        client,
        max_messages_per_conversation=3,
        max_messages_global=8,
        max_pages_per_conversation=2,
        max_pages_global=5,
        max_concurrency=2,
        page_size=1,
    )
    result = await loader.load(
        _job(
            channel_id="D1",
            conversation_kind=SlackConversationKind.DM,
            projection=projection,
        )
    )

    calls_per_conversation = Counter(str(call["channel"]) for call in client.calls)
    assert len(client.calls) == 5
    assert max(calls_per_conversation.values()) <= 2
    assert client.max_active == 2
    assert result.manifest.raw_messages_scanned <= 8
    assert result.manifest.history_requests == 5
    assert result.manifest.truncated is True


@pytest.mark.asyncio
async def test_response_metadata_cannot_expand_the_projection() -> None:
    projection = ("C1", "D1")
    client = _HistoryClient(
        {
            ("C1", ""): _page(
                [_message("100.000", "safe", channel="C-injected")],
                next_cursor="next",
            ),
            ("C1", "next"): _page([_message("99.000", "still safe", channel="C-injected")]),
            ("D1", ""): _page([_message("101.000", "dm safe", channel="C-injected")]),
            ("C-injected", ""): _page([_message("102.000", "must not load")]),
        }
    )
    result = await SlackHistoryContextLoader(client).load(
        _job(
            channel_id="D1",
            conversation_kind=SlackConversationKind.DM,
            projection=projection,
        )
    )

    assert {str(call["channel"]) for call in client.calls} == set(projection)
    assert {item.conversation_id for item in result.items} == set(projection)
    assert result.manifest.requested_conversation_ids == projection


@pytest.mark.asyncio
async def test_thread_loads_paginated_root_and_replies_in_order_including_leo() -> None:
    job = _job(message_ts="120.000", thread_root_ts="100.000")
    client = _HistoryClient(
        {("C1", ""): _page([_message("100.000", "duplicate root")])},
        reply_pages={
            ("C1", "100.000", ""): _page(
                [
                    _message("100.000", "Remember amber hexagons."),
                    _message(
                        "110.000",
                        "I will remember amber hexagons in this conversation.",
                        user="ULEO",
                        bot_id="BLEO",
                        subtype="bot_message",
                        thread_ts="100.000",
                    ),
                ],
                next_cursor="next",
                has_more=True,
            ),
            ("C1", "100.000", "next"): _page(
                [
                    _message(
                        "115.000",
                        "What did I ask you to remember?",
                        thread_ts="100.000",
                    ),
                    _message(
                        "120.000",
                        "current event must be excluded",
                        thread_ts="100.000",
                    ),
                ]
            ),
        },
    )

    result = await SlackHistoryContextLoader(client, user_history_client=client).load(job)

    thread_items = [item for item in result.items if item.id.startswith("slack-thread:")]
    assert [item.id.rsplit(":", 1)[-1] for item in thread_items] == [
        "100.000",
        "110.000",
        "115.000",
    ]
    assert "author_kind=bot" in thread_items[1].content
    assert "amber hexagons" in thread_items[1].content
    assert thread_items[0].retention is ContextItemRetention.THREAD_ROOT
    assert all(item.retention.pinned for item in thread_items)
    assert [call["channel"] for call in client.reply_calls] == ["C1", "C1"]
    assert all(call["ts"] == "100.000" for call in client.reply_calls)
    assert result.manifest.thread_complete is True
    assert result.manifest.thread_messages_loaded == 3
    assert result.manifest.thread_requests == 2


@pytest.mark.asyncio
async def test_channel_followup_keeps_exact_root_and_leo_outcome_without_other_threads() -> None:
    client = _HistoryClient(
        {
            ("C1", ""): _page(
                [
                    _message(
                        "90.000",
                        "Unrelated preference: only safe high-dividend stocks.",
                    )
                ]
            )
        },
        reply_pages={
            ("C1", "100.000", ""): _page(
                [
                    _message(
                        "100.000",
                        "What are some interesting investing opportunities right now?",
                    ),
                    _message(
                        "110.000",
                        "What goals, risk tolerance, and time horizon should I use?",
                        bot_id="BLEO",
                        thread_ts="100.000",
                    ),
                    _message(
                        "120.000",
                        "Focus on long-term growth and moderate risk.",
                        thread_ts="100.000",
                    ),
                ]
            )
        },
    )

    result = await SlackHistoryContextLoader(client, user_history_client=client).load(
        _job(
            prompt="Focus on long-term growth and moderate risk.",
            message_ts="120.000",
            thread_root_ts="100.000",
        )
    )

    combined = "\n".join(item.content for item in result.items)
    assert client.calls == []
    assert len(client.reply_calls) == 1
    assert client.reply_calls[0]["channel"] == "C1"
    assert client.reply_calls[0]["ts"] == "100.000"
    assert result.manifest.requested_conversation_ids == ()
    assert result.manifest.thread_complete is True
    assert result.manifest.thread_messages_loaded == 2
    assert all(item.id.startswith("slack-thread:") for item in result.items)
    assert all("Slack exact thread" in item.content for item in result.items)
    assert "What are some interesting investing opportunities right now?" in combined
    assert "What goals, risk tolerance, and time horizon should I use?" in combined
    assert "Unrelated preference" not in combined
    assert "Focus on long-term growth" not in combined


@pytest.mark.asyncio
async def test_thread_rejects_foreign_future_and_current_messages() -> None:
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={
            ("C1", "100.000", ""): _page(
                [
                    _message("100.000", "authorized root"),
                    _message("101.000", "authorized reply", thread_ts="100.000"),
                    _message("102.000", "foreign thread", thread_ts="90.000"),
                    _message("120.000", "current", thread_ts="100.000"),
                    _message("121.000", "future", thread_ts="100.000"),
                ]
            )
        },
    )

    result = await SlackHistoryContextLoader(client, user_history_client=client).load(
        _job(message_ts="120.000", thread_root_ts="100.000")
    )

    combined = "\n".join(item.content for item in result.items)
    assert "authorized root" in combined
    assert "authorized reply" in combined
    assert "foreign thread" not in combined
    assert "current" not in combined
    assert "future" not in combined


@pytest.mark.asyncio
async def test_dm_thread_uses_only_exact_thread_despite_authorized_continuity_union() -> None:
    projection = ("C1", "D1", "G1")
    client = _HistoryClient(
        {
            ("C1", ""): _page([_message("80.000", "channel background")]),
            ("D1", ""): _page([_message("81.000", "dm background")]),
            ("G1", ""): _page([_message("82.000", "group background")]),
        },
        reply_pages={
            ("D1", "100.000", ""): _page(
                [
                    _message("100.000", "dm thread root"),
                    _message("110.000", "dm thread reply", thread_ts="100.000"),
                ]
            )
        },
    )

    result = await SlackHistoryContextLoader(client).load(
        _job(
            channel_id="D1",
            conversation_kind=SlackConversationKind.DM,
            projection=projection,
            message_ts="120.000",
            thread_root_ts="100.000",
        )
    )

    assert client.calls == []
    assert {str(call["channel"]) for call in client.reply_calls} == {"D1"}
    thread_items = [item for item in result.items if item.id.startswith("slack-thread:")]
    assert {item.conversation_id for item in thread_items} == {"D1"}
    assert tuple(result.items) == tuple(thread_items)
    assert result.manifest.requested_conversation_ids == ()
    combined = "\n".join(item.content for item in result.items)
    assert "dm thread root" in combined
    assert "dm thread reply" in combined
    assert "channel background" not in combined
    assert "dm background" not in combined
    assert "group background" not in combined


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply_outcome", "error_code"),
    [
        (PermissionError("missing_scope"), "slack_thread_history_unavailable"),
        (_page([_message("101.000", "reply only", thread_ts="100.000")]), "root_missing"),
    ],
)
async def test_thread_api_failure_or_missing_root_fails_closed(
    reply_outcome: dict[str, object] | BaseException,
    error_code: str,
) -> None:
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={("C1", "100.000", ""): reply_outcome},
    )

    with pytest.raises(SlackThreadContextError, match=error_code):
        await SlackHistoryContextLoader(client, user_history_client=client).load(
            _job(message_ts="120.000", thread_root_ts="100.000")
        )
    assert client.calls == []


@pytest.mark.asyncio
async def test_thread_pagination_cap_fails_closed_instead_of_returning_partial_context() -> None:
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={
            ("C1", "100.000", ""): _page(
                [_message("100.000", "root")],
                next_cursor="more",
                has_more=True,
            )
        },
    )

    with pytest.raises(SlackThreadContextError, match="incomplete"):
        await SlackHistoryContextLoader(
            client,
            user_history_client=client,
            max_thread_pages=1,
        ).load(_job(message_ts="120.000", thread_root_ts="100.000"))


@pytest.mark.asyncio
async def test_large_thread_compacts_supporting_turns_but_keeps_decisive_turns_exact() -> None:
    replies = [_message("100.000", "Root objective")]
    for index in range(1, 20):
        text = (
            "We decided to ship option alpha."
            if index == 2
            else "Correction: use option beta instead."
            if index == 3
            else f"Supporting detail {index} " + ("x" * 1_000)
            if index < 7
            else f"Recent reply {index}"
        )
        replies.append(_message(f"{100 + index}.000", text, thread_ts="100.000"))
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={("C1", "100.000", ""): _page(replies)},
    )

    result = await SlackHistoryContextLoader(
        client,
        user_history_client=client,
        max_context_tokens=1_000,
        page_size=40,
    ).load(_job(message_ts="130.000", thread_root_ts="100.000"))

    assert result.manifest.thread_messages_loaded == 20
    assert result.manifest.thread_messages_compacted > 0
    assert result.manifest.thread_compaction_digest is not None
    summary = next(item for item in result.items if item.kind is ContextItemKind.THREAD_SUMMARY)
    assert summary.retention is ContextItemRetention.COMPACTION_SUMMARY
    assert "source_digest=sha256:" in summary.content
    exact = "\n".join(
        item.content for item in result.items if item.kind is ContextItemKind.CONVERSATION_TURN
    )
    assert "Root objective" in exact
    assert "We decided to ship option alpha." in exact
    assert "Correction: use option beta instead." in exact
    assert "Recent reply 19" in exact


@pytest.mark.asyncio
async def test_protected_thread_overflow_fails_before_returning_context() -> None:
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={("C1", "100.000", ""): _page([_message("100.000", "root " + ("x" * 4_000))])},
    )

    with pytest.raises(SlackThreadContextError, match="pinned"):
        await SlackHistoryContextLoader(
            client,
            user_history_client=client,
            max_context_tokens=128,
        ).load(_job(message_ts="120.000", thread_root_ts="100.000"))


@pytest.mark.asyncio
async def test_channel_thread_uses_only_workspace_attested_user_history_client() -> None:
    bot = _HistoryClient({("C1", ""): _page([])})
    user = _HistoryClient(
        {},
        reply_pages={
            ("C1", "100.000", ""): _page(
                [
                    _message("100.000", "root"),
                    _message("110.000", "prior reply", thread_ts="100.000"),
                ]
            )
        },
    )

    result = await SlackHistoryContextLoader(
        bot,
        user_history_client=user,
    ).load(_job(message_ts="120.000", thread_root_ts="100.000"))

    assert user.auth_calls == 1
    assert len(user.reply_calls) == 1
    assert bot.reply_calls == []
    assert result.manifest.thread_source == "slack_replies_user"


@pytest.mark.asyncio
async def test_wrong_workspace_user_history_is_never_queried() -> None:
    bot = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={("C1", "100.000", ""): PermissionError("missing_scope")},
    )
    wrong_workspace_user = _HistoryClient({}, auth_team_id="T-other")

    with pytest.raises(SlackThreadContextError, match="unavailable"):
        await SlackHistoryContextLoader(
            bot,
            user_history_client=wrong_workspace_user,
        ).load(_job(message_ts="120.000", thread_root_ts="100.000"))

    assert wrong_workspace_user.auth_calls == 1
    assert wrong_workspace_user.reply_calls == []
    assert wrong_workspace_user.calls == []
    assert bot.reply_calls == []


@pytest.mark.asyncio
async def test_dm_and_mpim_thread_replies_remain_on_bot_identity() -> None:
    bot = _HistoryClient(
        {("G1", ""): _page([])},
        reply_pages={
            ("G1", "100.000", ""): _page(
                [
                    _message("100.000", "group root"),
                    _message("110.000", "group reply", thread_ts="100.000"),
                ]
            )
        },
    )
    user = _HistoryClient({})

    result = await SlackHistoryContextLoader(bot, user_history_client=user).load(
        _job(
            channel_id="G1",
            conversation_kind=SlackConversationKind.MPIM,
            message_ts="120.000",
            thread_root_ts="100.000",
        )
    )

    assert len(bot.reply_calls) == 1
    assert user.auth_calls == 0
    assert user.reply_calls == []
    assert result.manifest.thread_source == "slack_replies_bot"


@pytest.mark.asyncio
async def test_failed_replies_uses_only_exact_attested_complete_persisted_thread() -> None:
    bot = _HistoryClient({("C1", ""): _page([])})
    user = _HistoryClient(
        {
            ("C1", ""): _page(
                [
                    _message(
                        "100.000",
                        "Remember amber hexagons.",
                        reply_count=2,
                        latest_reply="120.000",
                    )
                ]
            )
        },
        reply_pages={("C1", "100.000", ""): PermissionError("not_allowed_token_type")},
    )
    fallback = _ThreadFallback(_persisted_snapshot())

    result = await SlackHistoryContextLoader(
        bot,
        user_history_client=user,
        thread_fallback=fallback,
    ).load(_job(message_ts="120.000", thread_root_ts="100.000"))

    assert len(fallback.record_calls) == 1
    assert fallback.record_calls[0]["source"] is SlackThreadCoverageSource.USER_HISTORY
    assert fallback.record_calls[0]["raw_root"]["ts"] == "100.000"  # type: ignore[index]
    assert fallback.load_calls == [
        {
            "team_id": "T1",
            "channel_id": "C1",
            "thread_root_ts": "100.000",
            "current_message_ts": "120.000",
            "current_actor_id": "U-current",
            "current_event_id": "Ev-current",
            "max_messages": 200,
        }
    ]
    assert result.manifest.thread_source == "persisted_complete"
    assert result.manifest.thread_coverage_reason == "complete"
    assert result.manifest.thread_coverage_digest == "b" * 64
    assert result.manifest.thread_requests == 2
    assert [
        item.source_actor_id for item in result.items if item.id.startswith("slack-thread:")
    ] == ["U1", "leo"]


@pytest.mark.asyncio
async def test_progress_posted_before_history_load_uses_exact_prefix_at_current_boundary() -> None:
    root_ts = "1787412000.000001"
    prior_progress_ts = "1787412100.100001"
    prior_final_ts = "1787412200.200001"
    current_ts = "1787412219.905099"
    current_progress_ts = "1787412253.855439"
    rows = (
        _persisted_row(
            "root",
            root_ts,
            "Root request",
            actor_id="U-root",
            role="user",
            thread_root_ts=root_ts,
        ),
        _persisted_row(
            "prior-progress",
            prior_progress_ts,
            "Prior progress update",
            actor_id="leo",
            role="assistant",
            thread_root_ts=root_ts,
        ),
        _persisted_row(
            "prior-final",
            prior_final_ts,
            "Prior final answer",
            actor_id="leo",
            role="assistant",
            thread_root_ts=root_ts,
        ),
        _persisted_row(
            "current-user",
            current_ts,
            "Current follow-up must be the context boundary",
            actor_id="U-current",
            role="user",
            thread_root_ts=root_ts,
            event_id="Ev-current",
        ),
        _persisted_row(
            "current-progress",
            current_progress_ts,
            "Current progress must not leak backward into context",
            actor_id="leo",
            role="assistant",
            thread_root_ts=root_ts,
        ),
    )
    snapshot = _assess_thread_coverage(
        team_id="T1",
        channel_id="C1",
        thread_root_ts=root_ts,
        current_message_ts=current_ts,
        conversation_id="conversation-internal",
        rows=rows,
        authoritative_reply_count=4,
        authoritative_latest_reply_ts=current_progress_ts,
        coverage_source=SlackThreadCoverageSource.BOT_HISTORY,
        coverage_snapshot_hash="a" * 64,
        max_messages=200,
        current_actor_id="U-current",
        current_event_id="Ev-current",
    )
    bot = _HistoryClient(
        {
            ("C1", ""): _page(
                [
                    _message(
                        root_ts,
                        "Root request",
                        reply_count=4,
                        latest_reply=current_progress_ts,
                    )
                ]
            )
        },
        reply_pages={
            ("C1", root_ts, ""): AssertionError(
                "ordinary channel without a user token must not use bot conversations.replies"
            )
        },
    )
    fallback = _ThreadFallback(snapshot)

    result = await SlackHistoryContextLoader(bot, thread_fallback=fallback).load(
        _job(
            message_ts=current_ts,
            thread_root_ts=root_ts,
            prompt="Continue the prior request",
        )
    )

    thread_items = tuple(item for item in result.items if item.id.startswith("slack-thread:"))
    combined_context = "\n".join(item.content for item in thread_items)
    assert snapshot.complete is True
    assert snapshot.complete_through_ts == current_ts
    assert snapshot.authoritative_latest_reply_ts == current_progress_ts
    assert snapshot.persisted_message_count == 5
    assert result.manifest.thread_source == "persisted_complete"
    assert result.manifest.thread_complete is True
    assert result.manifest.thread_messages_loaded == 3
    assert len(thread_items) == 3
    assert "Root request" in combined_context
    assert "Prior progress update" in combined_context
    assert "Prior final answer" in combined_context
    assert "Current follow-up" not in combined_context
    assert "Current progress" not in combined_context
    assert bot.reply_calls == []


@pytest.mark.asyncio
async def test_channel_user_nonmember_falls_back_to_exact_bot_root_coverage_only() -> None:
    bot = _HistoryClient(
        {
            ("G-private", ""): _page(
                [
                    _message(
                        "100.000",
                        "Remember amber hexagons.",
                        reply_count=2,
                        latest_reply="120.000",
                    )
                ]
            )
        },
        reply_pages={
            ("G-private", "100.000", ""): AssertionError(
                "bot conversations.replies must not be used for a channel"
            )
        },
    )
    user = _HistoryClient(
        {("G-private", ""): PermissionError("not_in_channel")},
        reply_pages={("G-private", "100.000", ""): PermissionError("not_in_channel")},
    )
    snapshot = replace(
        _persisted_snapshot(channel_id="G-private"),
        coverage_source=SlackThreadCoverageSource.BOT_HISTORY,
    )
    fallback = _ThreadFallback(snapshot)

    result = await SlackHistoryContextLoader(
        bot,
        user_history_client=user,
        thread_fallback=fallback,
    ).load(
        _job(
            channel_id="G-private",
            message_ts="120.000",
            thread_root_ts="100.000",
        )
    )

    assert len(user.reply_calls) == 1
    assert len(user.calls) == 1
    assert bot.reply_calls == []
    assert fallback.record_calls[0]["source"] is SlackThreadCoverageSource.BOT_HISTORY
    assert fallback.record_calls[0]["channel_id"] == "G-private"
    assert fallback.record_calls[0]["team_id"] == "T1"
    assert fallback.record_calls[0]["current_message_ts"] == "120.000"
    assert result.manifest.thread_source == "persisted_complete"
    assert result.manifest.thread_requests == 3


@pytest.mark.asyncio
async def test_channel_user_and_bot_root_coverage_failure_stays_closed() -> None:
    bot = _HistoryClient(
        {("G-private", ""): PermissionError("missing_scope")},
        reply_pages={
            ("G-private", "100.000", ""): AssertionError(
                "bot conversations.replies must not be used for a channel"
            )
        },
    )
    user = _HistoryClient(
        {("G-private", ""): PermissionError("not_in_channel")},
        reply_pages={("G-private", "100.000", ""): PermissionError("not_in_channel")},
    )
    fallback = _ThreadFallback(
        _persisted_snapshot(
            complete=False,
            channel_id="G-private",
            reason=SlackThreadCoverageReason.COVERAGE_MISSING,
        )
    )

    with pytest.raises(SlackThreadContextError, match="unavailable"):
        await SlackHistoryContextLoader(
            bot,
            user_history_client=user,
            thread_fallback=fallback,
        ).load(
            _job(
                channel_id="G-private",
                message_ts="120.000",
                thread_root_ts="100.000",
            )
        )

    assert len(user.reply_calls) == 1
    assert len(user.calls) == 1
    assert len(bot.calls) == 1
    assert bot.reply_calls == []
    assert fallback.record_calls == []


@pytest.mark.asyncio
async def test_transient_user_auth_failure_is_retryable_across_loads() -> None:
    bot = _HistoryClient({("C1", ""): _page([])})
    user = _HistoryClient(
        {},
        auth_outcomes=[
            RuntimeError("temporary auth outage one"),
            RuntimeError("temporary auth outage two"),
            {"ok": True, "team_id": "T1"},
        ],
        reply_pages={
            ("C1", "100.000", ""): _page(
                [
                    _message("100.000", "root"),
                    _message("110.000", "prior reply", thread_ts="100.000"),
                ]
            )
        },
    )
    loader = SlackHistoryContextLoader(bot, user_history_client=user)
    job = _job(message_ts="120.000", thread_root_ts="100.000")

    with pytest.raises(SlackThreadContextError, match="unavailable"):
        await loader.load(job)
    result = await loader.load(job)

    assert user.auth_calls == 3
    assert len(user.reply_calls) == 1
    assert bot.reply_calls == []
    assert result.manifest.thread_source == "slack_replies_user"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "snapshot",
    [
        _persisted_snapshot(complete=False, reason=SlackThreadCoverageReason.COUNT_MISMATCH),
        _persisted_snapshot(team_id="T-other"),
        replace(_persisted_snapshot(), authoritative_reply_count=99),
        replace(_persisted_snapshot(), authoritative_latest_reply_ts="119.000"),
        replace(_persisted_snapshot(), boundary_actor_id="U-other"),
        replace(_persisted_snapshot(), boundary_event_id="Ev-other"),
        replace(
            _persisted_snapshot(),
            messages=(
                PersistedSlackThreadMessage(
                    id="message-root",
                    actor_id="U1",
                    role="user",
                    text="root",
                    message_ts="100.000",
                ),
                PersistedSlackThreadMessage(
                    id="message-current",
                    actor_id="U2",
                    role="user",
                    text="current must not leak",
                    message_ts="120.000",
                ),
            ),
        ),
    ],
)
async def test_incomplete_or_mismatched_persisted_thread_fails_closed(
    snapshot: PersistedSlackThreadSnapshot,
) -> None:
    bot = _HistoryClient(
        {("C1", ""): _page([_message("100.000", "root")])},
        reply_pages={("C1", "100.000", ""): PermissionError("missing_scope")},
    )
    fallback = _ThreadFallback(snapshot)

    with pytest.raises(SlackThreadContextError, match="unavailable"):
        await SlackHistoryContextLoader(bot, thread_fallback=fallback).load(
            _job(message_ts="120.000", thread_root_ts="100.000")
        )


@pytest.mark.asyncio
async def test_eighty_turn_question_and_progress_thread_compacts_without_protected_overflow() -> (
    None
):
    replies = [_message("100.000", "Root objective for the long thread")]
    for index in range(1, 80):
        if index % 5 == 1:
            text = f"What is the explanation for item {index}?"
            extra: dict[str, object] = {"thread_ts": "100.000"}
        elif index % 5 == 2:
            text = f"The material answer for item {index - 1} is value {index}. " + ("x" * 240)
            extra = {"thread_ts": "100.000", "bot_id": "BLEO"}
        elif index % 5 == 3:
            text = f"Still working on item {index}."
            extra = {"thread_ts": "100.000", "bot_id": "BLEO"}
        else:
            text = f"Supporting user detail {index}. " + ("y" * 240)
            extra = {"thread_ts": "100.000"}
        replies.append(_message(f"{100 + index}.000", text, **extra))
    client = _HistoryClient(
        {("C1", ""): _page([])},
        reply_pages={("C1", "100.000", ""): _page(replies)},
    )

    result = await SlackHistoryContextLoader(
        client,
        user_history_client=client,
        max_context_tokens=2_000,
        page_size=100,
    ).load(_job(message_ts="200.000", thread_root_ts="100.000"))

    assert result.manifest.thread_messages_loaded == 80
    assert result.manifest.thread_messages_compacted > 40
    assert len(result.manifest.protected_thread_item_ids) <= 16
    assert result.manifest.thread_reopen_handles
    assert result.reopen_ranges[0].items
    compacted_text = "\n".join(item.content for item in result.reopen_ranges[0].items)
    assert "Still working on item 3." in compacted_text
    assert "material answer for item 1" in compacted_text
