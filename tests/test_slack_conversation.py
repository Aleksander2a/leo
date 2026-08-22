from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    ScopeKey,
)
from leo.harness.thread_context import ThreadContextRange, thread_context_source_digest
from leo.integrations.fake import FixedClock
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackConversationKind,
    SlackLaunchRef,
    SlackMentionJob,
    SlackScopeResolution,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.memory.models import MemoryVisibility
from leo.memory.navigation import ProgressiveMemorySearchResult, membership_snapshot_hash
from leo.persistence.context_loader import (
    AuthorizedConversationContext,
    ConversationContextManifest,
)
from leo.persistence.task_leases import TaskLease
from leo.worker.slack_conversation import _merge_authorized_context, run_admitted_slack_conversation

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


class _ContextLoader:
    def __init__(self, sessions: object) -> None:
        del sessions

    async def load_authorized(self, scope: ScopeKey, request: Any) -> AuthorizedConversationContext:
        assert scope == SCOPE
        return AuthorizedConversationContext(
            items=(),
            manifest=ConversationContextManifest(
                access_hash=request.access_hash,
                membership_hash=membership_snapshot_hash(request.allowed_conversation_ids),
                allowed_conversation_ids=request.allowed_conversation_ids,
                harness_thread_id="thread-1",
                external_provenance="internal",
                membership_policy_version=1,
                item_ids=(),
                current_event_id=request.current_event_id,
                thread_root_ts=request.thread_root_ts,
            ),
        )


class _ProgressiveMemoryService:
    def __init__(self, sessions: object) -> None:
        del sessions

    async def search(self, *args: object, **kwargs: object) -> ProgressiveMemorySearchResult:
        del args, kwargs
        return ProgressiveMemorySearchResult(
            items=(),
            query_hash="a" * 64,
            selected_count=0,
            cache_status="miss",
        )


@pytest.fixture(autouse=True)
def _memory_runtime_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "leo.worker.slack_conversation.PostgresProgressiveMemoryService",
        _ProgressiveMemoryService,
    )


def _admitted(
    prompt: str,
    *,
    kind: SlackConversationKind = SlackConversationKind.ORDINARY_INTERNAL,
) -> AdmittedSlackMention:
    channel_id = "D1" if kind is SlackConversationKind.DM else "C1"
    projection = (channel_id,)
    job = SlackMentionJob(
        event_id="Ev1",
        team_id="T1",
        channel_id=channel_id,
        user_id="U1",
        message_ts="1710000000.001",
        thread_root_ts="1710000000.001",
        conversation_key=f"slack:T1:{channel_id}:1710000000.001",
        prompt=prompt,
        conversation_kind=kind,
        trigger_kind=(
            SlackTriggerKind.MESSAGE_IM
            if kind is SlackConversationKind.DM
            else SlackTriggerKind.APP_MENTION
        ),
        context_conversation_ids=projection,
        context_access_hash=build_context_access_hash(
            team_id="T1",
            user_id="U1",
            channel_id=channel_id,
            context_conversation_ids=projection,
        ),
    )
    return AdmittedSlackMention(
        job=job,
        resolution=SlackScopeResolution(scope=SCOPE, mapping_version=1, provisioned=True),
        launch=SlackLaunchRef(thread_id="thread-1", task_id="task-1", run_id="run-1"),
    )


def test_context_merge_preserves_pinned_thread_turn_before_live_composition() -> None:
    root = ContextItem(
        id="thread-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Exact Slack root",
        conversation_id="C1",
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    supporting = ContextItem(
        id="background",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="background " * 1_000,
        conversation_id="C1",
        budget_priority=1,
    )

    assert _merge_authorized_context(
        (root, supporting),
        allowed_conversation_ids=frozenset({"C1"}),
        destination_id="C1",
        team_id="T1",
        thread_root_ts="100.000",
        actor_id="U1",
        max_tokens=32,
    ) == (root,)


def test_context_merge_collapses_only_equivalent_authoritative_root_pair() -> None:
    durable_root = ContextItem(
        id="thread-message:root-row",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: What are some interesting investing opportunities right now?",
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    slack_root = ContextItem(
        id="slack-thread:T1:C1:100.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=100.000; "
            "author=U1; author_kind=user]\n"
            "<@ULEO> What are some interesting investing opportunities right now?"
        ),
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    prior_outcome = ContextItem(
        id="slack-thread:T1:C1:110.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Which goals, risk tolerance, and time horizon should I use?",
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.UNRESOLVED_QUESTION,
        budget_priority=98,
    )

    merged = _merge_authorized_context(
        (durable_root, slack_root, prior_outcome),
        allowed_conversation_ids=frozenset({"C1"}),
        destination_id="C1",
        team_id="T1",
        thread_root_ts="100.000",
        actor_id="U1",
    )

    assert merged == (durable_root, prior_outcome)
    assert sum(item.retention is ContextItemRetention.THREAD_ROOT for item in merged) == 1


@pytest.mark.parametrize(
    ("durable_update", "slack_update", "error"),
    [
        ({"content": "User: Different root"}, {}, "content mismatch"),
        ({"source_actor_id": "U-other"}, {}, "actor mismatch"),
        ({}, {"id": "slack-thread:T1:C1:99.000"}, "identity mismatch"),
        ({}, {"conversation_id": "D-other"}, "unauthorized conversation"),
    ],
)
def test_context_merge_rejects_conflicting_authoritative_root_pair(
    durable_update: dict[str, object],
    slack_update: dict[str, object],
    error: str,
) -> None:
    durable_root = ContextItem(
        id="thread-message:root-row",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: Exact root",
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
    ).model_copy(update=durable_update)
    slack_root = ContextItem(
        id="slack-thread:T1:C1:100.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=100.000; "
            "author=U1; author_kind=user]\n<@ULEO> Exact root"
        ),
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
    ).model_copy(update=slack_update)

    with pytest.raises(ValueError, match=error):
        _merge_authorized_context(
            (durable_root, slack_root),
            allowed_conversation_ids=frozenset({"C1"}),
            destination_id="C1",
            team_id="T1",
            thread_root_ts="100.000",
            actor_id="U1",
        )


def _lease() -> TaskLease:
    return TaskLease(
        task_id="task-1",
        owner="worker",
        token="lease-token",
        attempt=1,
        expires_at=datetime(2026, 8, 21, 12, 0, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_admitted_slack_explicit_memory_command_binds_exact_runtime_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    expected_result = object()

    async def fake_run_live_conversation(**kwargs: Any) -> object:
        captured.update(kwargs)
        return expected_result

    monkeypatch.setattr(
        "leo.worker.slack_conversation.PostgresConversationContextLoader",
        _ContextLoader,
    )
    monkeypatch.setattr(
        "leo.worker.slack_conversation.run_live_conversation",
        fake_run_live_conversation,
    )
    sessions = cast(async_sessionmaker[AsyncSession], object())
    omitted = ContextItem(
        id="omitted-thread-turn",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Exact progressively reopenable detail.",
        conversation_id="C1",
    )
    source_range = ThreadContextRange(
        handle="thr_" + ("a" * 32),
        digest=thread_context_source_digest((omitted,)),
        items=(omitted,),
    )
    provider_gates = ProviderGateRegistry(FixedClock(datetime(2026, 8, 21, tzinfo=UTC)))
    async with httpx.AsyncClient() as client:
        result = await run_admitted_slack_conversation(
            settings=Settings(_env_file=None),
            client=client,
            sessions=sessions,
            admitted=_admitted("remember that the demo is called Helios"),
            lease=_lease(),
            thread_context_ranges=(source_range,),
            provider_gates=provider_gates,
        )

    assert result is expected_result
    authority = captured["memory_authority"]
    assert authority.scope == SCOPE
    assert authority.actor_id == "U1"
    assert authority.event_id == "Ev1"
    assert authority.task_id == "task-1"
    assert authority.run_id == "run-1"
    assert authority.message_reference == "1710000000.001"
    assert authority.destination.external_id == "C1"
    assert authority.visibility is MemoryVisibility.CONVERSATION_LOCAL
    assert authority.namespace_id == "C1"
    assert [tool.spec.name for tool in captured["thread_context_tools"]] == ["thread_context.open"]
    assert captured["provider_gates"] is provider_gates


@pytest.mark.asyncio
async def test_admitted_slack_ordinary_conversation_does_not_expose_memory_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_live_conversation(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "leo.worker.slack_conversation.PostgresConversationContextLoader",
        _ContextLoader,
    )
    monkeypatch.setattr(
        "leo.worker.slack_conversation.run_live_conversation",
        fake_run_live_conversation,
    )
    sessions = cast(async_sessionmaker[AsyncSession], object())
    async with httpx.AsyncClient() as client:
        await run_admitted_slack_conversation(
            settings=Settings(_env_file=None),
            client=client,
            sessions=sessions,
            admitted=_admitted("What did we call the demo?"),
            lease=_lease(),
        )

    assert captured["memory_authority"] is None


@pytest.mark.asyncio
async def test_admitted_slack_dm_memory_authority_is_actor_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_run_live_conversation(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        "leo.worker.slack_conversation.PostgresConversationContextLoader",
        _ContextLoader,
    )
    monkeypatch.setattr(
        "leo.worker.slack_conversation.run_live_conversation",
        fake_run_live_conversation,
    )
    sessions = cast(async_sessionmaker[AsyncSession], object())
    async with httpx.AsyncClient() as client:
        await run_admitted_slack_conversation(
            settings=Settings(_env_file=None),
            client=client,
            sessions=sessions,
            admitted=_admitted("remember my private preference", kind=SlackConversationKind.DM),
            lease=_lease(),
        )

    authority = captured["memory_authority"]
    assert authority.visibility is MemoryVisibility.ACTOR_PRIVATE
    assert authority.namespace_id == "U1"
