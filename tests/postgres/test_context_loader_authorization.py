from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import build_context_access_hash
from leo.persistence.context_loader import (
    ConversationContextAuthorizationError,
    ConversationContextRequest,
    PostgresConversationContextLoader,
)
from leo.persistence.schema import (
    ConversationAccessSnapshotRow,
    ConversationActorMembershipRow,
    ConversationRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
    ThreadRow,
)

SCOPE = ScopeKey(organization_id="org-context-auth", strategy_id="strategy-context-auth")


@pytest.fixture
def context_sessions(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> async_sessionmaker[AsyncSession]:
    return preserved_postgres_sessions


def _request(
    suffix: str,
    *,
    destination_id: str,
    destination_kind: str,
    projection: tuple[str, ...],
) -> ConversationContextRequest:
    return ConversationContextRequest(
        team_id="T-context",
        destination_id=destination_id,
        destination_kind=destination_kind,  # type: ignore[arg-type]
        actor_id="U-context",
        objective="recall the decision",
        current_task_id=f"task-context-{suffix}",
        current_event_id=f"event-context-{suffix}",
        current_message_ts="1.1",
        thread_root_ts="1.0",
        allowed_conversation_ids=projection,
        access_hash=build_context_access_hash(
            team_id="T-context",
            user_id="U-context",
            channel_id=destination_id,
            context_conversation_ids=projection,
        ),
        current_thread_namespace_id=f"slack:T-context:{destination_id}:1.0",
        max_turns=0,
        max_memories=0,
    )


async def _seed_authority(
    sessions: async_sessionmaker[AsyncSession],
    request: ConversationContextRequest,
) -> str:
    suffix = request.current_task_id.removeprefix("task-context-")
    event_id = f"event-context-{suffix}"
    conversation_id = f"conversation-context-{suffix}"
    source_kind = (
        "dm_membership_intersection"
        if request.destination_kind == "dm" and len(request.allowed_conversation_ids) > 1
        else "exact_destination"
    )
    conversation_kind = {
        "channel": "ordinary_internal",
        "dm": "dm",
        "group_dm": "mpim",
        "shared": "shared",
        "external": "external",
    }[request.destination_kind]
    async with sessions() as session, session.begin():
        for source_external_id in request.allowed_conversation_ids:
            is_destination = source_external_id == request.destination_id
            source_conversation_id = (
                conversation_id
                if is_destination
                else f"conversation-context-{suffix}-{source_external_id}"
            )
            session.add(
                ConversationRow(
                    id=source_conversation_id,
                    provider="slack",
                    team_id=request.team_id,
                    external_id=source_external_id,
                    kind=(request.destination_kind if is_destination else "channel"),
                    actor_id=(
                        request.actor_id
                        if is_destination and request.destination_kind == "dm"
                        else None
                    ),
                    external_provenance=(
                        "not_applicable"
                        if is_destination and request.destination_kind in {"dm", "group_dm"}
                        else request.destination_kind
                        if is_destination and request.destination_kind in {"shared", "external"}
                        else "internal"
                    ),
                    version=1,
                )
            )
            session.add(
                ConversationActorMembershipRow(
                    id=f"membership-context-{suffix}-{source_external_id}",
                    organization_id=SCOPE.organization_id,
                    team_id=request.team_id,
                    actor_id=request.actor_id,
                    conversation_external_id=source_external_id,
                    status="active",
                    source_kind=source_kind,
                    context_access_hash=request.access_hash,
                    version=1,
                    observed_at=datetime.now(UTC),
                )
            )
        await session.flush()
        session.add(
            ThreadRow(
                id=f"thread-context-{suffix}",
                organization_id=SCOPE.organization_id,
                strategy_id=SCOPE.strategy_id,
                origin_provider="slack",
                external_thread_id=request.current_thread_namespace_id,
                external_event_id=event_id,
                external_channel_id=request.destination_id,
                conversation_id=conversation_id,
                mapping_version=1,
                version=0,
            )
        )
        await session.flush()
        session.add(
            TaskRow(
                id=request.current_task_id,
                thread_id=f"thread-context-{suffix}",
                organization_id=SCOPE.organization_id,
                strategy_id=SCOPE.strategy_id,
                objective=request.objective,
                continuation_kind="root",
                mapping_version=1,
                status="queued",
                observation_ids=[],
                verifier_feedback=[],
                version=0,
                attempt_count=0,
            )
        )
        await session.flush()
        session.add(
            SlackIngressEventRow(
                event_id=event_id,
                team_id=request.team_id,
                channel_id=request.destination_id,
                user_id=request.actor_id,
                message_ts="1.1",
                thread_root_ts="1.0",
                conversation_key=request.current_thread_namespace_id,
                prompt=request.objective,
                conversation_kind=conversation_kind,
                external_provenance=(
                    "not_applicable"
                    if request.destination_kind in {"dm", "group_dm"}
                    else request.destination_kind
                    if request.destination_kind in {"shared", "external"}
                    else "internal"
                ),
                trigger_kind=("message_im" if request.destination_kind == "dm" else "app_mention"),
                context_conversation_ids=list(request.allowed_conversation_ids),
                context_access_hash=request.access_hash,
                context_projection_source=source_kind,
                conversation_id=conversation_id,
                organization_id=SCOPE.organization_id,
                strategy_id=SCOPE.strategy_id,
                mapping_version=1,
                status="queued",
                task_id=request.current_task_id,
                launch_status="queued",
                launch_attempt_count=1,
                attempt_count=0,
            )
        )
        # The access-snapshot foreign key is migration-enforced but intentionally
        # not exposed as an ORM relationship. Persist its authority event first so
        # SQLAlchemy cannot reorder the following bulk snapshot insert ahead of it.
        await session.flush()
        for position, external_id in enumerate(request.allowed_conversation_ids):
            session.add(
                ConversationAccessSnapshotRow(
                    id=f"snapshot-context-{suffix}-{position}",
                    ingress_event_id=event_id,
                    organization_id=SCOPE.organization_id,
                    team_id=request.team_id,
                    actor_id=request.actor_id,
                    destination_external_id=request.destination_id,
                    conversation_external_id=external_id,
                    position=position,
                    source_kind=source_kind,
                    context_access_hash=request.access_hash,
                    observed_at=datetime.now(UTC),
                )
            )
    return event_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_request",
    [
        _request(
            "valid-channel",
            destination_id="C-context",
            destination_kind="channel",
            projection=("C-context",),
        ),
        _request(
            "valid-dm",
            destination_id="D-context",
            destination_kind="dm",
            projection=("C-context", "D-context", "G-context"),
        ),
    ],
    ids=("channel", "dm-union"),
)
async def test_persisted_snapshot_authorizes_valid_projection(
    context_sessions: async_sessionmaker[AsyncSession],
    context_request: ConversationContextRequest,
) -> None:
    await _seed_authority(context_sessions, context_request)

    assert (
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, context_request) == ()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation", ["membership", "bot_presence"])
async def test_dm_source_revocation_fails_before_any_context_is_returned(
    context_sessions: async_sessionmaker[AsyncSession],
    revocation: str,
) -> None:
    request = _request(
        f"revoked-{revocation}",
        destination_id="D-revoked",
        destination_kind="dm",
        projection=("C-revoked", "D-revoked"),
    )
    await _seed_authority(context_sessions, request)
    async with context_sessions() as session, session.begin():
        if revocation == "membership":
            await session.execute(
                update(ConversationActorMembershipRow)
                .where(
                    ConversationActorMembershipRow.organization_id == SCOPE.organization_id,
                    ConversationActorMembershipRow.team_id == request.team_id,
                    ConversationActorMembershipRow.actor_id == request.actor_id,
                    ConversationActorMembershipRow.conversation_external_id == "C-revoked",
                )
                .values(status="revoked", version=2)
            )
        else:
            await session.execute(
                update(ConversationRow)
                .where(
                    ConversationRow.team_id == request.team_id,
                    ConversationRow.external_id == "C-revoked",
                )
                .values(bot_presence="absent", version=2)
            )

    with pytest.raises(ConversationContextAuthorizationError, match=r"revoked|presence"):
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, request)


@pytest.mark.asyncio
async def test_missing_persisted_snapshot_fails_closed(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    request = _request(
        "missing",
        destination_id="C-missing",
        destination_kind="channel",
        projection=("C-missing",),
    )
    event_id = await _seed_authority(context_sessions, request)
    async with context_sessions() as session, session.begin():
        await session.execute(
            delete(ConversationAccessSnapshotRow).where(
                ConversationAccessSnapshotRow.ingress_event_id == event_id
            )
        )

    with pytest.raises(ConversationContextAuthorizationError):
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, request)


@pytest.mark.asyncio
async def test_mismatched_persisted_authority_fails_closed(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    request = _request(
        "mismatch",
        destination_id="C-mismatch",
        destination_kind="channel",
        projection=("C-mismatch",),
    )
    event_id = await _seed_authority(context_sessions, request)
    async with context_sessions() as session, session.begin():
        await session.execute(
            update(SlackIngressEventRow)
            .where(SlackIngressEventRow.event_id == event_id)
            .values(user_id="U-forged")
        )

    with pytest.raises(ConversationContextAuthorizationError):
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, request)


@pytest.mark.asyncio
async def test_reordered_persisted_snapshot_fails_closed(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    request = _request(
        "reordered",
        destination_id="D-reordered",
        destination_kind="dm",
        projection=("C-reordered", "D-reordered", "G-reordered"),
    )
    event_id = await _seed_authority(context_sessions, request)
    async with context_sessions() as session, session.begin():
        await session.execute(
            update(ConversationAccessSnapshotRow)
            .where(
                ConversationAccessSnapshotRow.ingress_event_id == event_id,
                ConversationAccessSnapshotRow.position == 0,
            )
            .values(position=1)
        )
        await session.execute(
            update(ConversationAccessSnapshotRow)
            .where(
                ConversationAccessSnapshotRow.ingress_event_id == event_id,
                ConversationAccessSnapshotRow.conversation_external_id == "D-reordered",
            )
            .values(position=0)
        )

    with pytest.raises(ConversationContextAuthorizationError):
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, request)


@pytest.mark.asyncio
async def test_forged_projection_cannot_expand_persisted_authority(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    durable = _request(
        "forged",
        destination_id="D-forged",
        destination_kind="dm",
        projection=("C-forged", "D-forged"),
    )
    await _seed_authority(context_sessions, durable)
    forged = _request(
        "forged",
        destination_id="D-forged",
        destination_kind="dm",
        projection=("C-forged", "C9-forged", "D-forged"),
    )

    with pytest.raises(ConversationContextAuthorizationError):
        await PostgresConversationContextLoader(context_sessions).load(SCOPE, forged)


@pytest.mark.asyncio
async def test_thread_plane_is_exact_excludes_current_and_accepts_legacy_strategy_metadata(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    request = _request(
        "thread-guardrails",
        destination_id="C-thread-guardrails",
        destination_kind="channel",
        projection=("C-thread-guardrails",),
    )
    await _seed_authority(context_sessions, request)
    conversation_id = "conversation-context-thread-guardrails"
    harness_thread_id = "thread-context-thread-guardrails"
    recorded_at = datetime(2026, 8, 22, tzinfo=UTC)

    def row(
        row_id: str,
        *,
        external_event_id: str,
        provider_message_ts: str,
        text: str,
        role: str,
        selected_conversation_id: str = conversation_id,
    ) -> SanitizedMessageRow:
        return SanitizedMessageRow(
            id=row_id,
            organization_id=SCOPE.organization_id,
            strategy_id="legacy-non-gating-strategy",
            destination_id=request.destination_id,
            external_event_id=external_event_id,
            text=text,
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
            recorded_at=recorded_at,
            conversation_id=selected_conversation_id,
            harness_thread_id=harness_thread_id,
            actor_id="leo" if role == "assistant" else request.actor_id,
            role=role,
            provider_message_ts=provider_message_ts,
            context_access_hash=request.access_hash,
        )

    async with context_sessions() as session, session.begin():
        foreign_conversation_id = "conversation-context-thread-guardrails-foreign"
        session.add(
            ConversationRow(
                id=foreign_conversation_id,
                provider="slack",
                team_id=request.team_id,
                external_id="C-thread-foreign",
                kind="channel",
                external_provenance="internal",
                version=1,
            )
        )
        await session.flush()
        session.add_all(
            [
                row(
                    "message-thread-root",
                    external_event_id="event-thread-root",
                    provider_message_ts=request.thread_root_ts,
                    text="Root objective",
                    role="user",
                ),
                row(
                    "message-thread-outcome",
                    external_event_id="event-thread-outcome",
                    provider_message_ts="1.05",
                    text="The verified tool-backed result completed.",
                    role="assistant",
                ),
                row(
                    "message-thread-foreign",
                    external_event_id="event-thread-foreign",
                    provider_message_ts="1.04",
                    text="foreign conversation secret",
                    role="user",
                    selected_conversation_id=foreign_conversation_id,
                ),
                row(
                    "message-thread-current",
                    external_event_id=request.current_event_id,
                    provider_message_ts=request.current_message_ts,
                    text="current event duplicate",
                    role="user",
                ),
            ]
        )

    result = await PostgresConversationContextLoader(context_sessions).load_authorized(
        SCOPE,
        request,
    )

    combined = "\n".join(item.content for item in result.items)
    assert "Root objective" in combined
    assert "verified tool-backed result" in combined
    assert "foreign conversation secret" not in combined
    assert "current event duplicate" not in combined
    assert result.manifest.current_event_id == request.current_event_id
    assert result.manifest.thread_root_ts == request.thread_root_ts
    assert set(result.manifest.protected_thread_item_ids) == {
        "thread-message:message-thread-root",
        "thread-message:message-thread-outcome",
    }


@pytest.mark.asyncio
async def test_task_turn_plane_is_current_thread_only_even_for_dm_membership_union(
    context_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = "task-turn-isolation"
    request = _request(
        suffix,
        destination_id="D-task-turn-isolation",
        destination_kind="dm",
        projection=(
            "C-task-turn-isolation",
            "D-task-turn-isolation",
            "G-task-turn-isolation",
        ),
    ).model_copy(update={"max_turns": 10})
    await _seed_authority(context_sessions, request)
    current_thread_id = f"thread-context-{suffix}"
    destination_conversation_id = f"conversation-context-{suffix}"
    union_conversation_id = f"conversation-context-{suffix}-C-task-turn-isolation"

    def task(
        row_id: str,
        *,
        thread_id: str,
        objective: str,
        answer: str,
    ) -> TaskRow:
        return TaskRow(
            id=row_id,
            thread_id=thread_id,
            organization_id=SCOPE.organization_id,
            strategy_id=SCOPE.strategy_id,
            objective=objective,
            continuation_kind="follow_up",
            mapping_version=1,
            status="completed",
            observation_ids=[],
            verifier_feedback=[],
            final_output=answer,
            version=1,
            attempt_count=1,
        )

    async with context_sessions() as session, session.begin():
        session.add_all(
            [
                ThreadRow(
                    id=f"thread-context-{suffix}-old-dm",
                    organization_id=SCOPE.organization_id,
                    strategy_id=SCOPE.strategy_id,
                    origin_provider="slack",
                    external_thread_id=("slack:T-context:D-task-turn-isolation:old-dm-root"),
                    external_event_id=f"event-context-{suffix}-old-dm",
                    external_channel_id=request.destination_id,
                    conversation_id=destination_conversation_id,
                    mapping_version=1,
                    version=0,
                ),
                ThreadRow(
                    id=f"thread-context-{suffix}-union-channel",
                    organization_id=SCOPE.organization_id,
                    strategy_id=SCOPE.strategy_id,
                    origin_provider="slack",
                    external_thread_id=("slack:T-context:C-task-turn-isolation:old-channel-root"),
                    external_event_id=f"event-context-{suffix}-union-channel",
                    external_channel_id="C-task-turn-isolation",
                    conversation_id=union_conversation_id,
                    mapping_version=1,
                    version=0,
                ),
            ]
        )
        await session.flush()
        session.add_all(
            [
                task(
                    f"task-context-{suffix}-same-thread",
                    thread_id=current_thread_id,
                    objective="What did we decide in this exact DM thread?",
                    answer="same-thread verified outcome",
                ),
                task(
                    f"task-context-{suffix}-old-dm",
                    thread_id=f"thread-context-{suffix}-old-dm",
                    objective="Unrelated earlier DM root",
                    answer="old-DM-thread secret",
                ),
                task(
                    f"task-context-{suffix}-union-channel",
                    thread_id=f"thread-context-{suffix}-union-channel",
                    objective="Unrelated channel in the DM membership union",
                    answer="membership-union channel secret",
                ),
            ]
        )

    result = await PostgresConversationContextLoader(context_sessions).load_authorized(
        SCOPE,
        request,
    )

    combined = "\n".join(item.content for item in result.items)
    assert "same-thread verified outcome" in combined
    assert "old-DM-thread secret" not in combined
    assert "membership-union channel secret" not in combined
    assert tuple(item.id for item in result.items) == (f"turn:task-context-{suffix}-same-thread",)
