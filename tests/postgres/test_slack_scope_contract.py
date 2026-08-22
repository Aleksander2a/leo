from __future__ import annotations

from hashlib import sha256
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import DataError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    SlackBotPresence,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.persistence.schema import ConversationRow, SlackChannelScopeRow, SlackIngressEventRow
from leo.persistence.slack_ingress import PostgresSlackIngressAdmission


def _event_id(label: str, suffix: str) -> str:
    """Return a readable fixture ID that stays below every provider/DB ID bound."""

    return f"Ev-{label[:12]}-{suffix[:12]}"


def _slack_timestamp(event_id: str, microseconds: int) -> str:
    epoch = int.from_bytes(sha256(event_id.encode("utf-8")).digest()[:6], "big")
    return f"{epoch}.{microseconds:06d}"


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _job(
    event_id: str,
    *,
    team_id: str,
    channel_id: str,
    user_id: str,
    kind: SlackConversationKind = SlackConversationKind.ORDINARY_INTERNAL,
    external_provenance: SlackExternalProvenance = SlackExternalProvenance.INTERNAL,
) -> SlackMentionJob:
    thread_root_ts = _slack_timestamp(event_id, 0)
    message_ts = _slack_timestamp(event_id, 1)
    return SlackMentionJob(
        event_id=event_id,
        team_id=team_id,
        channel_id=channel_id,
        user_id=user_id,
        message_ts=message_ts,
        thread_root_ts=thread_root_ts,
        conversation_key=f"slack:{team_id}:{channel_id}:{thread_root_ts}",
        prompt="quote NVDA",
        conversation_kind=kind,
        trigger_kind=(
            SlackTriggerKind.MESSAGE_IM
            if kind is SlackConversationKind.DM
            else SlackTriggerKind.APP_MENTION
        ),
        context_conversation_ids=(channel_id,),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=external_provenance,
        context_access_hash=build_context_access_hash(
            team_id=team_id,
            user_id=user_id,
            channel_id=channel_id,
            context_conversation_ids=(channel_id,),
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "external_provenance", "persisted_kind"),
    [
        (
            SlackConversationKind.ORDINARY_INTERNAL,
            SlackExternalProvenance.INTERNAL,
            "channel",
        ),
        (SlackConversationKind.DM, SlackExternalProvenance.NOT_APPLICABLE, "dm"),
        (SlackConversationKind.MPIM, SlackExternalProvenance.NOT_APPLICABLE, "group_dm"),
        (SlackConversationKind.SHARED, SlackExternalProvenance.SHARED, "shared"),
        (SlackConversationKind.EXTERNAL, SlackExternalProvenance.EXTERNAL, "external"),
    ],
)
async def test_every_authoritative_kind_persists_exact_non_gating_authority_snapshot(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    kind: SlackConversationKind,
    external_provenance: SlackExternalProvenance,
    persisted_kind: str,
) -> None:
    suffix = uuid4().hex
    team_id = f"T{suffix[:12].upper()}"
    channel_id = f"C{suffix[12:24].upper()}"
    user_id = f"U{suffix[20:32].upper()}"
    job = _job(
        _event_id("authority", suffix),
        team_id=team_id,
        channel_id=channel_id,
        user_id=user_id,
        kind=kind,
        external_provenance=external_provenance,
    )
    eligibility = SlackConversationEligibility(
        kind=kind,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=external_provenance,
    )

    admitted = await PostgresSlackIngressAdmission(preserved_postgres_sessions).admit(
        job,
        ScopeKey(organization_id="org-authority", strategy_id="strategy-optional"),
        eligibility=eligibility,
    )

    assert admitted is not None
    async with preserved_postgres_sessions() as session:
        ingress = await session.get(SlackIngressEventRow, job.event_id)
        conversation = await session.scalar(
            select(ConversationRow).where(
                ConversationRow.team_id == job.team_id,
                ConversationRow.external_id == channel_id,
            )
        )
    assert ingress is not None
    assert conversation is not None
    assert ingress.bot_presence == conversation.bot_presence == "present"
    assert ingress.conversation_lifecycle == conversation.lifecycle == "active"
    assert ingress.external_provenance == conversation.external_provenance
    assert ingress.external_provenance == external_provenance.value
    assert ingress.membership_policy_version == conversation.membership_policy_version == 1
    assert conversation.kind == persisted_kind


@pytest.mark.asyncio
async def test_repeated_admission_is_independent_of_legacy_mapping(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex
    team_id = f"T{suffix[:12].upper()}"
    channel_id = f"C{suffix[12:24].upper()}"
    user_id = f"U{suffix[20:32].upper()}"
    event_ids = (_event_id("a", suffix), _event_id("b", suffix))
    admission = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    outcomes = (
        await admission.admit(
            _job(
                event_ids[0],
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
            ),
            ScopeKey(organization_id="org-a", strategy_id="strategy-a"),
            eligibility=_eligibility(),
        ),
        await admission.admit(
            _job(
                event_ids[1],
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
            ),
            ScopeKey(organization_id="org-b", strategy_id="strategy-b"),
            eligibility=_eligibility(),
        ),
    )

    assert {outcome.resolution.mapping_version for outcome in outcomes} == {1}
    assert all(not outcome.resolution.provisioned for outcome in outcomes)
    assert outcomes[0].resolution.scope == ScopeKey(
        organization_id="org-a", strategy_id="strategy-a"
    )
    assert outcomes[1].resolution.scope == ScopeKey(
        organization_id="org-b", strategy_id="strategy-b"
    )

    async with preserved_postgres_sessions() as session:
        mapping = await session.scalar(
            select(SlackChannelScopeRow).where(
                SlackChannelScopeRow.team_id == team_id,
                SlackChannelScopeRow.channel_id == channel_id,
            )
        )
        events = list(
            (
                await session.scalars(
                    select(SlackIngressEventRow)
                    .where(SlackIngressEventRow.event_id.in_(event_ids))
                    .order_by(SlackIngressEventRow.event_id)
                )
            ).all()
        )

    assert mapping is None
    assert len(events) == 2
    assert all(event.status == "received" for event in events)
    assert [event.organization_id for event in events] == ["org-a", "org-b"]
    assert [event.strategy_id for event in events] == ["strategy-a", "strategy-b"]
    assert all(event.mapping_version == 1 for event in events)
    assert all(event.conversation_kind == "ordinary_internal" for event in events)
    assert all(event.context_conversation_ids == [channel_id] for event in events)


@pytest.mark.asyncio
async def test_invalid_admission_rolls_back_reservation_and_mapping(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex
    team_id = f"T{suffix[:12].upper()}"
    user_id = f"U{suffix[20:32].upper()}"
    event_id = _event_id("rollback", suffix)
    invalid_channel_id = f" C{suffix[12:24].upper()}"
    admission = PostgresSlackIngressAdmission(preserved_postgres_sessions)

    with pytest.raises(ValueError, match="channel_id"):
        await admission.admit(
            _job(
                event_id,
                team_id=team_id,
                channel_id=invalid_channel_id,
                user_id=user_id,
            ),
            ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
            eligibility=_eligibility(),
        )

    async with preserved_postgres_sessions() as session:
        event = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
        mapping = await session.scalar(
            select(SlackChannelScopeRow).where(
                SlackChannelScopeRow.team_id == team_id,
                SlackChannelScopeRow.channel_id == invalid_channel_id,
            )
        )

    assert event is None
    assert mapping is None


@pytest.mark.asyncio
async def test_revoked_legacy_mapping_does_not_reject_admission(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex
    team_id = f"T{suffix[:12].upper()}"
    channel_id = f"C{suffix[12:24].upper()}"
    user_id = f"U{suffix[20:32].upper()}"
    event_id = _event_id("policy", suffix)
    async with preserved_postgres_sessions() as session, session.begin():
        session.add(
            SlackChannelScopeRow(
                team_id=team_id,
                channel_id=channel_id,
                organization_id="org-demo",
                strategy_id="strategy-demo",
                status="revoked",
                provisioned_by_user_id=user_id,
                provisioned_via="test",
                version=1,
            )
        )

    admission = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    admitted = await admission.admit(
        _job(
            event_id,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
        ),
        ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
        eligibility=_eligibility(),
    )

    assert admitted is not None
    async with preserved_postgres_sessions() as session:
        event = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
    assert event is not None
    assert event.status == "received"
    assert event.last_error is None
    assert event.organization_id == "org-default"
    assert event.strategy_id == "strategy-default"
    assert event.mapping_version == 1


@pytest.mark.asyncio
async def test_database_failure_rolls_back_reservation_and_mapping(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex
    team_id = f"T{suffix[:12].upper()}"
    channel_id = f"C{suffix[12:24].upper()}"
    user_id = f"U{suffix[20:32].upper()}"
    event_id = _event_id("db-failure", suffix)
    admission = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    oversized_scope = ScopeKey(
        organization_id="org-demo",
        strategy_id="strategy-" + ("x" * 64),
    )

    with pytest.raises(DataError):
        await admission.admit(
            _job(
                event_id,
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
            ),
            oversized_scope,
            eligibility=_eligibility(),
        )

    async with preserved_postgres_sessions() as session:
        event = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
        mapping = await session.scalar(
            select(SlackChannelScopeRow).where(
                SlackChannelScopeRow.team_id == team_id,
                SlackChannelScopeRow.channel_id == channel_id,
            )
        )
    assert event is None
    assert mapping is None
