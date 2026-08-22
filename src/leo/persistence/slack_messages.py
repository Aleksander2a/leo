"""Passive Slack message-plane persistence and fail-closed thread coverage proofs."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Protocol

from sqlalchemy import or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    SlackConversationKind,
    SlackPassiveMessage,
    SlackPassiveMessageRole,
)
from leo.persistence.conversation_plane import (
    ConversationMessageRole,
    ConversationPlaneMessage,
    build_conversation_plane_message,
    canonical_slack_conversation_id,
    persist_conversation_plane_message,
)
from leo.persistence.schema import (
    ConversationRow,
    SanitizedMessageRow,
    SlackThreadCoverageRow,
    ThreadRow,
)

_SLACK_TS = re.compile(r"^[0-9]+\.[0-9]+$")


class SlackThreadCoverageReason(StrEnum):
    COMPLETE = "complete"
    CONVERSATION_MISSING = "conversation_missing"
    COVERAGE_MISSING = "coverage_missing"
    INVALID_TIMESTAMP = "invalid_timestamp"
    BOUNDED_LIMIT = "bounded_limit"
    METADATA_AFTER_BOUNDARY = "metadata_after_boundary"
    BOUNDARY_NOT_ATTESTED = "boundary_not_attested"
    ROOT_MISSING = "root_missing"
    COUNT_MISMATCH = "count_mismatch"
    LATEST_MISMATCH = "latest_mismatch"
    DUPLICATE_PROVIDER_TIMESTAMP = "duplicate_provider_timestamp"


class SlackThreadCoverageSource(StrEnum):
    """Read identity that supplied an exact conversations.history root snapshot."""

    BOT_HISTORY = "slack_conversations_history_bot"
    USER_HISTORY = "slack_conversations_history_user"


@dataclass(frozen=True, slots=True)
class PersistedSlackThreadMessage:
    id: str
    actor_id: str
    role: str
    text: str
    message_ts: str


@dataclass(frozen=True, slots=True)
class PersistedSlackThreadSnapshot:
    """Exact persisted prefix plus an independently derived whole-thread proof.

    ``messages`` is always strictly earlier than ``current_message_ts``. A complete
    snapshot may have an authoritative latest reply after that boundary, but only when
    the full persisted row count/latest tuple is exact and the unique boundary row is
    the expected user ingress event.
    """

    team_id: str
    channel_id: str
    thread_root_ts: str
    current_message_ts: str
    conversation_id: str | None
    messages: tuple[PersistedSlackThreadMessage, ...]
    complete: bool
    coverage_reason: SlackThreadCoverageReason
    authoritative_reply_count: int | None
    authoritative_latest_reply_ts: str | None
    coverage_source: SlackThreadCoverageSource | None
    coverage_snapshot_hash: str | None
    complete_through_ts: str | None
    coverage_digest: str
    persisted_message_count: int | None = None
    boundary_attested: bool = False
    boundary_actor_id: str | None = None
    boundary_event_id: str | None = None


class SlackThreadContextFallback(Protocol):
    async def record_root_coverage(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        raw_root: Mapping[str, object],
        source: SlackThreadCoverageSource,
    ) -> bool: ...

    async def load_complete_thread(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        current_actor_id: str,
        current_event_id: str,
        max_messages: int = 500,
    ) -> PersistedSlackThreadSnapshot: ...


class PostgresSlackMessagePlane:
    """Short-transaction passive writes and bounded exact-thread fallback reads."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record_passive_message(
        self,
        message: SlackPassiveMessage,
        default_scope: ScopeKey,
    ) -> None:
        observed_at = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            conversation = await _ensure_passive_conversation(session, message)
            harness_thread_id = await session.scalar(
                select(ThreadRow.id).where(
                    ThreadRow.origin_provider == "slack",
                    ThreadRow.external_thread_id
                    == _slack_thread_key(
                        message.team_id,
                        message.channel_id,
                        message.thread_root_ts,
                    ),
                    ThreadRow.conversation_id == conversation.id,
                )
            )
            plane_message = build_conversation_plane_message(
                scope=default_scope,
                conversation_id=conversation.id,
                harness_thread_id=harness_thread_id,
                destination_id=message.channel_id,
                external_event_id=message.event_id,
                actor_id=message.actor_id,
                role=(
                    ConversationMessageRole.ASSISTANT
                    if message.role is SlackPassiveMessageRole.ASSISTANT
                    else ConversationMessageRole.USER
                ),
                provider_message_ts=message.message_ts,
                provider_thread_root_ts=message.thread_root_ts,
                context_access_hash=None,
                text=message.text,
                recorded_at=observed_at,
            )
            if message.role is SlackPassiveMessageRole.ASSISTANT:
                reconciled = await _reconcile_assistant_delivery(
                    session,
                    plane_message=plane_message,
                )
                if not reconciled:
                    await persist_conversation_plane_message(session, plane_message)
            else:
                await persist_conversation_plane_message(session, plane_message)

    async def record_root_coverage(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        raw_root: Mapping[str, object],
        source: SlackThreadCoverageSource,
    ) -> bool:
        """Persist coverage only from an exact conversations.history root snapshot."""

        metadata = _validate_authoritative_root_snapshot(
            team_id=team_id,
            channel_id=channel_id,
            thread_root_ts=thread_root_ts,
            current_message_ts=current_message_ts,
            raw_root=raw_root,
            source=source,
        )
        if metadata is None:
            return False
        reply_count, latest_reply_ts, snapshot_hash = metadata
        observed_at = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            conversation = await session.scalar(
                select(ConversationRow).where(
                    ConversationRow.provider == "slack",
                    ConversationRow.team_id == team_id,
                    ConversationRow.external_id == channel_id,
                )
            )
            if conversation is None:
                return False
            root_row = await session.scalar(
                select(SanitizedMessageRow).where(
                    SanitizedMessageRow.conversation_id == conversation.id,
                    SanitizedMessageRow.provider_thread_root_ts == thread_root_ts,
                    SanitizedMessageRow.provider_message_ts == thread_root_ts,
                )
            )
            if root_row is None:
                return False
            await persist_slack_thread_coverage(
                session,
                conversation_id=conversation.id,
                team_id=team_id,
                channel_id=channel_id,
                thread_root_ts=thread_root_ts,
                authoritative_reply_count=reply_count,
                authoritative_latest_reply_ts=latest_reply_ts,
                authority_source=source,
                authority_snapshot_hash=snapshot_hash,
                observed_at=observed_at,
            )
        return True

    async def load_complete_thread(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        current_actor_id: str,
        current_event_id: str,
        max_messages: int = 500,
    ) -> PersistedSlackThreadSnapshot:
        _validate_thread_lookup(
            team_id=team_id,
            channel_id=channel_id,
            thread_root_ts=thread_root_ts,
            current_message_ts=current_message_ts,
            max_messages=max_messages,
        )
        if not current_actor_id.strip() or not current_event_id.strip():
            raise ValueError("Slack current-message authority IDs must be non-empty")
        async with self._sessions() as session:
            conversation = await session.scalar(
                select(ConversationRow).where(
                    ConversationRow.provider == "slack",
                    ConversationRow.team_id == team_id,
                    ConversationRow.external_id == channel_id,
                )
            )
            if conversation is None:
                return _empty_snapshot(
                    team_id=team_id,
                    channel_id=channel_id,
                    thread_root_ts=thread_root_ts,
                    current_message_ts=current_message_ts,
                    reason=SlackThreadCoverageReason.CONVERSATION_MISSING,
                )
            coverage = await session.scalar(
                select(SlackThreadCoverageRow).where(
                    SlackThreadCoverageRow.conversation_id == conversation.id,
                    SlackThreadCoverageRow.team_id == team_id,
                    SlackThreadCoverageRow.channel_id == channel_id,
                    SlackThreadCoverageRow.thread_root_ts == thread_root_ts,
                )
            )
            if coverage is None:
                return _empty_snapshot(
                    team_id=team_id,
                    channel_id=channel_id,
                    thread_root_ts=thread_root_ts,
                    current_message_ts=current_message_ts,
                    conversation_id=conversation.id,
                    reason=SlackThreadCoverageReason.COVERAGE_MISSING,
                )
            if coverage.authoritative_reply_count + 1 > max_messages:
                return _snapshot(
                    team_id=team_id,
                    channel_id=channel_id,
                    thread_root_ts=thread_root_ts,
                    current_message_ts=current_message_ts,
                    conversation_id=conversation.id,
                    messages=(),
                    reason=SlackThreadCoverageReason.BOUNDED_LIMIT,
                    authoritative_reply_count=coverage.authoritative_reply_count,
                    authoritative_latest_reply_ts=coverage.authoritative_latest_reply_ts,
                    coverage_source=SlackThreadCoverageSource(coverage.authority_source),
                    coverage_snapshot_hash=coverage.authority_snapshot_hash,
                )
            rows = tuple(
                (
                    await session.scalars(
                        select(SanitizedMessageRow)
                        .where(
                            SanitizedMessageRow.conversation_id == conversation.id,
                            SanitizedMessageRow.provider_thread_root_ts == thread_root_ts,
                            SanitizedMessageRow.provider_message_ts.is_not(None),
                        )
                        .order_by(
                            SanitizedMessageRow.provider_message_ts,
                            SanitizedMessageRow.id,
                        )
                        .limit(max_messages + 1)
                    )
                ).all()
            )
        return _assess_thread_coverage(
            team_id=team_id,
            channel_id=channel_id,
            thread_root_ts=thread_root_ts,
            current_message_ts=current_message_ts,
            conversation_id=conversation.id,
            rows=rows,
            authoritative_reply_count=coverage.authoritative_reply_count,
            authoritative_latest_reply_ts=coverage.authoritative_latest_reply_ts,
            coverage_source=SlackThreadCoverageSource(coverage.authority_source),
            coverage_snapshot_hash=coverage.authority_snapshot_hash,
            max_messages=max_messages,
            current_actor_id=current_actor_id,
            current_event_id=current_event_id,
        )


async def persist_slack_thread_coverage(
    session: AsyncSession,
    *,
    conversation_id: str,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    authoritative_reply_count: int,
    authoritative_latest_reply_ts: str | None,
    authority_source: SlackThreadCoverageSource,
    authority_snapshot_hash: str,
    observed_at: datetime,
) -> None:
    """Upsert the latest exact history snapshot without using event volume."""

    values = {
        "id": _coverage_id(team_id, channel_id, thread_root_ts),
        "conversation_id": conversation_id,
        "team_id": team_id,
        "channel_id": channel_id,
        "thread_root_ts": thread_root_ts,
        "authoritative_reply_count": authoritative_reply_count,
        "authoritative_latest_reply_ts": authoritative_latest_reply_ts,
        "authority_source": authority_source.value,
        "authority_snapshot_hash": authority_snapshot_hash,
        "metadata_observed_at": observed_at,
    }
    excluded = postgres_insert(SlackThreadCoverageRow).excluded
    await session.execute(
        postgres_insert(SlackThreadCoverageRow)
        .values(**values)
        .on_conflict_do_update(
            constraint="uq_slack_thread_coverage_root",
            set_={
                "authoritative_reply_count": excluded.authoritative_reply_count,
                "authoritative_latest_reply_ts": excluded.authoritative_latest_reply_ts,
                "authority_source": excluded.authority_source,
                "authority_snapshot_hash": excluded.authority_snapshot_hash,
                "metadata_observed_at": excluded.metadata_observed_at,
                "updated_at": observed_at,
            },
            where=(SlackThreadCoverageRow.metadata_observed_at <= excluded.metadata_observed_at),
        )
    )


async def _ensure_passive_conversation(
    session: AsyncSession,
    message: SlackPassiveMessage,
) -> ConversationRow:
    kind = "group_dm" if message.conversation_kind is SlackConversationKind.MPIM else "channel"
    external_provenance = (
        "not_applicable" if message.conversation_kind is SlackConversationKind.MPIM else "unknown"
    )
    await session.execute(
        postgres_insert(ConversationRow)
        .values(
            id=canonical_slack_conversation_id(message.team_id, message.channel_id),
            provider="slack",
            team_id=message.team_id,
            external_id=message.channel_id,
            kind=kind,
            actor_id=None,
            authority_source="slack_event",
            bot_presence="present",
            lifecycle="active",
            external_provenance=external_provenance,
            membership_policy_version=1,
            version=1,
        )
        .on_conflict_do_nothing(constraint="uq_conversations_provider_external")
    )
    conversation = await session.scalar(
        select(ConversationRow).where(
            ConversationRow.provider == "slack",
            ConversationRow.team_id == message.team_id,
            ConversationRow.external_id == message.channel_id,
        )
    )
    if conversation is None:
        raise RuntimeError("passive Slack conversation was not persisted")
    if message.conversation_kind is SlackConversationKind.MPIM:
        if conversation.kind != "group_dm":
            raise RuntimeError("passive MPIM conflicted with persisted conversation kind")
    elif conversation.kind not in {"channel", "shared", "external"}:
        raise RuntimeError("passive channel message conflicted with persisted conversation kind")
    return conversation


async def _reconcile_assistant_delivery(
    session: AsyncSession,
    *,
    plane_message: ConversationPlaneMessage,
) -> bool:
    if plane_message.harness_thread_id is None:
        return False
    existing_id = await session.scalar(
        select(SanitizedMessageRow.id)
        .where(
            SanitizedMessageRow.conversation_id == plane_message.conversation_id,
            SanitizedMessageRow.harness_thread_id == plane_message.harness_thread_id,
            SanitizedMessageRow.role == ConversationMessageRole.ASSISTANT.value,
            SanitizedMessageRow.content_hash == plane_message.content_hash,
            or_(
                SanitizedMessageRow.provider_thread_root_ts.is_(None),
                SanitizedMessageRow.provider_thread_root_ts
                == plane_message.provider_thread_root_ts,
            ),
        )
        .order_by(SanitizedMessageRow.recorded_at.desc(), SanitizedMessageRow.id.desc())
        .limit(1)
    )
    if existing_id is None:
        return False
    await session.execute(
        update(SanitizedMessageRow)
        .where(SanitizedMessageRow.id == existing_id)
        .values(
            provider_message_ts=plane_message.provider_message_ts,
            provider_thread_root_ts=plane_message.provider_thread_root_ts,
        )
    )
    return True


def _assess_thread_coverage(
    *,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    current_message_ts: str,
    conversation_id: str,
    rows: tuple[SanitizedMessageRow, ...],
    authoritative_reply_count: int,
    authoritative_latest_reply_ts: str | None,
    coverage_source: SlackThreadCoverageSource,
    coverage_snapshot_hash: str,
    max_messages: int,
    current_actor_id: str,
    current_event_id: str,
) -> PersistedSlackThreadSnapshot:
    persisted_message_count: int | None = None
    boundary_attested = False
    boundary_actor_id: str | None = None
    boundary_event_id: str | None = None
    if len(rows) > max_messages:
        reason = SlackThreadCoverageReason.BOUNDED_LIMIT
        selected_rows: tuple[SanitizedMessageRow, ...] = ()
    else:
        try:
            root_key = _slack_ts_key(thread_root_ts)
            current_key = _slack_ts_key(current_message_ts)
            latest_key = (
                _slack_ts_key(authoritative_latest_reply_ts)
                if authoritative_latest_reply_ts is not None
                else None
            )
            timestamped = tuple(
                sorted(
                    ((row, _slack_ts_key(row.provider_message_ts or "")) for row in rows),
                    key=lambda item: (item[1], item[0].id),
                )
            )
        except ValueError:
            reason = SlackThreadCoverageReason.INVALID_TIMESTAMP
            selected_rows = ()
        else:
            persisted_message_count = len(timestamped)
            all_keys = tuple(key for _row, key in timestamped)
            prior_rows = tuple(row for row, key in timestamped if key < current_key)
            boundary_rows = tuple(row for row, key in timestamped if key == current_key)
            if len(boundary_rows) == 1:
                boundary_actor_id = boundary_rows[0].actor_id
                boundary_event_id = boundary_rows[0].external_event_id
                boundary_attested = bool(
                    boundary_rows[0].role == "user"
                    and boundary_actor_id == current_actor_id
                    and boundary_event_id == current_event_id
                )
            if not any(key == root_key for key in all_keys):
                reason = SlackThreadCoverageReason.ROOT_MISSING
            elif len(set(all_keys)) != len(all_keys):
                reason = SlackThreadCoverageReason.DUPLICATE_PROVIDER_TIMESTAMP
            elif root_key > current_key or not boundary_attested:
                reason = SlackThreadCoverageReason.BOUNDARY_NOT_ATTESTED
            elif latest_key is not None and latest_key < current_key:
                reason = SlackThreadCoverageReason.BOUNDARY_NOT_ATTESTED
            elif latest_key is not None and latest_key > current_key and latest_key not in all_keys:
                reason = SlackThreadCoverageReason.METADATA_AFTER_BOUNDARY
            elif any(key > current_key for key in all_keys) and (
                latest_key is None or max(all_keys) != latest_key
            ):
                reason = SlackThreadCoverageReason.METADATA_AFTER_BOUNDARY
            elif len(timestamped) != authoritative_reply_count + 1:
                reason = SlackThreadCoverageReason.COUNT_MISMATCH
            elif authoritative_reply_count == 0:
                reason = (
                    SlackThreadCoverageReason.COMPLETE
                    if latest_key is None and all_keys == (root_key,)
                    else SlackThreadCoverageReason.LATEST_MISMATCH
                )
            elif latest_key is None or max(all_keys) != latest_key:
                reason = SlackThreadCoverageReason.LATEST_MISMATCH
            else:
                reason = SlackThreadCoverageReason.COMPLETE
            selected_rows = prior_rows if reason is SlackThreadCoverageReason.COMPLETE else ()
    messages = tuple(
        PersistedSlackThreadMessage(
            id=row.id,
            actor_id=row.actor_id or "unknown",
            role=row.role,
            text=row.text,
            message_ts=row.provider_message_ts or "",
        )
        for row in selected_rows
    )
    return _snapshot(
        team_id=team_id,
        channel_id=channel_id,
        thread_root_ts=thread_root_ts,
        current_message_ts=current_message_ts,
        conversation_id=conversation_id,
        messages=messages,
        reason=reason,
        authoritative_reply_count=authoritative_reply_count,
        authoritative_latest_reply_ts=authoritative_latest_reply_ts,
        coverage_source=coverage_source,
        coverage_snapshot_hash=coverage_snapshot_hash,
        persisted_message_count=persisted_message_count,
        boundary_attested=boundary_attested,
        boundary_actor_id=boundary_actor_id,
        boundary_event_id=boundary_event_id,
    )


def _snapshot(
    *,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    current_message_ts: str,
    conversation_id: str | None,
    messages: tuple[PersistedSlackThreadMessage, ...],
    reason: SlackThreadCoverageReason,
    authoritative_reply_count: int | None,
    authoritative_latest_reply_ts: str | None,
    coverage_source: SlackThreadCoverageSource | None,
    coverage_snapshot_hash: str | None,
    persisted_message_count: int | None = None,
    boundary_attested: bool = False,
    boundary_actor_id: str | None = None,
    boundary_event_id: str | None = None,
) -> PersistedSlackThreadSnapshot:
    complete = reason is SlackThreadCoverageReason.COMPLETE
    complete_through_ts = None
    if complete:
        complete_through_ts = current_message_ts
    digest_material = {
        "team_id": team_id,
        "channel_id": channel_id,
        "thread_root_ts": thread_root_ts,
        "current_message_ts": current_message_ts,
        "conversation_id": conversation_id,
        "coverage_reason": reason.value,
        "authoritative_reply_count": authoritative_reply_count,
        "authoritative_latest_reply_ts": authoritative_latest_reply_ts,
        "coverage_source": coverage_source.value if coverage_source is not None else None,
        "coverage_snapshot_hash": coverage_snapshot_hash,
        "persisted_message_count": persisted_message_count,
        "boundary_attested": boundary_attested,
        "boundary_actor_id": boundary_actor_id,
        "boundary_event_id": boundary_event_id,
        "messages": [
            [
                item.id,
                item.actor_id,
                item.role,
                item.message_ts,
                hashlib.sha256(item.text.encode()).hexdigest(),
            ]
            for item in messages
        ],
    }
    digest = hashlib.sha256(
        json.dumps(digest_material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PersistedSlackThreadSnapshot(
        team_id=team_id,
        channel_id=channel_id,
        thread_root_ts=thread_root_ts,
        current_message_ts=current_message_ts,
        conversation_id=conversation_id,
        messages=messages,
        complete=complete,
        coverage_reason=reason,
        authoritative_reply_count=authoritative_reply_count,
        authoritative_latest_reply_ts=authoritative_latest_reply_ts,
        coverage_source=coverage_source,
        coverage_snapshot_hash=coverage_snapshot_hash,
        complete_through_ts=complete_through_ts,
        coverage_digest=digest,
        persisted_message_count=persisted_message_count,
        boundary_attested=boundary_attested,
        boundary_actor_id=boundary_actor_id,
        boundary_event_id=boundary_event_id,
    )


def _empty_snapshot(
    *,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    current_message_ts: str,
    reason: SlackThreadCoverageReason,
    conversation_id: str | None = None,
) -> PersistedSlackThreadSnapshot:
    return _snapshot(
        team_id=team_id,
        channel_id=channel_id,
        thread_root_ts=thread_root_ts,
        current_message_ts=current_message_ts,
        conversation_id=conversation_id,
        messages=(),
        reason=reason,
        authoritative_reply_count=None,
        authoritative_latest_reply_ts=None,
        coverage_source=None,
        coverage_snapshot_hash=None,
    )


def _validate_thread_lookup(
    *,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    current_message_ts: str,
    max_messages: int,
) -> None:
    if any(not value.strip() for value in (team_id, channel_id)):
        raise ValueError("Slack thread lookup IDs must be non-empty")
    _slack_ts_key(thread_root_ts)
    _slack_ts_key(current_message_ts)
    if max_messages < 1 or max_messages > 500:
        raise ValueError("max_messages must be between 1 and 500")


def _validate_authoritative_root_snapshot(
    *,
    team_id: str,
    channel_id: str,
    thread_root_ts: str,
    current_message_ts: str,
    raw_root: Mapping[str, object],
    source: SlackThreadCoverageSource,
) -> tuple[int, str | None, str] | None:
    """Validate only the bounded root tuple returned by conversations.history."""

    _validate_thread_lookup(
        team_id=team_id,
        channel_id=channel_id,
        thread_root_ts=thread_root_ts,
        current_message_ts=current_message_ts,
        max_messages=500,
    )
    if _slack_ts_key(thread_root_ts) > _slack_ts_key(current_message_ts):
        return None
    root_ts = raw_root.get("ts")
    if not isinstance(root_ts, str) or root_ts != thread_root_ts:
        return None
    raw_thread_ts = raw_root.get("thread_ts")
    if raw_thread_ts not in {None, "", thread_root_ts}:
        return None
    reply_count = raw_root.get("reply_count")
    if isinstance(reply_count, bool) or not isinstance(reply_count, int) or reply_count < 0:
        return None
    raw_latest = raw_root.get("latest_reply")
    if raw_latest is not None and not isinstance(raw_latest, str):
        return None
    latest_reply_ts = raw_latest or None
    if reply_count == 0:
        if latest_reply_ts is not None:
            return None
    else:
        if latest_reply_ts is None:
            return None
        try:
            latest_key = _slack_ts_key(latest_reply_ts)
        except ValueError:
            return None
        if latest_key <= _slack_ts_key(thread_root_ts):
            return None
    material = {
        "team_id": team_id,
        "channel_id": channel_id,
        "thread_root_ts": thread_root_ts,
        "reply_count": reply_count,
        "latest_reply_ts": latest_reply_ts,
        "source": source.value,
    }
    snapshot_hash = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return reply_count, latest_reply_ts, snapshot_hash


def _slack_ts_key(value: str) -> Decimal:
    if not _SLACK_TS.fullmatch(value):
        raise ValueError("Slack message timestamp has an invalid shape")
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Slack message timestamp is invalid") from exc


def _slack_thread_key(team_id: str, channel_id: str, thread_root_ts: str) -> str:
    return f"slack:{team_id}:{channel_id}:{thread_root_ts}"


def _coverage_id(team_id: str, channel_id: str, thread_root_ts: str) -> str:
    material = f"{team_id}\x1f{channel_id}\x1f{thread_root_ts}"
    return f"coverage-{hashlib.sha256(material.encode()).hexdigest()[:55]}"
