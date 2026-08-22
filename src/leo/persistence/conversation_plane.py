"""Durable sanitized message-plane persistence for admitted conversations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.memory.planes import sanitize_message_text
from leo.persistence.schema import SanitizedMessageRow


class ConversationMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ConversationPlaneMessage:
    """One immutable message whose authority fields were bound before model execution."""

    id: str
    scope: ScopeKey
    conversation_id: str
    harness_thread_id: str | None
    destination_id: str
    external_event_id: str
    actor_id: str
    role: ConversationMessageRole
    provider_message_ts: str | None
    provider_thread_root_ts: str | None
    context_access_hash: str | None
    text: str
    content_hash: str
    recorded_at: datetime


def canonical_slack_conversation_id(team_id: str, destination_id: str) -> str:
    """Return the stable canonical ID used by Slack admission and migration backfills."""

    if not team_id.strip() or not destination_id.strip():
        raise ValueError("Slack conversation identity fields must be non-empty")
    digest = hashlib.sha256(f"{team_id}\x1f{destination_id}".encode()).hexdigest()
    return f"slack-{digest[:56]}"


def build_conversation_plane_message(
    *,
    scope: ScopeKey,
    conversation_id: str,
    harness_thread_id: str | None,
    destination_id: str,
    external_event_id: str,
    actor_id: str,
    role: ConversationMessageRole,
    provider_message_ts: str | None,
    context_access_hash: str | None,
    text: str,
    recorded_at: datetime,
    provider_thread_root_ts: str | None = None,
) -> ConversationPlaneMessage:
    """Sanitize and deterministically identify a message for idempotent replay."""

    required = {
        "conversation_id": conversation_id,
        "destination_id": destination_id,
        "external_event_id": external_event_id,
        "actor_id": actor_id,
    }
    if any(not value.strip() for value in required.values()):
        raise ValueError("conversation message authority fields must be non-empty")
    if context_access_hash is not None and (
        len(context_access_hash) != 64
        or any(character not in "0123456789abcdef" for character in context_access_hash)
    ):
        raise ValueError("context_access_hash must be a lowercase SHA-256 digest")
    # Slack can carry more text than the bounded message plane. Truncation happens before
    # sanitization so the persisted payload can never exceed its database contract.
    sanitized = sanitize_message_text(text[:8192])
    content_hash = hashlib.sha256(sanitized.encode("utf-8")).hexdigest()
    identity = "\x1f".join((conversation_id, external_event_id, role.value))
    message_id = f"message-{hashlib.sha256(identity.encode()).hexdigest()[:56]}"
    return ConversationPlaneMessage(
        id=message_id,
        scope=scope,
        conversation_id=conversation_id,
        harness_thread_id=harness_thread_id,
        destination_id=destination_id,
        external_event_id=external_event_id,
        actor_id=actor_id,
        role=role,
        provider_message_ts=provider_message_ts,
        provider_thread_root_ts=provider_thread_root_ts,
        context_access_hash=context_access_hash,
        text=sanitized,
        content_hash=content_hash,
        recorded_at=recorded_at,
    )


async def persist_conversation_plane_message(
    session: AsyncSession,
    message: ConversationPlaneMessage,
) -> None:
    """Insert exactly once inside an existing transaction."""

    await session.execute(
        postgres_insert(SanitizedMessageRow)
        .values(
            id=message.id,
            organization_id=message.scope.organization_id,
            strategy_id=message.scope.strategy_id,
            destination_id=message.destination_id,
            external_event_id=message.external_event_id,
            text=message.text,
            content_hash=message.content_hash,
            recorded_at=message.recorded_at,
            conversation_id=message.conversation_id,
            harness_thread_id=message.harness_thread_id,
            actor_id=message.actor_id,
            role=message.role.value,
            provider_message_ts=message.provider_message_ts,
            provider_thread_root_ts=message.provider_thread_root_ts,
            context_access_hash=message.context_access_hash,
        )
        .on_conflict_do_nothing(
            index_elements=[
                SanitizedMessageRow.conversation_id,
                SanitizedMessageRow.external_event_id,
                SanitizedMessageRow.role,
            ],
            index_where=SanitizedMessageRow.conversation_id.is_not(None),
        )
    )


class PostgresConversationPlaneStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def record(self, message: ConversationPlaneMessage) -> None:
        async with self._sessions() as session, session.begin():
            await persist_conversation_plane_message(session, message)
