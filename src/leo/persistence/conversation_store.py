"""SQLAlchemy adapter for server-derived conversation and pinned-thread scope state."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.domain.conversation import ConversationKind, ConversationRef, ThreadRef
from leo.domain.conversation_store import ConversationStoreError
from leo.harness.models import ScopeKey
from leo.harness.ports import IdGenerator
from leo.persistence.schema import ConversationRow, ConversationThreadRow


class PostgresConversationStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], ids: IdGenerator) -> None:
        self._sessions = sessions
        self._ids = ids

    async def pin_thread(
        self,
        scope: ScopeKey,
        destination: ConversationRef,
        *,
        root_ts: str,
        mapping_version: int | None = None,
    ) -> ThreadRef:
        if destination.kind is ConversationKind.UNKNOWN:
            raise ConversationStoreError("conversation_ineligible")
        if not root_ts or (mapping_version is not None and mapping_version < 1):
            raise ValueError("root_ts is required and mapping_version must be positive")
        async with self._sessions() as session, session.begin():
            await session.execute(
                postgres_insert(ConversationRow)
                .values(
                    id=self._ids.new("conversation"),
                    provider=destination.provider,
                    team_id=destination.team_id,
                    external_id=destination.external_id,
                    kind=destination.kind.value,
                    actor_id=destination.actor_id,
                    version=1,
                )
                .on_conflict_do_nothing(
                    index_elements=[
                        ConversationRow.provider,
                        ConversationRow.team_id,
                        ConversationRow.external_id,
                    ]
                )
            )
            conversation = await session.scalar(
                select(ConversationRow)
                .where(
                    ConversationRow.provider == destination.provider,
                    ConversationRow.team_id == destination.team_id,
                    ConversationRow.external_id == destination.external_id,
                )
                .with_for_update()
            )
            if conversation is None or conversation.kind != destination.kind.value:
                raise ConversationStoreError("conversation_identity_conflict")
            if conversation.actor_id != destination.actor_id:
                raise ConversationStoreError("conversation_actor_conflict")
            existing = await session.scalar(
                select(ConversationThreadRow)
                .where(
                    ConversationThreadRow.conversation_id == conversation.id,
                    ConversationThreadRow.root_ts == root_ts,
                    ConversationThreadRow.organization_id == scope.organization_id,
                )
                .with_for_update()
            )
            if existing is not None:
                return _thread_model(existing, destination)
            conflicting = await session.scalar(
                select(ConversationThreadRow).where(
                    ConversationThreadRow.conversation_id == conversation.id,
                    ConversationThreadRow.root_ts == root_ts,
                )
            )
            if conflicting is not None:
                raise ConversationStoreError("thread_organization_changed")
            row = ConversationThreadRow(
                id=self._ids.new("conversation-thread"),
                conversation_id=conversation.id,
                root_ts=root_ts,
                organization_id=scope.organization_id,
                strategy_id=scope.strategy_id,
                mapping_version=mapping_version or 1,
                version=1,
            )
            session.add(row)
            await session.flush()
            return _thread_model(row, destination)

    async def load_thread(
        self,
        scope: ScopeKey,
        destination: ConversationRef,
        *,
        root_ts: str,
    ) -> ThreadRef | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(ConversationThreadRow)
                .join(ConversationRow, ConversationRow.id == ConversationThreadRow.conversation_id)
                .where(
                    ConversationRow.provider == destination.provider,
                    ConversationRow.team_id == destination.team_id,
                    ConversationRow.external_id == destination.external_id,
                    ConversationRow.kind == destination.kind.value,
                    ConversationRow.actor_id == destination.actor_id,
                    ConversationThreadRow.root_ts == root_ts,
                    ConversationThreadRow.organization_id == scope.organization_id,
                )
            )
        return None if row is None else _thread_model(row, destination)


def _thread_model(row: ConversationThreadRow, destination: ConversationRef) -> ThreadRef:
    return ThreadRef(
        conversation=destination,
        root_ts=row.root_ts,
        scope=ScopeKey(organization_id=row.organization_id, strategy_id=row.strategy_id),
        mapping_version=row.mapping_version,
        version=row.version,
    )
