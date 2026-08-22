"""Load exact conversation-scoped history and memory for a live model turn."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Literal

from pydantic import Field, model_validator
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.context_budget import ContextBudgetError
from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContractModel,
    NonEmptyStr,
    ScopeKey,
)
from leo.harness.thread_context import (
    ThreadContextRange,
    ThreadTurnRetentionInput,
    classify_thread_transcript,
    select_context_with_thread_compaction,
)
from leo.memory.navigation import membership_snapshot_hash
from leo.memory.retrieval import channel_authorized_namespaces, dm_authorized_namespaces
from leo.persistence.schema import (
    ConversationAccessSnapshotRow,
    ConversationActorMembershipRow,
    ConversationRow,
    MemoryRecordRow,
    MemoryRevisionRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
    ThreadRow,
    ThreadSummaryRevisionRow,
)


class ConversationContextRequest(ContractModel):
    """An exact, server-derived visibility projection for one turn."""

    team_id: NonEmptyStr
    destination_id: NonEmptyStr
    destination_kind: Literal["channel", "dm", "group_dm", "shared", "external"]
    actor_id: NonEmptyStr
    objective: NonEmptyStr
    current_task_id: NonEmptyStr
    current_event_id: NonEmptyStr
    current_message_ts: NonEmptyStr
    thread_root_ts: NonEmptyStr
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    current_thread_namespace_id: NonEmptyStr
    max_turns: int = Field(default=24, ge=0, le=100)
    max_memories: int = Field(default=16, ge=0, le=100)
    max_context_tokens: int = Field(default=4_000, ge=128, le=32_000)
    max_thread_messages: int = Field(default=200, ge=1, le=1_000)

    @model_validator(mode="after")
    def exact_access_projection(self) -> ConversationContextRequest:
        normalized = tuple(sorted(set(self.allowed_conversation_ids)))
        if normalized != self.allowed_conversation_ids:
            raise ValueError("allowed conversation IDs must be sorted and unique")
        if self.destination_id not in normalized:
            raise ValueError("destination must be present in its context projection")
        if self.destination_kind != "dm" and normalized != (self.destination_id,):
            raise ValueError("non-DM context must be restricted to the exact destination")
        current_ts = _parse_slack_timestamp(self.current_message_ts)
        root_ts = _parse_slack_timestamp(self.thread_root_ts)
        if current_ts is None or root_ts is None or root_ts > current_ts:
            raise ValueError("Slack thread timestamps must define a valid current boundary")
        return self


class ConversationContextAuthorizationError(RuntimeError):
    """The durable admission snapshot did not authorize the requested context."""


class ConversationContextOverflowError(RuntimeError):
    """Protected thread context cannot fit or complete inside the configured bound."""


class ConversationContextManifest(ContractModel):
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    harness_thread_id: NonEmptyStr
    external_provenance: NonEmptyStr
    membership_policy_version: int = Field(ge=1)
    item_ids: tuple[NonEmptyStr, ...]
    current_event_id: NonEmptyStr
    thread_root_ts: NonEmptyStr
    protected_thread_item_ids: tuple[NonEmptyStr, ...] = ()
    compacted_thread_item_ids: tuple[NonEmptyStr, ...] = ()
    thread_compaction_digest: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    thread_reopen_handles: tuple[NonEmptyStr, ...] = ()


class AuthorizedConversationContext(ContractModel):
    items: tuple[ContextItem, ...]
    manifest: ConversationContextManifest
    reopen_ranges: tuple[ThreadContextRange, ...] = Field(default=(), exclude=True)


class PostgresConversationContextLoader:
    """Select authorization first, then relevance/ranking and token budgeting."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(
        self,
        scope: ScopeKey,
        request: ConversationContextRequest,
    ) -> tuple[ContextItem, ...]:
        return (await self.load_authorized(scope, request)).items

    async def load_authorized(
        self,
        scope: ScopeKey,
        request: ConversationContextRequest,
    ) -> AuthorizedConversationContext:
        async with self._sessions() as session:
            (
                membership_hash,
                harness_thread_id,
                external_provenance,
                membership_policy_version,
                destination_conversation_id,
            ) = await self._authorize(session, scope, request)
            turns = await self._load_turns(
                session,
                scope,
                request,
                harness_thread_id=harness_thread_id,
            )
            memories = await self._load_memories(session, scope, request)
            summary = await self._load_summary(
                session,
                scope,
                request,
                harness_thread_id=harness_thread_id,
            )
            recent = await self._load_recent_thread_messages(
                session,
                scope,
                request,
                harness_thread_id=harness_thread_id,
                destination_conversation_id=destination_conversation_id,
            )
        items = (*turns, *summary, *recent, *memories)
        if not items:
            return AuthorizedConversationContext(
                items=(),
                manifest=ConversationContextManifest(
                    access_hash=request.access_hash,
                    membership_hash=membership_hash,
                    allowed_conversation_ids=request.allowed_conversation_ids,
                    harness_thread_id=harness_thread_id,
                    external_provenance=external_provenance,
                    membership_policy_version=membership_policy_version,
                    item_ids=(),
                    current_event_id=request.current_event_id,
                    thread_root_ts=request.thread_root_ts,
                ),
            )
        prioritized = tuple(
            item
            if item.budget_priority is not None
            else item.model_copy(update={"budget_priority": _priority(item, request.objective)})
            for item in items
        )
        thread_item_ids = frozenset(
            item.id for item in recent if item.kind is ContextItemKind.CONVERSATION_TURN
        )
        try:
            selection = select_context_with_thread_compaction(
                prioritized,
                thread_item_ids=thread_item_ids,
                conversation_id=request.destination_id,
                summary_id_namespace=(
                    f"postgres-thread-compaction:{harness_thread_id}:{request.current_event_id}"
                ),
                max_tokens=request.max_context_tokens,
            )
        except ContextBudgetError as exc:
            raise ConversationContextOverflowError(exc.safe_code) from exc
        selected_items = selection.items
        return AuthorizedConversationContext(
            items=selected_items,
            manifest=ConversationContextManifest(
                access_hash=request.access_hash,
                membership_hash=membership_hash,
                allowed_conversation_ids=request.allowed_conversation_ids,
                harness_thread_id=harness_thread_id,
                external_provenance=external_provenance,
                membership_policy_version=membership_policy_version,
                item_ids=tuple(item.id for item in selected_items),
                current_event_id=request.current_event_id,
                thread_root_ts=request.thread_root_ts,
                protected_thread_item_ids=tuple(
                    item.id for item in recent if item.retention.pinned
                ),
                compacted_thread_item_ids=selection.compacted_item_ids,
                thread_compaction_digest=selection.compaction_digest,
                thread_reopen_handles=tuple(item.handle for item in selection.reopen_ranges),
            ),
            reopen_ranges=selection.reopen_ranges,
        )

    async def _authorize(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        request: ConversationContextRequest,
    ) -> tuple[str, str, str, int, str]:
        """Validate one immutable ingress projection before any content query."""

        rows = (
            await session.execute(
                select(
                    SlackIngressEventRow,
                    TaskRow,
                    ConversationAccessSnapshotRow,
                    ThreadRow,
                )
                .join(TaskRow, TaskRow.id == SlackIngressEventRow.task_id)
                .join(ThreadRow, ThreadRow.id == TaskRow.thread_id)
                .join(
                    ConversationAccessSnapshotRow,
                    ConversationAccessSnapshotRow.ingress_event_id == SlackIngressEventRow.event_id,
                )
                .where(SlackIngressEventRow.task_id == request.current_task_id)
                .order_by(
                    ConversationAccessSnapshotRow.position,
                    ConversationAccessSnapshotRow.id,
                )
            )
        ).all()
        if not rows:
            raise ConversationContextAuthorizationError(
                "context access snapshot did not authorize the request"
            )

        ingress, task, _first_snapshot, thread = rows[0]
        expected_kind = _slack_conversation_kind(request.destination_kind)
        ingress_matches = (
            task.id == request.current_task_id
            and ingress.task_id == task.id
            and ingress.event_id == request.current_event_id
            and ingress.message_ts == request.current_message_ts
            and task.organization_id == scope.organization_id
            and task.strategy_id == scope.strategy_id
            and ingress.organization_id == scope.organization_id
            and ingress.strategy_id == scope.strategy_id
            and ingress.team_id == request.team_id
            and ingress.user_id == request.actor_id
            and ingress.channel_id == request.destination_id
            and ingress.thread_root_ts == request.thread_root_ts
            and ingress.conversation_key == request.current_thread_namespace_id
            and ingress.conversation_kind == expected_kind
            and ingress.context_access_hash == request.access_hash
            and tuple(ingress.context_conversation_ids) == request.allowed_conversation_ids
            and ingress.bot_presence == "present"
            and ingress.conversation_lifecycle == "active"
        )
        thread_matches = (
            thread.id == task.thread_id
            and thread.organization_id == scope.organization_id
            and thread.strategy_id == scope.strategy_id
            and thread.origin_provider == "slack"
            and thread.external_thread_id == request.current_thread_namespace_id
            and thread.external_channel_id == request.destination_id
            and thread.conversation_id == ingress.conversation_id
        )
        snapshots = tuple(row[2] for row in rows)
        snapshot_matches = (
            tuple(snapshot.position for snapshot in snapshots) == tuple(range(len(snapshots)))
            and tuple(snapshot.conversation_external_id for snapshot in snapshots)
            == request.allowed_conversation_ids
            and all(
                snapshot.ingress_event_id == ingress.event_id
                and snapshot.organization_id == scope.organization_id
                and snapshot.team_id == request.team_id
                and snapshot.actor_id == request.actor_id
                and snapshot.destination_external_id == request.destination_id
                and snapshot.context_access_hash == request.access_hash
                and snapshot.source_kind == ingress.context_projection_source
                for snapshot in snapshots
            )
        )
        if not ingress_matches or not thread_matches or not snapshot_matches:
            raise ConversationContextAuthorizationError(
                "context access snapshot did not authorize the request"
            )
        current_conversations = tuple(
            await session.scalars(
                select(ConversationRow)
                .where(
                    ConversationRow.provider == "slack",
                    ConversationRow.team_id == request.team_id,
                    ConversationRow.external_id.in_(request.allowed_conversation_ids),
                )
                .with_for_update(read=True)
            )
        )
        current_by_external_id = {
            conversation.external_id: conversation for conversation in current_conversations
        }
        destination = current_by_external_id.get(request.destination_id)
        if (
            destination is None
            or destination.id != ingress.conversation_id
            or destination.kind != request.destination_kind
            or destination.bot_presence != "present"
            or destination.lifecycle != "active"
            or destination.external_provenance != ingress.external_provenance
            or destination.membership_policy_version != ingress.membership_policy_version
            or set(current_by_external_id) != set(request.allowed_conversation_ids)
            or any(
                conversation.bot_presence != "present" or conversation.lifecycle != "active"
                for conversation in current_conversations
            )
        ):
            raise ConversationContextAuthorizationError(
                "current source conversation or Leo presence no longer authorizes context"
            )
        membership_statement = (
            select(ConversationActorMembershipRow)
            .where(
                ConversationActorMembershipRow.organization_id == scope.organization_id,
                ConversationActorMembershipRow.team_id == request.team_id,
                ConversationActorMembershipRow.actor_id == request.actor_id,
                ConversationActorMembershipRow.conversation_external_id.in_(
                    request.allowed_conversation_ids
                ),
            )
            .with_for_update(read=True)
        )
        memberships = tuple(await session.scalars(membership_statement))
        membership_by_id = {row.conversation_external_id: row for row in memberships}
        if set(membership_by_id) != set(request.allowed_conversation_ids) or any(
            row.status != "active" for row in memberships
        ):
            raise ConversationContextAuthorizationError(
                "current conversation membership or Leo presence was revoked"
            )
        return (
            membership_snapshot_hash(request.allowed_conversation_ids),
            task.thread_id,
            destination.external_provenance,
            destination.membership_policy_version,
            destination.id,
        )

    async def _load_turns(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        request: ConversationContextRequest,
        *,
        harness_thread_id: str,
    ) -> tuple[ContextItem, ...]:
        """Load task outcomes only from the current authorized Slack thread.

        ``allowed_conversation_ids`` is an authorization projection for memory
        namespaces.  In particular, a 1:1 DM may contain the actor's membership
        union.  It must never widen the conversational task plane: prior tasks
        become antecedents only when they share the server-derived harness thread.
        Cross-thread recall is handled explicitly by the memory/navigation tools.
        """

        if request.max_turns == 0:
            return ()
        rows = (
            await session.execute(
                select(TaskRow, ConversationRow)
                .join(ThreadRow, ThreadRow.id == TaskRow.thread_id)
                .join(ConversationRow, ConversationRow.id == ThreadRow.conversation_id)
                .where(
                    TaskRow.organization_id == scope.organization_id,
                    TaskRow.thread_id == harness_thread_id,
                    TaskRow.id != request.current_task_id,
                    ThreadRow.id == harness_thread_id,
                    ThreadRow.organization_id == scope.organization_id,
                    ThreadRow.strategy_id == scope.strategy_id,
                    ThreadRow.origin_provider == "slack",
                    ThreadRow.external_thread_id == request.current_thread_namespace_id,
                    ThreadRow.external_channel_id == request.destination_id,
                    ConversationRow.provider == "slack",
                    ConversationRow.team_id == request.team_id,
                    ConversationRow.external_id == request.destination_id,
                )
                .order_by(TaskRow.created_at.desc(), TaskRow.id.desc())
                .limit(request.max_turns)
            )
        ).all()
        if any(
            task.thread_id != harness_thread_id
            or task.organization_id != scope.organization_id
            or task.id == request.current_task_id
            or conversation.external_id != request.destination_id
            for task, conversation in rows
        ):
            raise ConversationContextAuthorizationError(
                "task history query returned content outside the exact thread boundary"
            )
        chronological = reversed(rows)
        return tuple(
            ContextItem(
                id=f"turn:{task.id}",
                kind=ContextItemKind.CONVERSATION_TURN,
                content=_turn_content(task),
                conversation_id=conversation.external_id,
                source_scope=ScopeKey(
                    organization_id=task.organization_id,
                    strategy_id=task.strategy_id,
                ),
            )
            for task, conversation in chronological
        )

    async def _load_memories(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        request: ConversationContextRequest,
    ) -> tuple[ContextItem, ...]:
        if request.max_memories == 0:
            return ()
        if request.destination_kind == "dm":
            authorized = dm_authorized_namespaces(
                request.allowed_conversation_ids,
                actor_id=request.actor_id,
                thread_namespace_id=request.current_thread_namespace_id,
            )
        else:
            authorized = channel_authorized_namespaces(
                request.destination_id,
                thread_namespace_id=request.current_thread_namespace_id,
            )
        predicates = [
            and_(
                MemoryRecordRow.visibility == item.visibility.value,
                MemoryRecordRow.namespace_id == item.namespace_id,
                MemoryRevisionRow.visibility == item.visibility.value,
                MemoryRevisionRow.namespace_id == item.namespace_id,
            )
            for item in authorized
        ]
        now = datetime.now().astimezone()
        rows = (
            await session.execute(
                select(MemoryRevisionRow, MemoryRecordRow)
                .join(MemoryRecordRow, MemoryRecordRow.id == MemoryRevisionRow.record_id)
                .where(
                    MemoryRecordRow.organization_id == scope.organization_id,
                    MemoryRecordRow.status == "active",
                    MemoryRevisionRow.status == "active",
                    MemoryRevisionRow.number == MemoryRecordRow.current_revision,
                    MemoryRevisionRow.valid_from <= now,
                    or_(
                        MemoryRevisionRow.valid_until.is_(None),
                        MemoryRevisionRow.valid_until > now,
                    ),
                    or_(MemoryRevisionRow.expires_at.is_(None), MemoryRevisionRow.expires_at > now),
                    or_(*predicates),
                )
                .order_by(MemoryRevisionRow.recorded_at.desc(), MemoryRevisionRow.id.desc())
                .limit(max(request.max_memories * 4, request.max_memories))
            )
        ).all()
        ranked = sorted(
            rows,
            key=lambda row: (
                -_relevance(row[0].content, request.objective),
                -row[0].recorded_at.timestamp(),
                row[0].id,
            ),
        )[: request.max_memories]
        return tuple(
            ContextItem(
                id=f"memory:{revision.id}",
                kind=ContextItemKind.MEMORY,
                content=revision.content,
                conversation_id=(
                    request.destination_id
                    if record.visibility == "actor_private"
                    else record.namespace_id
                ),
                source_scope=ScopeKey(
                    organization_id=record.organization_id,
                    strategy_id=record.strategy_id,
                ),
                source_actor_id=revision.actor_id,
            )
            for revision, record in ranked
        )

    async def _load_summary(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        request: ConversationContextRequest,
        *,
        harness_thread_id: str,
    ) -> tuple[ContextItem, ...]:
        row = await session.scalar(
            select(ThreadSummaryRevisionRow)
            .where(
                ThreadSummaryRevisionRow.organization_id == scope.organization_id,
                ThreadSummaryRevisionRow.thread_id == harness_thread_id,
            )
            .order_by(ThreadSummaryRevisionRow.revision.desc())
            .limit(1)
        )
        if row is None:
            return ()
        return (
            ContextItem(
                id=f"thread-summary:{row.id}",
                kind=ContextItemKind.THREAD_SUMMARY,
                content=row.content,
                conversation_id=request.destination_id,
                source_scope=scope,
                budget_priority=92,
            ),
        )

    async def _load_recent_thread_messages(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        request: ConversationContextRequest,
        *,
        harness_thread_id: str,
        destination_conversation_id: str,
    ) -> tuple[ContextItem, ...]:
        rows = tuple(
            await session.scalars(
                select(SanitizedMessageRow)
                .where(
                    SanitizedMessageRow.organization_id == scope.organization_id,
                    or_(
                        and_(
                            SanitizedMessageRow.harness_thread_id == harness_thread_id,
                            SanitizedMessageRow.provider_thread_root_ts.is_(None),
                        ),
                        SanitizedMessageRow.provider_thread_root_ts == request.thread_root_ts,
                    ),
                    SanitizedMessageRow.conversation_id == destination_conversation_id,
                    SanitizedMessageRow.external_event_id != request.current_event_id,
                    SanitizedMessageRow.provider_message_ts.is_not(None),
                    SanitizedMessageRow.provider_message_ts < request.current_message_ts,
                )
                .order_by(
                    SanitizedMessageRow.recorded_at.desc(),
                    SanitizedMessageRow.id.desc(),
                )
                .limit(request.max_thread_messages + 1)
            )
        )
        if len(rows) > request.max_thread_messages:
            raise ConversationContextOverflowError("persisted_thread_history_incomplete")
        current_message_ts = _parse_slack_timestamp(request.current_message_ts)
        if current_message_ts is None or any(
            row.conversation_id != destination_conversation_id
            or row.external_event_id == request.current_event_id
            or (
                row.provider_thread_root_ts is not None
                and row.provider_thread_root_ts != request.thread_root_ts
            )
            or (row_ts := _parse_slack_timestamp(row.provider_message_ts)) is None
            or row_ts >= current_message_ts
            for row in rows
        ):
            raise ConversationContextAuthorizationError(
                "persisted thread query returned content outside the exact event boundary"
            )
        chronological = tuple(
            sorted(
                rows,
                key=lambda row: (
                    _parse_slack_timestamp(row.provider_message_ts) or Decimal(0),
                    row.id,
                ),
            )
        )
        recent_ids = frozenset(row.id for row in chronological[-12:])
        classifications = classify_thread_transcript(
            tuple(
                ThreadTurnRetentionInput(
                    content=row.text,
                    actor_id=row.actor_id or "unknown",
                    speaker_role=row.role,
                    is_root=row.provider_message_ts == request.thread_root_ts,
                    is_recent=row.id in recent_ids,
                )
                for row in chronological
            )
        )
        items: list[ContextItem] = []
        for row, (retention, priority) in zip(
            chronological,
            classifications,
            strict=True,
        ):
            content = f"{row.role.title()}: {row.text}"
            items.append(
                ContextItem(
                    id=f"thread-message:{row.id}",
                    kind=ContextItemKind.CONVERSATION_TURN,
                    content=content,
                    conversation_id=request.destination_id,
                    source_scope=ScopeKey(
                        organization_id=row.organization_id,
                        strategy_id=row.strategy_id,
                    ),
                    source_actor_id=row.actor_id,
                    retention=retention,
                    budget_priority=priority,
                )
            )
        return tuple(items)


def _turn_content(task: TaskRow) -> str:
    answer = task.final_output.strip() if task.final_output else ""
    if answer:
        return f"User: {task.objective.strip()}\nLeo: {answer}"
    return f"User: {task.objective.strip()}"


def _parse_slack_timestamp(value: object) -> Decimal | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return None
    return parsed if parsed > 0 else None


def _slack_conversation_kind(destination_kind: str) -> str:
    return {
        "channel": "ordinary_internal",
        "dm": "dm",
        "group_dm": "mpim",
        "shared": "shared",
        "external": "external",
    }[destination_kind]


def _tokens(value: str) -> frozenset[str]:
    return frozenset(re.findall(r"[A-Za-z0-9_.-]{2,64}", value.lower()))


def _relevance(content: str, objective: str) -> float:
    query = _tokens(objective)
    if not query:
        return 0.0
    return len(query & _tokens(content)) / len(query)


def _priority(item: ContextItem, objective: str) -> int:
    base = 75 if item.kind is ContextItemKind.MEMORY else 65
    return min(99, base + round(_relevance(item.content, objective) * 20))
