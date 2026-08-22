"""Durable progressive-memory search/open with repeat authorization."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta

from sqlalchemy import and_, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.memory.cache import RetrievalCacheKey
from leo.memory.models import MemoryStatus, MemoryVisibility
from leo.memory.navigation import (
    NAVIGATION_POLICY_VERSION,
    AuthorizedMemoryDocument,
    MemoryNavigationAuthority,
    MemoryNavigationError,
    MemoryResultKind,
    ProgressiveMemoryItem,
    ProgressiveMemoryOpenResult,
    ProgressiveMemorySearchResult,
    deterministic_memory_chunks,
    membership_snapshot_hash,
    opaque_memory_reference,
    project_open_window,
    source_conversation_label,
)
from leo.memory.retrieval import MemorySearchHit, MemorySearchRequest, normalized_query_hash
from leo.persistence.memory_retrieval import execute_memory_search
from leo.persistence.schema import (
    ConversationActorMembershipRow,
    ConversationRow,
    MemoryCapabilityHandleRow,
    MemoryRecordRow,
    MemoryRetrievalCacheRow,
    MemoryRevisionRow,
)


class PostgresProgressiveMemoryService:
    """Searches, issues capabilities, and atomically reauthorizes every open."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(
        self,
        authority: MemoryNavigationAuthority,
        *,
        query: str,
        now: datetime,
        limit: int = 8,
        per_namespace_limit: int = 3,
        inline_max_chars: int = 900,
        handle_ttl: timedelta = timedelta(minutes=20),
        max_opens: int = 8,
    ) -> ProgressiveMemorySearchResult:
        _validate_clock(now)
        if inline_max_chars < 128 or inline_max_chars > 1_200:
            raise ValueError("inline memory size must be between 128 and 1200 characters")
        if handle_ttl <= timedelta(0) or handle_ttl > timedelta(hours=2):
            raise ValueError("memory handle lifetime must be positive and at most two hours")
        if max_opens < 1 or max_opens > 64:
            raise ValueError("memory handle open budget must be between 1 and 64")
        request = MemorySearchRequest(
            scope=authority.scope,
            query=query,
            authorized_namespaces=authority.authorized_namespaces,
            access_hash=authority.access_hash,
            membership_hash=authority.membership_hash,
            as_of=now,
            limit=limit,
            per_namespace_limit=per_namespace_limit,
        )
        async with self._sessions() as session, session.begin():
            await _validate_current_authority(session, authority, lock=True)
            generation, content_digest = await _memory_generation_manifest(session, authority)
            cache_key = RetrievalCacheKey.from_request(
                request,
                generation=generation,
                policy_version="postgres-fts-scope-first-v2",
                content_digest=content_digest,
            )
            cached = await session.scalar(
                select(MemoryRetrievalCacheRow).where(
                    MemoryRetrievalCacheRow.organization_id == authority.scope.organization_id,
                    MemoryRetrievalCacheRow.strategy_id == authority.scope.strategy_id,
                    MemoryRetrievalCacheRow.key_hash == cache_key.digest(),
                    MemoryRetrievalCacheRow.generation == cache_key.generation,
                    or_(
                        MemoryRetrievalCacheRow.expires_at.is_(None),
                        MemoryRetrievalCacheRow.expires_at > now,
                    ),
                )
            )
            cache_status = "miss"
            hits: tuple[MemorySearchHit, ...]
            if cached is not None:
                hits = await execute_memory_search(
                    session,
                    request,
                    record_ids_hint=tuple(str(item) for item in cached.result_ids),
                )
                if tuple(hit.record_id for hit in hits) == tuple(cached.result_ids):
                    cache_status = "hit"
                else:
                    hits = await execute_memory_search(session, request)
            else:
                hits = await execute_memory_search(session, request)
            if cache_status == "miss":
                key_hash = cache_key.digest()
                await session.execute(
                    pg_insert(MemoryRetrievalCacheRow)
                    .values(
                        id=f"cache-{key_hash[:58]}",
                        organization_id=authority.scope.organization_id,
                        strategy_id=authority.scope.strategy_id,
                        key_hash=key_hash,
                        generation=cache_key.generation,
                        result_ids=[hit.record_id for hit in hits],
                        expires_at=now + timedelta(minutes=5),
                    )
                    .on_conflict_do_nothing()
                )
            projected: list[ProgressiveMemoryItem] = []
            for hit in hits:
                projected.append(
                    await _project_hit(
                        session,
                        authority,
                        hit,
                        now=now,
                        inline_max_chars=inline_max_chars,
                        expires_at=now + handle_ttl,
                        max_opens=max_opens,
                    )
                )
            items = tuple(projected)
        return ProgressiveMemorySearchResult(
            items=items,
            query_hash=normalized_query_hash(query),
            selected_count=len(items),
            cache_status=cache_status,
        )

    async def open(
        self,
        authority: MemoryNavigationAuthority,
        *,
        handle: str,
        now: datetime,
        start_ordinal: int = 0,
        max_chunks: int = 4,
    ) -> ProgressiveMemoryOpenResult:
        document = await self._authorize_open(authority, handle=handle, now=now)
        return project_open_window(
            document,
            start_ordinal=start_ordinal,
            max_chunks=max_chunks,
        )

    async def search_within(
        self,
        authority: MemoryNavigationAuthority,
        *,
        handle: str,
        query: str,
        now: datetime,
        max_chunks: int = 4,
    ) -> ProgressiveMemoryOpenResult:
        document = await self._authorize_open(authority, handle=handle, now=now)
        return project_open_window(document, query=query, max_chunks=max_chunks)

    async def _authorize_open(
        self,
        authority: MemoryNavigationAuthority,
        *,
        handle: str,
        now: datetime,
    ) -> AuthorizedMemoryDocument:
        _validate_clock(now)
        handle_hash = _handle_hash(handle)
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(MemoryCapabilityHandleRow)
                .where(MemoryCapabilityHandleRow.handle_hash == handle_hash)
                .with_for_update()
            )
            if row is None or not _handle_matches_authority(row, authority):
                raise MemoryNavigationError("memory_handle_not_authorized")
            if row.invalidated_at is not None:
                raise MemoryNavigationError("memory_handle_invalidated")
            if row.expires_at <= now:
                raise MemoryNavigationError("memory_handle_expired")
            if row.open_count >= row.max_opens:
                raise MemoryNavigationError("memory_handle_budget_exhausted")
            await _validate_current_authority(session, authority, lock=True)
            revision_row = await session.scalar(
                select(MemoryRevisionRow)
                .join(
                    MemoryRecordRow,
                    (MemoryRecordRow.id == MemoryRevisionRow.record_id)
                    & (MemoryRecordRow.current_revision == MemoryRevisionRow.number),
                )
                .where(
                    MemoryRecordRow.organization_id == authority.scope.organization_id,
                    MemoryRecordRow.id == row.record_id,
                    MemoryRecordRow.current_revision == row.revision,
                    MemoryRecordRow.status.in_(("active", "contested")),
                    MemoryRecordRow.visibility == row.visibility,
                    MemoryRecordRow.namespace_id == row.namespace_id,
                    MemoryRevisionRow.status.in_(("active", "contested")),
                    MemoryRevisionRow.visibility == row.visibility,
                    MemoryRevisionRow.namespace_id == row.namespace_id,
                    MemoryRevisionRow.valid_from <= now,
                    or_(
                        MemoryRevisionRow.valid_until.is_(None),
                        MemoryRevisionRow.valid_until > now,
                    ),
                    or_(
                        MemoryRevisionRow.expires_at.is_(None),
                        MemoryRevisionRow.expires_at > now,
                    ),
                )
            )
            if revision_row is None:
                raise MemoryNavigationError("memory_handle_source_changed")
            row.open_count += 1
            row.updated_at = now
            await session.flush()
            return AuthorizedMemoryDocument(
                record_id=revision_row.record_id,
                revision=revision_row.number,
                content=revision_row.content,
                content_hash=revision_row.content_hash,
                visibility=MemoryVisibility(revision_row.visibility),
                namespace_id=revision_row.namespace_id,
                status=MemoryStatus(revision_row.status),
                handle=handle,
                reference=opaque_memory_reference(
                    revision_row.record_id,
                    revision_row.number,
                    authority.access_hash,
                ),
            )


async def invalidate_actor_memory_handles(
    session: AsyncSession,
    *,
    organization_id: str,
    team_id: str,
    actor_id: str,
    now: datetime,
    reason: str,
) -> int:
    """Safely over-invalidate prior source-set capabilities in an authority transaction."""

    result = await session.execute(
        update(MemoryCapabilityHandleRow)
        .where(
            MemoryCapabilityHandleRow.organization_id == organization_id,
            MemoryCapabilityHandleRow.team_id == team_id,
            MemoryCapabilityHandleRow.actor_id == actor_id,
            MemoryCapabilityHandleRow.invalidated_at.is_(None),
        )
        .values(
            invalidated_at=now,
            invalidation_reason=reason,
            updated_at=now,
        )
    )
    value = getattr(result, "rowcount", 0)
    return value if isinstance(value, int) and value > 0 else 0


async def _project_hit(
    session: AsyncSession,
    authority: MemoryNavigationAuthority,
    hit: MemorySearchHit,
    *,
    now: datetime,
    inline_max_chars: int,
    expires_at: datetime,
    max_opens: int,
) -> ProgressiveMemoryItem:
    reference = opaque_memory_reference(hit.record_id, hit.revision, authority.access_hash)
    label = source_conversation_label(hit.visibility, hit.namespace_id)
    contested = hit.lifecycle_status is MemoryStatus.CONTESTED
    if len(hit.content) <= inline_max_chars:
        return ProgressiveMemoryItem(
            kind=MemoryResultKind.INLINE,
            reference=reference,
            content=hit.content,
            source_conversation=label,
            lifecycle_status=hit.lifecycle_status,
            contested=contested,
        )
    token = "mh_" + secrets.token_urlsafe(24)
    handle_hash = _handle_hash(token)
    await session.execute(
        pg_insert(MemoryCapabilityHandleRow).values(
            id=f"handle-{handle_hash[:57]}",
            handle_hash=handle_hash,
            organization_id=authority.scope.organization_id,
            strategy_id=authority.scope.strategy_id,
            task_id=authority.task_id,
            run_id=authority.run_id,
            team_id=authority.team_id,
            destination_id=authority.destination_id,
            destination_kind=authority.destination_kind.value,
            actor_id=authority.actor_id,
            access_hash=authority.access_hash,
            membership_hash=authority.membership_hash,
            source_conversation_ids=list(authority.allowed_conversation_ids),
            current_thread_namespace_id=authority.current_thread_namespace_id,
            record_id=hit.record_id,
            revision=hit.revision,
            visibility=hit.visibility.value,
            namespace_id=hit.namespace_id,
            policy_version=NAVIGATION_POLICY_VERSION,
            expires_at=expires_at,
            max_opens=max_opens,
            open_count=0,
            created_at=now,
            updated_at=now,
        )
    )
    chunks = deterministic_memory_chunks(hit.content)
    excerpt = hit.content[:560].rstrip()
    if len(hit.content) > len(excerpt):
        excerpt = f"{excerpt}…"
    return ProgressiveMemoryItem(
        kind=MemoryResultKind.CARD,
        reference=reference,
        excerpt=excerpt,
        handle=token,
        chunk_count=len(chunks),
        source_conversation=label,
        lifecycle_status=hit.lifecycle_status,
        contested=contested,
    )


async def _validate_current_authority(
    session: AsyncSession,
    authority: MemoryNavigationAuthority,
    *,
    lock: bool,
) -> None:
    conversation_statement = select(ConversationRow).where(
        ConversationRow.provider == "slack",
        ConversationRow.team_id == authority.team_id,
        ConversationRow.external_id.in_(authority.allowed_conversation_ids),
    )
    if lock:
        conversation_statement = conversation_statement.with_for_update(read=True)
    conversations = tuple(await session.scalars(conversation_statement))
    by_conversation_id = {row.external_id: row for row in conversations}
    destination = by_conversation_id.get(authority.destination_id)
    if (
        destination is None
        or destination.kind != authority.destination_kind.value
        or destination.bot_presence != "present"
        or destination.lifecycle != "active"
        or set(by_conversation_id) != set(authority.allowed_conversation_ids)
        or any(row.bot_presence != "present" or row.lifecycle != "active" for row in conversations)
    ):
        raise MemoryNavigationError("memory_destination_or_bot_not_current")
    statement = select(ConversationActorMembershipRow).where(
        ConversationActorMembershipRow.organization_id == authority.scope.organization_id,
        ConversationActorMembershipRow.team_id == authority.team_id,
        ConversationActorMembershipRow.actor_id == authority.actor_id,
        ConversationActorMembershipRow.conversation_external_id.in_(
            authority.allowed_conversation_ids
        ),
    )
    if lock:
        statement = statement.with_for_update(read=True)
    rows = tuple(await session.scalars(statement))
    by_id = {row.conversation_external_id: row for row in rows}
    if set(by_id) != set(authority.allowed_conversation_ids) or any(
        row.status != "active" for row in rows
    ):
        raise MemoryNavigationError("memory_access_revoked")
    if membership_snapshot_hash(authority.allowed_conversation_ids) != authority.membership_hash:
        raise MemoryNavigationError("memory_membership_snapshot_changed")


async def _memory_generation_manifest(
    session: AsyncSession,
    authority: MemoryNavigationAuthority,
) -> tuple[int, str]:
    predicates = tuple(
        and_(
            MemoryRecordRow.visibility == item.visibility.value,
            MemoryRecordRow.namespace_id == item.namespace_id,
        )
        for item in authority.authorized_namespaces
    )
    rows = (
        await session.execute(
            select(
                MemoryRecordRow.id,
                MemoryRecordRow.generation,
                MemoryRecordRow.current_revision,
                MemoryRecordRow.status,
            )
            .where(
                MemoryRecordRow.organization_id == authority.scope.organization_id,
                or_(*predicates),
            )
            .order_by(MemoryRecordRow.id)
        )
    ).all()
    payload = [
        (str(record_id), int(generation), int(revision), str(status))
        for record_id, generation, revision, status in rows
    ]
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
    generation = max((item[1] for item in payload), default=1)
    return generation, digest


def _handle_hash(handle: str) -> str:
    if not handle.startswith("mh_") or len(handle) < 16 or len(handle) > 256:
        raise MemoryNavigationError("memory_handle_not_authorized")
    return hashlib.sha256(handle.encode("utf-8")).hexdigest()


def _handle_matches_authority(
    row: MemoryCapabilityHandleRow,
    authority: MemoryNavigationAuthority,
) -> bool:
    return (
        row.organization_id == authority.scope.organization_id
        and row.strategy_id == authority.scope.strategy_id
        and row.task_id == authority.task_id
        and row.run_id == authority.run_id
        and row.team_id == authority.team_id
        and row.destination_id == authority.destination_id
        and row.destination_kind == authority.destination_kind.value
        and row.actor_id == authority.actor_id
        and row.access_hash == authority.access_hash
        and row.membership_hash == authority.membership_hash
        and tuple(row.source_conversation_ids) == authority.allowed_conversation_ids
        and row.current_thread_namespace_id == authority.current_thread_namespace_id
        and row.policy_version == NAVIGATION_POLICY_VERSION
    )


def _validate_clock(now: datetime) -> None:
    if now.utcoffset() is None:
        raise ValueError("memory navigation clock must be timezone-aware")
