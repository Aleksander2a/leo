"""Scope-first Postgres FTS adapter for deterministic memory retrieval."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import Select

from leo.harness.models import ScopeKey
from leo.memory.models import MemoryRevision, MemoryStatus, MemoryVisibility
from leo.memory.retrieval import (
    AuthorizedMemoryNamespace,
    MemoryRetrievalError,
    MemorySearchHit,
    MemorySearchRequest,
    conflict_group_id,
    normalize_memory_query,
    select_bounded_memory_hits,
)
from leo.persistence.schema import MemoryRecordRow, MemoryRevisionRow

_SEARCH_POOL_LIMIT = 1_000
_RETRIEVAL_POLICY = "postgres-fts-scope-first-v2"


class PostgresMemoryRetriever:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def search(self, request: MemorySearchRequest) -> tuple[MemorySearchHit, ...]:
        async with self._sessions() as session, session.begin():
            return await execute_memory_search(session, request)


async def execute_memory_search(
    session: AsyncSession,
    request: MemorySearchRequest,
    *,
    record_ids_hint: tuple[str, ...] | None = None,
) -> tuple[MemorySearchHit, ...]:
    """Execute scope-first retrieval inside a caller-owned authorization transaction."""

    statement = build_memory_search_statement(request, record_ids_hint=record_ids_hint)
    rows = (await session.execute(statement)).mappings().all()
    if len(rows) > _SEARCH_POOL_LIMIT:
        raise MemoryRetrievalError("authorized_search_pool_exhausted")
    hits = tuple(_hit_from_mapping(dict(row)) for row in rows)
    return select_bounded_memory_hits(hits, request)


def build_memory_search_statement(
    request: MemorySearchRequest,
    *,
    record_ids_hint: tuple[str, ...] | None = None,
) -> Select[Any]:
    """Build a bounded query whose hard filters precede ranking and limiting."""

    normalized = normalize_memory_query(request.query)
    query_vector = func.plainto_tsquery("simple", normalized)
    rank = func.ts_rank(MemoryRevisionRow.search_vector, query_vector)
    authorized = tuple(
        and_(
            MemoryRevisionRow.visibility == item.visibility.value,
            MemoryRevisionRow.namespace_id == item.namespace_id,
            MemoryRecordRow.visibility == item.visibility.value,
            MemoryRecordRow.namespace_id == item.namespace_id,
        )
        for item in sorted(
            request.authorized_namespaces,
            key=lambda item: (item.visibility.value, item.namespace_id),
        )
    )
    # PostgreSQL applies this SELECT's WHERE clause before computing rank/order.
    # Unauthorized, stale, superseded, retracted, expired, or over-sensitivity rows
    # therefore cannot enter the ranked pool. Overflow fails closed in the adapter.
    authorized_ranked = (
        select(
            MemoryRevisionRow.id.label("revision_id"),
            MemoryRevisionRow.record_id,
            MemoryRevisionRow.number,
            MemoryRevisionRow.content,
            MemoryRevisionRow.content_hash,
            MemoryRevisionRow.source_ids,
            MemoryRevisionRow.visibility,
            MemoryRevisionRow.namespace_id,
            MemoryRevisionRow.sensitivity,
            MemoryRevisionRow.valid_from,
            MemoryRevisionRow.valid_until,
            MemoryRevisionRow.recorded_at,
            MemoryRevisionRow.expires_at,
            MemoryRevisionRow.status.label("revision_status"),
            MemoryRevisionRow.actor_id,
            MemoryRevisionRow.reason,
            MemoryRevisionRow.supersedes_revision,
            MemoryRecordRow.status.label("record_status"),
            rank.label("score"),
        )
        .join(
            MemoryRecordRow,
            (MemoryRecordRow.id == MemoryRevisionRow.record_id)
            & (MemoryRecordRow.current_revision == MemoryRevisionRow.number),
        )
        .where(
            MemoryRevisionRow.organization_id == request.scope.organization_id,
            MemoryRecordRow.organization_id == request.scope.organization_id,
            MemoryRecordRow.status.in_((MemoryStatus.ACTIVE.value, MemoryStatus.CONTESTED.value)),
            MemoryRevisionRow.status.in_((MemoryStatus.ACTIVE.value, MemoryStatus.CONTESTED.value)),
            or_(*authorized),
            MemoryRevisionRow.sensitivity <= request.max_sensitivity,
            MemoryRevisionRow.valid_from <= request.as_of,
            (
                MemoryRevisionRow.valid_until.is_(None)
                | (MemoryRevisionRow.valid_until > request.as_of)
            ),
            (
                MemoryRevisionRow.expires_at.is_(None)
                | (MemoryRevisionRow.expires_at > request.as_of)
            ),
            MemoryRevisionRow.search_vector.op("@@")(query_vector),
        )
    )
    if record_ids_hint is not None:
        authorized_ranked = authorized_ranked.where(MemoryRecordRow.id.in_(record_ids_hint))
    ranked_cte = authorized_ranked.cte("authorized_current_ranked_memory")
    return (
        select(ranked_cte)
        .order_by(
            ranked_cte.c.score.desc(),
            ranked_cte.c.recorded_at.desc(),
            ranked_cte.c.revision_id,
        )
        .limit(_SEARCH_POOL_LIMIT + 1)
    )


def _hit_from_mapping(row: dict[str, Any]) -> MemorySearchHit:
    revision = MemoryRevision(
        id=str(row["revision_id"]),
        record_id=str(row["record_id"]),
        number=int(row["number"]),
        content=str(row["content"]),
        content_hash=str(row["content_hash"]),
        source_ids=tuple(str(item) for item in row["source_ids"]),
        visibility=MemoryVisibility(str(row["visibility"])),
        namespace_id=str(row["namespace_id"]),
        sensitivity=float(row["sensitivity"]),
        valid_from=row["valid_from"],
        valid_until=row["valid_until"],
        recorded_at=row["recorded_at"],
        expires_at=row["expires_at"],
        status=MemoryStatus(str(row["revision_status"])),
        actor_id=str(row["actor_id"]),
        reason=str(row["reason"]),
        supersedes_revision=row["supersedes_revision"],
    )
    lifecycle_status = (
        MemoryStatus.CONTESTED
        if MemoryStatus.CONTESTED.value in {str(row["record_status"]), str(row["revision_status"])}
        else MemoryStatus.ACTIVE
    )
    return MemorySearchHit(
        record_id=revision.record_id,
        revision=revision.number,
        content=revision.content,
        score=float(row["score"]),
        match_reason=_RETRIEVAL_POLICY,
        recorded_at=revision.recorded_at,
        visibility=revision.visibility,
        namespace_id=revision.namespace_id,
        source_ids=revision.source_ids,
        lifecycle_status=lifecycle_status,
        conflict_group_id=conflict_group_id(revision),
    )


def scope_request(
    scope: ScopeKey,
    *,
    query: str,
    authorized_namespaces: frozenset[AuthorizedMemoryNamespace],
    access_hash: str,
    membership_hash: str,
    as_of: datetime,
    max_sensitivity: float = 1,
    limit: int = 10,
    per_namespace_limit: int | None = None,
) -> MemorySearchRequest:
    return MemorySearchRequest(
        scope=scope,
        query=query,
        authorized_namespaces=authorized_namespaces,
        access_hash=access_hash,
        membership_hash=membership_hash,
        as_of=as_of,
        max_sensitivity=max_sensitivity,
        limit=limit,
        per_namespace_limit=per_namespace_limit,
    )
