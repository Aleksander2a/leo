"""Best-effort pgvector indexing for memory revisions.

Layered on top of ``PostgresMemoryStore`` rather than folded into it: a
revision is authoritative and fully retrievable through FTS the moment the
store commits it, so embedding it is a pure enhancement that must never make
a memory write fail. A gateway error or missing API key here degrades that
one revision to lexical-only recall, nothing more.
"""

from __future__ import annotations

import logging

from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.capabilities.embeddings import OpenRouterEmbeddingGateway
from leo.harness.models import ScopeKey
from leo.harness.ports import IdGenerator
from leo.memory.models import MemoryRevision
from leo.persistence.schema import MemoryEmbeddingRow

_logger = logging.getLogger(__name__)


class PostgresMemoryEmbeddingIndexer:
    """Embeds a revision's content and upserts it into ``memory_embeddings``."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        gateway: OpenRouterEmbeddingGateway,
        *,
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._gateway = gateway
        self._ids = ids

    async def index(self, scope: ScopeKey, revision: MemoryRevision, *, source_type: str) -> None:
        (vector,) = await self._gateway.embed((revision.content,))
        if vector is None:
            _logger.warning(
                "embedding gateway returned no vector for memory revision %s", revision.id
            )
            return
        async with self._sessions() as session, session.begin():
            await session.execute(
                pg_insert(MemoryEmbeddingRow)
                .values(
                    id=self._ids.new("memory-embedding"),
                    revision_id=revision.id,
                    record_id=revision.record_id,
                    organization_id=scope.organization_id,
                    strategy_id=scope.strategy_id,
                    content_hash=revision.content_hash,
                    model=self._gateway.model,
                    embedding=list(vector),
                )
                .on_conflict_do_nothing(index_elements=["revision_id", "content_hash", "model"])
            )
