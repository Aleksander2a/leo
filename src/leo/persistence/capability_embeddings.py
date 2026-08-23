"""Durable L2 cache for tool/capability embeddings, behind the in-process L1.

Tool descriptions only change on a code deploy, so re-embedding the entire
catalog on every process restart is pure waste. This persists each embedding
once and serves it back across restarts; a database error degrades to "cache
miss, re-embed via the gateway" rather than failing discovery.
"""

from __future__ import annotations

import logging

from sqlalchemy import select, tuple_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.capabilities.embeddings import CapabilityEmbeddingKey, EmbeddingVector
from leo.harness.ports import IdGenerator
from leo.persistence.schema import CapabilityEmbeddingRow

_logger = logging.getLogger(__name__)


class PostgresCapabilityEmbeddingStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, ids: IdGenerator) -> None:
        self._sessions = sessions
        self._ids = ids

    async def get_many(
        self, keys: tuple[CapabilityEmbeddingKey, ...]
    ) -> dict[CapabilityEmbeddingKey, EmbeddingVector]:
        if not keys:
            return {}
        try:
            async with self._sessions() as session:
                rows = await session.scalars(
                    select(CapabilityEmbeddingRow).where(
                        tuple_(
                            CapabilityEmbeddingRow.capability_id,
                            CapabilityEmbeddingRow.content_hash,
                            CapabilityEmbeddingRow.model,
                        ).in_(keys)
                    )
                )
                return {
                    (row.capability_id, row.content_hash, row.model): tuple(row.embedding)
                    for row in rows
                }
        except SQLAlchemyError:
            _logger.warning(
                "capability embedding L2 read failed; falling back to gateway", exc_info=True
            )
            return {}

    async def put_many(
        self, items: tuple[tuple[CapabilityEmbeddingKey, EmbeddingVector], ...]
    ) -> None:
        if not items:
            return
        try:
            async with self._sessions() as session, session.begin():
                for (capability_id, content_hash, model), vector in items:
                    await session.execute(
                        pg_insert(CapabilityEmbeddingRow)
                        .values(
                            id=self._ids.new("capability-embedding"),
                            capability_id=capability_id,
                            content_hash=content_hash,
                            model=model,
                            embedding=list(vector),
                        )
                        .on_conflict_do_nothing(
                            index_elements=["capability_id", "content_hash", "model"]
                        )
                    )
        except SQLAlchemyError:
            _logger.warning(
                "capability embedding L2 write failed; will re-embed next time", exc_info=True
            )
