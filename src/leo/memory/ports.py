"""Scope-first memory repository port."""

from __future__ import annotations

from typing import Protocol

from leo.harness.models import ScopeKey
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)


class MemoryStore(Protocol):
    async def create(
        self,
        record: MemoryRecord,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...],
    ) -> MemoryRecord: ...

    async def append_revision(
        self,
        scope: ScopeKey,
        record_id: str,
        expected_revision: int,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...] = (),
    ) -> MemoryRecord: ...

    async def current(self, scope: ScopeKey, record_id: str) -> MemoryRevision | None: ...

    async def forget(self, scope: ScopeKey, record_id: str, reason: str) -> MemoryRecord: ...

    async def list_active(
        self,
        scope: ScopeKey,
        *,
        visibility: MemoryVisibility,
        namespace_id: str,
        kind: MemoryKind | None = None,
        limit: int = 50,
    ) -> tuple[tuple[str, MemoryRevision], ...]:
        """Return (record_id, current_revision) pairs used for candidate governance.

        This is a bounded lookup for duplicate/contradiction assessment, not a
        retrieval-ranked search -- callers needing ranked recall use the retrieval
        module instead.
        """
        ...


class MemoryEmbeddingIndexer(Protocol):
    """Best-effort semantic index maintenance, separate from the durable store.

    A revision is authoritative the moment ``MemoryStore`` commits it; indexing
    is a pure enhancement layered on top, so failures here must never block or
    roll back a memory write. Implementations should embed the revision content
    and upsert it into whatever backs vector recall.
    """

    async def index(
        self, scope: ScopeKey, revision: MemoryRevision, *, source_type: str
    ) -> None: ...
