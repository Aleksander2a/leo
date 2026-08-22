"""Scope-first memory repository port."""

from __future__ import annotations

from typing import Protocol

from leo.harness.models import ScopeKey
from leo.memory.models import MemoryRecord, MemoryRevision, MemorySource


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
