"""Deterministic in-memory memory store used before retrieval adapters."""

from __future__ import annotations

import asyncio

from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.memory.lifecycle import next_record, validate_append_revision, validate_initial_revision
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.ports import MemoryStore


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._revisions: dict[tuple[str, int], MemoryRevision] = {}
        self._sources: dict[str, MemorySource] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        record: MemoryRecord,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...],
    ) -> MemoryRecord:
        validate_initial_revision(record, revision)
        if len({source.id for source in sources}) != len(sources):
            raise StoreError("duplicate memory source")
        if any(source.scope != record.scope for source in sources):
            raise StoreError("memory source is outside the record scope")
        if set(revision.source_ids) != {source.id for source in sources}:
            raise StoreError("memory revision source provenance is incomplete")
        async with self._lock:
            if record.id in self._records or any(source.id in self._sources for source in sources):
                raise ConcurrencyError("memory record or source already exists")
            self._records[record.id] = record
            self._revisions[(record.id, revision.number)] = revision
            for source in sources:
                self._sources[source.id] = source
            return record

    async def append_revision(
        self,
        scope: ScopeKey,
        record_id: str,
        expected_revision: int,
        revision: MemoryRevision,
        sources: tuple[MemorySource, ...] = (),
    ) -> MemoryRecord:
        async with self._lock:
            record = self._records.get(record_id)
            if record is None or record.scope != scope:
                raise NotFoundError("memory record not found")
            validate_append_revision(record, expected_revision, revision)
            if len({source.id for source in sources}) != len(sources):
                raise StoreError("duplicate memory source")
            if any(source.id in self._sources for source in sources):
                raise ConcurrencyError("memory source already exists")
            if any(
                source.scope != record.scope
                or source.visibility != revision.visibility
                or source.namespace_id != revision.namespace_id
                for source in sources
            ):
                raise StoreError("memory source is outside the revision scope")
            current = self._revisions[(record.id, record.current_revision)]
            new_source_ids = {source.id for source in sources}
            expected_sources = set(current.source_ids) | new_source_ids
            if set(revision.source_ids) != expected_sources:
                raise StoreError("memory revision must retain prior and current provenance")
            if any(
                source_id not in self._sources and source_id not in new_source_ids
                for source_id in revision.source_ids
            ):
                raise StoreError("memory revision references an unknown source")
            if (record_id, revision.number) in self._revisions:
                raise ConcurrencyError("memory revision already exists")
            updated = next_record(record, revision)
            for source in sources:
                self._sources[source.id] = source
            self._revisions[(record_id, revision.number)] = revision
            self._records[record_id] = updated
            return updated

    async def current(self, scope: ScopeKey, record_id: str) -> MemoryRevision | None:
        async with self._lock:
            record = self._records.get(record_id)
            if record is None or record.scope != scope:
                raise NotFoundError("memory record not found")
            if record.status is MemoryStatus.RETRACTED:
                return None
            return self._revisions[(record.id, record.current_revision)]

    async def forget(self, scope: ScopeKey, record_id: str, reason: str) -> MemoryRecord:
        if not reason.strip():
            raise StoreError("forget reason must be non-empty")
        async with self._lock:
            record = self._records.get(record_id)
            if record is None or record.scope != scope:
                raise NotFoundError("memory record not found")
            if record.status is MemoryStatus.RETRACTED:
                return record
            current = self._revisions[(record.id, record.current_revision)]
            revision = current.model_copy(
                update={
                    "id": f"{current.id}:forget:{record.generation + 1}",
                    "number": current.number + 1,
                    "status": MemoryStatus.RETRACTED,
                    "reason": reason,
                    "supersedes_revision": current.number,
                }
            )
            updated = next_record(record, revision)
            self._revisions[(record.id, revision.number)] = revision
            self._records[record.id] = updated
            return updated

    async def list_active(
        self,
        scope: ScopeKey,
        *,
        visibility: MemoryVisibility,
        namespace_id: str,
        kind: MemoryKind | None = None,
        limit: int = 50,
    ) -> tuple[tuple[str, MemoryRevision], ...]:
        async with self._lock:
            matches = tuple(
                (record.id, self._revisions[(record.id, record.current_revision)])
                for record in self._records.values()
                if record.scope == scope
                and record.visibility is visibility
                and record.namespace_id == namespace_id
                and record.status is MemoryStatus.ACTIVE
                and (kind is None or record.kind is kind)
            )
            return matches[:limit]
