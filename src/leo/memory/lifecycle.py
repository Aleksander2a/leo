"""Pure lifecycle checks for append-only memory revisions."""

from __future__ import annotations

from leo.harness.store_errors import StoreError
from leo.memory.models import MemoryRecord, MemoryRevision, MemoryStatus


def validate_initial_revision(record: MemoryRecord, revision: MemoryRevision) -> None:
    if revision.record_id != record.id or revision.number != 1:
        raise StoreError("initial memory revision identity or number is invalid")
    if revision.visibility != record.visibility or revision.namespace_id != record.namespace_id:
        raise StoreError("memory revision visibility does not match its record")
    if revision.status is not MemoryStatus.ACTIVE:
        raise StoreError("initial memory revision must be active")


def validate_append_revision(
    record: MemoryRecord,
    expected_revision: int,
    revision: MemoryRevision,
) -> None:
    if record.status is MemoryStatus.RETRACTED:
        raise StoreError("retracted memory cannot receive a new revision")
    if record.current_revision != expected_revision:
        raise StoreError("memory revision is stale")
    if revision.record_id != record.id or revision.number != expected_revision + 1:
        raise StoreError("memory revision must append exactly one revision")
    if revision.visibility != record.visibility or revision.namespace_id != record.namespace_id:
        raise StoreError("memory revision visibility cannot change")
    if revision.supersedes_revision not in {None, expected_revision}:
        raise StoreError("memory revision predecessor does not match current revision")


def next_record(record: MemoryRecord, revision: MemoryRevision) -> MemoryRecord:
    status = revision.status
    return record.model_copy(
        update={
            "current_revision": revision.number,
            "generation": record.generation + 1,
            "status": status,
        }
    )
