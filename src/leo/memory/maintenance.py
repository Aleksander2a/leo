"""Manual, bounded memory maintenance; no scheduler or wildcard deletion."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class MaintenanceJobStatus(StrEnum):
    QUEUED = "queued"
    RETRY = "retry"
    DEAD = "dead"
    COMPLETED = "completed"


class PurgeTarget(ContractModel):
    record_id: NonEmptyStr
    generation: int = Field(ge=1)
    current_revision: int = Field(ge=1)


class PurgePlan(ContractModel):
    scope: ScopeKey
    record_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=100)
    manifest_hash: NonEmptyStr
    confirmation_token: NonEmptyStr
    targets: tuple[PurgeTarget, ...] = ()

    @model_validator(mode="after")
    def target_snapshot_matches_ids(self) -> PurgePlan:
        if self.targets and tuple(item.record_id for item in self.targets) != self.record_ids:
            raise ValueError("purge target snapshot does not match record IDs")
        return self


class PurgeResult(ContractModel):
    scope: ScopeKey
    manifest_hash: NonEmptyStr
    purged_record_ids: tuple[NonEmptyStr, ...] = ()
    already_absent_record_ids: tuple[NonEmptyStr, ...] = ()
    deleted_revision_count: int = Field(ge=0)
    deleted_source_count: int = Field(ge=0)
    invalidated_cache_count: int = Field(ge=0)
    deleted_embedding_job_count: int = Field(ge=0)


class MaintenanceHealth(ContractModel):
    scope: ScopeKey
    expired_active_records: int = Field(ge=0)
    queued_embedding_jobs: int = Field(ge=0)
    retry_embedding_jobs: int = Field(ge=0)
    dead_embedding_jobs: int = Field(ge=0)
    retrieval_cache_entries: int = Field(ge=0)


def make_purge_plan(
    scope: ScopeKey,
    record_ids: tuple[str, ...],
    *,
    targets: tuple[PurgeTarget, ...] = (),
) -> PurgePlan:
    if not record_ids or any("*" in record_id or "?" in record_id for record_id in record_ids):
        raise ValueError("purge requires explicit non-wildcard record IDs")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("purge record IDs must be unique")
    payload = {
        "scope": scope.model_dump(mode="json"),
        "record_ids": record_ids,
        "targets": [item.model_dump(mode="json") for item in targets],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return PurgePlan(
        scope=scope,
        record_ids=record_ids,
        manifest_hash=digest,
        confirmation_token=f"confirm:{digest[:16]}",
        targets=targets,
    )


def validate_confirmation(plan: PurgePlan, confirmation_token: str, *, scope: ScopeKey) -> None:
    if plan.scope != scope or confirmation_token != plan.confirmation_token:
        raise ValueError("purge confirmation is stale or unauthorized")
