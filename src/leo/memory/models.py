"""Append-only memory records with explicit visibility and provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class MemoryVisibility(StrEnum):
    THREAD_LOCAL = "thread_local"
    CONVERSATION_LOCAL = "conversation_local"
    # Deprecated compatibility spelling for persisted pre-D-054 rows.
    CHANNEL_LOCAL = "channel_local"
    ACTOR_PRIVATE = "actor_private"
    STRATEGY_SHARED = "strategy_shared"
    ORGANIZATION_SHARED = "organization_shared"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONTESTED = "contested"
    RETRACTED = "retracted"


class MemoryKind(StrEnum):
    NOTE = "note"
    PREFERENCE = "preference"
    RESEARCH_CONTEXT = "research_context"


class MemorySource(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    source_kind: NonEmptyStr
    reference: NonEmptyStr
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr


class MemoryRevision(ContractModel):
    id: NonEmptyStr
    record_id: NonEmptyStr
    number: int = Field(ge=1)
    content: NonEmptyStr = Field(max_length=16_384)
    content_hash: str = Field(min_length=64, max_length=64)
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    sensitivity: float = Field(ge=0, le=1)
    valid_from: datetime
    valid_until: datetime | None = None
    recorded_at: datetime
    expires_at: datetime | None = None
    status: MemoryStatus = MemoryStatus.ACTIVE
    actor_id: NonEmptyStr
    reason: NonEmptyStr
    supersedes_revision: int | None = Field(default=None, ge=1)
    source_type: Literal["explicit", "autonomous"] = "explicit"

    @model_validator(mode="after")
    def validate_content_and_time(self) -> MemoryRevision:
        expected_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if self.content_hash != expected_hash:
            raise ValueError("memory content hash does not match content")
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("memory valid_until must be after valid_from")
        if self.expires_at is not None and self.expires_at <= self.recorded_at:
            raise ValueError("memory expires_at must be after recorded_at")
        if self.status is MemoryStatus.SUPERSEDED and self.supersedes_revision is None:
            raise ValueError("superseded memory revision requires a predecessor")
        return self

    @classmethod
    def from_content(
        cls,
        *,
        id: str,
        record_id: str,
        number: int,
        content: str,
        source_ids: tuple[str, ...],
        visibility: MemoryVisibility,
        namespace_id: str,
        sensitivity: float,
        valid_from: datetime,
        recorded_at: datetime,
        actor_id: str,
        reason: str,
        status: MemoryStatus = MemoryStatus.ACTIVE,
        supersedes_revision: int | None = None,
        valid_until: datetime | None = None,
        expires_at: datetime | None = None,
        source_type: Literal["explicit", "autonomous"] = "explicit",
    ) -> MemoryRevision:
        return cls(
            id=id,
            record_id=record_id,
            number=number,
            content=content,
            content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
            source_ids=source_ids,
            visibility=visibility,
            namespace_id=namespace_id,
            sensitivity=sensitivity,
            valid_from=valid_from,
            valid_until=valid_until,
            recorded_at=recorded_at,
            expires_at=expires_at,
            status=status,
            actor_id=actor_id,
            reason=reason,
            supersedes_revision=supersedes_revision,
            source_type=source_type,
        )


class MemoryRecord(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    kind: MemoryKind
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    current_revision: int = Field(default=1, ge=1)
    generation: int = Field(default=1, ge=1)
    status: MemoryStatus = MemoryStatus.ACTIVE
    created_at: datetime


def canonical_payload(revision: MemoryRevision) -> dict[str, JsonValue]:
    """Return a redaction-safe event payload without memory content."""

    return {
        "record_id": revision.record_id,
        "revision": revision.number,
        "content_hash": revision.content_hash,
        "source_count": len(revision.source_ids),
        "visibility": revision.visibility.value,
        "sensitivity": revision.sensitivity,
        "status": revision.status.value,
        "reason": revision.reason,
    }


def content_digest(content: str) -> str:
    return hashlib.sha256(json.dumps(content, ensure_ascii=False).encode("utf-8")).hexdigest()
