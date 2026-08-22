"""Scoped memory cards/chunks and run-bound opaque handles."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime

from pydantic import Field

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class MemoryCard(ContractModel):
    record_id: NonEmptyStr
    revision: int = Field(ge=1)
    scope: ScopeKey
    title: NonEmptyStr
    excerpt: NonEmptyStr = Field(max_length=1000)
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    created_at: datetime


class MemoryChunk(ContractModel):
    chunk_id: NonEmptyStr
    record_id: NonEmptyStr
    scope: ScopeKey
    text: NonEmptyStr = Field(max_length=4000)
    ordinal: int = Field(ge=0)


class MemoryHandle(ContractModel):
    handle: NonEmptyStr
    run_id: NonEmptyStr
    scope: ScopeKey
    record_id: NonEmptyStr
    revision: int = Field(ge=1)
    expires_at: datetime


class HandleStore:
    def __init__(self) -> None:
        self._handles: dict[str, MemoryHandle] = {}

    def issue(
        self, *, run_id: str, scope: ScopeKey, card: MemoryCard, expires_at: datetime
    ) -> MemoryHandle:
        token = "mh_" + secrets.token_urlsafe(18)
        handle = MemoryHandle(
            handle=token,
            run_id=run_id,
            scope=scope,
            record_id=card.record_id,
            revision=card.revision,
            expires_at=expires_at,
        )
        self._handles[token] = handle
        return handle

    def open(self, handle: str, *, run_id: str, scope: ScopeKey, now: datetime) -> MemoryHandle:
        item = self._handles.get(handle)
        if item is None or item.run_id != run_id or item.scope != scope or item.expires_at <= now:
            raise KeyError("memory_handle_not_authorized")
        return item

    def digest(self, handle: MemoryHandle) -> str:
        return hashlib.sha256(handle.handle.encode()).hexdigest()
