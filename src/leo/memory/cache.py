"""Scope/version/content keyed retrieval cache with explicit invalidation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import Field

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.retrieval import MemorySearchRequest, normalized_query_hash


class RetrievalCacheKey(ContractModel):
    scope: ScopeKey
    query_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    as_of: datetime
    max_sensitivity: float = Field(ge=0, le=1)
    limit: int = Field(ge=1, le=100)
    per_namespace_limit: int | None = Field(default=None, ge=1, le=50)
    generation: int = Field(ge=1)
    policy_version: NonEmptyStr
    content_digest: NonEmptyStr

    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @classmethod
    def from_request(
        cls,
        request: MemorySearchRequest,
        *,
        generation: int,
        policy_version: str,
        content_digest: str,
    ) -> RetrievalCacheKey:
        return cls(
            scope=request.scope,
            query_hash=normalized_query_hash(request.query),
            access_hash=request.access_hash,
            membership_hash=request.membership_hash,
            as_of=request.as_of,
            max_sensitivity=request.max_sensitivity,
            limit=request.limit,
            per_namespace_limit=request.per_namespace_limit,
            generation=generation,
            policy_version=policy_version,
            content_digest=content_digest,
        )


class RetrievalCacheEntry(ContractModel):
    key: RetrievalCacheKey
    record_ids: tuple[NonEmptyStr, ...]
    expires_at: datetime | None = None


class RetrievalCache:
    def __init__(self) -> None:
        self._entries: dict[str, RetrievalCacheEntry] = {}

    def put(self, entry: RetrievalCacheEntry) -> None:
        self._entries[entry.key.digest()] = entry

    def get(
        self, key: RetrievalCacheKey, *, now: datetime | None = None
    ) -> RetrievalCacheEntry | None:
        entry = self._entries.get(key.digest())
        if entry is not None and now is not None and entry.expires_at is not None:
            if entry.expires_at <= now:
                return None
        return entry

    def invalidate_scope(self, scope: ScopeKey) -> None:
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.key.scope.organization_id != scope.organization_id
        }

    def invalidate_authority(
        self,
        scope: ScopeKey,
        *,
        access_hash: str,
        membership_hash: str,
    ) -> None:
        """Drop entries made under superseded access or membership snapshots."""

        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.key.scope.organization_id != scope.organization_id
            or (
                entry.key.access_hash == access_hash
                and entry.key.membership_hash == membership_hash
            )
        }

    def invalidate_generation(self, scope: ScopeKey, *, current_generation: int) -> None:
        self._entries = {
            key: entry
            for key, entry in self._entries.items()
            if entry.key.scope.organization_id != scope.organization_id
            or entry.key.generation >= current_generation
        }
