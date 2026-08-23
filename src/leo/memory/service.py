"""Harness-owned explicit memory commands; model candidates never carry trusted scope."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.harness.ports import Clock, IdGenerator
from leo.harness.store_errors import NotFoundError
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.ports import MemoryEmbeddingIndexer, MemoryStore

if TYPE_CHECKING:
    from leo.memory.policy import PromotionDecision

_logger = logging.getLogger(__name__)


class MemoryCandidate(ContractModel):
    """Untrusted semantic proposal; it contains no scope, actor, grant, or lifecycle authority."""

    kind: MemoryKind
    content: NonEmptyStr = Field(max_length=16_384)
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    sensitivity: float = Field(ge=0, le=1)
    valid_from: datetime
    valid_until: datetime | None = None
    expires_at: datetime | None = None
    reason: NonEmptyStr


class MemoryCommandRejected(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class ExplicitMemoryService:
    def __init__(
        self,
        store: MemoryStore,
        clock: Clock,
        ids: IdGenerator,
        *,
        embedding_indexer: MemoryEmbeddingIndexer | None = None,
    ) -> None:
        self._store = store
        self._clock = clock
        self._ids = ids
        self._embedding_indexer = embedding_indexer

    async def remember(
        self,
        scope: ScopeKey,
        candidate: MemoryCandidate,
        *,
        actor_id: str,
        sources: tuple[MemorySource, ...],
        confirmed: bool,
        source_type: Literal["explicit", "autonomous"] = "explicit",
    ) -> MemoryRecord:
        if not confirmed:
            raise MemoryCommandRejected("explicit_confirmation_required")
        _validate_sources(scope, candidate, sources)
        record_id = self._ids.new("memory")
        revision = MemoryRevision.from_content(
            id=self._ids.new("memory-revision"),
            record_id=record_id,
            number=1,
            content=candidate.content,
            source_ids=candidate.source_ids,
            visibility=candidate.visibility,
            namespace_id=candidate.namespace_id,
            sensitivity=candidate.sensitivity,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            recorded_at=self._clock.now(),
            expires_at=candidate.expires_at,
            actor_id=actor_id,
            reason=candidate.reason,
            source_type=source_type,
        )
        record = MemoryRecord(
            id=record_id,
            scope=scope,
            kind=candidate.kind,
            visibility=candidate.visibility,
            namespace_id=candidate.namespace_id,
            created_at=revision.recorded_at,
        )
        created = await self._store.create(record, revision, sources)
        await self._index_best_effort(scope, revision, source_type=source_type)
        return created

    async def correct(
        self,
        scope: ScopeKey,
        record_id: str,
        candidate: MemoryCandidate,
        *,
        actor_id: str,
        sources: tuple[MemorySource, ...],
        confirmed: bool,
        source_type: Literal["explicit", "autonomous"] = "explicit",
    ) -> MemoryRecord:
        if not confirmed:
            raise MemoryCommandRejected("explicit_confirmation_required")
        try:
            current = await self._store.current(scope, record_id)
        except NotFoundError as exc:
            raise MemoryCommandRejected("memory_not_current") from exc
        if current is None:
            raise MemoryCommandRejected("memory_not_current")
        if (
            candidate.visibility is not current.visibility
            or candidate.namespace_id != current.namespace_id
        ):
            raise MemoryCommandRejected("memory_visibility_immutable")
        _validate_sources(scope, candidate, sources)
        source_ids = tuple(dict.fromkeys((*current.source_ids, *candidate.source_ids)))
        revision = MemoryRevision.from_content(
            id=self._ids.new("memory-revision"),
            record_id=record_id,
            number=current.number + 1,
            content=candidate.content,
            source_ids=source_ids,
            visibility=current.visibility,
            namespace_id=current.namespace_id,
            sensitivity=candidate.sensitivity,
            valid_from=candidate.valid_from,
            valid_until=candidate.valid_until,
            recorded_at=self._clock.now(),
            expires_at=candidate.expires_at,
            actor_id=actor_id,
            reason=candidate.reason,
            supersedes_revision=current.number,
            source_type=source_type,
        )
        updated = await self._store.append_revision(
            scope,
            record_id,
            current.number,
            revision,
            sources,
        )
        await self._index_best_effort(scope, revision, source_type=source_type)
        return updated

    async def _index_best_effort(
        self, scope: ScopeKey, revision: MemoryRevision, *, source_type: str
    ) -> None:
        if self._embedding_indexer is None:
            return
        try:
            await self._embedding_indexer.index(scope, revision, source_type=source_type)
        except Exception:
            _logger.warning(
                "memory embedding index failed for revision %s; content remains "
                "retrievable through lexical search",
                revision.id,
                exc_info=True,
            )

    async def propose_autonomous(
        self,
        scope: ScopeKey,
        candidate: MemoryCandidate,
        *,
        actor_id: str,
        sources: tuple[MemorySource, ...],
    ) -> tuple[PromotionDecision, MemoryRecord | None]:
        """Commit a model-proposed candidate without an interactive confirmation turn.

        This is the autonomous counterpart to ``remember``: the harness still derives
        every scope/visibility/namespace/source value (the candidate never carries
        authority), but instead of requiring an explicit user command, it runs the
        candidate through duplicate/contradiction governance and commits automatically.
        A single contested conflict is treated as an update (a new revision superseding
        the conflicting record); a duplicate is a no-op; anything else is a new record.
        """

        from leo.memory.policy import PromotionStatus, assess_candidate

        _validate_sources(scope, candidate, sources)
        current = await self._store.list_active(
            scope,
            visibility=candidate.visibility,
            namespace_id=candidate.namespace_id,
            kind=candidate.kind,
        )
        decision = assess_candidate(scope, candidate, current, confirmed=True)
        if decision.status is PromotionStatus.DUPLICATE:
            return decision, None
        if (
            decision.status is PromotionStatus.CONTESTED
            and len(decision.contradiction_record_ids) == 1
        ):
            record = await self.correct(
                scope,
                decision.contradiction_record_ids[0],
                candidate,
                actor_id=actor_id,
                sources=sources,
                confirmed=True,
                source_type="autonomous",
            )
            return decision, record
        record = await self.remember(
            scope,
            candidate,
            actor_id=actor_id,
            sources=sources,
            confirmed=True,
            source_type="autonomous",
        )
        return decision, record

    async def forget(
        self,
        scope: ScopeKey,
        record_id: str,
        *,
        actor_id: str,
        visibility: MemoryVisibility,
        namespace_id: str,
        sources: tuple[MemorySource, ...],
        confirmed: bool,
        reason: str,
    ) -> MemoryRecord:
        if not confirmed:
            raise MemoryCommandRejected("explicit_confirmation_required")
        if not reason.strip():
            raise MemoryCommandRejected("forget_reason_required")
        try:
            current = await self._store.current(scope, record_id)
        except NotFoundError as exc:
            raise MemoryCommandRejected("memory_not_current") from exc
        if current is None:
            raise MemoryCommandRejected("memory_not_current")
        if current.visibility is not visibility or current.namespace_id != namespace_id:
            raise MemoryCommandRejected("memory_not_authorized_for_destination")
        source_ids = tuple(dict.fromkeys((*current.source_ids, *(source.id for source in sources))))
        candidate = MemoryCandidate(
            kind=MemoryKind.NOTE,
            content=current.content,
            source_ids=tuple(source.id for source in sources),
            visibility=visibility,
            namespace_id=namespace_id,
            sensitivity=current.sensitivity,
            valid_from=current.valid_from,
            valid_until=current.valid_until,
            expires_at=current.expires_at,
            reason=reason,
        )
        _validate_sources(scope, candidate, sources)
        revision = MemoryRevision.from_content(
            id=self._ids.new("memory-revision"),
            record_id=record_id,
            number=current.number + 1,
            content=current.content,
            source_ids=source_ids,
            visibility=current.visibility,
            namespace_id=current.namespace_id,
            sensitivity=current.sensitivity,
            valid_from=current.valid_from,
            valid_until=current.valid_until,
            recorded_at=self._clock.now(),
            expires_at=current.expires_at,
            actor_id=actor_id,
            reason=reason,
            status=MemoryStatus.RETRACTED,
            supersedes_revision=current.number,
        )
        return await self._store.append_revision(
            scope,
            record_id,
            current.number,
            revision,
            sources,
        )


def _validate_sources(
    scope: ScopeKey,
    candidate: MemoryCandidate,
    sources: tuple[MemorySource, ...],
) -> None:
    if len({source.id for source in sources}) != len(sources):
        raise MemoryCommandRejected("duplicate_source")
    if set(candidate.source_ids) != {source.id for source in sources}:
        raise MemoryCommandRejected("source_provenance_incomplete")
    if any(source.scope != scope for source in sources):
        raise MemoryCommandRejected("source_scope_mismatch")
    if any(
        source.visibility is not candidate.visibility
        or source.namespace_id != candidate.namespace_id
        for source in sources
    ):
        raise MemoryCommandRejected("source_visibility_mismatch")
