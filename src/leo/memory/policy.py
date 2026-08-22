"""Conservative candidate promotion; semantic text never becomes commit authority."""

from __future__ import annotations

from enum import StrEnum

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.models import MemoryRevision
from leo.memory.service import MemoryCandidate


class PromotionStatus(StrEnum):
    CONFIRMATION_REQUIRED = "confirmation_required"
    PROMOTE = "promote"
    DUPLICATE = "duplicate"
    CONTESTED = "contested"
    REJECT = "reject"


class PromotionDecision(ContractModel):
    status: PromotionStatus
    reason: NonEmptyStr
    matched_record_id: str | None = None
    contradiction_record_ids: tuple[NonEmptyStr, ...] = ()


def assess_candidate(
    scope: ScopeKey,
    candidate: MemoryCandidate,
    current: tuple[tuple[str, MemoryRevision], ...],
    *,
    confirmed: bool,
) -> PromotionDecision:
    del scope
    if not confirmed:
        return PromotionDecision(
            status=PromotionStatus.CONFIRMATION_REQUIRED,
            reason="explicit confirmation is required before memory promotion",
        )
    exact = tuple(
        (record_id, revision)
        for record_id, revision in current
        if revision.content_hash
        == MemoryRevision.from_content(
            id="hash-check",
            record_id=record_id,
            number=revision.number,
            content=candidate.content,
            source_ids=revision.source_ids,
            visibility=revision.visibility,
            namespace_id=revision.namespace_id,
            sensitivity=candidate.sensitivity,
            valid_from=candidate.valid_from,
            recorded_at=revision.recorded_at,
            actor_id=revision.actor_id,
            reason=revision.reason,
        ).content_hash
    )
    if exact:
        return PromotionDecision(
            status=PromotionStatus.DUPLICATE,
            reason="candidate content already exists",
            matched_record_id=exact[0][0],
        )
    conflicts = tuple(
        record_id
        for record_id, revision in current
        if revision.visibility is candidate.visibility
        and revision.namespace_id == candidate.namespace_id
        and revision.status.value == "active"
        and revision.content != candidate.content
    )
    if conflicts:
        return PromotionDecision(
            status=PromotionStatus.CONTESTED,
            reason="candidate conflicts with an active memory in the same namespace",
            contradiction_record_ids=conflicts,
        )
    return PromotionDecision(
        status=PromotionStatus.PROMOTE,
        reason="candidate is eligible for explicit commit",
    )
