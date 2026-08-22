"""Deterministic whole-request context budgeting with pinned-segment safety."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from leo.harness.models import ContractModel, NonEmptyStr


class TokenEstimator(Protocol):
    """Provider-neutral deterministic token estimator used before model invocation."""

    @property
    def version(self) -> str: ...

    def estimate_tokens(self, text: str) -> int: ...


class Utf8TokenEstimator:
    """Stable dependency-free estimate: one token per four UTF-8 bytes, rounded up."""

    @property
    def version(self) -> str:
        return "utf8-bytes-div4-v1"

    def estimate_tokens(self, text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


class BudgetDecisionReason(StrEnum):
    INCLUDED_PINNED = "included_pinned"
    INCLUDED_PRIORITY = "included_priority"
    EXCLUDED_TOKEN_BUDGET = "excluded_token_budget"
    EXCLUDED_BYTE_BUDGET = "excluded_byte_budget"
    EXCLUDED_TOKEN_AND_BYTE_BUDGET = "excluded_token_and_byte_budget"


class BudgetSegment(ContractModel):
    name: NonEmptyStr
    source_type: NonEmptyStr = "generic"
    content_version: NonEmptyStr = "v1"
    text: NonEmptyStr
    priority: int = Field(ge=0, le=100)
    pinned: bool = False
    source_ids: tuple[NonEmptyStr, ...] = ()


class ContextBudget(ContractModel):
    max_tokens: int = Field(ge=1)
    max_bytes: int = Field(ge=1)


class SegmentBudgetDecision(ContractModel):
    name: NonEmptyStr
    estimated_tokens: int = Field(ge=1)
    estimated_bytes: int = Field(ge=1)
    included: bool
    reason: BudgetDecisionReason


class BudgetedContext(ContractModel):
    segments: tuple[BudgetSegment, ...]
    decisions: tuple[SegmentBudgetDecision, ...]
    estimator_version: NonEmptyStr
    estimated_tokens: int = Field(ge=0)
    estimated_bytes: int = Field(ge=0)
    candidate_estimated_tokens: int = Field(ge=0)
    candidate_estimated_bytes: int = Field(ge=0)
    manifest_digest: NonEmptyStr
    evicted_names: tuple[NonEmptyStr, ...] = ()


class ContextBudgetError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def assemble_budgeted_context(
    segments: tuple[BudgetSegment, ...],
    budget: ContextBudget,
    *,
    estimator: TokenEstimator | None = None,
) -> BudgetedContext:
    """Select a deterministic subset while retaining an auditable decision per candidate."""

    if len({segment.name for segment in segments}) != len(segments):
        raise ContextBudgetError("context_duplicate_segment")
    token_estimator = estimator or Utf8TokenEstimator()
    estimates = _estimate_segments(segments, token_estimator)
    if any(
        byte_count > budget.max_bytes
        for segment, (_, byte_count) in zip(segments, estimates, strict=True)
        if segment.pinned
    ):
        raise ContextBudgetError("pinned_segment_too_large")

    pinned_tokens = sum(
        token_count
        for segment, (token_count, _) in zip(segments, estimates, strict=True)
        if segment.pinned
    )
    pinned_bytes = sum(
        byte_count
        for segment, (_, byte_count) in zip(segments, estimates, strict=True)
        if segment.pinned
    )
    if pinned_tokens > budget.max_tokens or pinned_bytes > budget.max_bytes:
        raise ContextBudgetError("pinned_context_exceeds_budget")

    selected_indexes = list(range(len(segments)))
    exclusion_reasons: dict[int, BudgetDecisionReason] = {}

    def selected_totals() -> tuple[int, int]:
        return (
            sum(estimates[index][0] for index in selected_indexes),
            sum(estimates[index][1] for index in selected_indexes),
        )

    while True:
        selected_tokens, selected_bytes = selected_totals()
        token_overflow = selected_tokens > budget.max_tokens
        byte_overflow = selected_bytes > budget.max_bytes
        if not token_overflow and not byte_overflow:
            break
        candidates = [index for index in selected_indexes if not segments[index].pinned]
        if not candidates:
            raise ContextBudgetError("pinned_context_exceeds_budget")
        victim = min(
            candidates,
            key=lambda index: (segments[index].priority, index, segments[index].name),
        )
        exclusion_reasons[victim] = _exclusion_reason(
            token_overflow=token_overflow,
            byte_overflow=byte_overflow,
        )
        selected_indexes.remove(victim)

    selected_index_set = frozenset(selected_indexes)
    selected = tuple(
        segment for index, segment in enumerate(segments) if index in selected_index_set
    )
    decisions = tuple(
        SegmentBudgetDecision(
            name=segment.name,
            estimated_tokens=estimates[index][0],
            estimated_bytes=estimates[index][1],
            included=index in selected_index_set,
            reason=(
                BudgetDecisionReason.INCLUDED_PINNED
                if segment.pinned
                else BudgetDecisionReason.INCLUDED_PRIORITY
                if index in selected_index_set
                else exclusion_reasons[index]
            ),
        )
        for index, segment in enumerate(segments)
    )
    selected_tokens, selected_bytes = selected_totals()
    encoded = json.dumps(
        {
            "budget": budget.model_dump(mode="json"),
            "decisions": [item.model_dump(mode="json") for item in decisions],
            "estimator_version": token_estimator.version,
            "segments": [
                {
                    "content_hash": hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                    "content_version": item.content_version,
                    "name": item.name,
                    "pinned": item.pinned,
                    "priority": item.priority,
                    "source_ids": item.source_ids,
                    "source_type": item.source_type,
                }
                for item in segments
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return BudgetedContext(
        segments=selected,
        decisions=decisions,
        estimator_version=token_estimator.version,
        estimated_tokens=selected_tokens,
        estimated_bytes=selected_bytes,
        candidate_estimated_tokens=sum(item[0] for item in estimates),
        candidate_estimated_bytes=sum(item[1] for item in estimates),
        manifest_digest=hashlib.sha256(encoded.encode()).hexdigest(),
        evicted_names=tuple(
            segment.name
            for index, segment in enumerate(segments)
            if index not in selected_index_set
        ),
    )


def _exclusion_reason(*, token_overflow: bool, byte_overflow: bool) -> BudgetDecisionReason:
    if token_overflow and byte_overflow:
        return BudgetDecisionReason.EXCLUDED_TOKEN_AND_BYTE_BUDGET
    if token_overflow:
        return BudgetDecisionReason.EXCLUDED_TOKEN_BUDGET
    return BudgetDecisionReason.EXCLUDED_BYTE_BUDGET


def _estimate_segments(
    segments: tuple[BudgetSegment, ...],
    estimator: TokenEstimator,
) -> tuple[tuple[int, int], ...]:
    try:
        if not estimator.version.strip():
            raise ValueError("estimator version must be non-empty")
        estimates = tuple(
            (
                estimator.estimate_tokens(segment.text),
                len(segment.text.encode("utf-8")),
            )
            for segment in segments
        )
        if any(
            isinstance(token_count, bool) or not isinstance(token_count, int) or token_count < 1
            for token_count, _ in estimates
        ):
            raise ValueError("estimator returned an invalid token count")
    except Exception as exc:
        raise ContextBudgetError("context_token_estimator_failed") from exc
    return estimates
