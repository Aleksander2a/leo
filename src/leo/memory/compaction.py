"""Versioned source-linked thread summary proposals and deterministic checks."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.planes import SanitizedMessage


class SummaryProposal(ContractModel):
    objective: NonEmptyStr
    corrections: tuple[NonEmptyStr, ...] = ()
    decisions: tuple[NonEmptyStr, ...] = ()
    commitments: tuple[NonEmptyStr, ...] = ()
    unresolved_questions: tuple[NonEmptyStr, ...] = ()
    evidence_ids: tuple[NonEmptyStr, ...] = ()
    covered_message_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class SummaryRevision(ContractModel):
    thread_id: NonEmptyStr
    scope: ScopeKey
    version: int = Field(ge=1)
    proposal: SummaryProposal
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    digest: NonEmptyStr


class CompactionPolicy(ContractModel):
    trigger_messages: int = Field(default=50, ge=20, le=500)
    recent_window_messages: int = Field(default=12, ge=4, le=100)

    @model_validator(mode="after")
    def validate_window(self) -> CompactionPolicy:
        if self.recent_window_messages >= self.trigger_messages:
            raise ValueError("recent compaction window must be smaller than the trigger")
        return self


class CompactionWindow(ContractModel):
    should_compact: bool
    compactable_message_ids: tuple[NonEmptyStr, ...] = ()
    recent_message_ids: tuple[NonEmptyStr, ...]
    input_estimated_tokens: int = Field(ge=0)


class CompactionResult(ContractModel):
    summary: SummaryRevision
    recent_message_ids: tuple[NonEmptyStr, ...]
    input_estimated_tokens: int = Field(ge=1)
    summary_estimated_tokens: int = Field(ge=1)
    retained_estimated_tokens: int = Field(ge=1)
    token_reduction_ratio: float = Field(ge=0, le=1)


def validate_summary(proposal: SummaryProposal, *, available_source_ids: frozenset[str]) -> None:
    if not set(proposal.evidence_ids).issubset(available_source_ids):
        raise ValueError("summary cites unavailable evidence")
    if not proposal.objective.strip():
        raise ValueError("summary objective is required")


def make_summary(
    thread_id: str,
    scope: ScopeKey,
    version: int,
    proposal: SummaryProposal,
    *,
    available_source_ids: frozenset[str],
    previous: SummaryRevision | None = None,
) -> SummaryRevision:
    validate_summary(proposal, available_source_ids=available_source_ids)
    if not set(proposal.covered_message_ids).issubset(available_source_ids):
        raise ValueError("summary covers an unavailable message")
    if previous is not None:
        if previous.thread_id != thread_id or previous.scope != scope:
            raise ValueError("previous summary is outside the thread scope")
        if version != previous.version + 1:
            raise ValueError("summary version must append exactly one revision")
        _require_prior_facts(previous.proposal, proposal)
    elif version != 1:
        raise ValueError("initial summary version must be one")
    source_ids = tuple(dict.fromkeys((*proposal.covered_message_ids, *proposal.evidence_ids)))
    payload = {
        "thread_id": thread_id,
        "scope": scope.model_dump(mode="json"),
        "version": version,
        "proposal": proposal.model_dump(mode="json"),
        "source_ids": source_ids,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SummaryRevision(
        thread_id=thread_id,
        scope=scope,
        version=version,
        proposal=proposal,
        source_ids=source_ids,
        digest=digest,
    )


def render_summary_content(summary: SummaryRevision) -> str:
    """Render the structured proposal canonically for durable derived storage."""

    return json.dumps(
        summary.proposal.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def select_compaction_window(
    messages: tuple[SanitizedMessage, ...],
    policy: CompactionPolicy,
) -> CompactionWindow:
    """Choose an immutable prefix to summarize while retaining a recent source window."""

    ordered = tuple(sorted(messages, key=lambda item: (item.recorded_at, item.id)))
    if len({message.id for message in ordered}) != len(ordered):
        raise ValueError("compaction messages contain duplicate IDs")
    total_tokens = sum(_estimate_tokens(message.text) for message in ordered)
    if len(ordered) < policy.trigger_messages:
        return CompactionWindow(
            should_compact=False,
            recent_message_ids=tuple(message.id for message in ordered),
            input_estimated_tokens=total_tokens,
        )
    split = len(ordered) - policy.recent_window_messages
    return CompactionWindow(
        should_compact=True,
        compactable_message_ids=tuple(message.id for message in ordered[:split]),
        recent_message_ids=tuple(message.id for message in ordered[split:]),
        input_estimated_tokens=total_tokens,
    )


def compaction_result(
    summary: SummaryRevision,
    messages: tuple[SanitizedMessage, ...],
    window: CompactionWindow,
) -> CompactionResult:
    if not window.should_compact or not window.compactable_message_ids:
        raise ValueError("compaction result requires a triggered window")
    if not set(window.compactable_message_ids).issubset(summary.proposal.covered_message_ids):
        raise ValueError("summary did not cover the entire compactable prefix")
    by_id = {message.id: message for message in messages}
    if not set(window.recent_message_ids).issubset(by_id):
        raise ValueError("recent compaction window references unavailable messages")
    summary_tokens = _estimate_tokens(render_summary_content(summary))
    recent_tokens = sum(_estimate_tokens(by_id[item].text) for item in window.recent_message_ids)
    retained = summary_tokens + recent_tokens
    ratio = max(0.0, 1 - (retained / max(1, window.input_estimated_tokens)))
    return CompactionResult(
        summary=summary,
        recent_message_ids=window.recent_message_ids,
        input_estimated_tokens=window.input_estimated_tokens,
        summary_estimated_tokens=summary_tokens,
        retained_estimated_tokens=retained,
        token_reduction_ratio=min(1.0, ratio),
    )


def _require_prior_facts(previous: SummaryProposal, current: SummaryProposal) -> None:
    if not set(previous.covered_message_ids).issubset(current.covered_message_ids):
        raise ValueError("incremental summary dropped covered source messages")
    for field in (
        "corrections",
        "decisions",
        "commitments",
        "unresolved_questions",
        "evidence_ids",
    ):
        if not set(getattr(previous, field)).issubset(getattr(current, field)):
            raise ValueError(f"incremental summary dropped prior {field}")


def _estimate_tokens(value: str) -> int:
    return max(1, (len(value.encode("utf-8")) + 3) // 4)
