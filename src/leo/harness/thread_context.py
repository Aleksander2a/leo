"""Deterministic protection and compaction for authorized conversation threads."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass

from leo.harness.context_budget import (
    BudgetedContext,
    BudgetSegment,
    ContextBudget,
    ContextBudgetError,
    assemble_budgeted_context,
)
from leo.harness.models import ContextItem, ContextItemKind, ContextItemRetention

_CORRECTION_PATTERN = re.compile(
    r"\b(?:correction|corrected?|actually|instead|supersed(?:e|ed|es)|"
    r"replace(?:d|s)?\s+(?:that|the\s+prior)|not\s+.+?\s+but)\b",
    re.IGNORECASE,
)
_DECISION_PATTERN = re.compile(
    r"\b(?:decid(?:e|ed|ing)|decision|agreed?|agreement|committed?|commitment|"
    r"approved?|selected?|chosen|chose|resolved?)\b",
    re.IGNORECASE,
)
_PROGRESS_PATTERN = re.compile(
    r"(?:\b(?:working|processing|researching|starting|queued|still\s+working|"
    r"one\s+moment|hang\s+tight|in\s+progress|status\s+update)\b|^[.·•…\s]+$)",
    re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.-]{3,64}")
_STOP_WORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "been",
        "before",
        "from",
        "have",
        "into",
        "just",
        "message",
        "slack",
        "that",
        "their",
        "then",
        "there",
        "they",
        "this",
        "thread",
        "turn",
        "were",
        "what",
        "when",
        "where",
        "which",
        "with",
        "would",
    }
)


@dataclass(frozen=True, slots=True)
class ThreadContextSelection:
    """One budgeted selection plus the exact supporting turns it compacted."""

    items: tuple[ContextItem, ...]
    budgeted: BudgetedContext
    compacted_item_ids: tuple[str, ...]
    compaction_digest: str | None
    reopen_ranges: tuple[ThreadContextRange, ...]


@dataclass(frozen=True, slots=True)
class ThreadContextRange:
    """Opaque-handle source range retained server-side for progressive reopening."""

    handle: str
    digest: str
    items: tuple[ContextItem, ...]


@dataclass(frozen=True, slots=True)
class ThreadTurnRetentionInput:
    """Minimal trusted transcript metadata used for chronological retention."""

    content: str
    actor_id: str
    speaker_role: str
    is_root: bool
    is_recent: bool


def classify_thread_turn(
    content: str,
    *,
    is_root: bool,
    is_recent: bool,
    is_prior_outcome: bool = False,
) -> tuple[ContextItemRetention, int]:
    """Return a trusted retention class derived from transcript position and text."""

    if is_root:
        return ContextItemRetention.THREAD_ROOT, 100
    if _CORRECTION_PATTERN.search(content):
        return ContextItemRetention.CORRECTION, 100
    if _DECISION_PATTERN.search(content):
        return ContextItemRetention.DECISION, 99
    if is_prior_outcome:
        return ContextItemRetention.PRIOR_OUTCOME, 99
    if "?" in content:
        return ContextItemRetention.UNRESOLVED_QUESTION, 98
    if is_recent:
        return ContextItemRetention.RECENT, 97
    return ContextItemRetention.SUPPORTING, 76


def classify_thread_transcript(
    turns: tuple[ThreadTurnRetentionInput, ...],
) -> tuple[tuple[ContextItemRetention, int], ...]:
    """Protect unresolved questions and the last material assistant outcome.

    A later substantive opposite-role turn resolves an older question. Generic bot
    progress updates are supporting context, while the latest material assistant
    outcome remains exact. Decisions, corrections, the root, and recent turns keep
    their stronger independent protection.
    """

    material = tuple(_is_substantive(turn.content) for turn in turns)
    unresolved_questions: set[int] = set()
    for index, turn in enumerate(turns):
        if "?" not in turn.content:
            continue
        resolved = any(
            material[later]
            and turns[later].speaker_role != turn.speaker_role
            and not _PROGRESS_PATTERN.search(turns[later].content)
            and _question_answer_overlap(turn.content, turns[later].content)
            for later in range(index + 1, len(turns))
        )
        if not resolved:
            unresolved_questions.add(index)
    material_assistant = tuple(
        index
        for index, turn in enumerate(turns)
        if turn.speaker_role == "assistant"
        and material[index]
        and not _PROGRESS_PATTERN.search(turn.content)
        and index not in unresolved_questions
    )
    last_material_assistant = material_assistant[-1] if material_assistant else None

    classifications: list[tuple[ContextItemRetention, int]] = []
    for index, turn in enumerate(turns):
        if turn.is_root:
            classification = (ContextItemRetention.THREAD_ROOT, 100)
        elif _CORRECTION_PATTERN.search(turn.content):
            classification = (ContextItemRetention.CORRECTION, 100)
        elif _DECISION_PATTERN.search(turn.content):
            classification = (ContextItemRetention.DECISION, 99)
        elif index in unresolved_questions:
            classification = (ContextItemRetention.UNRESOLVED_QUESTION, 98)
        elif index == last_material_assistant:
            classification = (ContextItemRetention.PRIOR_OUTCOME, 99)
        elif turn.is_recent:
            classification = (ContextItemRetention.RECENT, 97)
        else:
            classification = (ContextItemRetention.SUPPORTING, 76)
        classifications.append(classification)
    return tuple(classifications)


def select_context_with_thread_compaction(
    items: tuple[ContextItem, ...],
    *,
    thread_item_ids: frozenset[str],
    conversation_id: str,
    summary_id_namespace: str,
    max_tokens: int,
    max_bytes: int | None = None,
    summary_max_bytes: int = 8_192,
) -> ThreadContextSelection:
    """Budget context without ever silently evicting an authorized thread turn.

    Protected thread turns are pinned.  Any supporting thread turns displaced by the
    whole-request budget are replaced by one deterministic, provenance-digested summary;
    ordinary background context may still be priority-evicted.
    """

    context_budget = ContextBudget(
        max_tokens=max_tokens,
        max_bytes=max_bytes if max_bytes is not None else max_tokens * 4,
    )
    by_id = {item.id: item for item in items}
    if len(by_id) != len(items):
        raise ValueError("context items must have unique IDs before thread compaction")
    thread_supporting_ids = frozenset(
        item.id for item in items if item.id in thread_item_ids and not item.retention.pinned
    )
    protected_bytes = sum(
        len(item.content.encode("utf-8")) for item in items if item.retention.pinned
    )
    protected_tokens = sum(
        max(1, (len(item.content.encode("utf-8")) + 3) // 4)
        for item in items
        if item.retention.pinned
    )
    summary_byte_budget = min(
        summary_max_bytes,
        context_budget.max_bytes - protected_bytes,
        max(0, (context_budget.max_tokens - protected_tokens) * 4),
    )
    compacted: set[str] = set()
    summary: ContextItem | None = None

    # The compacted set only grows, so this is deterministically bounded by the number
    # of supporting thread turns plus the initial pass.
    for _ in range(len(thread_supporting_ids) + 2):
        candidates = tuple(item for item in items if item.id not in compacted)
        if summary is not None:
            candidates = _insert_after_root(candidates, summary)
        budgeted = assemble_budgeted_context(
            tuple(_budget_segment(item) for item in candidates),
            context_budget,
        )
        newly_compacted = thread_supporting_ids.intersection(budgeted.evicted_names)
        if not newly_compacted:
            selected_names = {segment.name for segment in budgeted.segments}
            selected = tuple(item for item in candidates if item.id in selected_names)
            return ThreadContextSelection(
                items=selected,
                budgeted=budgeted,
                compacted_item_ids=tuple(item.id for item in items if item.id in compacted),
                compaction_digest=(
                    _source_digest(tuple(item for item in items if item.id in compacted))
                    if compacted
                    else None
                ),
                reopen_ranges=(
                    _reopen_ranges(
                        items,
                        compacted=frozenset(compacted),
                        summary_id_namespace=summary_id_namespace,
                    )
                    if compacted
                    else ()
                ),
            )
        compacted.update(newly_compacted)
        compacted_items = tuple(item for item in items if item.id in compacted)
        ranges = _reopen_ranges(
            items,
            compacted=frozenset(compacted),
            summary_id_namespace=summary_id_namespace,
        )
        if summary_byte_budget < 64:
            raise ContextBudgetError("pinned_context_exceeds_budget")
        summary = build_thread_compaction_item(
            compacted_items,
            conversation_id=conversation_id,
            summary_id_namespace=summary_id_namespace,
            max_bytes=summary_byte_budget,
            reopen_ranges=ranges,
        )

    raise RuntimeError("thread context compaction did not converge")


def build_thread_compaction_item(
    items: tuple[ContextItem, ...],
    *,
    conversation_id: str,
    summary_id_namespace: str,
    max_bytes: int,
    reopen_ranges: tuple[ThreadContextRange, ...],
) -> ContextItem:
    """Build a bounded extractive summary with an immutable full-source digest."""

    if not items:
        raise ValueError("thread compaction requires at least one source item")
    digest = _source_digest(items)
    header = (
        "[Deterministic Slack thread compaction; "
        f"source_count={len(items)}; source_digest=sha256:{digest}]\n"
        "Root, recent, decision, correction, unresolved-question, and final-outcome "
        "turns remain exact outside this summary.\n"
    )
    handles = "\n".join(
        (
            f"- handle={item.handle}; source_count={len(item.items)}; "
            f"source_digest=sha256:{item.digest}"
        )
        for item in reopen_ranges
    )
    terms = Counter(
        token.lower()
        for item in items
        for token in _TOKEN_PATTERN.findall(item.content)
        if token.lower() not in _STOP_WORDS
    )
    topic_terms = [
        token
        for token, count in sorted(terms.items(), key=lambda pair: (-pair[1], pair[0]))
        if count > 1
    ][:24]
    topic_line = "Recurring terms: " + ", ".join(topic_terms)
    prefix = f"{header}Reopen ranges:\n{handles}\n{topic_line}\nChronological extracts:\n"
    available = max_bytes - len(prefix.encode("utf-8"))
    if available < 64:
        raise ValueError("thread compaction byte budget is too small")
    per_item = max(12, min(320, available // len(items) - 8))
    lines = [f"- {_truncate_utf8(_flatten(item.content), per_item)}" for item in items]
    content = _truncate_utf8(prefix + "\n".join(lines), max_bytes)
    return ContextItem(
        id=f"{summary_id_namespace}:{digest[:24]}",
        kind=ContextItemKind.THREAD_SUMMARY,
        content=content,
        conversation_id=conversation_id,
        retention=ContextItemRetention.COMPACTION_SUMMARY,
        budget_priority=98,
    )


def thread_context_source_digest(items: tuple[ContextItem, ...]) -> str:
    """Canonical digest shared by compaction manifests and reopen services."""

    if not items:
        raise ValueError("thread context source digest requires at least one item")
    return _source_digest(items)


def _budget_segment(item: ContextItem) -> BudgetSegment:
    return BudgetSegment(
        name=item.id,
        source_type=f"context_item:{item.kind.value}",
        content_version=item.kind.value,
        text=item.content,
        priority=item.budget_priority if item.budget_priority is not None else 70,
        pinned=item.retention.pinned,
        source_ids=(item.id, item.conversation_id),
    )


def _reopen_ranges(
    items: tuple[ContextItem, ...],
    *,
    compacted: frozenset[str],
    summary_id_namespace: str,
) -> tuple[ThreadContextRange, ...]:
    source_items = tuple(item for item in items if item.id in compacted)
    if not source_items:
        return ()
    digest = _source_digest(source_items)
    material = f"{summary_id_namespace}\x1f{digest}"
    handle = f"thr_{hashlib.sha256(material.encode()).hexdigest()[:32]}"
    return (ThreadContextRange(handle=handle, digest=digest, items=source_items),)


def _insert_after_root(
    items: tuple[ContextItem, ...],
    summary: ContextItem,
) -> tuple[ContextItem, ...]:
    root_index = next(
        (
            index
            for index, item in enumerate(items)
            if item.retention is ContextItemRetention.THREAD_ROOT
        ),
        -1,
    )
    insertion = root_index + 1
    return (*items[:insertion], summary, *items[insertion:])


def _source_digest(items: tuple[ContextItem, ...]) -> str:
    encoded = json.dumps(
        [
            {
                "content": item.content,
                "conversation_id": item.conversation_id,
                "id": item.id,
            }
            for item in items
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _flatten(value: str) -> str:
    return " ".join(value.split())


def _is_substantive(value: str) -> bool:
    return len(_TOKEN_PATTERN.findall(value)) >= 3


def _question_answer_overlap(question: str, answer: str) -> bool:
    question_terms = {
        token.lower()
        for token in _TOKEN_PATTERN.findall(question)
        if token.lower() not in _STOP_WORDS
    }
    answer_terms = {
        token.lower()
        for token in _TOKEN_PATTERN.findall(answer)
        if token.lower() not in _STOP_WORDS
    }
    return bool(question_terms.intersection(answer_terms)) or bool(
        re.search(r"\b(?:yes|no|answer|because|result)\b", answer, re.IGNORECASE)
    )


def _truncate_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    if max_bytes <= 3:
        return encoded[:max_bytes].decode("utf-8", errors="ignore")
    return encoded[: max_bytes - 3].decode("utf-8", errors="ignore").rstrip() + "..."
