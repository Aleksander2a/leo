"""Provider-neutral Exa highlight normalization and canonical evidence helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re

from pydantic import JsonValue

from leo.url_policy import is_public_https_url

MAX_EXA_HIGHLIGHTS = 3


def normalize_complete_exa_result(value: object) -> dict[str, JsonValue] | None:
    """Return one bounded result when its URL/highlight evidence is complete.

    Exa has shipped otherwise useful highlight results without the optional title
    and/or highlight-score metadata.  Preserve that distinction explicitly instead
    of discarding the URL-bound text.  The normalized ``None`` values and ordered
    ``missing_fields`` marker make the representation stable when it is normalized
    again by the verifier.
    """

    if not isinstance(value, dict):
        return None
    raw_title = value.get("title")
    url = value.get("url")
    highlights = value.get("highlights")
    scores = value.get("highlightScores", value.get("highlight_scores"))
    if not (
        (raw_title is None or isinstance(raw_title, str))
        and isinstance(url, str)
        and is_public_https_url(url)
        and len(url) <= 2_048
        and isinstance(highlights, list)
        and highlights
    ):
        return None
    clean_title = _clean_text(raw_title, limit=240) if isinstance(raw_title, str) else ""
    missing_fields: list[JsonValue] = []
    normalized_title: JsonValue = clean_title or None
    if normalized_title is None:
        missing_fields.append("title")

    scores_missing = scores is None
    if not scores_missing and not (isinstance(scores, list) and len(highlights) == len(scores)):
        return None
    clean_highlights: list[JsonValue] = []
    clean_scores: list[JsonValue] = []
    for index, highlight in enumerate(highlights):
        if not isinstance(highlight, str):
            return None
        clean_highlight = _clean_text(highlight, limit=1_200)
        if not clean_highlight:
            return None
        clean_highlights.append(clean_highlight)
        if not scores_missing:
            assert isinstance(scores, list)
            score = scores[index]
            if not (
                isinstance(score, int | float)
                and not isinstance(score, bool)
                and math.isfinite(float(score))
            ):
                return None
            clean_scores.append(float(score))
        if len(clean_highlights) == MAX_EXA_HIGHLIGHTS:
            break
    if not clean_highlights:
        return None
    if scores_missing:
        missing_fields.append("highlight_scores")
    normalized: dict[str, JsonValue] = {
        "title": normalized_title,
        "url": url,
        "highlights": clean_highlights,
        "highlight_scores": None if scores_missing else clean_scores,
    }
    if missing_fields:
        normalized["missing_fields"] = missing_fields
    return normalized


def canonical_exa_highlight_statements(data: dict[str, JsonValue]) -> tuple[str, ...] | None:
    """Rebuild the only source-claim statements admitted from one Exa result."""

    result = data.get("result")
    if not isinstance(result, dict):
        return None
    normalized = normalize_complete_exa_result(result)
    if normalized is None or normalized != result:
        return None
    title = normalized.get("title")
    url = normalized.get("url")
    highlights = normalized.get("highlights")
    if not (
        (title is None or isinstance(title, str))
        and isinstance(url, str)
        and isinstance(highlights, list)
    ):
        return None
    statements = tuple(
        canonical_exa_highlight_statement(title=title, url=url, highlight=highlight)
        for highlight in highlights
        if isinstance(highlight, str)
    )
    return statements or None


def canonical_exa_highlight_statement(*, title: str | None, url: str, highlight: str) -> str:
    display_title = title or "Untitled Exa result"
    return f'Exa highlight from "{display_title}" ({url}): {highlight}'


def exa_result_hash(data: dict[str, JsonValue]) -> str | None:
    result = data.get("result")
    if not isinstance(result, dict):
        return None
    return hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _clean_text(value: str, *, limit: int) -> str:
    without_controls = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    # Keep the normalized representation idempotent when the byte/character
    # bound cuts immediately after a whitespace separator.
    return " ".join(without_controls.split())[:limit].rstrip()


__all__ = [
    "MAX_EXA_HIGHLIGHTS",
    "canonical_exa_highlight_statement",
    "canonical_exa_highlight_statements",
    "exa_result_hash",
    "normalize_complete_exa_result",
]
