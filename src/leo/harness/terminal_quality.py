"""Bounded terminal-answer quality predicates shared by policy and verification."""

from __future__ import annotations

import re

_PROTECTED_TEXT = re.compile(
    r"```[\s\S]*?```|`[^`\r\n]*`|"
    r'"(?:\\.|[^"\\\r\n])*"|'
    r"(?<!\w)'(?:[^'\r\n]|(?<=\w)'(?=\w))*'(?!\w)|"
    r"\u201c[^\u201d\r\n]*\u201d|\u2018[^\u2019\r\n]*\u2019"
)
_MARKDOWN_QUOTE_LINE = re.compile(r"(?m)^\s*>[^\r\n]*(?:\r?\n|$)")
_FUTURE_ACTION = re.compile(
    r"\b(?:"
    r"i(?:'ll| will)|let me|i (?:need|have) to|i(?:'m| am) going to"
    r")\s+(?:(?:just|first|now|quickly|briefly|next)\s+){0,3}"
    r"(?:pull|grab|check|search|look up|research|verify|browse|investigate|consult|gather|"
    r"find|fetch|retrieve|open|review|use|take a look at)\b"
)
_THEN_I_CAN = re.compile(r"\bthen\s+i\s+can\s+[a-z][a-z'-]*\b")
_COMPLETED_MARKET_DATA_ACTION = re.compile(
    r"\bi(?:'ve| have)?\s+(?:pulled|grabbed|fetched|retrieved|checked|looked up|"
    r"researched|verified|reviewed)\b[^.!?\r\n]{0,80}\b(?:quotes?|prices?|market data)\b"
)
_COMPLETED_FILING_ACTION = re.compile(
    r"\bi(?:'ve| have)?\s+(?:pulled|grabbed|fetched|retrieved|checked|looked up|"
    r"researched|verified|reviewed)\b[^.!?\r\n]{0,80}\bfilings?\b"
)
_COMPLETED_EXTERNAL_RESEARCH_ACTION = re.compile(
    r"\bi(?:'ve| have)?\s+(?:researched|browsed|searched|looked up|investigated)\b|"
    r"\bi(?:'ve| have)?\s+(?:pulled|grabbed|fetched|retrieved|checked|verified|reviewed)\b"
    r"[^.!?\r\n]{0,80}\b(?:current data|live data|sources?|filings?|news|financials?|"
    r"web results?|documentation|evidence|reports?|company data|market data)\b"
)


def contains_future_action_promise(answer: str) -> bool:
    """Return whether unquoted assistant prose promises a later research action.

    Quoted examples, Markdown blockquotes, and code are removed first so explanatory
    discussion of a bad phrase does not itself fail the terminal-answer gate. The
    remaining predicates require first-person future action; capability descriptions
    such as ``I can check this with Tavily`` remain valid.
    """

    normalized = _normalized_unquoted_prose(answer)
    return (
        _FUTURE_ACTION.search(normalized) is not None or _THEN_I_CAN.search(normalized) is not None
    )


def completed_research_action_claim(answer: str) -> str | None:
    """Classify an unquoted first-person claim that an external read already happened."""

    normalized = _normalized_unquoted_prose(answer)
    if _COMPLETED_MARKET_DATA_ACTION.search(normalized) is not None:
        return "market_quote"
    if _COMPLETED_FILING_ACTION.search(normalized) is not None:
        return "filing"
    if _COMPLETED_EXTERNAL_RESEARCH_ACTION.search(normalized) is not None:
        return "external_read"
    return None


def _normalized_unquoted_prose(answer: str) -> str:
    unquoted = _MARKDOWN_QUOTE_LINE.sub(" ", answer)
    unquoted = _PROTECTED_TEXT.sub(" ", unquoted)
    return " ".join(unquoted.casefold().replace("\u2019", "'").replace("\u2018", "'").split())
