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
    r"i(?:'ll| will)|let me|i (?:need|have) to|i(?:'m| am) going to|"
    r"i(?:'m| am) about to"
    r")\s+(?:(?:just|first|now|quickly|briefly|next)\s+){0,3}"
    r"(?:pull|grab|check|search|look up|research|verify|browse|investigate|gather|"
    r"fetch|retrieve|run|query)\b"
    # Present progressive is the same broken promise in a different tense.
    # "I'm pulling those now" reads as work in flight, but the turn is about to
    # end -- nothing is in flight, and the user is left waiting for a follow-up
    # that never arrives. This shape reached Slack twice in a row before it was
    # caught here, so it is treated exactly like "I'll pull those".
    r"|\bi(?:'m| am)\s+(?:(?:just|now|currently|quickly|briefly)\s+){0,3}"
    r"(?:pulling|grabbing|checking|searching|researching|verifying|"
    r"browsing|investigating|gathering|fetching|retrieving|running|querying"
    # "look up" separates around its object: "looking that up", "looking it up".
    r"|looking(?:\s+\w+){0,3}\s+up)\b"
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
_RETRIEVAL_VERB = r"(?:pulled|grabbed|fetched|retrieved|checked|looked up|verified|reviewed)"
# Nouns that only make sense as the object of a real external read. "rates",
# "yields", and "quotes" belong here: a model asserting it pulled *those*
# without an observation is claiming live data it never fetched.
_RETRIEVAL_OBJECT = (
    r"(?:current data|live data|live rates|rates|yields?|apys?|quotes?|prices?|"
    r"sources?|filings?|news|financials?(?!\s+advice\b)|web results?|"
    r"documentation|evidence|reports?|company data|market data)"
)
_COMPLETED_EXTERNAL_RESEARCH_ACTION = re.compile(
    # Unconditional research verbs need no object.
    r"\bi(?:'ve| have)?\s+(?:researched|browsed|searched|looked up|investigated)\b|"
    # Verb then object: "I pulled the current data".
    rf"\bi(?:'ve| have)?\s+{_RETRIEVAL_VERB}\b[^.!?\r\n]{{0,80}}\b{_RETRIEVAL_OBJECT}\b|"
    # Object then verb: "the live rates I pulled", "based on sources I checked".
    # Same claim, reversed clause order -- which previously slipped through and
    # let an answer assert it had fetched data while calling no tool at all.
    rf"\b{_RETRIEVAL_OBJECT}\b[^.!?\r\n]{{0,40}}\bi(?:'ve| have)?\s+{_RETRIEVAL_VERB}\b"
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
