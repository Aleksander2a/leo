"""Turn a model answer into something Slack renders correctly.

Models write Markdown. Slack does not speak Markdown -- it speaks mrkdwn, where
bold is ``*one asterisk*``, links are ``<url|label>``, and headings do not
exist. Asking the model nicely gets it right most of the time, which is another
way of saying it is wrong often enough to look broken.

So the conversion happens here, structurally, on the way out. Code spans and
fenced blocks are lifted out first and put back untouched, because their
contents are not markup.
"""

from __future__ import annotations

import re

#: Slack rejects a text block over 3000 characters, and a message over 4000.
#: Chunking below the block limit keeps a long answer intact across posts.
MAX_BLOCK_CHARS = 2900

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]+`")
_BOLD = re.compile(r"\*\*(.+?)\*\*", re.DOTALL)
_BOLD_ALT = re.compile(r"__(.+?)__", re.DOTALL)
_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![\w*])")
_STRIKE = re.compile(r"~~(.+?)~~", re.DOTALL)
# Only horizontal whitespace around the text: `\s*$` in multiline mode would
# swallow the blank line after a heading and glue it to the next paragraph.
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$", re.MULTILINE)
_LINK = re.compile(r"\[([^\]\n]+)\]\((\S+?)\)")
_BULLET = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$", re.MULTILINE)
_BOLD_SENTINEL = "\x00B\x00"


def to_mrkdwn(text: str) -> str:
    """Convert Markdown to Slack mrkdwn, leaving code untouched."""

    if not text:
        return ""
    protected: list[str] = []

    def stash(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00C{len(protected) - 1}\x00"

    body = _FENCE.sub(stash, text)
    body = _INLINE_CODE.sub(stash, body)

    # Headings and bold both become mrkdwn bold, but marked with a sentinel so
    # the single-asterisk italic pass below cannot mistake them for italics.
    body = _HEADING.sub(lambda m: f"{_BOLD_SENTINEL}{m.group(1)}{_BOLD_SENTINEL}", body)
    body = _LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", body)
    body = _BOLD.sub(lambda m: f"{_BOLD_SENTINEL}{m.group(1)}{_BOLD_SENTINEL}", body)
    body = _BOLD_ALT.sub(lambda m: f"{_BOLD_SENTINEL}{m.group(1)}{_BOLD_SENTINEL}", body)
    body = _STRIKE.sub(lambda m: f"~{m.group(1)}~", body)
    body = _ITALIC.sub(lambda m: f"_{m.group(1)}_", body)
    body = body.replace(_BOLD_SENTINEL, "*")
    body = _BULLET.sub(lambda m: f"{m.group(1)}• ", body)
    body = _RULE.sub("", body)

    for index, original in enumerate(protected):
        body = body.replace(f"\x00C{index}\x00", original)
    return body.strip()


def chunks(text: str, limit: int = MAX_BLOCK_CHARS) -> list[str]:
    """Split a long answer at natural boundaries, never inside a code fence."""

    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    out: list[str] = []
    current = ""
    for block in _split_preserving_fences(text):
        if len(block) > limit:
            if current:
                out.append(current.strip())
                current = ""
            out.extend(_hard_split(block, limit))
            continue
        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) > limit:
            out.append(current.strip())
            current = block
        else:
            current = candidate
    if current.strip():
        out.append(current.strip())
    return [chunk for chunk in out if chunk]


def _split_preserving_fences(text: str) -> list[str]:
    parts: list[str] = []
    cursor = 0
    for match in _FENCE.finditer(text):
        before = text[cursor : match.start()]
        parts.extend(p for p in before.split("\n\n") if p.strip())
        parts.append(match.group(0))
        cursor = match.end()
    parts.extend(p for p in text[cursor:].split("\n\n") if p.strip())
    return parts


def _hard_split(block: str, limit: int) -> list[str]:
    """Split one oversize block, keeping each piece independently valid.

    A code fence longer than Slack's block limit has to be cut somewhere; each
    piece is re-fenced so no chunk arrives with an unterminated ``` and turns
    the rest of the message into code.
    """

    fenced = block.startswith("```")
    budget = limit - 8 if fenced else limit
    body = block.strip("`").lstrip("\n") if fenced else block

    out: list[str] = []
    remaining = body
    while len(remaining) > budget:
        cut = remaining.rfind("\n", 0, budget)
        if cut <= 0:
            cut = remaining.rfind(" ", 0, budget)
        if cut <= 0:
            cut = budget
        out.append(remaining[:cut].strip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining.strip():
        out.append(remaining.strip("\n"))
    if fenced:
        return [f"```\n{piece}\n```" for piece in out]
    return [piece.strip() for piece in out]


def blocks_for(text: str) -> list[dict[str, object]]:
    return [
        {"type": "section", "text": {"type": "mrkdwn", "text": chunk}} for chunk in chunks(text)
    ]


_MENTION = re.compile(r"<[@!#][A-Z0-9]+(\|[^>]*)?>")
_LABELLED_LINK = re.compile(r"<(https?://[^|>]+)\|([^>]*)>")
_BARE_LINK = re.compile(r"<(https?://[^|>\s]+)>")


def clean_prompt(text: str) -> str:
    """Strip Slack's own markup from an incoming message.

    Slack wraps URLs and mentions in angle brackets before the event reaches
    us. Left in place they read to the model as malformed markup; the URLs in
    particular have to survive, because "summarise this link" is a real request.
    """

    body = _LABELLED_LINK.sub(
        lambda m: f"{m.group(2)} ({m.group(1)})" if m.group(2) else m.group(1), text
    )
    body = _BARE_LINK.sub(lambda m: m.group(1), body)
    body = _MENTION.sub("", body)
    return " ".join(body.split()).strip()
