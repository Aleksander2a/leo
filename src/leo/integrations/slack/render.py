"""Pure conservative rendering for Slack delivery payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlsplit

from leo.harness.models import ClaimKind, CoordinatorResult, Observation, RunStatus
from leo.url_policy import is_public_https_url

# Version 4 keeps durable run IDs as internal correlation metadata while removing
# them from every user-facing Slack payload. It also replaces mechanical terminal
# status boilerplate with bounded, conversational recovery copy. The version is
# part of the durable outbox part identity, so semantic changes must increment it.
RENDERER_VERSION = 4
MAX_SLACK_CHARS = 3500
RESEARCH_DISCLAIMER = "Research evidence, not financial advice."
_CONTROL_CHARS = (
    frozenset(chr(value) for value in range(0x00, 0x20))
    | frozenset(chr(value) for value in range(0x7F, 0xA0))
) - {"\n", "\t"}
_BIDI_CHARS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"\bxox(?:a|b|p|r|s)-[A-Za-z0-9-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bxapp-[A-Za-z0-9-]{8,}\b", re.IGNORECASE),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(
        r"https://hooks\.slack\.com/services/[A-Za-z0-9/_-]+",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpostgres(?:ql)?://[^\s:/@]+:[^\s@]+@[^\s]+",
        re.IGNORECASE,
    ),
)
_SENSITIVE_QUERY_KEYS = frozenset(
    {"access_token", "api_key", "apikey", "auth", "key", "password", "secret", "signature", "token"}
)
_INTERNAL_RUN_ID_PATTERN = re.compile(
    r"\brun-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class RenderedSlackText:
    version: int
    chunks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SlackSource:
    """A verifier-selected source eligible for a user-facing link."""

    label: str
    url: str


@dataclass(frozen=True, slots=True)
class SlackClaim:
    """One verifier-approved claim and its already-selected sources."""

    statement: str
    sources: tuple[SlackSource, ...] = ()


@dataclass(frozen=True, slots=True)
class SlackVerifiedResult:
    """Typed, verifier-owned content accepted by the Slack renderer."""

    run_id: str
    answer: str
    claims: tuple[SlackClaim, ...] = ()
    inferences: tuple[str, ...] = ()
    affected_assumption: str | None = None
    uncertainty: str | None = None
    research_disclaimer_required: bool | None = None


@dataclass(frozen=True, slots=True)
class SlackTerminalResult:
    """Trusted durable terminal state rendered without exposing internal diagnostics.

    ``terminal_reason`` is used only to select a bounded, prewritten explanation. It
    is never copied into the Slack payload. Partial results must already have passed
    a verifier before a caller places them in ``verified_partial_results``.
    """

    run_id: str
    status: RunStatus | str
    terminal_reason: str | None = None
    completed_output: str | None = None
    verified_partial_results: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SlackResearchResult:
    """Verifier-selected rich research sections; no model-owned source is rendered."""

    run_id: str
    facts: tuple[SlackClaim, ...] = ()
    inferences: tuple[str, ...] = ()
    affected_assumption: str | None = None
    uncertainty: str | None = None


def render_research_result(
    result: SlackResearchResult,
    *,
    max_chars: int = MAX_SLACK_CHARS,
) -> RenderedSlackText:
    run_id = _validated_run_id(result.run_id, result_kind="research")
    lines: list[str] = []
    if result.facts:
        lines.append("Facts")
        for claim in result.facts:
            if not claim.statement.strip():
                raise SlackRenderPolicyError("research fact must be non-empty")
            lines.append(f"• {_escape_text(claim.statement)}")
            for source in claim.sources:
                source_line = _source_line(source, max_chars=max_chars)
                if source_line is not None:
                    lines.append(source_line)
    if result.inferences:
        lines.append("Inferences")
        lines.extend(f"• {_escape_text(item)}" for item in result.inferences if item.strip())
    if result.affected_assumption:
        lines.append(f"Affected assumption: {_escape_text(result.affected_assumption)}")
    if result.uncertainty:
        lines.append(f"Uncertainty: {_escape_text(result.uncertainty)}")
    lines.append(RESEARCH_DISCLAIMER)
    return _chunk_sanitized_text(
        _hide_internal_run_id("\n".join(lines), run_id),
        max_chars=max_chars,
    )


_MD_FENCE = re.compile(r"```[\s\S]*?```")
_MD_INLINE_CODE = re.compile(r"`[^`\r\n]+`")
_MD_LINK = re.compile(r"\[([^\]\r\n]{1,256})\]\((https?://[^\s)]{1,2000})\)")
_MD_HEADING = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
_MD_BOLD = re.compile(r"\*\*(?!\s)([^*\r\n]+?)(?<!\s)\*\*")
_MD_BOLD_UNDERSCORE = re.compile(r"(?<![\w_])__(?!\s)([^_\r\n]+?)(?<!\s)__(?![\w_])")
_MD_BULLET = re.compile(r"(?m)^(\s*)[-*+]\s+")
_MD_RULE = re.compile(r"(?m)^\s*(?:---+|\*\*\*+|___+)\s*$")


def markdown_to_mrkdwn(text: str) -> str:
    """Convert the Markdown models actually emit into Slack's mrkdwn dialect.

    Slack does not render Markdown. Answers were posted verbatim, so a model's
    ``**bold**`` arrived as literal asterisks, ``[label](url)`` as literal
    brackets, and ``## Heading`` as a stray hash -- a good answer that looked
    broken. Fenced and inline code are masked first so their contents survive
    untouched, and the substitutions are deliberately conservative: anything not
    confidently recognized is left exactly as written.
    """

    protected: list[str] = []

    def _mask(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    masked = _MD_FENCE.sub(_mask, text)
    masked = _MD_INLINE_CODE.sub(_mask, masked)

    # Links first: their label may itself contain emphasis markers.
    masked = _MD_LINK.sub(lambda m: f"<{m.group(2)}|{m.group(1)}>", masked)
    # Slack has no headings; bold carries the same weight.
    masked = _MD_HEADING.sub(lambda m: f"*{m.group(1)}*", masked)
    masked = _MD_BOLD.sub(lambda m: f"*{m.group(1)}*", masked)
    masked = _MD_BOLD_UNDERSCORE.sub(lambda m: f"*{m.group(1)}*", masked)
    masked = _MD_BULLET.sub(lambda m: f"{m.group(1)}• ", masked)
    masked = _MD_RULE.sub("", masked)

    # Restore highest index first. Inline-code spans are masked after fenced
    # blocks, so an inline-code original can itself contain a fence placeholder;
    # restoring forwards would reinsert that placeholder after its turn had
    # already passed and leave the marker in the delivered text.
    for index in range(len(protected) - 1, -1, -1):
        masked = masked.replace(f"\x00{index}\x00", protected[index])
    return masked.replace("\x00", "")


class SlackRenderPolicyError(ValueError):
    """The typed result cannot be rendered under the conservative policy."""


def render_slack_text(text: str, *, max_chars: int = MAX_SLACK_CHARS) -> RenderedSlackText:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if not text:
        raise ValueError("Slack text must be non-empty")
    # Conversion runs *after* escaping so the `<url|label>` markup it emits is
    # not itself escaped, and so no `<` in the model's prose can forge markup.
    return _chunk_sanitized_text(markdown_to_mrkdwn(_escape_text(text)), max_chars=max_chars)


def render_terminal_result(
    result: SlackTerminalResult,
    *,
    max_chars: int = MAX_SLACK_CHARS,
) -> RenderedSlackText:
    """Render a durable terminal outcome as safe, useful conversational text.

    Durable status, reason, and run ID remain machine truth. The user sees only a
    stable category explanation, any explicitly verifier-approved partials, and a
    useful recovery action. Unknown statuses and reasons deliberately fall back to
    the same safe generic message.
    """

    run_id = _validated_run_id(result.run_id, result_kind="terminal")
    status = result.status.value if isinstance(result.status, RunStatus) else result.status
    normalized_status = status.strip()[:64].casefold()
    if normalized_status == RunStatus.COMPLETED.value and result.completed_output:
        return _chunk_sanitized_text(
            _hide_internal_run_id(
                markdown_to_mrkdwn(_escape_text(result.completed_output)),
                run_id,
            ),
            max_chars=max_chars,
        )

    explanation, next_step = _terminal_conversation_copy(
        status=normalized_status,
        terminal_reason=result.terminal_reason,
    )
    lines = [explanation]
    partials = _bounded_verified_partials(result.verified_partial_results)
    if partials:
        lines.append("I did confirm this before stopping:")
        lines.extend(f"• {_escape_text(item)}" for item in partials)
    lines.append(next_step)
    return _chunk_sanitized_text(
        _hide_internal_run_id("\n".join(lines), run_id),
        max_chars=max_chars,
    )


def render_verified_result(
    result: SlackVerifiedResult,
    *,
    max_chars: int = MAX_SLACK_CHARS,
) -> RenderedSlackText:
    """Render verifier-selected content while keeping run metadata internal."""

    run_id = _validated_run_id(result.run_id, result_kind="verified")
    if not result.answer.strip():
        raise SlackRenderPolicyError("verified result requires an answer")
    lines = [_escape_text(result.answer)]
    if result.claims:
        lines.append("Facts")
    for claim in result.claims:
        if not claim.statement.strip():
            raise SlackRenderPolicyError("verified claim must be non-empty")
        lines.append(f"• {_escape_text(claim.statement)}")
        for source in claim.sources:
            source_line = _source_line(source, max_chars=max_chars)
            if source_line is not None:
                lines.append(source_line)
    if result.inferences:
        lines.append("Inferences")
        lines.extend(f"• {_escape_text(item)}" for item in result.inferences if item.strip())
    if result.affected_assumption:
        lines.append(f"Affected assumption: {_escape_text(result.affected_assumption)}")
    if result.uncertainty:
        lines.append(f"Uncertainty: {_escape_text(result.uncertainty)}")
    disclaimer_required = result.research_disclaimer_required
    if disclaimer_required is None:
        disclaimer_required = _result_looks_external_or_financial(result)
    if disclaimer_required:
        lines.append(RESEARCH_DISCLAIMER)
    return _chunk_sanitized_text(
        _hide_internal_run_id("\n".join(lines), run_id),
        max_chars=max_chars,
    )


def verified_result_from_coordinator(
    result: CoordinatorResult,
    *,
    include_evidence_details: bool = True,
) -> SlackVerifiedResult:
    """Convert the coordinator result, optionally hiding internal evidence details."""

    if result.run.final_output is None or result.run.status.value != "completed":
        raise SlackRenderPolicyError("only a completed coordinator result can be rendered")
    if not include_evidence_details:
        return SlackVerifiedResult(
            run_id=result.run.id,
            answer=result.run.final_output,
        )
    observations = {observation.id: observation for observation in result.observations}
    claims: list[SlackClaim] = []
    inferences: list[str] = []
    affected_assumption: str | None = None
    uncertainty: str | None = None
    cited_observations: list[Observation] = []
    for claim in result.claims:
        cited_observations.extend(
            observation
            for observation_id in claim.observation_ids
            if (observation := observations.get(observation_id)) is not None
        )
        if claim.kind is ClaimKind.INFERENCE:
            inferences.append(claim.statement)
            continue
        if claim.kind is ClaimKind.AFFECTED_ASSUMPTION:
            affected_assumption = claim.statement
            continue
        if claim.kind is ClaimKind.UNCERTAINTY:
            uncertainty = claim.statement
            continue
        sources: list[SlackSource] = []
        for observation_id in claim.observation_ids:
            observation = observations.get(observation_id)
            if observation is None:
                raise SlackRenderPolicyError("claim references an unavailable observation")
            sources.extend(_sources_for_verified_claim(claim.statement, observation))
        claims.append(SlackClaim(statement=claim.statement, sources=tuple(sources)))
    return SlackVerifiedResult(
        run_id=result.run.id,
        answer=result.run.final_output,
        claims=tuple(claims),
        inferences=tuple(inferences),
        affected_assumption=affected_assumption,
        uncertainty=uncertainty,
        research_disclaimer_required=(
            any(_observation_is_external_research(item) for item in cited_observations)
            or any(_text_looks_financial(item.statement) for item in result.claims)
            or _is_substantive_financial_answer(result.run.final_output)
        ),
    )


def _result_looks_external_or_financial(result: SlackVerifiedResult) -> bool:
    if any(claim.sources for claim in result.claims):
        return True
    structured_content = " ".join(
        value
        for value in (
            *(claim.statement for claim in result.claims),
            *result.inferences,
            result.affected_assumption,
            result.uncertainty,
        )
        if value
    )
    return _text_looks_financial(structured_content) or _is_substantive_financial_answer(
        result.answer
    )


def _observation_is_external_research(observation: Observation) -> bool:
    kind = observation.kind
    provider = observation.source.provider
    return (
        observation.source.url is not None
        or kind.startswith(("market.", "sec.", "web."))
        or (
            provider in {"finnhub", "sec-edgar", "web", "wikipedia-opensearch"}
            or provider.startswith("mcp:")
        )
    )


def _text_looks_financial(text: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
    if tokens.intersection(
        {
            "allocation",
            "bond",
            "bonds",
            "buy",
            "dividend",
            "dividends",
            "equity",
            "etf",
            "filing",
            "filings",
            "fund",
            "investment",
            "investing",
            "market",
            "portfolio",
            "price",
            "quote",
            "sec",
            "sell",
            "stock",
            "stocks",
            "thesis",
            "yield",
        }
    ):
        return True

    # "Share" is common conversational language. Treat it as financial only
    # when a second, unambiguous trading term supplies the missing context.
    return bool(
        tokens.intersection({"share", "shares"})
        and tokens.intersection(
            {"dividend", "dividends", "earnings", "ticker", "trade", "trades", "trading"}
        )
    )


def _is_substantive_financial_answer(text: str) -> bool:
    if not _text_looks_financial(text):
        return False
    return not _looks_like_clarification_only(text)


def _looks_like_clarification_only(text: str) -> bool:
    normalized = " ".join(text.casefold().split())
    if "?" not in normalized:
        return False
    if normalized.startswith(
        (
            "are you after ",
            "are you looking for ",
            "can you clarify ",
            "can you tell me ",
            "could you clarify ",
            "could you tell me ",
            "do you mean ",
            "what is your ",
            "what kind of ",
            "what's your ",
            "which ",
        )
    ):
        return True
    asks_for_missing_detail = any(
        phrase in normalized
        for phrase in (
            "are you after ",
            "are you looking for ",
            "can you clarify",
            "can you tell me",
            "could you clarify",
            "could you tell me",
            "tell me a bit about",
            "what you're looking for",
            "what you are looking for",
            "what's your risk",
            "what is your risk",
        )
    )
    defers_answer_until_reply = any(
        phrase in normalized
        for phrase in (
            "before i can",
            "depends on your",
            "once i know",
            "once you tell me",
        )
    )
    return asks_for_missing_detail and defers_answer_until_reply


def _sources_for_verified_claim(
    statement: str,
    observation: Observation,
) -> tuple[SlackSource, ...]:
    """Select only links whose meaning matches the verified claim.

    Finnhub endpoint documentation describes an API contract; it does not
    substantiate a returned quote, profile, earnings value, or financial metric.
    Company-news data is different because each canonical item carries its article
    URL. Bind that URL by reconstructing the exact canonical statement, and omit a
    link if the observation is malformed or the mapping is ambiguous.
    """

    if observation.source.provider == "finnhub":
        if observation.kind != "market.get_company_news":
            return ()
        return _finnhub_news_sources(statement, observation)
    if observation.source.url is None:
        return ()
    return (
        SlackSource(
            label=observation.source.reference,
            url=observation.source.url,
        ),
    )


def _finnhub_news_sources(
    statement: str,
    observation: Observation,
) -> tuple[SlackSource, ...]:
    symbol = observation.data.get("symbol")
    items = observation.data.get("items")
    if not isinstance(symbol, str) or not isinstance(items, list):
        return ()
    normalized_statement = " ".join(statement.split()).casefold()
    matches: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            return ()
        published_at = item.get("published_at")
        provider = item.get("source")
        headline = item.get("headline")
        url = item.get("url")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (published_at, provider, headline, url)
        ):
            return ()
        # The all() guard above narrows values for runtime, but keep the explicit
        # checks for static analyzers and future contract changes.
        if not (
            isinstance(published_at, str)
            and isinstance(provider, str)
            and isinstance(headline, str)
            and isinstance(url, str)
            and is_public_https_url(url)
        ):
            return ()
        canonical = (
            f"On {published_at}, {provider} reported for {symbol}: {headline} Source URL: {url}"
        )
        if " ".join(canonical.split()).casefold() == normalized_statement:
            matches.add((provider, url))
    if len(matches) != 1:
        return ()
    provider, url = matches.pop()
    return (SlackSource(label=provider, url=url),)


def _terminal_conversation_copy(
    *,
    status: str,
    terminal_reason: str | None,
) -> tuple[str, str]:
    reason = (terminal_reason or "").strip()[:512].casefold()
    if status == RunStatus.CANCELLED.value:
        if reason == "slack_user_cancelled":
            return (
                "You asked me to stop, so I stopped before finishing this request.",
                "If you want to resume, send the part you'd like me to pick up.",
            )
        return (
            "I stopped before I could finish this request.",
            "If you want to resume, send the part you'd like me to pick up.",
        )
    if status == RunStatus.TIMED_OUT.value:
        return (
            "I ran out of time before I could finish this request.",
            "Try again, or tell me which part to handle first.",
        )
    if status == RunStatus.BUDGET_EXHAUSTED.value:
        return (
            "I reached this request's processing limit before I could finish.",
            "Reply “continue” and I'll focus on the most important unfinished part, "
            "or narrow the request.",
        )
    if status == RunStatus.REQUIRES_ACTION.value:
        return (
            "I need a little more information or permission before I can continue.",
            "Reply with the missing detail or approval and I'll pick it up from there.",
        )
    if status == RunStatus.FAILED.value:
        if any(
            marker in reason
            for marker in (
                "verif",
                "citation",
                "ground",
                "source_claim",
                "completion_contract",
            )
        ):
            return (
                "I found information, but I couldn't verify it strongly enough "
                "to give you a reliable answer.",
                "Ask me to try different sources or narrow the claim you want checked.",
            )
        if any(
            marker in reason
            for marker in (
                "model",
                "llm",
                "inference",
                "reasoning",
            )
        ):
            return (
                "The reasoning service stopped unexpectedly before I could finish the answer.",
                "Ask me to retry; if it happens again, tell me which part to handle first.",
            )
        if any(
            marker in reason
            for marker in (
                "tool",
                "provider",
                "gateway",
                "source",
                "rate_limit",
                "normalization",
                "required_action",
            )
        ):
            return (
                "One of the sources or tools I needed wasn't available, "
                "so I couldn't finish the answer.",
                "Ask me to retry, or tell me to answer with the information "
                "I can verify without it.",
            )
        if any(
            marker in reason for marker in ("context", "authority", "scope", "membership", "access")
        ):
            return (
                "I'm missing conversation context or access I need to answer this correctly.",
                "Share the missing detail here or point me to the relevant conversation, "
                "then I'll try again.",
            )
        if "retry" in reason or "attempt" in reason:
            return (
                "I tried again but kept hitting the same temporary problem.",
                "Ask me to retry later, or narrow the request so I can use a different path.",
            )
        if any(marker in reason for marker in ("store", "storage", "persist", "lease")):
            return (
                "I couldn't save or recover enough progress to finish this request reliably.",
                "Ask me to retry and I'll start a fresh attempt.",
            )
    return (
        "I hit an unexpected problem before I could finish the answer.",
        "Please try again. If it happens again, tell me which part matters most "
        "and I'll focus there.",
    )


def _validated_run_id(value: str, *, result_kind: str) -> str:
    run_id = value.strip()
    if not run_id:
        raise SlackRenderPolicyError(f"{result_kind} result requires a run ID")
    if len(run_id) > 256:
        raise SlackRenderPolicyError(f"{result_kind} run ID is too long")
    return run_id


def _hide_internal_run_id(text: str, run_id: str) -> str:
    """Remove internal correlation metadata even if verifier content repeats it."""

    escaped_run_id = _escape_text(run_id)
    exact_metadata_line = re.compile(
        rf"(?im)^[ \t]*(?:run(?:[ \t]+id)?|request[ \t]+id)[ \t]*:[ \t]*"
        rf"{re.escape(escaped_run_id)}[ \t]*(?:\n|$)"
    )
    without_metadata_line = exact_metadata_line.sub("", text)
    return without_metadata_line.replace(escaped_run_id, "this request")


def _bounded_verified_partials(values: tuple[str, ...]) -> tuple[str, ...]:
    partials: list[str] = []
    for value in values:
        normalized = " ".join(value.split())
        if not normalized:
            continue
        if len(normalized) > 1_000:
            normalized = f"{normalized[:999]}…"
        partials.append(normalized)
        if len(partials) == 5:
            break
    return tuple(partials)


def _escape_text(text: str) -> str:
    sanitized = "".join(
        character
        for character in text
        if character not in _CONTROL_CHARS and character not in _BIDI_CHARS
    )
    for pattern in _SECRET_PATTERNS:
        sanitized = pattern.sub("[redacted credential]", sanitized)
    sanitized = _INTERNAL_RUN_ID_PATTERN.sub("this request", sanitized)
    return sanitized.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chunk_sanitized_text(text: str, *, max_chars: int) -> RenderedSlackText:
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    chunks_list: list[str] = []
    remaining = text
    if not remaining:
        raise ValueError("Slack text became empty after sanitization")
    while len(remaining) > max_chars:
        # Keep complete logical lines together whenever they fit.  A long single
        # line still uses the deterministic fixed-width fallback.
        newline = remaining.rfind("\n", 0, max_chars)
        cut = newline + 1 if newline > 0 else max_chars
        chunks_list.append(remaining[:cut])
        remaining = remaining[cut:]
    chunks_list.append(remaining)
    chunks = tuple(chunks_list)
    return RenderedSlackText(version=RENDERER_VERSION, chunks=chunks)


def _safe_source_url(url: str) -> bool:
    if not is_public_https_url(url):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.hostname is not None
        and parsed.username is None
        and parsed.password is None
        and not any(
            key.casefold() in _SENSITIVE_QUERY_KEYS for key, _value in parse_qsl(parsed.query)
        )
        and not any(pattern.search(url) for pattern in _SECRET_PATTERNS)
        and "<" not in url
        and ">" not in url
        and not any(character in url for character in _CONTROL_CHARS | _BIDI_CHARS)
    )


def _source_line(source: SlackSource, *, max_chars: int) -> str | None:
    if not _safe_source_url(source.url):
        return None
    label = _escape_text(source.label.strip())
    if not label:
        return None
    if len(label) > 256:
        label = f"{label[:255]}…"
    line = f"  Source: <{source.url}|{label}>"
    # Never split Slack's trusted link syntax across delivery chunks. A source whose
    # complete markup cannot fit is omitted while the verifier-backed statement stays.
    return line if len(line) <= max_chars else None
