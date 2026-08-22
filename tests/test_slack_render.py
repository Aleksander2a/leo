from __future__ import annotations

import random
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from leo.harness.models import (
    Claim,
    ClaimKind,
    CoordinatorResult,
    EvidenceQuality,
    Observation,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    TaskStatus,
    Thread,
)
from leo.integrations.slack.render import (
    RENDERER_VERSION,
    RESEARCH_DISCLAIMER,
    SlackClaim,
    SlackResearchResult,
    SlackSource,
    SlackTerminalResult,
    SlackVerifiedResult,
    render_research_result,
    render_slack_text,
    render_terminal_result,
    render_verified_result,
    verified_result_from_coordinator,
)


def _completed_result_for_observation(
    observation: Observation,
    statement: str,
) -> CoordinatorResult:
    scope = observation.scope
    answer = statement
    return CoordinatorResult(
        thread=Thread(
            id="thread-source",
            scope=scope,
            origin=OriginRef(provider="slack", external_thread_id="C1"),
        ),
        task=Task(
            id="task-source",
            thread_id="thread-source",
            scope=scope,
            objective="Research the source.",
            status=TaskStatus.COMPLETED,
            observation_ids=(observation.id,),
            final_output=answer,
        ),
        run=Run(
            id=observation.run_id,
            task_id="task-source",
            scope=scope,
            status=RunStatus.COMPLETED,
            started_at=observation.observed_at,
            final_output=answer,
            terminal_reason="verified_completion",
        ),
        observations=(observation,),
        claims=(
            Claim(
                id="claim-source",
                scope=scope,
                run_id=observation.run_id,
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(observation.id,),
            ),
        ),
        events=(),
    )


def test_renderer_escapes_markup_and_unsafe_controls() -> None:
    rendered = render_slack_text("<!channel> & <@U123>\u202e safe\x00")
    assert RENDERER_VERSION == 4
    assert rendered.version == RENDERER_VERSION
    assert rendered.chunks == ("&lt;!channel&gt; &amp; &lt;@U123&gt; safe",)


def test_renderer_is_deterministic_and_chunks_at_fixed_limit() -> None:
    first = render_slack_text("abcdef", max_chars=2)
    second = render_slack_text("abcdef", max_chars=2)
    assert first == second
    assert first.chunks == ("ab", "cd", "ef")


@pytest.mark.parametrize(
    ("status", "reason", "expected", "recovery"),
    [
        (
            RunStatus.BUDGET_EXHAUSTED,
            "iteration_budget_exhausted",
            "reached this request's processing limit",
            "Reply “continue”",
        ),
        (
            RunStatus.FAILED,
            "tool_failure:FINNHUB_RATE_LIMIT",
            "sources or tools I needed wasn't available",
            "Ask me to retry",
        ),
        (
            RunStatus.FAILED,
            "non_retryable_verification_failure",
            "couldn't verify it strongly enough",
            "try different sources",
        ),
        (
            RunStatus.FAILED,
            "model_gateway_error",
            "reasoning service stopped unexpectedly",
            "which part to handle first",
        ),
        (
            RunStatus.CANCELLED,
            "slack_user_cancelled",
            "You asked me to stop",
            "If you want to resume",
        ),
        (
            RunStatus.TIMED_OUT,
            "slack_runtime_deadline_exceeded",
            "ran out of time",
            "which part to handle first",
        ),
        (
            "future_unknown_terminal",
            "internal_error",
            "hit an unexpected problem",
            "Please try again",
        ),
    ],
)
def test_terminal_renderer_maps_durable_failures_to_safe_conversational_recovery(
    status: RunStatus | str,
    reason: str,
    expected: str,
    recovery: str,
) -> None:
    rendered = render_terminal_result(
        SlackTerminalResult(
            run_id="run-terminal",
            status=status,
            terminal_reason=(
                reason if status == RunStatus.CANCELLED else reason + ":Bearer abcdefghijklmnop"
            ),
        )
    )

    payload = "".join(rendered.chunks)
    assert expected in payload
    assert recovery in payload
    assert "run-terminal" not in payload
    assert "Run:" not in payload
    assert "I haven't presented unverified work" not in payload
    assert reason not in payload
    assert "abcdefghijklmnop" not in payload
    assert "budget_exhausted" not in payload
    assert "future_unknown_terminal" not in payload


def test_terminal_renderer_preserves_only_explicit_verified_partials_and_redacts_them() -> None:
    rendered = render_terminal_result(
        SlackTerminalResult(
            run_id="run-partial",
            status=RunStatus.TIMED_OUT,
            terminal_reason="provider_timeout:postgresql://demo:secret@example.com/leo",
            verified_partial_results=(
                "The primary filing date was verified.",
                "Credential " + "xox" + "b-1234567890-sensitive",
                "",
            ),
        ),
        max_chars=96,
    )

    payload = "".join(rendered.chunks)
    assert "I did confirm this before stopping:" in payload
    assert "• The primary filing date was verified." in payload
    assert "[redacted credential]" in payload
    assert "sensitive" not in payload
    assert "postgresql" not in payload
    assert "run-partial" not in payload
    assert "Run:" not in payload
    assert all(0 < len(chunk) <= 96 for chunk in rendered.chunks)


def test_terminal_renderer_keeps_completed_output_while_sanitizing_metadata() -> None:
    rendered = render_terminal_result(
        SlackTerminalResult(
            run_id="run-completed",
            status=RunStatus.COMPLETED,
            completed_output="Verified answer <without broadcast>.",
        )
    )

    assert rendered.chunks == ("Verified answer &lt;without broadcast&gt;.",)


def test_renderer_prefers_logical_line_boundaries_for_long_rich_output() -> None:
    rendered = render_slack_text("Facts\n" + ("value " * 4), max_chars=14)

    assert rendered.chunks == ("Facts\n", "value value va", "lue value ")
    assert all(0 < len(chunk) <= 14 for chunk in rendered.chunks)


def test_verified_renderer_includes_claim_sources_and_uncertainty_without_internal_id() -> None:
    rendered = render_verified_result(
        SlackVerifiedResult(
            run_id="run-123",
            answer="NVDA is quoted at 216.85.",
            claims=(
                SlackClaim(
                    statement="NVDA is quoted at 216.85.",
                    sources=(
                        SlackSource(label="synthetic quote", url="https://example.com/quote"),
                    ),
                ),
            ),
            uncertainty="Quote is current as of the provider observation.",
        )
    )
    assert rendered.chunks == (
        "NVDA is quoted at 216.85.\n"
        "Facts\n"
        "• NVDA is quoted at 216.85.\n"
        "  Source: <https://example.com/quote|synthetic quote>\n"
        "Uncertainty: Quote is current as of the provider observation.\n"
        "Research evidence, not financial advice.",
    )


def test_verified_renderer_omits_research_disclaimer_for_pure_internal_memory() -> None:
    rendered = render_verified_result(
        SlackVerifiedResult(
            run_id="run-memory",
            answer="The synthetic Project Borealis preference is amber hexagons.",
            inferences=("The synthetic Project Borealis preference is amber hexagons.",),
            research_disclaimer_required=False,
        )
    )

    payload = "".join(rendered.chunks)
    assert "Inferences\n" in payload
    assert RESEARCH_DISCLAIMER not in payload
    assert payload.endswith("• The synthetic Project Borealis preference is amber hexagons.")
    assert "run-memory" not in payload


@pytest.mark.parametrize(
    "answer",
    [
        "Context stays private unless you deliberately share it.",
        "Share this context with the group when you are ready.",
        "The assistant shares a memory only when the policy allows it.",
    ],
)
def test_verified_renderer_does_not_treat_conversational_sharing_as_finance(
    answer: str,
) -> None:
    rendered = render_verified_result(SlackVerifiedResult(run_id="run-generic", answer=answer))

    assert RESEARCH_DISCLAIMER not in "".join(rendered.chunks)


@pytest.mark.parametrize(
    "answer",
    [
        "NVDA shares trade higher today.",
        "The share price is 214.72.",
        "Buy ten shares only if that allocation fits your plan.",
    ],
)
def test_verified_renderer_keeps_disclaimer_for_financial_share_language(
    answer: str,
) -> None:
    rendered = render_verified_result(SlackVerifiedResult(run_id="run-finance", answer=answer))

    assert RESEARCH_DISCLAIMER in "".join(rendered.chunks)


def test_financial_clarification_without_claims_has_no_research_disclaimer_or_run_id() -> None:
    now = datetime(2026, 8, 22, tzinfo=UTC)
    scope = ScopeKey(organization_id="org", strategy_id="conversation")
    answer = (
        "That's a broad investing question, and the useful answer depends on your goals. "
        "Could you tell me what you're looking for: dividend income, growth stocks, "
        "or a specific sector? What's your risk tolerance and time horizon? "
        "Once I know that, I can research concrete ideas."
    )
    result = CoordinatorResult(
        thread=Thread(
            id="thread-clarification",
            scope=scope,
            origin=OriginRef(provider="slack", external_thread_id="C1"),
        ),
        task=Task(
            id="task-clarification",
            thread_id="thread-clarification",
            scope=scope,
            objective="What are some interesting investing opportunities?",
            status=TaskStatus.COMPLETED,
            final_output=answer,
        ),
        run=Run(
            id="run-12345678-1234-4abc-8def-1234567890ab",
            task_id="task-clarification",
            scope=scope,
            status=RunStatus.COMPLETED,
            started_at=now,
            final_output=answer,
            terminal_reason="verified_completion",
        ),
        observations=(),
        claims=(),
        events=(),
    )

    coordinator_view = verified_result_from_coordinator(result)
    inferred_view = SlackVerifiedResult(run_id=result.run.id, answer=answer)
    payloads = tuple(
        "".join(render_verified_result(view).chunks) for view in (coordinator_view, inferred_view)
    )

    assert coordinator_view.research_disclaimer_required is False
    assert all(RESEARCH_DISCLAIMER not in payload for payload in payloads)
    assert all(result.run.id not in payload for payload in payloads)
    assert all("Run:" not in payload for payload in payloads)


def test_coordinator_memory_result_does_not_request_research_disclaimer() -> None:
    now = datetime(2026, 8, 21, tzinfo=UTC)
    scope = ScopeKey(organization_id="org", strategy_id="conversation")
    answer = "The synthetic Project Borealis preference is amber hexagons."
    result = CoordinatorResult(
        thread=Thread(
            id="thread-memory",
            scope=scope,
            origin=OriginRef(provider="slack", external_thread_id="D1"),
        ),
        task=Task(
            id="task-memory",
            thread_id="thread-memory",
            scope=scope,
            objective="What do you remember about Project Borealis?",
            status=TaskStatus.COMPLETED,
            observation_ids=("obs-memory",),
            final_output=answer,
        ),
        run=Run(
            id="run-memory",
            task_id="task-memory",
            scope=scope,
            status=RunStatus.COMPLETED,
            started_at=now,
            final_output=answer,
            terminal_reason="verified_completion",
        ),
        observations=(
            Observation(
                id="obs-memory",
                scope=scope,
                run_id="run-memory",
                tool_call_id="call-memory",
                kind="memory.search",
                data={"items": [{"kind": "inline", "content": answer}]},
                source=SourceRef(provider="leo_memory", reference="query-hash"),
                observed_at=now,
                raw_hash="a" * 64,
                quality=EvidenceQuality.INTERNAL_CONTEXT,
            ),
        ),
        claims=(
            Claim(
                id="claim-memory",
                scope=scope,
                run_id="run-memory",
                kind=ClaimKind.INFERENCE,
                statement=answer,
                observation_ids=("obs-memory",),
            ),
        ),
        events=(),
    )

    view = verified_result_from_coordinator(result)
    payload = "".join(render_verified_result(view).chunks)

    assert view.research_disclaimer_required is False
    assert RESEARCH_DISCLAIMER not in payload


def test_finnhub_news_link_is_bound_to_exact_verified_article_statement() -> None:
    now = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
    scope = ScopeKey(organization_id="org", strategy_id="conversation")
    first_url = "https://example.org/news/first"
    second_url = "https://example.org/news/second"
    second_statement = (
        f"On {now.isoformat()}, Second Wire reported for MSFT: Second update "
        f"Source URL: {second_url}"
    )
    observation = Observation(
        id="obs-news",
        scope=scope,
        run_id="run-news",
        tool_call_id="call-news",
        kind="market.get_company_news",
        data={
            "symbol": "MSFT",
            "items": [
                {
                    "published_at": now.isoformat(),
                    "source": "First Wire",
                    "headline": "First update",
                    "url": first_url,
                },
                {
                    "published_at": now.isoformat(),
                    "source": "Second Wire",
                    "headline": "Second update",
                    "url": second_url,
                },
            ],
        },
        # An old or malformed observation-level URL must never select the link.
        source=SourceRef(
            provider="finnhub",
            reference="company-news:MSFT:2026-08-15:2026-08-22",
            url="https://finnhub.io/docs/api/company-news",
        ),
        observed_at=now,
        raw_hash="b" * 64,
    )

    view = verified_result_from_coordinator(
        _completed_result_for_observation(observation, second_statement)
    )
    payload = "".join(render_verified_result(view).chunks)

    assert view.claims[0].sources == (SlackSource(label="Second Wire", url=second_url),)
    assert second_url in payload
    assert first_url not in payload
    assert "finnhub.io/docs" not in payload


def test_finnhub_source_projection_fails_closed_for_mismatch_and_api_docs() -> None:
    now = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
    scope = ScopeKey(organization_id="org", strategy_id="conversation")
    article_url = "https://example.org/news/exact"
    mismatched_statement = (
        f"On {now.isoformat()}, Example Wire reported for MSFT: Different headline "
        f"Source URL: {article_url}"
    )
    news = Observation(
        id="obs-news-mismatch",
        scope=scope,
        run_id="run-news-mismatch",
        tool_call_id="call-news",
        kind="market.get_company_news",
        data={
            "symbol": "MSFT",
            "items": [
                {
                    "published_at": now.isoformat(),
                    "source": "Example Wire",
                    "headline": "Exact headline",
                    "url": article_url,
                }
            ],
        },
        source=SourceRef(provider="finnhub", reference="company-news:MSFT", url=article_url),
        observed_at=now,
        raw_hash="c" * 64,
    )
    quote_statement = "MSFT is quoted at 420."
    quote = Observation(
        id="obs-quote-docs",
        scope=scope,
        run_id="run-quote-docs",
        tool_call_id="call-quote",
        kind="market.get_quote",
        data={"symbol": "MSFT", "price": 420.0},
        source=SourceRef(
            provider="finnhub",
            reference="quote:MSFT:1787390400",
            url="https://finnhub.io/docs/api/quote",
        ),
        observed_at=now,
        raw_hash="d" * 64,
    )

    mismatched_view = verified_result_from_coordinator(
        _completed_result_for_observation(news, mismatched_statement)
    )
    quote_view = verified_result_from_coordinator(
        _completed_result_for_observation(quote, quote_statement)
    )

    assert mismatched_view.claims[0].sources == ()
    assert quote_view.claims[0].sources == ()


def test_verified_renderer_neutralizes_markup_and_drops_unsafe_sources() -> None:
    rendered = render_verified_result(
        SlackVerifiedResult(
            run_id="run-unsafe",
            answer="<!channel> <@U123>",
            claims=(
                SlackClaim(
                    statement="safe <b>text</b>",
                    sources=(
                        SlackSource(label="secret", url="file:///private/token"),
                        SlackSource(label="insecure", url="http://example.com/report"),
                        SlackSource(label="loopback", url="https://127.0.0.1/report"),
                        SlackSource(label="localhost", url="https://localhost/report"),
                        SlackSource(
                            label="query secret",
                            url="https://example.com/report?token=private-value",
                        ),
                        SlackSource(label="safe", url="https://example.com"),
                    ),
                ),
            ),
        )
    )
    assert rendered.chunks == (
        "&lt;!channel&gt; &lt;@U123&gt;\n"
        "Facts\n"
        "• safe &lt;b&gt;text&lt;/b&gt;\n"
        "  Source: <https://example.com|safe>\n"
        "Research evidence, not financial advice.",
    )


@pytest.mark.parametrize(
    ("payload", "forbidden"),
    [
        ("notify <!here> and <@U123>", ("<!here>", "<@U123>")),
        ("action <!date^1^{date}|today>", ("<!date",)),
        ("token " + "xox" + "b-1234567890-secretvalue", ("xox" + "b-", "secretvalue")),
        ("key " + "s" + "k-proj-abcdefghijklmnop", ("s" + "k-proj-", "abcdefghijklmnop")),
        ("Bearer abcdefghijklmnop", ("abcdefghijklmnop",)),
        (
            "postgresql://demo:do-not-leak@example.com/leo",
            ("do-not-leak", "example.com/leo"),
        ),
        ("left\u061cright\u200f\x7f\x85", ("\u061c", "\u200f", "\x7f", "\x85")),
    ],
)
def test_renderer_adversarial_matrix_neutralizes_actions_secrets_and_controls(
    payload: str,
    forbidden: tuple[str, ...],
) -> None:
    first = render_slack_text(payload, max_chars=24)
    second = render_slack_text(payload, max_chars=24)

    assert first == second
    rendered = "".join(first.chunks)
    assert all(value not in rendered for value in forbidden)
    assert all(0 < len(chunk) <= 24 for chunk in first.chunks)


def test_secret_is_redacted_atomically_before_chunking() -> None:
    rendered = render_slack_text(
        "prefix " + "xap" + "p-1234567890-sensitive suffix",
        max_chars=12,
    )

    assert "xap" + "p-" not in "".join(rendered.chunks)
    assert "[redacted credential]" in "".join(rendered.chunks)


def test_renderer_never_splits_or_emits_oversized_source_markup() -> None:
    rendered = render_verified_result(
        SlackVerifiedResult(
            run_id="run-long-source",
            answer="Grounded fact.",
            claims=(
                SlackClaim(
                    statement="Grounded fact.",
                    sources=(
                        SlackSource(
                            label="too long",
                            url=f"https://example.com/{'x' * 3000}",
                        ),
                        SlackSource(label="fits", url="https://example.com/ok"),
                    ),
                ),
            ),
        ),
        max_chars=80,
    )

    assert "too long" not in "".join(rendered.chunks)
    assert "<https://example.com/ok|fits>" in "".join(rendered.chunks)
    assert all(chunk.count("<") == chunk.count(">") for chunk in rendered.chunks)


def test_research_renderer_separates_content_without_internal_id() -> None:
    rendered = render_research_result(
        SlackResearchResult(
            run_id="run-research",
            facts=(SlackClaim(statement="Primary fact."),),
            inferences=("Bounded inference.",),
            affected_assumption="Demand remains durable.",
            uncertainty="One provider was unavailable.",
        )
    )

    payload = "".join(rendered.chunks)
    assert "Facts\n• Primary fact." in payload
    assert "Inferences\n• Bounded inference." in payload
    assert "Affected assumption: Demand remains durable." in payload
    assert "Uncertainty: One provider was unavailable." in payload
    assert payload.endswith("Research evidence, not financial advice.")
    assert "run-research" not in payload
    assert "Run:" not in payload


def test_structured_renderer_removes_repeated_internal_id_from_verified_content() -> None:
    run_id = "run-12345678-1234-4abc-8def-1234567890ab"
    rendered = render_verified_result(
        SlackVerifiedResult(
            run_id=run_id,
            answer=f"Run: {run_id}\nI can still explain this request: {run_id}.",
            research_disclaimer_required=False,
        )
    )

    payload = "".join(rendered.chunks)
    assert payload == "I can still explain this request: this request."
    assert run_id not in payload
    assert "Run:" not in payload


def test_plain_slack_renderer_suppresses_production_shaped_internal_run_ids() -> None:
    run_id = "run-12345678-1234-4abc-8def-1234567890ab"

    payload = "".join(render_slack_text(f"Internal reference {run_id}").chunks)

    assert payload == "Internal reference this request"
    assert run_id not in payload


@pytest.mark.parametrize(
    "render",
    [
        lambda: render_terminal_result(SlackTerminalResult(run_id=" ", status=RunStatus.FAILED)),
        lambda: render_verified_result(SlackVerifiedResult(run_id="", answer="Answer.")),
        lambda: render_research_result(SlackResearchResult(run_id="\t")),
    ],
)
def test_structured_renderers_require_internal_run_id_for_correlation(
    render: Callable[[], object],
) -> None:
    with pytest.raises(ValueError, match="requires a run ID"):
        render()


@pytest.mark.parametrize("value", ["", "\x00\x01"])
def test_renderer_rejects_empty_payload(value: str) -> None:
    with pytest.raises(ValueError):
        render_slack_text(value)


def test_renderer_seeded_unicode_fuzz_is_bounded_deterministic_and_inert() -> None:
    generator = random.Random(20260822)
    alphabet = (
        "abc XYZ 012\n\t<>&@!|`*_~"
        "\u061c\u200e\u200f\u202a\u202e\u2066\u2069"
        "\x00\x01\x1f\x7f\x85"
        "\u00e9\u03bb\u4e2d\U0001f680"
    )
    for _ in range(256):
        payload = "seed " + "".join(
            generator.choice(alphabet) for _ in range(generator.randint(1, 8_000))
        )
        limit = generator.randint(1, 256)
        first = render_slack_text(payload, max_chars=limit)
        second = render_slack_text(payload, max_chars=limit)
        joined = "".join(first.chunks)

        assert first == second
        assert all(first.chunks)
        assert all(len(chunk) <= limit for chunk in first.chunks)
        assert not any(
            character in joined for character in "\u061c\u200e\u200f\u202a\u202e\u2066\u2069"
        )
        assert not any(character in joined for character in "\x00\x01\x1f\x7f\x85")
        assert "<" not in joined and ">" not in joined


def test_verified_renderer_handles_very_long_answer_without_splitting_source_markup() -> None:
    answer = (
        "A long verified conversational answer with unicode λ and U0001f680.\n" * 2_000
    ).strip()
    result = SlackVerifiedResult(
        run_id="run-long-conversation",
        answer=answer,
        claims=(
            SlackClaim(
                statement="One bounded supported statement.",
                sources=(SlackSource(label="primary", url="https://example.com/source"),),
            ),
        ),
    )

    first = render_verified_result(result)
    second = render_verified_result(result)
    payload = "".join(first.chunks)

    assert first == second
    assert len(first.chunks) > 20
    assert all(0 < len(chunk) <= 3_500 for chunk in first.chunks)
    assert payload.count("<https://example.com/source|primary>") == 1
    assert all(chunk.count("<") == chunk.count(">") for chunk in first.chunks)
    assert payload.endswith("Research evidence, not financial advice.")
    assert "run-long-conversation" not in payload
