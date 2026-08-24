from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.demo import run_conversation_smoke
from leo.harness.deliberation import ElasticDeliberationGateway, ElasticDeliberationPolicy
from leo.harness.models import (
    BudgetLimits,
    CompletionProposal,
    EventType,
    ModelRequest,
    ModelTurnResult,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    VerificationOutcome,
    VerifierStatus,
)
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="answer-quality-org", strategy_id="default-domain")
LIVE_INCOMPLETE_ANSWER = (
    "Here are a few dividend-focused ideas ... some that are steadier, higher-yield "
)
LIVE_FUTURE_PROMISE = (
    "Happy to help — let me pull a few current quotes and recent dividend data, and then I "
    "can narrow in on the strongest candidates."
)
LIVE_FALSE_ACTION_ANSWER = (
    "Here's a mix of dividend-focused ideas. I pulled current quotes for a few names. "
    "Let me grab live data first."
)
LIVE_PREAMBLE_ONLY_ANSWER = (
    "Here are a couple of buckets to consider, based on current market data I pulled for "
    "well-known dividend payers. (Note: I don't have your risk tolerance or time horizon yet, "
    "so treat these as starting points, not personalized advice.)"
)
RECOMMENDATION_OBJECTIVE = (
    "Some dividend based stocks with growth potential over time, and some safe bets with high "
    "dividends."
)


def _bundle(
    observation: Observation | None = None,
    *,
    objective: str = "Provide a useful answer.",
) -> RunBundle:
    thread = Thread(
        id="answer-thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="answer-thread"),
    )
    task = Task(
        id="answer-task",
        thread_id=thread.id,
        scope=SCOPE,
        objective=objective,
    )
    run = Run(id="answer-run", task_id=task.id, scope=SCOPE)
    return RunBundle(
        thread=thread,
        task=task,
        run=run,
        observations=() if observation is None else (observation,),
    )


def _verify(
    answer: str,
    observation: Observation | None = None,
    *,
    objective: str = "Provide a useful answer.",
) -> VerificationOutcome:
    return DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        require_source_claim=False,
    ).verify(CompletionProposal(answer=answer), _bundle(observation, objective=objective))


@pytest.mark.parametrize(
    ("answer", "feedback_fragment"),
    [
        (LIVE_INCOMPLETE_ANSWER, "without trailing whitespace"),
        ("A complete-looking prefix\n", "without trailing whitespace"),
        ("The main trade-offs are:", "punctuation that requires continuation"),
        ("Prefer resilient cash flows,", "punctuation that requires continuation"),
        ("Balance income and", "trailing connective 'and'"),
        ("The alternatives include growth or", "trailing connective 'or'"),
        ("One useful comparison is for example", "trailing connective 'for example'"),
        ("Choose the strongest option --", "punctuation that requires continuation"),
        (LIVE_FUTURE_PROMISE, "do not complete with a promise of future work"),
        ("Let me check the latest filing before I answer.", "promise of future work"),
        ("I'll research the company first.", "promise of future work"),
        ("I will look up the quote and get back to you.", "promise of future work"),
        ("After you reply, then I can narrow the list.", "promise of future work"),
        ("I\u2019ll quickly pull current prices.", "promise of future work"),
        (LIVE_FALSE_ACTION_ANSWER, "promise of future work"),
        ("Let me grab live data first.", "promise of future work"),
        (
            "I pulled current quotes before preparing this answer.",
            "matching retrieved observation",
        ),
        ("I checked the latest filings.", "matching retrieved observation"),
        ("I researched the company before replying.", "matching retrieved observation"),
    ],
)
def test_clear_terminal_fragments_are_rejected_with_actionable_feedback(
    answer: str,
    feedback_fragment: str,
) -> None:
    outcome = _verify(answer)

    check = next(item for item in outcome.result.checks if item.name == "answer_completeness")
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.result.retryable is True
    assert outcome.completion is None
    assert check.passed is False
    assert feedback_fragment in check.detail.lower()


@pytest.mark.parametrize(
    "answer",
    [
        "Yes",
        "It depends on your time horizon",
        "Which risk level do you prefer?",
        "- Dividend growers\n- High-yield utilities",
        "1. Preserve capital\n2. Add income",
        "```python\nprint('complete')\n```",
        "`SELECT 1;`",
        "SELECT 1;",
        "const answer = 42;",
        "https://example.com/research/",
        "http://example.com/research/",
        "42",
        "$1,234.50",
        "NVDA is quoted at 181.25 USD.",
        "C++",
        "Log in",
        'The phrase "let me pull current quotes" is a future-work promise.',
        "The assistant said: 'I'll check that.'",
        "Avoid saying `then I can narrow in`; answer now.",
        "I can check current quotes with the market tool.",
        "> I'll research that later.\nThat quoted line is the problem.",
        'The sentence "I pulled current quotes" is an unsupported-action example.',
        "The assistant said: 'I researched the company.'",
        "I can grab live data with an eligible read tool.",
        "I checked the arithmetic and the result is 42.",
        "I grabbed lunch before this conversation.",
    ],
)
def test_concise_and_structured_complete_answers_are_not_rejected(answer: str) -> None:
    outcome = _verify(answer)

    check = next(item for item in outcome.result.checks if item.name == "answer_completeness")
    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert check.passed is True


REWRITE_OBJECTIVE = (
    "Rewrite this in a friendly tone under 45 words: The deployment is delayed because the "
    "migration review found two safety issues. We will share a revised date after the fixes "
    "are verified."
)
REWRITTEN_ANSWER = (
    "Thanks for your patience. The deployment is delayed while we address two safety issues "
    "found during migration review. We'll share a revised date once the fixes are verified."
)
NAMES_OBJECTIVE = (
    "Give me exactly three friendly code names for a project that helps teams find reliable "
    "information faster. Names only, one per line."
)
NAMES_ANSWER = "Beacon\nVerity\nSignalPath"
BULLETS_OBJECTIVE = (
    "Explain the difference between a limit order and a market order in exactly four concise "
    "bullets."
)
BULLETS_ANSWER = (
    "- A limit order sets the worst price you will accept.\n"
    "- A market order prioritizes immediate execution.\n"
    "- Limit orders may remain unfilled.\n"
    "- Market orders can fill at an unexpected price."
)
GENERIC_CLARIFICATION = "Which specific target, options, or missing details should I use?"


@pytest.mark.parametrize(
    ("objective", "answer", "expected_pass"),
    [
        (REWRITE_OBJECTIVE, GENERIC_CLARIFICATION, False),
        (REWRITE_OBJECTIVE, REWRITTEN_ANSWER, True),
        (NAMES_OBJECTIVE, GENERIC_CLARIFICATION, False),
        (NAMES_OBJECTIVE, NAMES_ANSWER, True),
        (NAMES_OBJECTIVE, "- Beacon\n- Verity\n- SignalPath", False),
        (NAMES_OBJECTIVE, "Beacon, Verity, SignalPath", False),
        (
            BULLETS_OBJECTIVE,
            (
                "1. A limit order controls price. 2. A market order prioritizes speed. "
                "3. Limit orders may not fill. 4. Market orders can slip."
            ),
            False,
        ),
        (BULLETS_OBJECTIVE, BULLETS_ANSWER, True),
        (
            "Explain limit orders in exactly two concise sentences.",
            (
                "A limit order sets a maximum purchase price, such as $10.50. "
                "In the U.S., execution is not guaranteed."
            ),
            True,
        ),
        (
            "Explain limit orders in exactly two concise sentences.",
            "A limit order controls price but may not execute.",
            False,
        ),
        ("Reply in under 5 words.", "One two three four", True),
        ("Reply in under 5 words.", "One two three four five", False),
        ("Reply in at most 5 words.", "One two three four five", True),
        ("Rewrite this in two sentences.", "What text should I rewrite?", True),
        ("Compare these in exactly two sentences.", "Which options should I compare?", True),
    ],
)
def test_explicit_answer_formats_are_enforced_without_becoming_a_style_grader(
    objective: str,
    answer: str,
    expected_pass: bool,
) -> None:
    outcome = _verify(answer, objective=objective)

    check = next(item for item in outcome.result.checks if item.name == "answer_format")
    assert check.passed is expected_pass
    assert (outcome.result.status is VerifierStatus.PASS) is expected_pass
    assert outcome.result.retryable is (not expected_pass)


def _observation(kind: str) -> Observation:
    return Observation(
        id=f"obs-{kind}",
        scope=SCOPE,
        run_id="answer-run",
        tool_call_id=f"call-{kind}",
        kind=kind,
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="fixture", reference=f"fixture:{kind}"),
        observed_at=NOW,
        raw_hash="a" * 64,
    )


def test_completed_quote_action_requires_matching_quote_observation() -> None:
    answer = "I pulled current quotes before preparing this answer."

    wrong_kind = _verify(answer, _observation("web.search_exa"))
    matched = _verify(answer, _observation("market.get_quote_finnhub"))

    assert wrong_kind.result.status is VerifierStatus.FAIL
    assert matched.result.status is VerifierStatus.PASS
    assert matched.completion is not None


def test_completed_research_action_accepts_matching_external_read_observation() -> None:
    outcome = _verify(
        "I researched the company before preparing this answer.",
        _observation("web.search_exa"),
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None


def test_completed_filing_action_requires_a_filing_observation_not_any_read() -> None:
    answer = "I checked the latest filings before preparing this answer."

    wrong_kind = _verify(answer, _observation("market.get_quote"))
    matched = _verify(answer, _observation("sec.get_recent_filings"))

    assert wrong_kind.result.status is VerifierStatus.FAIL
    assert matched.result.status is VerifierStatus.PASS


def test_live_preamble_only_answer_fails_both_completeness_and_sufficiency() -> None:
    """A captured live answer that is both unsourced and contentless.

    It claims "current market data I pulled" with no retrieved observation *and*
    stops after the list introduction, so both gates must fire. It is kept out of
    the sufficiency-only matrix below because it is not a single-defect example.
    """

    outcome = _verify(LIVE_PREAMBLE_ONLY_ANSWER, objective=RECOMMENDATION_OBJECTIVE)

    completeness = next(
        item for item in outcome.result.checks if item.name == "answer_completeness"
    )
    sufficiency = next(item for item in outcome.result.checks if item.name == "answer_sufficiency")
    assert completeness.passed is False
    assert sufficiency.passed is False
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.result.retryable is True


@pytest.mark.parametrize(
    "answer",
    [
        "Here are three options to consider.",
        "Some examples include the following. This is not financial advice.",
        "There are several alternatives worth considering. It depends on your goals.",
        "I would consider the following options. Does that help?",
        "Here's a mix to consider. This is not personalized advice.",
    ],
)
def test_requested_output_preamble_without_items_fails_sufficiency(answer: str) -> None:
    outcome = _verify(answer, objective=RECOMMENDATION_OBJECTIVE)

    completeness = next(
        item for item in outcome.result.checks if item.name == "answer_completeness"
    )
    sufficiency = next(item for item in outcome.result.checks if item.name == "answer_sufficiency")
    assert completeness.passed is True
    assert sufficiency.passed is False
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.result.retryable is True
    assert "do not stop after a list introduction" in sufficiency.detail.lower()


UK_DIVIDEND_SHORTLIST_OBJECTIVE = (
    "Give me a concise shortlist of three UK dividend-growth stocks, with one key trade-off "
    "for each, and include a brief not-financial-advice caveat."
)
UK_DIVIDEND_SHORTLIST_PREAMBLE_ONLY_ANSWER = (
    "Here's a concise shortlist of three UK dividend-growth stocks, each with one key "
    "trade-off. (Not financial advice — this is general information, not a "
    "recommendation; do your own research and consider your own circumstances before "
    "investing.)"
)
UK_DIVIDEND_SHORTLIST_REPAIRED_ANSWER = (
    "Here's a concise shortlist of three UK dividend-growth stocks, each with one key "
    "trade-off: Unilever offers defensive income but slower growth; National Grid offers a "
    "high yield but is rate-sensitive; Diageo offers dividend consistency but faces "
    "consumer-spending risk. Not financial advice — do your own research."
)


def test_shortlist_preamble_without_items_fails_sufficiency() -> None:
    """Regression test for the 2026-08-23 Railway smoke-test incident.

    A real Slack reply consisted only of this announcement sentence plus the research
    disclaimer, with no actual stocks or trade-offs. ``_requests_concrete_options`` missed
    "shortlist" (compound word, no ``\\blist\\b`` boundary) and ``_is_output_preamble_only``
    didn't recognize this phrasing as an introduction, so the empty answer passed
    verification and was delivered to the user as-is.
    """

    outcome = _verify(
        UK_DIVIDEND_SHORTLIST_PREAMBLE_ONLY_ANSWER,
        objective=UK_DIVIDEND_SHORTLIST_OBJECTIVE,
    )

    sufficiency = next(item for item in outcome.result.checks if item.name == "answer_sufficiency")
    assert sufficiency.passed is False
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.result.retryable is True


def test_shortlist_answer_with_real_content_passes_sufficiency() -> None:
    outcome = _verify(
        UK_DIVIDEND_SHORTLIST_REPAIRED_ANSWER,
        objective=UK_DIVIDEND_SHORTLIST_OBJECTIVE,
    )

    sufficiency = next(item for item in outcome.result.checks if item.name == "answer_sufficiency")
    assert sufficiency.passed is True
    assert outcome.result.status is VerifierStatus.PASS


@pytest.mark.asyncio
async def test_shortlist_preamble_only_answer_retries_into_concrete_recommendations() -> None:
    delegate = _RepairingGateway(
        (UK_DIVIDEND_SHORTLIST_PREAMBLE_ONLY_ANSWER, UK_DIVIDEND_SHORTLIST_REPAIRED_ANSWER)
    )
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(UK_DIVIDEND_SHORTLIST_OBJECTIVE),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=UK_DIVIDEND_SHORTLIST_OBJECTIVE,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == UK_DIVIDEND_SHORTLIST_REPAIRED_ANSWER
    assert len(delegate.requests) == 2


@pytest.mark.parametrize(
    "answer",
    [
        "Which market, risk tolerance, and time horizon should I use?",
        "I need some details. Which market and time horizon should I use?",
        "JNJ and KO.",
        "Here are two options to consider: dividend growers such as JNJ, and utilities such "
        "as DUK.",
        "Here are two options based on different risk profiles: JNJ and KO.",
        "Here's a mix to consider: JNJ for stability and DUK for higher yield.",
        "JNJ emphasizes stability, while DUK offers a higher starting yield.",
        "Two candidates are JNJ and KO. This is not financial advice.",
        "NVDA is quoted at 181.25 USD.",
    ],
)
def test_concrete_answers_clarifications_and_canonical_completions_pass_sufficiency(
    answer: str,
) -> None:
    outcome = _verify(answer, objective=RECOMMENDATION_OBJECTIVE)

    sufficiency = next(item for item in outcome.result.checks if item.name == "answer_sufficiency")
    assert sufficiency.passed is True
    assert outcome.result.status is VerifierStatus.PASS


class _RepairingGateway:
    def __init__(self, answers: tuple[str, ...]) -> None:
        self._answers = answers
        self.requests: list[ModelRequest] = []

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.requests.append(request)
        answer = self._answers[len(self.requests) - 1]
        return ModelTurnResult(
            decision=CompletionProposal(answer=answer),
            provider="openrouter",
            model="live-shaped-fixture",
            finish_reason="stop",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("objective", "first_answer", "repaired_answer", "feedback_fragment"),
    [
        (
            REWRITE_OBJECTIVE,
            GENERIC_CLARIFICATION,
            REWRITTEN_ANSWER,
            "already supplies the target",
        ),
        (
            NAMES_OBJECTIVE,
            GENERIC_CLARIFICATION,
            NAMES_ANSWER,
            "already supplies the target",
        ),
        (
            BULLETS_OBJECTIVE,
            (
                "1. A limit order controls price. 2. A market order prioritizes speed. "
                "3. Limit orders may not fill. 4. Market orders can slip."
            ),
            BULLETS_ANSWER,
            "exactly 4 bullet lines",
        ),
    ],
)
async def test_live_shaped_short_prompt_format_failure_repairs_in_coordinator_loop(
    objective: str,
    first_answer: str,
    repaired_answer: str,
    feedback_fragment: str,
) -> None:
    delegate = _RepairingGateway((first_answer, repaired_answer))
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(objective),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=objective,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == repaired_answer
    assert result.run.usage.model_calls == 2
    assert any(
        feedback_fragment in feedback.lower() for feedback in delegate.requests[1].verifier_feedback
    )
    failed = [event for event in result.events if event.type is EventType.VERIFICATION_FAILED]
    assert len(failed) == 1
    assert any(
        isinstance(check, dict)
        and check.get("name") == "answer_format"
        and check.get("passed") is False
        for check in failed[0].payload["checks"]
    )


@pytest.mark.asyncio
async def test_repeated_names_format_failure_is_repaired_without_another_model_call() -> None:
    delegate = _RepairingGateway(
        ("Here are three friendly code names:\n1. Beacon\n2. Verity\n3. SignalPath",) * 3
    )
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(NAMES_OBJECTIVE),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=NAMES_OBJECTIVE,
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == NAMES_ANSWER
    assert result.run.usage.model_calls == 2


@pytest.mark.asyncio
async def test_names_format_recovery_handles_different_invalid_retries() -> None:
    delegate = _RepairingGateway(
        (
            "Here are three friendly code names:\n1. Beacon\n2. Verity\n3. SignalPath",
            "Beacon, Verity, SignalPath",
            "- Beacon\n- Verity\n- SignalPath",
        )
    )
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(NAMES_OBJECTIVE),
        max_no_progress_turns=2,
    )

    result = await run_conversation_smoke(
        model=model,
        objective=NAMES_OBJECTIVE,
        limits=BudgetLimits(max_iterations=5, max_model_calls=5, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == NAMES_ANSWER
    assert result.run.usage.model_calls == 4


@pytest.mark.asyncio
async def test_names_format_recovery_never_fabricates_the_answer() -> None:
    """A format mismatch must not license the harness to author the content.

    This previously asserted the opposite: that an unrepairable answer was
    replaced with a hardcoded list ("Signal Harbor", "Trust Compass", ...) and
    delivered as Leo's own words. That is a fabrication path inside the component
    whose entire purpose is preventing fabrication. Whatever Leo finally says
    here must be traceable to the model, never invented by the harness.
    """

    delegate = _RepairingGateway(("I can help with that, but first let me think.",) * 2)
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(NAMES_OBJECTIVE),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=NAMES_OBJECTIVE,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    delivered = result.run.final_output or ""
    for invented in ("Signal Harbor", "Trust Compass", "Clarity Trail"):
        assert invented not in delivered


@pytest.mark.asyncio
async def test_live_shaped_stop_completion_retries_and_repairs_in_normal_loop() -> None:
    objective = (
        "Some dividend based stocks with growth potential over time, some safe bets with "
        "high dividends"
    )
    repaired = (
        "A useful starting screen is durable cash flow, a sustainable payout ratio, and "
        "a history of dividend growth."
    )
    delegate = _RepairingGateway((LIVE_INCOMPLETE_ANSWER, repaired))
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(objective),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=objective,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == repaired
    assert result.run.usage.model_calls == 2
    assert len(delegate.requests) == 2
    assert any(
        "without trailing whitespace" in feedback.lower()
        for feedback in delegate.requests[1].verifier_feedback
    )
    failed = [event for event in result.events if event.type is EventType.VERIFICATION_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["retryable"] is True


@pytest.mark.asyncio
async def test_live_future_promise_retries_into_one_concrete_clarification() -> None:
    objective = "What are some interesting investing opportunities currently?"
    clarification = (
        "Which market or asset class, risk tolerance, and time horizon should I focus on?"
    )
    delegate = _RepairingGateway((LIVE_FUTURE_PROMISE, clarification))

    result = await run_conversation_smoke(
        model=delegate,
        objective=objective,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == clarification
    assert clarification.count("?") == 1
    assert len(delegate.requests) == 2
    assert any(
        "call an eligible read tool now" in feedback.lower()
        and "one concrete input-seeking question" in feedback.lower()
        for feedback in delegate.requests[1].verifier_feedback
    )


@pytest.mark.asyncio
async def test_unobserved_completed_quote_claim_retries_in_normal_loop() -> None:
    first = "Here's a preliminary mix. I pulled current quotes for a few names."
    repaired = "Which ticker or market should I check?"
    delegate = _RepairingGateway((first, repaired))

    result = await run_conversation_smoke(
        model=delegate,
        objective="Show me a few current dividend opportunities.",
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == repaired
    assert len(delegate.requests) == 2
    assert any(
        "without a matching retrieved observation" in feedback.lower()
        for feedback in delegate.requests[1].verifier_feedback
    )


@pytest.mark.asyncio
async def test_live_preamble_only_answer_retries_into_concrete_recommendations() -> None:
    repaired = (
        "Two concrete buckets are dividend growers such as JNJ, and higher-yield utilities "
        "such as DUK."
    )
    delegate = _RepairingGateway((LIVE_PREAMBLE_ONLY_ANSWER, repaired))
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(RECOMMENDATION_OBJECTIVE),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=RECOMMENDATION_OBJECTIVE,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == repaired
    assert len(delegate.requests) == 2
    assert any(
        "requested concrete recommendations" in feedback.lower()
        for feedback in delegate.requests[1].verifier_feedback
    )
    failed = [event for event in result.events if event.type is EventType.VERIFICATION_FAILED]
    assert len(failed) == 1
    failed_checks = failed[0].payload["checks"]
    assert isinstance(failed_checks, list)
    assert any(
        isinstance(check, dict)
        and check.get("name") == "answer_sufficiency"
        and check.get("passed") is False
        for check in failed_checks
    )


@pytest.mark.asyncio
async def test_repeated_preamble_only_answer_delivers_best_effort_fallback() -> None:
    """Once the bounded repair loop gives up, a claim-free answer is still delivered.

    The model never produces real content here, so the run cannot succeed through
    ordinary verification. But failing the user with no answer at all is worse than
    delivering the last (imperfect) attempt: it carries no unverified claims, so it
    is safe to hand back as a best-effort completion instead of a terminal error.
    """

    delegate = _RepairingGateway((LIVE_PREAMBLE_ONLY_ANSWER, LIVE_PREAMBLE_ONLY_ANSWER))
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(RECOMMENDATION_OBJECTIVE),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=RECOMMENDATION_OBJECTIVE,
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=0),
    )

    assert len(delegate.requests) == 2
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == LIVE_PREAMBLE_ONLY_ANSWER
    assert sum(event.type is EventType.VERIFICATION_FAILED for event in result.events) == 1
    fallback = next(
        check
        for event in result.events
        if event.type is EventType.VERIFICATION_PASSED
        for check in event.payload["checks"]
        if check["name"] == "best_effort_fallback"
    )
    assert fallback["passed"] is True


@pytest.mark.asyncio
async def test_repeated_incomplete_repair_delivers_best_effort_fallback() -> None:
    objective = "Suggest a few dividend-focused ideas."
    delegate = _RepairingGateway((LIVE_INCOMPLETE_ANSWER, LIVE_INCOMPLETE_ANSWER))
    model = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(objective),
    )

    result = await run_conversation_smoke(
        model=model,
        objective=objective,
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=0),
    )

    assert len(delegate.requests) == 2
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == LIVE_INCOMPLETE_ANSWER.strip()
    assert sum(event.type is EventType.VERIFICATION_FAILED for event in result.events) == 1
