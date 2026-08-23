"""Ordinary questions must reach an answer, not a canned refusal.

Leo's regression history runs one direction: each tightening of a gate improved
a tracked safety number, and no metric ever objected when a real question went
unanswered. The reported production failure -- "@Leo what's the year end
forecast for GOOG?" answered with "I'm missing a reliable source needed for that
answer" -- was the end state of that gradient, not a one-off bug.

These tests are the counterweight. They assert the property no previous test
asserted: that ordinary questions a person actually types into Slack have a
route to a substantive reply. They are deliberately about *reachability*, not
about answer content -- correctness of the content stays the verifier's job, and
the anti-fabrication guards elsewhere in this suite stay untouched.
"""

from __future__ import annotations

import pytest

from leo.harness.deliberation import (
    ElasticDeliberationPolicy,
    _is_non_terminal_deferral,
    answer_is_substantive,
)
from leo.live import (
    _is_self_contained_conversational_turn,
    _primary_source_search_query,
    _requires_external_evidence,
    _research_is_available,
)

# Questions a person actually asks Leo. None of these may dead-end.
RESEARCH_QUESTIONS = (
    "what's the year end forecast for GOOG?",
    "should I buy NVDA at these levels?",
    "what's the target price for TSLA?",
    "how did Microsoft's cloud margins trend last quarter?",
    "what's the outlook for semiconductor demand?",
    "is the Fed likely to cut rates in December?",
    "compare AMD and NVDA on gross margin",
    "what happened to Intel's foundry business?",
    "give me a read on the AI infrastructure trade",
    "where do analysts see AAPL by year end?",
)

# Turns that genuinely need no external lookup. Advertising web search for these
# is waste, so the split has to hold in both directions to be worth anything.
SELF_CONTAINED_TURNS = (
    "hi",
    "thanks!",
    "good morning",
    "what did we call the demo?",
    "what did you say about margins?",
    "summarize this",
    "rewrite the above more concisely",
)


@pytest.mark.parametrize("objective", RESEARCH_QUESTIONS)
def test_ordinary_questions_keep_research_available(objective: str) -> None:
    """No keyword list decides whether a real question deserves a lookup.

    `_requires_external_evidence` is a high-confidence *yes* signal built from
    literal token sets. Treating its absence as a *no* is what broke the GOOG
    turn: "forecast" is in no market vocabulary and "year end" is in no freshness
    vocabulary, so the turn was routed as needing nothing at all.
    """

    assert _research_is_available(objective, ()) is True


@pytest.mark.parametrize("objective", SELF_CONTAINED_TURNS)
def test_self_contained_turns_do_not_pull_in_research(objective: str) -> None:
    assert _is_self_contained_conversational_turn(objective) is True
    assert _research_is_available(objective, ()) is False


@pytest.mark.parametrize("objective", RESEARCH_QUESTIONS)
def test_ordinary_questions_never_route_into_a_dead_end(objective: str) -> None:
    """An envelope that forbids tools while demanding clarification cannot answer.

    That combination is reserved for genuinely unresolvable input. Reaching it
    from a specific, well-formed question means the run is over before the model
    is consulted.
    """

    envelope = ElasticDeliberationPolicy().assess(
        objective,
        available_tool_names=frozenset({"agent.delegate_research", "agent.execute_research_plan"}),
    )
    assert not (envelope.hard_disable_tools and envelope.hard_require_clarification)
    assert envelope.maximum_depth >= 1


def test_search_queries_are_the_users_question() -> None:
    """Discovery is not steered away from the subject by a fixed suffix.

    Every query used to carry " official documentation primary source", which
    turned a finance question into a developer-docs lookup that returned nothing
    usable -- and the empty result then read as "no reliable source exists".
    """

    objective = "what's the year end forecast for GOOG?"
    assert _primary_source_search_query(objective) == objective


def test_confident_market_signal_still_forces_a_read() -> None:
    """Widening availability must not weaken the obligation where it was real."""

    assert _requires_external_evidence("what is the current price of NVDA stock?", ()) is True


class TestHedgedAnswersSurvive:
    """A useful answer that admits uncertainty is an answer, not a deferral.

    Rewriting these was doubly harmful: it discarded good work, and it applied
    steady pressure toward unhedged, overconfident prose -- the harness punished
    exactly the epistemic honesty the product needs.
    """

    GOOD_HEDGED = (
        "I don't have current market data for GOOG, but Wall Street consensus "
        "year-end price targets cluster around $210-$230, implying modest upside "
        "from here.",
        "I can't verify live quotes right now. Analyst year-end targets for TSLA "
        "average about $250, with a range of $120 to $400 reflecting unusually "
        "wide disagreement on the robotaxi timeline.",
    )

    EMPTY_DEFERRALS = (
        "I don't have enough information.",
        "I do not have reliable sources for that.",
        "I'm missing a reliable source needed for that answer.",
    )

    @pytest.mark.parametrize("answer", GOOD_HEDGED)
    def test_substantive_hedged_answers_are_delivered(self, answer: str) -> None:
        assert answer_is_substantive(answer) is True
        assert _is_non_terminal_deferral(answer) is False

    @pytest.mark.parametrize("answer", EMPTY_DEFERRALS)
    def test_contentless_deferrals_are_still_caught(self, answer: str) -> None:
        assert _is_non_terminal_deferral(answer) is True

    def test_a_promise_of_future_work_is_a_deferral_at_any_length(self) -> None:
        """Length is not a licence to promise work Leo will not actually do."""

        promise = (
            "GOOG closed around $185 last I knew, and consensus year-end targets "
            "sit near $210. I'll pull the latest analyst revisions and confirm the "
            "exact figures for you."
        )
        assert answer_is_substantive(promise) is True
        assert _is_non_terminal_deferral(promise) is True
