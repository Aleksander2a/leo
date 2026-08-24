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

from pathlib import Path

import pytest

from leo.harness.deliberation import (
    ElasticDeliberationPolicy,
    _is_non_terminal_deferral,
    answer_is_substantive,
)
from leo.harness.terminal_quality import contains_future_action_promise
from leo.harness.verifier import presents_unretrieved_data
from leo.live import (
    _is_self_contained_conversational_turn,
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


def test_no_harness_authored_search_query_path_remains() -> None:
    """The model writes its own query; the harness no longer writes one for it.

    Queries used to be built by the harness from the raw objective (once with a
    fixed " official documentation primary source" suffix, which turned a finance
    question into a developer-docs lookup). Both the suffix and the whole
    harness-authored query path are gone: the model composes the query as part of
    the tool call it chose, which is the only way it can refine one after a poor
    result.
    """

    source = (Path(__file__).resolve().parents[1] / "src" / "leo" / "live.py").read_text(
        encoding="utf-8"
    )
    assert "_primary_source_search_query" not in source
    assert "official documentation primary source" not in source


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


def test_present_progressive_promises_are_caught_like_future_tense() -> None:
    """A promise in flight is still a promise the run cannot keep.

    Leo shipped "...and I'm pulling those now to give you a more grounded read"
    to Slack as a *final* answer, twice in a row. Nothing was in flight: the turn
    was ending, so the user was told to wait for a follow-up that could not come.
    Only "I'll pull" and "let me check" were recognized before.
    """

    promises = (
        "I'm pulling those now to give you a more grounded read.",
        "I am gathering the earnings data.",
        "I'm looking that up.",
        "I'm looking it up now.",
        "I'm checking the recent filings.",
        "I'm currently researching that.",
        "I'm about to query the provider.",
    )
    for answer in promises:
        assert contains_future_action_promise(answer) is True, answer


def test_capability_and_past_tense_statements_are_not_promises() -> None:
    """The guard must not punish describing what Leo can do, or already did."""

    allowed = (
        "I can pull those figures if you want me to.",
        "I pulled the filings already, and they show a 12% rise.",
        "I'm looking forward to helping with the next one.",
        "Looking at the data, GOOG margins are stable.",
        "Analysts are checking the numbers this quarter.",
    )
    for answer in allowed:
        assert contains_future_action_promise(answer) is False, answer


def test_presenting_live_data_requires_a_retrieved_observation() -> None:
    """Framing beats phrasing: claim retrieved data, hold a retrieved observation.

    Chasing each new promissory phrasing with its own regex is a losing game.
    This is the structural rule underneath them: an answer that presents itself
    as showing live or current data is false when the run retrieved nothing.
    """

    claiming = (
        "I don't have a target, but here's what the current data shows.",
        "Here it is, built from live data rather than a made-up number.",
        "Based on the latest figures, GOOG looks strong.",
    )
    for answer in claiming:
        assert presents_unretrieved_data(answer, ()) is True, answer


def test_honest_general_knowledge_answers_are_not_blocked() -> None:
    """The structural rule must not push Leo away from honest sourcing."""

    honest = (
        "Analyst consensus is around $215, from general knowledge as of my cutoff.",
        "I could not pull live figures in this run, so these are approximate ranges.",
        "GOOG year-end consensus: ~$215 mean target, range $190-$250.",
    )
    for answer in honest:
        assert presents_unretrieved_data(answer, ()) is False, answer
