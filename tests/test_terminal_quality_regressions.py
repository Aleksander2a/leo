"""Regression tests for narrow-regex false positives in terminal_quality.py.

These guard two production false positives in the deterministic completion
verifier's terminal-answer checks:

* ``contains_future_action_promise`` misfired on ``I'll use X`` / ``I will use X``
  because the generic verb ``use`` shared a trailing-verb alternation with genuine
  forward-looking research verbs (pull, grab, fetch, ...). ``I'll use dividend
  cover as the key risk metric`` is a complete analytical sentence, not a promise
  of unfinished work.
* ``completed_research_action_claim`` misfired on the ordinary not-financial-advice
  disclaimer sentence because the word "financial" inside "financial advice"
  satisfied the classifier's required-noun alternation (``financials?``).

Each false-positive case is paired with a genuine case that must keep failing
verification, so a future regex change cannot silently widen the loophole back
open in the name of fixing the false positive.
"""

from __future__ import annotations

from leo.harness.terminal_quality import (
    completed_research_action_claim,
    contains_future_action_promise,
)


def test_use_as_analytical_verb_is_not_a_future_action_promise() -> None:
    """'I'll use X as the metric' is a complete analytical claim, not a promise."""

    answer = (
        "I'll use dividend cover as the key risk metric; at roughly 1.6x it leaves "
        "less margin if earnings dip."
    )

    assert contains_future_action_promise(answer) is False


def test_genuine_future_research_promise_is_still_flagged() -> None:
    """A real forward-looking research verb must still trip the promise check."""

    assert contains_future_action_promise("Let me check the latest filing before I answer.") is True
    assert contains_future_action_promise("I'll research the company first.") is True


def test_not_financial_advice_disclaimer_is_not_a_completed_research_claim() -> None:
    """The routine disclaimer must not be misread as an unobserved research claim."""

    answer = "I have verified this is not financial advice."

    assert completed_research_action_claim(answer) is None


def test_genuine_completed_research_claim_is_still_classified() -> None:
    """A real claim of completed external research must still require grounding."""

    assert completed_research_action_claim("I checked the latest filings.") == "filing"
    assert (
        completed_research_action_claim("I pulled current quotes before preparing this answer.")
        == "market_quote"
    )
