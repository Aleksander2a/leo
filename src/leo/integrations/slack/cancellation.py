"""Explicit, bounded Slack cancellation intent.

Cancellation is a transport control request, not a model classification.  The
matcher therefore accepts only a small exact vocabulary after conservative
whitespace/case/punctuation normalization.  Sentences that merely discuss
stopping or cancellation continue through the conversational harness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from leo.integrations.slack.events import AdmittedSlackMention

_TRAILING_PUNCTUATION = re.compile(r"[.!?]+$")
_WHITESPACE = re.compile(r"\s+")
_CANCEL_REQUESTS = frozenset(
    {
        "cancel",
        "cancel this run",
        "cancel this task",
        "please cancel",
        "please cancel this run",
        "please cancel this task",
        "stop",
        "stop leo",
        "stop this run",
        "stop this task",
        "please stop",
        "please stop leo",
        "please stop this run",
        "please stop this task",
    }
)


class SlackCancellationOutcome(StrEnum):
    APPLIED = "applied"
    NO_ACTIVE_TASK = "no_active_task"
    NOT_AUTHORIZED = "not_authorized"
    TERMINAL_RACE = "terminal_race"


@dataclass(frozen=True, slots=True)
class SlackCancellationResult:
    admitted: AdmittedSlackMention
    outcome: SlackCancellationOutcome
    message: str


def is_slack_cancellation_request(prompt: str) -> bool:
    normalized = _WHITESPACE.sub(" ", prompt.strip().casefold())
    normalized = _TRAILING_PUNCTUATION.sub("", normalized).strip()
    return normalized in _CANCEL_REQUESTS


def cancellation_message(outcome: SlackCancellationOutcome) -> str:
    return {
        SlackCancellationOutcome.APPLIED: "Leo cancelled the active task and its child work.",
        SlackCancellationOutcome.NO_ACTIVE_TASK: (
            "There is no active Leo task in this conversation to cancel."
        ),
        SlackCancellationOutcome.NOT_AUTHORIZED: (
            "Only the person who started the active task can cancel it."
        ),
        SlackCancellationOutcome.TERMINAL_RACE: (
            "The active task reached a terminal state before cancellation won."
        ),
    }[outcome]
