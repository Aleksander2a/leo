"""Shared test-only controls for matched eval compositions."""

from __future__ import annotations

from dataclasses import dataclass

from leo.harness.models import (
    CompletionProposal,
    RunBundle,
    VerificationOutcome,
)
from leo.harness.ports import CompletionVerifier


class NoCorrectionVerifier:
    """Keep hard verification while making the first rejection terminal.

    This wrapper exists only under ``leo.evals``. Production composition neither
    imports it nor accepts a request/config flag that can select it.
    """

    def __init__(self, delegate: CompletionVerifier) -> None:
        self._delegate = delegate

    def verify(
        self,
        proposal: CompletionProposal,
        bundle: RunBundle,
    ) -> VerificationOutcome:
        outcome = self._delegate.verify(proposal, bundle)
        if outcome.completion is not None or not outcome.result.retryable:
            return outcome
        return VerificationOutcome(result=outcome.result.model_copy(update={"retryable": False}))


@dataclass(frozen=True)
class BaselineExecution:
    """Observed state from one real coordinator-backed baseline run."""

    invariants: frozenset[str]
    metrics: dict[str, float | int | str]
    hard_failures: tuple[str, ...]
    eligible_schema_count: int
    admitted_destination: str
    model_fixture: str
    matched_tool_catalog: tuple[str, ...]
    exposed_tool_catalog: tuple[str, ...]
