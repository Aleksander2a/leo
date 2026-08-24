"""End-to-end proof that a committed plan holds the run open until it is done.

The bug this exists to prevent shipped to Slack twice: Leo answered
"...and I'm pulling those now to give you a more grounded read", the run ended,
and no follow-up was possible. Nothing in the harness could tell that answer
apart from a finished one, because intent left no trace it could check.
"""

from __future__ import annotations

import pytest

from leo.demo import run_quote_smoke
from leo.harness.models import (
    BudgetLimits,
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EventType,
    ModelDecision,
    ModelRequest,
    PlanStepDraft,
    PlanStepStatus,
    RunStatus,
    ToolRequest,
    ToolRequests,
)
from leo.integrations.fake import FixtureModel


class _PromisesThenDelivers(FixtureModel):
    """Plans a quote step, tries to answer without it, then actually does it.

    Turn 0 is the exact production failure: a plan is committed and the model
    immediately narrates the work as if it were finished. The harness must reject
    that, and the model must then really call the tool.
    """

    def __init__(self) -> None:
        super().__init__()
        self.answered_early = False

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        if not request.observations:
            if not self.answered_early:
                self.answered_early = True
                return CompletionProposal(
                    answer=(
                        "Alphabet's fundamentals remain strong and I'm pulling the live "
                        "quote now to give you a more grounded read than a made-up number."
                    ),
                    steps=(
                        PlanStepDraft(
                            key="quote",
                            intent="Read the live quote",
                            tool="market.get_quote",
                        ),
                    ),
                )
            return ToolRequests(
                calls=(
                    ToolRequest(
                        id=f"call-{request.iteration + 1:03d}",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            )
        observation = request.observations[-1]
        statement = (
            f"The current quote for {observation.data.get('symbol')} is "
            f"{observation.data.get('price')}."
        )
        return CompletionProposal(
            answer=statement,
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=statement,
                    observation_ids=(observation.id,),
                ),
            ),
        )


class _PlansWorkItNeverDoes(FixtureModel):
    """Commits to a step and then keeps answering without ever doing it."""

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        return CompletionProposal(
            answer=(
                "Here is my read on the request, based on what I already know about the "
                "company and its sector positioning over the last several quarters."
            ),
            steps=(
                PlanStepDraft(
                    key="quote",
                    intent="Read the live quote",
                    tool="market.get_quote",
                ),
            ),
        )


@pytest.mark.asyncio
async def test_answering_with_a_pending_step_sends_the_model_back_to_do_it() -> None:
    result = await run_quote_smoke(
        model=_PromisesThenDelivers(),
        limits=BudgetLimits(max_iterations=6, max_model_calls=6, max_tool_calls=2),
    )

    assert result.run.status is RunStatus.COMPLETED
    # The promissory answer was rejected, the tool actually ran, and the answer
    # the user receives is the one built from the tool's result.
    assert EventType.PLAN_STEP_OUTSTANDING in {event.type for event in result.events}
    assert len(result.observations) == 1
    assert result.run.final_output is not None
    assert "pulling the live quote now" not in result.run.final_output
    assert "181.25" in result.run.final_output
    assert [item.status for item in result.task.step_plan] == [PlanStepStatus.SATISFIED]


@pytest.mark.asyncio
async def test_an_undone_plan_blocks_while_turns_remain_then_still_delivers() -> None:
    """Blocking must be bounded: push back while it can, then answer anyway.

    A model that will never do its own step must not deadlock the turn. The plan
    holds the run open for as long as there are turns to act on the feedback, and
    once there are none it delivers the best answer it has -- marked best-effort
    -- rather than failing the run or hanging. Recoverability beats purity here:
    a user waiting in Slack gets the substantive answer, not silence.
    """

    result = await run_quote_smoke(
        model=_PlansWorkItNeverDoes(),
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=1),
    )

    outstanding = [
        event for event in result.events if event.type is EventType.PLAN_STEP_OUTSTANDING
    ]
    assert outstanding, "the run must have blocked on the undone step"
    assert outstanding[0].payload["pending"] == ["quote"]
    # After the first block the tool choice is pinned to the step's own tool, so
    # later turns are refused at the tool-choice policy rather than here.

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output is not None
    checks = {check.name for check in result.claims} if result.claims else set()
    assert "quote" not in checks
    verification = next(
        event for event in result.events if event.type is EventType.VERIFICATION_PASSED
    )
    # Delivered explicitly as unverified best effort, never as a clean pass.
    assert any(
        str(check.get("name")) == "best_effort_fallback"
        for check in verification.payload.get("checks", [])
    )


@pytest.mark.asyncio
async def test_a_plan_free_turn_still_completes_in_one_pass() -> None:
    """Planning must not tax simple requests: no steps means no extra loop."""

    result = await run_quote_smoke(
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=2),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert result.task.step_plan == ()
    assert EventType.PLAN_STEP_OUTSTANDING not in {event.type for event in result.events}
