"""Leo's plan/act/observe/reason loop must actually carry memory forward.

Before this, every iteration rebuilt a stateless two-message prompt from the
objective, the observation set, and the verifier's complaints. Nothing carried
the model's own reasoning across turns, so on iteration four it could not tell
which tools it had already called, with what arguments, or what it had been
trying to establish. Genuine multi-step work was impossible, and near-identical
prompts produced near-identical decisions that then tripped the no-progress
guard -- a loop that punished the model for the harness erasing its memory.

These tests pin the two properties that make the loop real: the trace reaches
the next turn, and the harness (not the model) writes down what happened.
"""

from __future__ import annotations

import pytest

from leo.demo import run_conversation_smoke
from leo.harness.models import (
    BudgetLimits,
    CompletionProposal,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    ReasoningStep,
    RunStatus,
    ToolRequest,
    ToolRequests,
)
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import FakeQuoteTool, FixedClock


class _RecordingGateway:
    """Answers on the second turn, recording the scratchpad it was handed."""

    def __init__(self) -> None:
        self.seen_scratchpads: list[tuple[ReasoningStep, ...]] = []
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.seen_scratchpads.append(request.scratchpad)
        self.calls += 1
        if self.calls == 1:
            decision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="call-1",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                ),
                plan="Get the NVDA quote so I can compare it against the target price.",
            )
        else:
            decision = CompletionProposal(
                answer="NVDA is quoted at 181.25, modestly below the consensus target.",
                plan="Answer now that the quote is in hand.",
            )
        return ModelTurnResult(
            decision=decision,
            provider="fixture",
            model="fixture/model",
            request_id=f"req-{self.calls}",
            finish_reason="tool_calls" if self.calls == 1 else "stop",
            usage=ModelUsage(),
        )


@pytest.mark.asyncio
async def test_the_next_iteration_sees_what_the_previous_one_did() -> None:
    gateway = _RecordingGateway()

    result = await run_conversation_smoke(
        model=gateway,
        objective="Where is NVDA trading versus its target?",
        tool_registry=ToolRegistry((FakeQuoteTool(FixedClock()),)),
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=4),
    )

    assert result.run.status is RunStatus.COMPLETED
    assert gateway.calls == 2

    # Turn one is a genuine cold start; turn two must not be.
    assert gateway.seen_scratchpads[0] == ()
    carried = gateway.seen_scratchpads[1]
    assert len(carried) == 1

    step = carried[0]
    assert step.iteration == 0
    # The model's own stated intent survives into the next turn.
    assert "compare it against the target price" in step.plan
    # And so does what it actually did, with the argument names it used.
    assert "market.get_quote" in step.action
    assert "symbol" in step.action


@pytest.mark.asyncio
async def test_the_harness_writes_the_outcome_not_the_model() -> None:
    """A model cannot record its own success into its history.

    The plan is model-authored because only the model knows its intent. The
    result is always summarized by the harness from what actually happened, so a
    hallucinated "succeeded" can never enter the trace and be read back as fact
    on a later turn.
    """

    gateway = _RecordingGateway()

    await run_conversation_smoke(
        model=gateway,
        objective="Where is NVDA trading versus its target?",
        tool_registry=ToolRegistry((FakeQuoteTool(FixedClock()),)),
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=4),
    )

    outcome = gateway.seen_scratchpads[1][0].outcome
    assert outcome == "retrieved market.get_quote"


@pytest.mark.asyncio
async def test_the_trace_is_durable_on_the_task() -> None:
    """The scratchpad is task state, so a resumed run does not cold-start."""

    gateway = _RecordingGateway()

    result = await run_conversation_smoke(
        model=gateway,
        objective="Where is NVDA trading versus its target?",
        tool_registry=ToolRegistry((FakeQuoteTool(FixedClock()),)),
        limits=BudgetLimits(max_iterations=4, max_model_calls=4, max_tool_calls=4),
    )

    assert len(result.task.scratchpad) >= 1
    assert result.task.scratchpad[0].action.startswith("market.get_quote")


def test_a_reasoning_step_renders_compactly_for_the_prompt() -> None:
    step = ReasoningStep(
        iteration=2,
        plan="Check earnings before comparing margins.",
        action="market.get_earnings_surprises(symbol)",
        outcome="retrieved market.get_earnings_surprises",
    )

    rendered = step.render()

    assert rendered.startswith("[2] plan: Check earnings")
    assert "action: market.get_earnings_surprises(symbol)" in rendered
    assert "result: retrieved market.get_earnings_surprises" in rendered
