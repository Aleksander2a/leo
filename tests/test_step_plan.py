"""The committed step plan: what makes the ReAct loop finish only when it is done.

The scratchpad records what happened. The step plan records what the model said
it would do, and is the thing that stops a run from shipping a half-finished
answer: a turn that narrates "I'm pulling the earnings data now" and then stops
leaves a pending step, so the coordinator sends the model back to actually do it.
"""

from __future__ import annotations

from datetime import UTC, datetime

from leo.harness.models import (
    EvidenceQuality,
    Observation,
    ObservationStatus,
    PlannedStep,
    PlanStepDraft,
    PlanStepStatus,
    ScopeKey,
    SourceRef,
)
from leo.harness.step_plan import (
    abandon_unreachable_steps,
    discharge_satisfied_steps,
    merge_step_drafts,
    outstanding_work_feedback,
    render_step_plan,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _observation(
    kind: str,
    *,
    status: ObservationStatus = ObservationStatus.RETRIEVED,
) -> Observation:
    return Observation(
        id=f"obs-{kind}",
        run_id="run-1",
        scope=SCOPE,
        tool_call_id=f"call-{kind}",
        kind=kind,
        status=status,
        quality=EvidenceQuality.PROVIDER_REPORTED,
        data={"value": 1},
        source=SourceRef(provider="p", reference="r"),
        observed_at=NOW,
        raw_hash="0" * 64,
        rejection_code=(None if status is ObservationStatus.RETRIEVED else "PROVIDER_REJECTED"),
    )


def test_a_planned_step_is_discharged_only_by_its_tool_returning_data() -> None:
    """The model cannot mark its own work done; evidence does that."""

    plan = (
        PlannedStep(key="quote", intent="Get the live quote", tool="market.get_quote"),
        PlannedStep(key="earnings", intent="Get earnings", tool="market.get_earnings"),
    )

    unchanged = discharge_satisfied_steps(plan, ())
    assert [item.status for item in unchanged] == [PlanStepStatus.PENDING] * 2

    partial = discharge_satisfied_steps(plan, (_observation("market.get_quote"),))
    assert partial[0].status is PlanStepStatus.SATISFIED
    assert partial[1].status is PlanStepStatus.PENDING


def test_a_rejected_observation_does_not_discharge_a_step() -> None:
    """A tool that ran but returned nothing usable has not done the step."""

    plan = (PlannedStep(key="quote", intent="Get the quote", tool="market.get_quote"),)
    rejected = discharge_satisfied_steps(
        plan,
        (_observation("market.get_quote", status=ObservationStatus.REJECTED),),
    )
    assert rejected[0].status is PlanStepStatus.PENDING


def test_a_step_can_be_abandoned_only_with_a_stated_reason() -> None:
    """Silent attrition is what let half-finished answers ship."""

    plan = (PlannedStep(key="news", intent="Read recent news", tool="market.news"),)

    ignored = merge_step_drafts(plan, (PlanStepDraft(key="news", intent="Read recent news"),))
    assert ignored[0].status is PlanStepStatus.PENDING

    abandoned = merge_step_drafts(
        plan,
        (
            PlanStepDraft(
                key="news",
                intent="Read recent news",
                abandon_reason="No news provider is configured.",
            ),
        ),
    )
    assert abandoned[0].status is PlanStepStatus.ABANDONED
    assert "No news provider" in abandoned[0].note


def test_evidence_backed_steps_cannot_be_reopened() -> None:
    """Re-proposing a satisfied step must not discard the evidence that closed it."""

    plan = (
        PlannedStep(
            key="quote",
            intent="Get the quote",
            tool="market.get_quote",
            status=PlanStepStatus.SATISFIED,
        ),
    )
    merged = merge_step_drafts(
        plan,
        (PlanStepDraft(key="quote", intent="Get the quote", abandon_reason="changed my mind"),),
    )
    assert merged[0].status is PlanStepStatus.SATISFIED


def test_new_steps_are_appended_in_commitment_order() -> None:
    """Plans are adaptive: the model may add steps as it learns."""

    plan = (PlannedStep(key="a", intent="First", tool="tool.a"),)
    merged = merge_step_drafts(
        plan,
        (PlanStepDraft(key="b", intent="Second", tool="tool.b"),),
    )
    assert [item.key for item in merged] == ["a", "b"]


def test_steps_naming_an_unavailable_tool_are_retired_not_blocked_on() -> None:
    """A step the run cannot act on must never hold the loop open.

    Otherwise a model planning against a tool that was never advertised -- or one
    withdrawn after failing unrecoverably -- would spin the run to budget
    exhaustion instead of answering with an honest gap.
    """

    plan = (
        PlannedStep(key="quote", intent="Get the quote", tool="market.get_quote"),
        PlannedStep(key="ghost", intent="Use a missing tool", tool="tool.absent"),
    )
    retired = abandon_unreachable_steps(plan, frozenset({"market.get_quote"}))
    assert retired[0].status is PlanStepStatus.PENDING
    assert retired[1].status is PlanStepStatus.ABANDONED
    assert "not available" in retired[1].note


def test_reasoning_only_steps_need_no_tool_and_are_never_retired() -> None:
    plan = (PlannedStep(key="synth", intent="Compare the two figures"),)
    assert abandon_unreachable_steps(plan, frozenset())[0].status is PlanStepStatus.PENDING


def test_outstanding_feedback_names_the_steps_and_forbids_narrating_progress() -> None:
    pending = (PlannedStep(key="earnings", intent="Get earnings", tool="market.get_earnings"),)
    feedback = outstanding_work_feedback(pending)
    assert "earnings" in feedback
    assert "market.get_earnings" in feedback
    assert "abandon_reason" in feedback
    assert "in progress" in feedback


def test_rendered_plan_shows_each_step_state_for_the_prompt() -> None:
    plan = (
        PlannedStep(
            key="quote",
            intent="Get the quote",
            tool="market.get_quote",
            status=PlanStepStatus.SATISFIED,
        ),
        PlannedStep(key="synth", intent="Compare"),
    )
    rendered = render_step_plan(plan)
    assert rendered[0].startswith("[satisfied] quote:")
    assert "market.get_quote" in rendered[0]
    assert rendered[1].startswith("[pending] synth:")


def test_provider_spelled_tool_names_still_discharge_their_steps() -> None:
    """The model plans in the only spelling it is shown.

    Leo's canonical tool names are dotted, but providers reject dots in function
    names, so the model sees `market_get_quote` and plans with that. Matching the
    two literally meant a step was never discharged by its own observation: the
    model did exactly what it planned and was told it had not, then either looped
    or abandoned the step with a confused reason.
    """

    plan = (PlannedStep(key="quote", intent="Get the quote", tool="market_get_quote"),)

    discharged = discharge_satisfied_steps(plan, (_observation("market.get_quote"),))
    assert discharged[0].status is PlanStepStatus.SATISFIED

    # The same spelling gap must not make a real tool look unavailable either.
    assert (
        abandon_unreachable_steps(plan, frozenset({"market.get_quote"}))[0].status
        is PlanStepStatus.PENDING
    )


def test_two_steps_on_the_same_tool_need_two_observations() -> None:
    """One read cannot discharge two steps that happen to name the same tool.

    A NVDA-vs-GOOG comparison plans `nvda_quote` and `goog_quote`, both via
    `market.get_quote`. Crediting by tool name alone let the single NVDA quote
    close both, so the run reported a complete plan while half the comparison had
    no evidence behind it at all.
    """

    plan = (
        PlannedStep(key="nvda_quote", intent="NVDA price", tool="market.get_quote"),
        PlannedStep(key="goog_quote", intent="GOOG price", tool="market.get_quote"),
    )

    one = discharge_satisfied_steps(plan, (_observation("market.get_quote"),))
    assert [item.status for item in one] == [
        PlanStepStatus.SATISFIED,
        PlanStepStatus.PENDING,
    ]

    both = discharge_satisfied_steps(
        one,
        (
            _observation("market.get_quote"),
            _observation("market.get_quote").model_copy(update={"id": "obs-second-quote"}),
        ),
    )
    assert [item.status for item in both] == [
        PlanStepStatus.SATISFIED,
        PlanStepStatus.SATISFIED,
    ]


def test_a_still_doable_step_cannot_be_abandoned() -> None:
    """Abandonment is for work that cannot be done, not work being dodged.

    Faced with the completion gate, a model will retire all of its own steps with
    a reason like "Replaced by direct tool calls this turn" and then make no tool
    calls at all. If the tool is advertised, the step stays pending.
    """

    plan = (PlannedStep(key="quote", intent="Get the quote", tool="market.get_quote"),)
    draft = (
        PlanStepDraft(
            key="quote",
            intent="Get the quote",
            tool="market.get_quote",
            abandon_reason="Replaced by direct tool calls this turn.",
        ),
    )

    dodged = merge_step_drafts(plan, draft, available_tools=frozenset({"market.get_quote"}))
    assert dodged[0].status is PlanStepStatus.PENDING

    # When the tool really is gone, the abandonment stands.
    genuine = merge_step_drafts(plan, draft, available_tools=frozenset())
    assert genuine[0].status is PlanStepStatus.ABANDONED
