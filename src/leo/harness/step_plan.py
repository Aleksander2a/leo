"""Reconciliation of the model's committed step plan against real run evidence.

The scratchpad answers "what happened". This module answers "what did the model
say it would do, and has it actually done it" -- the difference between a
finished answer and an abandoned one.

The rule that makes this worth having: a step naming a tool is discharged only
by a *retrieved observation from that tool*. The model cannot mark its own work
complete, so narrating "I'm pulling the earnings data now" and then stopping
leaves a pending step, and the coordinator sends the model back to do it.
"""

from __future__ import annotations

from leo.harness.models import (
    Observation,
    ObservationStatus,
    PlannedStep,
    PlanStepDraft,
    PlanStepStatus,
)

MAX_PLAN_STEPS = 12


def _tool_key(name: str) -> str:
    """Compare tool names in the one spelling both sides can agree on.

    Leo's canonical names are dotted (``market.get_quote``), but providers do not
    accept dots in function names, so the model only ever *sees* the underscored
    form (``market_get_quote``) and plans its steps with that. Comparing the two
    literally meant no tool-backed step ever matched its own observation: every
    step stayed pending, and a model that had done exactly what it planned was
    told it had not.
    """

    return name.strip().replace(".", "_").casefold()


def merge_step_drafts(
    existing: tuple[PlannedStep, ...],
    drafts: tuple[PlanStepDraft, ...],
    *,
    available_tools: frozenset[str] = frozenset(),
) -> tuple[PlannedStep, ...]:
    """Apply the model's proposed steps and revisions to the committed plan.

    Plans are adaptive: the model may add steps as it learns, and may abandon a
    step it can no longer do. What it may not do is quietly drop one -- an
    abandonment without a reason is ignored, so the step stays pending and the
    loop keeps asking for it.

    Nor may it abandon work it is still able to do. Faced with the completion
    gate, a model will happily retire all seven of its own steps with
    "Replaced by direct tool calls this turn" and then make no tool calls at
    all -- turning the plan into a formality it can dismiss. So an abandonment
    is only honoured for a reasoning-only step, or one whose tool this run
    genuinely cannot call. Everything else stays pending: if the tool is right
    there, the way past the gate is to use it.

    A satisfied step is immutable. Re-proposing it cannot reopen work that real
    evidence already discharged.
    """

    available = {_tool_key(name) for name in available_tools}
    merged = {item.key: item for item in existing}
    for draft in drafts:
        current = merged.get(draft.key)
        if current is not None and current.status is PlanStepStatus.SATISFIED:
            continue
        reason = draft.abandon_reason.strip()
        if reason:
            base = current or PlannedStep(key=draft.key, intent=draft.intent, tool=draft.tool)
            if base.needs_evidence and _tool_key(base.tool) in available:
                merged[draft.key] = base.model_copy(
                    update={"intent": draft.intent, "status": PlanStepStatus.PENDING}
                )
                continue
            merged[draft.key] = base.model_copy(
                update={"status": PlanStepStatus.ABANDONED, "note": reason}
            )
            continue
        if current is not None:
            # Keep the original status; only intent/tool may be refined.
            merged[draft.key] = current.model_copy(
                update={"intent": draft.intent, "tool": draft.tool}
            )
            continue
        if len(merged) >= MAX_PLAN_STEPS:
            continue
        merged[draft.key] = PlannedStep(key=draft.key, intent=draft.intent, tool=draft.tool)
    # Preserve commitment order: existing steps first, then newly added ones.
    ordered: list[PlannedStep] = []
    seen: set[str] = set()
    for item in existing:
        ordered.append(merged[item.key])
        seen.add(item.key)
    for draft in drafts:
        if draft.key not in seen and draft.key in merged:
            ordered.append(merged[draft.key])
            seen.add(draft.key)
    return tuple(ordered)


def discharge_satisfied_steps(
    plan: tuple[PlannedStep, ...],
    observations: tuple[Observation, ...],
) -> tuple[PlannedStep, ...]:
    """Mark tool-backed steps satisfied once their tool actually returned data."""

    available: dict[str, int] = {}
    for observation in observations:
        if observation.status is not ObservationStatus.RETRIEVED:
            continue
        key = _tool_key(observation.kind)
        available[key] = available.get(key, 0) + 1
    if not available:
        return plan

    # Each step consumes its *own* observation. Matching on tool name alone let a
    # single read discharge every step that named the same tool: one NVDA quote
    # closed both "nvda_quote" and "goog_quote", and the comparison shipped with
    # half its evidence missing while the plan claimed to be complete.
    #
    # Steps already satisfied on an earlier turn still hold the observation that
    # discharged them, so they are debited first; the remainder is what is
    # genuinely unclaimed. Pending steps are then credited in commitment order.
    for item in plan:
        if item.status is PlanStepStatus.SATISFIED and item.needs_evidence:
            key = _tool_key(item.tool)
            available[key] = max(0, available.get(key, 0) - 1)

    discharged: list[PlannedStep] = []
    for item in plan:
        key = _tool_key(item.tool)
        if (
            item.status is PlanStepStatus.PENDING
            and item.needs_evidence
            and available.get(key, 0) > 0
        ):
            available[key] -= 1
            discharged.append(item.model_copy(update={"status": PlanStepStatus.SATISFIED}))
            continue
        discharged.append(item)
    return tuple(discharged)


def abandon_unreachable_steps(
    plan: tuple[PlannedStep, ...],
    available_tools: frozenset[str],
) -> tuple[PlannedStep, ...]:
    """Retire steps whose tool this run cannot call, so the loop cannot spin.

    A step is only worth blocking completion for if it is still achievable. A
    model can plan against a tool that was never advertised, or one withdrawn
    after failing unrecoverably; without this the run would block on a step that
    can never be discharged and burn its whole budget before answering.

    The step is abandoned with an explicit note rather than deleted, so the
    answer can tell the user which part went unsourced and why.
    """

    if not plan:
        return plan
    available = {_tool_key(name) for name in available_tools}
    return tuple(
        item.model_copy(
            update={
                "status": PlanStepStatus.ABANDONED,
                "note": f"tool {item.tool!r} was not available in this run",
            }
        )
        if (
            item.status is PlanStepStatus.PENDING
            and item.needs_evidence
            and _tool_key(item.tool) not in available
        )
        else item
        for item in plan
    )


def outstanding_work_feedback(pending: tuple[PlannedStep, ...]) -> str:
    """Tell the model exactly which committed steps still have no evidence."""

    lines = "; ".join(
        f"{item.key} ({item.intent})" + (f" via {item.tool}" if item.needs_evidence else "")
        for item in pending
    )
    return (
        "You committed to steps that are not done yet: "
        f"{lines}. "
        "Do them now: call the named tool for each remaining step, then answer from the "
        "results. If a step turned out to be impossible or unnecessary, resubmit it in "
        "`steps` with an `abandon_reason` explaining why, and say so in your answer. Do not "
        "describe work as in progress -- this run ends when you answer."
    )


def render_step_plan(plan: tuple[PlannedStep, ...]) -> tuple[str, ...]:
    """Compact prompt rendering of the committed plan and each step's state."""

    return tuple(
        f"[{item.status.value}] {item.key}: {item.intent}"
        + (f" (tool: {item.tool})" if item.needs_evidence else "")
        + (f" -- {item.note}" if item.note else "")
        for item in plan
    )
