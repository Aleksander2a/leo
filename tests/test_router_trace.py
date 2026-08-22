from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import InMemoryToolCatalog
from leo.capabilities.router import AdaptiveRouter
from leo.harness.events import EventKind, build_event
from leo.harness.models import RunPhase, ScopeKey, ToolEffect, ToolSpec
from leo.harness.planning import (
    EvidenceRequirement,
    PlanValidationError,
    ReadPlan,
    ReadPlanNode,
    validate_plan_eligibility,
)
from leo.harness.trace import TraceError, append_trace, new_trace
from leo.integrations.slack.render import (
    SlackClaim,
    SlackResearchResult,
    SlackSource,
    render_research_result,
)

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _catalog() -> InMemoryToolCatalog:
    catalog = InMemoryToolCatalog()
    catalog.register(
        catalog_tool_from_spec(
            ToolSpec(
                name="market.get_quote",
                version="1",
                description="Return current market quote.",
                domain="MARKET",
                input_schema={"type": "object"},
                effect=ToolEffect.READ,
                allowed_phases=frozenset({RunPhase.RESEARCH}),
            ),
            provider="demo",
            tags=frozenset({"market", "quote"}),
        )
    )
    return catalog


def test_router_uses_only_eligible_catalog_records() -> None:
    decision = AdaptiveRouter(_catalog()).route(
        "market.get_quote",
        phase=RunPhase.RESEARCH,
        profile="research",
        role="reader",
        remaining_cost=1,
    )
    assert decision.mode == "direct"
    assert decision.selected == ("market.get_quote",)


def test_plan_eligibility_and_trace_identity_are_strict() -> None:
    plan = ReadPlan(
        id="plan-1",
        version="v1",
        goal="quote",
        nodes=(
            ReadPlanNode(
                id="node-1",
                capability_id="market.get_quote",
                tool_version="1.0.0",
                purpose="quote",
                arguments={"symbol": "NVDA"},
                expected_observation_kind="market.get_quote",
                evidence=(EvidenceRequirement(kind="market.get_quote"),),
            ),
        ),
        max_tool_calls=1,
        max_cost=1,
    )
    validate_plan_eligibility(
        plan,
        _catalog(),
        phase=RunPhase.RESEARCH,
        profile="research",
        role="reader",
        remaining_cost=1,
    )
    trace = new_trace("run-1", "task-1", SCOPE)
    event = build_event(
        event_id="event-1",
        run_id="run-1",
        task_id="task-1",
        scope=SCOPE,
        sequence=0,
        occurred_at=NOW,
        kind=EventKind.PLAN_VALIDATED,
        correlation_id="corr-1",
        payload={"status": "ok"},
    )
    trace = append_trace(trace, event)
    assert len(trace.digest) == 64
    with pytest.raises(TraceError, match="sequence"):
        append_trace(trace, event)
    with pytest.raises(PlanValidationError, match="not_eligible"):
        validate_plan_eligibility(
            plan.model_copy(
                update={"nodes": (plan.nodes[0].model_copy(update={"tool_version": "9.0.0"}),)}
            ),
            _catalog(),
            phase=RunPhase.RESEARCH,
            profile="research",
            role="reader",
            remaining_cost=1,
        )


def test_research_renderer_separates_facts_inferences_and_sources() -> None:
    rendered = render_research_result(
        SlackResearchResult(
            run_id="run-1",
            facts=(
                SlackClaim(
                    statement="Synthetic fact",
                    sources=(SlackSource(label="SEC", url="https://www.sec.gov/demo"),),
                ),
            ),
            inferences=("Synthetic inference",),
            affected_assumption="Synthetic assumption",
            uncertainty="Synthetic uncertainty",
        )
    )
    assert "Facts" in rendered.chunks[0]
    assert "Inferences" in rendered.chunks[0]
    assert "sec.gov" in rendered.chunks[0]
