from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.harness.models import ToolEffect
from leo.harness.planning import ReadPlan, ReadPlanNode, independent_batches


def _node(node_id: str, *, depends_on: tuple[str, ...] = ()) -> ReadPlanNode:
    return ReadPlanNode(
        id=node_id,
        capability_id="market.quote",
        tool_version="1.0.0",
        purpose="read synthetic quote",
        arguments={"symbol": "NVDA"},
        depends_on=depends_on,
        expected_observation_kind="quote",
    )


def test_read_plan_validates_dag_and_returns_deterministic_batches() -> None:
    plan = ReadPlan(
        id="plan-1",
        version="1",
        goal="challenge thesis",
        nodes=(_node("b", depends_on=("a",)), _node("a"), _node("c")),
        max_tool_calls=3,
        max_cost=1,
    )
    assert independent_batches(plan) == (("a", "c"), ("b",))


@pytest.mark.parametrize(
    "nodes",
    [
        (_node("a", depends_on=("b",)), _node("b", depends_on=("a",))),
        (
            _node(
                "a",
            ),
            _node(
                "a",
            ),
        ),
        (
            _node(
                "a",
            ),
            _node("b", depends_on=("missing",)),
        ),
    ],
)
def test_read_plan_rejects_cycles_duplicates_and_unknown_dependencies(nodes) -> None:
    with pytest.raises(ValidationError):
        ReadPlan(
            id="plan-1",
            version="1",
            goal="read",
            nodes=nodes,
            max_tool_calls=4,
            max_cost=1,
        )


def test_read_plan_rejects_effectful_nodes() -> None:
    with pytest.raises(ValidationError):
        ReadPlan(
            id="plan-1",
            version="1",
            goal="write",
            nodes=(_node("write").model_copy(update={"effect": ToolEffect.WRITE}),),
            max_tool_calls=1,
            max_cost=1,
        )
