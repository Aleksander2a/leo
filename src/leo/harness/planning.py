"""Strict read-only research plans with deterministic DAG validation."""

from __future__ import annotations

from pydantic import Field, model_validator

from leo.capabilities.catalog import InMemoryToolCatalog
from leo.harness.models import ContractModel, NonEmptyStr, RunPhase, ToolEffect


class EvidenceRequirement(ContractModel):
    kind: NonEmptyStr
    minimum: int = Field(default=1, ge=1, le=32)
    freshness_seconds: int | None = Field(default=None, ge=1)


class ReadPlanNode(ContractModel):
    id: NonEmptyStr
    capability_id: NonEmptyStr
    tool_version: NonEmptyStr
    purpose: NonEmptyStr
    arguments: dict[str, object]
    depends_on: tuple[NonEmptyStr, ...] = ()
    expected_observation_kind: NonEmptyStr
    evidence: tuple[EvidenceRequirement, ...] = ()
    fallback_node_id: str | None = None
    effect: ToolEffect = ToolEffect.READ


class ReadPlan(ContractModel):
    id: NonEmptyStr
    version: NonEmptyStr
    goal: NonEmptyStr
    nodes: tuple[ReadPlanNode, ...] = Field(min_length=1, max_length=32)
    max_tool_calls: int = Field(ge=1, le=64)
    max_cost: float = Field(ge=0)
    sequential: bool = True

    @model_validator(mode="after")
    def validate_dag_and_read_only(self) -> ReadPlan:
        node_ids = [node.id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("plan node IDs must be unique")
        nodes = {node.id: node for node in self.nodes}
        if any(node.effect is not ToolEffect.READ for node in self.nodes):
            raise ValueError("read plans cannot contain effectful nodes")
        if any(dependency not in nodes for node in self.nodes for dependency in node.depends_on):
            raise ValueError("plan dependency is unknown")
        if any(node.fallback_node_id not in nodes for node in self.nodes if node.fallback_node_id):
            raise ValueError("plan fallback is unknown")
        _assert_acyclic(nodes)
        if len(self.nodes) > self.max_tool_calls:
            raise ValueError("plan exceeds tool-call budget")
        return self


class PlanValidationError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def independent_batches(plan: ReadPlan) -> tuple[tuple[str, ...], ...]:
    """Return deterministic topological batches; execution remains a separate adapter concern."""

    remaining = {node.id: set(node.depends_on) for node in plan.nodes}
    batches: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(sorted(node_id for node_id, deps in remaining.items() if not deps))
        if not ready:
            raise PlanValidationError("plan_cycle")
        batches.append(ready)
        for node_id in ready:
            del remaining[node_id]
        for deps in remaining.values():
            deps.difference_update(ready)
    return tuple(batches)


def validate_plan_eligibility(
    plan: ReadPlan,
    catalog: InMemoryToolCatalog,
    *,
    phase: RunPhase,
    profile: str,
    role: str,
    remaining_cost: float,
) -> None:
    eligible = {
        (record.id, record.semantic_version)
        for record in catalog.eligible(
            phase=phase, profile=profile, role=role, remaining_cost=remaining_cost
        )
    }
    for node in plan.nodes:
        if (node.capability_id, node.tool_version) not in eligible:
            raise PlanValidationError("plan_capability_not_eligible")


def _assert_acyclic(nodes: dict[str, ReadPlanNode]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("plan dependencies must be acyclic")
        if node_id in visited:
            return
        visiting.add(node_id)
        for dependency in nodes[node_id].depends_on:
            visit(dependency)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in nodes:
        visit(node_id)
