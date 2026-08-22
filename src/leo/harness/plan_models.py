"""Durable, provider-neutral contracts for bounded multi-step plan execution."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey

MAX_PLAN_NODES = 64
MAX_PLAN_REVISIONS = 8
PlanNodeKey = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")]


class PlanStatus(StrEnum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class PlanNodeStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DelegationStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class PlanNodeDefinition(ContractModel):
    """One immutable node in a revision DAG."""

    key: PlanNodeKey
    objective: Annotated[str, Field(min_length=1, max_length=4_000)]
    depends_on: tuple[PlanNodeKey, ...] = Field(default=(), max_length=16)
    max_attempts: int = Field(default=3, ge=1, le=8)

    @model_validator(mode="after")
    def validate_dependencies(self) -> PlanNodeDefinition:
        if self.objective != self.objective.strip():
            raise ValueError("plan node objective must not have surrounding whitespace")
        if tuple(sorted(self.depends_on)) != self.depends_on:
            raise ValueError("plan node dependencies must be sorted")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("plan node dependencies must be unique")
        if self.key in self.depends_on:
            raise ValueError("plan node cannot depend on itself")
        return self


class PlanRevision(ContractModel):
    """An immutable, content-addressed plan revision."""

    id: NonEmptyStr
    plan_id: NonEmptyStr
    number: int = Field(ge=1, le=MAX_PLAN_REVISIONS)
    goal: Annotated[str, Field(min_length=1, max_length=4_000)]
    nodes: tuple[PlanNodeDefinition, ...] = Field(min_length=1, max_length=MAX_PLAN_NODES)
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    parent_revision_id: str | None = None
    parent_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    reason: Annotated[str, Field(min_length=1, max_length=1_000)]
    created_at: datetime

    @model_validator(mode="after")
    def validate_revision(self) -> PlanRevision:
        validate_revision_graph(self.nodes)
        if self.goal != self.goal.strip():
            raise ValueError("plan goal must not have surrounding whitespace")
        if self.reason != self.reason.strip():
            raise ValueError("plan revision reason must not have surrounding whitespace")
        if self.number == 1:
            if self.parent_revision_id is not None or self.parent_digest is not None:
                raise ValueError("initial revision cannot carry a parent revision")
        elif self.parent_revision_id is None or self.parent_digest is None:
            raise ValueError("non-initial revision requires its parent revision and digest")
        expected = revision_digest(self.goal, self.nodes)
        if self.digest != expected:
            raise ValueError("plan revision digest does not match its immutable DAG")
        return self


class Plan(ContractModel):
    """Stable parent-owned identity and lifecycle for a revisioned plan."""

    id: NonEmptyStr
    scope: ScopeKey
    parent_task_id: NonEmptyStr
    parent_run_id: NonEmptyStr
    idempotency_key: Annotated[str, Field(min_length=1, max_length=128)]
    initial_digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    status: PlanStatus = PlanStatus.ACTIVE
    current_revision: int = Field(default=1, ge=1, le=MAX_PLAN_REVISIONS)
    max_revisions: int = Field(default=4, ge=1, le=MAX_PLAN_REVISIONS)
    output: str | None = None
    error: str | None = None
    version: int = Field(default=1, ge=1)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_lifecycle(self) -> Plan:
        if self.current_revision > self.max_revisions:
            raise ValueError("current revision exceeds the plan revision bound")
        if self.status is PlanStatus.ACTIVE and (self.output is not None or self.error is not None):
            raise ValueError("active plan cannot carry a terminal result")
        if self.status is PlanStatus.COMPLETED and (not self.output or self.error is not None):
            raise ValueError("completed plan requires output and cannot carry an error")
        if self.status is PlanStatus.FAILED and (not self.error or self.output is not None):
            raise ValueError("failed plan requires an error and cannot carry output")
        return self


class PlanNode(ContractModel):
    """Durable execution state for one immutable revision node."""

    id: NonEmptyStr
    plan_id: NonEmptyStr
    revision_id: NonEmptyStr
    revision_number: int = Field(ge=1, le=MAX_PLAN_REVISIONS)
    definition: PlanNodeDefinition
    status: PlanNodeStatus = PlanNodeStatus.PENDING
    attempt: int = Field(default=0, ge=0, le=8)
    claim_owner: str | None = None
    claim_token: str | None = None
    lease_expires_at: datetime | None = None
    child_task_id: str | None = None
    child_run_id: str | None = None
    output: str | None = None
    error: str | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_execution_state(self) -> PlanNode:
        if self.attempt > self.definition.max_attempts:
            raise ValueError("plan node attempt exceeds its bound")
        claimed = (
            self.claim_owner is not None,
            self.claim_token is not None,
            self.lease_expires_at is not None,
        )
        if self.status is PlanNodeStatus.RUNNING:
            if claimed != (True, True, True) or self.attempt < 1:
                raise ValueError("running plan node requires one complete current claim")
        elif any(claimed):
            raise ValueError("only a running plan node may carry claim state")
        if (self.child_task_id is None) != (self.child_run_id is None):
            raise ValueError("child task and run IDs must be recorded together")
        if self.status is PlanNodeStatus.PENDING:
            if self.output is not None or self.error is not None:
                raise ValueError("pending plan node cannot carry a result")
        elif self.status is PlanNodeStatus.RUNNING:
            if self.output is not None or self.error is not None:
                raise ValueError("running plan node cannot carry a result")
        elif self.status is PlanNodeStatus.COMPLETED:
            if not self.output or self.error is not None:
                raise ValueError("completed plan node requires output")
        elif not self.error or self.output is not None:
            raise ValueError("failed plan node requires an error")
        return self


class Delegation(ContractModel):
    """Append-only attempt record; supersession preserves stale-claim history."""

    id: NonEmptyStr
    plan_id: NonEmptyStr
    revision_id: NonEmptyStr
    node_id: NonEmptyStr
    parent_task_id: NonEmptyStr
    parent_run_id: NonEmptyStr
    attempt: int = Field(ge=1, le=8)
    owner: NonEmptyStr
    claim_token: NonEmptyStr
    status: DelegationStatus
    child_task_id: str | None = None
    child_run_id: str | None = None
    output: str | None = None
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None

    @model_validator(mode="after")
    def validate_attempt_state(self) -> Delegation:
        if (self.child_task_id is None) != (self.child_run_id is None):
            raise ValueError("delegation child task and run IDs must be recorded together")
        if self.status is DelegationStatus.RUNNING:
            if self.finished_at is not None or self.output is not None or self.error is not None:
                raise ValueError("running delegation cannot carry a terminal result")
        elif self.finished_at is None:
            raise ValueError("terminal delegation requires finished_at")
        elif self.status is DelegationStatus.COMPLETED:
            if not self.output or self.error is not None:
                raise ValueError("completed delegation requires output")
        elif not self.error or self.output is not None:
            raise ValueError("failed or superseded delegation requires an error")
        return self


class PlanNodeClaim(ContractModel):
    """Opaque fenced claim returned to a worker for one dependency-ready node."""

    scope: ScopeKey
    plan_id: NonEmptyStr
    revision_id: NonEmptyStr
    node_id: NonEmptyStr
    node_key: PlanNodeKey
    parent_task_id: NonEmptyStr
    parent_run_id: NonEmptyStr
    objective: NonEmptyStr
    depends_on: tuple[PlanNodeKey, ...] = ()
    owner: NonEmptyStr
    token: NonEmptyStr
    attempt: int = Field(ge=1, le=8)
    expires_at: datetime


class PlanSnapshot(ContractModel):
    """Replayable aggregate snapshot, including every immutable revision and attempt."""

    plan: Plan
    revisions: tuple[PlanRevision, ...]
    nodes: tuple[PlanNode, ...]
    delegations: tuple[Delegation, ...]

    @model_validator(mode="after")
    def validate_aggregate(self) -> PlanSnapshot:
        if not self.revisions:
            raise ValueError("plan snapshot requires at least one revision")
        if tuple(revision.number for revision in self.revisions) != tuple(
            range(1, len(self.revisions) + 1)
        ):
            raise ValueError("plan revisions must form a contiguous append-only chain")
        if self.plan.current_revision != self.revisions[-1].number:
            raise ValueError("plan current revision does not match the revision chain")
        if self.plan.initial_digest != self.revisions[0].digest:
            raise ValueError("plan initial digest does not match its first revision")
        for previous, current in zip(self.revisions, self.revisions[1:], strict=False):
            if (
                current.parent_revision_id != previous.id
                or current.parent_digest != previous.digest
            ):
                raise ValueError("plan revision parent chain is invalid")
        revision_ids = {revision.id for revision in self.revisions}
        node_ids = {node.id for node in self.nodes}
        if any(
            node.plan_id != self.plan.id or node.revision_id not in revision_ids
            for node in self.nodes
        ):
            raise ValueError("plan node is outside the snapshot aggregate")
        for revision in self.revisions:
            expected = {definition.key: definition for definition in revision.nodes}
            actual = {
                node.definition.key: node for node in self.nodes if node.revision_id == revision.id
            }
            if set(actual) != set(expected):
                raise ValueError("snapshot nodes do not exactly replay the revision DAG")
            if any(
                node.revision_number != revision.number or node.definition != expected[key]
                for key, node in actual.items()
            ):
                raise ValueError("snapshot node definition diverges from its immutable revision")
        if any(
            delegation.plan_id != self.plan.id
            or delegation.revision_id not in revision_ids
            or delegation.node_id not in node_ids
            or delegation.parent_task_id != self.plan.parent_task_id
            or delegation.parent_run_id != self.plan.parent_run_id
            for delegation in self.delegations
        ):
            raise ValueError("delegation is outside the parent plan authority")
        delegations_by_node: dict[str, list[Delegation]] = {}
        for delegation in self.delegations:
            delegations_by_node.setdefault(delegation.node_id, []).append(delegation)
        for node in self.nodes:
            attempts_for_node = sorted(
                delegations_by_node.get(node.id, []), key=lambda item: item.attempt
            )
            attempts = [item.attempt for item in attempts_for_node]
            if attempts != list(range(1, node.attempt + 1)):
                raise ValueError("delegation attempts do not exactly replay node state")
            if not attempts_for_node:
                continue
            latest = attempts_for_node[-1]
            if (latest.child_task_id, latest.child_run_id) != (
                node.child_task_id,
                node.child_run_id,
            ):
                raise ValueError("latest delegation child identity diverges from node state")
            if node.status is PlanNodeStatus.RUNNING:
                if latest.status is not DelegationStatus.RUNNING:
                    raise ValueError("running node requires a running latest delegation")
            elif node.status is PlanNodeStatus.COMPLETED:
                if latest.status is not DelegationStatus.COMPLETED or latest.output != node.output:
                    raise ValueError("completed node diverges from its latest delegation")
            elif node.status is PlanNodeStatus.FAILED and latest.status not in {
                DelegationStatus.FAILED,
                DelegationStatus.SUPERSEDED,
            }:
                raise ValueError("failed node requires a failed or exhausted latest delegation")
        current_nodes = tuple(
            node for node in self.nodes if node.revision_number == self.plan.current_revision
        )
        if self.plan.status is PlanStatus.COMPLETED and any(
            node.status is not PlanNodeStatus.COMPLETED for node in current_nodes
        ):
            raise ValueError("completed plan requires every current node to complete")
        if self.plan.status is PlanStatus.FAILED:
            if any(node.status is PlanNodeStatus.RUNNING for node in current_nodes):
                raise ValueError("failed plan cannot retain a running child")
            if not any(node.status is PlanNodeStatus.FAILED for node in current_nodes) and not (
                self.plan.error or ""
            ).startswith("parent_cancelled:"):
                raise ValueError("failed plan requires a failed current node")
        return self

    @property
    def current_nodes(self) -> tuple[PlanNode, ...]:
        return tuple(
            node for node in self.nodes if node.revision_number == self.plan.current_revision
        )


def revision_digest(goal: str, nodes: tuple[PlanNodeDefinition, ...]) -> str:
    """Return the canonical SHA-256 identity of a goal and immutable node DAG."""

    validate_revision_graph(nodes)
    payload = {
        "goal": goal,
        "nodes": [
            {
                "depends_on": list(node.depends_on),
                "key": node.key,
                "max_attempts": node.max_attempts,
                "objective": node.objective,
            }
            for node in sorted(nodes, key=lambda item: item.key)
        ],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_revision_graph(nodes: tuple[PlanNodeDefinition, ...]) -> None:
    if not nodes:
        raise ValueError("plan revision requires at least one node")
    if len(nodes) > MAX_PLAN_NODES:
        raise ValueError("plan revision exceeds its node bound")
    by_key = {node.key: node for node in nodes}
    if len(by_key) != len(nodes):
        raise ValueError("plan node keys must be unique")
    if any(dependency not in by_key for node in nodes for dependency in node.depends_on):
        raise ValueError("plan node dependency is unknown")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(key: str) -> None:
        if key in visiting:
            raise ValueError("plan node dependencies must be acyclic")
        if key in visited:
            return
        visiting.add(key)
        for dependency in by_key[key].depends_on:
            visit(dependency)
        visiting.remove(key)
        visited.add(key)

    for key in sorted(by_key):
        visit(key)


def cancel_plan_snapshot(
    snapshot: PlanSnapshot,
    *,
    parent_task_id: str,
    parent_run_id: str,
    reason: str,
    cancelled_at: datetime,
) -> PlanSnapshot:
    """Propagate a parent cancellation through nonterminal plan and child attempts."""

    if (
        snapshot.plan.parent_task_id != parent_task_id
        or snapshot.plan.parent_run_id != parent_run_id
    ):
        raise ValueError("only the stable parent may cancel its plan")
    if snapshot.plan.status is not PlanStatus.ACTIVE:
        raise ValueError("terminal plan cannot be cancelled")
    safe_reason = reason.strip()
    if not safe_reason:
        raise ValueError("plan cancellation reason must be non-empty")
    error = f"parent_cancelled:{safe_reason}"
    nodes = tuple(
        node
        if node.status in {PlanNodeStatus.COMPLETED, PlanNodeStatus.FAILED}
        else node.model_copy(
            update={
                "status": PlanNodeStatus.FAILED,
                "claim_owner": None,
                "claim_token": None,
                "lease_expires_at": None,
                "output": None,
                "error": error,
                "updated_at": cancelled_at,
            }
        )
        for node in snapshot.nodes
    )
    delegations = tuple(
        delegation
        if delegation.status is not DelegationStatus.RUNNING
        else delegation.model_copy(
            update={
                "status": DelegationStatus.SUPERSEDED,
                "output": None,
                "error": error,
                "finished_at": cancelled_at,
            }
        )
        for delegation in snapshot.delegations
    )
    plan = snapshot.plan.model_copy(
        update={
            "status": PlanStatus.FAILED,
            "output": None,
            "error": error,
            "version": snapshot.plan.version + 1,
            "updated_at": cancelled_at,
        }
    )
    return PlanSnapshot(
        plan=plan,
        revisions=snapshot.revisions,
        nodes=nodes,
        delegations=delegations,
    )
