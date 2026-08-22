from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import cast

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.schema import CreateTable

from leo.harness.models import RunStatus, ScopeKey, TaskStatus
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    PlanStatus,
    cancel_plan_snapshot,
    revision_digest,
)
from leo.persistence.plan_store import PlanTerminalError, PostgresPlanStore
from leo.persistence.schema import Base

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


class _ScalarSession:
    def __init__(self, *rows: object) -> None:
        self.rows = list(rows)
        self.statements: list[object] = []

    async def scalar(self, statement: object) -> object:
        self.statements.append(statement)
        return self.rows.pop(0)


def _definition(
    key: str,
    *,
    objective: str | None = None,
    depends_on: tuple[str, ...] = (),
    max_attempts: int = 3,
) -> PlanNodeDefinition:
    return PlanNodeDefinition(
        key=key,
        objective=objective or f"Complete {key}",
        depends_on=depends_on,
        max_attempts=max_attempts,
    )


def _revision(
    *,
    revision_id: str = "revision-1",
    number: int = 1,
    nodes: tuple[PlanNodeDefinition, ...] | None = None,
    parent_revision_id: str | None = None,
    parent_digest: str | None = None,
) -> PlanRevision:
    definitions = nodes or (_definition("a"), _definition("b", depends_on=("a",)))
    return PlanRevision(
        id=revision_id,
        plan_id="plan-1",
        number=number,
        goal="Answer a multi-step question",
        nodes=definitions,
        digest=revision_digest("Answer a multi-step question", definitions),
        parent_revision_id=parent_revision_id,
        parent_digest=parent_digest,
        reason="initial_plan" if number == 1 else "repair failed branch",
        created_at=NOW,
    )


def _plan() -> Plan:
    first = _revision()
    return Plan(
        id="plan-1",
        scope=ScopeKey(organization_id="org-a", strategy_id="strategy-provenance"),
        parent_task_id="task-parent",
        parent_run_id="run-parent",
        idempotency_key="event-1:plan",
        initial_digest=first.digest,
        created_at=NOW,
        updated_at=NOW,
    )


def _pending_node(definition: PlanNodeDefinition, *, node_id: str) -> PlanNode:
    return PlanNode(
        id=node_id,
        plan_id="plan-1",
        revision_id="revision-1",
        revision_number=1,
        definition=definition,
        created_at=NOW,
        updated_at=NOW,
    )


def test_revision_digest_is_canonical_and_mutation_sensitive() -> None:
    a = _definition("a")
    b = _definition("b", depends_on=("a",))

    assert revision_digest("goal", (a, b)) == revision_digest("goal", (b, a))
    assert revision_digest("goal", (a, b)) != revision_digest(
        "goal", (a, b.model_copy(update={"objective": "Different work"}))
    )


@pytest.mark.parametrize(
    "nodes",
    [
        (_definition("a"), _definition("a")),
        (_definition("a", depends_on=("missing",)),),
        (_definition("a", depends_on=("b",)), _definition("b", depends_on=("a",))),
    ],
)
def test_revision_graph_rejects_duplicate_unknown_and_cyclic_dependencies(
    nodes: tuple[PlanNodeDefinition, ...],
) -> None:
    with pytest.raises(ValueError):
        revision_digest("goal", nodes)


def test_revision_rejects_forged_digest_and_incomplete_parent_link() -> None:
    revision = _revision()
    with pytest.raises(ValidationError, match="digest"):
        revision.model_copy(update={"digest": "0" * 64}, deep=True).__class__.model_validate(
            {**revision.model_dump(), "digest": "0" * 64}
        )

    with pytest.raises(ValidationError, match="parent"):
        PlanRevision(
            **{
                **revision.model_dump(),
                "id": "revision-2",
                "number": 2,
                "parent_revision_id": "revision-1",
                "parent_digest": None,
            }
        )


def test_plan_node_and_delegation_fail_closed_on_malformed_claim_state() -> None:
    pending = _pending_node(_definition("a"), node_id="node-a")
    with pytest.raises(ValidationError, match="claim"):
        PlanNode(
            **{
                **pending.model_dump(),
                "status": PlanNodeStatus.RUNNING,
                "attempt": 1,
                "claim_owner": "worker",
                "claim_token": None,
                "lease_expires_at": NOW + timedelta(seconds=60),
            }
        )

    with pytest.raises(ValidationError, match="finished_at"):
        Delegation(
            id="delegation-1",
            plan_id="plan-1",
            revision_id="revision-1",
            node_id="node-a",
            parent_task_id="task-parent",
            parent_run_id="run-parent",
            attempt=1,
            owner="worker",
            claim_token="opaque-token",
            status=DelegationStatus.FAILED,
            error="safe failure",
            created_at=NOW,
        )


def test_snapshot_replay_rejects_revision_and_parent_authority_tampering() -> None:
    revision = _revision()
    plan = _plan()
    nodes = tuple(
        _pending_node(definition, node_id=f"node-{definition.key}") for definition in revision.nodes
    )
    snapshot = PlanSnapshot(plan=plan, revisions=(revision,), nodes=nodes, delegations=())
    assert snapshot.current_nodes == nodes

    bad_delegation = Delegation(
        id="delegation-1",
        plan_id=plan.id,
        revision_id=revision.id,
        node_id=nodes[0].id,
        parent_task_id="forged-parent",
        parent_run_id=plan.parent_run_id,
        attempt=1,
        owner="worker",
        claim_token="opaque-token",
        status=DelegationStatus.RUNNING,
        created_at=NOW,
    )
    with pytest.raises(ValidationError, match="authority"):
        PlanSnapshot(
            plan=plan,
            revisions=(revision,),
            nodes=nodes,
            delegations=(bad_delegation,),
        )


def test_snapshot_replay_requires_attached_child_to_match_latest_delegation() -> None:
    definition = _definition("only")
    revision = _revision(nodes=(definition,))
    plan = _plan().model_copy(update={"initial_digest": revision.digest})
    running = PlanNode(
        id="node-only",
        plan_id=plan.id,
        revision_id=revision.id,
        revision_number=1,
        definition=definition,
        status=PlanNodeStatus.RUNNING,
        attempt=1,
        claim_owner="worker",
        claim_token="claim-token",
        lease_expires_at=NOW + timedelta(seconds=60),
        child_task_id="child-task",
        child_run_id="child-run",
        created_at=NOW,
        updated_at=NOW,
    )
    delegation = Delegation(
        id="delegation-1",
        plan_id=plan.id,
        revision_id=revision.id,
        node_id=running.id,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
        attempt=1,
        owner="worker",
        claim_token="claim-token",
        status=DelegationStatus.RUNNING,
        child_task_id="child-task",
        child_run_id="child-run",
        created_at=NOW,
    )
    PlanSnapshot(
        plan=plan,
        revisions=(revision,),
        nodes=(running,),
        delegations=(delegation,),
    )

    with pytest.raises(ValidationError, match="child identity"):
        PlanSnapshot(
            plan=plan,
            revisions=(revision,),
            nodes=(running,),
            delegations=(
                delegation.model_copy(
                    update={"child_task_id": "other-task", "child_run_id": "other-run"}
                ),
            ),
        )


def test_plan_schema_has_durable_journals_and_organization_only_authority() -> None:
    tables = Base.metadata.tables
    assert {"plans", "plan_revisions", "plan_nodes", "delegations"}.issubset(tables)
    assert not tables["plans"].c.strategy_id.foreign_keys
    assert {"organization_id", "idempotency_key"} == {
        column.name
        for constraint in tables["plans"].constraints
        if constraint.name == "uq_plans_org_idempotency_key"
        for column in constraint.columns
    }
    assert {
        "ix_plan_nodes_claim_eligibility",
        "ix_plan_nodes_plan_revision",
    }.issubset(index.name for index in tables["plan_nodes"].indexes)


def test_plan_tables_compile_as_postgres_ddl_offline() -> None:
    dialect = postgresql.dialect()
    ddl = "\n".join(
        str(CreateTable(Base.metadata.tables[name]).compile(dialect=dialect))
        for name in ("plans", "plan_revisions", "plan_nodes", "delegations")
    )
    assert "JSONB" in ddl
    assert "uq_plans_org_idempotency_key" in ddl
    assert "ck_plan_nodes_claim_state" in ddl


def test_memory_visibility_metadata_matches_forward_compatibility_migration() -> None:
    for table_name, constraint_name in (
        ("memory_records", "ck_memory_records_visibility"),
        ("memory_sources", "ck_memory_sources_visibility"),
        ("memory_revisions", "ck_memory_revisions_visibility"),
    ):
        constraint = next(
            item
            for item in Base.metadata.tables[table_name].constraints
            if item.name == constraint_name
        )
        expression = str(constraint.sqltext)
        assert "conversation_local" in expression
        assert "channel_local" in expression


def test_parent_cancellation_propagates_through_running_and_pending_plan_work() -> None:
    revision = _revision()
    plan = _plan()
    running = PlanNode(
        id="node-a",
        plan_id=plan.id,
        revision_id=revision.id,
        revision_number=1,
        definition=revision.nodes[0],
        status=PlanNodeStatus.RUNNING,
        attempt=1,
        claim_owner="worker",
        claim_token="claim-token",
        lease_expires_at=NOW + timedelta(seconds=60),
        child_task_id="child-task",
        child_run_id="child-run",
        created_at=NOW,
        updated_at=NOW,
    )
    pending = _pending_node(revision.nodes[1], node_id="node-b")
    delegation = Delegation(
        id="delegation-1",
        plan_id=plan.id,
        revision_id=revision.id,
        node_id=running.id,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
        attempt=1,
        owner="worker",
        claim_token="claim-token",
        status=DelegationStatus.RUNNING,
        child_task_id="child-task",
        child_run_id="child-run",
        created_at=NOW,
    )
    snapshot = PlanSnapshot(
        plan=plan,
        revisions=(revision,),
        nodes=(running, pending),
        delegations=(delegation,),
    )
    cancelled_at = NOW + timedelta(seconds=5)

    cancelled = cancel_plan_snapshot(
        snapshot,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
        reason="operator_cancelled",
        cancelled_at=cancelled_at,
    )

    assert cancelled.plan.status is PlanStatus.FAILED
    assert cancelled.plan.error == "parent_cancelled:operator_cancelled"
    assert all(node.status is PlanNodeStatus.FAILED for node in cancelled.current_nodes)
    assert all(node.claim_token is None for node in cancelled.current_nodes)
    assert cancelled.current_nodes[0].child_run_id == "child-run"
    assert cancelled.delegations[0].status is DelegationStatus.SUPERSEDED
    assert cancelled.delegations[0].finished_at == cancelled_at
    with pytest.raises(ValueError, match="stable parent"):
        cancel_plan_snapshot(
            snapshot,
            parent_task_id="forged",
            parent_run_id=plan.parent_run_id,
            reason="operator_cancelled",
            cancelled_at=cancelled_at,
        )
    with pytest.raises(ValueError, match="terminal plan"):
        cancel_plan_snapshot(
            cancelled,
            parent_task_id=plan.parent_task_id,
            parent_run_id=plan.parent_run_id,
            reason="again",
            cancelled_at=cancelled_at,
        )


@pytest.mark.asyncio
async def test_durable_plan_parent_authority_is_locked_and_terminal_fails_closed() -> None:
    scope = ScopeKey(organization_id="org-a", strategy_id="strategy-provenance")
    task = SimpleNamespace(
        organization_id=scope.organization_id,
        status=TaskStatus.CANCELLED.value,
    )
    run = SimpleNamespace(
        task_id="task-parent",
        organization_id=scope.organization_id,
        status=RunStatus.CANCELLED.value,
    )
    session = _ScalarSession(task, run)
    store = object.__new__(PostgresPlanStore)

    with pytest.raises(PlanTerminalError, match="terminal parent"):
        await store._require_active_parent(
            cast(AsyncSession, session),
            scope,
            "task-parent",
            "run-parent",
        )

    dialect = postgresql.dialect()
    assert len(session.statements) == 2
    assert all(
        "FOR UPDATE" in str(statement.compile(dialect=dialect)) for statement in session.statements
    )
