from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.harness.models import EventDraft, EventType, OriginRef, Run, ScopeKey, Task, Thread
from leo.harness.plan_models import PlanNodeDefinition, PlanNodeStatus, PlanStatus
from leo.harness.transitions import cancel_task_and_run, start_task_and_run
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.plan_store import (
    PlanClaimConflictError,
    PlanConflictError,
    PlanNoProgressError,
    PlanRevisionLimitError,
    PlanScopeMismatchError,
    PlanTerminalError,
    PostgresPlanStore,
)
from leo.persistence.run_store import PostgresRunStore


@dataclass(frozen=True)
class PlanHarness:
    store: PostgresPlanStore
    run_store: Any
    clock: FixedClock
    sessions: async_sessionmaker[AsyncSession]
    # Every row this harness writes is namespaced under a unique organization so
    # committed state cannot collide with a concurrent run or with demo data.
    organization_id: str


class _NamespacedIds:
    """Sequential ids with a per-harness suffix.

    Ordering still matters -- delegations are replayed by (created_at, id), and
    a FixedClock makes ties on created_at routine -- so a random id generator
    reorders them. The constant suffix preserves the monotonic lexical order
    while keeping committed rows unique between harnesses.
    """

    def __init__(self, suffix: str) -> None:
        self._inner = SequentialIdGenerator()
        self._suffix = suffix

    def new(self, prefix: str) -> str:
        return f"{self._inner.new(prefix)}-{self._suffix}"


@pytest_asyncio.fixture
async def plan_harness() -> AsyncIterator[PlanHarness]:
    """Commit plan state on a real pool, namespaced and cleaned up per test.

    The single-connection rollback harness cannot serve these tests: its rows are
    invisible to any other connection (so a separately-built plan store failed
    with "parent task or run not found"), and several of these tests claim nodes
    concurrently, which needs genuinely independent connections. So this harness
    follows the documented alternative -- commit for real under a unique
    organization id, then delete exactly what it wrote.
    """

    database_url = Settings().database_url
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured")
    suffix = uuid4().hex[:12]
    organization_id = f"org-plan-{suffix}"
    engine = create_database_engine(database_url.get_secret_value())
    sessions = create_session_factory(engine)
    clock = FixedClock()
    try:
        yield PlanHarness(
            store=PostgresPlanStore(sessions, clock, _NamespacedIds(suffix)),
            run_store=PostgresRunStore(sessions, clock, _NamespacedIds(suffix)),
            clock=clock,
            sessions=sessions,
            organization_id=organization_id,
        )
    finally:
        try:
            await _cleanup_organization(sessions, organization_id)
        finally:
            await engine.dispose()


# Children first, then parents: every statement is scoped to this harness's own
# organization id, so a failure part-way through cannot touch unrelated rows.
# run_events carries no organization column, so it is reached through its runs.
_CLEANUP_SQL = (
    "DELETE FROM delegations WHERE organization_id = :organization_id",
    "DELETE FROM plan_nodes WHERE organization_id = :organization_id",
    "DELETE FROM plan_revisions WHERE organization_id = :organization_id",
    "DELETE FROM plans WHERE organization_id = :organization_id",
    "DELETE FROM run_events WHERE run_id IN"
    " (SELECT id FROM runs WHERE organization_id = :organization_id)",
    "DELETE FROM observations WHERE organization_id = :organization_id",
    "DELETE FROM runs WHERE organization_id = :organization_id",
    "DELETE FROM tasks WHERE organization_id = :organization_id",
    "DELETE FROM threads WHERE organization_id = :organization_id",
)


async def _cleanup_organization(
    sessions: async_sessionmaker[AsyncSession],
    organization_id: str,
) -> None:
    async with sessions() as session:
        for statement in _CLEANUP_SQL:
            await session.execute(text(statement), {"organization_id": organization_id})
        await session.commit()


def _node(
    key: str,
    *,
    depends_on: tuple[str, ...] = (),
    max_attempts: int = 3,
) -> PlanNodeDefinition:
    return PlanNodeDefinition(
        key=key,
        objective=f"Complete {key}",
        depends_on=depends_on,
        max_attempts=max_attempts,
    )


async def _seed_parent(
    harness: PlanHarness,
    *,
    scope: ScopeKey | None = None,
    suffix: str = "parent",
) -> tuple[ScopeKey, Task, Run]:
    effective_scope = scope or ScopeKey(
        organization_id=harness.organization_id,
        strategy_id="strategy-a",
    )
    # These rows are committed, so their identifiers must be unique per harness
    # or two concurrent runs of this suite would collide on the primary key.
    unique = f"{suffix}-{harness.organization_id.rsplit('-', 1)[-1]}"
    thread = Thread(
        id=f"thread-{unique}",
        scope=effective_scope,
        origin=OriginRef(provider="plan-test", external_thread_id=f"thread-{unique}"),
    )
    task = Task(
        id=f"task-{unique}",
        thread_id=thread.id,
        scope=effective_scope,
        objective="Coordinate a durable plan",
    )
    run = Run(id=f"run-{unique}", task_id=task.id, scope=effective_scope)
    await harness.run_store.seed(thread, task, run)
    # Seeding admits a QUEUED pair. Delegated work requires *live* parent
    # authority (ACTIVE task, RUNNING run), so drive the same started transition
    # the coordinator does -- otherwise every plan operation fails closed with
    # PlanTerminalError against a parent that was simply never started. Reload
    # first: the durable versions after seeding are what the CAS commit expects.
    seeded = await harness.run_store.load(task.id, run.id, effective_scope)
    started_task, started_run = start_task_and_run(
        seeded.task,
        seeded.run,
        started_at=harness.clock.now(),
    )
    committed = await harness.run_store.commit(
        expected_task_version=seeded.task.version,
        expected_run_version=seeded.run.version,
        task=started_task,
        run=started_run,
        events=(
            EventDraft(
                type=EventType.TASK_STARTED,
                iteration=0,
                payload={"phase": started_run.phase.value},
            ),
        ),
    )
    return effective_scope, committed.task, committed.run


async def _create(
    harness: PlanHarness,
    *,
    suffix: str = "parent",
    nodes: tuple[PlanNodeDefinition, ...] | None = None,
    max_revisions: int = 4,
):
    scope, task, run = await _seed_parent(harness, suffix=suffix)
    snapshot = await harness.store.create_or_load(
        scope=scope,
        parent_task_id=task.id,
        parent_run_id=run.id,
        idempotency_key=f"idem-{suffix}",
        goal="Complete durable research",
        nodes=nodes or (_node("a"), _node("b")),
        max_revisions=max_revisions,
    )
    return scope, task, run, snapshot


@pytest.mark.asyncio
async def test_idempotency_digest_and_strategy_mapping_are_non_gating(
    plan_harness: PlanHarness,
) -> None:
    scope, task, run, first = await _create(plan_harness)
    remapped = ScopeKey(organization_id=scope.organization_id, strategy_id="strategy-remapped")
    second = await plan_harness.store.create_or_load(
        scope=remapped,
        parent_task_id=task.id,
        parent_run_id=run.id,
        idempotency_key="idem-parent",
        goal="Complete durable research",
        nodes=(_node("a"), _node("b")),
    )
    assert second == first
    assert second.plan.scope == scope

    with pytest.raises(PlanConflictError, match="idempotency"):
        await plan_harness.store.create_or_load(
            scope=scope,
            parent_task_id=task.id,
            parent_run_id=run.id,
            idempotency_key="idem-parent",
            goal="Mutated request",
            nodes=(_node("a"),),
        )


@pytest.mark.asyncio
async def test_parallel_claims_are_disjoint_and_restart_replays_state(
    plan_harness: PlanHarness,
) -> None:
    scope, _, _, snapshot = await _create(plan_harness)
    first, second = await asyncio.gather(
        plan_harness.store.claim_ready_node(
            scope=scope, plan_id=snapshot.plan.id, owner="worker-a"
        ),
        plan_harness.store.claim_ready_node(
            scope=scope, plan_id=snapshot.plan.id, owner="worker-b"
        ),
    )
    assert first is not None and second is not None
    assert {first.node_key, second.node_key} == {"a", "b"}
    assert first.token != second.token

    restarted_store = PostgresPlanStore(
        plan_harness.sessions,
        plan_harness.clock,
        SequentialIdGenerator(),
    )
    reloaded = await restarted_store.replay(scope=scope, plan_id=snapshot.plan.id)
    assert {node.status for node in reloaded.current_nodes} == {PlanNodeStatus.RUNNING}
    assert len(reloaded.delegations) == 2


@pytest.mark.asyncio
async def test_dependency_claim_waits_and_stale_reclaim_fences_old_worker(
    plan_harness: PlanHarness,
) -> None:
    scope, _, _, snapshot = await _create(
        plan_harness,
        nodes=(_node("a", max_attempts=2), _node("b", depends_on=("a",))),
    )
    first = await plan_harness.store.claim_ready_node(
        scope=scope, plan_id=snapshot.plan.id, owner="worker-a", lease_seconds=10
    )
    assert first is not None and first.node_key == "a"
    assert (
        await plan_harness.store.claim_ready_node(
            scope=scope, plan_id=snapshot.plan.id, owner="worker-b"
        )
        is None
    )

    plan_harness.clock.advance(seconds=11)
    reclaimed = await plan_harness.store.claim_ready_node(
        scope=scope, plan_id=snapshot.plan.id, owner="worker-b"
    )
    assert reclaimed is not None
    assert reclaimed.node_id == first.node_id
    assert reclaimed.attempt == 2
    with pytest.raises(PlanClaimConflictError):
        await plan_harness.store.complete_node(scope=scope, claim=first, output="stale")

    await plan_harness.store.complete_node(scope=scope, claim=reclaimed, output="finding a")
    dependent = await plan_harness.store.claim_ready_node(
        scope=scope, plan_id=snapshot.plan.id, owner="worker-c"
    )
    assert dependent is not None and dependent.node_key == "b"
    replayed = await plan_harness.store.reload(scope=scope, plan_id=snapshot.plan.id)
    assert [item.status.value for item in replayed.delegations[:2]] == [
        "superseded",
        "completed",
    ]


@pytest.mark.asyncio
async def test_crash_on_last_attempt_becomes_durable_no_progress(
    plan_harness: PlanHarness,
) -> None:
    scope, _, _, snapshot = await _create(
        plan_harness,
        nodes=(_node("only", max_attempts=1),),
    )
    claim = await plan_harness.store.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="crashing-worker",
        lease_seconds=10,
    )
    assert claim is not None
    plan_harness.clock.advance(seconds=11)
    with pytest.raises(PlanNoProgressError):
        await plan_harness.store.claim_ready_node(
            scope=scope,
            plan_id=snapshot.plan.id,
            owner="recovery-worker",
        )
    replayed = await plan_harness.store.replay(scope=scope, plan_id=snapshot.plan.id)
    assert replayed.current_nodes[0].status is PlanNodeStatus.FAILED
    assert replayed.current_nodes[0].error == "plan_node_attempts_exhausted"
    assert replayed.delegations[0].status.value == "superseded"


@pytest.mark.asyncio
async def test_durable_parent_cancellation_supersedes_running_children(
    plan_harness: PlanHarness,
) -> None:
    scope, task, run, snapshot = await _create(
        plan_harness,
        suffix="cancel",
        nodes=(_node("running"), _node("pending", depends_on=("running",))),
    )
    claim = await plan_harness.store.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="worker",
    )
    assert claim is not None
    cancelled_task, cancelled_run = cancel_task_and_run(
        task,
        run,
        "operator_cancelled",
    )
    await plan_harness.run_store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=cancelled_task,
        run=cancelled_run,
        events=(
            EventDraft(
                type=EventType.RUN_CANCELLED,
                iteration=0,
                payload={"reason": "operator_cancelled"},
            ),
        ),
    )

    cancelled = await plan_harness.store.cancel(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=task.id,
        parent_run_id=run.id,
        reason="operator_cancelled",
    )

    assert cancelled.plan.status is PlanStatus.FAILED
    assert all(node.status is PlanNodeStatus.FAILED for node in cancelled.current_nodes)
    assert cancelled.delegations[0].status.value == "superseded"
    assert (
        await plan_harness.store.cancel(
            scope=scope,
            plan_id=snapshot.plan.id,
            parent_task_id=task.id,
            parent_run_id=run.id,
            reason="operator_cancelled",
        )
        == cancelled
    )


@pytest.mark.asyncio
async def test_attached_child_is_idempotent_recoverable_and_fenced(
    plan_harness: PlanHarness,
) -> None:
    scope, _, _, snapshot = await _create(
        plan_harness,
        nodes=(_node("only"),),
    )
    claim = await plan_harness.store.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="worker",
    )
    assert claim is not None
    _, child_task, child_run = await _seed_parent(
        plan_harness,
        scope=scope,
        suffix="attached-child",
    )
    attached = await plan_harness.store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    assert attached.current_nodes[0].child_task_id == child_task.id
    assert attached.delegations[0].child_run_id == child_run.id

    repeated = await plan_harness.store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    assert repeated == attached

    _, other_task, other_run = await _seed_parent(
        plan_harness,
        scope=scope,
        suffix="different-child",
    )
    with pytest.raises(PlanClaimConflictError, match="different child"):
        await plan_harness.store.attach_child(
            scope=scope,
            claim=claim,
            child_task_id=other_task.id,
            child_run_id=other_run.id,
        )
    with pytest.raises(PlanClaimConflictError, match="stale"):
        await plan_harness.store.attach_child(
            scope=scope,
            claim=claim.model_copy(update={"token": "forged-token"}),
            child_task_id=child_task.id,
            child_run_id=child_run.id,
        )

    completed = await plan_harness.store.complete_node(
        scope=scope,
        claim=claim,
        output="child result",
    )
    assert completed.current_nodes[0].child_task_id == child_task.id
    assert completed.delegations[0].child_run_id == child_run.id


@pytest.mark.asyncio
async def test_replan_is_append_only_idempotent_and_bounded(plan_harness: PlanHarness) -> None:
    scope, task, run, snapshot = await _create(plan_harness, max_revisions=2)
    replacement = (_node("repair"),)
    replanned = await plan_harness.store.append_revision(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=task.id,
        parent_run_id=run.id,
        goal="Repair durable research",
        nodes=replacement,
        reason="dependency failed",
    )
    assert replanned.plan.current_revision == 2
    assert len(replanned.revisions) == 2
    same = await plan_harness.store.append_revision(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=task.id,
        parent_run_id=run.id,
        goal="Repair durable research",
        nodes=replacement,
        reason="duplicate delivery",
    )
    assert same == replanned

    with pytest.raises(PlanRevisionLimitError):
        await plan_harness.store.append_revision(
            scope=scope,
            plan_id=snapshot.plan.id,
            parent_task_id=task.id,
            parent_run_id=run.id,
            goal="Third revision",
            nodes=(_node("third"),),
            reason="another repair",
        )


@pytest.mark.asyncio
async def test_deadlock_scope_and_parent_terminal_authority_fail_closed(
    plan_harness: PlanHarness,
) -> None:
    scope, task, run, snapshot = await _create(
        plan_harness,
        nodes=(_node("a", max_attempts=1), _node("b", depends_on=("a",))),
    )
    claim = await plan_harness.store.claim_ready_node(
        scope=scope, plan_id=snapshot.plan.id, owner="worker"
    )
    assert claim is not None
    await plan_harness.store.fail_node(scope=scope, claim=claim, error="source unavailable")

    with pytest.raises(PlanNoProgressError):
        await plan_harness.store.claim_ready_node(
            scope=scope, plan_id=snapshot.plan.id, owner="worker"
        )
    # The impostor must be a *live* task/run pair that simply is not this plan's
    # parent. A wholly invented id is rejected earlier, by the active-parent
    # lookup, so it never reaches the stable-parent authority check this asserts.
    _, other_task, other_run = await _seed_parent(plan_harness, suffix="impostor")
    with pytest.raises(PlanTerminalError, match="stable parent"):
        await plan_harness.store.finalize(
            scope=scope,
            plan_id=snapshot.plan.id,
            parent_task_id=other_task.id,
            parent_run_id=other_run.id,
            status=PlanStatus.FAILED,
            result="plan could not progress",
        )
    failed = await plan_harness.store.finalize(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=task.id,
        parent_run_id=run.id,
        status=PlanStatus.FAILED,
        result="plan could not progress",
    )
    assert failed.plan.status is PlanStatus.FAILED

    with pytest.raises(PlanScopeMismatchError):
        await plan_harness.store.reload(
            scope=ScopeKey(organization_id="other-org", strategy_id=scope.strategy_id),
            plan_id=snapshot.plan.id,
        )


@pytest.mark.asyncio
async def test_parent_finalizes_only_after_all_nodes_complete(plan_harness: PlanHarness) -> None:
    scope, task, run, snapshot = await _create(plan_harness, nodes=(_node("only"),))
    with pytest.raises(PlanTerminalError, match="every current node"):
        await plan_harness.store.finalize(
            scope=scope,
            plan_id=snapshot.plan.id,
            parent_task_id=task.id,
            parent_run_id=run.id,
            status=PlanStatus.COMPLETED,
            result="premature",
        )
    claim = await plan_harness.store.claim_ready_node(
        scope=scope, plan_id=snapshot.plan.id, owner="worker"
    )
    assert claim is not None
    await plan_harness.store.complete_node(scope=scope, claim=claim, output="verified finding")
    completed = await plan_harness.store.finalize(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=task.id,
        parent_run_id=run.id,
        status=PlanStatus.COMPLETED,
        result="parent synthesis",
    )
    assert completed.plan.status is PlanStatus.COMPLETED
    assert completed.plan.output == "parent synthesis"
