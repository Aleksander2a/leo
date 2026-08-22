from __future__ import annotations

import warnings
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import update
from sqlalchemy.exc import SAWarning
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import EventDraft, EventType, OriginRef, Run, ScopeKey, Task, Thread
from leo.harness.plan_models import PlanNodeDefinition
from leo.harness.ports import IdGenerator
from leo.harness.transitions import start_task_and_run
from leo.health import HealthState, probe_database, probe_operational_metadata
from leo.integrations.fake import FixedClock
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import (
    ConversationActorMembershipRow,
    ConversationRow,
    DelegationRow,
)


def _fixture_suffix() -> str:
    return uuid4().hex[:12]


def _bounded_id(prefix: str, suffix: str, counter: int | None = None) -> str:
    tail = suffix if counter is None else f"{suffix}-{counter:x}"
    prefix_budget = 32 - len(tail) - 1
    return f"{prefix[:prefix_budget]}-{tail}"


class _UniqueIds(IdGenerator):
    def __init__(self) -> None:
        self._suffix = _fixture_suffix()
        self._counter = 0

    def new(self, prefix: str) -> str:
        self._counter += 1
        return _bounded_id(prefix, self._suffix, self._counter)


async def _seed_running_parent(
    sessions: async_sessionmaker[AsyncSession],
    *,
    scope: ScopeKey,
    suffix: str,
    clock: FixedClock,
    ids: IdGenerator,
) -> tuple[PostgresRunStore, str, str]:
    task_id = f"task-{suffix}"
    run_id = f"run-{suffix}"
    thread_id = f"thread-{suffix}"
    store = PostgresRunStore(sessions, clock, ids)
    bundle = await store.seed(
        Thread(
            id=thread_id,
            scope=scope,
            origin=OriginRef(provider="fixture", external_thread_id=f"health-{suffix}"),
        ),
        Task(
            id=task_id,
            thread_id=thread_id,
            scope=scope,
            objective="health orchestration fixture",
        ),
        Run(id=run_id, task_id=task_id, scope=scope),
    )
    task, run = start_task_and_run(bundle.task, bundle.run, started_at=clock.now())
    await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=task,
        run=run,
        events=(EventDraft(type=EventType.TASK_STARTED, iteration=0),),
    )
    return store, task_id, run_id


@pytest.mark.asyncio
async def test_postgres_health_reports_queue_work_without_fake_success(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = _fixture_suffix()
    scope = ScopeKey(
        organization_id=f"health-org-{suffix}",
        strategy_id=f"health-strategy-{suffix}",
    )
    observed_at = datetime(2026, 8, 21, tzinfo=UTC)
    _database, queue_before, _outbox, _last_success = await probe_database(
        preserved_postgres_sessions,
        observed_at=observed_at,
    )
    store = PostgresRunStore(preserved_postgres_sessions, FixedClock(), _UniqueIds())
    await store.seed(
        Thread(
            id=f"thread-{suffix}",
            scope=scope,
            origin=OriginRef(provider="fixture", external_thread_id=f"health-{suffix}"),
        ),
        Task(
            id=f"task-{suffix}",
            thread_id=f"thread-{suffix}",
            scope=scope,
            objective="health probe fixture",
        ),
        Run(id=f"run-{suffix}", task_id=f"task-{suffix}", scope=scope),
    )

    database, queue, outbox, last_success = await probe_database(
        preserved_postgres_sessions,
        observed_at=observed_at,
    )

    assert database.state is HealthState.OK
    assert queue.state is HealthState.DEGRADED
    assert int(queue.details["queued"]) == int(queue_before.details["queued"]) + 1
    assert outbox.state in {HealthState.OK, HealthState.DEGRADED, HealthState.UNHEALTHY}
    assert last_success.state in {HealthState.OK, HealthState.DEGRADED, HealthState.UNKNOWN}


@pytest.mark.asyncio
async def test_postgres_operational_health_uses_persisted_conversation_membership_age(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = _fixture_suffix()
    observed_at = datetime(2099, 8, 21, 12, 0, tzinfo=UTC)
    team_id = f"T{suffix.upper()}"
    channel_id = f"C{suffix[::-1].upper()}"
    async with preserved_postgres_sessions() as session, session.begin():
        session.add(
            ConversationRow(
                id=f"conversation-{suffix}",
                provider="slack",
                team_id=team_id,
                external_id=channel_id,
                kind="channel",
                actor_id=None,
                version=1,
                created_at=observed_at,
                updated_at=observed_at,
            )
        )
        session.add(
            ConversationActorMembershipRow(
                id=f"membership-{suffix}",
                organization_id=f"org-health-{suffix}",
                team_id=team_id,
                actor_id=f"U{suffix.upper()}",
                conversation_external_id=channel_id,
                status="active",
                source_kind="exact_destination",
                context_access_hash="a" * 64,
                version=1,
                observed_at=observed_at,
                created_at=observed_at,
                updated_at=observed_at,
            )
        )

    conversation, membership, orchestration, model = await probe_operational_metadata(
        preserved_postgres_sessions,
        observed_at=observed_at.replace(hour=13),
        organization_id=f"org-health-{suffix}",
        team_id=team_id,
        membership_stale_seconds=3_600,
    )

    assert conversation.state is HealthState.OK
    assert membership.state is HealthState.OK
    assert membership.age_seconds == 3_600
    assert orchestration.state is HealthState.OK
    assert model.state is HealthState.UNKNOWN


@pytest.mark.asyncio
async def test_orchestration_health_counts_each_table_without_cartesian_multiplication(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    observed_at = datetime.now(UTC)
    clock = FixedClock(observed_at)
    ids = _UniqueIds()
    suffix = _fixture_suffix()
    scope = ScopeKey(
        organization_id=f"health-safe-org-{suffix}",
        strategy_id=f"health-safe-strategy-{suffix}",
    )
    _, task_id, run_id = await _seed_running_parent(
        preserved_postgres_sessions,
        scope=scope,
        suffix=suffix,
        clock=clock,
        ids=ids,
    )
    _, _, before, _ = await probe_operational_metadata(
        preserved_postgres_sessions,
        observed_at=observed_at,
        organization_id=scope.organization_id,
        team_id=f"TH{suffix.upper()}",
    )
    plan_store = PostgresPlanStore(preserved_postgres_sessions, clock, ids)
    snapshot = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=task_id,
        parent_run_id=run_id,
        idempotency_key=f"health-safe-orchestration-{suffix}",
        goal="Prove independent health aggregates",
        nodes=(
            PlanNodeDefinition(
                key="only",
                objective="Remain running during the health probe",
            ),
        ),
    )
    claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="health-worker",
        lease_seconds=60,
        now=observed_at,
    )
    assert claim is not None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", SAWarning)
        _, _, orchestration, _ = await probe_operational_metadata(
            preserved_postgres_sessions,
            observed_at=observed_at,
            organization_id=scope.organization_id,
            team_id=f"TH{suffix.upper()}",
        )

    assert not [item for item in caught if issubclass(item.category, SAWarning)]
    expected_delta = {
        "active_plans": 1,
        "running_nodes": 1,
        "expired_node_leases": 0,
        "running_delegations": 1,
        "orphaned_running_nodes": 0,
        "orphaned_running_delegations": 0,
        "blocked_dependency_nodes": 0,
    }
    assert {
        key: int(orchestration.details[key]) - int(before.details[key]) for key in expected_delta
    } == expected_delta


@pytest.mark.asyncio
async def test_orchestration_health_fails_closed_for_orphaned_and_blocked_child_work(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    observed_at = datetime.now(UTC)
    clock = FixedClock(observed_at)
    ids = _UniqueIds()
    suffix = _fixture_suffix()
    scope = ScopeKey(
        organization_id=f"health-fault-org-{suffix}",
        strategy_id=f"health-fault-strategy-{suffix}",
    )
    _, task_id, run_id = await _seed_running_parent(
        preserved_postgres_sessions,
        scope=scope,
        suffix=suffix,
        clock=clock,
        ids=ids,
    )
    _, _, before, _ = await probe_operational_metadata(
        preserved_postgres_sessions,
        observed_at=observed_at,
        organization_id=scope.organization_id,
        team_id=f"TH{suffix.upper()}",
    )
    plan_store = PostgresPlanStore(preserved_postgres_sessions, clock, ids)
    orphan_plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=task_id,
        parent_run_id=run_id,
        idempotency_key=f"health-orphan-{suffix}",
        goal="Expose an orphaned running child",
        nodes=(PlanNodeDefinition(key="orphan", objective="Remain running"),),
    )
    orphan_claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=orphan_plan.plan.id,
        owner="health-worker",
        lease_seconds=60,
        now=observed_at,
    )
    assert orphan_claim is not None
    async with preserved_postgres_sessions() as session, session.begin():
        await session.execute(
            update(DelegationRow)
            .where(DelegationRow.node_id == orphan_claim.node_id)
            .values(
                status="superseded",
                error="injected_orphan",
                finished_at=observed_at,
            )
        )

    blocked_plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=task_id,
        parent_run_id=run_id,
        idempotency_key=f"health-blocked-{suffix}",
        goal="Expose a dependency blocked by failed child work",
        nodes=(
            PlanNodeDefinition(key="first", objective="Fail", max_attempts=1),
            PlanNodeDefinition(key="second", objective="Wait", depends_on=("first",)),
        ),
    )
    failed_claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=blocked_plan.plan.id,
        owner="health-worker",
        lease_seconds=60,
        now=observed_at,
    )
    assert failed_claim is not None
    await plan_store.fail_node(
        scope=scope,
        claim=failed_claim,
        error="injected_child_failure",
        now=observed_at,
    )

    _, _, orchestration, _ = await probe_operational_metadata(
        preserved_postgres_sessions,
        observed_at=observed_at,
        organization_id=scope.organization_id,
        team_id=f"TH{suffix.upper()}",
    )

    assert orchestration.state is HealthState.UNHEALTHY
    assert (
        int(orchestration.details["orphaned_running_nodes"])
        - int(before.details["orphaned_running_nodes"])
        == 1
    )
    assert (
        int(orchestration.details["blocked_dependency_nodes"])
        - int(before.details["blocked_dependency_nodes"])
        == 1
    )
