from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select

from leo.harness.models import (
    EventDraft,
    EventType,
    OriginRef,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    Thread,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
)
from leo.harness.plan_models import (
    DelegationStatus,
    PlanNodeDefinition,
    PlanNodeStatus,
)
from leo.harness.store_errors import ConcurrencyError
from leo.harness.transitions import (
    cancel_task_and_run,
    start_task_and_run,
    time_out_task_and_run,
)
from leo.integrations.fake import FixedClock
from leo.integrations.slack.events import (
    SlackBotPresence,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.integrations.system import UuidIdGenerator
from leo.persistence.outbox import DeliveryState, PostgresDeliveryOutbox
from leo.persistence.plan_store import PlanClaimConflictError, PostgresPlanStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import (
    ConversationAccessSnapshotRow,
    ConversationRow,
    DelegationRow,
    DeliveryOutboxRow,
    PlanNodeRow,
    PlanRevisionRow,
    PlanRow,
    RunEventRow,
    RunRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
    ThreadRow,
)
from leo.persistence.slack_ingress import PostgresSlackIngressAdmission
from leo.persistence.slack_scope import SlackScopeStoreInvariantError
from leo.persistence.task_leases import PostgresTaskLeaseStore, TaskLeaseConflictError


def _bounded_id(prefix: str, suffix: str) -> str:
    prefix_budget = 32 - len(suffix) - 1
    return f"{prefix[:prefix_budget]}-{suffix}"


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _job(harness: Any, *, label: str) -> SlackMentionJob:
    event_id = _bounded_id(f"Ev-{label}", harness.suffix)
    return SlackMentionJob(
        event_id=event_id,
        team_id=harness.team_id,
        channel_id=harness.channel_id,
        user_id=harness.user_id,
        message_ts=f"{int(harness.suffix[:8], 16)}.100",
        thread_root_ts=f"{int(harness.suffix[:8], 16)}.000",
        conversation_key=(
            f"slack:{harness.team_id}:{harness.channel_id}:{int(harness.suffix[:8], 16)}.000"
        ),
        prompt=f"Synthetic two-connection contract {label}.",
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=(harness.channel_id,),
        context_access_hash=build_context_access_hash(
            team_id=harness.team_id,
            user_id=harness.user_id,
            channel_id=harness.channel_id,
            context_conversation_ids=(harness.channel_id,),
        ),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _launch_models(
    harness: Any,
    job: SlackMentionJob,
    *,
    label: str,
) -> tuple[Thread, Task, Run]:
    thread = Thread(
        id=_bounded_id(f"thread-{label}", harness.suffix),
        scope=harness.scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=job.event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id(f"task-{label}", harness.suffix),
        thread_id=thread.id,
        scope=harness.scope,
        objective=job.prompt,
    )
    run = Run(
        id=_bounded_id(f"run-{label}", harness.suffix),
        task_id=task.id,
        scope=harness.scope,
    )
    return thread, task, run


async def _seed_launch(harness: Any, *, label: str) -> tuple[SlackMentionJob, Task, Run]:
    job = _job(harness, label=label)
    admission = PostgresSlackIngressAdmission(harness.sessions_a)
    admitted = await admission.admit(job, harness.scope, eligibility=_eligibility())
    assert admitted is not None
    thread, task, run = _launch_models(harness, job, label=label)
    await admission.materialize_initial_launch(
        event_id=job.event_id,
        thread=thread,
        task=task,
        run=run,
    )
    return job, task, run


async def _start_run(harness: Any, task: Task, run: Run) -> tuple[PostgresRunStore, Any]:
    store = PostgresRunStore(harness.sessions_a, FixedClock(), UuidIdGenerator())
    queued = await store.load(task.id, run.id, harness.scope)
    active_task, active_run = start_task_and_run(
        queued.task,
        queued.run,
        started_at=FixedClock().now(),
    )
    active = await store.commit(
        expected_task_version=queued.task.version,
        expected_run_version=queued.run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(
                type=EventType.TASK_STARTED,
                iteration=0,
                payload={"phase": "research"},
            ),
        ),
    )
    return store, active


def _verified_completion() -> VerifiedCompletion:
    return VerifiedCompletion(
        answer="The exact terminal winner committed.",
        claims=(),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=(
                VerifierCheck(
                    name="two_connection_contract",
                    passed=True,
                    detail="The deterministic fixture permits a context-only completion.",
                ),
            ),
            retryable=False,
            allow_unsourced_completion=True,
        ),
    )


@pytest.mark.asyncio
async def test_two_connections_admit_one_canonical_event_and_reject_envelope_drift(
    two_connection_postgres: Any,
) -> None:
    harness = two_connection_postgres
    assert harness.backend_pids[0] != harness.backend_pids[1]
    job = _job(harness, label="admission-race")
    admission_a = PostgresSlackIngressAdmission(harness.sessions_a)
    admission_b = PostgresSlackIngressAdmission(harness.sessions_b)

    first, second = await asyncio.wait_for(
        asyncio.gather(
            admission_a.admit(job, harness.scope, eligibility=_eligibility()),
            admission_b.admit(job, harness.scope, eligibility=_eligibility()),
        ),
        timeout=15,
    )

    assert sum(item is not None for item in (first, second)) == 1
    with pytest.raises(SlackScopeStoreInvariantError, match="different envelope"):
        await admission_b.admit(
            job.model_copy(update={"prompt": "Drifted event envelope."}),
            harness.scope,
            eligibility=_eligibility(),
        )
    async with harness.sessions_a() as session:
        ingress_count = await session.scalar(
            select(func.count())
            .select_from(SlackIngressEventRow)
            .where(SlackIngressEventRow.event_id == job.event_id)
        )
        conversation_count = await session.scalar(
            select(func.count())
            .select_from(ConversationRow)
            .where(
                ConversationRow.team_id == job.team_id,
                ConversationRow.external_id == job.channel_id,
            )
        )
        snapshot_count = await session.scalar(
            select(func.count())
            .select_from(ConversationAccessSnapshotRow)
            .where(ConversationAccessSnapshotRow.ingress_event_id == job.event_id)
        )
        message_count = await session.scalar(
            select(func.count())
            .select_from(SanitizedMessageRow)
            .where(SanitizedMessageRow.external_event_id == job.event_id)
        )
        conversation = await session.scalar(
            select(ConversationRow).where(
                ConversationRow.team_id == job.team_id,
                ConversationRow.external_id == job.channel_id,
            )
        )
    assert (ingress_count, conversation_count, snapshot_count, message_count) == (1, 1, 1, 1)
    assert conversation is not None
    assert conversation.version == 1
    assert conversation.authority_source == job.conversation_authority_source
    assert conversation.bot_presence == job.bot_presence.value


@pytest.mark.asyncio
async def test_two_connections_materialize_one_canonical_launch(
    two_connection_postgres: Any,
) -> None:
    harness = two_connection_postgres
    job = _job(harness, label="launch-race")
    admission_a = PostgresSlackIngressAdmission(harness.sessions_a)
    admission_b = PostgresSlackIngressAdmission(harness.sessions_b)
    admitted = await admission_a.admit(job, harness.scope, eligibility=_eligibility())
    assert admitted is not None
    thread, task, run = _launch_models(harness, job, label="launch-race")

    first, second = await asyncio.wait_for(
        asyncio.gather(
            admission_a.materialize_initial_launch(
                event_id=job.event_id,
                thread=thread,
                task=task,
                run=run,
            ),
            admission_b.materialize_initial_launch(
                event_id=job.event_id,
                thread=thread,
                task=task,
                run=run,
            ),
        ),
        timeout=15,
    )

    assert {first.created, second.created} == {False, True}
    assert first.thread_id == second.thread_id == thread.id
    assert first.task_id == second.task_id == task.id
    assert first.run_id == second.run_id == run.id
    async with harness.sessions_a() as session:
        thread_count = await session.scalar(
            select(func.count()).select_from(ThreadRow).where(ThreadRow.id == thread.id)
        )
        task_count = await session.scalar(
            select(func.count()).select_from(TaskRow).where(TaskRow.id == task.id)
        )
        run_count = await session.scalar(
            select(func.count()).select_from(RunRow).where(RunRow.id == run.id)
        )
        ingress = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == job.event_id)
        )
    assert (thread_count, task_count, run_count) == (1, 1, 1)
    assert ingress is not None
    assert ingress.task_id == task.id
    assert ingress.launch_status == "queued"


@pytest.mark.asyncio
async def test_two_connections_claim_reclaim_and_fence_stale_task_lease_owner(
    two_connection_postgres: Any,
) -> None:
    harness = two_connection_postgres
    _job_record, task, run = await _seed_launch(harness, label="lease-race")
    clock = FixedClock()
    store_a = PostgresTaskLeaseStore(harness.sessions_a, UuidIdGenerator())
    store_b = PostgresTaskLeaseStore(harness.sessions_b, UuidIdGenerator())

    first, second = await asyncio.wait_for(
        asyncio.gather(
            store_a.claim_task(
                task.id,
                "worker-a",
                lease_seconds=60,
                now=clock.now(),
            ),
            store_b.claim_task(
                task.id,
                "worker-b",
                lease_seconds=60,
                now=clock.now(),
            ),
        ),
        timeout=15,
    )

    winner = first or second
    assert winner is not None
    assert sum(item is not None for item in (first, second)) == 1
    reclaimer = store_b if winner.owner == "worker-a" else store_a
    reclaimed = await reclaimer.claim_task(
        task.id,
        "worker-reclaimer",
        lease_seconds=60,
        now=clock.now().replace(microsecond=0) + timedelta(seconds=61),
    )
    assert reclaimed is not None
    assert reclaimed.attempt == 2
    with pytest.raises(TaskLeaseConflictError, match="stale or owned"):
        await store_a.heartbeat(
            winner,
            now=clock.now().replace(microsecond=0) + timedelta(seconds=62),
        )
    refreshed = await reclaimer.heartbeat(
        reclaimed,
        now=clock.now().replace(microsecond=0) + timedelta(seconds=62),
    )
    assert refreshed.token == reclaimed.token
    assert refreshed.task_id == run.task_id


@pytest.mark.asyncio
async def test_two_connections_converge_on_plan_child_and_fence_reclaimed_node_owner(
    two_connection_postgres: Any,
) -> None:
    harness = two_connection_postgres
    assert harness.backend_pids[0] != harness.backend_pids[1]
    _job_record, parent_task, parent_run = await _seed_launch(
        harness,
        label="plan-node-race",
    )
    run_store, active = await _start_run(harness, parent_task, parent_run)
    child_thread = Thread(
        id=_bounded_id("thread-plan-child", harness.suffix),
        scope=harness.scope,
        origin=OriginRef(
            provider="plan-contract",
            external_thread_id=f"plan-child-{harness.suffix}",
        ),
    )
    child_task = Task(
        id=_bounded_id("task-plan-child", harness.suffix),
        thread_id=child_thread.id,
        scope=harness.scope,
        objective="Execute the canonical delegated child.",
    )
    child_run = Run(
        id=_bounded_id("run-plan-child", harness.suffix),
        task_id=child_task.id,
        scope=harness.scope,
    )
    await run_store.seed(child_thread, child_task, child_run)

    clock = FixedClock()
    plan_a = PostgresPlanStore(harness.sessions_a, clock, UuidIdGenerator())
    plan_b = PostgresPlanStore(harness.sessions_b, clock, UuidIdGenerator())
    nodes = (
        PlanNodeDefinition(
            key="research",
            objective="Produce one bounded delegated finding.",
            max_attempts=2,
        ),
    )
    create_arguments = {
        "scope": harness.scope,
        "parent_task_id": active.task.id,
        "parent_run_id": active.run.id,
        "idempotency_key": f"plan-node-race-{harness.suffix}",
        "goal": "Prove canonical plan and child identity under contention.",
        "nodes": nodes,
    }
    created_a, created_b = await asyncio.wait_for(
        asyncio.gather(
            plan_a.create_or_load(**create_arguments),
            plan_b.create_or_load(**create_arguments),
        ),
        timeout=15,
    )

    assert created_a.plan.id == created_b.plan.id
    assert created_a.revisions[0].id == created_b.revisions[0].id
    assert created_a.current_nodes[0].id == created_b.current_nodes[0].id
    plan_id = created_a.plan.id

    first, second = await asyncio.wait_for(
        asyncio.gather(
            plan_a.claim_ready_node(
                scope=harness.scope,
                plan_id=plan_id,
                owner="plan-worker-a",
                lease_seconds=60,
                now=clock.now(),
            ),
            plan_b.claim_ready_node(
                scope=harness.scope,
                plan_id=plan_id,
                owner="plan-worker-b",
                lease_seconds=60,
                now=clock.now(),
            ),
        ),
        timeout=15,
    )
    winner = first or second
    assert winner is not None
    assert sum(item is not None for item in (first, second)) == 1
    assert winner.node_id == created_a.current_nodes[0].id

    attached_a, attached_b = await asyncio.wait_for(
        asyncio.gather(
            plan_a.attach_child(
                scope=harness.scope,
                claim=winner,
                child_task_id=child_task.id,
                child_run_id=child_run.id,
            ),
            plan_b.attach_child(
                scope=harness.scope,
                claim=winner,
                child_task_id=child_task.id,
                child_run_id=child_run.id,
            ),
        ),
        timeout=15,
    )
    assert attached_a.current_nodes[0].child_task_id == child_task.id
    assert attached_a.current_nodes[0].child_run_id == child_run.id
    assert attached_b.current_nodes[0].child_task_id == child_task.id
    assert attached_b.current_nodes[0].child_run_id == child_run.id
    assert len(attached_a.delegations) == len(attached_b.delegations) == 1

    reclaimer = plan_b if winner.owner == "plan-worker-a" else plan_a
    reclaimed = await reclaimer.claim_ready_node(
        scope=harness.scope,
        plan_id=plan_id,
        owner="plan-worker-reclaimer",
        lease_seconds=60,
        now=winner.expires_at,
    )
    assert reclaimed is not None
    assert reclaimed.node_id == winner.node_id
    assert reclaimed.attempt == 2
    assert reclaimed.token != winner.token

    with pytest.raises(PlanClaimConflictError, match="stale, expired, or owned elsewhere"):
        await plan_a.complete_node(
            scope=harness.scope,
            claim=winner,
            output="A stale owner must not commit this result.",
            now=winner.expires_at,
        )
    await reclaimer.attach_child(
        scope=harness.scope,
        claim=reclaimed,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
        now=winner.expires_at,
    )

    restarted = PostgresPlanStore(harness.sessions_a, clock, UuidIdGenerator())
    replayed = await restarted.replay(scope=harness.scope, plan_id=plan_id)
    current = replayed.current_nodes[0]
    assert current.status is PlanNodeStatus.RUNNING
    assert current.attempt == 2
    assert current.claim_owner == "plan-worker-reclaimer"
    assert current.claim_token == reclaimed.token
    assert current.child_task_id == child_task.id
    assert current.child_run_id == child_run.id
    assert [(item.attempt, item.status) for item in replayed.delegations] == [
        (1, DelegationStatus.SUPERSEDED),
        (2, DelegationStatus.RUNNING),
    ]

    async with harness.sessions_b() as session:
        plan_count = await session.scalar(
            select(func.count())
            .select_from(PlanRow)
            .where(PlanRow.organization_id == harness.scope.organization_id)
        )
        revision_count = await session.scalar(
            select(func.count())
            .select_from(PlanRevisionRow)
            .where(PlanRevisionRow.organization_id == harness.scope.organization_id)
        )
        node_count = await session.scalar(
            select(func.count())
            .select_from(PlanNodeRow)
            .where(PlanNodeRow.organization_id == harness.scope.organization_id)
        )
        delegation_count = await session.scalar(
            select(func.count())
            .select_from(DelegationRow)
            .where(DelegationRow.organization_id == harness.scope.organization_id)
        )
    assert (plan_count, revision_count, node_count, delegation_count) == (1, 1, 1, 2)


@pytest.mark.asyncio
async def test_two_connections_repair_one_final_intent_and_one_dispatcher_claims_it(
    two_connection_postgres: Any,
) -> None:
    harness = two_connection_postgres
    job, task, run = await _seed_launch(harness, label="outbox-race")
    run_store, active = await _start_run(harness, task, run)
    timed_out_task, timed_out_run = time_out_task_and_run(
        active.task,
        active.run,
        "two_connection_timeout",
        usage=active.run.usage,
    )
    await run_store.commit(
        expected_task_version=active.task.version,
        expected_run_version=active.run.version,
        task=timed_out_task,
        run=timed_out_run,
        events=(
            EventDraft(
                type=EventType.RUN_TIMED_OUT,
                iteration=active.run.iteration,
                payload={"reason": "two_connection_timeout"},
            ),
        ),
    )
    outbox_a = PostgresDeliveryOutbox(harness.sessions_a, UuidIdGenerator())
    outbox_b = PostgresDeliveryOutbox(harness.sessions_b, UuidIdGenerator())
    first, second = await asyncio.wait_for(
        asyncio.gather(
            outbox_a.reconcile_terminal(
                lambda _task, _run: "safe timeout",
                payload_version=2,
                task_id=task.id,
                run_id=run.id,
                ingress_event_id=job.event_id,
            ),
            outbox_b.reconcile_terminal(
                lambda _task, _run: "safe timeout",
                payload_version=2,
                task_id=task.id,
                run_id=run.id,
                ingress_event_id=job.event_id,
            ),
        ),
        timeout=15,
    )

    repaired_ids = {intent.id for intent in (*first, *second)}
    assert len(repaired_ids) == 1
    intent_id = repaired_ids.pop()
    claimed_a, claimed_b = await asyncio.wait_for(
        asyncio.gather(
            outbox_a.claim_next("dispatcher-a", intent_id=intent_id),
            outbox_b.claim_next("dispatcher-b", intent_id=intent_id),
        ),
        timeout=15,
    )
    claimed = claimed_a or claimed_b
    assert claimed is not None
    assert sum(item is not None for item in (claimed_a, claimed_b)) == 1
    lease, _intent = claimed
    await outbox_a.mark_delivered(lease, "1787361066.999999")
    assert (
        await outbox_b.reconcile_terminal(
            lambda _task, _run: "must not duplicate",
            payload_version=2,
            task_id=task.id,
            run_id=run.id,
            ingress_event_id=job.event_id,
        )
        == ()
    )
    async with harness.sessions_b() as session:
        row = await session.scalar(
            select(DeliveryOutboxRow).where(DeliveryOutboxRow.id == intent_id)
        )
    assert row is not None
    assert row.state == DeliveryState.DELIVERED.value
    assert row.receipt_message_ts == "1787361066.999999"


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_kind", ("timeout", "cancel"))
async def test_two_connections_have_one_completion_or_terminal_control_winner(
    two_connection_postgres: Any,
    terminal_kind: str,
) -> None:
    harness = two_connection_postgres
    _job_record, task, run = await _seed_launch(harness, label=f"terminal-{terminal_kind}")
    store_a, active = await _start_run(harness, task, run)
    store_b = PostgresRunStore(harness.sessions_b, FixedClock(), UuidIdGenerator())
    if terminal_kind == "timeout":
        terminal_task, terminal_run = time_out_task_and_run(
            active.task,
            active.run,
            "two_connection_timeout",
            usage=active.run.usage,
        )
        terminal_event = EventDraft(
            type=EventType.RUN_TIMED_OUT,
            iteration=active.run.iteration,
            payload={"reason": "two_connection_timeout"},
        )
        competing_status = RunStatus.TIMED_OUT
    else:
        terminal_task, terminal_run = cancel_task_and_run(
            active.task,
            active.run,
            "two_connection_cancel",
        )
        terminal_event = EventDraft(
            type=EventType.RUN_CANCELLED,
            iteration=active.run.iteration,
            payload={"reason": "two_connection_cancel"},
        )
        competing_status = RunStatus.CANCELLED

    completed, terminal = await asyncio.wait_for(
        asyncio.gather(
            store_a.complete_verified(
                expected_task_version=active.task.version,
                expected_run_version=active.run.version,
                task_id=task.id,
                run_id=run.id,
                scope=harness.scope,
                usage=active.run.usage,
                completion=_verified_completion(),
            ),
            store_b.commit(
                expected_task_version=active.task.version,
                expected_run_version=active.run.version,
                task=terminal_task,
                run=terminal_run,
                events=(terminal_event,),
            ),
            return_exceptions=True,
        ),
        timeout=15,
    )

    outcomes = (completed, terminal)
    assert sum(isinstance(item, ConcurrencyError) for item in outcomes) == 1
    assert sum(not isinstance(item, BaseException) for item in outcomes) == 1
    final = await store_a.load(task.id, run.id, harness.scope)
    assert final.run.status in {RunStatus.COMPLETED, competing_status}
    if final.run.status is RunStatus.COMPLETED:
        assert final.task.status is TaskStatus.COMPLETED
    elif final.run.status is RunStatus.TIMED_OUT:
        assert final.task.status is TaskStatus.FAILED
    else:
        assert final.task.status is TaskStatus.CANCELLED
    async with harness.sessions_a() as session:
        terminal_event_count = await session.scalar(
            select(func.count())
            .select_from(RunEventRow)
            .where(
                RunEventRow.run_id == run.id,
                RunEventRow.type.in_(
                    (
                        EventType.RUN_COMPLETED.value,
                        EventType.RUN_TIMED_OUT.value,
                        EventType.RUN_CANCELLED.value,
                    )
                ),
            )
        )
    assert terminal_event_count == 1
