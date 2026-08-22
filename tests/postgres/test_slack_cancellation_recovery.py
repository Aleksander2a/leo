from __future__ import annotations

from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import (
    EventDraft,
    EventType,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
)
from leo.harness.plan_models import PlanNodeDefinition, PlanStatus
from leo.harness.transitions import start_task_and_run, time_out_task_and_run
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.slack.cancellation import SlackCancellationOutcome
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
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.schema import PlanRow, RunRow, SlackIngressEventRow, TaskRow
from leo.persistence.slack_cancellation import PostgresSlackCancellationService
from leo.persistence.slack_ingress import PostgresSlackIngressAdmission
from leo.persistence.task_leases import PostgresTaskLeaseStore
from leo.worker.slack_conversation import reconcile_terminal_parent_plans


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _job(
    suffix: str,
    *,
    event_id: str,
    message_ts: str,
    prompt: str,
    user_id: str = "U-cancel-owner",
) -> SlackMentionJob:
    channel_id = f"C-cancel-{suffix}"
    thread_root_ts = "100.0"
    context_ids = (channel_id,)
    return SlackMentionJob(
        event_id=event_id,
        team_id=f"T-cancel-{suffix}",
        channel_id=channel_id,
        user_id=user_id,
        message_ts=message_ts,
        thread_root_ts=thread_root_ts,
        conversation_key=f"slack:T-cancel-{suffix}:{channel_id}:{thread_root_ts}",
        prompt=prompt,
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=context_ids,
        context_access_hash=build_context_access_hash(
            team_id=f"T-cancel-{suffix}",
            user_id=user_id,
            channel_id=channel_id,
            context_conversation_ids=context_ids,
        ),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


class _LaunchPreparer:
    def __init__(self, ingress: PostgresSlackIngressAdmission, suffix: str) -> None:
        self._ingress = ingress
        self._suffix = suffix

    async def prepare(self, admitted):  # type: ignore[no-untyped-def]
        job = admitted.job
        scope = admitted.resolution.scope
        thread = Thread(
            id=f"thread-control-{self._suffix}-{job.event_id}",
            scope=scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=job.conversation_key,
                external_event_id=job.event_id,
                external_channel_id=job.channel_id,
            ),
        )
        task = Task(
            id=f"task-control-{self._suffix}-{job.event_id}",
            thread_id=thread.id,
            scope=scope,
            objective=job.prompt,
        )
        run = Run(
            id=f"run-control-{self._suffix}-{job.event_id}",
            task_id=task.id,
            scope=scope,
        )
        launch = await self._ingress.materialize_initial_launch(
            event_id=job.event_id,
            thread=thread,
            task=task,
            run=run,
        )
        return await self._ingress.load_linked_mention(launch.task_id)


class _CrashAt:
    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._armed = True

    def __call__(self, stage: str) -> None:
        if self._armed and stage == self._stage:
            self._armed = False
            raise RuntimeError(f"injected:{stage}")


def _seed_for_job(
    suffix: str,
    job: SlackMentionJob,
    scope: ScopeKey,
) -> tuple[Thread, Task, Run]:
    thread = Thread(
        id=f"thread-crash-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=job.event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=f"task-crash-{suffix}",
        thread_id=thread.id,
        scope=scope,
        objective=job.prompt,
    )
    return thread, task, Run(id=f"run-crash-{suffix}", task_id=task.id, scope=scope)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        "admission_after_reserve",
        "admission_after_scope",
        "admission_before_commit",
        "admission_after_commit",
    ],
)
async def test_admission_crash_matrix_rolls_back_or_leaves_recoverable_intent(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    stage: str,
) -> None:
    suffix = f"{stage[-6:]}-{uuid4().hex[:8]}"
    scope = ScopeKey(
        organization_id=f"org-admit-crash-{suffix}",
        strategy_id=f"strategy-admit-crash-{suffix}",
    )
    job = _job(
        suffix,
        event_id=f"Ev-admit-crash-{suffix}",
        message_ts="100.0",
        prompt="Arbitrary admitted work survives process death.",
    )
    faulted = PostgresSlackIngressAdmission(
        preserved_postgres_sessions,
        fault_hook=_CrashAt(stage),
    )
    with pytest.raises(RuntimeError, match=f"injected:{stage}"):
        await faulted.admit(job, scope, eligibility=_eligibility())

    clean = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    async with preserved_postgres_sessions() as session:
        persisted = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == job.event_id)
        )
    if stage == "admission_after_commit":
        assert persisted is not None and persisted.launch_status == "unlaunched"
        assert await clean.admit(job, scope, eligibility=_eligibility()) is None
    else:
        assert persisted is None
        assert await clean.admit(job, scope, eligibility=_eligibility()) is not None

    def seed_factory(recovered_job: SlackMentionJob, recovered_scope: ScopeKey):
        return _seed_for_job(suffix, recovered_job, recovered_scope)

    recovered = await clean.recover_startup_launches(
        seed_factory,
        include_queued=True,
        event_ids=(job.event_id,),
    )
    assert len(recovered) == 1
    assert recovered[0].launch is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stage",
    [
        "launch_after_materializing",
        "launch_before_commit",
        "launch_after_commit",
    ],
)
async def test_launch_crash_matrix_converges_on_one_canonical_task_run(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    stage: str,
) -> None:
    suffix = f"{stage[-6:]}-{uuid4().hex[:8]}"
    scope = ScopeKey(
        organization_id=f"org-launch-crash-{suffix}",
        strategy_id=f"strategy-launch-crash-{suffix}",
    )
    job = _job(
        suffix,
        event_id=f"Ev-launch-crash-{suffix}",
        message_ts="100.0",
        prompt="Materialize one canonical launch.",
    )
    clean = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    assert await clean.admit(job, scope, eligibility=_eligibility()) is not None
    thread, task, run = _seed_for_job(suffix, job, scope)
    faulted = PostgresSlackIngressAdmission(
        preserved_postgres_sessions,
        fault_hook=_CrashAt(stage),
    )
    with pytest.raises(RuntimeError, match=f"injected:{stage}"):
        await faulted.materialize_initial_launch(
            event_id=job.event_id,
            thread=thread,
            task=task,
            run=run,
        )

    canonical = await clean.materialize_initial_launch(
        event_id=job.event_id,
        thread=thread,
        task=task,
        run=run,
    )
    assert canonical.task_id == task.id
    assert canonical.run_id == run.id
    assert canonical.created is (stage != "launch_after_commit")
    async with preserved_postgres_sessions() as session:
        task_count = len(
            (await session.scalars(select(TaskRow.id).where(TaskRow.id == task.id))).all()
        )
        run_count = len((await session.scalars(select(RunRow.id).where(RunRow.id == run.id))).all())
    assert (task_count, run_count) == (1, 1)


@pytest.mark.asyncio
async def test_crash_after_launch_commit_before_notify_is_resignalled_on_startup(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex[:12]
    scope = ScopeKey(
        organization_id=f"org-notify-crash-{suffix}",
        strategy_id=f"strategy-notify-crash-{suffix}",
    )
    job = _job(
        suffix,
        event_id=f"Ev-notify-crash-{suffix}",
        message_ts="100.0",
        prompt="Committed work receives no volatile wake-up.",
    )
    admission = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    assert await admission.admit(job, scope, eligibility=_eligibility()) is not None
    thread, task, run = _seed_for_job(suffix, job, scope)
    await admission.materialize_initial_launch(
        event_id=job.event_id,
        thread=thread,
        task=task,
        run=run,
    )

    def should_not_seed(
        recovered_job: SlackMentionJob,
        recovered_scope: ScopeKey,
    ) -> tuple[Thread, Task, Run]:
        del recovered_job, recovered_scope
        raise AssertionError("queued launch recovery must reuse the committed seed")

    recovered = await admission.recover_startup_launches(
        should_not_seed,
        include_queued=True,
        event_ids=(job.event_id,),
    )
    assert len(recovered) == 1
    assert recovered[0].launch is not None
    assert recovered[0].launch.task_id == task.id


@pytest.mark.asyncio
async def test_authorized_slack_cancel_terminalizes_parent_and_control_idempotently(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex[:12]
    scope = ScopeKey(
        organization_id=f"org-cancel-{suffix}",
        strategy_id=f"strategy-cancel-{suffix}",
    )
    ingress = PostgresSlackIngressAdmission(preserved_postgres_sessions)
    root_job = _job(
        suffix,
        event_id=f"Ev-root-{suffix}",
        message_ts="100.0",
        prompt="Work until explicitly cancelled.",
    )
    root = await ingress.admit(root_job, scope, eligibility=_eligibility())
    assert root is not None
    root_thread = Thread(
        id=f"thread-root-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=root_job.conversation_key,
            external_event_id=root_job.event_id,
            external_channel_id=root_job.channel_id,
        ),
    )
    root_task = Task(
        id=f"task-root-{suffix}",
        thread_id=root_thread.id,
        scope=scope,
        objective=root_job.prompt,
    )
    root_run = Run(id=f"run-root-{suffix}", task_id=root_task.id, scope=scope)
    await ingress.materialize_initial_launch(
        event_id=root_job.event_id,
        thread=root_thread,
        task=root_task,
        run=root_run,
    )
    lease = await PostgresTaskLeaseStore(
        preserved_postgres_sessions,
        SequentialIdGenerator(),
    ).claim_task(root_task.id, "cancel-worker", lease_seconds=60)
    assert lease is not None
    root_store = LeaseBoundRunStore(
        PostgresRunStore(preserved_postgres_sessions, FixedClock(), UuidIdGenerator()),
        lease,
    )
    queued = await root_store.load(root_task.id, root_run.id, scope)
    active_task, active_run = start_task_and_run(
        queued.task,
        queued.run,
        started_at=FixedClock().now(),
    )
    await root_store.commit(
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

    cancel_job = _job(
        suffix,
        event_id=f"Ev-cancel-{suffix}",
        message_ts="101.0",
        prompt="please cancel this task",
    )
    cancel_admitted = await ingress.admit(cancel_job, scope, eligibility=_eligibility())
    assert cancel_admitted is not None
    service = PostgresSlackCancellationService(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
        ingress,
    )
    preparer = _LaunchPreparer(ingress, suffix)
    first = await service.handle(cancel_admitted, preparer)
    second = await service.handle(first.admitted, preparer)

    assert first.outcome is SlackCancellationOutcome.APPLIED
    assert second == first
    parent = await PostgresRunStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(root_task.id, root_run.id, scope)
    assert parent.task.status is TaskStatus.CANCELLED
    assert parent.run.status is RunStatus.CANCELLED
    assert parent.run.terminal_reason == "slack_user_cancelled"
    assert first.admitted.launch is not None
    control = await PostgresRunStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(
        first.admitted.launch.task_id,
        first.admitted.launch.run_id,
        first.admitted.resolution.scope,
    )
    assert control.task.status is TaskStatus.CANCELLED
    assert control.run.terminal_reason == "slack_cancel_control_applied"
    async with preserved_postgres_sessions() as session:
        ingress_row = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == cancel_job.event_id)
        )
    assert ingress_row is not None
    assert ingress_row.status == "cancel_control_applied"


@pytest.mark.asyncio
async def test_startup_reconciliation_closes_terminal_parent_plan_and_child(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    suffix = uuid4().hex[:12]
    scope = ScopeKey(
        organization_id=f"org-timeout-{suffix}",
        strategy_id=f"strategy-timeout-{suffix}",
    )
    clock = FixedClock()
    ids = UuidIdGenerator()
    run_store = PostgresRunStore(preserved_postgres_sessions, clock, ids)
    parent_thread = Thread(
        id=f"thread-timeout-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="m2-timeout-recovery",
            external_thread_id=f"parent-{suffix}",
        ),
    )
    parent_task = Task(
        id=f"task-timeout-{suffix}",
        thread_id=parent_thread.id,
        scope=scope,
        objective="Parent whose timeout committed before child propagation.",
    )
    parent_run = Run(id=f"run-timeout-{suffix}", task_id=parent_task.id, scope=scope)
    await run_store.seed(parent_thread, parent_task, parent_run)
    started_task, started_run = start_task_and_run(
        parent_task,
        parent_run,
        started_at=clock.now(),
    )
    parent_active = await run_store.commit(
        expected_task_version=parent_task.version,
        expected_run_version=parent_run.version,
        task=started_task,
        run=started_run,
        events=(
            EventDraft(
                type=EventType.TASK_STARTED,
                iteration=0,
                payload={"phase": "research"},
            ),
        ),
    )
    plan_store = PostgresPlanStore(preserved_postgres_sessions, clock, ids)
    plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        idempotency_key=f"timeout-repair-{suffix}",
        goal="Attach one child before the timeout crash window.",
        nodes=(PlanNodeDefinition(key="child", objective="Remain active until fenced."),),
    )
    claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=plan.plan.id,
        owner="timeout-child-worker",
    )
    assert claim is not None
    child_thread = Thread(
        id=f"thread-timeout-child-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="m2-timeout-recovery",
            external_thread_id=f"child-{suffix}",
        ),
    )
    child_task = Task(
        id=f"task-timeout-child-{suffix}",
        thread_id=child_thread.id,
        scope=scope,
        objective="Child that must be cancelled after restart.",
        parent_task_id=parent_task.id,
        continuation_kind="subagent",
    )
    child_run = Run(
        id=f"run-timeout-child-{suffix}",
        task_id=child_task.id,
        scope=scope,
    )
    await run_store.seed(child_thread, child_task, child_run)
    await plan_store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    child_started_task, child_started_run = start_task_and_run(
        child_task,
        child_run,
        started_at=clock.now(),
    )
    await run_store.commit(
        expected_task_version=child_task.version,
        expected_run_version=child_run.version,
        task=child_started_task,
        run=child_started_run,
        events=(
            EventDraft(
                type=EventType.TASK_STARTED,
                iteration=0,
                payload={"phase": "research"},
            ),
        ),
    )
    timed_out_task, timed_out_run = time_out_task_and_run(
        parent_active.task,
        parent_active.run,
        "slack_runtime_deadline_exceeded",
        usage=parent_active.run.usage,
    )
    await run_store.commit(
        expected_task_version=parent_active.task.version,
        expected_run_version=parent_active.run.version,
        task=timed_out_task,
        run=timed_out_run,
        events=(
            EventDraft(
                type=EventType.RUN_TIMED_OUT,
                iteration=timed_out_run.iteration,
                payload={"reason": "slack_runtime_deadline_exceeded"},
            ),
        ),
    )

    assert await reconcile_terminal_parent_plans(preserved_postgres_sessions) == 1
    assert await reconcile_terminal_parent_plans(preserved_postgres_sessions) == 0
    child_after = await run_store.load(child_task.id, child_run.id, scope)
    assert child_after.task.status is TaskStatus.CANCELLED
    assert child_after.run.status is RunStatus.CANCELLED
    assert child_after.run.terminal_reason == "parent_deadline_exceeded"
    async with preserved_postgres_sessions() as session:
        plan_row = await session.scalar(select(PlanRow).where(PlanRow.id == plan.plan.id))
        active_child = await session.scalar(
            select(TaskRow.id).where(
                TaskRow.id == child_task.id,
                TaskRow.status.in_(("queued", "active", "requires_action")),
            )
        )
        parent_row = await session.scalar(select(RunRow).where(RunRow.id == parent_run.id))
    assert plan_row is not None and plan_row.status == PlanStatus.FAILED.value
    assert active_child is None
    assert parent_row is not None and parent_row.status == RunStatus.TIMED_OUT.value
