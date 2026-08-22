from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import select, text, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import OriginRef, Run, ScopeKey, Task, Thread
from leo.harness.ports import IdGenerator
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackBotPresence,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackLaunchRef,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.persistence.schema import RunRow, SlackIngressEventRow, TaskRow, ThreadRow
from leo.persistence.slack_ingress import (
    PostgresSlackIngressAdmission,
    SlackLaunchInvariantError,
)
from leo.persistence.task_leases import PostgresTaskLeaseStore
from leo.worker.slack_conversation import reconcile_admitted_slack_timeout


@pytest_asyncio.fixture
async def launch_sessions(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    yield preserved_postgres_sessions


class _UniqueIds(IdGenerator):
    def __init__(self) -> None:
        self._suffix = uuid4().hex[:12]
        self._counter = 0

    def new(self, prefix: str) -> str:
        self._counter += 1
        return _bounded_id(f"{prefix}-{self._counter:x}", self._suffix)


def _bounded_id(prefix: str, suffix: str) -> str:
    candidate = f"{prefix}-{suffix}"
    if len(candidate) <= 32:
        return candidate
    prefix_digest = sha256(prefix.encode("utf-8")).hexdigest()[:6]
    prefix_budget = 32 - len(suffix) - len(prefix_digest) - 2
    return f"{prefix[:prefix_budget]}-{prefix_digest}-{suffix}"


def _slack_timestamp(event_id: str, microseconds: int) -> str:
    epoch = int.from_bytes(sha256(event_id.encode("utf-8")).digest()[:6], "big")
    return f"{epoch}.{microseconds:06d}"


def _job(
    event_id: str,
    *,
    team_id: str = "T-launch",
    channel_id: str = "C-launch",
    user_id: str = "U-launch",
) -> SlackMentionJob:
    thread_root_ts = _slack_timestamp(event_id, 0)
    return SlackMentionJob(
        event_id=event_id,
        team_id=team_id,
        channel_id=channel_id,
        user_id=user_id,
        message_ts=_slack_timestamp(event_id, 1),
        thread_root_ts=thread_root_ts,
        conversation_key=f"slack:{team_id}:{channel_id}:{thread_root_ts}",
        prompt="quote NVDA",
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=(channel_id,),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
        context_access_hash=build_context_access_hash(
            team_id=team_id,
            user_id=user_id,
            channel_id=channel_id,
            context_conversation_ids=(channel_id,),
        ),
    )


def _unique_job(prefix: str) -> tuple[SlackMentionJob, ScopeKey, str]:
    suffix = uuid4().hex[:12]
    job = _job(
        _bounded_id(prefix, suffix),
        team_id=f"T{suffix.upper()}",
        channel_id=f"C{suffix[::-1].upper()}",
        user_id=f"U{suffix.upper()}",
    )
    return (
        job,
        ScopeKey(
            organization_id=_bounded_id("org-launch", suffix),
            strategy_id=_bounded_id("strategy-launch", suffix),
        ),
        suffix,
    )


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


@pytest.mark.asyncio
async def test_admitted_event_has_recoverable_launch_intent(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, _suffix = _unique_job("Ev-launch-intent")
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(
        job,
        scope,
        eligibility=_eligibility(),
    )

    async with launch_sessions() as session:
        row = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == job.event_id)
        )

    assert row is not None
    assert row.launch_status == "unlaunched"
    assert row.launch_attempt_count == 0
    assert row.launch_error is None
    assert row.task_id is None
    assert row.organization_id == scope.organization_id
    assert row.strategy_id == scope.strategy_id
    assert row.mapping_version == 1
    assert row.conversation_authority_source == "slack_conversations_info"
    assert row.bot_presence == "present"
    assert row.conversation_lifecycle == "active"
    assert row.external_provenance == "internal"
    assert row.membership_policy_version == 1


@pytest.mark.asyncio
async def test_launch_link_is_unique_and_indexed(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    async with launch_sessions() as session:
        index = (
            await session.execute(
                text(
                    """
                    select indexdef
                    from pg_indexes
                    where schemaname = 'public'
                      and indexname = 'uq_slack_ingress_task_id'
                    """
                )
            )
        ).scalar_one()

    assert "UNIQUE INDEX" in index.upper()
    assert "task_id" in index
    assert "IS NOT NULL" in index.upper()


@pytest.mark.asyncio
async def test_initial_launch_materializes_one_seed_and_is_idempotent(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-materialize")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, scope, eligibility=_eligibility())
    thread = Thread(
        id=_bounded_id("thread", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task", suffix), thread_id=thread.id, scope=scope, objective=job.prompt
    )
    run = Run(id=_bounded_id("run", suffix), task_id=task.id, scope=scope)

    first = await admission.materialize_initial_launch(
        event_id=event_id,
        thread=thread,
        task=task,
        run=run,
    )
    second = await admission.materialize_initial_launch(
        event_id=event_id,
        thread=thread,
        task=task,
        run=run,
    )

    assert first.thread_id == second.thread_id == thread.id
    assert first.task_id == second.task_id == task.id
    assert first.run_id == second.run_id == run.id
    assert first.created is True
    assert second.created is False
    async with launch_sessions() as session:
        ingress = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
        assert ingress is not None
        assert ingress.launch_status == "queued"
        assert ingress.task_id == task.id
        assert (
            await session.scalar(select(ThreadRow.id).where(ThreadRow.id == thread.id)) == thread.id
        )
        assert await session.scalar(select(TaskRow.id).where(TaskRow.id == task.id)) == task.id
        assert await session.scalar(select(RunRow.id).where(RunRow.id == run.id)) == run.id


@pytest.mark.asyncio
async def test_initial_launch_rejects_scope_drift_without_materializing(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, admitted_scope, suffix = _unique_job("Ev-launch-scope-drift")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, admitted_scope, eligibility=_eligibility())
    drifted_scope = ScopeKey(
        organization_id=_bounded_id("org-other", suffix),
        strategy_id=_bounded_id("strategy-other", suffix),
    )
    thread = Thread(
        id=_bounded_id("thread-drift", suffix),
        scope=drifted_scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task-drift", suffix),
        thread_id=thread.id,
        scope=drifted_scope,
        objective=job.prompt,
    )
    run = Run(id=_bounded_id("run-drift", suffix), task_id=task.id, scope=drifted_scope)

    with pytest.raises(SlackLaunchInvariantError, match="identity or scope"):
        await admission.materialize_initial_launch(
            event_id=event_id,
            thread=thread,
            task=task,
            run=run,
        )

    async with launch_sessions() as session:
        ingress = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
        assert ingress is not None
        assert ingress.launch_status == "unlaunched"
        assert ingress.task_id is None
        assert await session.scalar(select(ThreadRow.id).where(ThreadRow.id == thread.id)) is None


@pytest.mark.asyncio
async def test_linked_launch_status_update_preserves_launch_link(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-status")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, scope, eligibility=_eligibility())
    thread = Thread(
        id=_bounded_id("thread-status", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task-status", suffix),
        thread_id=thread.id,
        scope=scope,
        objective=job.prompt,
    )
    run = Run(id=_bounded_id("run-status", suffix), task_id=task.id, scope=scope)
    await admission.materialize_initial_launch(
        event_id=event_id,
        thread=thread,
        task=task,
        run=run,
    )

    await admission.mark_linked_status(event_id, "run_completed", None)

    async with launch_sessions() as session:
        ingress = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == event_id)
        )
    assert ingress is not None
    assert ingress.status == "run_completed"
    assert ingress.launch_status == "queued"
    assert ingress.task_id == task.id


@pytest.mark.asyncio
async def test_startup_recovery_materializes_and_resignals_without_cloning(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-recovery")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, scope, eligibility=_eligibility())

    def seed(recovered_job: SlackMentionJob, recovered_scope: ScopeKey) -> tuple[Thread, Task, Run]:
        thread = Thread(
            id=_bounded_id("thread-recovery", suffix),
            scope=recovered_scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=recovered_job.conversation_key,
                external_event_id=recovered_job.event_id,
                external_channel_id=recovered_job.channel_id,
            ),
        )
        task = Task(
            id=_bounded_id("task-recovery", suffix),
            thread_id=thread.id,
            scope=recovered_scope,
            objective=recovered_job.prompt,
        )
        return (
            thread,
            task,
            Run(id=_bounded_id("run-recovery", suffix), task_id=task.id, scope=recovered_scope),
        )

    first = await admission.recover_startup_launches(seed, event_ids=(event_id,))
    second = await admission.recover_startup_launches(
        lambda _job, _scope: (_ for _ in ()).throw(
            AssertionError("queued recovery must not reseed")
        ),
        event_ids=(event_id,),
    )

    assert len(first) == len(second) == 1
    assert first[0].launch == second[0].launch
    async with launch_sessions() as session:
        assert await session.scalar(
            select(ThreadRow.id).where(ThreadRow.id == _bounded_id("thread-recovery", suffix))
        ) == _bounded_id("thread-recovery", suffix)
        assert await session.scalar(
            select(TaskRow.id).where(TaskRow.id == _bounded_id("task-recovery", suffix))
        ) == _bounded_id("task-recovery", suffix)
        assert await session.scalar(
            select(RunRow.id).where(RunRow.id == _bounded_id("run-recovery", suffix))
        ) == _bounded_id("run-recovery", suffix)


@pytest.mark.asyncio
async def test_repeated_startup_recovery_converges_on_one_canonical_launch(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-recovery-competing-seeds")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, scope, eligibility=_eligibility())

    def seed(candidate: str) -> tuple[Thread, Task, Run]:
        thread = Thread(
            id=_bounded_id(f"thread-launch-{candidate}", suffix),
            scope=scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=job.conversation_key,
                external_event_id=event_id,
                external_channel_id=job.channel_id,
            ),
        )
        task = Task(
            id=_bounded_id(f"task-launch-{candidate}", suffix),
            thread_id=thread.id,
            scope=scope,
            objective=job.prompt,
        )
        return (
            thread,
            task,
            Run(
                id=_bounded_id(f"run-launch-{candidate}", suffix),
                task_id=task.id,
                scope=scope,
            ),
        )

    first = await admission.recover_startup_launches(
        lambda _job, _scope: seed("a"),
        event_ids=(event_id,),
    )
    second = await admission.recover_startup_launches(
        lambda _job, _scope: seed("b"),
        event_ids=(event_id,),
    )
    launches = [item.launch for item in (*first, *second) if item.launch is not None]
    assert launches
    assert len({launch.task_id for launch in launches}) == 1
    assert len({launch.run_id for launch in launches}) == 1
    async with launch_sessions() as session:
        assert await session.scalar(
            select(ThreadRow.id).where(ThreadRow.id == _bounded_id("thread-launch-a", suffix))
        ) == _bounded_id("thread-launch-a", suffix)
        assert (
            await session.scalar(
                select(TaskRow.id).where(TaskRow.id == _bounded_id("task-launch-b", suffix))
            )
            is None
        )
        assert (
            await session.scalar(
                select(RunRow.id).where(RunRow.id == _bounded_id("run-launch-b", suffix))
            )
            is None
        )


@pytest.mark.asyncio
async def test_startup_resignal_skips_live_lease_and_reclaims_expired_lease(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-expired-lease")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    await admission.admit(job, scope, eligibility=_eligibility())
    thread = Thread(
        id=_bounded_id("thread-expired", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=job.event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task-expired", suffix),
        thread_id=thread.id,
        scope=scope,
        objective=job.prompt,
    )
    await admission.materialize_initial_launch(
        event_id=event_id,
        thread=thread,
        task=task,
        run=Run(id=_bounded_id("run-expired", suffix), task_id=task.id, scope=scope),
    )
    leases = PostgresTaskLeaseStore(launch_sessions, _UniqueIds())
    claimed = await leases.claim_task(
        task.id,
        "worker-live",
        lease_seconds=60,
        now=datetime.now(UTC),
    )
    assert claimed is not None

    def no_seed(
        _job: SlackMentionJob,
        _scope: ScopeKey,
    ) -> tuple[Thread, Task, Run]:
        raise AssertionError("queued recovery must not reseed")

    assert await admission.recover_startup_launches(no_seed, event_ids=(event_id,)) == ()

    async with launch_sessions() as session, session.begin():
        await session.execute(
            update(TaskRow)
            .where(TaskRow.id == task.id)
            # PostgreSQL ``now()`` is fixed at the outer transaction start. Use an
            # unambiguously old instant so remote latency cannot make this look live.
            .values(lease_expires_at=datetime(2000, 1, 1, tzinfo=UTC))
        )

    recovered = await admission.recover_startup_launches(no_seed, event_ids=(event_id,))
    assert len(recovered) == 1
    assert recovered[0].launch is not None
    assert recovered[0].launch.task_id == task.id


@pytest.mark.asyncio
async def test_process_deadline_reconciles_durable_timeout_before_timeout_ux(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    job, scope, suffix = _unique_job("Ev-launch-timeout")
    event_id = job.event_id
    admission = PostgresSlackIngressAdmission(launch_sessions)
    admitted = await admission.admit(job, scope, eligibility=_eligibility())
    assert admitted is not None
    thread = Thread(
        id=_bounded_id("thread-timeout", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=job.event_id,
            external_channel_id=job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task-timeout", suffix),
        thread_id=thread.id,
        scope=scope,
        objective=job.prompt,
    )
    materialized = await admission.materialize_initial_launch(
        event_id=event_id,
        thread=thread,
        task=task,
        run=Run(id=_bounded_id("run-timeout", suffix), task_id=task.id, scope=scope),
    )
    admitted = replace(
        admitted,
        launch=SlackLaunchRef(
            thread_id=materialized.thread_id,
            task_id=materialized.task_id,
            run_id=materialized.run_id,
        ),
    )
    assert isinstance(admitted, AdmittedSlackMention)
    leases = PostgresTaskLeaseStore(launch_sessions, _UniqueIds())
    lease = await leases.claim_task(task.id, "worker-timeout", lease_seconds=60)
    assert lease is not None

    first = await reconcile_admitted_slack_timeout(
        sessions=launch_sessions,
        admitted=admitted,
        lease=lease,
    )
    second = await reconcile_admitted_slack_timeout(
        sessions=launch_sessions,
        admitted=admitted,
        lease=lease,
    )

    assert first.task.status.value == "failed"
    assert first.run.status.value == "timed_out"
    assert first.run.terminal_reason == "slack_runtime_deadline_exceeded"
    assert second.task.version == first.task.version
    assert second.run.version == first.run.version
    async with launch_sessions() as session:
        row = await session.scalar(select(TaskRow).where(TaskRow.id == task.id))
    assert row is not None
    assert row.lease_owner is None
    assert row.lease_token is None


@pytest.mark.asyncio
async def test_followup_creates_parent_link_under_same_thread_after_terminal_run(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    root_job, scope, suffix = _unique_job("Ev-followup-root")
    admission = PostgresSlackIngressAdmission(launch_sessions)
    root_job = root_job.model_copy(
        update={
            "message_ts": "100.0",
            "thread_root_ts": "100.0",
            "conversation_key": f"slack:{root_job.team_id}:{root_job.channel_id}:100.0",
        }
    )
    await admission.admit(root_job, scope, eligibility=_eligibility())
    root_thread = Thread(
        id=_bounded_id("thread-followup-root", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=root_job.conversation_key,
            external_event_id=root_job.event_id,
            external_channel_id=root_job.channel_id,
        ),
    )
    root_task = Task(
        id=_bounded_id("task-followup-root", suffix),
        thread_id=root_thread.id,
        scope=scope,
        objective=root_job.prompt,
    )
    await admission.materialize_initial_launch(
        event_id=root_job.event_id,
        thread=root_thread,
        task=root_task,
        run=Run(
            id=_bounded_id("run-followup-root", suffix),
            task_id=root_task.id,
            scope=scope,
        ),
    )
    async with launch_sessions() as session, session.begin():
        await session.execute(
            update(TaskRow)
            .where(TaskRow.id == root_task.id)
            .values(status="completed", final_output="root answer")
        )
        await session.execute(
            update(RunRow)
            .where(RunRow.id == _bounded_id("run-followup-root", suffix))
            .values(
                status="completed",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                final_output="root answer",
                terminal_reason="verified_completion",
            )
        )

    follow_job = _job(
        _bounded_id("Ev-followup-child", suffix),
        team_id=root_job.team_id,
        channel_id=root_job.channel_id,
        user_id=root_job.user_id,
    ).model_copy(
        update={
            "message_ts": "100.1",
            "thread_root_ts": "100.0",
            "conversation_key": root_job.conversation_key,
        }
    )
    await admission.admit(follow_job, scope, eligibility=_eligibility())
    materialized = await admission.materialize_initial_launch(
        event_id=follow_job.event_id,
        thread=Thread(
            id=_bounded_id("thread-followup-child-proposed", suffix),
            scope=scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=follow_job.conversation_key,
                external_event_id=follow_job.event_id,
                external_channel_id=follow_job.channel_id,
            ),
        ),
        task=Task(
            id=_bounded_id("task-followup-child", suffix),
            thread_id=_bounded_id("thread-followup-child-proposed", suffix),
            scope=scope,
            objective=follow_job.prompt,
        ),
        run=Run(
            id=_bounded_id("run-followup-child", suffix),
            task_id=_bounded_id("task-followup-child", suffix),
            scope=scope,
        ),
    )
    async with launch_sessions() as session:
        child = await session.scalar(select(TaskRow).where(TaskRow.id == materialized.task_id))
    assert child is not None
    assert materialized.thread_id == root_thread.id
    assert child.parent_task_id == root_task.id
    assert child.continuation_kind == "follow_up"
    assert child.mapping_version == 1
    reloaded = await admission.load_linked_mention(materialized.task_id)
    assert reloaded.job.event_id == follow_job.event_id
    assert reloaded.launch is not None
    assert reloaded.launch.thread_id == root_thread.id
    assert reloaded.launch.task_id == materialized.task_id


@pytest.mark.asyncio
async def test_followup_while_active_stays_durable_and_fifo_eligible(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    root_job, scope, suffix = _unique_job("Ev-busy-root")
    admission = PostgresSlackIngressAdmission(launch_sessions)
    root_job = root_job.model_copy(
        update={
            "message_ts": "200.0",
            "thread_root_ts": "200.0",
            "conversation_key": f"slack:{root_job.team_id}:{root_job.channel_id}:200.0",
        }
    )
    await admission.admit(root_job, scope, eligibility=_eligibility())
    thread = Thread(
        id=_bounded_id("thread-busy", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=root_job.conversation_key,
            external_event_id=root_job.event_id,
            external_channel_id=root_job.channel_id,
        ),
    )
    task = Task(
        id=_bounded_id("task-busy", suffix),
        thread_id=thread.id,
        scope=scope,
        objective=root_job.prompt,
    )
    await admission.materialize_initial_launch(
        event_id=root_job.event_id,
        thread=thread,
        task=task,
        run=Run(id=_bounded_id("run-busy", suffix), task_id=task.id, scope=scope),
    )
    follow_job = _job(
        _bounded_id("Ev-busy-child", suffix),
        team_id=root_job.team_id,
        channel_id=root_job.channel_id,
        user_id=root_job.user_id,
    ).model_copy(
        update={
            "message_ts": "200.1",
            "thread_root_ts": "200.0",
            "conversation_key": root_job.conversation_key,
        }
    )
    await admission.admit(follow_job, scope, eligibility=_eligibility())
    with pytest.raises(SlackLaunchInvariantError, match="active Task"):
        await admission.materialize_initial_launch(
            event_id=follow_job.event_id,
            thread=Thread(
                id=_bounded_id("thread-busy-proposed", suffix),
                scope=scope,
                origin=OriginRef(
                    provider="slack",
                    external_thread_id=follow_job.conversation_key,
                    external_event_id=follow_job.event_id,
                    external_channel_id=follow_job.channel_id,
                ),
            ),
            task=Task(
                id=_bounded_id("task-busy-child", suffix),
                thread_id=_bounded_id("thread-busy-proposed", suffix),
                scope=scope,
                objective=follow_job.prompt,
            ),
            run=Run(
                id=_bounded_id("run-busy-child", suffix),
                task_id=_bounded_id("task-busy-child", suffix),
                scope=scope,
            ),
        )
    await admission.mark_followup_pending(follow_job.event_id, "thread_task_active")
    async with launch_sessions() as session:
        pending = await session.scalar(
            select(SlackIngressEventRow).where(SlackIngressEventRow.event_id == follow_job.event_id)
        )
    assert pending is not None
    assert pending.launch_status == "unlaunched"
    assert pending.status == "followup_pending"
    assert pending.launch_error == "thread_task_active"
    assert pending.task_id is None


@pytest.mark.asyncio
async def test_busy_followups_recover_one_at_a_time_in_immutable_admission_order(
    launch_sessions: async_sessionmaker[AsyncSession],
) -> None:
    root_job, scope, suffix = _unique_job("Ev-fifo-root")
    admission = PostgresSlackIngressAdmission(launch_sessions)
    root_job = root_job.model_copy(
        update={
            "message_ts": "300.0",
            "thread_root_ts": "300.0",
            "conversation_key": f"slack:{root_job.team_id}:{root_job.channel_id}:300.0",
        }
    )
    await admission.admit(root_job, scope, eligibility=_eligibility())
    root_thread = Thread(
        id=_bounded_id("thread-fifo", suffix),
        scope=scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=root_job.conversation_key,
            external_event_id=root_job.event_id,
            external_channel_id=root_job.channel_id,
        ),
    )
    root_task = Task(
        id=_bounded_id("task-fifo-root", suffix),
        thread_id=root_thread.id,
        scope=scope,
        objective=root_job.prompt,
    )
    await admission.materialize_initial_launch(
        event_id=root_job.event_id,
        thread=root_thread,
        task=root_task,
        run=Run(
            id=_bounded_id("run-fifo-root", suffix),
            task_id=root_task.id,
            scope=scope,
        ),
    )

    followups: list[SlackMentionJob] = []
    for label, message_ts in (("a", "300.1"), ("b", "300.2")):
        event_id = _bounded_id(f"Ev-fifo-{label}", suffix)
        job = _job(
            event_id,
            team_id=root_job.team_id,
            channel_id=root_job.channel_id,
            user_id=root_job.user_id,
        ).model_copy(
            update={
                "message_ts": message_ts,
                "thread_root_ts": "300.0",
                "conversation_key": root_job.conversation_key,
            }
        )
        await admission.admit(job, scope, eligibility=_eligibility())
        followups.append(job)
        with pytest.raises(SlackLaunchInvariantError, match="active Task"):
            await admission.materialize_initial_launch(
                event_id=job.event_id,
                thread=Thread(
                    id=_bounded_id(f"thread-proposed-{label}", suffix),
                    scope=scope,
                    origin=OriginRef(
                        provider="slack",
                        external_thread_id=job.conversation_key,
                        external_event_id=job.event_id,
                        external_channel_id=job.channel_id,
                    ),
                ),
                task=Task(
                    id=_bounded_id(f"task-proposed-{label}", suffix),
                    thread_id=_bounded_id(f"thread-proposed-{label}", suffix),
                    scope=scope,
                    objective=job.prompt,
                ),
                run=Run(
                    id=_bounded_id(f"run-proposed-{label}", suffix),
                    task_id=_bounded_id(f"task-proposed-{label}", suffix),
                    scope=scope,
                ),
            )
        await admission.mark_followup_pending(job.event_id, "thread_task_active")

    await _complete_task_run(
        launch_sessions,
        root_task.id,
        _bounded_id("run-fifo-root", suffix),
    )

    def seed(job: SlackMentionJob, recovered_scope: ScopeKey) -> tuple[Thread, Task, Run]:
        task_id = _bounded_id(f"task-recovered-{job.event_id}", suffix)
        thread = Thread(
            id=_bounded_id(f"thread-recovered-{job.event_id}", suffix),
            scope=recovered_scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=job.conversation_key,
                external_event_id=job.event_id,
                external_channel_id=job.channel_id,
            ),
        )
        task = Task(
            id=task_id,
            thread_id=thread.id,
            scope=recovered_scope,
            objective=job.prompt,
        )
        return (
            thread,
            task,
            Run(
                id=_bounded_id(f"run-recovered-{job.event_id}", suffix),
                task_id=task_id,
                scope=recovered_scope,
            ),
        )

    followup_event_ids = tuple(job.event_id for job in followups)
    first = await admission.recover_startup_launches(seed, event_ids=followup_event_ids)
    assert [item.job.event_id for item in first] == [followups[0].event_id]
    assert first[0].launch is not None
    await _complete_task_run(
        launch_sessions,
        first[0].launch.task_id,
        first[0].launch.run_id,
    )

    second = await admission.recover_startup_launches(seed, event_ids=followup_event_ids)
    assert [item.job.event_id for item in second] == [followups[1].event_id]


async def _complete_task_run(
    sessions: async_sessionmaker[AsyncSession],
    task_id: str,
    run_id: str,
) -> None:
    async with sessions() as session, session.begin():
        await session.execute(
            update(TaskRow)
            .where(TaskRow.id == task_id)
            .values(status="completed", final_output="done")
        )
        await session.execute(
            update(RunRow)
            .where(RunRow.id == run_id)
            .values(
                status="completed",
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                final_output="done",
                terminal_reason="verified_completion",
            )
        )
