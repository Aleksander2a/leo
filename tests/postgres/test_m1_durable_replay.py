from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.context import DefaultContextAssembler, context_manifest_event_payload
from leo.harness.models import (
    BudgetUsage,
    ContextItem,
    ContextItemKind,
    EventDraft,
    EventType,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
)
from leo.harness.plan_models import PlanNodeDefinition, PlanNodeStatus, PlanStatus
from leo.harness.store_errors import NotFoundError
from leo.harness.transitions import (
    cancel_task_and_run,
    start_task_and_run,
    time_out_task_and_run,
)
from leo.integrations.fake import FixedClock
from leo.integrations.system import UuidIdGenerator
from leo.persistence.plan_store import PlanClaimConflictError, PostgresPlanStore
from leo.persistence.replay_store import PostgresReplayStore
from leo.persistence.run_store import PostgresRunStore
from leo.replay import ReplayLane, export_replay


def _suffix() -> str:
    return uuid4().hex


async def _assert_current_head(sessions: async_sessionmaker[AsyncSession]) -> None:
    config = Config("alembic.ini")
    config.set_main_option("script_location", "migrations")
    expected = ScriptDirectory.from_config(config).get_current_head()
    assert expected is not None
    async with sessions() as session:
        actual = await session.scalar(text("select version_num from alembic_version"))
    assert actual == expected


async def _seed_run(
    sessions: async_sessionmaker[AsyncSession],
    *,
    clock: FixedClock,
    suffix: str,
    scope: ScopeKey | None = None,
    parent_task_id: str | None = None,
) -> tuple[PostgresRunStore, ScopeKey, Task, Run]:
    trusted_scope = scope or ScopeKey(
        organization_id=f"org-m1-{suffix}",
        strategy_id=f"strategy-m1-{suffix}",
    )
    thread = Thread(
        id=f"thread-{suffix}",
        scope=trusted_scope,
        origin=OriginRef(
            provider="m1-durable-replay",
            external_thread_id=f"conversation-{suffix}",
            external_channel_id=f"conversation-{suffix}",
        ),
    )
    task = Task(
        id=f"task-{suffix}",
        thread_id=thread.id,
        scope=trusted_scope,
        objective=f"Exercise durable replay for {suffix}.",
        parent_task_id=parent_task_id,
        continuation_kind="subagent" if parent_task_id is not None else "root",
    )
    run = Run(id=f"run-{suffix}", task_id=task.id, scope=trusted_scope)
    store = PostgresRunStore(sessions, clock, UuidIdGenerator())
    await store.seed(thread, task, run)
    return store, trusted_scope, task, run


async def _start(
    store: PostgresRunStore,
    task: Task,
    run: Run,
    clock: FixedClock,
) -> RunBundle:
    started_task, started_run = start_task_and_run(task, run, started_at=clock.now())
    return await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
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


def _context_event(
    bundle: RunBundle,
    *,
    item_id: str,
    conversation_id: str,
    authority_ids: tuple[str, ...],
) -> EventDraft:
    request = DefaultContextAssembler(
        context_items=(
            ContextItem(
                id=item_id,
                kind=ContextItemKind.CONVERSATION_TURN,
                content="Synthetic untrusted context; never exported as durable manifest data.",
                conversation_id=conversation_id,
                source_scope=bundle.run.scope,
            ),
        ),
        authority_snapshot_ids=authority_ids,
    ).assemble(bundle, ())
    return EventDraft(
        type=EventType.CONTEXT_BUILT,
        iteration=bundle.run.iteration,
        payload={
            "segments": [segment.name for segment in request.manifest.segments],
            "tool_count": 0,
            "tool_choice": request.tool_choice.mode.value,
            "source_manifest": context_manifest_event_payload(request.manifest),
        },
    )


def _unsourced_completion(answer: str) -> VerifiedCompletion:
    return VerifiedCompletion(
        answer=answer,
        claims=(),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=(
                VerifierCheck(
                    name="trusted_context_only",
                    passed=True,
                    detail="The deterministic fixture explicitly permits context-only output.",
                ),
            ),
            retryable=False,
            allow_unsourced_completion=True,
        ),
    )


async def _complete(
    store: PostgresRunStore,
    bundle: RunBundle,
    *,
    answer: str,
    context_event: EventDraft,
) -> RunBundle:
    return await store.complete_verified(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        scope=bundle.run.scope,
        usage=BudgetUsage(model_calls=1),
        completion=_unsourced_completion(answer),
        preceding_events=(context_event,),
    )


@pytest.mark.asyncio
async def test_parent_plan_child_source_manifest_restart_and_export_use_outer_rollback(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    await _assert_current_head(preserved_postgres_sessions)
    clock = FixedClock()
    suffix = _suffix()
    run_store, scope, parent_task, parent_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"parent-{suffix}",
    )
    parent_active = await _start(run_store, parent_task, parent_run, clock)
    plan_store = PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    )
    plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        idempotency_key=f"m1-replay-{suffix}",
        goal="Delegate one bounded child and replay all durable provenance.",
        nodes=(
            PlanNodeDefinition(
                key="research",
                objective="Produce one deterministic child result.",
            ),
        ),
    )
    claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=plan.plan.id,
        owner="m1-worker",
    )
    assert claim is not None
    child_store, _, child_task, child_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"child-{suffix}",
        scope=scope,
        parent_task_id=parent_task.id,
    )
    attached = await plan_store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    assert attached.current_nodes[0].child_run_id == child_run.id

    child_active = await _start(child_store, child_task, child_run, clock)
    child_completed = await _complete(
        child_store,
        child_active,
        answer="The child completed without parent completion authority.",
        context_event=_context_event(
            child_active,
            item_id=f"child-message-{suffix}",
            conversation_id=f"conversation-child-{suffix}",
            authority_ids=(f"child-access-{suffix}",),
        ),
    )
    assert child_completed.run.status is RunStatus.COMPLETED
    assert parent_active.run.status is RunStatus.RUNNING
    active_replay = await PostgresReplayStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(scope=scope, run_id=parent_run.id)
    assert active_replay.status is RunStatus.RUNNING
    assert "source_manifest_not_persisted" in active_replay.omissions
    assert any(
        entry.lane is ReplayLane.CHILD_EVENT and entry.kind == EventType.RUN_COMPLETED.value
        for entry in active_replay.entries
    )
    await plan_store.complete_node(
        scope=scope,
        claim=claim,
        output=child_completed.run.final_output or "child complete",
    )
    finalized = await plan_store.finalize(
        scope=scope,
        plan_id=plan.plan.id,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        status=PlanStatus.COMPLETED,
        result="The parent accepted the bounded child result.",
    )
    assert finalized.plan.status is PlanStatus.COMPLETED

    parent_item = f"parent-message-{suffix}"
    parent_conversation = f"conversation-parent-{suffix}"
    parent_authority = (f"access-{suffix}", f"membership-{suffix}")
    parent_completed = await _complete(
        run_store,
        parent_active,
        answer="The parent alone completed after verification.",
        context_event=_context_event(
            parent_active,
            item_id=parent_item,
            conversation_id=parent_conversation,
            authority_ids=parent_authority,
        ),
    )
    assert parent_completed.run.status is RunStatus.COMPLETED

    replay_store = PostgresReplayStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    )
    first = await replay_store.load(scope=scope, run_id=parent_run.id)
    second = await PostgresReplayStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(scope=scope, run_id=parent_run.id)
    assert second == first
    assert first.status is RunStatus.COMPLETED
    assert first.omissions == ()
    assert first.source_manifest is not None
    assert {
        parent_item,
        parent_conversation,
        *parent_authority,
    }.issubset(first.source_manifest.included_source_ids)
    assert {
        ReplayLane.PARENT_EVENT,
        ReplayLane.PLAN_REVISION,
        ReplayLane.PLAN_NODE,
        ReplayLane.DELEGATION,
        ReplayLane.CHILD_EVENT,
    }.issubset({entry.lane for entry in first.entries})
    assert any(entry.child_run_id == child_run.id for entry in first.entries if entry.child_run_id)
    assert any(
        entry.lane is ReplayLane.CHILD_EVENT and entry.kind == EventType.CONTEXT_BUILT.value
        for entry in first.entries
    )

    destination = tmp_path / "m1-durable-replay.json"
    assert export_replay(first, destination) == destination.resolve()
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["digest"] == first.digest
    assert "Synthetic untrusted context" not in destination.read_text(encoding="utf-8")

    wrong_scope = ScopeKey(
        organization_id=f"wrong-{scope.organization_id}",
        strategy_id=scope.strategy_id,
    )
    with pytest.raises(NotFoundError):
        await replay_store.load(scope=wrong_scope, run_id=parent_run.id)


@pytest.mark.asyncio
async def test_parent_cancellation_atomically_stops_attached_child_and_fences_stale_claim(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    await _assert_current_head(preserved_postgres_sessions)
    clock = FixedClock()
    suffix = _suffix()
    parent_store, scope, parent_task, parent_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"cancel-parent-{suffix}",
    )
    parent_active = await _start(parent_store, parent_task, parent_run, clock)
    plan_store = PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    )
    plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        idempotency_key=f"m1-cancel-{suffix}",
        goal="Cancel all unfinished child work.",
        nodes=(
            PlanNodeDefinition(key="running", objective="Start a child."),
            PlanNodeDefinition(
                key="dependent",
                objective="Never start after parent cancellation.",
                depends_on=("running",),
            ),
        ),
    )
    claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=plan.plan.id,
        owner="cancel-worker",
    )
    assert claim is not None and claim.node_key == "running"
    child_store, _, child_task, child_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"cancel-child-{suffix}",
        scope=scope,
        parent_task_id=parent_task.id,
    )
    await plan_store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    child_active = await _start(child_store, child_task, child_run, clock)

    cancelled_task, cancelled_run = cancel_task_and_run(
        parent_active.task,
        parent_active.run,
        "operator_cancelled",
    )
    await parent_store.commit(
        expected_task_version=parent_active.task.version,
        expected_run_version=parent_active.run.version,
        task=cancelled_task,
        run=cancelled_run,
        events=(
            _context_event(
                parent_active,
                item_id=f"cancel-message-{suffix}",
                conversation_id=f"cancel-conversation-{suffix}",
                authority_ids=(f"cancel-access-{suffix}",),
            ),
            EventDraft(
                type=EventType.RUN_CANCELLED,
                iteration=cancelled_run.iteration,
                payload={"reason": "operator_cancelled"},
            ),
        ),
    )
    cancelled_plan = await plan_store.cancel(
        scope=scope,
        plan_id=plan.plan.id,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        reason="operator_cancelled",
    )
    assert cancelled_plan.plan.status is PlanStatus.FAILED
    assert all(node.status is PlanNodeStatus.FAILED for node in cancelled_plan.current_nodes)
    assert cancelled_plan.delegations[0].status.value == "superseded"
    assert cancelled_plan.delegations[0].child_run_id == child_run.id

    child_after = await child_store.load(child_task.id, child_run.id, scope)
    assert child_after.task.status is TaskStatus.CANCELLED
    assert child_after.run.status is RunStatus.CANCELLED
    assert child_after.run.terminal_reason == "parent_plan_cancelled"
    assert child_after.run.version == child_active.run.version + 1
    assert child_after.events[-1].type is EventType.RUN_CANCELLED
    assert child_after.events[-1].payload == {"reason": "parent_plan_cancelled"}
    with pytest.raises(PlanClaimConflictError, match="terminal plan"):
        await plan_store.complete_node(scope=scope, claim=claim, output="late child result")

    replay = await PostgresReplayStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(scope=scope, run_id=parent_run.id)
    assert replay.status is RunStatus.CANCELLED
    assert replay.final_output is None
    assert replay.source_manifest is not None
    assert any(
        entry.lane is ReplayLane.CHILD_EVENT and entry.kind == EventType.RUN_CANCELLED.value
        for entry in replay.entries
    )
    assert not any(entry.kind == EventType.RUN_COMPLETED.value for entry in replay.entries)


@pytest.mark.asyncio
async def test_parent_timeout_uses_the_same_durable_child_fence_and_restart_path(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    await _assert_current_head(preserved_postgres_sessions)
    clock = FixedClock()
    suffix = _suffix()
    parent_store, scope, parent_task, parent_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"timeout-parent-{suffix}",
    )
    parent_active = await _start(parent_store, parent_task, parent_run, clock)
    plan_store = PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    )
    plan = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        idempotency_key=f"m1-timeout-{suffix}",
        goal="Stop a durable child when the parent deadline wins.",
        nodes=(PlanNodeDefinition(key="child", objective="Run until fenced."),),
    )
    claim = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=plan.plan.id,
        owner="timeout-worker",
    )
    assert claim is not None
    child_store, _, child_task, child_run = await _seed_run(
        preserved_postgres_sessions,
        clock=clock,
        suffix=f"timeout-child-{suffix}",
        scope=scope,
        parent_task_id=parent_task.id,
    )
    await plan_store.attach_child(
        scope=scope,
        claim=claim,
        child_task_id=child_task.id,
        child_run_id=child_run.id,
    )
    await _start(child_store, child_task, child_run, clock)

    timed_out_task, timed_out_run = time_out_task_and_run(
        parent_active.task,
        parent_active.run,
        "deadline_exceeded",
        usage=parent_active.run.usage,
    )
    await parent_store.commit(
        expected_task_version=parent_active.task.version,
        expected_run_version=parent_active.run.version,
        task=timed_out_task,
        run=timed_out_run,
        events=(
            EventDraft(
                type=EventType.RUN_TIMED_OUT,
                iteration=timed_out_run.iteration,
                payload={"reason": "deadline_exceeded"},
            ),
        ),
    )
    terminated = await plan_store.terminate_for_parent(
        scope=scope,
        plan_id=plan.plan.id,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        parent_status=RunStatus.TIMED_OUT,
        reason="deadline_exceeded",
        child_terminal_reason="parent_deadline_exceeded",
    )
    assert terminated.plan.status is PlanStatus.FAILED
    assert terminated.plan.error == "parent_timed_out:deadline_exceeded"
    child_after = await child_store.load(child_task.id, child_run.id, scope)
    assert child_after.task.status is TaskStatus.CANCELLED
    assert child_after.run.status is RunStatus.CANCELLED
    assert child_after.run.terminal_reason == "parent_deadline_exceeded"
    assert child_after.events[-1].payload == {"reason": "parent_deadline_exceeded"}
    with pytest.raises(PlanClaimConflictError, match="terminal plan"):
        await plan_store.fail_node(scope=scope, claim=claim, error="late timeout result")

    replay = await PostgresReplayStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    ).load(scope=scope, run_id=parent_run.id)
    assert replay.status is RunStatus.TIMED_OUT
    assert replay.terminal_reason == "deadline_exceeded"
    assert any(
        entry.lane is ReplayLane.CHILD_EVENT and entry.kind == EventType.RUN_CANCELLED.value
        for entry in replay.entries
    )
