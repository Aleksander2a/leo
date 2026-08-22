from __future__ import annotations

import io
from pathlib import Path
from uuid import uuid4

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.evals.durable_recovery import (
    DurableRecoveryArtifact,
    DurableRecoveryOutcome,
    make_durable_recovery_artifact,
    make_durable_recovery_case,
)
from leo.evals.failure import (
    FailureExportAuthority,
    FailureExportNotFound,
    import_failure_bundle,
)
from leo.evals.operator_cli import run_operator_cli_async
from leo.evals.postgres_failure_source import PostgresFailureEventSource
from leo.harness.events import normalize_run_timeline
from leo.harness.models import (
    BudgetUsage,
    EventDraft,
    EventType,
    OriginRef,
    Run,
    ScopeKey,
    Task,
    Thread,
)
from leo.harness.plan_models import PlanNodeDefinition, PlanStatus
from leo.harness.store_errors import ConcurrencyError
from leo.harness.transitions import advance_step, start_task_and_run, time_out_task_and_run
from leo.integrations.fake import FixedClock
from leo.integrations.system import UuidIdGenerator
from leo.persistence.plan_store import (
    PlanClaimConflictError,
    PlanScopeMismatchError,
    PostgresPlanStore,
)
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import RunEventRow


def _suffix() -> str:
    return uuid4().hex[:12]


async def _seed_run(
    sessions: async_sessionmaker[AsyncSession],
    *,
    suffix: str,
    clock: FixedClock,
    scope: ScopeKey | None = None,
) -> tuple[PostgresRunStore, ScopeKey, Task, Run]:
    scope = scope or ScopeKey(
        organization_id=f"org-m5-{suffix}",
        strategy_id=f"strategy-m5-{suffix}",
    )
    thread = Thread(
        id=f"thread-{suffix}",
        scope=scope,
        origin=OriginRef(
            provider="m5-pg-eval",
            external_thread_id=f"external-{suffix}",
        ),
    )
    task = Task(
        id=f"task-{suffix}",
        thread_id=thread.id,
        scope=scope,
        objective="Exercise rollback-safe M5 durable recovery.",
    )
    run = Run(id=f"run-{suffix}", task_id=task.id, scope=scope)
    store = PostgresRunStore(sessions, clock, UuidIdGenerator())
    await store.seed(thread, task, run)
    return store, scope, task, run


async def _assert_current_head(sessions: async_sessionmaker[AsyncSession]) -> str:
    config = Config("alembic.ini")
    config.set_main_option("script_location", "migrations")
    expected = ScriptDirectory.from_config(config).get_current_head()
    assert expected is not None
    async with sessions() as session:
        actual = await session.scalar(text("select version_num from alembic_version"))
    assert actual == expected
    return expected


@pytest.mark.asyncio
async def test_current_head_event_sequence_cas_restart_and_operator_export_are_rollback_safe(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    head = await _assert_current_head(preserved_postgres_sessions)
    clock = FixedClock()
    store, scope, task, run = await _seed_run(
        preserved_postgres_sessions,
        suffix=_suffix(),
        clock=clock,
    )
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    active = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    candidate_task, candidate_run = advance_step(
        active.task,
        active.run,
        usage=BudgetUsage(model_calls=1),
    )
    winner = await store.commit(
        expected_task_version=active.task.version,
        expected_run_version=active.run.version,
        task=candidate_task,
        run=candidate_run,
        events=(
            EventDraft(
                type=EventType.MODEL_CALLED,
                iteration=candidate_run.iteration,
                payload={"decision": "completion", "provider": "offline"},
            ),
        ),
    )
    before_stale = winner.model_dump(mode="json")
    with pytest.raises(ConcurrencyError, match="stale task version"):
        await store.commit(
            expected_task_version=active.task.version,
            expected_run_version=active.run.version,
            task=candidate_task,
            run=candidate_run,
            events=(
                EventDraft(
                    type=EventType.MODEL_CALLED,
                    iteration=candidate_run.iteration,
                    payload={"decision": "completion", "provider": "offline"},
                ),
            ),
        )
    after_stale = await store.load(task.id, run.id, scope)
    assert after_stale.model_dump(mode="json") == before_stale

    duplicate_sequence_rejected = False
    with pytest.raises(IntegrityError):
        async with preserved_postgres_sessions() as session, session.begin():
            session.add(
                RunEventRow(
                    id=UuidIdGenerator().new("evt"),
                    run_id=run.id,
                    task_id=task.id,
                    sequence=2,
                    type=EventType.MODEL_CALLED.value,
                    occurred_at=clock.now(),
                    iteration=1,
                    schema_version=1,
                    payload={"decision": "completion"},
                )
            )
            await session.flush()
    duplicate_sequence_rejected = True
    after_sequence_conflict = await store.load(task.id, run.id, scope)
    assert after_sequence_conflict == after_stale

    timeout_task, timeout_run = time_out_task_and_run(
        after_stale.task,
        after_stale.run,
        "m5_synthetic_timeout",
        usage=after_stale.run.usage,
    )
    terminal = await store.commit(
        expected_task_version=after_stale.task.version,
        expected_run_version=after_stale.run.version,
        task=timeout_task,
        run=timeout_run,
        events=(
            EventDraft(
                type=EventType.RUN_TIMED_OUT,
                iteration=timeout_run.iteration,
                payload={"reason": "m5_synthetic_timeout"},
            ),
        ),
    )
    restarted = PostgresRunStore(
        preserved_postgres_sessions,
        FixedClock(),
        UuidIdGenerator(),
    )
    reloaded = await restarted.load(task.id, run.id, scope)
    assert reloaded == terminal
    normalized = normalize_run_timeline(reloaded.events, scope)
    assert [item.sequence for item in normalized] == [1, 2, 3]
    assert all(item.schema_version == "v2" for item in normalized)
    assert normalized[-1].causation_id == normalized[-2].event_id

    authority = FailureExportAuthority(
        organization_id=scope.organization_id,
        actor_id="m5-operator",
        allowed_run_ids=(run.id,),
    )
    source = PostgresFailureEventSource(
        preserved_postgres_sessions,
        config_versions={
            "alembic_head": head,
            "source": "rollback-safe-postgres",
        },
    )
    destination = tmp_path / "durable-failure.json"
    output = io.StringIO()
    assert (
        await run_operator_cli_async(
            ["export", "--run-id", run.id, "--output", str(destination)],
            source=source,
            authority=authority,
            stdout=output,
        )
        == 0
    )
    exported = import_failure_bundle(destination)
    assert exported.failure.run_id == run.id
    assert exported.failure.event_ids == tuple(item.id for item in terminal.events)
    assert all("scope" not in item for item in exported.sanitized_events)
    assert all(item["schema_version"] == "v2" for item in exported.sanitized_events)
    with pytest.raises(FailureExportNotFound):
        await source.load(
            authority=authority.model_copy(update={"organization_id": "other-org"}),
            run_id=run.id,
        )

    cases = (
        make_durable_recovery_case(
            case_id="stale-cas",
            boundary="run_store_commit",
            outcome=DurableRecoveryOutcome.REJECTED_SAFE,
            before=before_stale,
            after=after_stale.model_dump(mode="json"),
            mutation_applied=False,
            detail_code="stale_task_version",
        ),
        make_durable_recovery_case(
            case_id="duplicate-event-sequence",
            boundary="run_event_unique_sequence",
            outcome=DurableRecoveryOutcome.REJECTED_SAFE,
            before=before_stale,
            after=after_sequence_conflict.model_dump(mode="json"),
            mutation_applied=not duplicate_sequence_rejected,
            detail_code="uq_run_event_sequence",
        ),
        make_durable_recovery_case(
            case_id="restart-replay",
            boundary="run_store_reload",
            outcome=DurableRecoveryOutcome.RELOAD_EXACT,
            before=terminal.model_dump(mode="json"),
            after=reloaded.model_dump(mode="json"),
            mutation_applied=False,
            detail_code="exact_snapshot_reloaded",
        ),
        make_durable_recovery_case(
            case_id="operator-export",
            boundary="failure_event_source",
            outcome=DurableRecoveryOutcome.EXPORTED,
            before={"run_id": run.id},
            after={"bundle_digest": exported.digest},
            mutation_applied=destination.is_file(),
            detail_code="sanitized_bundle_exported",
        ),
    )
    artifact = make_durable_recovery_artifact(cases)
    assert artifact.case_count == 4
    assert artifact.false_success_count == artifact.duplicate_commit_count == 0
    artifact_path = tmp_path / "event-recovery-artifact.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    assert (
        DurableRecoveryArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        == artifact
    )


@pytest.mark.asyncio
async def test_durable_plan_child_reclaim_fencing_and_final_replay_use_outer_rollback(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
    tmp_path: Path,
) -> None:
    await _assert_current_head(preserved_postgres_sessions)
    clock = FixedClock()
    run_store, scope, parent_task, parent_run = await _seed_run(
        preserved_postgres_sessions,
        suffix=_suffix(),
        clock=clock,
    )
    started_task, started_run = start_task_and_run(
        parent_task,
        parent_run,
        started_at=clock.now(),
    )
    active_parent = await run_store.commit(
        expected_task_version=parent_task.version,
        expected_run_version=parent_run.version,
        task=started_task,
        run=started_run,
        events=(
            EventDraft(
                type=EventType.TASK_STARTED,
                iteration=0,
                payload={"phase": "planning"},
            ),
        ),
    )
    parent_task = active_parent.task
    parent_run = active_parent.run
    plan_store = PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    )
    snapshot = await plan_store.create_or_load(
        scope=scope,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        idempotency_key=f"m5-plan-{_suffix()}",
        goal="Recover one durable delegated child.",
        nodes=(
            PlanNodeDefinition(
                key="research",
                objective="Complete synthetic research.",
                max_attempts=2,
            ),
        ),
    )
    first = await plan_store.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="worker-before-crash",
        lease_seconds=10,
    )
    assert first is not None
    _, _, foreign_task, foreign_run = await _seed_run(
        preserved_postgres_sessions,
        suffix=_suffix(),
        clock=clock,
    )
    with pytest.raises(PlanScopeMismatchError, match="trusted organization"):
        await plan_store.attach_child(
            scope=scope,
            claim=first,
            child_task_id=foreign_task.id,
            child_run_id=foreign_run.id,
        )
    _, _, first_child_task, first_child_run = await _seed_run(
        preserved_postgres_sessions,
        suffix=_suffix(),
        clock=clock,
        scope=scope,
    )
    attached = await plan_store.attach_child(
        scope=scope,
        claim=first,
        child_task_id=first_child_task.id,
        child_run_id=first_child_run.id,
    )

    restarted = PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    )
    replayed_running = await restarted.replay(scope=scope, plan_id=snapshot.plan.id)
    assert replayed_running == attached
    clock.advance(seconds=11)
    reclaimed = await restarted.claim_ready_node(
        scope=scope,
        plan_id=snapshot.plan.id,
        owner="worker-after-restart",
    )
    assert reclaimed is not None and reclaimed.attempt == 2
    stale_fenced = False
    with pytest.raises(PlanClaimConflictError):
        await restarted.complete_node(scope=scope, claim=first, output="stale result")
    stale_fenced = True
    _, _, replacement_task, replacement_run = await _seed_run(
        preserved_postgres_sessions,
        suffix=_suffix(),
        clock=clock,
        scope=scope,
    )
    await restarted.attach_child(
        scope=scope,
        claim=reclaimed,
        child_task_id=replacement_task.id,
        child_run_id=replacement_run.id,
    )
    settled = await restarted.complete_node(
        scope=scope,
        claim=reclaimed,
        output="verified synthetic result",
    )
    final = await restarted.finalize(
        scope=scope,
        plan_id=snapshot.plan.id,
        parent_task_id=parent_task.id,
        parent_run_id=parent_run.id,
        status=PlanStatus.COMPLETED,
        result="parent verified synthesis",
    )
    replayed_final = await PostgresPlanStore(
        preserved_postgres_sessions,
        clock,
        UuidIdGenerator(),
    ).reload(scope=scope, plan_id=snapshot.plan.id)
    assert replayed_final == final
    assert settled.delegations[0].status.value == "superseded"
    assert settled.delegations[-1].status.value == "completed"

    artifact = make_durable_recovery_artifact(
        (
            make_durable_recovery_case(
                case_id="running-child-reload",
                boundary="plan_store_replay",
                outcome=DurableRecoveryOutcome.RELOAD_EXACT,
                before=attached.model_dump(mode="json"),
                after=replayed_running.model_dump(mode="json"),
                mutation_applied=False,
                detail_code="attached_child_reloaded",
            ),
            make_durable_recovery_case(
                case_id="expired-child-reclaim",
                boundary="plan_node_lease",
                outcome=DurableRecoveryOutcome.RECLAIMED,
                before=first.model_dump(mode="json"),
                after=reclaimed.model_dump(mode="json"),
                mutation_applied=reclaimed.token != first.token,
                detail_code="stale_running_reclaimed",
            ),
            make_durable_recovery_case(
                case_id="stale-child-fenced",
                boundary="plan_node_claim",
                outcome=DurableRecoveryOutcome.FENCED,
                before=settled.model_dump(mode="json"),
                after=settled.model_dump(mode="json"),
                mutation_applied=not stale_fenced,
                detail_code="stale_claim_rejected",
            ),
            make_durable_recovery_case(
                case_id="terminal-plan-reload",
                boundary="plan_store_reload",
                outcome=DurableRecoveryOutcome.RELOAD_EXACT,
                before=final.model_dump(mode="json"),
                after=replayed_final.model_dump(mode="json"),
                mutation_applied=False,
                detail_code="parent_terminal_reloaded",
            ),
        )
    )
    assert artifact.case_count == 4
    assert artifact.false_success_count == artifact.duplicate_commit_count == 0
    artifact_path = tmp_path / "plan-recovery-artifact.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    assert (
        DurableRecoveryArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8"))
        == artifact
    )
