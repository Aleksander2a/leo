"""Supabase-compatible Postgres implementation of Leo's atomic run-store port."""

from __future__ import annotations

from typing import cast

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import (
    BudgetLimits,
    BudgetUsage,
    Claim,
    ClaimKind,
    EventDraft,
    EventType,
    Observation,
    OriginRef,
    PlannedStep,
    ReasoningStep,
    Run,
    RunBundle,
    RunEvent,
    RunPhase,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    TaskStatus,
    Thread,
    VerifiedCompletion,
)
from leo.harness.persistence_rules import (
    build_verification_passed_event,
    sanitize_event_drafts,
    validate_commit,
    validate_seed,
    validate_verified_completion,
)
from leo.harness.ports import Clock, IdGenerator
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.harness.transitions import complete_task_and_run
from leo.persistence.schema import (
    ClaimRow,
    ObservationRow,
    RunEventRow,
    RunRow,
    TaskRow,
    ThreadRow,
)
from leo.persistence.task_leases import TaskLease


class PostgresRunStore:
    """Persist one aggregate step and its journal records in a single transaction."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._clock = clock
        self._ids = ids

    async def seed(self, thread: Thread, task: Task, run: Run) -> RunBundle:
        validate_seed(thread, task, run)
        try:
            async with self._sessions() as session, session.begin():
                effective_thread = await self._resolve_thread(session, thread)
                effective_task = task.model_copy(update={"thread_id": effective_thread.id})
                session.add(_task_row(effective_task))
                # The mapped rows intentionally have no ORM relationships; flush the
                # parent explicitly so Postgres never sees the run before its task.
                await session.flush()
                session.add(_run_row(run))
                await session.flush()
                return await self._load_bundle(
                    session,
                    effective_task.id,
                    run.id,
                    effective_task.scope,
                )
        except IntegrityError as exc:
            raise ConcurrencyError("thread, task, or run already exists") from exc

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        async with self._sessions() as session:
            return await self._load_bundle(session, task_id, run_id, scope)

    async def load_run(self, run_id: str, scope: ScopeKey) -> RunBundle:
        """Operator/replay lookup that still requires the trusted durable namespace."""

        async with self._sessions() as session:
            task_id = await session.scalar(
                select(RunRow.task_id).where(
                    RunRow.id == run_id,
                    RunRow.organization_id == scope.organization_id,
                    RunRow.strategy_id == scope.strategy_id,
                )
            )
            if task_id is None:
                raise NotFoundError("run not found")
            return await self._load_bundle(session, task_id, run_id, scope)

    async def commit(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task: Task,
        run: Run,
        observations: tuple[Observation, ...] = (),
        events: tuple[EventDraft, ...] = (),
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> RunBundle:
        try:
            async with self._sessions() as session, session.begin():
                task_row = await session.scalar(
                    select(TaskRow)
                    .where(
                        TaskRow.id == task.id,
                        TaskRow.organization_id == task.scope.organization_id,
                        TaskRow.strategy_id == task.scope.strategy_id,
                    )
                    .with_for_update()
                )
                run_row = await session.scalar(
                    select(RunRow)
                    .where(
                        RunRow.id == run.id,
                        RunRow.task_id == task.id,
                        RunRow.organization_id == task.scope.organization_id,
                        RunRow.strategy_id == task.scope.strategy_id,
                    )
                    .with_for_update()
                )
                if task_row is None or run_row is None:
                    raise NotFoundError("task or run not found")
                _require_lease(task_row, lease_owner, lease_token)

                current_task = _task_model(task_row)
                current_run = _run_model(run_row)
                if current_task.version != expected_task_version:
                    raise ConcurrencyError("stale task version")
                if current_run.version != expected_run_version:
                    raise ConcurrencyError("stale run version")
                if task.version != expected_task_version + 1:
                    raise StoreError("next task version must increment exactly once")
                if run.version != expected_run_version + 1:
                    raise StoreError("next run version must increment exactly once")
                if run.task_id != task.id or run.scope != task.scope:
                    raise StoreError("task and run identity mismatch")
                events = validate_commit(
                    current_task,
                    current_run,
                    task,
                    run,
                    events,
                    observations=observations,
                )
                _validate_new_records(run, observations)
                persisted_observation_ids = frozenset(
                    (
                        await session.scalars(
                            select(ObservationRow.id).where(
                                ObservationRow.run_id == run.id,
                                ObservationRow.organization_id == run.scope.organization_id,
                                ObservationRow.strategy_id == run.scope.strategy_id,
                            )
                        )
                    ).all()
                )
                available_observation_ids = persisted_observation_ids.union(
                    item.id for item in observations
                )
                if any(item not in available_observation_ids for item in task.observation_ids):
                    raise StoreError("task references an unavailable observation")

                _apply_task(task_row, task)
                _apply_run(run_row, run)
                for observation in observations:
                    session.add(_observation_row(observation))
                self._append_events(session, run_row, task, run, events)
                await session.flush()
                return await self._load_bundle(session, task.id, run.id, task.scope)
        except IntegrityError as exc:
            raise ConcurrencyError("atomic run step conflicted with persisted state") from exc

    async def complete_verified(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task_id: str,
        run_id: str,
        scope: ScopeKey,
        usage: BudgetUsage,
        completion: VerifiedCompletion,
        preceding_events: tuple[EventDraft, ...] = (),
        lease_owner: str | None = None,
        lease_token: str | None = None,
    ) -> RunBundle:
        try:
            async with self._sessions() as session, session.begin():
                task_row = await session.scalar(
                    select(TaskRow)
                    .where(
                        TaskRow.id == task_id,
                        TaskRow.organization_id == scope.organization_id,
                        TaskRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update()
                )
                run_row = await session.scalar(
                    select(RunRow)
                    .where(
                        RunRow.id == run_id,
                        RunRow.task_id == task_id,
                        RunRow.organization_id == scope.organization_id,
                        RunRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update()
                )
                if task_row is None or run_row is None:
                    raise NotFoundError("task or run not found")
                _require_lease(task_row, lease_owner, lease_token)
                current_task = _task_model(task_row)
                current_run = _run_model(run_row)
                if current_task.version != expected_task_version:
                    raise ConcurrencyError("stale task version")
                if current_run.version != expected_run_version:
                    raise ConcurrencyError("stale run version")

                available_observation_ids = frozenset(
                    (
                        await session.scalars(
                            select(ObservationRow.id).where(
                                ObservationRow.run_id == run_id,
                                ObservationRow.organization_id == scope.organization_id,
                                ObservationRow.strategy_id == scope.strategy_id,
                            )
                        )
                    ).all()
                )
                safe_preceding_events = sanitize_event_drafts(preceding_events, current_run)
                validate_verified_completion(
                    current_task,
                    current_run,
                    usage,
                    completion,
                    available_observation_ids,
                    safe_preceding_events,
                )
                task, run = complete_task_and_run(
                    current_task,
                    current_run,
                    completion,
                    usage=usage,
                )
                _apply_task(task_row, task)
                _apply_run(run_row, run)
                for claim in completion.claims:
                    session.add(_claim_row(claim))

                authoritative_events = (
                    *safe_preceding_events,
                    build_verification_passed_event(completion, run),
                    EventDraft(
                        type=EventType.RUN_COMPLETED,
                        iteration=run.iteration,
                        payload={"reason": "verified_completion"},
                    ),
                )
                safe_authoritative_events = sanitize_event_drafts(authoritative_events, run)
                self._append_events(session, run_row, task, run, safe_authoritative_events)
                await session.flush()
                return await self._load_bundle(session, task.id, run.id, scope)
        except IntegrityError as exc:
            raise ConcurrencyError("verified completion conflicted with persisted state") from exc

    def _append_events(
        self,
        session: AsyncSession,
        run_row: RunRow,
        task: Task,
        run: Run,
        events: tuple[EventDraft, ...],
    ) -> None:
        for draft in events:
            run_row.event_sequence += 1
            session.add(
                RunEventRow(
                    id=self._ids.new("evt"),
                    run_id=run.id,
                    task_id=task.id,
                    sequence=run_row.event_sequence,
                    type=draft.type.value,
                    occurred_at=self._clock.now(),
                    iteration=draft.iteration,
                    schema_version=1,
                    payload=cast(dict[str, object], draft.payload),
                )
            )

    async def _resolve_thread(self, session: AsyncSession, thread: Thread) -> Thread:
        statement = (
            postgres_insert(ThreadRow)
            .values(
                id=thread.id,
                organization_id=thread.scope.organization_id,
                strategy_id=thread.scope.strategy_id,
                origin_provider=thread.origin.provider,
                external_thread_id=thread.origin.external_thread_id,
                external_event_id=thread.origin.external_event_id,
                external_channel_id=thread.origin.external_channel_id,
                mapping_version=thread.mapping_version,
                version=thread.version,
            )
            .on_conflict_do_nothing(
                index_elements=[ThreadRow.origin_provider, ThreadRow.external_thread_id]
            )
            .returning(ThreadRow.id)
        )
        inserted_id = (await session.execute(statement)).scalar_one_or_none()
        if inserted_id is not None:
            return thread

        existing = await session.scalar(
            select(ThreadRow).where(
                ThreadRow.origin_provider == thread.origin.provider,
                ThreadRow.external_thread_id == thread.origin.external_thread_id,
            )
        )
        if existing is None:
            raise ConcurrencyError("thread conflict could not be resolved")
        if (
            existing.organization_id != thread.scope.organization_id
            or existing.strategy_id != thread.scope.strategy_id
        ):
            raise StoreError("existing thread belongs to a different trusted scope")
        return _thread_model(existing)

    async def _load_bundle(
        self,
        session: AsyncSession,
        task_id: str,
        run_id: str,
        scope: ScopeKey,
    ) -> RunBundle:
        task_row = await session.scalar(
            select(TaskRow).where(
                TaskRow.id == task_id,
                TaskRow.organization_id == scope.organization_id,
                TaskRow.strategy_id == scope.strategy_id,
            )
        )
        run_row = await session.scalar(
            select(RunRow).where(
                RunRow.id == run_id,
                RunRow.task_id == task_id,
                RunRow.organization_id == scope.organization_id,
                RunRow.strategy_id == scope.strategy_id,
            )
        )
        if task_row is None or run_row is None:
            raise NotFoundError("task or run not found")
        thread_row = await session.scalar(
            select(ThreadRow).where(
                ThreadRow.id == task_row.thread_id,
                ThreadRow.organization_id == scope.organization_id,
                ThreadRow.strategy_id == scope.strategy_id,
            )
        )
        if thread_row is None:
            raise NotFoundError("thread not found")

        observation_rows = (
            await session.scalars(
                select(ObservationRow)
                .where(
                    ObservationRow.run_id == run_id,
                    ObservationRow.organization_id == scope.organization_id,
                    ObservationRow.strategy_id == scope.strategy_id,
                )
                .order_by(ObservationRow.observed_at, ObservationRow.id)
            )
        ).all()
        claim_rows = (
            await session.scalars(
                select(ClaimRow)
                .where(
                    ClaimRow.run_id == run_id,
                    ClaimRow.organization_id == scope.organization_id,
                    ClaimRow.strategy_id == scope.strategy_id,
                )
                .order_by(ClaimRow.id)
            )
        ).all()
        event_rows = (
            await session.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id, RunEventRow.task_id == task_id)
                .order_by(RunEventRow.sequence)
            )
        ).all()
        return RunBundle(
            thread=_thread_model(thread_row),
            task=_task_model(task_row),
            run=_run_model(run_row),
            observations=tuple(_observation_model(row) for row in observation_rows),
            claims=tuple(_claim_model(row) for row in claim_rows),
            events=tuple(_event_model(row) for row in event_rows),
        )


def _validate_new_records(
    run: Run,
    observations: tuple[Observation, ...],
) -> None:
    for observation in observations:
        if observation.run_id != run.id or observation.scope != run.scope:
            raise StoreError("observation is outside the run scope")


def _task_row(task: Task) -> TaskRow:
    return TaskRow(
        id=task.id,
        thread_id=task.thread_id,
        organization_id=task.scope.organization_id,
        strategy_id=task.scope.strategy_id,
        objective=task.objective,
        parent_task_id=task.parent_task_id,
        continuation_kind=task.continuation_kind,
        mapping_version=task.mapping_version,
        status=task.status.value,
        observation_ids=list(task.observation_ids),
        verifier_feedback=list(task.verifier_feedback),
        scratchpad=[step.model_dump(mode="json") for step in task.scratchpad],
        step_plan=[step.model_dump(mode="json") for step in task.step_plan],
        final_output=task.final_output,
        version=task.version,
    )


def _run_row(run: Run) -> RunRow:
    return RunRow(
        id=run.id,
        task_id=run.task_id,
        organization_id=run.scope.organization_id,
        strategy_id=run.scope.strategy_id,
        status=run.status.value,
        phase=run.phase.value,
        iteration=run.iteration,
        limits=cast(dict[str, object], run.limits.model_dump(mode="json")),
        usage=cast(dict[str, object], run.usage.model_dump(mode="json")),
        started_at=run.started_at,
        deadline_at=run.deadline_at,
        final_output=run.final_output,
        terminal_reason=run.terminal_reason,
        event_sequence=0,
        version=run.version,
    )


def _observation_row(observation: Observation) -> ObservationRow:
    return ObservationRow(
        id=observation.id,
        run_id=observation.run_id,
        organization_id=observation.scope.organization_id,
        strategy_id=observation.scope.strategy_id,
        tool_call_id=observation.tool_call_id,
        kind=observation.kind,
        data=cast(dict[str, object], observation.data),
        source=cast(dict[str, object], observation.source.model_dump(mode="json")),
        observed_at=observation.observed_at,
        expires_at=observation.expires_at,
        raw_hash=observation.raw_hash,
        status=observation.status.value,
        quality=observation.quality.value,
        schema_version=observation.schema_version,
        normalization_version=observation.normalization_version,
        rejection_code=observation.rejection_code,
    )


def _claim_row(claim: Claim) -> ClaimRow:
    return ClaimRow(
        id=claim.id,
        run_id=claim.run_id,
        organization_id=claim.scope.organization_id,
        strategy_id=claim.scope.strategy_id,
        kind=claim.kind.value,
        statement=claim.statement,
        observation_ids=list(claim.observation_ids),
    )


def _apply_task(row: TaskRow, task: Task) -> None:
    row.status = task.status.value
    row.observation_ids = list(task.observation_ids)
    row.verifier_feedback = list(task.verifier_feedback)
    row.scratchpad = [step.model_dump(mode="json") for step in task.scratchpad]
    row.step_plan = [step.model_dump(mode="json") for step in task.step_plan]
    row.final_output = task.final_output
    row.parent_task_id = task.parent_task_id
    row.continuation_kind = task.continuation_kind
    row.mapping_version = task.mapping_version
    row.version = task.version
    if task.status.value in {"completed", "failed", "cancelled"}:
        row.lease_owner = None
        row.lease_token = None
        row.lease_expires_at = None
        row.heartbeat_at = None
        row.retry_after = None


def _require_lease(
    row: TaskRow,
    lease_owner: str | None,
    lease_token: str | None,
) -> None:
    if (lease_owner is None) != (lease_token is None):
        raise ValueError("lease_owner and lease_token must be supplied together")
    if lease_owner is not None and (
        row.lease_owner != lease_owner or row.lease_token != lease_token
    ):
        raise ConcurrencyError("task lease is stale or owned by another worker")


class LeaseBoundRunStore:
    """Run-store view that fences every mutation with one durable Task lease."""

    def __init__(self, store: PostgresRunStore, lease: TaskLease) -> None:
        self._store = store
        self._lease = lease

    async def seed(self, thread: Thread, task: Task, run: Run) -> RunBundle:
        return await self._store.seed(thread, task, run)

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        return await self._store.load(task_id, run_id, scope)

    async def commit(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task: Task,
        run: Run,
        observations: tuple[Observation, ...] = (),
        events: tuple[EventDraft, ...] = (),
    ) -> RunBundle:
        return await self._store.commit(
            expected_task_version=expected_task_version,
            expected_run_version=expected_run_version,
            task=task,
            run=run,
            observations=observations,
            events=events,
            lease_owner=self._lease.owner,
            lease_token=self._lease.token,
        )

    async def complete_verified(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task_id: str,
        run_id: str,
        scope: ScopeKey,
        usage: BudgetUsage,
        completion: VerifiedCompletion,
        preceding_events: tuple[EventDraft, ...] = (),
    ) -> RunBundle:
        return await self._store.complete_verified(
            expected_task_version=expected_task_version,
            expected_run_version=expected_run_version,
            task_id=task_id,
            run_id=run_id,
            scope=scope,
            usage=usage,
            completion=completion,
            preceding_events=preceding_events,
            lease_owner=self._lease.owner,
            lease_token=self._lease.token,
        )


def _apply_run(row: RunRow, run: Run) -> None:
    row.status = run.status.value
    row.phase = run.phase.value
    row.iteration = run.iteration
    row.limits = cast(dict[str, object], run.limits.model_dump(mode="json"))
    row.usage = cast(dict[str, object], run.usage.model_dump(mode="json"))
    row.started_at = run.started_at
    row.deadline_at = run.deadline_at
    row.final_output = run.final_output
    row.terminal_reason = run.terminal_reason
    row.version = run.version


def _thread_model(row: ThreadRow) -> Thread:
    return Thread(
        id=row.id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        origin=OriginRef(
            provider=row.origin_provider,
            external_thread_id=row.external_thread_id,
            external_event_id=row.external_event_id,
            external_channel_id=row.external_channel_id,
        ),
        mapping_version=row.mapping_version,
        version=row.version,
    )


def _task_model(row: TaskRow) -> Task:
    return Task(
        id=row.id,
        thread_id=row.thread_id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        objective=row.objective,
        parent_task_id=row.parent_task_id,
        continuation_kind=row.continuation_kind,
        mapping_version=row.mapping_version,
        status=TaskStatus(row.status),
        observation_ids=tuple(row.observation_ids),
        verifier_feedback=tuple(row.verifier_feedback),
        scratchpad=tuple(ReasoningStep.model_validate(item) for item in (row.scratchpad or ())),
        step_plan=tuple(PlannedStep.model_validate(item) for item in (row.step_plan or ())),
        final_output=row.final_output,
        version=row.version,
    )


def _run_model(row: RunRow) -> Run:
    return Run(
        id=row.id,
        task_id=row.task_id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        status=RunStatus(row.status),
        phase=RunPhase(row.phase),
        iteration=row.iteration,
        limits=BudgetLimits.model_validate(row.limits),
        usage=BudgetUsage.model_validate(row.usage),
        started_at=row.started_at,
        deadline_at=row.deadline_at,
        final_output=row.final_output,
        terminal_reason=row.terminal_reason,
        version=row.version,
    )


def _observation_model(row: ObservationRow) -> Observation:
    return Observation(
        id=row.id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        run_id=row.run_id,
        tool_call_id=row.tool_call_id,
        kind=row.kind,
        data=cast(dict[str, JsonValue], row.data),
        source=SourceRef.model_validate(row.source),
        observed_at=row.observed_at,
        expires_at=row.expires_at,
        raw_hash=row.raw_hash,
        status=row.status,
        quality=row.quality,
        schema_version=row.schema_version,
        normalization_version=row.normalization_version,
        rejection_code=row.rejection_code,
    )


def _claim_model(row: ClaimRow) -> Claim:
    return Claim(
        id=row.id,
        scope=ScopeKey(
            organization_id=row.organization_id,
            strategy_id=row.strategy_id,
        ),
        run_id=row.run_id,
        kind=ClaimKind(row.kind),
        statement=row.statement,
        observation_ids=tuple(row.observation_ids),
    )


def _event_model(row: RunEventRow) -> RunEvent:
    return RunEvent(
        id=row.id,
        run_id=row.run_id,
        task_id=row.task_id,
        sequence=row.sequence,
        type=EventType(row.type),
        occurred_at=row.occurred_at,
        iteration=row.iteration,
        schema_version=row.schema_version,
        payload=cast(dict[str, JsonValue], row.payload),
    )
