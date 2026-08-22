from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from leo.harness.models import (
    BudgetUsage,
    Claim,
    ClaimKind,
    EventDraft,
    EventType,
    Observation,
    OriginRef,
    Run,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
)
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.harness.transitions import advance_step, start_task_and_run, time_out_task_and_run


def _records() -> tuple[ScopeKey, Thread, Task, Run]:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread",
        scope=scope,
        origin=OriginRef(provider="test", external_thread_id="thread"),
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    return scope, thread, task, run


def _observation(scope: ScopeKey, run_id: str) -> Observation:
    return Observation(
        id="obs",
        scope=scope,
        run_id=run_id,
        tool_call_id="call",
        kind="quote",
        data={"price": 181.25},
        source=SourceRef(provider="test", reference="quote"),
        observed_at=datetime(2026, 8, 20, 12, 0, tzinfo=UTC),
        raw_hash="hash",
    )


def _completion(scope: ScopeKey, run_id: str) -> VerifiedCompletion:
    claim = Claim(
        id="claim",
        scope=scope,
        run_id=run_id,
        kind=ClaimKind.SOURCE_CLAIM,
        statement="The quote is 181.25.",
        observation_ids=("obs",),
    )
    return VerifiedCompletion(
        answer="The quote is 181.25.",
        claims=(claim,),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=(VerifierCheck(name="quote", passed=True, detail="quote is supported"),),
            retryable=False,
        ),
    )


async def _seed(harness: Any) -> tuple[ScopeKey, Thread, Task, Run]:
    scope, thread, task, run = _records()
    await harness.store.seed(thread, task, run)
    return scope, thread, task, run


async def _start(harness: Any, task: Task, run: Run) -> tuple[Task, Run]:
    active_task, active_run = start_task_and_run(task, run, started_at=harness.clock.now())
    await harness.store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    return active_task, active_run


async def _observe(harness: Any, bundle: Any, scope: ScopeKey, run_id: str) -> Any:
    observation = _observation(scope, run_id)
    observed_task, observed_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=1),
        observation_ids=(observation.id,),
    )
    return await harness.store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=observed_task,
        run=observed_run,
        observations=(observation,),
        events=(
            EventDraft(
                type=EventType.OBSERVATION_CREATED,
                iteration=observed_run.iteration,
                payload={
                    "observation_id": observation.id,
                    "tool_call_id": observation.tool_call_id,
                },
            ),
        ),
    )


async def _completed_bundle(harness: Any) -> Any:
    scope, _, task, run = await _seed(harness)
    await _start(harness, task, run)
    started = await harness.store.load(task.id, run.id, scope)
    observed = await _observe(harness, started, scope, run.id)
    return await harness.store.complete_verified(
        expected_task_version=observed.task.version,
        expected_run_version=observed.run.version,
        task_id=task.id,
        run_id=run.id,
        scope=scope,
        usage=BudgetUsage(model_calls=2, tool_calls=1),
        completion=_completion(scope, run.id),
    )


def _normalized(bundle: Any) -> dict[str, Any]:
    return bundle.model_dump(mode="json")


@pytest.mark.asyncio
async def test_shared_store_contract_seed_observe_complete_reload(store_harness: Any) -> None:
    completed = await _completed_bundle(store_harness)
    loaded = await store_harness.store.load(
        completed.task.id, completed.run.id, completed.run.scope
    )

    assert _normalized(loaded) == _normalized(completed)
    assert completed.task.version == 3
    assert completed.run.version == 3
    assert completed.run.status.value == "completed"
    assert [event.sequence for event in completed.events] == [1, 2, 3, 4]
    assert completed.events[-1].type is EventType.RUN_COMPLETED


@pytest.mark.asyncio
async def test_shared_store_contract_scope_and_cas_fail_closed(store_harness: Any) -> None:
    scope, _, task, run = await _seed(store_harness)
    with pytest.raises(NotFoundError):
        await store_harness.store.load(
            task.id,
            run.id,
            ScopeKey(organization_id="other", strategy_id=scope.strategy_id),
        )

    active_task, active_run = await _start(store_harness, task, run)
    active = await store_harness.store.load(task.id, run.id, scope)
    other_scope = ScopeKey(organization_id="other", strategy_id=scope.strategy_id)
    wrong_observation = _observation(other_scope, run.id)
    changed_task, changed_run = advance_step(
        active.task,
        active.run,
        usage=BudgetUsage(model_calls=1, tool_calls=1),
        observation_ids=(wrong_observation.id,),
    )
    with pytest.raises(StoreError, match="outside the run scope"):
        await store_harness.store.commit(
            expected_task_version=active.task.version,
            expected_run_version=active.run.version,
            task=changed_task,
            run=changed_run,
            observations=(wrong_observation,),
            events=(
                EventDraft(
                    type=EventType.OBSERVATION_CREATED,
                    iteration=changed_run.iteration,
                    payload={
                        "observation_id": wrong_observation.id,
                        "tool_call_id": wrong_observation.tool_call_id,
                    },
                ),
            ),
        )

    observed = await _observe(store_harness, active, scope, run.id)
    wrong_completion = _completion(other_scope, run.id)
    with pytest.raises(StoreError, match="outside the run scope"):
        await store_harness.store.complete_verified(
            expected_task_version=observed.task.version,
            expected_run_version=observed.run.version,
            task_id=task.id,
            run_id=run.id,
            scope=scope,
            usage=BudgetUsage(model_calls=2, tool_calls=1),
            completion=wrong_completion,
        )

    with pytest.raises(ConcurrencyError, match="stale task version"):
        await store_harness.store.commit(
            expected_task_version=task.version,
            expected_run_version=run.version,
            task=active_task,
            run=active_run,
            events=(
                EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
            ),
        )


@pytest.mark.asyncio
async def test_shared_store_contract_generic_completion_and_failed_commit_rollback(
    store_harness: Any,
) -> None:
    scope, _, task, run = await _seed(store_harness)
    await _start(store_harness, task, run)
    active = await store_harness.store.load(task.id, run.id, scope)

    with pytest.raises(StoreError, match="illegal active task/run transition"):
        await store_harness.store.commit(
            expected_task_version=active.task.version,
            expected_run_version=active.run.version,
            task=active.task.model_copy(
                update={"status": "completed", "final_output": "forged", "version": 2}
            ),
            run=active.run.model_copy(
                update={
                    "status": "completed",
                    "iteration": 1,
                    "final_output": "forged",
                    "terminal_reason": "forged",
                    "started_at": active.run.started_at,
                    "version": 2,
                }
            ),
            events=(
                EventDraft(type=EventType.VERIFICATION_PASSED, iteration=1),
                EventDraft(type=EventType.RUN_COMPLETED, iteration=1),
            ),
        )
    assert _normalized(await store_harness.store.load(task.id, run.id, scope)) == _normalized(
        active
    )

    observed = await _observe(store_harness, active, scope, run.id)
    before = await store_harness.store.load(task.id, run.id, scope)
    duplicate = _observation(scope, run.id)
    changed_task, changed_run = advance_step(
        observed.task,
        observed.run,
        usage=BudgetUsage(model_calls=2, tool_calls=1),
        observation_ids=(duplicate.id,),
    )
    with pytest.raises(ConcurrencyError):
        await store_harness.store.commit(
            expected_task_version=observed.task.version,
            expected_run_version=observed.run.version,
            task=changed_task,
            run=changed_run,
            observations=(duplicate,),
            events=(
                EventDraft(
                    type=EventType.MODEL_CALLED,
                    iteration=changed_run.iteration,
                    payload={"decision": "tool_requests"},
                ),
            ),
        )
    after = await store_harness.store.load(task.id, run.id, scope)
    assert _normalized(after) == _normalized(before)


@pytest.mark.asyncio
async def test_shared_store_contract_concurrent_cas_has_one_winner(store_harness: Any) -> None:
    _, _, task, run = await _seed(store_harness)
    active_task, active_run = start_task_and_run(task, run, started_at=store_harness.clock.now())

    async def attempt() -> Any:
        try:
            return await store_harness.store.commit(
                expected_task_version=task.version,
                expected_run_version=run.version,
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
        except Exception as exc:
            return exc

    results = await asyncio.gather(attempt(), attempt())
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ConcurrencyError) for result in results) == 1


@pytest.mark.asyncio
async def test_shared_store_contract_concurrent_completion_has_one_winner(
    store_harness: Any,
) -> None:
    scope, _, task, run = await _seed(store_harness)
    await _start(store_harness, task, run)
    active = await store_harness.store.load(task.id, run.id, scope)
    observed = await _observe(store_harness, active, scope, run.id)
    completion = _completion(scope, run.id)

    async def attempt() -> Any:
        try:
            return await store_harness.store.complete_verified(
                expected_task_version=observed.task.version,
                expected_run_version=observed.run.version,
                task_id=task.id,
                run_id=run.id,
                scope=scope,
                usage=BudgetUsage(model_calls=2, tool_calls=1),
                completion=completion,
            )
        except Exception as exc:
            return exc

    results = await asyncio.gather(attempt(), attempt())
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ConcurrencyError) for result in results) == 1
    final = await store_harness.store.load(task.id, run.id, scope)
    assert final.run.status.value == "completed"
    assert len(final.claims) == 1


@pytest.mark.asyncio
async def test_postgres_completion_timeout_race_has_one_terminal_winner(
    postgres_store: Any,
) -> None:
    scope, _, task, run = await _seed(postgres_store)
    await _start(postgres_store, task, run)
    active = await postgres_store.store.load(task.id, run.id, scope)
    observed = await _observe(postgres_store, active, scope, run.id)
    timeout_task, timeout_run = time_out_task_and_run(
        observed.task,
        observed.run,
        "run_deadline_exceeded",
        usage=observed.run.usage,
    )
    completion = _completion(scope, run.id)

    async def timeout_attempt() -> Any:
        try:
            return await postgres_store.store.commit(
                expected_task_version=observed.task.version,
                expected_run_version=observed.run.version,
                task=timeout_task,
                run=timeout_run,
                events=(
                    EventDraft(
                        type=EventType.RUN_TIMED_OUT,
                        iteration=timeout_run.iteration,
                        payload={"reason": "run_deadline_exceeded"},
                    ),
                ),
            )
        except Exception as exc:
            return exc

    async def completion_attempt() -> Any:
        try:
            return await postgres_store.store.complete_verified(
                expected_task_version=observed.task.version,
                expected_run_version=observed.run.version,
                task_id=task.id,
                run_id=run.id,
                scope=scope,
                usage=BudgetUsage(model_calls=2, tool_calls=1),
                completion=completion,
            )
        except Exception as exc:
            return exc

    results = await asyncio.gather(timeout_attempt(), completion_attempt())
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, ConcurrencyError) for result in results) == 1
    final = await postgres_store.store.load(task.id, run.id, scope)
    assert final.run.status.value in {"completed", "timed_out"}
    assert final.task.status.value in {"completed", "failed"}


@pytest.mark.asyncio
async def test_postgres_and_memory_event_timelines_normalize_identically(
    postgres_store: Any,
) -> None:
    from leo.harness.storage import InMemoryRunStore
    from leo.integrations.fake import FixedClock, SequentialIdGenerator

    memory_clock = FixedClock()
    memory = type(postgres_store)(
        store=InMemoryRunStore(memory_clock, SequentialIdGenerator()),
        clock=memory_clock,
    )
    memory_completed = await _completed_bundle(memory)
    postgres_completed = await _completed_bundle(postgres_store)

    assert _normalized(memory_completed) == _normalized(postgres_completed)
