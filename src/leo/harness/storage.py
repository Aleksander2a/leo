"""Deterministic in-memory implementation of the runtime persistence port."""

from __future__ import annotations

import asyncio

from leo.harness.models import (
    BudgetUsage,
    Claim,
    EventDraft,
    EventType,
    Observation,
    Run,
    RunBundle,
    RunEvent,
    ScopeKey,
    Task,
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


class InMemoryRunStore:
    """Atomic snapshot/event store used by deterministic tests and smoke runs."""

    def __init__(self, clock: Clock, ids: IdGenerator) -> None:
        self._clock = clock
        self._ids = ids
        self._threads: dict[str, Thread] = {}
        self._tasks: dict[str, Task] = {}
        self._runs: dict[str, Run] = {}
        self._observations: dict[str, Observation] = {}
        self._claims: dict[str, Claim] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._lock = asyncio.Lock()

    async def seed(self, thread: Thread, task: Task, run: Run) -> RunBundle:
        async with self._lock:
            if thread.id in self._threads or task.id in self._tasks or run.id in self._runs:
                raise ConcurrencyError("thread, task, or run already exists")
            validate_seed(thread, task, run)
            self._threads[thread.id] = thread.model_copy(deep=True)
            self._tasks[task.id] = task.model_copy(deep=True)
            self._runs[run.id] = run.model_copy(deep=True)
            self._events[run.id] = []
            return self._bundle(task.id, run.id)

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle:
        async with self._lock:
            bundle = self._bundle(task_id, run_id)
            if bundle.task.scope != scope or bundle.run.scope != scope:
                raise NotFoundError("task or run not found")
            return bundle

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
        async with self._lock:
            current_task = self._tasks.get(task.id)
            current_run = self._runs.get(run.id)
            if current_task is None or current_run is None:
                raise NotFoundError("task or run not found")
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

            for observation in observations:
                if observation.id in self._observations:
                    raise ConcurrencyError("observation already exists")
                if observation.run_id != run.id or observation.scope != run.scope:
                    raise StoreError("observation is outside the run scope")

            if len({item.id for item in observations}) != len(observations):
                raise ConcurrencyError("duplicate observation in atomic step")
            available_observation_ids = {
                item.id for item in self._observations.values() if item.run_id == run.id
            }.union(item.id for item in observations)
            if any(item not in available_observation_ids for item in task.observation_ids):
                raise StoreError("task references an unavailable observation")

            self._tasks[task.id] = task.model_copy(deep=True)
            self._runs[run.id] = run.model_copy(deep=True)
            for observation in observations:
                self._observations[observation.id] = observation.model_copy(deep=True)
            self._append_events(task, run, events)

            return self._bundle(task.id, run.id)

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
        async with self._lock:
            current_task = self._tasks.get(task_id)
            current_run = self._runs.get(run_id)
            if current_task is None or current_run is None:
                raise NotFoundError("task or run not found")
            if current_task.scope != scope or current_run.scope != scope:
                raise NotFoundError("task or run not found")
            if current_task.version != expected_task_version:
                raise ConcurrencyError("stale task version")
            if current_run.version != expected_run_version:
                raise ConcurrencyError("stale run version")

            available_observation_ids = frozenset(
                item.id for item in self._observations.values() if item.run_id == run_id
            )
            validate_verified_completion(
                current_task,
                current_run,
                usage,
                completion,
                available_observation_ids,
                sanitize_event_drafts(preceding_events, current_run),
            )
            for claim in completion.claims:
                if claim.id in self._claims:
                    raise ConcurrencyError("claim already exists")

            task, run = complete_task_and_run(
                current_task,
                current_run,
                completion,
                usage=usage,
            )
            safe_preceding_events = sanitize_event_drafts(preceding_events, current_run)
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
            self._tasks[task.id] = task.model_copy(deep=True)
            self._runs[run.id] = run.model_copy(deep=True)
            for claim in completion.claims:
                self._claims[claim.id] = claim.model_copy(deep=True)
            self._append_events(task, run, safe_authoritative_events)
            return self._bundle(task.id, run.id)

    def _append_events(
        self,
        task: Task,
        run: Run,
        drafts: tuple[EventDraft, ...],
    ) -> None:
        run_events = self._events.setdefault(run.id, [])
        for draft in drafts:
            event = RunEvent(
                id=self._ids.new("evt"),
                run_id=run.id,
                task_id=task.id,
                sequence=len(run_events) + 1,
                type=draft.type,
                occurred_at=self._clock.now(),
                iteration=draft.iteration,
                payload=draft.payload,
            )
            run_events.append(event)

    def _bundle(self, task_id: str, run_id: str) -> RunBundle:
        task = self._tasks.get(task_id)
        run = self._runs.get(run_id)
        if task is None or run is None:
            raise NotFoundError("task or run not found")
        thread = self._threads.get(task.thread_id)
        if thread is None:
            raise NotFoundError("thread not found")
        observations = tuple(
            observation.model_copy(deep=True)
            for observation in self._observations.values()
            if observation.run_id == run_id
        )
        events = tuple(event.model_copy(deep=True) for event in self._events.get(run_id, []))
        claims = tuple(
            claim.model_copy(deep=True) for claim in self._claims.values() if claim.run_id == run_id
        )
        return RunBundle(
            thread=thread.model_copy(deep=True),
            task=task.model_copy(deep=True),
            run=run.model_copy(deep=True),
            observations=observations,
            claims=claims,
            events=events,
        )
