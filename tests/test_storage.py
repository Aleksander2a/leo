from __future__ import annotations

import json

import pytest

from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.events import EventKind, normalize_run_timeline
from leo.harness.models import (
    BudgetUsage,
    Claim,
    ClaimKind,
    ContextItem,
    ContextItemKind,
    EventDraft,
    EventType,
    Observation,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    TaskStatus,
    Thread,
    TrustedScope,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
)
from leo.harness.persistence_rules import (
    CONTEXT_BUILT_PROJECTION_VERSION,
    EVENT_PAYLOAD_MAX_BYTES,
    VERIFICATION_CHECK_PROJECTION_VERSION,
    build_verification_passed_event,
    sanitize_event_drafts,
)
from leo.harness.storage import ConcurrencyError, InMemoryRunStore, NotFoundError, StoreError
from leo.harness.tools import ToolRegistry
from leo.harness.transitions import (
    advance_step,
    require_action_task_and_run,
    resume_task_and_run,
    start_task_and_run,
)
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.replay import ReplaySourceManifest


@pytest.mark.asyncio
async def test_compare_and_swap_rejects_stale_version() -> None:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(FixedClock(), SequentialIdGenerator())
    await store.seed(thread, task, run)

    next_task, next_run = start_task_and_run(task, run, started_at=FixedClock().now())
    await store.commit(
        expected_task_version=0,
        expected_run_version=0,
        task=next_task,
        run=next_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    with pytest.raises(ConcurrencyError, match="stale task version"):
        await store.commit(
            expected_task_version=0,
            expected_run_version=0,
            task=next_task,
            run=next_run,
        )


def test_six_conversation_dm_context_event_uses_content_free_whole_payload_projection() -> None:
    authority_marker = f"authority-source-set-count:15:sha256:{'a' * 64}"
    included_source_ids = sorted(
        {
            authority_marker,
            *(f"dm-union-source-{index:02d}-" + "x" * 136 for index in range(31)),
        }
    )
    payload = {
        "segments": [f"dm-union-segment-{index:02d}-" + "s" * 96 for index in range(64)],
        "tool_count": 12,
        "tool_choice": "auto",
        "required_tool": None,
        "required_arguments": [
            {"path": f"argument-{index:02d}", "value": "v" * 96} for index in range(24)
        ],
        "completion_contract": {
            "guidance": ["complete authorized DM union context" * 8 for _ in range(32)]
        },
        "source_manifest": {
            "schema_version": 1,
            "manifest_digest": "b" * 64,
            "budget_profile": "parent",
            "estimator_version": "utf8-v1",
            "included_source_ids": included_source_ids,
            "excluded_source_ids": [],
            "omitted_source_id_count": 80,
            "included_estimated_tokens": 12_000,
            "excluded_estimated_tokens": 4_000,
            "included_estimated_bytes": 48_000,
            "excluded_estimated_bytes": 16_000,
        },
        "catalog_version": "catalog-v1",
        "catalog_fingerprint": "c" * 64,
        "selection_fingerprint": "d" * 64,
        "selection_mode": "adaptive",
        "selection_reason": "bounded lexical eligible shortlist" * 32,
        "capability_candidates": [f"candidate-{index:03d}" for index in range(128)],
        "capability_selected": [f"selected-{index:02d}" for index in range(16)],
        "skill_selected": [f"skill-{index:02d}" for index in range(16)],
        "capability_query_hash": "e" * 64,
        "eligible_capability_count": 128,
    }
    raw_bytes = len(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    )
    assert raw_bytes > EVENT_PAYLOAD_MAX_BYTES
    run = Run(
        id="run-dm-union",
        task_id="task-dm-union",
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
    )
    draft = EventDraft(type=EventType.CONTEXT_BUILT, iteration=0, payload=payload)

    first = sanitize_event_drafts((draft,), run)[0]
    second = sanitize_event_drafts((draft,), run)[0]

    assert first == second
    encoded = json.dumps(
        first.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert len(encoded) <= EVENT_PAYLOAD_MAX_BYTES
    projection = first.payload["projection"]
    assert isinstance(projection, dict)
    assert projection["version"] == CONTEXT_BUILT_PROJECTION_VERSION
    assert projection["original_payload_bytes"] == raw_bytes
    assert projection["segments_count"] == 64
    assert projection["source_id_count"] == 111
    source_manifest = first.payload["source_manifest"]
    assert isinstance(source_manifest, dict)
    replayable = ReplaySourceManifest.model_validate(source_manifest)
    assert replayable.included_source_ids == (authority_marker,)
    assert replayable.omitted_source_id_count == 111
    serialized = encoded.decode("utf-8")
    assert "dm-union-source" not in serialized
    assert "complete authorized DM union context" not in serialized


class _FailAfterDmUnionContextModel:
    async def decide(self, request: object) -> object:
        del request
        raise RuntimeError("synthetic provider failure after context assembly")


@pytest.mark.asyncio
async def test_coordinator_persists_projected_six_conversation_dm_union_context() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    scope = ScopeKey(organization_id="org-dm-union", strategy_id="conversation")
    thread = Thread(
        id="thread-dm-union",
        scope=scope,
        origin=OriginRef(provider="slack", external_thread_id="slack:T1:D1:1.0"),
    )
    task = Task(
        id="task-dm-union",
        thread_id=thread.id,
        scope=scope,
        objective="Answer from my authorized conversations.",
    )
    run = Run(id="run-dm-union", task_id=task.id, scope=scope)
    context_items = tuple(
        ContextItem(
            id=(
                f"slack-history:T1:C{index % 6}:1787393{index:03d}.000001:"
                f"authorized-actor-{index:03d}"
            ),
            kind=ContextItemKind.CONVERSATION_TURN,
            content=f"Authorized synthetic conversation turn {index}.",
            conversation_id=f"C{index % 6}",
            source_scope=scope,
        )
        for index in range(120)
    )
    authority_ids = tuple(f"dm-union-authority-{index:02d}" for index in range(15))
    store = InMemoryRunStore(clock, ids)
    await store.seed(thread, task, run)
    coordinator = RunCoordinator(
        store=store,
        model=_FailAfterDmUnionContextModel(),  # type: ignore[arg-type]
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(
            context_items=context_items,
            authority_snapshot_ids=authority_ids,
        ),
        verifier=DeterministicCompletionVerifier(ids, clock),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=TrustedScope(namespace=scope, actor_id="U1"),
    )

    assert result.run.status is RunStatus.FAILED
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    encoded = json.dumps(
        context_event.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    assert len(encoded) <= EVENT_PAYLOAD_MAX_BYTES
    projection = context_event.payload["projection"]
    assert isinstance(projection, dict)
    assert projection["version"] == CONTEXT_BUILT_PROJECTION_VERSION
    assert projection["segments_count"] >= 120
    assert context_event.payload["segments"] == projection["segments_count"]
    source_manifest = context_event.payload["source_manifest"]
    assert isinstance(source_manifest, dict)
    replayable = ReplaySourceManifest.model_validate(source_manifest)
    assert any(
        source_id.startswith("authority-source-set-count:15:sha256:")
        for source_id in replayable.included_source_ids
    )
    timeline = normalize_run_timeline(result.events, scope)
    normalized_context = next(event for event in timeline if event.kind is EventKind.CONTEXT_BUILT)
    assert normalized_context.payload["count"] == projection["segments_count"]


@pytest.mark.asyncio
async def test_store_rejects_direct_unverified_completion() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)

    fabricated_task = task.model_copy(
        update={"status": TaskStatus.COMPLETED, "final_output": "fabricated", "version": 1}
    )
    fabricated_run = run.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "started_at": clock.now(),
            "final_output": "fabricated",
            "version": 1,
        }
    )
    with pytest.raises(StoreError, match="queued task/run"):
        await store.commit(
            expected_task_version=0,
            expected_run_version=0,
            task=fabricated_task,
            run=fabricated_run,
        )


@pytest.mark.asyncio
async def test_active_run_cannot_forge_completion_through_generic_commit() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    fabricated_task = bundle.task.model_copy(
        update={
            "status": TaskStatus.COMPLETED,
            "final_output": "fabricated",
            "version": bundle.task.version + 1,
        }
    )
    fabricated_run = bundle.run.model_copy(
        update={
            "status": RunStatus.COMPLETED,
            "iteration": bundle.run.iteration + 1,
            "final_output": "fabricated",
            "terminal_reason": "forged",
            "version": bundle.run.version + 1,
        }
    )
    with pytest.raises(StoreError, match="illegal active task/run transition"):
        await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=fabricated_task,
            run=fabricated_run,
            events=(
                EventDraft(type=EventType.VERIFICATION_PASSED, iteration=1),
                EventDraft(type=EventType.RUN_COMPLETED, iteration=1),
            ),
        )


@pytest.mark.asyncio
async def test_generic_commit_cannot_forge_authoritative_completion_events() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    next_task, next_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )

    with pytest.raises(StoreError, match="completion events require"):
        await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=next_task,
            run=next_run,
            events=(EventDraft(type=EventType.VERIFICATION_PASSED, iteration=1),),
        )


@pytest.mark.asyncio
async def test_store_load_fails_closed_for_wrong_scope() -> None:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(FixedClock(), SequentialIdGenerator())
    await store.seed(thread, task, run)

    with pytest.raises(NotFoundError, match="not found"):
        await store.load(
            task.id,
            run.id,
            ScopeKey(organization_id="org", strategy_id="other-strategy"),
        )


@pytest.mark.asyncio
async def test_store_persists_requires_action_and_resume_transitions() -> None:
    clock = FixedClock()
    task, run = (
        Task(
            id="task",
            thread_id="thread",
            scope=ScopeKey(organization_id="org", strategy_id="strategy"),
            objective="test",
        ),
        Run(
            id="run",
            task_id="task",
            scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        ),
    )
    thread = Thread(
        id="thread",
        scope=task.scope,
        origin=OriginRef(provider="test", external_thread_id="thread"),
    )
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    paused_task, paused_run = require_action_task_and_run(
        bundle.task,
        bundle.run,
        "needs_user_input",
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )
    paused = await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=paused_task,
        run=paused_run,
        events=(
            EventDraft(
                type=EventType.RUN_REQUIRES_ACTION,
                iteration=paused_run.iteration,
                payload={"reason": "needs_user_input"},
            ),
        ),
    )
    assert paused.task.status is TaskStatus.REQUIRES_ACTION
    assert paused.run.status is RunStatus.REQUIRES_ACTION
    assert paused.run.terminal_reason == "needs_user_input"

    resumed_task, resumed_run = resume_task_and_run(paused.task, paused.run)
    resumed = await store.commit(
        expected_task_version=paused.task.version,
        expected_run_version=paused.run.version,
        task=resumed_task,
        run=resumed_run,
        events=(EventDraft(type=EventType.RUN_RESUMED, iteration=resumed_run.iteration),),
    )
    assert resumed.task.status is TaskStatus.ACTIVE
    assert resumed.run.status is RunStatus.RUNNING
    assert resumed.run.terminal_reason is None


@pytest.mark.asyncio
async def test_store_rejects_unreachable_task_run_status_pair() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    mismatched_run = bundle.run.model_copy(
        update={
            "status": RunStatus.REQUIRES_ACTION,
            "terminal_reason": "needs_user_input",
            "version": bundle.run.version + 1,
        }
    )
    with pytest.raises(StoreError, match="lifecycle pair"):
        await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=bundle.task.model_copy(update={"version": bundle.task.version + 1}),
            run=mismatched_run,
        )


@pytest.mark.asyncio
async def test_state_change_requires_event_but_version_only_noop_is_allowed() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    changed_task, changed_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )
    with pytest.raises(StoreError, match="require at least one event"):
        await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=changed_task,
            run=changed_run,
        )

    no_op = await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=bundle.task.model_copy(update={"version": bundle.task.version + 1}),
        run=bundle.run.model_copy(update={"version": bundle.run.version + 1}),
    )
    assert no_op.task.version == bundle.task.version + 1
    assert no_op.run.version == bundle.run.version + 1
    assert no_op.events == bundle.events


@pytest.mark.asyncio
async def test_event_payload_is_redacted_and_store_owned_correlation_is_stable() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    next_task, next_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )

    result = await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=next_task,
        run=next_run,
        events=(
            EventDraft(
                type=EventType.RUN_FAILED,
                iteration=next_run.iteration,
                payload={
                    "reason": "model_gateway_error:http_401",
                    "detail": "Authorization: Bearer super-secret-token",
                },
            ),
        ),
    )

    event = result.events[-1]
    assert event.run_id == result.run.id
    assert event.task_id == result.task.id
    assert event.sequence == 2
    assert event.payload["reason"] == "model_gateway_error:http_401"
    assert event.payload["detail"] == "[REDACTED]"


@pytest.mark.asyncio
async def test_event_payload_rejects_forged_correlation_and_unknown_fields() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    next_task, next_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=0),
    )

    for payload, message in (
        ({"reason": "failed", "run_id": "forged"}, "correlation field"),
        ({"reason": "failed", "prompt": "untrusted"}, "not allowlisted"),
    ):
        with pytest.raises(StoreError, match=message):
            await store.commit(
                expected_task_version=bundle.task.version,
                expected_run_version=bundle.run.version,
                task=next_task,
                run=next_run,
                events=(
                    EventDraft(
                        type=EventType.RUN_FAILED,
                        iteration=next_run.iteration,
                        payload=payload,
                    ),
                ),
            )


@pytest.mark.asyncio
async def test_verified_completion_events_use_shared_redaction() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="test", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="test")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )
    observation = Observation(
        id="obs",
        scope=scope,
        run_id=run.id,
        tool_call_id="call",
        kind="quote",
        data={"price": 181.25},
        source=SourceRef(provider="test", reference="quote"),
        observed_at=clock.now(),
        raw_hash="hash",
    )
    observed_task, observed_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=1),
        observation_ids=(observation.id,),
    )
    bundle = await store.commit(
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
    claim = Claim(
        id="claim",
        scope=scope,
        run_id=run.id,
        kind=ClaimKind.SOURCE_CLAIM,
        statement="The quote is 181.25.",
        observation_ids=(observation.id,),
    )
    completion = VerifiedCompletion(
        answer="The quote is 181.25.",
        claims=(claim,),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=(
                VerifierCheck(
                    name="safe_check",
                    passed=True,
                    detail="ProviderError: Authorization: Bearer verifier-secret",
                ),
            ),
            retryable=False,
        ),
    )

    completed = await store.complete_verified(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task_id=task.id,
        run_id=run.id,
        scope=scope,
        usage=BudgetUsage(model_calls=2, tool_calls=1),
        completion=completion,
    )

    verification_event = next(
        event for event in completed.events if event.type is EventType.VERIFICATION_PASSED
    )
    detail = verification_event.payload["checks"][0]["detail"]
    assert isinstance(detail, str)
    assert "ProviderError" not in detail
    assert "Bearer" not in detail


@pytest.mark.asyncio
async def test_live_shaped_long_plan_completion_is_bounded_and_replayable() -> None:
    clock = FixedClock()
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    thread = Thread(
        id="thread", scope=scope, origin=OriginRef(provider="slack", external_thread_id="thread")
    )
    task = Task(id="task", thread_id=thread.id, scope=scope, objective="Compare NVDA evidence.")
    run = Run(id="run", task_id=task.id, scope=scope)
    store = InMemoryRunStore(clock, SequentialIdGenerator())
    await store.seed(thread, task, run)
    active_task, active_run = start_task_and_run(task, run, started_at=clock.now())
    bundle = await store.commit(
        expected_task_version=task.version,
        expected_run_version=run.version,
        task=active_task,
        run=active_run,
        events=(
            EventDraft(type=EventType.TASK_STARTED, iteration=0, payload={"phase": "research"}),
        ),
    )

    quote_statement = "NVDA's latest provider quote is $177.79 USD."
    filing_statement = "NVIDIA's latest 10-Q is available from the SEC filing document URL."
    plan_observation = Observation(
        id="obs-parent-plan",
        scope=scope,
        run_id=run.id,
        tool_call_id="call-parent-plan",
        kind="agent.execute_research_plan",
        data={
            "schema_version": "research-plan-result-v1",
            "plan_id": "plan-0ce33e3d",
            "status": "completed",
            "nodes": [
                {
                    "node_id": "quote",
                    "child_evidence": {
                        "schema_version": "child-evidence-v1",
                        "statement": quote_statement,
                        "provider": "finnhub",
                    },
                },
                {
                    "node_id": "filing",
                    "child_evidence": {
                        "schema_version": "child-evidence-v1",
                        "statement": filing_statement,
                        "provider": "sec",
                    },
                },
            ],
        },
        source=SourceRef(provider="leo", reference="plan-0ce33e3d"),
        observed_at=clock.now(),
        raw_hash="parent-plan-hash",
    )
    observed_task, observed_run = advance_step(
        bundle.task,
        bundle.run,
        usage=BudgetUsage(model_calls=1, tool_calls=2),
        observation_ids=(plan_observation.id,),
    )
    bundle = await store.commit(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task=observed_task,
        run=observed_run,
        observations=(plan_observation,),
        events=(
            EventDraft(
                type=EventType.OBSERVATION_CREATED,
                iteration=observed_run.iteration,
                payload={
                    "observation_id": plan_observation.id,
                    "tool_call_id": plan_observation.tool_call_id,
                },
            ),
        ),
    )

    answer = f"{quote_statement} {filing_statement} These sources report different facts."
    claims = (
        Claim(
            id="claim-quote",
            scope=scope,
            run_id=run.id,
            kind=ClaimKind.SOURCE_CLAIM,
            statement=quote_statement,
            observation_ids=(plan_observation.id,),
        ),
        Claim(
            id="claim-filing",
            scope=scope,
            run_id=run.id,
            kind=ClaimKind.SOURCE_CLAIM,
            statement=filing_statement,
            observation_ids=(plan_observation.id,),
        ),
    )
    checks = tuple(
        VerifierCheck(
            name=f"nested_child_evidence_{index:02d}",
            passed=True,
            detail=(
                f"Check {index}: {quote_statement} Source finnhub/quote/NVDA; "
                f"{filing_statement} Source sec/Archives/edgar/data/NVDA; "
                "both child-evidence-v1 digests, freshness windows, exact statements, "
                "distinct providers, plan coverage, uncertainty, and parent citations passed."
            ),
        )
        for index in range(48)
    )
    raw_event_payload = {
        "claim_count": len(claims),
        "check_count": len(checks),
        "checks": [check.model_dump(mode="json") for check in checks],
    }
    assert (
        len(
            json.dumps(
                raw_event_payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        )
        > EVENT_PAYLOAD_MAX_BYTES
    )
    completion = VerifiedCompletion(
        answer=answer,
        claims=claims,
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=checks,
            retryable=False,
        ),
    )

    completed = await store.complete_verified(
        expected_task_version=bundle.task.version,
        expected_run_version=bundle.run.version,
        task_id=task.id,
        run_id=run.id,
        scope=scope,
        usage=BudgetUsage(model_calls=2, tool_calls=2),
        completion=completion,
    )

    verification_event = next(
        event for event in completed.events if event.type is EventType.VERIFICATION_PASSED
    )
    encoded_event_size = len(
        json.dumps(
            verification_event.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    assert encoded_event_size <= EVENT_PAYLOAD_MAX_BYTES
    projection = verification_event.payload["projection"]
    assert isinstance(projection, dict)
    assert projection["version"] == VERIFICATION_CHECK_PROJECTION_VERSION
    assert projection["total_check_count"] == len(checks)
    assert projection["included_check_count"] + projection["omitted_check_count"] == len(checks)
    assert len(str(projection["checks_sha256"])) == 64

    replayed = await store.load(task.id, run.id, scope)
    assert replayed.task.final_output == answer
    assert replayed.run.final_output == answer
    assert replayed.claims == completed.claims == claims
    assert replayed.events == completed.events
    timeline = normalize_run_timeline(replayed.events, scope)
    normalized_verification = next(
        event for event in timeline if event.kind is EventKind.VERIFICATION
    )
    assert normalized_verification.payload["status"] == EventType.VERIFICATION_PASSED.value
    assert normalized_verification.payload["count"] == len(claims)
    assert normalized_verification.payload["check_count"] == len(checks)


def test_verification_projection_is_deterministic_for_multibyte_overflow() -> None:
    checks = tuple(
        VerifierCheck(
            name=f"nested_{index:03d}_" + "東京🚀" * 64,
            passed=True,
            detail=f"Verified nested source envelope {index}. " + "evidence " * 80,
        )
        for index in range(512)
    )
    completion = VerifiedCompletion(
        answer="The bounded nested evidence was verified.",
        claims=(),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=checks,
            retryable=False,
            allow_unsourced_completion=True,
        ),
    )
    run = Run(
        id="run-multibyte",
        task_id="task-multibyte",
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
    )

    first = build_verification_passed_event(completion, run)
    second = build_verification_passed_event(completion, run)

    assert first == second
    encoded_size = len(
        json.dumps(
            first.payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    assert encoded_size <= EVENT_PAYLOAD_MAX_BYTES
    projection = first.payload["projection"]
    assert isinstance(projection, dict)
    assert projection["omitted_check_count"] > 0
    projected_checks = first.payload["checks"]
    assert isinstance(projected_checks, list)
    assert projected_checks[0]["name"].startswith("check_0_")
