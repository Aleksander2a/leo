from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.harness.context import DefaultContextAssembler, context_manifest_event_payload
from leo.harness.context_budget import ContextBudget, Utf8TokenEstimator
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    EvidenceToolRequirement,
    ModelRequest,
    ModelTurnResult,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolArgumentConstraint,
    ToolChoiceMode,
    ToolEffect,
    ToolSpec,
    TrustedScope,
)
from leo.harness.storage import InMemoryRunStore
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FakeQuoteTool, FixedClock, SequentialIdGenerator
from leo.integrations.slack.context import (
    SlackHistoryContextManifest,
    slack_history_authority_ids,
)
from leo.replay import ReplaySourceManifest

SCOPE = ScopeKey(organization_id="demo-org", strategy_id="demo-strategy")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _bundle(
    *,
    parent_task_id: str | None = None,
    observations: tuple[Observation, ...] = (),
) -> RunBundle:
    thread = Thread(
        id="thread-1",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation-1"),
    )
    task = Task(
        id="task-1",
        thread_id=thread.id,
        scope=SCOPE,
        objective="Compare the scoped context and prepare the smallest supported answer.",
        parent_task_id=parent_task_id,
        continuation_kind="subagent" if parent_task_id is not None else "root",
    )
    run = Run(id="run-1", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=observations)


def _context_items() -> tuple[ContextItem, ...]:
    return (
        ContextItem(
            id="turn-old",
            kind=ContextItemKind.CONVERSATION_TURN,
            content="Old exact-conversation history " + "old " * 60,
            conversation_id="C-EXACT",
            source_scope=SCOPE,
        ),
        ContextItem(
            id="summary-1",
            kind=ContextItemKind.THREAD_SUMMARY,
            content="Source-linked thread summary " + "summary " * 50,
            conversation_id="C-EXACT",
            source_scope=SCOPE,
        ),
        ContextItem(
            id="memory-1",
            kind=ContextItemKind.MEMORY,
            content="Authorized exact-conversation memory " + "memory " * 40,
            conversation_id="C-EXACT",
            source_scope=SCOPE,
        ),
        ContextItem(
            id="child-result-1",
            kind=ContextItemKind.SUBAGENT_RESULT,
            content="Verified dependency result " + "child " * 30,
            conversation_id="C-EXACT",
            source_scope=SCOPE,
        ),
    )


def _optional_tool() -> ToolSpec:
    return ToolSpec(
        name="demo.optional_read",
        description="Return a large optional synthetic result. " + "schema " * 30,
        domain="demo",
        input_schema={"type": "object", "properties": {}},
        effect=ToolEffect.READ,
    )


def _observation() -> Observation:
    return Observation(
        id="observation-1",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="call-1",
        kind="demo.optional_read",
        data={"finding": "scoped observation " + "evidence " * 50},
        source=SourceRef(provider="fixture", reference="observation-1"),
        observed_at=NOW,
        raw_hash="a" * 64,
    )


def _included_ids(request: ModelRequest, source_type: str) -> tuple[str, ...]:
    return tuple(
        segment.source_ids[0]
        for segment in request.manifest.segments
        if segment.source_type == source_type and segment.included
    )


def test_whole_request_budget_manifest_exactly_matches_deterministic_selection() -> None:
    bundle = _bundle(observations=(_observation(),))
    items = _context_items()
    tool = _optional_tool()
    roomy = DefaultContextAssembler(
        context_items=items,
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(bundle, (tool,))
    pinned_tokens = sum(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.pinned and segment.source_type != "collection_summary"
    )
    child_tokens = next(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.source_ids and segment.source_ids[0] == "child-result-1"
    )
    assembler = DefaultContextAssembler(
        context_items=items,
        context_budget=ContextBudget(
            max_tokens=pinned_tokens + child_tokens,
            max_bytes=1_000_000,
        ),
    )

    first = assembler.assemble(bundle, (tool,))
    second = assembler.assemble(bundle, (tool,))

    assert first == second
    assert first.context_items == (items[-1],)
    assert first.observations == ()
    assert first.tools == ()
    assert first.tool_choice.mode is ToolChoiceMode.AUTO
    assert _included_ids(first, "context_item") == tuple(item.id for item in first.context_items)
    assert _included_ids(first, "observation") == tuple(item.id for item in first.observations)
    assert _included_ids(first, "tool_schema") == tuple(item.name for item in first.tools)
    manifest = first.manifest
    assert manifest.schema_version == 2
    assert manifest.budget_profile == "parent"
    assert manifest.estimator_version == Utf8TokenEstimator().version
    assert manifest.included_estimated_tokens <= manifest.max_tokens
    assert manifest.candidate_estimated_tokens == (
        manifest.included_estimated_tokens + manifest.excluded_estimated_tokens
    )
    assert manifest.candidate_estimated_bytes == (
        manifest.included_estimated_bytes + manifest.excluded_estimated_bytes
    )
    assert len(manifest.manifest_digest) == 64
    assert all(len(segment.content_hash) == 64 for segment in manifest.segments)
    assert all(segment.content_version for segment in manifest.segments)
    assert all(segment.reason for segment in manifest.segments)
    assert all(
        segment.reason.startswith("excluded_")
        for segment in manifest.segments
        if not segment.included
    )
    assert next(segment for segment in manifest.segments if segment.name == "task_lineage").included
    destination = next(
        segment for segment in manifest.segments if segment.name == "exact_destination"
    )
    assert destination.pinned and destination.included
    assert destination.source_ids == ("thread-1", "fixture", "conversation-1")
    selected_item = next(
        segment
        for segment in manifest.segments
        if segment.source_ids and segment.source_ids[0] == "child-result-1"
    )
    assert selected_item.source_ids == ("child-result-1", "C-EXACT")

    invalid = first.model_dump(mode="python")
    invalid["context_items"] = ()
    with pytest.raises(ValidationError, match="payload selection differs"):
        ModelRequest.model_validate(invalid)


def test_required_tool_schema_and_satisfying_observation_are_pinned() -> None:
    clock = FixedClock()
    quote = FakeQuoteTool(clock).spec
    requirement = EvidenceToolRequirement(
        observation_kind=quote.name,
        tool_name=quote.name,
        required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
    )
    bundle = _bundle()
    roomy_assembler = DefaultContextAssembler(
        evidence_requirements=(requirement,),
        clock=clock,
        context_items=_context_items(),
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    )
    roomy = roomy_assembler.assemble(bundle, (_optional_tool(), quote))
    required_tokens = sum(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.pinned and segment.source_type != "collection_summary"
    )
    request = DefaultContextAssembler(
        evidence_requirements=(requirement,),
        clock=clock,
        context_items=_context_items(),
        context_budget=ContextBudget(max_tokens=required_tokens, max_bytes=1_000_000),
    ).assemble(bundle, (_optional_tool(), quote))

    assert request.tool_choice.mode is ToolChoiceMode.REQUIRED
    assert request.tools == (quote,)
    required_schema = next(
        segment
        for segment in request.manifest.segments
        if segment.source_type == "tool_schema" and segment.source_ids == (quote.name,)
    )
    assert required_schema.pinned and required_schema.included

    observation = Observation(
        id="quote-observation",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="quote-call",
        kind=quote.name,
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="fixture", reference="quote"),
        observed_at=clock.now(),
        raw_hash="b" * 64,
    )
    observed = roomy_assembler.assemble(_bundle(observations=(observation,)), (quote,))
    observation_segment = next(
        segment
        for segment in observed.manifest.segments
        if segment.source_ids and segment.source_ids[0] == observation.id
    )
    assert observation_segment.pinned and observation_segment.included
    assert observed.observations == (observation,)


def test_thread_retention_is_pinned_through_the_final_model_manifest() -> None:
    root = ContextItem(
        id="thread-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Pinned Slack thread root",
        conversation_id="C-EXACT",
        source_scope=SCOPE,
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    supporting = ContextItem(
        id="thread-supporting",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="supporting " * 500,
        conversation_id="C-EXACT",
        source_scope=SCOPE,
        budget_priority=1,
    )
    roomy = DefaultContextAssembler(
        context_items=(root, supporting),
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(_bundle(), ())
    pinned_tokens = sum(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.pinned and segment.source_type != "collection_summary"
    )

    request = DefaultContextAssembler(
        context_items=(root, supporting),
        context_budget=ContextBudget(max_tokens=pinned_tokens, max_bytes=1_000_000),
    ).assemble(_bundle(), ())

    assert request.context_items == (root,)
    root_segment = next(
        segment for segment in request.manifest.segments if segment.source_ids[:1] == (root.id,)
    )
    supporting_segment = next(
        segment
        for segment in request.manifest.segments
        if segment.source_ids[:1] == (supporting.id,)
    )
    assert root_segment.pinned and root_segment.included
    assert not supporting_segment.pinned and not supporting_segment.included


def test_child_profile_applies_smaller_budget_without_broadening_context() -> None:
    items = _context_items()
    child_bundle = _bundle(parent_task_id="parent-task")
    roomy = DefaultContextAssembler(
        context_items=items,
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
        child_context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(child_bundle, ())
    pinned_tokens = sum(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.pinned and segment.source_type != "collection_summary"
    )
    child_tokens = next(
        segment.estimated_tokens
        for segment in roomy.manifest.segments
        if segment.source_ids and segment.source_ids[0] == "child-result-1"
    )
    child_limit = pinned_tokens + child_tokens

    request = DefaultContextAssembler(
        context_items=items,
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
        child_context_budget=ContextBudget(max_tokens=child_limit, max_bytes=1_000_000),
    ).assemble(child_bundle, ())

    assert request.manifest.budget_profile == "child"
    assert request.manifest.max_tokens == child_limit
    assert request.context_items == (items[-1],)
    assert {item.id for item in request.context_items}.issubset({item.id for item in items})
    lineage = next(
        segment for segment in request.manifest.segments if segment.name == "task_lineage"
    )
    assert lineage.source_type == "plan_child"
    assert lineage.source_ids == ("task-1", "parent-task")
    assert lineage.pinned and lineage.included


def test_context_source_manifest_projection_is_bounded_complete_and_replayable() -> None:
    context_items = tuple(
        ContextItem(
            id=f"turn-{index:02d}",
            kind=ContextItemKind.CONVERSATION_TURN,
            content=f"Synthetic authorized turn {index}",
            conversation_id=f"C-{index:02d}",
            source_scope=SCOPE,
        )
        for index in range(40)
    )
    request = DefaultContextAssembler(
        context_items=context_items,
        authority_snapshot_ids=("access:v7", "membership:v9"),
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(_bundle(), ())

    projected = context_manifest_event_payload(request.manifest)
    replayable = ReplaySourceManifest.model_validate(projected)

    assert replayable.manifest_digest == request.manifest.manifest_digest
    assert replayable.budget_profile == "parent"
    assert replayable.estimator_version == Utf8TokenEstimator().version
    assert replayable.included_estimated_tokens == request.manifest.included_estimated_tokens
    assert replayable.included_estimated_bytes == request.manifest.included_estimated_bytes
    assert len(replayable.included_source_ids) <= 32
    assert replayable.omitted_source_id_count > 0
    assert replayable.excluded_source_ids == ()
    assert {"access:v7", "membership:v9"}.issubset(replayable.included_source_ids)


def test_full_thread_authority_proof_is_content_free_bounded_and_replayable() -> None:
    history = SlackHistoryContextManifest(
        context_access_hash="a" * 64,
        requested_conversation_ids=("G-private",),
        loaded_conversation_ids=("G-private",),
        history_requests=2,
        raw_messages_scanned=8,
        eligible_messages_ranked=8,
        selected_messages=5,
        estimated_tokens=200,
        selection_digest="b" * 64,
        truncated=False,
        thread_triggered=True,
        thread_root_ts="100.000",
        thread_requests=3,
        thread_raw_messages_scanned=4,
        thread_messages_loaded=7,
        thread_messages_compacted=3,
        thread_compaction_digest="c" * 64,
        protected_thread_item_ids=("raw-root-id", "raw-decision-id"),
        thread_reopen_handles=("thr_private_one", "thr_private_two"),
        thread_complete=True,
        thread_source="persisted_complete",
        thread_coverage_reason="complete",
        thread_coverage_digest="d" * 64,
    )

    authority_ids = slack_history_authority_ids(history)
    request = DefaultContextAssembler(
        authority_snapshot_ids=authority_ids,
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(_bundle(), ())
    replayable = ReplaySourceManifest.model_validate(
        context_manifest_event_payload(request.manifest)
    )

    assert set(authority_ids).issubset(replayable.included_source_ids)
    assert len(authority_ids) == 11
    assert max(len(item.encode("utf-8")) for item in authority_ids) < 160
    encoded = "\n".join(replayable.included_source_ids)
    assert "raw-root-id" not in encoded
    assert "raw-decision-id" not in encoded
    assert "thr_private_one" not in encoded
    assert "thr_private_two" not in encoded
    assert "slack-thread-protected-count:2" in authority_ids
    assert "slack-thread-compacted-count:3" in authority_ids
    assert "slack-thread-reopen-handle-count:2" in authority_ids
    assert f"slack-thread-coverage-digest:{'d' * 64}" in authority_ids
    assert f"slack-history-selection-digest:{'b' * 64}" in authority_ids
    assert f"slack-thread-compaction-digest:{'c' * 64}" in authority_ids


def test_context_manifest_projection_hashes_an_excessive_authority_set_without_crashing() -> None:
    authority_ids = tuple(f"membership-source:{index:03d}" for index in range(40))
    request = DefaultContextAssembler(
        authority_snapshot_ids=authority_ids,
        context_budget=ContextBudget(max_tokens=100_000, max_bytes=1_000_000),
    ).assemble(_bundle(), ())

    replayable = ReplaySourceManifest.model_validate(
        context_manifest_event_payload(request.manifest)
    )

    assert len(replayable.included_source_ids) == 32
    assert replayable.included_source_ids[-1].startswith("membership-source:")
    assert any(
        source_id.startswith("authority-source-set-count:40:sha256:")
        for source_id in replayable.included_source_ids
    )
    assert replayable.omitted_source_id_count >= 9


def test_membership_change_rebuilds_manifest_without_stale_context_sources() -> None:
    before_items = _context_items()[:2]
    after_items = (
        before_items[0],
        ContextItem(
            id="turn-current-membership",
            kind=ContextItemKind.CONVERSATION_TURN,
            content="Only the refreshed authorized source remains eligible.",
            conversation_id="C-REFRESHED",
            source_scope=SCOPE,
        ),
    )
    before = DefaultContextAssembler(
        context_items=before_items,
        authority_snapshot_ids=("membership:v1",),
    ).assemble(_bundle(), ())
    after = DefaultContextAssembler(
        context_items=after_items,
        authority_snapshot_ids=("membership:v2",),
    ).assemble(_bundle(), ())

    before_projection = ReplaySourceManifest.model_validate(
        context_manifest_event_payload(before.manifest)
    )
    after_projection = ReplaySourceManifest.model_validate(
        context_manifest_event_payload(after.manifest)
    )

    assert before.manifest.manifest_digest != after.manifest.manifest_digest
    assert "summary-1" in before_projection.included_source_ids
    assert "summary-1" not in after_projection.included_source_ids
    assert "turn-current-membership" in after_projection.included_source_ids
    assert "membership:v1" not in after_projection.included_source_ids
    assert "membership:v2" in after_projection.included_source_ids


class _NeverCalledModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        self.calls += 1
        raise AssertionError("model must not run after pinned context overflow")


class _FailingEstimator:
    @property
    def version(self) -> str:
        return "failing-v1"

    def estimate_tokens(self, text: str) -> int:
        del text
        raise RuntimeError("provider tokenizer unavailable")


@pytest.mark.asyncio
async def test_pinned_overflow_fails_closed_before_model_call() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    model = _NeverCalledModel()
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(
            context_budget=ContextBudget(max_tokens=1, max_bytes=1_000_000)
        ),
        verifier=DeterministicCompletionVerifier(ids, clock, require_source_claim=False),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
    )

    assert model.calls == 0
    assert result.run.status is RunStatus.FAILED
    assert result.run.terminal_reason == ("context_assembly_error:pinned_context_exceeds_budget")


@pytest.mark.asyncio
async def test_estimator_failure_fails_closed_before_model_call() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    bundle = _bundle()
    store = InMemoryRunStore(clock, ids)
    await store.seed(bundle.thread, bundle.task, bundle.run)
    model = _NeverCalledModel()
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=ToolRegistry(()),
        context=DefaultContextAssembler(token_estimator=_FailingEstimator()),
        verifier=DeterministicCompletionVerifier(ids, clock, require_source_claim=False),
        clock=clock,
        ids=ids,
    )

    result = await coordinator.run(
        task_id=bundle.task.id,
        run_id=bundle.run.id,
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="U1"),
    )

    assert model.calls == 0
    assert result.run.status is RunStatus.FAILED
    assert result.run.terminal_reason == ("context_assembly_error:context_token_estimator_failed")
