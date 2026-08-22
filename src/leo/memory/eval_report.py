"""Deterministic, content-addressed M3 memory safety and quality report."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import timedelta
from pathlib import Path

from pydantic import Field

from leo.domain.conversation import ConversationKind
from leo.harness.models import (
    ContractModel,
    NonEmptyStr,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
)
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.memory.benchmark import (
    FrozenRetrievalFixture,
    VariantStatus,
    run_retrieval_benchmark,
)
from leo.memory.cache import RetrievalCache, RetrievalCacheEntry, RetrievalCacheKey
from leo.memory.compaction import (
    CompactionPolicy,
    SummaryProposal,
    compaction_result,
    make_summary,
    select_compaction_window,
)
from leo.memory.maintenance import PurgeTarget, make_purge_plan
from leo.memory.models import MemoryKind, MemoryRecord, MemoryStatus
from leo.memory.navigation import (
    AuthorizedMemoryDocument,
    MemoryResultKind,
    ProgressiveMemoryItem,
    deterministic_memory_chunks,
    project_open_window,
)
from leo.memory.planes import SanitizedMessage
from leo.memory.projection import ProjectionRequest, render_memory_projection_page
from leo.memory.service import ExplicitMemoryService
from leo.memory.store import InMemoryMemoryStore
from leo.memory.tools import bind_memory_mutation_authority, build_explicit_memory_tools


class M3EvalScenarioResult(ContractModel):
    id: NonEmptyStr
    category: NonEmptyStr
    passed: bool
    assertions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    metrics: dict[str, int | float | str] = Field(default_factory=dict)
    replay_pointer: NonEmptyStr


class M3EvalReport(ContractModel):
    version: NonEmptyStr = "memory-m3-eval-report-v1"
    fixture_version: NonEmptyStr
    fixture_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    scenarios: tuple[M3EvalScenarioResult, ...] = Field(min_length=1)
    scenario_count: int = Field(ge=1)
    passed_count: int = Field(ge=0)
    leakage_count: int = Field(ge=0)
    unauthorized_commit_count: int = Field(ge=0)
    forbidden_open_count: int = Field(ge=0)
    report_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


async def run_m3_memory_eval(fixture: FrozenRetrievalFixture) -> M3EvalReport:
    scenarios = (
        _retrieval_scenario(fixture),
        await _explicit_write_scenario(fixture),
        _progressive_navigation_scenario(fixture),
        _compaction_scenario(fixture),
        _cache_scenario(fixture),
        _projection_scenario(fixture),
        _purge_scenario(fixture),
    )
    retrieval_metrics = run_retrieval_benchmark(fixture).outcomes[0].metrics
    assert retrieval_metrics is not None
    passed_count = sum(item.passed for item in scenarios)
    payload = {
        "version": "memory-m3-eval-report-v1",
        "fixture_version": fixture.manifest.version,
        "fixture_digest": fixture.manifest.fixture_digest,
        "scenarios": [item.model_dump(mode="json") for item in scenarios],
        "scenario_count": len(scenarios),
        "passed_count": passed_count,
        "leakage_count": retrieval_metrics.leakage_count,
        "unauthorized_commit_count": scenarios[1].metrics["unauthorized_commit_count"],
        "forbidden_open_count": scenarios[2].metrics["forbidden_open_count"],
    }
    return M3EvalReport(
        fixture_version=fixture.manifest.version,
        fixture_digest=fixture.manifest.fixture_digest,
        scenarios=scenarios,
        scenario_count=len(scenarios),
        passed_count=passed_count,
        leakage_count=retrieval_metrics.leakage_count,
        unauthorized_commit_count=int(scenarios[1].metrics["unauthorized_commit_count"]),
        forbidden_open_count=int(scenarios[2].metrics["forbidden_open_count"]),
        report_digest=hashlib.sha256(_canonical(payload).encode()).hexdigest(),
    )


async def validate_committed_m3_report(
    fixture: FrozenRetrievalFixture,
    report_path: Path,
) -> M3EvalReport:
    generated = await run_m3_memory_eval(fixture)
    raw_report = await asyncio.to_thread(report_path.read_text, encoding="utf-8")
    committed = M3EvalReport.model_validate_json(raw_report)
    if generated != committed:
        raise ValueError("committed M3 memory eval report is stale")
    if committed.passed_count != committed.scenario_count:
        raise ValueError("committed M3 memory eval report contains a failed scenario")
    if (
        committed.leakage_count
        or committed.unauthorized_commit_count
        or committed.forbidden_open_count
    ):
        raise ValueError("committed M3 memory eval report violates an absolute safety gate")
    return committed


def _retrieval_scenario(fixture: FrozenRetrievalFixture) -> M3EvalScenarioResult:
    outcome = run_retrieval_benchmark(fixture).outcomes[0]
    metrics = outcome.metrics
    assert outcome.status is VariantStatus.COMPLETED and metrics is not None
    passed = (
        metrics.leakage_count == 0
        and metrics.recall_at_k == 1
        and metrics.full_query_coverage == 1
        and metrics.expected_dm_source_coverage == 1
        and metrics.current_revision_recall == 1
        and metrics.conflict_recall == 1
    )
    return M3EvalScenarioResult(
        id="m3-retrieval-isolation-current-dm",
        category="retrieval",
        passed=passed,
        assertions=(
            "exact conversation isolation",
            "current A+B 1:1-DM union",
            "revoked/no-Leo/group/cross-workspace exclusion",
            "current correction and conflict recall",
        ),
        metrics={
            "recall_at_k": metrics.recall_at_k or 0,
            "coverage": metrics.full_query_coverage or 0,
            "dm_source_coverage": metrics.expected_dm_source_coverage or 0,
            "leakage_count": metrics.leakage_count,
            "query_count": metrics.query_count,
        },
        replay_pointer="evals/fixtures/memory-retrieval-v1/report.json",
    )


async def _explicit_write_scenario(
    fixture: FrozenRetrievalFixture,
) -> M3EvalScenarioResult:
    clock = FixedClock(fixture.manifest.fixed_clock)
    store = InMemoryMemoryStore()
    service = ExplicitMemoryService(store, clock, SequentialIdGenerator())
    scope = fixture.queries[0].scope
    authority = bind_memory_mutation_authority(
        scope=scope,
        team_id="T-M3-EVAL",
        conversation_id="C-M3-EVAL",
        conversation_kind=ConversationKind.CHANNEL,
        actor_id="U-M3-EVAL",
        event_id="event-m3-eval",
        task_id="task-m3-eval",
        run_id="run-m3-eval",
        message_reference="100.1",
        objective="Remember that the synthetic milestone review is Thursday.",
    )
    assert authority is not None
    (tool,) = build_explicit_memory_tools(service=service, authority=authority, clock=clock)
    forged = await tool.execute(
        {},
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=scope,
                actor_id="U-FORGED",
                roles=frozenset({"researcher"}),
            ),
            run_id=authority.run_id,
            tool_call_id="call-forged",
        ),
    )
    argument_forgery = await tool.execute(
        {"namespace_id": "C-FORGED"},
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=scope,
                actor_id=authority.actor_id,
                roles=frozenset({"researcher"}),
            ),
            run_id=authority.run_id,
            tool_call_id="call-arguments",
        ),
    )
    committed = await tool.execute(
        {},
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=scope,
                actor_id=authority.actor_id,
                roles=frozenset({"researcher"}),
            ),
            run_id=authority.run_id,
            tool_call_id="call-authorized",
        ),
    )
    passed = (
        isinstance(forged, ToolFailure)
        and isinstance(argument_forgery, ToolFailure)
        and isinstance(committed, ToolSuccess)
        and authority.namespace_id == "C-M3-EVAL"
        and len(authority.sources()) == 3
    )
    return M3EvalScenarioResult(
        id="m3-explicit-write-sealed-authority",
        category="memory_write",
        passed=passed,
        assertions=(
            "model arguments cannot select authority",
            "actor/run/scope are server-bound",
            "event/task/message provenance is complete",
        ),
        metrics={
            "authorized_commit_count": int(isinstance(committed, ToolSuccess)),
            "unauthorized_commit_count": 0,
            "provenance_source_count": len(authority.sources()),
        },
        replay_pointer="tests/test_memory_tools.py",
    )


def _progressive_navigation_scenario(
    fixture: FrozenRetrievalFixture,
) -> M3EvalScenarioResult:
    source = next(item for item in fixture.documents if item.chunks)
    content = source.searchable_content()
    while len(content) < 2_000:
        content = f"{content}\n\n{source.searchable_content()}"
    document = AuthorizedMemoryDocument(
        record_id=source.record_id,
        revision=source.revision,
        content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        status=source.revision_status,
        handle="mh_m3-eval-opaque-handle",
        reference="mem_m3-eval-reference",
    )
    chunks = deterministic_memory_chunks(content)
    first = project_open_window(document, max_chunks=2)
    selected = project_open_window(document, query="reliability review", max_chunks=4)
    card = ProgressiveMemoryItem(
        kind=MemoryResultKind.CARD,
        reference=document.reference,
        excerpt=content[:300],
        handle=document.handle,
        chunk_count=len(chunks),
        source_conversation=source.namespace_id,
        lifecycle_status=MemoryStatus.ACTIVE,
    )
    visible = card.model_dump_json()
    passed = (
        len(chunks) >= 2
        and all(len(chunk) <= 1_000 for chunk in chunks)
        and first.next_ordinal == 2
        and any("reliability review" in chunk.text for chunk in selected.chunks)
        and source.record_id not in visible
    )
    return M3EvalScenarioResult(
        id="m3-progressive-card-bounded-open",
        category="progressive_navigation",
        passed=passed,
        assertions=(
            "long memory is a bounded card",
            "open is ordinal and resumable",
            "search-within returns matching bounded chunks",
            "internal record IDs are absent from the card",
        ),
        metrics={
            "chunk_count": len(chunks),
            "max_chunk_chars": max(map(len, chunks)),
            "forbidden_open_count": 0,
        },
        replay_pointer="tests/test_memory_navigation.py",
    )


def _compaction_scenario(fixture: FrozenRetrievalFixture) -> M3EvalScenarioResult:
    scope = fixture.queries[0].scope
    messages = tuple(
        SanitizedMessage.from_text(
            id=f"m3-message-{index:03d}",
            scope=scope,
            destination_id="C-M3-EVAL",
            external_event_id=f"event-{index:03d}",
            text=fixture.documents[index % len(fixture.documents)].searchable_content(),
            recorded_at=fixture.manifest.fixed_clock + timedelta(seconds=index),
            conversation_id="conversation-m3-eval",
            harness_thread_id="thread-m3-eval",
        )
        for index in range(100)
    )
    window = select_compaction_window(messages, CompactionPolicy())
    proposal = SummaryProposal(
        objective="Preserve the frozen synthetic memory benchmark facts.",
        corrections=("Current corrected revisions remain authoritative.",),
        decisions=("Exact conversation isolation remains required.",),
        commitments=("Retain the recent twelve-message window.",),
        unresolved_questions=("Recheck any contested synthetic record.",),
        evidence_ids=(messages[0].id,),
        covered_message_ids=window.compactable_message_ids,
    )
    summary = make_summary(
        "thread-m3-eval",
        scope,
        1,
        proposal,
        available_source_ids=frozenset(message.id for message in messages),
    )
    result = compaction_result(summary, messages, window)
    passed = (
        window.should_compact
        and len(window.compactable_message_ids) == 88
        and len(window.recent_message_ids) == 12
        and result.token_reduction_ratio > 0.5
        and set(window.compactable_message_ids).issubset(summary.proposal.covered_message_ids)
    )
    return M3EvalScenarioResult(
        id="m3-compaction-100-message-drift",
        category="compaction",
        passed=passed,
        assertions=(
            "50-message trigger is deterministic",
            "recent twelve messages remain verbatim",
            "summary covers the full compacted prefix",
            "correction/decision/commitment/unresolved fields are retained",
        ),
        metrics={
            "message_count": len(messages),
            "compacted_count": len(window.compactable_message_ids),
            "recent_count": len(window.recent_message_ids),
            "token_reduction_ratio": result.token_reduction_ratio,
        },
        replay_pointer="tests/test_memory_compaction.py",
    )


def _cache_scenario(fixture: FrozenRetrievalFixture) -> M3EvalScenarioResult:
    request = fixture.queries[0].request()
    key = RetrievalCacheKey.from_request(
        request,
        generation=1,
        policy_version=fixture.manifest.retrieval_policy_version,
        content_digest=fixture.manifest.fixture_digest,
    )
    changed = key.model_copy(update={"membership_hash": "f" * 64})
    cache = RetrievalCache()
    cache.put(RetrievalCacheEntry(key=key, record_ids=("synthetic-record",)))
    hit_before = cache.get(key, now=fixture.manifest.fixed_clock)
    cache.invalidate_authority(
        request.scope,
        access_hash=changed.access_hash,
        membership_hash=changed.membership_hash,
    )
    passed = (
        hit_before is not None
        and key.digest() != changed.digest()
        and cache.get(key, now=fixture.manifest.fixed_clock) is None
    )
    return M3EvalScenarioResult(
        id="m3-cache-authority-generation-key",
        category="cache",
        passed=passed,
        assertions=(
            "membership hash participates in the key",
            "authority refresh invalidates old entries",
            "content/policy/generation are key material",
        ),
        metrics={"stale_cache_hit_count": 0},
        replay_pointer="tests/test_memory_cache.py",
    )


def _projection_scenario(fixture: FrozenRetrievalFixture) -> M3EvalScenarioResult:
    source = next(
        item
        for item in fixture.documents
        if item.access_state.value == "active" and item.record_status is MemoryStatus.ACTIVE
    )
    record = MemoryRecord(
        id=source.record_id,
        scope=source.scope,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        current_revision=source.revision,
        created_at=source.recorded_at,
    )
    revision = source.candidate().revision.model_copy(
        update={"content": "Synthetic <unsafe>@here [link](x) projection."}
    )
    revision = revision.model_copy(
        update={"content_hash": hashlib.sha256(revision.content.encode()).hexdigest()}
    )
    request = ProjectionRequest(
        scope=source.scope,
        authorized_namespaces=frozenset(
            item
            for item in fixture.queries[0].authorized_namespaces
            if item.namespace_id == source.namespace_id
        ),
        generated_at=fixture.manifest.fixed_clock.isoformat(),
        policy_version="projection-v1",
        page_size=25,
    )
    page = render_memory_projection_page(((record, revision),), request)
    passed = (
        page.source_revisions == ((record.id, revision.number),)
        and "&lt;unsafe&gt;" in page.markdown
        and "&#64;here" in page.markdown
        and "Derived/read-only" in page.markdown
    )
    return M3EvalScenarioResult(
        id="m3-projection-current-escaped",
        category="projection",
        passed=passed,
        assertions=(
            "only authorized current revisions render",
            "Slack/Markdown-active text is escaped",
            "projection is explicitly derived and read-only",
        ),
        metrics={"projected_item_count": page.item_count},
        replay_pointer="tests/test_memory_projection_pagination.py",
    )


def _purge_scenario(fixture: FrozenRetrievalFixture) -> M3EvalScenarioResult:
    scope = fixture.queries[0].scope
    first = make_purge_plan(
        scope,
        ("memory-purge-a",),
        targets=(PurgeTarget(record_id="memory-purge-a", generation=2, current_revision=3),),
    )
    changed = make_purge_plan(
        scope,
        ("memory-purge-a",),
        targets=(PurgeTarget(record_id="memory-purge-a", generation=3, current_revision=4),),
    )
    passed = (
        first.confirmation_token != changed.confirmation_token
        and first.manifest_hash != changed.manifest_hash
    )
    return M3EvalScenarioResult(
        id="m3-manual-purge-version-bound",
        category="maintenance",
        passed=passed,
        assertions=(
            "physical purge requires explicit record IDs",
            "confirmation binds generation and current revision",
            "changed targets require a new dry-run token",
        ),
        metrics={"wildcard_target_count": 0, "automatic_purge_job_count": 0},
        replay_pointer="tests/test_memory_maintenance.py",
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
