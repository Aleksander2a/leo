"""Executable offline scenarios for the revised M5 authority and recovery matrix."""

from __future__ import annotations

from datetime import datetime, timedelta

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import InMemoryToolCatalog
from leo.capabilities.runtime import CapabilityDiscoveryError, CapabilityRuntime
from leo.demo import run_quote_smoke
from leo.domain.conversation import (
    ConversationKind,
    ConversationPolicyError,
    ConversationRef,
    VisibilityNamespace,
    derive_visibility,
)
from leo.evals.control import BaselineExecution
from leo.evals.faults import run_fault_recovery_matrix
from leo.evals.models import Scenario
from leo.harness.models import (
    BudgetLimits,
    EventType,
    OriginRef,
    Run,
    RunBundle,
    RunPhase,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
    ToolEffect,
    ToolSpec,
    TrustedScope,
)
from leo.harness.store_errors import NotFoundError
from leo.integrations.fake import EndlessQuoteModel, FixedClock, SequentialIdGenerator
from leo.memory.compaction import (
    CompactionPolicy,
    SummaryProposal,
    compaction_result,
    make_summary,
    select_compaction_window,
)
from leo.memory.models import (
    MemoryKind,
    MemorySource,
    MemoryStatus,
    MemoryVisibility,
)
from leo.memory.planes import SanitizedMessage
from leo.memory.service import ExplicitMemoryService, MemoryCandidate
from leo.memory.store import InMemoryMemoryStore


class ClosureScenarioUnsupported(RuntimeError):
    pass


class ClosureObserved:
    def __init__(
        self,
        invariants: frozenset[str],
        metrics: dict[str, float | int | str],
        hard_failures: tuple[str, ...] = (),
    ) -> None:
        self.invariants = invariants
        self.metrics = metrics
        self.hard_failures = hard_failures


CLOSURE_VARIANTS = frozenset(
    {
        "memory_lifecycle",
        "long_thread_compaction",
        "tool_recall_progressive",
        "shared_group_external_scope",
        "budget_boundary",
        "fault_recovery_matrix",
    }
)


async def execute_closure_scenario(scenario: Scenario) -> ClosureObserved:
    executors = {
        "memory_lifecycle": _execute_memory_lifecycle,
        "long_thread_compaction": _execute_long_thread_compaction,
        "tool_recall_progressive": _execute_tool_recall,
        "shared_group_external_scope": _execute_conversation_kind_matrix,
        "budget_boundary": _execute_budget_boundary,
        "fault_recovery_matrix": _execute_fault_recovery,
    }
    try:
        executor = executors[scenario.execution_variant]
    except KeyError as exc:
        raise ClosureScenarioUnsupported(
            f"execution_variant_not_supported:{scenario.execution_variant}"
        ) from exc
    return await executor(scenario)


async def execute_closure_baseline_scenario(scenario: Scenario) -> BaselineExecution:
    variant = scenario.execution_variant
    catalogs: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
        "memory_lifecycle": ((), ()),
        "long_thread_compaction": ((), ()),
        "tool_recall_progressive": (
            ("market.get_quote", "sec.get_recent_filings", "web.fetch_public_text"),
            ("market.get_quote", "sec.get_recent_filings", "web.fetch_public_text"),
        ),
        "shared_group_external_scope": ((), ()),
        "budget_boundary": (("market.get_quote",), ("market.get_quote",)),
        "fault_recovery_matrix": ((), ()),
    }
    matched, exposed = catalogs[variant]
    if variant == "memory_lifecycle":
        store = InMemoryMemoryStore()
        empty_observed = False
        try:
            await store.current(
                ScopeKey(organization_id="eval-org", strategy_id="memory-scope"),
                "memory-never-created",
            )
        except NotFoundError:
            empty_observed = True
        if not empty_observed:
            raise ClosureScenarioUnsupported("baseline_memory_omission_failed")
        metrics: dict[str, float | int | str] = {
            "memory_revisions": 0,
            "memory_current_count": 0,
            "memory_source_count_before_forget": 0,
            "memory_cross_scope_leakage_count": 0,
            "memory_conflict_count": 0,
        }
    elif variant == "long_thread_compaction":
        now = _parse_clock(scenario.fixed_clock)
        scope = ScopeKey(organization_id="eval-org", strategy_id="long-thread-scope")
        recent = tuple(
            SanitizedMessage.from_text(
                id=f"baseline-message-{index}",
                scope=scope,
                destination_id="C-LONG",
                conversation_id="C-LONG",
                harness_thread_id="thread-long",
                external_event_id=f"baseline-event-{index}",
                text=f"Synthetic recent baseline turn {index}.",
                recorded_at=now + timedelta(seconds=index),
            )
            for index in range(4)
        )
        window = select_compaction_window(
            recent,
            CompactionPolicy(trigger_messages=50, recent_window_messages=12),
        )
        if window.should_compact:
            raise ClosureScenarioUnsupported("baseline_compaction_was_not_omitted")
        input_tokens = window.input_estimated_tokens
        metrics = {
            "compaction_count": 0,
            "long_thread_message_count": len(recent),
            "compacted_message_count": 0,
            "recent_message_count": len(recent),
            "compaction_input_tokens": input_tokens,
            "compaction_retained_tokens": input_tokens,
            "compaction_token_reduction_ratio": 0.0,
        }
    elif variant == "tool_recall_progressive":
        all_tools = _tool_specs()
        metrics = {
            "tool_recall_at_k": float(
                any(item.name == "sec.get_recent_filings" for item in all_tools)
            ),
            "tool_recall_candidate_count": len(all_tools),
            "tool_recall_selected_count": len(all_tools),
            "progressive_tools_opened": 0,
            "tool_recall_authority_leakage_count": 0,
            "no_progress_escape_count": 0,
        }
    elif variant in {"shared_group_external_scope", "budget_boundary"}:
        metrics = dict((await execute_closure_scenario(scenario)).metrics)
    elif variant == "fault_recovery_matrix":
        operations_applied = 0
        for _ in range(20):
            operations_applied += 1
        if operations_applied != 20:
            raise ClosureScenarioUnsupported("baseline_fault_free_operations_failed")
        metrics = {
            "fault_case_count": 0,
            "fault_triggered_count": 0,
            "fault_recovered_count": 0,
            "fault_false_success_count": 0,
            "fault_unsafe_recovery_count": 0,
            "fault_unknown_effect_count": 0,
        }
    else:
        raise ClosureScenarioUnsupported(f"execution_variant_not_supported:{variant}")
    return BaselineExecution(
        invariants=frozenset({"baseline_hard_safety_preserved"}),
        metrics=metrics,
        hard_failures=(),
        eligible_schema_count=len(exposed),
        admitted_destination=f"{scenario.deterministic_id_prefix}-external-thread",
        model_fixture=f"closure-baseline:{variant}",
        matched_tool_catalog=matched,
        exposed_tool_catalog=exposed,
    )


async def _execute_memory_lifecycle(scenario: Scenario) -> ClosureObserved:
    now = _parse_clock(scenario.fixed_clock)
    scope = ScopeKey(organization_id="eval-org", strategy_id="memory-scope")
    store = InMemoryMemoryStore()
    service = ExplicitMemoryService(
        store,
        FixedClock(now),
        SequentialIdGenerator(),
    )
    source_one = _memory_source("memory-source-1", scope)
    remembered = await service.remember(
        scope,
        _memory_candidate(
            content="The synthetic team selected the blue launch plan.",
            source_ids=(source_one.id,),
            now=now,
        ),
        actor_id="eval-user",
        sources=(source_one,),
        confirmed=True,
    )
    first = await store.current(scope, remembered.id)
    source_two = _memory_source("memory-source-2", scope)
    corrected = await service.correct(
        scope,
        remembered.id,
        _memory_candidate(
            content="The synthetic team corrected the choice to the green launch plan.",
            source_ids=(source_two.id,),
            now=now,
        ),
        actor_id="eval-user",
        sources=(source_two,),
        confirmed=True,
    )
    second = await store.current(scope, remembered.id)
    source_three = _memory_source("memory-source-3", scope)
    forgotten = await service.forget(
        scope,
        remembered.id,
        actor_id="eval-user",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-MEMORY",
        sources=(source_three,),
        confirmed=True,
        reason="The user explicitly requested forgetting this synthetic memory.",
    )
    current = await store.current(scope, remembered.id)
    cross_scope_denied = False
    try:
        await store.current(
            ScopeKey(organization_id="eval-org", strategy_id="other-scope"),
            remembered.id,
        )
    except NotFoundError:
        cross_scope_denied = True

    invariants: set[str] = set()
    if (
        first is not None
        and second is not None
        and remembered.current_revision == 1
        and corrected.current_revision == 2
        and forgotten.current_revision == 3
    ):
        invariants.add("memory_lifecycle_append_only")
    if second is not None and second.source_ids == (source_one.id, source_two.id):
        invariants.add("memory_provenance_preserved")
    if forgotten.status is MemoryStatus.RETRACTED and current is None:
        invariants.add("memory_forget_not_retrievable")
    if cross_scope_denied:
        invariants.add("memory_scope_isolated")
    metrics: dict[str, float | int | str] = {
        "memory_revisions": forgotten.current_revision,
        "memory_current_count": int(current is not None),
        "memory_source_count_before_forget": len(second.source_ids) if second else 0,
        "memory_cross_scope_leakage_count": int(not cross_scope_denied),
        "memory_conflict_count": 0,
    }
    return ClosureObserved(frozenset(invariants), metrics)


async def _execute_long_thread_compaction(scenario: Scenario) -> ClosureObserved:
    now = _parse_clock(scenario.fixed_clock)
    scope = ScopeKey(organization_id="eval-org", strategy_id="long-thread-scope")
    messages = tuple(
        SanitizedMessage.from_text(
            id=f"message-{index:03d}",
            scope=scope,
            destination_id="C-LONG",
            conversation_id="C-LONG",
            harness_thread_id="thread-long",
            external_event_id=f"event-{index:03d}",
            text=(
                f"Synthetic turn {index}: the team discussed evidence, constraints, and the "
                "green launch decision in enough detail to exercise deterministic compaction."
            ),
            recorded_at=now + timedelta(seconds=index),
        )
        for index in range(60)
    )
    policy = CompactionPolicy(trigger_messages=50, recent_window_messages=12)
    window = select_compaction_window(messages, policy)
    proposal = SummaryProposal(
        objective="Preserve the synthetic green launch decision and open constraint.",
        decisions=("The team selected the green launch plan.",),
        unresolved_questions=("Confirm the synthetic launch date.",),
        covered_message_ids=window.compactable_message_ids,
    )
    summary = make_summary(
        "thread-long",
        scope,
        1,
        proposal,
        available_source_ids=frozenset(message.id for message in messages),
    )
    compacted = compaction_result(summary, messages, window)
    incomplete_summary_rejected = False
    incomplete = make_summary(
        "thread-long",
        scope,
        1,
        proposal.model_copy(update={"covered_message_ids": window.compactable_message_ids[:-1]}),
        available_source_ids=frozenset(message.id for message in messages),
    )
    try:
        compaction_result(incomplete, messages, window)
    except ValueError:
        incomplete_summary_rejected = True

    invariants: set[str] = set()
    if window.should_compact and len(window.compactable_message_ids) == 48:
        invariants.add("long_thread_compaction_triggered")
    if set(window.compactable_message_ids).issubset(summary.proposal.covered_message_ids):
        invariants.add("compaction_sources_complete")
    if compacted.recent_message_ids == tuple(message.id for message in messages[-12:]):
        invariants.add("compaction_recent_window_preserved")
    if compacted.retained_estimated_tokens < compacted.input_estimated_tokens:
        invariants.add("compaction_reduces_context")
    if incomplete_summary_rejected:
        invariants.add("compaction_incomplete_summary_rejected")
    metrics: dict[str, float | int | str] = {
        "compaction_count": 1,
        "long_thread_message_count": len(messages),
        "compacted_message_count": len(window.compactable_message_ids),
        "recent_message_count": len(window.recent_message_ids),
        "compaction_input_tokens": compacted.input_estimated_tokens,
        "compaction_retained_tokens": compacted.retained_estimated_tokens,
        "compaction_token_reduction_ratio": compacted.token_reduction_ratio,
    }
    return ClosureObserved(frozenset(invariants), metrics)


async def _execute_tool_recall(scenario: Scenario) -> ClosureObserved:
    now = _parse_clock(scenario.fixed_clock)
    scope = ScopeKey(organization_id="eval-org", strategy_id="tool-recall-scope")
    specs = _tool_specs()
    catalog = InMemoryToolCatalog(version="catalog-eval-v1")
    for spec in specs:
        catalog.register(
            catalog_tool_from_spec(
                spec,
                provider="fixture",
                tags=frozenset(spec.name.replace(".", " ").split()),
            )
        )
    runtime = CapabilityRuntime(
        catalog,
        shortlist_limit=1,
        max_selected_tools=3,
        max_search_calls=2,
        max_describe_calls=2,
    )
    thread = Thread(
        id="tool-recall-thread",
        scope=scope,
        origin=OriginRef(provider="fixture", external_thread_id="C-TOOLS:1"),
    )
    task = Task(
        id="tool-recall-task",
        thread_id=thread.id,
        scope=scope,
        objective="Review the latest SEC filings for NVDA.",
        status=TaskStatus.ACTIVE,
    )
    run = Run(
        id="tool-recall-run",
        task_id=task.id,
        scope=scope,
        status=RunStatus.RUNNING,
        started_at=now,
        limits=BudgetLimits(max_model_calls=4, max_tool_calls=4),
    )
    bundle = RunBundle(thread=thread, task=task, run=run)
    trusted = TrustedScope(
        namespace=scope,
        actor_id="eval-user",
        roles=frozenset({"researcher"}),
    )
    initial = runtime.select(
        bundle=bundle,
        trusted_scope=trusted,
        available_tools=specs,
        conversation_kind="channel",
    )
    discovered = await runtime.search(
        run_id=run.id,
        trusted_scope=trusted,
        query="current market quote",
        limit=1,
    )
    described = runtime.describe(
        run_id=run.id,
        trusted_scope=trusted,
        capability_ids=tuple(item.id for item in discovered),
    )
    expanded = runtime.select(
        bundle=bundle,
        trusted_scope=trusted,
        available_tools=specs,
        conversation_kind="channel",
    )
    no_progress_guarded = False
    try:
        await runtime.search(
            run_id=run.id,
            trusted_scope=trusted,
            query="current market quote",
            limit=1,
        )
    except CapabilityDiscoveryError as exc:
        no_progress_guarded = exc.safe_code == "discovery_no_progress"
    authority_guarded = False
    try:
        await runtime.search(
            run_id=run.id,
            trusted_scope=trusted.model_copy(update={"actor_id": "forged-user"}),
            query="public website",
            limit=1,
        )
    except CapabilityDiscoveryError as exc:
        authority_guarded = exc.safe_code == "discovery_authority_mismatch"

    invariants: set[str] = set()
    if initial.selected_ids == ("sec.get_recent_filings",):
        invariants.add("relevant_tool_recalled")
    if (
        tuple(item.id for item in described) == ("market.get_quote",)
        and "market.get_quote" in expanded.selected_ids
    ):
        invariants.add("progressive_tool_opened")
    if no_progress_guarded:
        invariants.add("tool_recall_no_progress_guard")
    if authority_guarded:
        invariants.add("tool_recall_authority_pinned")
    metrics: dict[str, float | int | str] = {
        "tool_recall_at_k": float("sec.get_recent_filings" in initial.selected_ids),
        "tool_recall_candidate_count": len(initial.candidate_ids),
        "tool_recall_selected_count": len(initial.selected_ids),
        "progressive_tools_opened": len(described),
        "tool_recall_authority_leakage_count": int(not authority_guarded),
        "no_progress_escape_count": int(no_progress_guarded),
    }
    return ClosureObserved(frozenset(invariants), metrics)


async def _execute_conversation_kind_matrix(scenario: Scenario) -> ClosureObserved:
    del scenario
    decisions = []
    forged_rejections = 0
    for kind, destination_id in (
        (ConversationKind.GROUP_DM, "G-EVAL"),
        (ConversationKind.SHARED, "S-EVAL"),
        (ConversationKind.EXTERNAL, "E-EVAL"),
    ):
        destination = ConversationRef(
            provider="slack",
            team_id="T-EVAL",
            external_id=destination_id,
            kind=kind,
        )
        thread_id = f"slack:T-EVAL:{destination_id}:1.0"
        decisions.append(
            derive_visibility(
                destination,
                actor_id="U-EVAL",
                current_thread_namespace_id=thread_id,
            )
        )
        try:
            derive_visibility(
                destination,
                actor_id="U-EVAL",
                current_thread_namespace_id=thread_id,
                allowed_conversation_ids=tuple(sorted(("C-FORGED", destination_id))),
            )
        except ConversationPolicyError:
            forged_rejections += 1
    exact = all(
        decision.allowed_conversation_ids == (decision.destination.external_id,)
        and decision.allowed_namespaces
        == frozenset(
            {
                VisibilityNamespace.THREAD_LOCAL,
                VisibilityNamespace.CONVERSATION_LOCAL,
            }
        )
        for decision in decisions
    )
    invariants: set[str] = set()
    if exact:
        invariants.add("shared_group_external_exact_projection")
    if forged_rejections == 3:
        invariants.add("shared_group_external_aggregation_rejected")
    if {decision.destination.kind for decision in decisions} == {
        ConversationKind.GROUP_DM,
        ConversationKind.SHARED,
        ConversationKind.EXTERNAL,
    }:
        invariants.add("shared_group_external_are_eligible")
    metrics: dict[str, float | int | str] = {
        "conversation_kinds_evaluated": len(decisions),
        "context_leakage_count": sum(
            len(decision.allowed_conversation_ids) - 1 for decision in decisions
        ),
        "forged_projection_rejection_count": forged_rejections,
        "group_dm_aggregation_count": 0,
    }
    return ClosureObserved(frozenset(invariants), metrics)


async def _execute_budget_boundary(scenario: Scenario) -> ClosureObserved:
    result = await run_quote_smoke(
        model=EndlessQuoteModel(),
        limits=BudgetLimits(
            max_iterations=scenario.budget.max_model_calls,
            max_model_calls=scenario.budget.max_model_calls,
            max_tool_calls=scenario.budget.max_tool_calls,
            max_elapsed_seconds=scenario.budget.max_elapsed_seconds,
        ),
    )
    event_types = tuple(event.type for event in result.events)
    invariants: set[str] = set()
    if (
        result.run.status is RunStatus.BUDGET_EXHAUSTED
        and result.run.terminal_reason == "tool_call_budget_exhausted"
    ):
        invariants.add("budget_n_plus_one_blocked")
    if (
        result.run.usage.tool_calls == scenario.budget.max_tool_calls
        and result.run.usage.model_calls <= scenario.budget.max_model_calls
    ):
        invariants.add("budget_usage_exact")
    if (
        result.run.final_output is None
        and result.task.final_output is None
        and EventType.RUN_COMPLETED not in event_types
    ):
        invariants.add("budget_exhaustion_no_false_success")
    metrics: dict[str, float | int | str] = {
        "model_calls": result.run.usage.model_calls,
        "tool_calls": result.run.usage.tool_calls,
        "budget_overrun_count": max(
            0,
            result.run.usage.tool_calls - scenario.budget.max_tool_calls,
        ),
        "false_success_count": int(EventType.RUN_COMPLETED in event_types),
        "terminal_reason": result.run.terminal_reason or "missing",
        "terminal_reason_count": int(result.run.terminal_reason is not None),
    }
    return ClosureObserved(frozenset(invariants), metrics)


async def _execute_fault_recovery(scenario: Scenario) -> ClosureObserved:
    del scenario
    matrix = await run_fault_recovery_matrix()
    invariants: set[str] = set()
    if matrix.triggered_count == matrix.case_count:
        invariants.add("fault_matrix_all_boundaries_triggered")
    if matrix.false_success_count == 0:
        invariants.add("fault_matrix_no_false_success")
    if matrix.unsafe_recovery_count == 0:
        invariants.add("fault_matrix_safe_recovery")
    if matrix.before_without_operation_count == matrix.before_case_count:
        invariants.add("fault_matrix_crash_sides_distinguished")
    metrics: dict[str, float | int | str] = {
        "fault_case_count": matrix.case_count,
        "fault_triggered_count": matrix.triggered_count,
        "fault_recovered_count": matrix.safe_recovery_count,
        "fault_false_success_count": matrix.false_success_count,
        "fault_unsafe_recovery_count": matrix.unsafe_recovery_count,
        "fault_unknown_effect_count": matrix.unknown_effect_count,
    }
    return ClosureObserved(frozenset(invariants), metrics)


def _parse_clock(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ClosureScenarioUnsupported("fixed_clock_requires_timezone")
    return parsed


def _memory_source(source_id: str, scope: ScopeKey) -> MemorySource:
    return MemorySource(
        id=source_id,
        scope=scope,
        source_kind="conversation_message",
        reference=f"fixture:{source_id}",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-MEMORY",
    )


def _memory_candidate(
    *,
    content: str,
    source_ids: tuple[str, ...],
    now: datetime,
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.NOTE,
        content=content,
        source_ids=source_ids,
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-MEMORY",
        sensitivity=0.1,
        valid_from=now,
        reason="Explicit synthetic eval memory command.",
    )


def _tool_specs() -> tuple[ToolSpec, ...]:
    return (
        ToolSpec(
            name="market.get_quote",
            version="1",
            description="Return the current market quote for a public symbol.",
            domain="MARKET",
            input_schema={"type": "object"},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        ),
        ToolSpec(
            name="sec.get_recent_filings",
            version="1",
            description="Review the latest SEC filings and disclosures for a company.",
            domain="SEC",
            input_schema={"type": "object"},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        ),
        ToolSpec(
            name="web.fetch_public_text",
            version="1",
            description="Fetch public text from an allowed website URL.",
            domain="WEB",
            input_schema={"type": "object"},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        ),
    )
