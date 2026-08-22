from __future__ import annotations

import socket
from pathlib import Path

import pytest

from leo.evals.baseline import (
    BaselineResult,
    paired_baseline_scenario,
    run_baseline,
)
from leo.evals.failure import (
    FailureClass,
    FailureExportAuthority,
    FailureExportNotFound,
    RegressionClosure,
    ScopedFailureBundleStore,
    classify_failure,
    export_failure_bundle,
    import_failure_bundle,
    make_bundle,
    validate_failure_bundle,
    validate_regression_closure,
)
from leo.evals.faults import (
    FaultAction,
    FaultBoundaryProbe,
    FaultController,
    FaultPlan,
    FaultPoint,
    FaultSide,
    FaultTrigger,
    InjectedFault,
    fault_controller_for_test,
)
from leo.evals.loader import load_scenarios
from leo.evals.metrics import (
    MetricThreshold,
    ThresholdOperator,
    aggregate_scenario_results,
    build_comparison_report,
    evaluate_thresholds,
    paired_delta,
    paired_result_deltas,
    percentile_metric,
    ratio_metric,
)
from leo.evals.models import ProviderMode
from leo.evals.proof import (
    REQUIRED_PROOF_SCENARIOS,
    ProofManifest,
    build_offline_proof_manifest,
    make_proof_artifact,
    validate_proof_manifest,
)
from leo.evals.recordings import (
    RecordingLane,
    RecordingMiss,
    RecordingMode,
    RecordingModelGateway,
    RecordingReplayCursor,
    RecordingSanitizationError,
    RecordingStore,
    RecordingTool,
    sanitize_payload,
)
from leo.evals.runner import run_scenario
from leo.evals.variants import (
    ConversationAuthorityFixture,
    DeliveryFixture,
    OrchestrationFixture,
    PlanNodeFixture,
    ScenarioVariant,
    VariantScenario,
    VariantSupport,
    build_variant_matrix,
    validate_variant_compatibility,
)
from leo.harness.models import (
    ContextManifest,
    ContextSegment,
    ModelRequest,
    ScopeKey,
    ToolChoiceMode,
    ToolChoicePolicy,
    ToolExecutionContext,
    TrustedScope,
)
from leo.integrations.fake import FakeQuoteTool, FixedClock, ScriptedQuoteModel

ROOT = Path("evals/scenarios")
BASELINE_SCHEMA_COUNTS = {
    "budget_boundary": 1,
    "channel_isolation": 0,
    "contextual_conversation": 0,
    "conversational_terminal_recovery": 0,
    "delegated_dependency_plan": 0,
    "dm_context_union": 0,
    "elastic_deliberation": 3,
    "fault_recovery_matrix": 0,
    "long_thread_compaction": 0,
    "memory_lifecycle": 0,
    "parallel_read_batch": 1,
    "quote_control": 1,
    "restart_replay_idempotency": 1,
    "safe_failure": 1,
    "shared_group_external_scope": 0,
    "slack_thread_context_authority": 1,
    "tavily_verified_research": 2,
    "tool_recall_progressive": 3,
    "verifier_correction": 1,
}


def test_recording_capture_is_sanitized_content_addressed_and_strict() -> None:
    store = RecordingStore()
    exchange = store.capture(
        provider="demo",
        operation="quote",
        version="v1",
        request={"symbol": "NVDA", "authorization": "Bearer " + "secret"},
        response={"price": 1.25},
        sequence=0,
    )
    replayed = store.replay(
        provider="demo",
        operation="quote",
        version="v1",
        request={"symbol": "NVDA", "authorization": "Bearer changed"},
        sequence=0,
    )
    assert replayed.digest == exchange.digest
    with pytest.raises(RecordingMiss, match="recording_miss"):
        store.replay(
            provider="demo",
            operation="quote",
            version="v1",
            request={"symbol": "MSFT"},
            sequence=0,
        )
    with pytest.raises(RecordingSanitizationError, match="recording_secret_detected"):
        sanitize_payload({"message": "xoxb-" + "123456789012345"})
    assert sanitize_payload({"prompt_tokens": 12, "max_tokens": 50}) == {
        "prompt_tokens": 12,
        "max_tokens": 50,
    }
    for private_value in (
        "person@example.com",
        "C:\\Users\\private-user\\recording.json",
        "sk-" + "private-token-value",
    ):
        with pytest.raises(RecordingSanitizationError, match="recording_secret_detected"):
            sanitize_payload({"safe_field": private_value})


def test_recording_replay_is_exact_per_parent_node_call_and_sequence() -> None:
    store = RecordingStore()
    parent = store.capture(
        provider="fixture-model",
        operation="response",
        version="v1",
        request={"prompt": "parent prompt"},
        response={"content": "parent answer"},
        sequence=0,
        lane=RecordingLane.PARENT_MODEL,
        parent_id="run-parent",
        call_id="parent-0",
    )
    child_a_0 = store.capture(
        provider="fixture-model",
        operation="response",
        version="v1",
        request={"prompt": "child A first"},
        response={"content": "A0"},
        sequence=0,
        lane=RecordingLane.CHILD_MODEL,
        parent_id="run-parent",
        node_id="node-a",
        call_id="a-0",
    )
    child_a_1 = store.capture(
        provider="fixture-model",
        operation="response",
        version="v1",
        request={"prompt": "child A second"},
        response={"content": "A1"},
        sequence=1,
        lane=RecordingLane.CHILD_MODEL,
        parent_id="run-parent",
        node_id="node-a",
        call_id="a-1",
    )
    store.capture(
        provider="fixture-model",
        operation="response",
        version="v1",
        request={"prompt": "child B first"},
        response={"content": "B0"},
        sequence=0,
        lane=RecordingLane.CHILD_MODEL,
        parent_id="run-parent",
        node_id="node-b",
        call_id="b-0",
    )

    cursor = RecordingReplayCursor(store)
    assert (
        cursor.replay_next(
            provider="fixture-model",
            operation="response",
            version="v1",
            request={"prompt": "child A first"},
            lane=RecordingLane.CHILD_MODEL,
            parent_id="run-parent",
            node_id="node-a",
            call_id="a-0",
        )
        == child_a_0
    )
    assert (
        cursor.replay_next(
            provider="fixture-model",
            operation="response",
            version="v1",
            request={"prompt": "child A second"},
            lane=RecordingLane.CHILD_MODEL,
            parent_id="run-parent",
            node_id="node-a",
            call_id="a-1",
        )
        == child_a_1
    )
    with pytest.raises(RecordingMiss, match="recording_miss"):
        store.replay(
            provider="fixture-model",
            operation="response",
            version="v1",
            request={"prompt": "parent prompt"},
            sequence=0,
            lane=RecordingLane.PARENT_MODEL,
            parent_id="run-parent",
            node_id="forged-node",
            call_id="parent-0",
        )
    with pytest.raises(RecordingMiss, match="recording_digest_mismatch"):
        store.put(parent.model_copy(update={"digest": "f" * 64}))


def test_parallel_recordings_match_stable_calls_not_completion_order() -> None:
    store = RecordingStore()
    first = store.capture(
        provider="fixture-tool",
        operation="market.get_quote",
        version="v1",
        request={"symbol": "NVDA"},
        response={"price": 181.25},
        sequence=0,
        parent_id="run-parent",
        call_id="call-a",
    )
    second = store.capture(
        provider="fixture-tool",
        operation="market.get_quote",
        version="v1",
        request={"symbol": "MSFT"},
        response={"price": 412.0},
        sequence=0,
        parent_id="run-parent",
        call_id="call-b",
    )
    cursor = RecordingReplayCursor(store)
    assert (
        cursor.replay_parallel_call(
            provider="fixture-tool",
            operation="market.get_quote",
            version="v1",
            request={"symbol": "MSFT"},
            lane=RecordingLane.TOOL,
            parent_id="run-parent",
            node_id=None,
            call_id="call-b",
        )
        == second
    )
    assert (
        cursor.replay_parallel_call(
            provider="fixture-tool",
            operation="market.get_quote",
            version="v1",
            request={"symbol": "NVDA"},
            lane=RecordingLane.TOOL,
            parent_id="run-parent",
            node_id=None,
            call_id="call-a",
        )
        == first
    )
    with pytest.raises(RecordingMiss, match="already_consumed"):
        cursor.replay_parallel_call(
            provider="fixture-tool",
            operation="market.get_quote",
            version="v1",
            request={"symbol": "NVDA"},
            lane=RecordingLane.TOOL,
            parent_id="run-parent",
            node_id=None,
            call_id="call-a",
        )


def _recording_model_request() -> ModelRequest:
    tool = FakeQuoteTool(FixedClock()).spec
    return ModelRequest(
        objective="Get the NVDA quote",
        iteration=0,
        observations=(),
        verifier_feedback=(),
        tools=(tool,),
        tool_choice=ToolChoicePolicy(mode=ToolChoiceMode.AUTO),
        manifest=ContextManifest(
            segments=(ContextSegment(name="objective", priority=100, pinned=True),)
        ),
    )


@pytest.mark.asyncio
async def test_recording_wrappers_capture_and_replay_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = RecordingStore()
    request = _recording_model_request()
    capture_model = RecordingModelGateway(
        store,
        mode=RecordingMode.CAPTURE,
        provider="fixture-model",
        version="v1",
        lane=RecordingLane.PARENT_MODEL,
        parent_id="run-parent",
        delegate=ScriptedQuoteModel(),
    )
    captured_turn = await capture_model.decide(request)
    capture_child_model = RecordingModelGateway(
        store,
        mode=RecordingMode.CAPTURE,
        provider="fixture-model",
        version="v1",
        lane=RecordingLane.CHILD_MODEL,
        parent_id="run-parent",
        node_id="node-child",
        delegate=ScriptedQuoteModel(),
    )
    captured_child_turn = await capture_child_model.decide(request)

    clock = FixedClock()
    delegate_tool = FakeQuoteTool(clock)
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org-eval", strategy_id="strategy-eval"),
            actor_id="user-eval",
        ),
        run_id="run-parent",
        tool_call_id="call-quote",
    )
    capture_tool = RecordingTool(
        store,
        mode=RecordingMode.CAPTURE,
        provider="fixture-tool",
        version="v1",
        spec=delegate_tool.spec,
        delegate=delegate_tool,
    )
    captured_outcome = await capture_tool.execute({"symbol": "NVDA"}, context)

    def deny_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("recording replay attempted network access")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    replay_model = RecordingModelGateway(
        store,
        mode=RecordingMode.REPLAY,
        provider="fixture-model",
        version="v1",
        lane=RecordingLane.PARENT_MODEL,
        parent_id="run-parent",
    )
    replay_child_model = RecordingModelGateway(
        store,
        mode=RecordingMode.REPLAY,
        provider="fixture-model",
        version="v1",
        lane=RecordingLane.CHILD_MODEL,
        parent_id="run-parent",
        node_id="node-child",
    )
    replay_tool = RecordingTool(
        store,
        mode=RecordingMode.REPLAY,
        provider="fixture-tool",
        version="v1",
        spec=delegate_tool.spec,
    )
    assert await replay_model.decide(request) == captured_turn
    assert await replay_child_model.decide(request) == captured_child_turn
    assert await replay_tool.execute({"symbol": "NVDA"}, context) == captured_outcome

    mutated = request.model_copy(update={"objective": "Get the MSFT quote"})
    drifted_replay = RecordingModelGateway(
        store,
        mode=RecordingMode.REPLAY,
        provider="fixture-model",
        version="v1",
        lane=RecordingLane.PARENT_MODEL,
        parent_id="run-parent",
    )
    with pytest.raises(RecordingMiss, match="recording_miss"):
        await drifted_replay.decide(mutated)


def test_executable_baseline_is_matched_safe_and_omits_only_frozen_features() -> None:
    scenarios = {scenario.id: scenario for scenario in load_scenarios(ROOT)}
    results: dict[str, BaselineResult] = {
        scenario_id: run_baseline(
            scenario,
            eligible_schema_count=BASELINE_SCHEMA_COUNTS[scenario_id],
        )
        for scenario_id, scenario in scenarios.items()
    }

    assert all(result.status.value == "passed" for result in results.values())
    assert all(
        "baseline_hard_safety_preserved" in result.observed_invariants
        for result in results.values()
    )
    assert results["channel_isolation"].admitted_destination == "C-BETA"
    assert results["quote_control"].budget == scenarios["quote_control"].budget
    assert results["quote_control"].matched_tool_catalog == ("market.get_quote",)
    assert results["quote_control"].metrics["task_success_count"] == 1
    assert results["dm_context_union"].metrics["context_items_seen"] == 0
    assert "no_dm_union" in results["dm_context_union"].feature_flags
    assert results["delegated_dependency_plan"].exposed_tool_catalog == ()
    assert results["delegated_dependency_plan"].metrics["plan_nodes_completed"] == 0
    assert results["verifier_correction"].metrics["correction_retry_count"] == 0
    assert results["verifier_correction"].metrics["task_success_count"] == 0
    assert results["memory_lifecycle"].metrics["memory_revisions"] == 0
    assert results["long_thread_compaction"].metrics["compaction_count"] == 0
    assert results["tool_recall_progressive"].metrics["progressive_tools_opened"] == 0
    assert results["tool_recall_progressive"].matched_tool_catalog == (
        "market.get_quote",
        "sec.get_recent_filings",
        "web.fetch_public_text",
    )
    assert results["shared_group_external_scope"].admitted_destination.endswith("-external-thread")
    assert results["budget_boundary"].budget == scenarios["budget_boundary"].budget
    assert results["fault_recovery_matrix"].metrics["fault_case_count"] == 0

    with pytest.raises(ValueError, match="eligible schema count"):
        run_baseline(scenarios["quote_control"], eligible_schema_count=0)


def test_baseline_pairing_rejects_an_unmatched_fixture() -> None:
    scenario = next(item for item in load_scenarios(ROOT) if item.id == "quote_control")
    baseline = run_baseline(scenario, eligible_schema_count=1)
    assert paired_baseline_scenario(scenario, baseline).status.value == "passed"
    forged = baseline.model_copy(update={"fixture_digest": "f" * 64})
    with pytest.raises(ValueError, match="not matched"):
        paired_baseline_scenario(scenario, forged)
    invalid_catalog = baseline.model_dump(mode="json") | {"tool_schema_count": 0}
    with pytest.raises(ValueError, match="schema count"):
        BaselineResult.model_validate(invalid_catalog)


def test_scenario_variants_remain_explicit() -> None:
    variant = VariantScenario(
        id="delivery-demo",
        version="v1",
        variant=ScenarioVariant.DELIVERY,
        fixture_digest="a" * 64,
        support_status="unsupported",
        expected_outcome="unsupported until the live delivery fixture is bound",
        delivery=DeliveryFixture(boundary="slack"),
    )
    assert not variant.supported


def test_variant_matrix_exactly_covers_executors_and_reserved_variants() -> None:
    scenarios = load_scenarios(ROOT)
    matrix = build_variant_matrix(scenarios)
    supported = tuple(item for item in matrix if item.supported)
    reserved = tuple(item for item in matrix if not item.supported)

    assert {item.id for item in supported} == {scenario.id for scenario in scenarios}
    assert {item.variant for item in matrix} == set(ScenarioVariant)
    assert all(
        item.support_status is VariantSupport.UNSUPPORTED_BY_RUNNER
        and item.expected_outcome == "unsupported_by_runner"
        and item.executor_variant is None
        for item in reserved
    )
    validate_variant_compatibility(matrix, scenarios)

    forged = matrix[0].model_copy(update={"fixture_digest": "f" * 64})
    with pytest.raises(ValueError, match="does not match"):
        validate_variant_compatibility((forged, *matrix[1:]), scenarios)


def test_variant_authority_and_plan_graph_fail_closed() -> None:
    with pytest.raises(ValueError, match="sorted and unique"):
        ConversationAuthorityFixture(
            team_id="T-EVAL",
            destination_id="D-EVAL",
            destination_kind="dm",
            actor_id="U-EVAL",
            allowed_conversation_ids=("D-EVAL", "C-ALPHA"),
        )
    with pytest.raises(ValueError, match="only the exact destination"):
        ConversationAuthorityFixture(
            team_id="T-EVAL",
            destination_id="G-EVAL",
            destination_kind="group_dm",
            allowed_conversation_ids=("C-FORGED", "G-EVAL"),
        )
    with pytest.raises(ValueError, match="acyclic"):
        OrchestrationFixture(
            parent_task_id="task-parent",
            parent_run_id="run-parent",
            plan_id="plan-1",
            nodes=(
                PlanNodeFixture(key="a", depends_on=("b",)),
                PlanNodeFixture(key="b", depends_on=("a",)),
            ),
        )
    with pytest.raises(ValueError, match="cannot carry runtime authority"):
        VariantScenario(
            id="forged-authority",
            version="v1",
            variant=ScenarioVariant.DELIVERY,
            fixture_digest="a" * 64,
            support_status=VariantSupport.UNSUPPORTED,
            expected_outcome="explicit test-only unsupported case",
            delivery=DeliveryFixture(boundary="slack"),
            payload={"trusted_scope": "forged"},
        )


def test_metrics_keep_denominators_aggregate_raw_counts_and_pair_exactly() -> None:
    unavailable = ratio_metric("success", 0, 0)
    assert unavailable.status == "not_available"
    latency = percentile_metric("latency_p95", [10, 20, 30], 95)
    assert latency.value == 30
    delta = paired_delta(
        "latency",
        scenario_id="quote-control",
        fixture_digest="a" * 64,
        baseline_fixture_digest="a" * 64,
        leo=percentile_metric("latency", [20], 50),
        baseline=percentile_metric("latency", [30], 50),
    )
    assert delta.value == -10
    with pytest.raises(ValueError, match="matched fixture"):
        paired_delta(
            "latency",
            scenario_id="quote-control",
            fixture_digest="a" * 64,
            baseline_fixture_digest="b" * 64,
            leo=percentile_metric("latency", [20], 50),
            baseline=percentile_metric("latency", [30], 50),
        )
    with pytest.raises(ValueError, match="finite"):
        percentile_metric("bad", [float("nan")], 50)

    scenarios = [
        item for item in load_scenarios(ROOT) if item.id in {"quote_control", "safe_failure"}
    ]
    leo_results = tuple(run_scenario(scenario) for scenario in scenarios)
    aggregate = aggregate_scenario_results(
        reversed(leo_results),
        label="leo-offline",
        provider_mode=ProviderMode.OFFLINE,
        config_digest="c" * 64,
    )
    assert aggregate.status_counts == {"passed": 2, "failed": 0, "unsupported": 0}
    model_calls = next(item for item in aggregate.raw_counts if item.name == "model_calls")
    assert model_calls.raw_total == 4
    assert model_calls.observed_scenarios == model_calls.eligible_scenarios == 2
    assert aggregate == aggregate_scenario_results(
        leo_results,
        label="leo-offline",
        provider_mode=ProviderMode.OFFLINE,
        config_digest="c" * 64,
    )
    wrong_provider = leo_results[0].model_copy(update={"provider_mode": ProviderMode.RECORDED})
    with pytest.raises(ValueError, match="provider mode"):
        aggregate_scenario_results(
            (wrong_provider,),
            label="leo-offline",
            provider_mode=ProviderMode.OFFLINE,
            config_digest="c" * 64,
        )
    partial_counts = dict(leo_results[1].raw_counts)
    partial_counts.pop("tool_calls")
    partial = leo_results[1].model_copy(update={"raw_counts": partial_counts})
    partial_aggregate = aggregate_scenario_results(
        (leo_results[0], partial),
        label="leo-offline",
        provider_mode=ProviderMode.OFFLINE,
        config_digest="c" * 64,
    )
    partial_tool_calls = next(
        item for item in partial_aggregate.raw_counts if item.name == "tool_calls"
    )
    assert partial_tool_calls.observed_scenarios == 1
    assert partial_tool_calls.eligible_scenarios == 2
    negative = leo_results[0].model_copy(update={"raw_counts": {"bad": -1}})
    with pytest.raises(ValueError, match="non-negative"):
        aggregate_scenario_results(
            (negative,),
            label="leo-offline",
            provider_mode=ProviderMode.OFFLINE,
            config_digest="c" * 64,
        )

    baselines = tuple(
        run_baseline(
            scenario,
            eligible_schema_count=BASELINE_SCHEMA_COUNTS[scenario.id],
        )
        for scenario in scenarios
    )
    paired = paired_result_deltas(
        leo_results,
        baselines,
        metric_names=frozenset({"model_calls", "tool_calls"}),
    )
    assert len(paired) == 4
    assert all(item.status == "available" for item in paired)
    unavailable_pair = paired_result_deltas(
        leo_results,
        baselines,
        metric_names=frozenset({"not_observed"}),
    )
    assert all(item.status == "not_available" for item in unavailable_pair)
    forged = baselines[0].model_copy(update={"fixture_digest": "f" * 64})
    with pytest.raises(ValueError, match="unmatched fixture"):
        paired_result_deltas(leo_results, (forged, baselines[1]))
    wrong_provider_baseline = baselines[0].model_copy(
        update={"provider_mode": ProviderMode.RECORDED}
    )
    with pytest.raises(ValueError, match="unmatched fixture"):
        paired_result_deltas(
            leo_results,
            (wrong_provider_baseline, baselines[1]),
        )


def test_thresholds_and_paired_comparison_are_typed_and_reproducible() -> None:
    scenarios = tuple(
        item for item in load_scenarios(ROOT) if item.id in {"quote_control", "safe_failure"}
    )
    leo_results = tuple(run_scenario(scenario) for scenario in scenarios)
    baselines = tuple(
        run_baseline(
            scenario,
            eligible_schema_count=BASELINE_SCHEMA_COUNTS[scenario.id],
        )
        for scenario in scenarios
    )
    thresholds = (
        MetricThreshold(
            id="zero-context-leakage",
            metric_name="context_leakage_count",
            operator=ThresholdOperator.MAXIMUM,
            value=0,
            source="D-054",
            required=False,
            safety_absolute=True,
        ),
        MetricThreshold(
            id="tool-calls-observed",
            metric_name="tool_calls",
            operator=ThresholdOperator.MINIMUM,
            value=1,
            source="M5-T07",
        ),
    )
    first = build_comparison_report(
        leo_results,
        baselines,
        config_digest="c" * 64,
        thresholds=thresholds,
        metric_names=frozenset({"model_calls", "tool_calls"}),
    )
    second = build_comparison_report(
        reversed(leo_results),
        reversed(baselines),
        config_digest="c" * 64,
        thresholds=reversed(thresholds),
        metric_names=frozenset({"model_calls", "tool_calls"}),
    )
    assert first == second
    assert first.passed
    assert {item.status for item in first.thresholds} == {"passed", "not_available"}
    assert first.leo.fixture_set_digest == first.baseline.fixture_set_digest
    assert len(first.paired_deltas) == 4

    blocking = evaluate_thresholds(
        first.leo,
        (
            MetricThreshold(
                id="impossible",
                metric_name="tool_calls",
                operator=ThresholdOperator.MINIMUM,
                value=99,
                source="test",
            ),
            MetricThreshold(
                id="missing-required",
                metric_name="not_observed",
                operator=ThresholdOperator.MAXIMUM,
                value=0,
                source="test",
            ),
        ),
    )
    assert [(item.status, item.blocking) for item in blocking] == [
        ("failed", True),
        ("not_available", True),
    ]
    with pytest.raises(ValueError, match="maximum of zero"):
        MetricThreshold(
            id="unsafe-average",
            metric_name="context_leakage_count",
            operator=ThresholdOperator.MAXIMUM,
            value=0.01,
            source="test",
            safety_absolute=True,
        )

    forged = baselines[0].model_copy(update={"fixture_digest": "f" * 64})
    with pytest.raises(ValueError, match="unmatched fixture"):
        build_comparison_report(
            leo_results,
            (forged, baselines[1]),
            config_digest="c" * 64,
        )


@pytest.mark.parametrize("point", tuple(FaultPoint))
def test_fault_controller_covers_every_named_boundary_with_deterministic_logs(
    point: FaultPoint,
) -> None:
    plan = FaultPlan(
        triggers=(
            FaultTrigger(
                point=point,
                call_index=2,
                side=FaultSide.AFTER,
                action=FaultAction.DISCONNECT,
                safe_code=f"{point.value}_disconnect",
            ),
        )
    )
    first = fault_controller_for_test(plan)
    second = fault_controller_for_test(plan)
    for controller in (first, second):
        assert controller.observe(point, side=FaultSide.AFTER) is None
        trigger = controller.observe(point, side=FaultSide.AFTER)
        assert trigger is not None and trigger.safe_code == f"{point.value}_disconnect"
    assert first.log == second.log
    assert first.plan.digest == second.plan.digest
    assert [entry.fired for entry in first.log] == [False, True]


def test_fault_injection_is_repeatable_but_not_production_selectable() -> None:
    plan = FaultPlan(
        triggers=(
            FaultTrigger(
                point=FaultPoint.LEASE,
                call_index=1,
                repeat_every=2,
                action=FaultAction.RETURN_FAILURE,
                safe_code="lease_failure",
            ),
        )
    )
    controller = fault_controller_for_test(plan)
    assert [controller.observe(FaultPoint.LEASE) is not None for _ in range(5)] == [
        True,
        False,
        True,
        False,
        True,
    ]
    with pytest.raises(PermissionError, match="test_fault_authority_required"):
        FaultController(plan, _authority=object())
    production_sources = tuple(
        path for path in Path("src/leo").rglob("*.py") if "evals" not in path.parts
    )
    assert all(
        "leo.evals.faults" not in path.read_text(encoding="utf-8") for path in production_sources
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("point", tuple(FaultPoint))
@pytest.mark.parametrize("side", tuple(FaultSide))
async def test_each_fault_boundary_executes_on_the_declared_side(
    point: FaultPoint,
    side: FaultSide,
) -> None:
    plan = FaultPlan(
        triggers=(
            FaultTrigger(
                point=point,
                call_index=1,
                side=side,
                action=FaultAction.RAISE,
                safe_code=f"{point.value}_{side.value}",
            ),
        )
    )
    probe = FaultBoundaryProbe(fault_controller_for_test(plan), point)
    operation_calls = 0

    async def operation() -> str:
        nonlocal operation_calls
        operation_calls += 1
        return "completed"

    with pytest.raises(InjectedFault) as raised:
        await probe.invoke(operation)
    assert raised.value.safe_code == f"{point.value}_{side.value}"
    assert operation_calls == (0 if side is FaultSide.BEFORE else 1)


@pytest.mark.parametrize(
    ("root_code", "expected"),
    [
        ("scope_policy_denied", FailureClass.POLICY),
        ("invalid_plan_decision", FailureClass.INVALID_DECISION),
        ("membership_source_set_changed", FailureClass.MEMBERSHIP_SOURCE),
        ("provider_unavailable", FailureClass.PROVIDER_TRANSIENT),
        ("provider_authentication", FailureClass.PROVIDER_PERMANENT),
        ("child_unavailable", FailureClass.CHILD_TRANSIENT),
        ("child_contract_invalid", FailureClass.CHILD_PERMANENT),
        ("orphan_child", FailureClass.ORPHAN),
        ("duplicate_delivery_work", FailureClass.DUPLICATE),
        ("plan_deadlock", FailureClass.DEADLOCK),
        ("plan_no_progress", FailureClass.NO_PROGRESS),
        ("lease_cas_conflict", FailureClass.CONCURRENCY),
        ("provider_timeout", FailureClass.BUDGET),
        ("synthesis_invalid", FailureClass.SYNTHESIS),
        ("unknown_effect", FailureClass.UNKNOWN_EFFECT),
        ("slack_delivery_failed", FailureClass.DELIVERY),
        ("malformed_data", FailureClass.INVARIANT),
    ],
)
def test_failure_taxonomy_is_total(root_code: str, expected: FailureClass) -> None:
    failure = classify_failure(
        "run-1",
        root_code,
        reproduction_command="leo eval --id quote_control",
    )
    assert failure.failure_class is expected


def test_failure_bundle_sanitization_and_regression_closure_are_enforced() -> None:
    failure = classify_failure(
        "run-1",
        "provider_timeout",
        reproduction_command="leo eval --id quote_control",
        event_ids=("event-1",),
        recording_ids=("recording-1",),
    )
    bundle = make_bundle(
        failure,
        fixture_id="fixture-1",
        sanitized_config={
            "mode": "recorded",
            "authorization": "Bearer secret-value",
            "prompt": "private synthetic prompt",
        },
        events=({"id": "event-1", "body": "private event content"},),
    )
    validate_failure_bundle(bundle)
    assert bundle.sanitized_config["authorization"] == "[REDACTED]"
    assert isinstance(bundle.sanitized_config["prompt"], dict)
    assert isinstance(bundle.sanitized_events[0]["body"], dict)
    incomplete = RegressionClosure(
        failure_digest=bundle.digest,
        fixture_id=bundle.fixture_id,
        focused_tests_passed=True,
        aggregate_gate_passed=True,
    )
    with pytest.raises(ValueError, match="focused and aggregate"):
        validate_regression_closure(bundle, incomplete)
    closed = incomplete.model_copy(
        update={
            "focused_evidence_ids": ("test_eval_failure",),
            "aggregate_evidence_id": "quality-gate-1",
        }
    )
    validate_regression_closure(bundle, closed)
    assert closed.closed
    with pytest.raises(ValueError, match="unsafe shell"):
        classify_failure(
            "run-1",
            "provider_timeout",
            reproduction_command="leo eval --id quote; print-secret",
        )


def test_failure_export_is_exact_scope_deterministic_and_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = classify_failure(
        "run-export",
        "provider_timeout",
        reproduction_command="leo eval --id quote_control",
        event_ids=("event-1",),
    )
    bundle = make_bundle(
        failure,
        fixture_id="quote_control",
        sanitized_config={"authorization": "Bearer private-value", "mode": "recorded"},
        events=({"id": "event-1", "body": "private event content"},),
    )
    store = ScopedFailureBundleStore()
    store.put(organization_id="org-a", bundle=bundle)
    authority = FailureExportAuthority(
        organization_id="org-a",
        actor_id="operator-1",
        allowed_run_ids=("run-export",),
    )
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first = export_failure_bundle(
        store,
        authority=authority,
        run_id="run-export",
        destination=first_path,
    )
    second = export_failure_bundle(
        store,
        authority=authority,
        run_id="run-export",
        destination=second_path,
    )
    assert first.export_digest == second.export_digest
    assert first_path.read_bytes() == second_path.read_bytes()
    assert import_failure_bundle(first_path) == bundle

    wrong_org = authority.model_copy(update={"organization_id": "org-b"})
    wrong_run = authority.model_copy(update={"allowed_run_ids": ("run-other",)})
    for forged_authority in (wrong_org, wrong_run):
        with pytest.raises(FailureExportNotFound, match="failure_bundle_not_found"):
            export_failure_bundle(
                store,
                authority=forged_authority,
                run_id="run-export",
                destination=tmp_path / "forged.json",
            )

    interrupted_path = tmp_path / "interrupted.json"

    def interrupt_replace(source: str, destination: Path) -> None:
        del source, destination
        raise OSError("simulated atomic replacement interruption")

    monkeypatch.setattr("leo.evals.failure.os.replace", interrupt_replace)
    with pytest.raises(OSError, match="simulated"):
        export_failure_bundle(
            store,
            authority=authority,
            run_id="run-export",
            destination=interrupted_path,
        )
    assert not interrupted_path.exists()
    assert not tuple(tmp_path.glob(".interrupted.json.*.tmp"))


def _proof_manifest() -> ProofManifest:
    scenarios = {scenario.id: scenario for scenario in load_scenarios(ROOT)}
    artifacts = tuple(
        make_proof_artifact(
            artifact_id=artifact_id,
            kind="deterministic_golden",
            command=(
                "python -m leo.evals --baseline"
                if artifact_id == "paired_baseline_report"
                else "python -m leo.evals "
                + " ".join(f"--id {scenario_id}" for scenario_id in sorted(scenario_ids))
            ),
            scenario_ids=tuple(sorted(scenario_ids)),
            fixture_digests=tuple(
                scenarios[scenario_id].fixture_digest for scenario_id in sorted(scenario_ids)
            ),
        )
        for artifact_id, scenario_ids in REQUIRED_PROOF_SCENARIOS.items()
    )
    return ProofManifest(
        code_version="demo-code-v1",
        fixture_versions=tuple(
            f"{scenario.id}:{scenario.version}" for scenario in scenarios.values()
        ),
        model_catalog_version="fixture-models-v1",
        tool_catalog_version="fixture-tools-v1",
        policy_versions=("baseline-v2", "verifier-v1"),
        artifacts=artifacts,
    )


def test_proof_manifest_requires_all_resolved_goldens_and_recovery_artifacts() -> None:
    manifest = _proof_manifest()
    validate_proof_manifest(manifest)
    assert manifest.reproducible
    with pytest.raises(ValueError, match="missing required artifacts"):
        validate_proof_manifest(manifest.model_copy(update={"artifacts": manifest.artifacts[:-1]}))
    with pytest.raises(ValueError, match="P004-D03 is resolved"):
        validate_proof_manifest(manifest.model_copy(update={"final_golden_bound": False}))


def test_offline_proof_manifest_is_built_from_observed_results_and_baselines() -> None:
    scenarios = load_scenarios(ROOT)
    results = tuple(run_scenario(scenario) for scenario in scenarios)
    baselines = tuple(run_baseline(scenario) for scenario in scenarios)
    first = build_offline_proof_manifest(
        scenarios,
        results,
        baselines,
        code_version="demo-code-v1",
        model_catalog_version="fixture-models-v1",
        tool_catalog_version="fixture-tools-v1",
        policy_versions=("verifier-v1", "baseline-v2"),
    )
    second = build_offline_proof_manifest(
        tuple(reversed(scenarios)),
        tuple(reversed(results)),
        tuple(reversed(baselines)),
        code_version="demo-code-v1",
        model_catalog_version="fixture-models-v1",
        tool_catalog_version="fixture-tools-v1",
        policy_versions=("baseline-v2", "verifier-v1"),
    )
    assert first == second
    assert first.digest == second.digest
    assert first.reproducible
    paired = next(item for item in first.artifacts if item.id == "paired_baseline_report")
    assert paired.metadata["comparison_digest"]
    assert paired.command == "python -m leo.evals --baseline"

    failed = results[0].model_copy(update={"status": "failed"})
    with pytest.raises(ValueError, match="failed or unmatched"):
        build_offline_proof_manifest(
            scenarios,
            (failed, *results[1:]),
            baselines,
            code_version="demo-code-v1",
            model_catalog_version="fixture-models-v1",
            tool_catalog_version="fixture-tools-v1",
            policy_versions=("baseline-v2", "verifier-v1"),
        )
    tampered = first.artifacts[0].model_copy(update={"command": "python -m leo.evals"})
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_proof_manifest(
            first.model_copy(update={"artifacts": (tampered, *first.artifacts[1:])})
        )
