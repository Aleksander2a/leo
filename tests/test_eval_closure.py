from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from leo.evals.failure import (
    FailureExportAuthority,
    ScopedFailureBundleStore,
    classify_failure,
    make_bundle,
)
from leo.evals.faults import (
    FaultPoint,
    FaultRecoveryOutcome,
    FaultSide,
    run_fault_recovery_matrix,
)
from leo.evals.frozen_report import (
    EXTERNAL_EVIDENCE_CONTRACTS,
    FrozenAggregateReport,
    build_frozen_offline_report,
    export_frozen_offline_artifacts,
)
from leo.evals.loader import load_scenarios
from leo.evals.metrics import (
    DEFAULT_METRIC_REGISTRY,
    MetricDefinition,
    ThresholdOperator,
    evaluate_metric_registry,
    metric_registry_digest,
    validate_metric_registry,
)
from leo.evals.operator_cli import main as unbound_operator_main
from leo.evals.operator_cli import run_operator_cli
from leo.evals.proof import ProofManifest
from leo.evals.runner import run_scenarios

ROOT = Path("evals/scenarios")


@pytest.mark.asyncio
async def test_fault_recovery_matrix_executes_every_crash_side_deterministically() -> None:
    first = await run_fault_recovery_matrix()
    second = await run_fault_recovery_matrix()
    assert first == second
    assert first.case_count == 20
    assert first.before_case_count == first.before_without_operation_count == 10
    assert first.safe_recovery_count == 20
    assert first.false_success_count == first.unsafe_recovery_count == 0
    by_boundary = {(item.point, item.side): item for item in first.records}
    assert by_boundary[(FaultPoint.DATABASE, FaultSide.AFTER)].outcome is (
        FaultRecoveryOutcome.RELOAD_REQUIRED
    )
    assert by_boundary[(FaultPoint.LEASE, FaultSide.AFTER)].outcome is (
        FaultRecoveryOutcome.RECLAIM_REQUIRED
    )
    assert by_boundary[(FaultPoint.SLACK, FaultSide.AFTER)].outcome is (
        FaultRecoveryOutcome.UNKNOWN_EFFECT
    )
    assert all(
        not item.operation_applied for item in first.records if item.side is FaultSide.BEFORE
    )
    assert all(item.operation_applied for item in first.records if item.side is FaultSide.AFTER)


def test_metric_registry_is_traceable_complete_and_missingness_is_not_zero() -> None:
    results = run_scenarios(load_scenarios(ROOT))
    validate_metric_registry()
    observations = evaluate_metric_registry(results)
    assert len(observations) == len(DEFAULT_METRIC_REGISTRY) >= 35
    required = tuple(item for item in observations if item.required_for_offline_gate)
    assert all(item.status == "available" for item in required)
    assert all(item.threshold is None or item.threshold.status == "passed" for item in required)
    assert any(item.status == "not_available" for item in observations)
    assert len(metric_registry_digest()) == 64

    without_memory = tuple(item for item in results if item.scenario_id != "memory_lifecycle")
    missing = {item.metric_id: item for item in evaluate_metric_registry(without_memory)}
    assert missing["memory-revision-lifecycle"].status == "not_available"
    assert missing["memory-revision-lifecycle"].value is None
    assert missing["memory-revision-lifecycle"].threshold is not None
    assert missing["memory-revision-lifecycle"].threshold.blocking


def test_metric_registry_mutation_blocks_instead_of_weakening_expectations() -> None:
    results = run_scenarios(load_scenarios(ROOT))
    definition = next(
        item for item in DEFAULT_METRIC_REGISTRY if item.id == "memory-revision-lifecycle"
    )
    assert definition.threshold is not None
    stricter = definition.model_copy(
        update={
            "threshold": definition.threshold.model_copy(
                update={"operator": ThresholdOperator.MINIMUM, "value": 4}
            )
        }
    )
    registry: tuple[MetricDefinition, ...] = tuple(
        stricter if item.id == definition.id else item for item in DEFAULT_METRIC_REGISTRY
    )
    observed = {item.metric_id: item for item in evaluate_metric_registry(results, registry)}
    assert observed[definition.id].threshold is not None
    assert observed[definition.id].threshold.status == "failed"
    assert observed[definition.id].threshold.blocking


def _failure_store() -> tuple[ScopedFailureBundleStore, FailureExportAuthority]:
    failure = classify_failure(
        "run-eval",
        "child_timeout",
        reproduction_command="python -m leo.evals --id delegated_dependency_plan",
        boundary="child_model",
        event_ids=("event-1",),
    )
    bundle = make_bundle(
        failure,
        fixture_id="delegated_dependency_plan",
        sanitized_config={"provider": "offline", "policy_version": "v1"},
        events=({"event_id": "event-1", "status": "failed"},),
    )
    store = ScopedFailureBundleStore()
    store.put(organization_id="org-eval", bundle=bundle)
    authority = FailureExportAuthority(
        organization_id="org-eval",
        actor_id="trusted-operator",
        allowed_run_ids=("run-eval",),
    )
    return store, authority


def test_trusted_operator_cli_round_trips_and_cannot_select_authority(
    tmp_path: Path,
) -> None:
    store, authority = _failure_store()
    destination = tmp_path / "failure.json"
    output = io.StringIO()
    assert (
        run_operator_cli(
            ["export", "--run-id", "run-eval", "--output", str(destination)],
            store=store,
            authority=authority,
            stdout=output,
        )
        == 0
    )
    first_bytes = destination.read_bytes()
    assert '"status":"ok"' in output.getvalue()
    imported = io.StringIO()
    assert (
        run_operator_cli(
            ["import", "--input", str(destination)],
            store=store,
            authority=authority,
            stdout=imported,
        )
        == 0
    )
    assert '"fixture_id":"delegated_dependency_plan"' in imported.getvalue()
    run_operator_cli(
        ["export", "--run-id", "run-eval", "--output", str(destination)],
        store=store,
        authority=authority,
        stdout=io.StringIO(),
    )
    assert destination.read_bytes() == first_bytes

    forged = FailureExportAuthority(
        organization_id="other-org",
        actor_id="forged",
        allowed_run_ids=("run-eval",),
    )
    assert (
        run_operator_cli(
            [
                "export",
                "--run-id",
                "run-eval",
                "--output",
                str(tmp_path / "forged.json"),
            ],
            store=store,
            authority=forged,
            stdout=io.StringIO(),
        )
        == 1
    )
    with pytest.raises(SystemExit):
        run_operator_cli(
            [
                "export",
                "--run-id",
                "run-eval",
                "--output",
                str(destination),
                "--organization-id",
                "other-org",
            ],
            store=store,
            authority=authority,
            stdout=io.StringIO(),
        )
    assert unbound_operator_main([]) == 2


def test_frozen_report_is_reproducible_and_keeps_external_proof_pending() -> None:
    scenarios = load_scenarios(ROOT)
    first = build_frozen_offline_report(scenarios, code_version="test-revision")
    second = build_frozen_offline_report(scenarios, code_version="test-revision")
    assert first == second
    assert first.offline_passed
    assert first.external_evidence_status == "pending"
    assert len(first.scenario_ids) == 19
    assert first.proof_manifest.reproducible
    assert set(first.proof_manifest.artifacts[-1].scenario_ids) == set(first.scenario_ids)
    assert first.external_evidence == EXTERNAL_EVIDENCE_CONTRACTS
    assert {item.evidence_level for item in first.external_evidence} == {"postgres", "live"}
    postgres_contract = next(
        item for item in first.external_evidence if item.id == "postgres-current-head-contracts"
    )
    assert postgres_contract.command[-1] == "tests/postgres/test_m5_reliability.py"
    assert "tests/postgres" not in postgres_contract.command
    assert "rollback_safe" in postgres_contract.expected_artifact_fields
    with pytest.raises(ValueError, match="exact complete proof scenario cohort"):
        build_frozen_offline_report(scenarios[:1], code_version="test-revision")
    with pytest.raises(ValidationError, match="cannot be self-attested"):
        FrozenAggregateReport.model_validate(
            first.model_dump(mode="json") | {"external_evidence_status": "collected"}
        )


def test_frozen_artifact_export_is_atomic_parseable_json_with_real_newline(
    tmp_path: Path,
) -> None:
    report = build_frozen_offline_report(load_scenarios(ROOT), code_version="test-revision")
    report_path = tmp_path / "frozen-report.json"
    proof_path = tmp_path / "proof-v2.json"

    digests = export_frozen_offline_artifacts(
        report,
        report_destination=report_path,
        proof_destination=proof_path,
    )

    assert digests == (report.digest, report.proof_manifest.digest)
    for path in (report_path, proof_path):
        raw = path.read_bytes()
        assert raw.endswith(b"\n")
        assert not raw.endswith(b"\\n")
        assert isinstance(json.loads(raw), dict)
    FrozenAggregateReport.model_validate_json(report_path.read_text(encoding="utf-8"))
    ProofManifest.model_validate_json(proof_path.read_text(encoding="utf-8"))
