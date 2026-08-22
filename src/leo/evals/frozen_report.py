"""Frozen deterministic M5 aggregate and external evidence collection contracts."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from leo.evals.baseline import BaselineResult, run_baseline_async
from leo.evals.faults import run_fault_recovery_matrix
from leo.evals.metrics import (
    DEFAULT_METRIC_REGISTRY,
    EvaluationComparisonReport,
    RegisteredMetricObservation,
    build_comparison_report,
    evaluate_metric_registry,
    metric_registry_digest,
)
from leo.evals.models import Scenario, ScenarioResult, ScenarioStatus
from leo.evals.proof import (
    REQUIRED_PROOF_SCENARIOS,
    ProofManifest,
    build_offline_proof_manifest,
)
from leo.evals.runner import run_scenarios_async
from leo.harness.events import event_contract_digest
from leo.harness.models import ContractModel, NonEmptyStr


class ExternalEvidenceContract(ContractModel):
    id: NonEmptyStr
    evidence_level: Literal["postgres", "live"]
    command: tuple[NonEmptyStr, ...] = Field(min_length=1)
    required_environment: tuple[NonEmptyStr, ...] = ()
    expected_artifact_fields: tuple[NonEmptyStr, ...] = Field(min_length=1)
    purpose: NonEmptyStr
    blocks_final_milestone: bool = True

    @model_validator(mode="after")
    def command_is_direct_and_environment_is_unique(self) -> ExternalEvidenceContract:
        if any(
            token in argument
            for argument in self.command
            for token in ("\n", "\r", ";", "&&", "||", "|", ">", "<")
        ):
            raise ValueError("external evidence command must be direct and non-chained")
        if tuple(sorted(set(self.required_environment))) != self.required_environment:
            raise ValueError("external evidence environment names must be sorted and unique")
        if len(self.expected_artifact_fields) != len(set(self.expected_artifact_fields)):
            raise ValueError("external evidence artifact fields must be unique")
        return self


EXTERNAL_EVIDENCE_CONTRACTS: tuple[ExternalEvidenceContract, ...] = (
    ExternalEvidenceContract(
        id="postgres-current-head-contracts",
        evidence_level="postgres",
        command=(
            ".venv\\Scripts\\python.exe",
            "-m",
            "pytest",
            "-q",
            "tests/postgres/test_m5_reliability.py",
        ),
        required_environment=("DATABASE_URL",),
        expected_artifact_fields=(
            "alembic_head",
            "concurrent_sequence_result",
            "durable_recovery_digest",
            "durable_plan_restart_result",
            "failure_bundle_digest",
            "lease_reclaim_result",
            "rollback_safe",
            "scenario_fixture_digests",
        ),
        purpose=(
            "Run current-head event/plan parity, conflict, restart, and authorized export proofs "
            "inside one rollback-only outer transaction per case."
        ),
    ),
    ExternalEvidenceContract(
        id="slack-primary-acceptance",
        evidence_level="live",
        command=("leo", "slack-live"),
        required_environment=(
            "DATABASE_URL",
            "FINNHUB_API_KEY",
            "OPENROUTER_API_KEY",
            "SLACK_APP_TOKEN",
            "SLACK_BOT_TOKEN",
        ),
        expected_artifact_fields=(
            "case_invariant_digest",
            "channel_id",
            "conversation_kind",
            "context_access_hash",
            "current_membership_digest",
            "delegated_evidence_digest",
            "evidence_id",
            "grounded_memory_claim_digest",
            "message_ts",
            "memory_recall_observation_digest",
            "objective_digest",
            "run_id",
            "slack_response_ts",
        ),
        purpose=(
            "Ping Leo in real Slack across required conversation kinds, including distinct "
            "positive 1:1-DM membership-union recall and isolated group-DM cases, and capture "
            "sanitized IDs."
        ),
    ),
    ExternalEvidenceContract(
        id="supabase-durable-reconciliation",
        evidence_level="live",
        command=("leo", "replay", "{run_id}"),
        required_environment=("DATABASE_URL",),
        expected_artifact_fields=(
            "case_invariant_digest",
            "context_access_hash",
            "context_snapshot_digest",
            "conversation_source_set_digest",
            "current_membership_count",
            "current_membership_digest",
            "delegated_child_count",
            "delegated_evidence_digest",
            "delegated_overlap_verified",
            "delivery_state",
            "event_terminal_state",
            "grounded_claim_count",
            "grounded_memory_claim_digest",
            "memory_mutation_digest",
            "memory_mutation_record_count",
            "memory_mutation_revision_count",
            "memory_mutation_source_count",
            "memory_recall_observation_digest",
            "memory_recall_source_digest",
            "observed_evidence_count",
            "plan_revision_digest",
            "run_id",
            "run_terminal_state",
            "task_terminal_state",
        ),
        purpose=(
            "Reconcile Slack-visible output with Supabase task/run/plan/context/event/outbox "
            "truth, including the exact DM source-set/access hash and grounded channel-memory "
            "recall."
        ),
    ),
)


class FrozenAggregateReport(ContractModel):
    version: NonEmptyStr = "m5-frozen-report-v1"
    code_version: NonEmptyStr
    scenario_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    fixture_set_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_set_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_set_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    event_contract_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_registry_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    fault_matrix_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    comparison: EvaluationComparisonReport
    metrics: tuple[RegisteredMetricObservation, ...] = Field(min_length=1)
    proof_manifest: ProofManifest
    external_evidence: tuple[ExternalEvidenceContract, ...] = Field(min_length=1)
    offline_passed: bool
    external_evidence_status: Literal["pending", "collected"] = "pending"
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def state_and_digest_reconcile(self) -> FrozenAggregateReport:
        metrics_pass = all(
            not item.required_for_offline_gate
            or (
                item.status == "available"
                and (item.threshold is None or not item.threshold.blocking)
            )
            for item in self.metrics
        )
        expected_passed = (
            self.comparison.passed and metrics_pass and self.proof_manifest.reproducible
        )
        if self.offline_passed != expected_passed:
            raise ValueError("frozen report pass state does not reconcile")
        if self.external_evidence_status == "collected":
            raise ValueError("external evidence cannot be self-attested by an offline report")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("frozen report digest mismatch")
        return self


async def build_frozen_offline_report_async(
    scenarios: Sequence[Scenario],
    *,
    code_version: str,
    model_catalog_version: str = "offline-model-catalog-v1",
    tool_catalog_version: str = "offline-tool-catalog-v1",
    policy_versions: tuple[str, ...] = ("baseline-v2", "event-v2", "verifier-v1"),
) -> FrozenAggregateReport:
    """Execute, pair, threshold, and bind one deterministic all-scenario report."""

    ordered = tuple(sorted(scenarios, key=lambda item: item.id))
    if not ordered or len({item.id for item in ordered}) != len(ordered):
        raise ValueError("frozen report requires a non-empty unique scenario cohort")
    required_ids = REQUIRED_PROOF_SCENARIOS["paired_baseline_report"]
    if {item.id for item in ordered} != required_ids:
        raise ValueError("frozen report requires the exact complete proof scenario cohort")
    results = await run_scenarios_async(ordered)
    baselines = tuple([await run_baseline_async(item) for item in ordered])
    if any(item.status is not ScenarioStatus.PASSED for item in results) or any(
        item.status is not ScenarioStatus.PASSED for item in baselines
    ):
        raise ValueError("frozen report requires every executable scenario and baseline to pass")
    config_digest = _digest(
        [
            item.model_dump(
                mode="json",
                include={
                    "id",
                    "version",
                    "provider_mode",
                    "fixed_clock",
                    "budget",
                    "inputs",
                },
            )
            for item in ordered
        ]
    )
    thresholds = tuple(
        item.threshold for item in DEFAULT_METRIC_REGISTRY if item.threshold is not None
    )
    comparison = build_comparison_report(
        results,
        baselines,
        config_digest=config_digest,
        thresholds=thresholds,
    )
    metric_observations = evaluate_metric_registry(results)
    proof = build_offline_proof_manifest(
        ordered,
        results,
        baselines,
        code_version=code_version,
        model_catalog_version=model_catalog_version,
        tool_catalog_version=tool_catalog_version,
        policy_versions=policy_versions,
    )
    faults = await run_fault_recovery_matrix()
    scenario_ids = tuple(item.id for item in ordered)
    payload = {
        "version": "m5-frozen-report-v1",
        "code_version": code_version,
        "scenario_ids": list(scenario_ids),
        "fixture_set_digest": _digest(
            [(item.id, item.version, item.fixture_digest) for item in ordered]
        ),
        "result_set_digest": _result_digest(results),
        "baseline_set_digest": _baseline_digest(baselines),
        "event_contract_digest": event_contract_digest(),
        "metric_registry_digest": metric_registry_digest(),
        "fault_matrix_digest": faults.digest,
        "comparison": comparison.model_dump(mode="json"),
        "metrics": [item.model_dump(mode="json") for item in metric_observations],
        "proof_manifest": proof.model_dump(mode="json"),
        "external_evidence": [item.model_dump(mode="json") for item in EXTERNAL_EVIDENCE_CONTRACTS],
        "offline_passed": comparison.passed
        and proof.reproducible
        and all(
            not item.required_for_offline_gate
            or (
                item.status == "available"
                and (item.threshold is None or not item.threshold.blocking)
            )
            for item in metric_observations
        ),
        "external_evidence_status": "pending",
    }
    return FrozenAggregateReport.model_validate({**payload, "digest": _digest(payload)})


def build_frozen_offline_report(
    scenarios: Sequence[Scenario],
    *,
    code_version: str,
    model_catalog_version: str = "offline-model-catalog-v1",
    tool_catalog_version: str = "offline-tool-catalog-v1",
    policy_versions: tuple[str, ...] = ("baseline-v2", "event-v2", "verifier-v1"),
) -> FrozenAggregateReport:
    return asyncio.run(
        build_frozen_offline_report_async(
            scenarios,
            code_version=code_version,
            model_catalog_version=model_catalog_version,
            tool_catalog_version=tool_catalog_version,
            policy_versions=policy_versions,
        )
    )


def export_frozen_offline_artifacts(
    report: FrozenAggregateReport,
    *,
    report_destination: Path,
    proof_destination: Path,
) -> tuple[str, str]:
    """Atomically export parseable report/proof JSON with one real trailing newline."""

    if report_destination.resolve() == proof_destination.resolve():
        raise ValueError("frozen report and proof destinations must be distinct")
    _atomic_write_contract_json(report_destination, report.model_dump_json(indent=2))
    _atomic_write_contract_json(proof_destination, report.proof_manifest.model_dump_json(indent=2))
    return report.digest, report.proof_manifest.digest


def _atomic_write_contract_json(destination: Path, payload: str) -> None:
    if destination.suffix.casefold() != ".json" or destination.name in {".json", "..json"}:
        raise ValueError("frozen artifact destination must be a named JSON file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _result_digest(results: tuple[ScenarioResult, ...]) -> str:
    return _digest([item.model_dump(mode="json") for item in results])


def _baseline_digest(results: tuple[BaselineResult, ...]) -> str:
    return _digest([item.model_dump(mode="json") for item in results])


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
