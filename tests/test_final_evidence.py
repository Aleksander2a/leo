from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory
from pydantic import ValidationError

import leo.evals.final_evidence as final_evidence
from leo.evals.durable_recovery import (
    DurableRecoveryOutcome,
    make_durable_recovery_artifact,
    make_durable_recovery_case,
)
from leo.evals.failure import classify_failure, make_bundle
from leo.evals.final_evidence import (
    EvidenceState,
    FinalEvidenceAggregate,
    LiveRestartEvidence,
    PostgresReliabilityEvidence,
    build_final_evidence_aggregate,
    export_final_evidence_aggregate,
    make_live_restart_case,
    make_live_restart_evidence,
    make_postgres_reliability_evidence,
)
from leo.evals.live_proof import LIVE_PROOF_ARTIFACT_ID, LiveProofCollection
from leo.evals.models import Scenario, ScenarioResult
from leo.evals.postgres_evidence_operator import collect_postgres_reliability_evidence
from leo.evals.proof import ProofManifest
from leo.evals.runner import run_scenarios_async

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts"
OFFLINE_REPORT = ARTIFACT_ROOT / "m5-frozen-offline-report.json"
LIVE_PROOF = ARTIFACT_ROOT / "m5-live-proof-v2.json"
TOPOLOGY = ARTIFACT_ROOT / "m5-slack-topology-v1.json"


@pytest.mark.asyncio
async def test_final_aggregate_derives_partial_state_from_observed_artifacts() -> None:
    aggregate = await build_final_evidence_aggregate(
        offline_report_path=OFFLINE_REPORT,
        live_proof_path=LIVE_PROOF,
        topology_path=TOPOLOGY,
    )

    assert aggregate.offline_freeze_ready
    assert not aggregate.final_milestone_ready
    assert aggregate.offline_scenario_count == 19
    assert aggregate.live_case_count == 9
    assert aggregate.live_pending_evidence_ids == ()
    assert aggregate.pending_requirements == (
        "live:listener_restart_no_redelivery",
        "live:revised_d063_d066_acceptance",
        "postgres:current_head_reliability",
    )
    assert aggregate.component_states == {
        "frozen_offline_report": EvidenceState.COLLECTED,
        "offline_cancellation_restart": EvidenceState.COLLECTED,
        "slack_topology": EvidenceState.COLLECTED,
        "strict_live_reconciliation": EvidenceState.COLLECTED,
        "live_listener_restart": EvidenceState.PENDING,
        "postgres_current_head_reliability": EvidenceState.PENDING,
        "revised_d063_d066_live_acceptance": EvidenceState.PENDING,
    }
    assert aggregate.lifecycle.cancellation_failed_node_count == 2
    assert aggregate.lifecycle.cancellation_claims_cleared_count == 2
    assert aggregate.lifecycle.cancellation_child_identity_retained
    assert aggregate.lifecycle.restart_event_delta == 0
    assert aggregate.lifecycle.duplicate_delivery_attempt_count == 1
    assert aggregate.lifecycle.duplicate_delivery_count == 0
    assert aggregate.lifecycle.physical_delivery_count == 1
    assert aggregate.lifecycle.false_success_count == 0


@pytest.mark.asyncio
async def test_final_export_is_parseable_content_addressed_json_with_real_newline(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "nested" / "final-evidence.json"
    exported = await export_final_evidence_aggregate(
        offline_report_path=OFFLINE_REPORT,
        live_proof_path=LIVE_PROOF,
        topology_path=TOPOLOGY,
        destination=destination,
    )

    raw = destination.read_bytes()
    assert raw.endswith(b"\n")
    assert not raw.endswith(b"\\n")
    parsed = json.loads(raw)
    assert parsed["digest"] == exported.digest
    assert FinalEvidenceAggregate.model_validate(parsed) == exported


@pytest.mark.asyncio
async def test_live_proof_cannot_swap_its_frozen_offline_base(tmp_path: Path) -> None:
    live = ProofManifest.model_validate_json(LIVE_PROOF.read_bytes())
    swapped = live.model_copy(update={"code_version": "unmatched-offline-build"})
    swapped_path = tmp_path / "swapped-live-proof.json"
    swapped_path.write_text(swapped.model_dump_json(indent=2) + "\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="live proof is not derived from the frozen offline proof",
    ):
        await build_final_evidence_aggregate(
            offline_report_path=OFFLINE_REPORT,
            live_proof_path=swapped_path,
            topology_path=TOPOLOGY,
        )


@pytest.mark.asyncio
async def test_mutated_restart_observation_cannot_self_attest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def run_with_duplicate_delivery(
        scenarios: Iterable[Scenario],
    ) -> tuple[ScenarioResult, ...]:
        observed = await run_scenarios_async(scenarios)
        result = observed[0]
        metrics = {**result.metrics, "duplicate_delivery_count": 1}
        return (result.model_copy(update={"metrics": metrics}),)

    monkeypatch.setattr(
        "leo.evals.final_evidence.run_scenarios_async",
        run_with_duplicate_delivery,
    )

    with pytest.raises(ValidationError, match="does not prove safe convergence"):
        await final_evidence.build_offline_lifecycle_evidence()


@pytest.mark.asyncio
async def test_optional_pg_contract_only_closes_pg_pending_state(tmp_path: Path) -> None:
    postgres = _postgres_evidence()
    postgres_path = tmp_path / "postgres-reliability.json"
    postgres_path.write_text(postgres.model_dump_json(indent=2) + "\n", encoding="utf-8")

    aggregate = await build_final_evidence_aggregate(
        offline_report_path=OFFLINE_REPORT,
        live_proof_path=LIVE_PROOF,
        topology_path=TOPOLOGY,
        postgres_path=postgres_path,
    )

    assert aggregate.postgres == postgres
    assert aggregate.pending_requirements == (
        "live:listener_restart_no_redelivery",
        "live:revised_d063_d066_acceptance",
    )
    assert aggregate.component_states["postgres_current_head_reliability"] is (
        EvidenceState.COLLECTED
    )
    assert not aggregate.final_milestone_ready


@pytest.mark.asyncio
async def test_tampered_pg_summary_is_rejected(tmp_path: Path) -> None:
    postgres = _postgres_evidence()
    payload = postgres.model_dump(mode="json")
    payload["event_recovery"]["digest"] = "d" * 64
    postgres_path = tmp_path / "tampered-postgres.json"
    postgres_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="digest mismatch"):
        await build_final_evidence_aggregate(
            offline_report_path=OFFLINE_REPORT,
            live_proof_path=LIVE_PROOF,
            topology_path=TOPOLOGY,
            postgres_path=postgres_path,
        )


def test_postgres_contract_cannot_claim_nonpassing_states() -> None:
    valid = _postgres_evidence()
    wrong = make_durable_recovery_case(
        case_id="stale-cas",
        boundary="run_store_commit",
        outcome=DurableRecoveryOutcome.RECLAIMED,
        before={"version": 1},
        after={"version": 2},
        mutation_applied=True,
        detail_code="unsafe_stale_commit",
    )
    other_cases = tuple(item for item in valid.event_recovery.cases if item.id != "stale-cas")

    with pytest.raises(ValidationError, match="unmatched outcome"):
        make_postgres_reliability_evidence(
            alembic_head=valid.alembic_head,
            event_recovery=make_durable_recovery_artifact((wrong, *other_cases)),
            plan_recovery=valid.plan_recovery,
            failure_bundle=valid.failure_bundle,
        )


def test_postgres_operator_discovers_and_binds_full_observed_artifacts(
    tmp_path: Path,
) -> None:
    expected = _postgres_evidence()
    source = tmp_path / "pytest-artifacts"
    _write_postgres_inputs(source, expected)
    destination = tmp_path / "m5-postgres-reliability-v1.json"

    observed = collect_postgres_reliability_evidence(
        pytest_artifact_root=source,
        destination=destination,
    )

    assert observed == expected
    raw = destination.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\\n")
    assert PostgresReliabilityEvidence.model_validate_json(raw) == expected


def test_postgres_operator_rejects_missing_or_ambiguous_pytest_artifacts(
    tmp_path: Path,
) -> None:
    expected = _postgres_evidence()
    source = tmp_path / "pytest-artifacts"
    _write_postgres_inputs(source, expected)
    duplicate = source / "duplicate" / "event-recovery-artifact.json"
    duplicate.parent.mkdir()
    duplicate.write_text(
        expected.event_recovery.model_dump_json(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one observed"):
        collect_postgres_reliability_evidence(
            pytest_artifact_root=source,
            destination=tmp_path / "rejected.json",
        )


@pytest.mark.asyncio
async def test_live_restart_artifact_must_bind_every_existing_live_case(
    tmp_path: Path,
) -> None:
    restart = _live_restart_evidence()
    restart_path = tmp_path / "live-restart.json"
    restart_path.write_text(restart.model_dump_json(indent=2) + "\n", encoding="utf-8")

    aggregate = await build_final_evidence_aggregate(
        offline_report_path=OFFLINE_REPORT,
        live_proof_path=LIVE_PROOF,
        topology_path=TOPOLOGY,
        live_restart_path=restart_path,
    )

    assert aggregate.live_restart == restart
    assert aggregate.component_states["live_listener_restart"] is EvidenceState.COLLECTED
    assert aggregate.pending_requirements == (
        "live:revised_d063_d066_acceptance",
        "postgres:current_head_reliability",
    )

    payload = restart.model_dump(mode="json")
    payload["cases"] = payload["cases"][:-1]
    payload["case_count"] -= 1
    without_one = make_live_restart_evidence(
        listener_epoch_digest=payload["listener_epoch_digest"],
        listener_started_at=datetime.fromisoformat(payload["listener_started_at"]),
        collection_observed_at=datetime.fromisoformat(payload["collection_observed_at"]),
        cases=restart.cases[:-1],
    )
    missing_path = tmp_path / "missing-live-restart.json"
    missing_path.write_text(without_one.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="exact collected live cohort"):
        await build_final_evidence_aggregate(
            offline_report_path=OFFLINE_REPORT,
            live_proof_path=LIVE_PROOF,
            topology_path=TOPOLOGY,
            live_restart_path=missing_path,
        )


def _postgres_evidence() -> PostgresReliabilityEvidence:
    config = Config(REPOSITORY_ROOT / "alembic.ini")
    config.set_main_option("script_location", str(REPOSITORY_ROOT / "migrations"))
    head = ScriptDirectory.from_config(config).get_current_head()
    assert head is not None
    failure = classify_failure(
        "run-pg-evidence",
        "synthetic_timeout",
        reproduction_command="leo replay run-pg-evidence",
        event_ids=("event-pg-evidence",),
    )
    bundle = make_bundle(
        failure,
        fixture_id="durable-run-failure-v1",
        sanitized_config={"alembic_head": head, "source": "rollback-safe-postgres"},
        events=({"event_id": "event-pg-evidence", "schema_version": "v2"},),
    )
    event_recovery = make_durable_recovery_artifact(
        (
            make_durable_recovery_case(
                case_id="stale-cas",
                boundary="run_store_commit",
                outcome=DurableRecoveryOutcome.REJECTED_SAFE,
                before={"version": 1},
                after={"version": 1},
                mutation_applied=False,
                detail_code="stale_task_version",
            ),
            make_durable_recovery_case(
                case_id="duplicate-event-sequence",
                boundary="run_event_unique_sequence",
                outcome=DurableRecoveryOutcome.REJECTED_SAFE,
                before={"sequence": 2},
                after={"sequence": 2},
                mutation_applied=False,
                detail_code="uq_run_event_sequence",
            ),
            make_durable_recovery_case(
                case_id="restart-replay",
                boundary="run_store_reload",
                outcome=DurableRecoveryOutcome.RELOAD_EXACT,
                before={"terminal": "timed_out"},
                after={"terminal": "timed_out"},
                mutation_applied=False,
                detail_code="exact_snapshot_reloaded",
            ),
            make_durable_recovery_case(
                case_id="operator-export",
                boundary="failure_event_source",
                outcome=DurableRecoveryOutcome.EXPORTED,
                before={"run_id": failure.run_id},
                after={"bundle_digest": bundle.digest},
                mutation_applied=True,
                detail_code="sanitized_bundle_exported",
            ),
        )
    )
    plan_recovery = make_durable_recovery_artifact(
        (
            make_durable_recovery_case(
                case_id="running-child-reload",
                boundary="plan_store_replay",
                outcome=DurableRecoveryOutcome.RELOAD_EXACT,
                before={"child": "attached"},
                after={"child": "attached"},
                mutation_applied=False,
                detail_code="attached_child_reloaded",
            ),
            make_durable_recovery_case(
                case_id="expired-child-reclaim",
                boundary="plan_node_lease",
                outcome=DurableRecoveryOutcome.RECLAIMED,
                before={"claim": 1},
                after={"claim": 2},
                mutation_applied=True,
                detail_code="stale_running_reclaimed",
            ),
            make_durable_recovery_case(
                case_id="stale-child-fenced",
                boundary="plan_node_claim",
                outcome=DurableRecoveryOutcome.FENCED,
                before={"claim": 2},
                after={"claim": 2},
                mutation_applied=False,
                detail_code="stale_claim_rejected",
            ),
            make_durable_recovery_case(
                case_id="terminal-plan-reload",
                boundary="plan_store_reload",
                outcome=DurableRecoveryOutcome.RELOAD_EXACT,
                before={"plan": "completed"},
                after={"plan": "completed"},
                mutation_applied=False,
                detail_code="parent_terminal_reloaded",
            ),
        )
    )
    return make_postgres_reliability_evidence(
        alembic_head=head,
        event_recovery=event_recovery,
        plan_recovery=plan_recovery,
        failure_bundle=bundle,
    )


def _live_restart_evidence() -> LiveRestartEvidence:
    manifest = ProofManifest.model_validate_json(LIVE_PROOF.read_bytes())
    artifact = next(item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID)
    metadata = artifact.metadata
    collection = LiveProofCollection.model_validate(
        {
            "version": metadata["version"],
            "profile": metadata["profile"],
            "authority_digest": metadata["authority_digest"],
            "cases": metadata["case_summaries"],
            "pending_evidence_ids": metadata["pending_evidence_ids"],
            "status": metadata["status"],
            "digest": metadata["collection_digest"],
        }
    )
    latest_receipt = max(float(item.slack_response_ts) for item in collection.cases)
    listener_started_at = datetime.fromtimestamp(latest_receipt, tz=UTC) + timedelta(hours=1)
    cases = tuple(
        make_live_restart_case(
            evidence_id=item.evidence_id,
            run_id=item.run_id,
            slack_response_ts=item.slack_response_ts,
            final_outbox_row_digest=item.outbox_digest,
            final_delivered_at=datetime.fromtimestamp(float(item.slack_response_ts), tz=UTC),
            post_restart_slack_readback_digest="f" * 64,
        )
        for item in collection.cases
    )
    return make_live_restart_evidence(
        listener_epoch_digest="e" * 64,
        listener_started_at=listener_started_at,
        collection_observed_at=listener_started_at + timedelta(minutes=5),
        cases=cases,
    )


def _write_postgres_inputs(
    root: Path,
    evidence: PostgresReliabilityEvidence,
) -> None:
    event_directory = root / "event-probe"
    plan_directory = root / "plan-probe"
    event_directory.mkdir(parents=True)
    plan_directory.mkdir(parents=True)
    (event_directory / "event-recovery-artifact.json").write_text(
        evidence.event_recovery.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (plan_directory / "plan-recovery-artifact.json").write_text(
        evidence.plan_recovery.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (event_directory / "durable-failure.json").write_text(
        json.dumps(
            {
                "version": "failure-export-v1",
                "bundle": evidence.failure_bundle.model_dump(mode="json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
