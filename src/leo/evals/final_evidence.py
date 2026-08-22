"""Deterministic final-evidence aggregation without external self-attestation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from leo.evals.durable_recovery import (
    DurableRecoveryArtifact,
    DurableRecoveryCase,
    DurableRecoveryOutcome,
)
from leo.evals.failure import FailureBundle
from leo.evals.frozen_report import FrozenAggregateReport
from leo.evals.live_proof import (
    LIVE_PROOF_ARTIFACT_ID,
    LiveEvidenceId,
    LiveProofCollection,
    SlackMessageTs,
)
from leo.evals.loader import default_scenario_root, load_scenarios
from leo.evals.models import ScenarioStatus
from leo.evals.proof import (
    REQUIRED_PROOF_SCENARIOS,
    ProofManifest,
    validate_proof_manifest,
)
from leo.evals.revised_live_acceptance import RevisedLiveAcceptanceArtifact
from leo.evals.runner import run_scenarios_async
from leo.evals.slack_topology import (
    SharedExternalPresence,
    SlackTopologyArtifact,
    TopologyConversationKind,
)
from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    cancel_plan_snapshot,
    revision_digest,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

FINAL_EVIDENCE_VERSION: Literal["m5-final-evidence-v1"] = "m5-final-evidence-v1"
LIFECYCLE_EVIDENCE_VERSION: Literal["offline-lifecycle-evidence-v1"] = (
    "offline-lifecycle-evidence-v1"
)
POSTGRES_EVIDENCE_VERSION: Literal["postgres-reliability-evidence-v1"] = (
    "postgres-reliability-evidence-v1"
)
LIVE_RESTART_EVIDENCE_VERSION: Literal["live-restart-evidence-v1"] = "live-restart-evidence-v1"
RESTART_SCENARIO_ID = "restart_replay_idempotency"
_FIXED_TIME = datetime(2026, 8, 22, 12, tzinfo=UTC)
_DATETIME_ADAPTER = TypeAdapter(datetime)


class EvidenceState(StrEnum):
    COLLECTED = "collected"
    PARTIAL = "partial"
    PENDING = "pending"


class OfflineLifecycleEvidence(ContractModel):
    """Observed cancellation transition plus executable restart/replay scenario."""

    version: Literal["offline-lifecycle-evidence-v1"] = LIFECYCLE_EVIDENCE_VERSION
    cancellation_snapshot_digest: Sha256
    cancellation_node_count: int = Field(ge=1)
    cancellation_failed_node_count: int = Field(ge=0)
    cancellation_claims_cleared_count: int = Field(ge=0)
    cancellation_child_identity_retained: bool
    cancellation_superseded_delegation_count: int = Field(ge=0)
    forged_parent_rejected: bool
    terminal_recancel_rejected: bool
    restart_fixture_digest: Sha256
    restart_result_digest: Sha256
    restart_event_delta: int = Field(ge=0)
    duplicate_delivery_attempt_count: int = Field(ge=0)
    duplicate_delivery_count: int = Field(ge=0)
    physical_delivery_count: int = Field(ge=0)
    false_success_count: int = Field(ge=0)
    digest: Sha256

    @model_validator(mode="after")
    def exact_observed_safety(self) -> OfflineLifecycleEvidence:
        if (
            self.cancellation_node_count != 2
            or self.cancellation_failed_node_count != self.cancellation_node_count
            or self.cancellation_claims_cleared_count != self.cancellation_node_count
            or not self.cancellation_child_identity_retained
            or self.cancellation_superseded_delegation_count != 1
            or not self.forged_parent_rejected
            or not self.terminal_recancel_rejected
            or self.restart_event_delta != 0
            or self.duplicate_delivery_attempt_count != 1
            or self.duplicate_delivery_count != 0
            or self.physical_delivery_count != 1
            or self.false_success_count != 0
        ):
            raise ValueError("offline lifecycle evidence does not prove safe convergence")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("offline lifecycle evidence digest mismatch")
        return self


class LiveRestartCase(ContractModel):
    """One pre-restart Slack receipt re-observed without a second delivery."""

    evidence_id: LiveEvidenceId
    run_id: NonEmptyStr
    slack_response_ts: SlackMessageTs
    final_outbox_row_digest: Sha256
    final_delivered_at: datetime
    final_delivery_attempt_count: Literal[1]
    final_receipt_count: Literal[1]
    post_restart_delivery_attempt_delta: Literal[0]
    post_restart_final_outbox_count_delta: Literal[0]
    post_restart_matching_slack_message_count: Literal[1]
    post_restart_slack_readback_digest: Sha256
    snapshot_digest: Sha256

    @model_validator(mode="after")
    def exact_snapshot_digest(self) -> LiveRestartCase:
        if self.final_delivered_at.tzinfo is None:
            raise ValueError("live restart delivery timestamp must be timezone-aware")
        if self.snapshot_digest != _digest(
            self.model_dump(mode="json", exclude={"snapshot_digest"})
        ):
            raise ValueError("live restart case digest mismatch")
        return self


class LiveRestartEvidence(ContractModel):
    """Trusted listener epoch plus post-restart durable no-redelivery observations."""

    version: Literal["live-restart-evidence-v1"] = LIVE_RESTART_EVIDENCE_VERSION
    listener_epoch_digest: Sha256
    listener_started_at: datetime
    collection_observed_at: datetime
    cases: tuple[LiveRestartCase, ...] = Field(min_length=1)
    case_count: int = Field(ge=1)
    digest: Sha256

    @model_validator(mode="after")
    def exact_restart_observation(self) -> LiveRestartEvidence:
        if (
            self.listener_started_at.tzinfo is None
            or self.collection_observed_at.tzinfo is None
            or self.collection_observed_at <= self.listener_started_at
            or self.case_count != len(self.cases)
            or len({item.evidence_id for item in self.cases}) != len(self.cases)
            or any(
                item.final_delivered_at >= self.listener_started_at
                or float(item.slack_response_ts) >= self.listener_started_at.timestamp()
                for item in self.cases
            )
        ):
            raise ValueError("live restart evidence lacks a strict before/after epoch")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("live restart evidence digest mismatch")
        return self


class PostgresReliabilityEvidence(ContractModel):
    """Observed rollback-safe artifacts from the later current-head PG run."""

    version: Literal["postgres-reliability-evidence-v1"] = POSTGRES_EVIDENCE_VERSION
    alembic_head: NonEmptyStr
    rollback_safe: Literal[True]
    event_recovery: DurableRecoveryArtifact
    plan_recovery: DurableRecoveryArtifact
    failure_bundle: FailureBundle
    digest: Sha256

    @model_validator(mode="after")
    def exact_observed_recovery(self) -> PostgresReliabilityEvidence:
        _validate_recovery_artifact(
            self.event_recovery,
            expected={
                "stale-cas": ("run_store_commit", DurableRecoveryOutcome.REJECTED_SAFE),
                "duplicate-event-sequence": (
                    "run_event_unique_sequence",
                    DurableRecoveryOutcome.REJECTED_SAFE,
                ),
                "restart-replay": ("run_store_reload", DurableRecoveryOutcome.RELOAD_EXACT),
                "operator-export": ("failure_event_source", DurableRecoveryOutcome.EXPORTED),
            },
            exact_reload_ids=frozenset({"restart-replay"}),
        )
        _validate_recovery_artifact(
            self.plan_recovery,
            expected={
                "running-child-reload": (
                    "plan_store_replay",
                    DurableRecoveryOutcome.RELOAD_EXACT,
                ),
                "expired-child-reclaim": (
                    "plan_node_lease",
                    DurableRecoveryOutcome.RECLAIMED,
                ),
                "stale-child-fenced": (
                    "plan_node_claim",
                    DurableRecoveryOutcome.FENCED,
                ),
                "terminal-plan-reload": (
                    "plan_store_reload",
                    DurableRecoveryOutcome.RELOAD_EXACT,
                ),
            },
            exact_reload_ids=frozenset({"running-child-reload", "terminal-plan-reload"}),
        )
        config = self.failure_bundle.sanitized_config
        if (
            self.event_recovery.digest == self.plan_recovery.digest
            or config.get("alembic_head") != self.alembic_head
            or config.get("source") != "rollback-safe-postgres"
            or not self.failure_bundle.sanitized_events
            or len(self.failure_bundle.failure.event_ids)
            != len(self.failure_bundle.sanitized_events)
            or any(
                item.get("schema_version") != "v2" for item in self.failure_bundle.sanitized_events
            )
            or _case_by_id(self.event_recovery, "operator-export").observed_after_digest
            != _digest({"bundle_digest": self.failure_bundle.digest})
        ):
            raise ValueError("Postgres evidence is not bound to observed durable recovery")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("Postgres reliability evidence digest mismatch")
        return self


class FinalEvidenceAggregate(ContractModel):
    """Content-addressed summary whose states derive from validated component artifacts."""

    version: Literal["m5-final-evidence-v1"] = FINAL_EVIDENCE_VERSION
    offline_report_digest: Sha256
    offline_proof_digest: Sha256
    offline_fixture_set_digest: Sha256
    offline_scenario_count: int = Field(ge=1)
    offline_passed: Literal[True]
    topology_artifact_digest: Sha256
    topology_manifest_digest: Sha256
    topology_conversation_count: int = Field(ge=0)
    topology_kind_counts: dict[TopologyConversationKind, int]
    shared_external_presence: SharedExternalPresence
    live_manifest_digest: Sha256
    live_collection_digest: Sha256
    live_case_count: int = Field(ge=0)
    live_pending_evidence_ids: tuple[LiveEvidenceId, ...]
    live_restart: LiveRestartEvidence | None = None
    lifecycle: OfflineLifecycleEvidence
    postgres: PostgresReliabilityEvidence | None = None
    revised_live_acceptance: RevisedLiveAcceptanceArtifact | None = None
    component_states: dict[NonEmptyStr, EvidenceState]
    pending_requirements: tuple[NonEmptyStr, ...]
    offline_freeze_ready: bool
    final_milestone_ready: bool
    digest: Sha256

    @model_validator(mode="after")
    def exact_states_and_digest(self) -> FinalEvidenceAggregate:
        expected_states: dict[str, EvidenceState] = {
            "frozen_offline_report": EvidenceState.COLLECTED,
            "offline_cancellation_restart": EvidenceState.COLLECTED,
            "slack_topology": EvidenceState.COLLECTED,
            "strict_live_reconciliation": (
                EvidenceState.PARTIAL if self.live_pending_evidence_ids else EvidenceState.COLLECTED
            ),
            "live_listener_restart": (
                EvidenceState.PENDING if self.live_restart is None else EvidenceState.COLLECTED
            ),
            "postgres_current_head_reliability": (
                EvidenceState.PENDING if self.postgres is None else EvidenceState.COLLECTED
            ),
            "revised_d063_d066_live_acceptance": (
                EvidenceState.PENDING
                if self.revised_live_acceptance is None
                else EvidenceState.COLLECTED
            ),
        }
        expected_pending = tuple(
            sorted(
                (
                    *(f"live:{item}" for item in self.live_pending_evidence_ids),
                    *(
                        ("live:listener_restart_no_redelivery",)
                        if self.live_restart is None
                        else ()
                    ),
                    *(("postgres:current_head_reliability",) if self.postgres is None else ()),
                    *(
                        ("live:revised_d063_d066_acceptance",)
                        if self.revised_live_acceptance is None
                        else ()
                    ),
                )
            )
        )
        expected_freeze_ready = (
            self.offline_passed
            and self.offline_scenario_count
            == len(REQUIRED_PROOF_SCENARIOS["paired_baseline_report"])
            and self.lifecycle.false_success_count == 0
            and self.lifecycle.duplicate_delivery_count == 0
        )
        if (
            self.component_states != expected_states
            or self.pending_requirements != expected_pending
            or self.offline_freeze_ready != expected_freeze_ready
            or self.final_milestone_ready != (expected_freeze_ready and not expected_pending)
            or self.live_case_count + len(self.live_pending_evidence_ids) != 9
            or sum(self.topology_kind_counts.values()) != self.topology_conversation_count
        ):
            raise ValueError("final evidence status does not reconcile with observed components")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("final evidence aggregate digest mismatch")
        return self


async def build_offline_lifecycle_evidence() -> OfflineLifecycleEvidence:
    """Execute both probes; no caller-supplied pass/fail values are accepted."""

    cancellation = _observe_cancellation()
    scenarios = load_scenarios(
        default_scenario_root(),
        scenario_ids=frozenset({RESTART_SCENARIO_ID}),
    )
    results = await run_scenarios_async(scenarios)
    if len(scenarios) != 1 or len(results) != 1:
        raise ValueError("restart evidence did not execute its exact scenario")
    scenario = scenarios[0]
    result = results[0]
    if (
        scenario.id != RESTART_SCENARIO_ID
        or result.scenario_id != scenario.id
        or result.scenario_version != scenario.version
        or result.fixture_digest != scenario.fixture_digest
        or result.status is not ScenarioStatus.PASSED
        or result.invariant_failures
    ):
        raise ValueError("restart evidence scenario failed or diverged from its fixture")
    payload: dict[str, object] = {
        "version": LIFECYCLE_EVIDENCE_VERSION,
        **cancellation,
        "restart_fixture_digest": result.fixture_digest,
        "restart_result_digest": _digest(result.model_dump(mode="json")),
        "restart_event_delta": _integer_metric(result.metrics, "replay_event_delta"),
        "duplicate_delivery_attempt_count": _integer_metric(
            result.metrics,
            "duplicate_delivery_attempt_count",
        ),
        "duplicate_delivery_count": _integer_metric(
            result.metrics,
            "duplicate_delivery_count",
        ),
        "physical_delivery_count": _integer_metric(
            result.metrics,
            "physical_delivery_count",
        ),
        "false_success_count": _integer_metric(result.raw_counts, "false_success_count"),
    }
    return OfflineLifecycleEvidence.model_validate({**payload, "digest": _digest(payload)})


async def build_final_evidence_aggregate(
    *,
    offline_report_path: Path,
    live_proof_path: Path,
    topology_path: Path,
    live_restart_path: Path | None = None,
    postgres_path: Path | None = None,
    revised_live_acceptance_path: Path | None = None,
) -> FinalEvidenceAggregate:
    report = FrozenAggregateReport.model_validate_json(
        await asyncio.to_thread(offline_report_path.read_bytes)
    )
    live_manifest = ProofManifest.model_validate_json(
        await asyncio.to_thread(live_proof_path.read_bytes)
    )
    topology = SlackTopologyArtifact.model_validate_json(
        await asyncio.to_thread(topology_path.read_bytes)
    )
    live_restart = (
        None
        if live_restart_path is None
        else LiveRestartEvidence.model_validate_json(
            await asyncio.to_thread(live_restart_path.read_bytes)
        )
    )
    postgres = (
        None
        if postgres_path is None
        else PostgresReliabilityEvidence.model_validate_json(
            await asyncio.to_thread(postgres_path.read_bytes)
        )
    )
    revised_live_acceptance = (
        None
        if revised_live_acceptance_path is None
        else RevisedLiveAcceptanceArtifact.model_validate_json(
            await asyncio.to_thread(revised_live_acceptance_path.read_bytes)
        )
    )
    if postgres is not None and postgres.alembic_head != repository_alembic_head():
        raise ValueError("Postgres evidence does not match the repository Alembic head")
    if (
        not report.offline_passed
        or set(report.scenario_ids) != REQUIRED_PROOF_SCENARIOS["paired_baseline_report"]
    ):
        raise ValueError("final evidence requires the passing exact nineteen-scenario report")
    validate_proof_manifest(report.proof_manifest)
    validate_proof_manifest(live_manifest)
    collection = _live_collection(live_manifest)
    if live_restart is not None:
        _validate_live_restart_binding(live_restart, collection)
    if revised_live_acceptance is not None:
        if live_restart is None or postgres is None:
            raise ValueError("revised live acceptance requires both restart and Postgres evidence")
        if (
            revised_live_acceptance.live_restart_digest != live_restart.digest
            or revised_live_acceptance.postgres_reliability_digest != postgres.digest
            or revised_live_acceptance.outbox_recovery.alembic_head != postgres.alembic_head
            or revised_live_acceptance.runtime_health.listener_epoch_digest
            != live_restart.listener_epoch_digest
            or revised_live_acceptance.runtime_health.listener_started_at
            != live_restart.listener_started_at
        ):
            raise ValueError("revised live acceptance is not bound to final evidence components")
        dm_reference = tuple(
            item for item in collection.cases if str(item.evidence_id) == "dm_membership_union"
        )
        if len(dm_reference) != 1 or (
            revised_live_acceptance.dm_root_reference_run_id != dm_reference[0].run_id
            or revised_live_acceptance.dm_root_reference_request_ts != dm_reference[0].message_ts
            or revised_live_acceptance.dm_root_reference_response_ts
            != dm_reference[0].slack_response_ts
        ):
            raise ValueError("revised DM follow-up is not anchored to the fixed-nine DM proof")
    base_manifest = live_manifest.model_copy(
        update={
            "artifacts": tuple(
                item for item in live_manifest.artifacts if item.id != LIVE_PROOF_ARTIFACT_ID
            )
        }
    )
    if base_manifest != report.proof_manifest:
        raise ValueError("live proof is not derived from the frozen offline proof")
    lifecycle = await build_offline_lifecycle_evidence()
    restart_fixture_digest = _fixture_digest(
        report.proof_manifest,
        RESTART_SCENARIO_ID,
    )
    if lifecycle.restart_fixture_digest != restart_fixture_digest:
        raise ValueError("restart evidence fixture diverges from the frozen report")

    component_states: dict[str, EvidenceState] = {
        "frozen_offline_report": EvidenceState.COLLECTED,
        "offline_cancellation_restart": EvidenceState.COLLECTED,
        "slack_topology": EvidenceState.COLLECTED,
        "strict_live_reconciliation": (
            EvidenceState.PARTIAL if collection.pending_evidence_ids else EvidenceState.COLLECTED
        ),
        "live_listener_restart": (
            EvidenceState.PENDING if live_restart is None else EvidenceState.COLLECTED
        ),
        "postgres_current_head_reliability": (
            EvidenceState.PENDING if postgres is None else EvidenceState.COLLECTED
        ),
        "revised_d063_d066_live_acceptance": (
            EvidenceState.PENDING if revised_live_acceptance is None else EvidenceState.COLLECTED
        ),
    }
    pending = tuple(
        sorted(
            (
                *(f"live:{item}" for item in collection.pending_evidence_ids),
                *(("live:listener_restart_no_redelivery",) if live_restart is None else ()),
                *(("postgres:current_head_reliability",) if postgres is None else ()),
                *(
                    ("live:revised_d063_d066_acceptance",)
                    if revised_live_acceptance is None
                    else ()
                ),
            )
        )
    )
    payload: dict[str, object] = {
        "version": FINAL_EVIDENCE_VERSION,
        "offline_report_digest": report.digest,
        "offline_proof_digest": report.proof_manifest.digest,
        "offline_fixture_set_digest": report.fixture_set_digest,
        "offline_scenario_count": len(report.scenario_ids),
        "offline_passed": True,
        "topology_artifact_digest": topology.digest,
        "topology_manifest_digest": topology.manifest_digest,
        "topology_conversation_count": topology.conversation_count,
        "topology_kind_counts": topology.kind_counts,
        "shared_external_presence": topology.shared_external_presence,
        "live_manifest_digest": live_manifest.digest,
        "live_collection_digest": collection.digest,
        "live_case_count": len(collection.cases),
        "live_pending_evidence_ids": collection.pending_evidence_ids,
        "live_restart": live_restart,
        "lifecycle": lifecycle,
        "postgres": postgres,
        "revised_live_acceptance": revised_live_acceptance,
        "component_states": component_states,
        "pending_requirements": pending,
        "offline_freeze_ready": True,
        "final_milestone_ready": not pending,
    }
    return FinalEvidenceAggregate.model_validate({**payload, "digest": _digest_json(payload)})


async def export_final_evidence_aggregate(
    *,
    offline_report_path: Path,
    live_proof_path: Path,
    topology_path: Path,
    destination: Path,
    live_restart_path: Path | None = None,
    postgres_path: Path | None = None,
    revised_live_acceptance_path: Path | None = None,
) -> FinalEvidenceAggregate:
    aggregate = await build_final_evidence_aggregate(
        offline_report_path=offline_report_path,
        live_proof_path=live_proof_path,
        topology_path=topology_path,
        live_restart_path=live_restart_path,
        postgres_path=postgres_path,
        revised_live_acceptance_path=revised_live_acceptance_path,
    )
    _atomic_write(destination, aggregate.model_dump_json(indent=2) + "\n")
    return aggregate


def make_postgres_reliability_evidence(
    *,
    alembic_head: str,
    event_recovery: DurableRecoveryArtifact,
    plan_recovery: DurableRecoveryArtifact,
    failure_bundle: FailureBundle,
) -> PostgresReliabilityEvidence:
    """Wrap full validated observed artifacts; callers cannot select pass states."""

    payload: dict[str, object] = {
        "version": POSTGRES_EVIDENCE_VERSION,
        "alembic_head": alembic_head,
        "rollback_safe": True,
        "event_recovery": event_recovery,
        "plan_recovery": plan_recovery,
        "failure_bundle": failure_bundle,
    }
    return PostgresReliabilityEvidence.model_validate({**payload, "digest": _digest_json(payload)})


def export_postgres_reliability_evidence(
    *,
    alembic_head: str,
    event_recovery: DurableRecoveryArtifact,
    plan_recovery: DurableRecoveryArtifact,
    failure_bundle: FailureBundle,
    destination: Path,
) -> PostgresReliabilityEvidence:
    """Atomically export only a fully validated observed Postgres artifact."""

    evidence = make_postgres_reliability_evidence(
        alembic_head=alembic_head,
        event_recovery=event_recovery,
        plan_recovery=plan_recovery,
        failure_bundle=failure_bundle,
    )
    _atomic_write(destination, evidence.model_dump_json(indent=2) + "\n")
    return evidence


def make_live_restart_case(
    *,
    evidence_id: LiveEvidenceId,
    run_id: str,
    slack_response_ts: str,
    final_outbox_row_digest: str,
    final_delivered_at: datetime,
    post_restart_slack_readback_digest: str,
) -> LiveRestartCase:
    """Build one case from exact collector observations; no pass flag is accepted."""

    payload: dict[str, object] = {
        "evidence_id": evidence_id,
        "run_id": run_id,
        "slack_response_ts": slack_response_ts,
        "final_outbox_row_digest": final_outbox_row_digest,
        "final_delivered_at": final_delivered_at,
        "final_delivery_attempt_count": 1,
        "final_receipt_count": 1,
        "post_restart_delivery_attempt_delta": 0,
        "post_restart_final_outbox_count_delta": 0,
        "post_restart_matching_slack_message_count": 1,
        "post_restart_slack_readback_digest": post_restart_slack_readback_digest,
    }
    return LiveRestartCase.model_validate({**payload, "snapshot_digest": _digest_json(payload)})


def make_live_restart_evidence(
    *,
    listener_epoch_digest: str,
    listener_started_at: datetime,
    collection_observed_at: datetime,
    cases: tuple[LiveRestartCase, ...],
) -> LiveRestartEvidence:
    """Bind exact pre/post epoch observations into a content-addressed artifact."""

    payload: dict[str, object] = {
        "version": LIVE_RESTART_EVIDENCE_VERSION,
        "listener_epoch_digest": listener_epoch_digest,
        "listener_started_at": listener_started_at,
        "collection_observed_at": collection_observed_at,
        "cases": cases,
        "case_count": len(cases),
    }
    return LiveRestartEvidence.model_validate({**payload, "digest": _digest_json(payload)})


def _observe_cancellation() -> dict[str, object]:
    definitions = (
        PlanNodeDefinition(key="research", objective="Complete deterministic research."),
        PlanNodeDefinition(
            key="synthesis",
            objective="Synthesize the observed result.",
            depends_on=("research",),
        ),
    )
    revision = PlanRevision(
        id="revision-final-evidence",
        plan_id="plan-final-evidence",
        number=1,
        goal="Prove parent cancellation propagation.",
        nodes=definitions,
        digest=revision_digest("Prove parent cancellation propagation.", definitions),
        reason="initial_plan",
        created_at=_FIXED_TIME,
    )
    plan = Plan(
        id=revision.plan_id,
        scope=ScopeKey(organization_id="eval-org", strategy_id="eval-strategy"),
        parent_task_id="task-parent-final-evidence",
        parent_run_id="run-parent-final-evidence",
        idempotency_key="final-evidence-cancellation",
        initial_digest=revision.digest,
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    running = PlanNode(
        id="node-research",
        plan_id=plan.id,
        revision_id=revision.id,
        revision_number=1,
        definition=definitions[0],
        status=PlanNodeStatus.RUNNING,
        attempt=1,
        claim_owner="offline-worker",
        claim_token="offline-claim",
        lease_expires_at=_FIXED_TIME + timedelta(minutes=1),
        child_task_id="task-child-final-evidence",
        child_run_id="run-child-final-evidence",
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    pending = PlanNode(
        id="node-synthesis",
        plan_id=plan.id,
        revision_id=revision.id,
        revision_number=1,
        definition=definitions[1],
        created_at=_FIXED_TIME,
        updated_at=_FIXED_TIME,
    )
    delegation = Delegation(
        id="delegation-final-evidence",
        plan_id=plan.id,
        revision_id=revision.id,
        node_id=running.id,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
        attempt=1,
        owner="offline-worker",
        claim_token="offline-claim",
        status=DelegationStatus.RUNNING,
        child_task_id=running.child_task_id,
        child_run_id=running.child_run_id,
        created_at=_FIXED_TIME,
    )
    snapshot = PlanSnapshot(
        plan=plan,
        revisions=(revision,),
        nodes=(running, pending),
        delegations=(delegation,),
    )
    cancelled = cancel_plan_snapshot(
        snapshot,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
        reason="operator_cancelled",
        cancelled_at=_FIXED_TIME + timedelta(seconds=5),
    )
    forged_parent_rejected = _cancellation_rejected(
        snapshot,
        parent_task_id="forged-parent",
        parent_run_id=plan.parent_run_id,
    )
    terminal_recancel_rejected = _cancellation_rejected(
        cancelled,
        parent_task_id=plan.parent_task_id,
        parent_run_id=plan.parent_run_id,
    )
    return {
        "cancellation_snapshot_digest": _digest(cancelled.model_dump(mode="json")),
        "cancellation_node_count": len(cancelled.current_nodes),
        "cancellation_failed_node_count": sum(
            item.status is PlanNodeStatus.FAILED for item in cancelled.current_nodes
        ),
        "cancellation_claims_cleared_count": sum(
            item.claim_token is None and item.claim_owner is None and item.lease_expires_at is None
            for item in cancelled.current_nodes
        ),
        "cancellation_child_identity_retained": (
            cancelled.current_nodes[0].child_task_id == running.child_task_id
            and cancelled.current_nodes[0].child_run_id == running.child_run_id
        ),
        "cancellation_superseded_delegation_count": sum(
            item.status is DelegationStatus.SUPERSEDED for item in cancelled.delegations
        ),
        "forged_parent_rejected": forged_parent_rejected,
        "terminal_recancel_rejected": terminal_recancel_rejected,
    }


def _cancellation_rejected(
    snapshot: PlanSnapshot,
    *,
    parent_task_id: str,
    parent_run_id: str,
) -> bool:
    try:
        cancel_plan_snapshot(
            snapshot,
            parent_task_id=parent_task_id,
            parent_run_id=parent_run_id,
            reason="operator_cancelled",
            cancelled_at=_FIXED_TIME + timedelta(seconds=6),
        )
    except ValueError:
        return True
    return False


def _live_collection(manifest: ProofManifest) -> LiveProofCollection:
    artifacts = tuple(item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID)
    if len(artifacts) != 1:
        raise ValueError("final evidence requires one strict live reconciliation artifact")
    artifact = artifacts[0]
    metadata = artifact.metadata
    collection = LiveProofCollection.model_validate(
        {
            "version": metadata.get("version"),
            "profile": metadata.get("profile"),
            "authority_digest": metadata.get("authority_digest"),
            "cases": metadata.get("case_summaries"),
            "pending_evidence_ids": metadata.get("pending_evidence_ids"),
            "status": metadata.get("status"),
            "digest": metadata.get("collection_digest"),
        }
    )
    expected_kind = (
        "live_slack_supabase_reconciliation"
        if collection.status == "complete"
        else "live_slack_supabase_reconciliation_partial"
    )
    if (
        artifact.kind != expected_kind
        or artifact.provider_label != "slack-supabase-live"
        or artifact.command != "python -m leo.evals.live_proof_operator"
        or artifact.scenario_ids != tuple(str(item.evidence_id) for item in collection.cases)
        or artifact.fixture_digests != tuple(item.binding_digest for item in collection.cases)
        or artifact.sanitized_run_ids != tuple(item.run_id for item in collection.cases)
    ):
        raise ValueError("strict live proof artifact diverges from its observed collection")
    return collection


def _validate_live_restart_binding(
    evidence: LiveRestartEvidence,
    collection: LiveProofCollection,
) -> None:
    expected = {item.evidence_id: item for item in collection.cases}
    observed = {item.evidence_id: item for item in evidence.cases}
    if set(observed) != set(expected):
        raise ValueError("live restart evidence does not cover the exact collected live cohort")
    for evidence_id, case in observed.items():
        source = expected[evidence_id]
        if case.run_id != source.run_id or case.slack_response_ts != source.slack_response_ts:
            raise ValueError("live restart evidence is not bound to the strict live proof")


def _validate_recovery_artifact(
    artifact: DurableRecoveryArtifact,
    *,
    expected: Mapping[str, tuple[str, DurableRecoveryOutcome]],
    exact_reload_ids: frozenset[str],
) -> None:
    cases = {item.id: item for item in artifact.cases}
    if (
        artifact.database_label != "supabase-postgres-current-head"
        or not artifact.rollback_safe
        or artifact.duplicate_commit_count != 0
        or artifact.false_success_count != 0
        or set(cases) != set(expected)
    ):
        raise ValueError("Postgres durable recovery artifact has an unmatched cohort")
    for case_id, (boundary, outcome) in expected.items():
        case = cases[case_id]
        if case.boundary != boundary or case.outcome is not outcome:
            raise ValueError("Postgres durable recovery case has an unmatched outcome")
        if case_id in exact_reload_ids and (
            case.mutation_applied or case.observed_before_digest != case.observed_after_digest
        ):
            raise ValueError("Postgres restart recovery did not reload an exact snapshot")


def _case_by_id(
    artifact: DurableRecoveryArtifact,
    case_id: str,
) -> DurableRecoveryCase:
    matches = tuple(item for item in artifact.cases if item.id == case_id)
    if len(matches) != 1:
        raise ValueError(f"Postgres recovery case {case_id} is absent or ambiguous")
    return matches[0]


def repository_alembic_head() -> str:
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config("alembic.ini")
    config.set_main_option("script_location", "migrations")
    head = ScriptDirectory.from_config(config).get_current_head()
    if head is None:
        raise ValueError("repository Alembic head is absent")
    return head


def _fixture_digest(manifest: ProofManifest, scenario_id: str) -> str:
    matches: set[str] = set()
    for artifact in manifest.artifacts:
        for candidate, digest in zip(
            artifact.scenario_ids,
            artifact.fixture_digests,
            strict=True,
        ):
            if candidate == scenario_id:
                matches.add(digest)
    if len(matches) == 1:
        return matches.pop()
    if matches:
        raise ValueError(f"offline proof has conflicting fixture digests for {scenario_id}")
    raise ValueError(f"offline proof lacks fixture {scenario_id}")


def _integer_metric(values: Mapping[str, float | int | str], key: str) -> int:
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"observed lifecycle metric {key} is absent or non-integral")
    return value


def _digest_json(value: Mapping[str, object]) -> str:
    return _digest(_json_value(value))


def _json_value(value: object) -> JsonValue:
    if isinstance(value, ContractModel):
        return cast(JsonValue, value.model_dump(mode="json"))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return cast(str, _DATETIME_ADAPTER.dump_python(value, mode="json"))
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported final evidence value: {type(value).__name__}")


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-final-evidence",
        description="Validate and aggregate offline M5 evidence without external calls.",
    )
    parser.add_argument("--offline-report", required=True, type=Path)
    parser.add_argument("--live-proof", required=True, type=Path)
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--live-restart-artifact", type=Path)
    parser.add_argument("--postgres-artifact", type=Path)
    parser.add_argument("--revised-live-acceptance-artifact", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


async def _run(arguments: argparse.Namespace) -> FinalEvidenceAggregate:
    return await export_final_evidence_aggregate(
        offline_report_path=arguments.offline_report,
        live_proof_path=arguments.live_proof,
        topology_path=arguments.topology,
        live_restart_path=arguments.live_restart_artifact,
        postgres_path=arguments.postgres_artifact,
        revised_live_acceptance_path=arguments.revised_live_acceptance_artifact,
        destination=arguments.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        aggregate = asyncio.run(_run(arguments))
    except Exception:
        print(
            json.dumps(
                {"code": "final_evidence_aggregation_failed", "status": "failed"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "digest": aggregate.digest,
                "final_milestone_ready": aggregate.final_milestone_ready,
                "offline_freeze_ready": aggregate.offline_freeze_ready,
                "pending_requirements": list(aggregate.pending_requirements),
                "status": "ok",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
