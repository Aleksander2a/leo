"""Typed, content-addressed acceptance for the revised D-063--D-066 live surface.

The fixed-nine M5 proof remains an unchanged baseline.  This companion proves
the later conversational recovery, elastic deliberation, complete thread
context, and expanded research tools from exact Slack/Supabase observations.
The database source performs SELECTs only; Slack text is bound by digest through
a content-free connector readback.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, TypeAdapter, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ContractModel, NonEmptyStr
from leo.persistence.schema import (
    ClaimRow,
    DeliveryOutboxRow,
    ObservationRow,
    RunEventRow,
    RunRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SlackMessageTs = Annotated[str, Field(pattern=r"^[0-9]+[.][0-9]+$")]

REVISED_LIVE_ACCEPTANCE_VERSION: Literal["revised-live-acceptance-v1"] = (
    "revised-live-acceptance-v1"
)
REVISED_LIVE_REQUEST_VERSION: Literal["revised-live-acceptance-request-v1"] = (
    "revised-live-acceptance-request-v1"
)
SLACK_REVISED_READBACK_VERSION: Literal["slack-revised-live-readback-v1"] = (
    "slack-revised-live-readback-v1"
)
RUNTIME_HEALTH_READBACK_VERSION: Literal["runtime-health-readback-v1"] = (
    "runtime-health-readback-v1"
)
OUTBOX_RECOVERY_PROBE_VERSION: Literal["outbox-recovery-probe-v1"] = "outbox-recovery-probe-v1"
OUTBOX_RECOVERY_EVIDENCE_VERSION: Literal["outbox-recovery-postgres-v1"] = (
    "outbox-recovery-postgres-v1"
)

_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_DATETIME_ADAPTER = TypeAdapter(datetime)
_HEALTH_COMPONENTS: dict[str, str] = {
    "database": "ok",
    "metadata": "ok",
    "membership": "ok",
    "model": "ok",
    "orchestration": "ok",
    "queue": "ok",
    "outbox": "ok",
    "last_success": "ok",
    "socket": "unknown_cross_process",
}


class RevisedLiveNotFound(LookupError):
    """An exact bound durable row is absent or ambiguous."""


class RevisedLiveIntegrityError(ValueError):
    """A revised live observation does not satisfy the exact contract."""


class RevisedLiveCaseId(StrEnum):
    TERMINAL_RECOVERY = "conversational_terminal_recovery"
    MPIM_CLARIFICATION = "mpim_short_clarification"
    MPIM_THREAD_FOLLOWUP = "mpim_complete_thread_followup"
    DM_MEMORY_ROOT = "dm_membership_memory_root"
    DM_THREAD_FOLLOWUP = "dm_complete_thread_followup"
    TAVILY_RESEARCH = "natural_tavily_research"
    FINNHUB_EARNINGS = "natural_finnhub_earnings"


REVISED_LIVE_BINDING_IDS: frozenset[RevisedLiveCaseId] = frozenset(RevisedLiveCaseId)
REVISED_LIVE_CASE_IDS: frozenset[RevisedLiveCaseId] = REVISED_LIVE_BINDING_IDS - {
    RevisedLiveCaseId.DM_MEMORY_ROOT
}


class RevisedLiveRunBinding(ContractModel):
    """Trusted exact Slack/run identity for one revised live case."""

    case_id: RevisedLiveCaseId
    run_id: NonEmptyStr
    channel_id: NonEmptyStr
    request_message_ts: SlackMessageTs
    thread_root_ts: SlackMessageTs
    slack_response_ts: SlackMessageTs
    expected_context_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_binding_shape(self) -> RevisedLiveRunBinding:
        if (
            self.expected_context_conversation_ids
            != tuple(sorted(self.expected_context_conversation_ids))
            or len(self.expected_context_conversation_ids)
            != len(set(self.expected_context_conversation_ids))
            or self.channel_id not in self.expected_context_conversation_ids
            or float(self.slack_response_ts) <= float(self.request_message_ts)
            or (
                self.case_id
                not in {
                    RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP,
                    RevisedLiveCaseId.DM_THREAD_FOLLOWUP,
                }
                and self.request_message_ts != self.thread_root_ts
            )
            or (
                self.case_id
                in {
                    RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP,
                    RevisedLiveCaseId.DM_THREAD_FOLLOWUP,
                }
                and self.request_message_ts == self.thread_root_ts
            )
        ):
            raise ValueError("revised live binding has an invalid destination or timeline")
        return self

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class RevisedLiveAcceptanceRequest(ContractModel):
    """Exact seven-case collection request; omission fails before database access."""

    version: Literal["revised-live-acceptance-request-v1"] = REVISED_LIVE_REQUEST_VERSION
    organization_id: NonEmptyStr
    team_id: NonEmptyStr
    bindings: tuple[RevisedLiveRunBinding, ...] = Field(min_length=7, max_length=7)
    post_restart_case_id: RevisedLiveCaseId

    @model_validator(mode="after")
    def exact_case_partition(self) -> RevisedLiveAcceptanceRequest:
        ids = tuple(item.case_id for item in self.bindings)
        if (
            ids != tuple(sorted(ids, key=str))
            or set(ids) != REVISED_LIVE_BINDING_IDS
            or self.post_restart_case_id not in REVISED_LIVE_CASE_IDS
        ):
            raise ValueError("revised live request must bind the sorted exact seven-case cohort")
        return self


class SlackRevisedReadbackCase(ContractModel):
    """Content-free connector result for one exact Slack response and thread."""

    case_id: RevisedLiveCaseId
    run_id: NonEmptyStr
    channel_id: NonEmptyStr
    thread_root_ts: SlackMessageTs
    slack_response_ts: SlackMessageTs
    matching_response_count: Literal[1]
    response_text_digest: Sha256
    thread_message_timestamps: tuple[SlackMessageTs, ...] = Field(min_length=2)
    readback_digest: Sha256

    @model_validator(mode="after")
    def exact_content_free_readback(self) -> SlackRevisedReadbackCase:
        timestamps = self.thread_message_timestamps
        if (
            timestamps != tuple(sorted(timestamps, key=float))
            or len(timestamps) != len(set(timestamps))
            or self.thread_root_ts not in timestamps
            or self.slack_response_ts not in timestamps
            or self.readback_digest
            != _digest(self.model_dump(mode="json", exclude={"readback_digest"}))
        ):
            raise ValueError("Slack revised readback is not exact and content-addressed")
        return self


class SlackRevisedReadback(ContractModel):
    """Content-free Slack connector readback for the exact seven-case cohort."""

    version: Literal["slack-revised-live-readback-v1"] = SLACK_REVISED_READBACK_VERSION
    observed_at: datetime
    cases: tuple[SlackRevisedReadbackCase, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def exact_readback_cohort(self) -> SlackRevisedReadback:
        ids = tuple(item.case_id for item in self.cases)
        if (
            self.observed_at.tzinfo is None
            or ids != tuple(sorted(ids, key=str))
            or set(ids) != REVISED_LIVE_CASE_IDS
        ):
            raise ValueError("Slack revised readback must cover the exact seven-case cohort")
        return self


class RuntimeHealthReadback(ContractModel):
    """Content-free listener epoch and deep-health observation from the operator."""

    version: Literal["runtime-health-readback-v1"] = RUNTIME_HEALTH_READBACK_VERSION
    listener_epoch_digest: Sha256
    listener_started_at: datetime
    listener_connected_at: datetime
    observed_at: datetime
    last_success_at: datetime
    component_states: dict[NonEmptyStr, NonEmptyStr]
    digest: Sha256

    @model_validator(mode="after")
    def exact_listener_health(self) -> RuntimeHealthReadback:
        timestamps = (
            self.listener_started_at,
            self.listener_connected_at,
            self.observed_at,
            self.last_success_at,
        )
        if (
            any(item.tzinfo is None for item in timestamps)
            or not (self.listener_started_at <= self.listener_connected_at < self.observed_at)
            or not (self.listener_started_at <= self.last_success_at <= self.observed_at)
            or self.component_states != _HEALTH_COMPONENTS
            or self.digest != _digest(self.model_dump(mode="json", exclude={"digest"}))
        ):
            raise ValueError("runtime health readback is unhealthy or not content-addressed")
        return self


class OutboxRecoveryCaseId(StrEnum):
    PENDING_FINAL = "pending_final_delivery"
    MISSING_FINAL = "missing_final_repair"


class OutboxRecoveryProbe(ContractModel):
    """Rollback-preserved content-free observation emitted by one Postgres test."""

    version: Literal["outbox-recovery-probe-v1"] = OUTBOX_RECOVERY_PROBE_VERSION
    case_id: OutboxRecoveryCaseId
    database_label: Literal["supabase-postgres-current-head"]
    pytest_node_id: NonEmptyStr
    initial_final_outbox_count: int = Field(ge=0)
    repaired_final_outbox_count: Literal[1]
    repair_created_count: int = Field(ge=0)
    claimed_count: Literal[1]
    final_outbox_count: Literal[1]
    final_state: Literal["delivered"]
    final_delivery_attempt_count: Literal[1]
    final_receipt_count: Literal[1]
    physical_delivery_count: Literal[1]
    duplicate_delivery_count: Literal[0]
    second_repair_count: Literal[0]
    second_dispatch_count: Literal[0]
    before_digest: Sha256
    after_digest: Sha256
    digest: Sha256

    @model_validator(mode="after")
    def exact_recovery_transition(self) -> OutboxRecoveryProbe:
        expected = {
            OutboxRecoveryCaseId.PENDING_FINAL: (
                "tests/postgres/test_outbox.py::test_pending_final_intent_is_delivered_once",
                1,
                0,
            ),
            OutboxRecoveryCaseId.MISSING_FINAL: (
                "tests/postgres/test_outbox.py::test_terminal_reconciliation_repairs_missing_final_intent",
                0,
                1,
            ),
        }[self.case_id]
        if (
            (self.pytest_node_id, self.initial_final_outbox_count, self.repair_created_count)
            != expected
            or self.before_digest == self.after_digest
            or self.digest != _digest(self.model_dump(mode="json", exclude={"digest"}))
        ):
            raise ValueError("outbox recovery probe does not prove its exact transition")
        return self


class OutboxRecoveryPostgresEvidence(ContractModel):
    """Exact pending-plus-missing final-outbox recovery cohort."""

    version: Literal["outbox-recovery-postgres-v1"] = OUTBOX_RECOVERY_EVIDENCE_VERSION
    alembic_head: NonEmptyStr
    probes: tuple[OutboxRecoveryProbe, ...] = Field(min_length=2, max_length=2)
    case_count: Literal[2]
    digest: Sha256

    @model_validator(mode="after")
    def exact_probe_cohort(self) -> OutboxRecoveryPostgresEvidence:
        ids = tuple(item.case_id for item in self.probes)
        if (
            ids != tuple(sorted(ids, key=str))
            or set(ids) != set(OutboxRecoveryCaseId)
            or self.digest != _digest(self.model_dump(mode="json", exclude={"digest"}))
        ):
            raise ValueError("outbox recovery evidence lacks the exact two-case cohort")
        return self


class RevisedLiveCaseObservation(ContractModel):
    """Content-free database/Slack reconciliation for one revised behavior."""

    case_id: RevisedLiveCaseId
    binding_digest: Sha256
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    channel_id: NonEmptyStr
    request_message_ts: SlackMessageTs
    thread_root_ts: SlackMessageTs
    slack_response_ts: SlackMessageTs
    received_at: datetime
    delivered_at: datetime
    ingress_latency_ms: int = Field(ge=0, lt=3000)
    conversation_kind: NonEmptyStr
    context_projection_source: NonEmptyStr
    context_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    context_access_hash: Sha256
    task_status: NonEmptyStr
    run_status: NonEmptyStr
    terminal_reason: NonEmptyStr
    final_output_digest: Sha256
    final_payload_digest: Sha256
    final_payload_utf8_bytes: int = Field(ge=1)
    event_sequence_digest: Sha256
    context_manifest_digest: Sha256
    semantic_markers: tuple[NonEmptyStr, ...] = Field(min_length=2)
    observation_kind_digests: tuple[Sha256, ...]
    claim_count: int = Field(ge=0)
    slack_readback_digest: Sha256
    snapshot_digest: Sha256

    @model_validator(mode="after")
    def exact_case_snapshot(self) -> RevisedLiveCaseObservation:
        expected_markers = tuple(
            sorted(
                {
                    RevisedLiveCaseId.TERMINAL_RECOVERY: (
                        "conversational_safe_terminal",
                        "durable_budget_exhausted",
                    ),
                    RevisedLiveCaseId.MPIM_CLARIFICATION: (
                        "elastic_short_clarification",
                        "mpim_singleton_isolation",
                    ),
                    RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP: (
                        "complete_thread_context",
                        "mpim_singleton_isolation",
                        "thread_follow_up",
                    ),
                    RevisedLiveCaseId.DM_MEMORY_ROOT: (
                        "dm_membership_projection",
                        "memory_grounded",
                    ),
                    RevisedLiveCaseId.DM_THREAD_FOLLOWUP: (
                        "complete_thread_context",
                        "dm_membership_projection",
                        "memory_grounded",
                        "thread_follow_up",
                    ),
                    RevisedLiveCaseId.TAVILY_RESEARCH: (
                        "natural_language_prompt",
                        "selected_public_fetch",
                        "tavily_search_discovery",
                        "verified_source_claim",
                    ),
                    RevisedLiveCaseId.FINNHUB_EARNINGS: (
                        "expanded_finnhub_earnings",
                        "natural_language_prompt",
                        "verified_source_claim",
                    ),
                }[self.case_id]
            )
        )
        expected_terminal = self.case_id is RevisedLiveCaseId.TERMINAL_RECOVERY
        if (
            self.received_at.tzinfo is None
            or self.delivered_at.tzinfo is None
            or self.delivered_at <= self.received_at
            or self.context_conversation_ids != tuple(sorted(self.context_conversation_ids))
            or self.semantic_markers != expected_markers
            or (
                expected_terminal
                and (
                    (self.task_status, self.run_status, self.terminal_reason)
                    != ("failed", "budget_exhausted", "iteration_budget_exhausted")
                    or self.final_output_digest != _EMPTY_SHA256
                )
            )
            or (
                not expected_terminal
                and (self.task_status, self.run_status, self.terminal_reason)
                != ("completed", "completed", "verified_completion")
            )
            or self.snapshot_digest
            != _digest(self.model_dump(mode="json", exclude={"snapshot_digest"}))
        ):
            raise ValueError("revised live case is not an exact content-addressed snapshot")
        return self


class RevisedLiveAcceptanceArtifact(ContractModel):
    """Complete revised live acceptance required by the M5 final aggregate."""

    version: Literal["revised-live-acceptance-v1"] = REVISED_LIVE_ACCEPTANCE_VERSION
    request_digest: Sha256
    slack_readback_digest: Sha256
    postgres_reliability_digest: Sha256
    outbox_recovery: OutboxRecoveryPostgresEvidence
    live_restart_digest: Sha256
    runtime_health: RuntimeHealthReadback
    dm_root_reference_binding_digest: Sha256
    dm_root_reference_run_id: NonEmptyStr
    dm_root_reference_request_ts: SlackMessageTs
    dm_root_reference_response_ts: SlackMessageTs
    post_restart_case_id: RevisedLiveCaseId
    cases: tuple[RevisedLiveCaseObservation, ...] = Field(min_length=6, max_length=6)
    case_count: Literal[6]
    max_ingress_latency_ms: int = Field(ge=0, lt=3000)
    observed_at: datetime
    digest: Sha256

    @model_validator(mode="after")
    def exact_complete_acceptance(self) -> RevisedLiveAcceptanceArtifact:
        ids = tuple(item.case_id for item in self.cases)
        post_restart = next(
            (item for item in self.cases if item.case_id is self.post_restart_case_id),
            None,
        )
        if (
            self.observed_at.tzinfo is None
            or ids != tuple(sorted(ids, key=str))
            or set(ids) != REVISED_LIVE_CASE_IDS
            or self.dm_root_reference_run_id in {item.run_id for item in self.cases}
            or float(self.dm_root_reference_response_ts) <= float(self.dm_root_reference_request_ts)
            or self.max_ingress_latency_ms != max(item.ingress_latency_ms for item in self.cases)
            or self.observed_at < max(item.delivered_at for item in self.cases)
            or post_restart is None
            or post_restart.received_at <= self.runtime_health.listener_connected_at
            or post_restart.delivered_at <= self.runtime_health.listener_connected_at
            or self.digest != _digest(self.model_dump(mode="json", exclude={"digest"}))
        ):
            raise ValueError("revised live artifact does not prove the complete exact cohort")
        return self


class DurableRevisedRun(ContractModel):
    """Internal SELECT-only projection revalidated across custom/fake sources."""

    case_id: RevisedLiveCaseId
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    parent_task_id: str | None
    continuation_kind: NonEmptyStr
    channel_id: NonEmptyStr
    request_message_ts: SlackMessageTs
    thread_root_ts: SlackMessageTs
    received_at: datetime
    conversation_kind: NonEmptyStr
    context_projection_source: NonEmptyStr
    context_conversation_ids: tuple[NonEmptyStr, ...]
    context_access_hash: Sha256
    prompt: NonEmptyStr
    task_status: NonEmptyStr
    run_status: NonEmptyStr
    terminal_reason: str | None
    final_output: str
    final_payload: NonEmptyStr
    final_payload_hash: Sha256
    final_state: NonEmptyStr
    final_attempt_count: int = Field(ge=0)
    final_receipt_message_ts: SlackMessageTs | None
    final_delivered_at: datetime
    event_types: tuple[NonEmptyStr, ...]
    tool_names: tuple[NonEmptyStr, ...]
    context_markers: tuple[NonEmptyStr, ...]
    context_projection_commitments: tuple[Sha256, ...]
    context_projection_source_counts: tuple[int, ...]
    persisted_thread_message_timestamps: tuple[SlackMessageTs, ...]
    observations: tuple[dict[str, object], ...]
    claims: tuple[dict[str, object], ...]


class DurableRevisedObservation(ContractModel):
    observed_at: datetime
    runs: tuple[DurableRevisedRun, ...] = Field(min_length=7, max_length=7)


class AsyncRevisedLiveSource(Protocol):
    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        bindings: tuple[RevisedLiveRunBinding, ...],
    ) -> DurableRevisedObservation: ...


class PostgresRevisedLiveSource:
    """SELECT-only revised acceptance source; session close rolls back reads."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        bindings: tuple[RevisedLiveRunBinding, ...],
    ) -> DurableRevisedObservation:
        runs: list[DurableRevisedRun] = []
        async with self._sessions() as session:
            for binding in bindings:
                run = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(RunRow).where(
                                    RunRow.id == binding.run_id,
                                    RunRow.organization_id == organization_id,
                                )
                            )
                        ).all()
                    )
                )
                task = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(TaskRow).where(
                                    TaskRow.id == run.task_id,
                                    TaskRow.organization_id == organization_id,
                                )
                            )
                        ).all()
                    )
                )
                ingress = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(SlackIngressEventRow).where(
                                    SlackIngressEventRow.task_id == task.id,
                                    SlackIngressEventRow.team_id == team_id,
                                    SlackIngressEventRow.organization_id == organization_id,
                                )
                            )
                        ).all()
                    )
                )
                events = tuple(
                    (
                        await session.scalars(
                            select(RunEventRow)
                            .where(
                                RunEventRow.run_id == run.id,
                                RunEventRow.task_id == task.id,
                            )
                            .order_by(RunEventRow.sequence)
                        )
                    ).all()
                )
                observations = tuple(
                    (
                        await session.scalars(
                            select(ObservationRow)
                            .where(
                                ObservationRow.run_id == run.id,
                                ObservationRow.organization_id == organization_id,
                            )
                            .order_by(ObservationRow.id)
                        )
                    ).all()
                )
                claims = tuple(
                    (
                        await session.scalars(
                            select(ClaimRow)
                            .where(
                                ClaimRow.run_id == run.id,
                                ClaimRow.organization_id == organization_id,
                            )
                            .order_by(ClaimRow.id)
                        )
                    ).all()
                )
                thread_messages = tuple(
                    (
                        await session.scalars(
                            select(SanitizedMessageRow)
                            .where(
                                SanitizedMessageRow.organization_id == organization_id,
                                SanitizedMessageRow.conversation_id == ingress.conversation_id,
                                SanitizedMessageRow.provider_thread_root_ts
                                == ingress.thread_root_ts,
                                SanitizedMessageRow.provider_message_ts.is_not(None),
                            )
                            .order_by(SanitizedMessageRow.provider_message_ts)
                        )
                    ).all()
                )
                finals = tuple(
                    (
                        await session.scalars(
                            select(DeliveryOutboxRow).where(
                                DeliveryOutboxRow.run_id == run.id,
                                DeliveryOutboxRow.task_id == task.id,
                                DeliveryOutboxRow.organization_id == organization_id,
                                DeliveryOutboxRow.kind == "final",
                            )
                        )
                    ).all()
                )
                final = _require_one(finals)
                runs.append(
                    _durable_run(
                        binding=binding,
                        run=run,
                        task=task,
                        ingress=ingress,
                        final=final,
                        events=events,
                        observations=observations,
                        claims=claims,
                        thread_messages=thread_messages,
                    )
                )
            observed_at = await session.scalar(select(func.now()))
        if observed_at is None:
            raise RevisedLiveNotFound
        return DurableRevisedObservation(observed_at=observed_at, runs=tuple(runs))


async def collect_revised_live_acceptance(
    source: AsyncRevisedLiveSource,
    *,
    request: RevisedLiveAcceptanceRequest,
    slack_readback: SlackRevisedReadback,
    runtime_health: RuntimeHealthReadback,
    postgres_reliability_digest: str,
    outbox_recovery: OutboxRecoveryPostgresEvidence,
    live_restart_digest: str,
) -> RevisedLiveAcceptanceArtifact:
    """Collect the exact revised cohort or fail without exporting an artifact."""

    request = RevisedLiveAcceptanceRequest.model_validate(request.model_dump(mode="json"))
    slack_readback = SlackRevisedReadback.model_validate(slack_readback.model_dump(mode="json"))
    runtime_health = RuntimeHealthReadback.model_validate(runtime_health.model_dump(mode="json"))
    outbox_recovery = OutboxRecoveryPostgresEvidence.model_validate(
        outbox_recovery.model_dump(mode="json")
    )
    durable_raw = await source.observe(
        organization_id=request.organization_id,
        team_id=request.team_id,
        bindings=request.bindings,
    )
    durable = DurableRevisedObservation.model_validate(durable_raw.model_dump(mode="json"))
    if durable.observed_at < slack_readback.observed_at:
        raise RevisedLiveIntegrityError("database observation predates Slack readback")

    bindings = {item.case_id: item for item in request.bindings}
    readbacks = {item.case_id: item for item in slack_readback.cases}
    runs = {item.case_id: item for item in durable.runs}
    if (
        set(bindings) != REVISED_LIVE_BINDING_IDS
        or set(runs) != REVISED_LIVE_BINDING_IDS
        or set(readbacks) != REVISED_LIVE_CASE_IDS
    ):
        raise RevisedLiveIntegrityError("revised sources do not cover the exact cohort")

    _validate_thread_pairs(bindings, readbacks, runs)
    dm_root_binding = bindings[RevisedLiveCaseId.DM_MEMORY_ROOT]
    _validate_dm_root_reference(
        dm_root_binding,
        runs[RevisedLiveCaseId.DM_MEMORY_ROOT],
    )
    cases = tuple(
        _reconcile_case(bindings[case_id], readbacks[case_id], runs[case_id])
        for case_id in sorted(REVISED_LIVE_CASE_IDS, key=str)
    )
    post_restart = next(item for item in cases if item.case_id is request.post_restart_case_id)
    if (
        post_restart.received_at <= runtime_health.listener_connected_at
        or post_restart.delivered_at <= runtime_health.listener_connected_at
    ):
        raise RevisedLiveIntegrityError("bound post-restart success predates listener connection")
    payload: dict[str, object] = {
        "version": REVISED_LIVE_ACCEPTANCE_VERSION,
        "request_digest": _digest(request.model_dump(mode="json")),
        "slack_readback_digest": _digest(slack_readback.model_dump(mode="json")),
        "postgres_reliability_digest": postgres_reliability_digest,
        "outbox_recovery": outbox_recovery,
        "live_restart_digest": live_restart_digest,
        "runtime_health": runtime_health,
        "dm_root_reference_binding_digest": dm_root_binding.digest,
        "dm_root_reference_run_id": dm_root_binding.run_id,
        "dm_root_reference_request_ts": dm_root_binding.request_message_ts,
        "dm_root_reference_response_ts": dm_root_binding.slack_response_ts,
        "post_restart_case_id": request.post_restart_case_id,
        "cases": cases,
        "case_count": len(cases),
        "max_ingress_latency_ms": max(item.ingress_latency_ms for item in cases),
        "observed_at": durable.observed_at,
    }
    return RevisedLiveAcceptanceArtifact.model_validate(
        {**payload, "digest": _digest(_json_value(payload))}
    )


def make_outbox_recovery_probe(
    *,
    case_id: OutboxRecoveryCaseId,
    initial_final_outbox_count: int,
    repair_created_count: int,
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> OutboxRecoveryProbe:
    """Build one exact test observation; pass/fail is derived by model validation."""

    node_id = {
        OutboxRecoveryCaseId.PENDING_FINAL: (
            "tests/postgres/test_outbox.py::test_pending_final_intent_is_delivered_once"
        ),
        OutboxRecoveryCaseId.MISSING_FINAL: (
            "tests/postgres/test_outbox.py::test_terminal_reconciliation_repairs_missing_final_intent"
        ),
    }[case_id]
    payload: dict[str, object] = {
        "version": OUTBOX_RECOVERY_PROBE_VERSION,
        "case_id": case_id,
        "database_label": "supabase-postgres-current-head",
        "pytest_node_id": node_id,
        "initial_final_outbox_count": initial_final_outbox_count,
        "repaired_final_outbox_count": 1,
        "repair_created_count": repair_created_count,
        "claimed_count": 1,
        "final_outbox_count": 1,
        "final_state": "delivered",
        "final_delivery_attempt_count": 1,
        "final_receipt_count": 1,
        "physical_delivery_count": 1,
        "duplicate_delivery_count": 0,
        "second_repair_count": 0,
        "second_dispatch_count": 0,
        "before_digest": _digest(dict(before)),
        "after_digest": _digest(dict(after)),
    }
    return OutboxRecoveryProbe.model_validate({**payload, "digest": _digest(_json_value(payload))})


def make_slack_revised_readback_case(
    *,
    case_id: RevisedLiveCaseId,
    run_id: str,
    channel_id: str,
    thread_root_ts: str,
    slack_response_ts: str,
    response_text_digest: str,
    thread_message_timestamps: tuple[str, ...],
) -> SlackRevisedReadbackCase:
    """Content-address one connector projection; exact matching count is fixed at one."""

    payload: dict[str, object] = {
        "case_id": case_id,
        "run_id": run_id,
        "channel_id": channel_id,
        "thread_root_ts": thread_root_ts,
        "slack_response_ts": slack_response_ts,
        "matching_response_count": 1,
        "response_text_digest": response_text_digest,
        "thread_message_timestamps": tuple(sorted(thread_message_timestamps, key=float)),
    }
    return SlackRevisedReadbackCase.model_validate(
        {**payload, "readback_digest": _digest(_json_value(payload))}
    )


def make_runtime_health_readback(
    *,
    listener_epoch_digest: str,
    listener_started_at: datetime,
    listener_connected_at: datetime,
    observed_at: datetime,
    last_success_at: datetime,
    component_states: Mapping[str, str],
) -> RuntimeHealthReadback:
    """Content-address one observed deep-health snapshot without accepting a pass flag."""

    payload: dict[str, object] = {
        "version": RUNTIME_HEALTH_READBACK_VERSION,
        "listener_epoch_digest": listener_epoch_digest,
        "listener_started_at": listener_started_at,
        "listener_connected_at": listener_connected_at,
        "observed_at": observed_at,
        "last_success_at": last_success_at,
        "component_states": dict(component_states),
    }
    return RuntimeHealthReadback.model_validate(
        {**payload, "digest": _digest(_json_value(payload))}
    )


def make_outbox_recovery_evidence(
    *,
    alembic_head: str,
    probes: tuple[OutboxRecoveryProbe, ...],
) -> OutboxRecoveryPostgresEvidence:
    ordered = tuple(sorted(probes, key=lambda item: str(item.case_id)))
    payload: dict[str, object] = {
        "version": OUTBOX_RECOVERY_EVIDENCE_VERSION,
        "alembic_head": alembic_head,
        "probes": ordered,
        "case_count": len(ordered),
    }
    return OutboxRecoveryPostgresEvidence.model_validate(
        {**payload, "digest": _digest(_json_value(payload))}
    )


def export_contract(model: ContractModel, destination: Path) -> None:
    """Atomically export one already validated evidence contract."""

    _atomic_write(destination, model.model_dump_json(indent=2) + "\n")


def _durable_run(
    *,
    binding: RevisedLiveRunBinding,
    run: RunRow,
    task: TaskRow,
    ingress: SlackIngressEventRow,
    final: DeliveryOutboxRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    thread_messages: tuple[SanitizedMessageRow, ...],
) -> DurableRevisedRun:
    context_markers: set[str] = set()
    context_projection_commitments: set[str] = set()
    context_projection_source_counts: set[int] = set()
    tool_names: list[str] = []
    for event in events:
        if event.type == "tool_started":
            tool = event.payload.get("tool")
            if isinstance(tool, str) and tool:
                tool_names.append(tool)
        if event.type == "context_built":
            projection = event.payload.get("projection")
            if (
                isinstance(projection, dict)
                and projection.get("version") == "context-built-v1"
                and isinstance(projection.get("source_ids_sha256"), str)
                and isinstance(projection.get("source_id_count"), int)
            ):
                context_projection_commitments.add(str(projection["source_ids_sha256"]))
                context_projection_source_counts.add(int(projection["source_id_count"]))
            manifest = event.payload.get("source_manifest")
            if isinstance(manifest, dict):
                values = manifest.get("included_source_ids")
                if isinstance(values, list):
                    context_markers.update(
                        item
                        for item in values
                        if isinstance(item, str) and item.startswith("slack-thread-")
                    )
    observation_payloads = tuple(
        {
            "id": item.id,
            "kind": item.kind,
            "status": item.status,
            "quality": item.quality,
            "provider": item.source.get("provider"),
        }
        for item in observations
    )
    claim_payloads = tuple(
        {
            "id": item.id,
            "kind": item.kind,
            "observation_ids": tuple(item.observation_ids),
        }
        for item in claims
    )
    persisted_thread_timestamps = tuple(
        sorted(
            {
                item.provider_message_ts
                for item in thread_messages
                if item.provider_message_ts is not None
                and float(item.provider_message_ts) <= float(binding.request_message_ts)
            },
            key=float,
        )
    )
    return DurableRevisedRun(
        case_id=binding.case_id,
        run_id=run.id,
        task_id=task.id,
        parent_task_id=task.parent_task_id,
        continuation_kind=task.continuation_kind,
        channel_id=ingress.channel_id,
        request_message_ts=ingress.message_ts,
        thread_root_ts=ingress.thread_root_ts,
        received_at=ingress.received_at,
        conversation_kind=ingress.conversation_kind,
        context_projection_source=ingress.context_projection_source,
        context_conversation_ids=tuple(sorted(ingress.context_conversation_ids)),
        context_access_hash=ingress.context_access_hash,
        prompt=ingress.prompt,
        task_status=task.status,
        run_status=run.status,
        terminal_reason=run.terminal_reason,
        final_output=run.final_output or "",
        final_payload=final.payload,
        final_payload_hash=final.payload_hash,
        final_state=final.state,
        final_attempt_count=final.attempt_count,
        final_receipt_message_ts=final.receipt_message_ts,
        final_delivered_at=final.updated_at,
        event_types=tuple(item.type for item in events),
        tool_names=tuple(tool_names),
        context_markers=tuple(sorted(context_markers)),
        context_projection_commitments=tuple(sorted(context_projection_commitments)),
        context_projection_source_counts=tuple(sorted(context_projection_source_counts)),
        persisted_thread_message_timestamps=persisted_thread_timestamps,
        observations=observation_payloads,
        claims=claim_payloads,
    )


def _validate_thread_pairs(
    bindings: Mapping[RevisedLiveCaseId, RevisedLiveRunBinding],
    readbacks: Mapping[RevisedLiveCaseId, SlackRevisedReadbackCase],
    runs: Mapping[RevisedLiveCaseId, DurableRevisedRun],
) -> None:
    pairs = (
        (RevisedLiveCaseId.MPIM_CLARIFICATION, RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP),
        (RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP),
    )
    for root_id, followup_id in pairs:
        root_binding = bindings[root_id]
        followup_binding = bindings[followup_id]
        root_run = runs[root_id]
        followup_run = runs[followup_id]
        followup_readback = readbacks[followup_id]
        required_timestamps = {
            root_binding.request_message_ts,
            root_binding.slack_response_ts,
            followup_binding.request_message_ts,
            followup_binding.slack_response_ts,
        }
        readback_before_followup = {
            item
            for item in followup_readback.thread_message_timestamps
            if float(item) <= float(followup_binding.request_message_ts)
        }
        if (
            root_binding.channel_id != followup_binding.channel_id
            or root_binding.thread_root_ts != followup_binding.thread_root_ts
            or float(root_binding.slack_response_ts) >= float(followup_binding.request_message_ts)
            or root_run.task_id != followup_run.parent_task_id
            or followup_run.continuation_kind != "follow_up"
            or not required_timestamps.issubset(followup_readback.thread_message_timestamps)
            or readback_before_followup != set(followup_run.persisted_thread_message_timestamps)
        ):
            raise RevisedLiveIntegrityError("revised follow-up is not bound to its complete thread")


def _validate_dm_root_reference(
    binding: RevisedLiveRunBinding,
    run: DurableRevisedRun,
) -> None:
    """Reference the fixed-nine DM case without counting it as revised acceptance."""

    if (
        run.run_id != binding.run_id
        or run.channel_id != binding.channel_id
        or run.request_message_ts != binding.request_message_ts
        or run.thread_root_ts != binding.thread_root_ts
        or run.context_conversation_ids != binding.expected_context_conversation_ids
        or run.final_state != "delivered"
        or run.final_attempt_count != 1
        or run.final_receipt_message_ts != binding.slack_response_ts
        or set(_semantic_markers(RevisedLiveCaseId.DM_MEMORY_ROOT, run))
        != {"dm_membership_projection", "memory_grounded"}
    ):
        raise RevisedLiveIntegrityError("DM root reference diverges from its fixed-nine binding")


def _reconcile_case(
    binding: RevisedLiveRunBinding,
    readback: SlackRevisedReadbackCase,
    run: DurableRevisedRun,
) -> RevisedLiveCaseObservation:
    final_digest = hashlib.sha256(run.final_payload.encode("utf-8")).hexdigest()
    if (
        run.run_id != binding.run_id
        or run.channel_id != binding.channel_id
        or run.request_message_ts != binding.request_message_ts
        or run.thread_root_ts != binding.thread_root_ts
        or run.context_conversation_ids != binding.expected_context_conversation_ids
        or readback.run_id != binding.run_id
        or readback.channel_id != binding.channel_id
        or readback.thread_root_ts != binding.thread_root_ts
        or readback.slack_response_ts != binding.slack_response_ts
        or binding.request_message_ts not in readback.thread_message_timestamps
        or run.final_state != "delivered"
        or run.final_attempt_count != 1
        or run.final_receipt_message_ts != binding.slack_response_ts
        or run.final_payload_hash != final_digest
        or readback.response_text_digest != final_digest
    ):
        raise RevisedLiveIntegrityError("revised case diverges across its exact bindings")
    latency_ms = round((run.received_at.timestamp() - float(run.request_message_ts)) * 1000)
    if latency_ms < 0 or latency_ms >= 3000:
        raise RevisedLiveIntegrityError(
            "live ingress receipt did not meet the sub-three-second SLO"
        )

    markers = _semantic_markers(binding.case_id, run)
    observation_digests = tuple(sorted(_digest(item) for item in run.observations))
    payload: dict[str, object] = {
        "case_id": binding.case_id,
        "binding_digest": binding.digest,
        "run_id": run.run_id,
        "task_id": run.task_id,
        "channel_id": run.channel_id,
        "request_message_ts": run.request_message_ts,
        "thread_root_ts": run.thread_root_ts,
        "slack_response_ts": binding.slack_response_ts,
        "received_at": run.received_at,
        "delivered_at": run.final_delivered_at,
        "ingress_latency_ms": latency_ms,
        "conversation_kind": run.conversation_kind,
        "context_projection_source": run.context_projection_source,
        "context_conversation_ids": run.context_conversation_ids,
        "context_access_hash": run.context_access_hash,
        "task_status": run.task_status,
        "run_status": run.run_status,
        "terminal_reason": run.terminal_reason or "absent",
        "final_output_digest": hashlib.sha256(run.final_output.encode("utf-8")).hexdigest(),
        "final_payload_digest": final_digest,
        "final_payload_utf8_bytes": len(run.final_payload.encode("utf-8")),
        "event_sequence_digest": _digest(run.event_types),
        "context_manifest_digest": _digest(run.context_markers),
        "semantic_markers": tuple(sorted(markers)),
        "observation_kind_digests": observation_digests,
        "claim_count": len(run.claims),
        "slack_readback_digest": readback.readback_digest,
    }
    return RevisedLiveCaseObservation.model_validate(
        {**payload, "snapshot_digest": _digest(_json_value(payload))}
    )


def _semantic_markers(case_id: RevisedLiveCaseId, run: DurableRevisedRun) -> tuple[str, ...]:
    if case_id is RevisedLiveCaseId.TERMINAL_RECOVERY:
        bare = {
            "failed",
            "budget_exhausted",
            "budget exhausted",
            "iteration_budget_exhausted",
        }
        lowered = run.final_payload.strip().lower()
        unsafe_tokens = ("traceback", "stack trace", "api_key", "password=")
        if (
            run.task_status != "failed"
            or run.run_status != "budget_exhausted"
            or run.terminal_reason != "iteration_budget_exhausted"
            or run.final_output
            or len(run.final_payload.encode("utf-8")) < 80
            or lowered in bare
            or any(item in lowered for item in unsafe_tokens)
            or "budget_exhausted" in lowered
            or "iteration_budget_exhausted" in lowered
            or "budget_exhausted" not in run.event_types
        ):
            raise RevisedLiveIntegrityError("terminal recovery is bare, unsafe, or not durable")
        return ("conversational_safe_terminal", "durable_budget_exhausted")

    if (
        run.task_status != "completed"
        or run.run_status != "completed"
        or run.terminal_reason != "verified_completion"
        or "verification_passed" not in run.event_types
        or "run_completed" not in run.event_types
        or not run.final_output
    ):
        raise RevisedLiveIntegrityError("revised success lacks verified durable completion")

    if case_id is RevisedLiveCaseId.MPIM_CLARIFICATION:
        lowered = run.final_output.lower()
        if (
            run.conversation_kind != "mpim"
            or run.context_projection_source != "exact_destination"
            or len(run.context_conversation_ids) != 1
            or run.tool_names
            or "?" not in run.final_output
            or not any(item in lowered for item in ("could you", "which", "what", "do you"))
        ):
            raise RevisedLiveIntegrityError(
                "short MPIM prompt did not produce an elastic clarification"
            )
        return ("elastic_short_clarification", "mpim_singleton_isolation")

    if case_id is RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP:
        _require_complete_thread(run)
        if (
            run.conversation_kind != "mpim"
            or run.context_projection_source != "exact_destination"
            or len(run.context_conversation_ids) != 1
        ):
            raise RevisedLiveIntegrityError("MPIM follow-up lost singleton isolation")
        return ("complete_thread_context", "mpim_singleton_isolation", "thread_follow_up")

    if case_id in {RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP}:
        if (
            run.conversation_kind != "dm"
            or run.context_projection_source != "dm_membership_intersection"
            or len(run.context_conversation_ids) < 2
            or not _has_grounded_observation(run, prefixes=("memory.",), provider="leo_memory")
        ):
            raise RevisedLiveIntegrityError(
                "DM case lacks exact membership projection or grounding"
            )
        markers = ["dm_membership_projection", "memory_grounded"]
        if case_id is RevisedLiveCaseId.DM_THREAD_FOLLOWUP:
            _require_complete_thread(run)
            markers.extend(("complete_thread_context", "thread_follow_up"))
        return tuple(markers)

    if case_id is RevisedLiveCaseId.TAVILY_RESEARCH:
        if (
            not _is_natural_prompt(run.prompt)
            or not _ordered_subsequence(
                run.tool_names,
                ("web.search_tavily", "web.fetch_public_text"),
            )
            or not _has_observation(
                run,
                kind="web.search_tavily",
                status="rejected",
                quality="discovery_only",
                provider="tavily",
            )
            or not _has_grounded_observation(
                run,
                prefixes=("web.fetch_public_text",),
                provider="web",
            )
        ):
            raise RevisedLiveIntegrityError("Tavily case lacks search-fetch-verified progression")
        return (
            "natural_language_prompt",
            "selected_public_fetch",
            "tavily_search_discovery",
            "verified_source_claim",
        )

    if case_id is RevisedLiveCaseId.FINNHUB_EARNINGS:
        if (
            not _is_natural_prompt(run.prompt)
            or "market.get_earnings_surprises" not in run.tool_names
            or not _has_grounded_observation(
                run,
                prefixes=("market.get_earnings_surprises",),
                provider="finnhub",
            )
        ):
            raise RevisedLiveIntegrityError("Finnhub case lacks natural expanded-tool grounding")
        return (
            "expanded_finnhub_earnings",
            "natural_language_prompt",
            "verified_source_claim",
        )
    raise RevisedLiveIntegrityError("unknown revised case")


def _require_complete_thread(run: DurableRevisedRun) -> None:
    direct_manifest = (
        "slack-thread-complete:true" in run.context_markers
        and "slack-thread-compacted-count:0" in run.context_markers
        and any(
            item in run.context_markers
            for item in (
                "slack-thread-source:slack_replies_bot",
                "slack-thread-source:slack_replies_user",
                "slack-thread-source:persisted_coverage",
            )
        )
    )
    if direct_manifest:
        protected = _marker_integer(run.context_markers, "slack-thread-protected-count:")
        if protected >= 2:
            return
    projected_manifest = (
        bool(run.context_projection_commitments)
        and bool(run.context_projection_source_counts)
        and len(run.persisted_thread_message_timestamps) >= 2
        and min(run.context_projection_source_counts)
        >= len(run.persisted_thread_message_timestamps)
    )
    if not projected_manifest:
        raise RevisedLiveIntegrityError("thread context is not complete and uncompacted")


def _has_observation(
    run: DurableRevisedRun,
    *,
    kind: str,
    status: str,
    quality: str,
    provider: str,
) -> bool:
    return any(
        item.get("kind") == kind
        and item.get("status") == status
        and item.get("quality") == quality
        and item.get("provider") == provider
        for item in run.observations
    )


def _has_grounded_observation(
    run: DurableRevisedRun,
    *,
    prefixes: tuple[str, ...],
    provider: str,
) -> bool:
    eligible_ids = {
        str(item["id"])
        for item in run.observations
        if isinstance(item.get("kind"), str)
        and str(item["kind"]).startswith(prefixes)
        and item.get("provider") == provider
        and item.get("status") == "retrieved"
    }
    return bool(eligible_ids) and any(
        eligible_ids.intersection(_claim_observation_ids(claim)) for claim in run.claims
    )


def _claim_observation_ids(claim: Mapping[str, object]) -> tuple[str, ...]:
    values = claim.get("observation_ids")
    if not isinstance(values, (tuple, list)):
        return ()
    return tuple(str(item) for item in values)


def _is_natural_prompt(prompt: str) -> bool:
    lowered = " ".join(prompt.lower().split())
    return (
        len(prompt.encode("utf-8")) <= 512
        and not lowered.startswith("perform ")
        and "step workflow" not in lowered
        and "tavily" not in lowered
        and "finnhub" not in lowered
        and "call the tool" not in lowered
        and "use the tool" not in lowered
    )


def _ordered_subsequence(values: Sequence[str], expected: tuple[str, ...]) -> bool:
    iterator = iter(values)
    return all(any(candidate == item for candidate in iterator) for item in expected)


def _marker_integer(markers: Sequence[str], prefix: str) -> int:
    values = {
        int(item.removeprefix(prefix))
        for item in markers
        if item.startswith(prefix) and item.removeprefix(prefix).isdigit()
    }
    if len(values) != 1:
        raise RevisedLiveIntegrityError("thread marker is absent or ambiguous")
    return values.pop()


def _require_one[T](items: Sequence[T]) -> T:
    if len(items) != 1:
        raise RevisedLiveNotFound
    return items[0]


def _json_value(value: object) -> object:
    if isinstance(value, ContractModel):
        return value.model_dump(mode="json")
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return _DATETIME_ADAPTER.dump_python(value, mode="json")
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_value(value),
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
