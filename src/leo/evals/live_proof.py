"""Read-only Slack/Postgres reconciliation for trusted milestone-five live proof.

The collector deliberately has no write path and no Slack client.  Its authority,
freshness boundary, and exact Slack-message/run pairs are constructor supplied by a
trusted operator composition.  A missing or inconsistent durable row fails the whole
collection; request fixtures can never attest their own success.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal, Protocol, cast

from pydantic import Field, JsonValue, model_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.evals.proof import ProofManifest, make_proof_artifact, validate_proof_manifest
from leo.evals.recordings import sanitize_payload
from leo.harness.child_evidence import ChildEvidenceError, parse_child_evidence_envelope
from leo.harness.events import normalize_run_timeline
from leo.harness.models import (
    LEGAL_TASK_RUN_PAIRS,
    ContractModel,
    EventType,
    NonEmptyStr,
    RunEvent,
    RunStatus,
    ScopeKey,
    TaskStatus,
)
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    PlanStatus,
)
from leo.memory.navigation import ProgressiveMemorySearchResult
from leo.persistence.schema import (
    ClaimRow,
    ConversationAccessSnapshotRow,
    ConversationActorMembershipRow,
    ConversationRow,
    DelegationRow,
    DeliveryOutboxRow,
    MemoryRecordRow,
    MemoryRevisionRow,
    MemorySourceRow,
    ObservationRow,
    PlanNodeRow,
    PlanRevisionRow,
    PlanRow,
    RunEventRow,
    RunRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
    ThreadRow,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SlackMessageTs = Annotated[str, Field(pattern=r"^[0-9]{10,}\.[0-9]{6}$")]
RunId = Annotated[
    str,
    Field(
        pattern=(
            r"^run-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
            r"[0-9a-f]{4}-[0-9a-f]{12}$"
        )
    ),
]

LIVE_PROOF_ARTIFACT_ID = "live_slack_supabase_reconciliation"
LIVE_PROOF_PROFILE: Literal["m5-live-v1"] = "m5-live-v1"
_EMPTY_MEMORY_SCOPE_INFERENCE = (
    "No matching authorized memory was found in this conversation scope."
)


class LiveEvidenceId(StrEnum):
    MEMORY_WRITE = "memory_write"
    MEMORY_RECALL = "memory_recall"
    DM_MEMBERSHIP_UNION = "dm_membership_union"
    CROSS_CHANNEL_NEGATIVE = "cross_channel_negative"
    QUOTE = "quote"
    SEC = "sec"
    PRIVATE_CHANNEL = "private_channel"
    GROUP_DM = "group_dm"
    DELEGATED_REPLANNING = "delegated_replanning"


M5_LIVE_EVIDENCE_IDS = tuple(LiveEvidenceId)


class LiveProofNotFound(LookupError):
    """Fail-closed public error for absent, stale, or unauthorized evidence."""


class LiveProofIntegrityError(ValueError):
    """The selected rows existed but did not form one valid durable trace."""


class LiveProofBinding(ContractModel):
    evidence_id: LiveEvidenceId
    message_ts: SlackMessageTs
    run_id: RunId
    expected_destination_id: NonEmptyStr
    expected_conversation_kind: Literal["ordinary_internal", "dm", "mpim", "shared", "external"]
    expected_context_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    expected_context_access_hash: Sha256
    expected_recall_source_conversation_id: str | None = None
    plan_expectation: Literal["required", "forbidden", "optional"] = "optional"
    expected_response_ts: SlackMessageTs | None = None

    @model_validator(mode="after")
    def exact_destination_and_memory_semantics(self) -> LiveProofBinding:
        normalized = tuple(sorted(set(self.expected_context_conversation_ids)))
        if normalized != self.expected_context_conversation_ids:
            raise ValueError("live proof context conversations must be sorted and unique")
        if self.expected_destination_id not in normalized:
            raise ValueError("live proof destination is absent from its context projection")
        if self.expected_conversation_kind != "dm" and normalized != (
            self.expected_destination_id,
        ):
            raise ValueError("non-DM live proof context must be exact-destination only")
        if self.expected_recall_source_conversation_id is not None and (
            self.expected_recall_source_conversation_id not in normalized
        ):
            raise ValueError("live proof recall source is outside the context projection")
        if self.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION:
            if (
                self.expected_conversation_kind != "dm"
                or len(normalized) < 2
                or self.expected_recall_source_conversation_id is None
                or self.expected_recall_source_conversation_id == self.expected_destination_id
            ):
                raise ValueError(
                    "DM membership-union proof requires a non-DM positive recall source"
                )
        elif self.evidence_id is LiveEvidenceId.GROUP_DM:
            if self.expected_conversation_kind != "mpim" or normalized != (
                self.expected_destination_id,
            ):
                raise ValueError("group-DM proof must remain exact-destination isolated")
        elif self.evidence_id is LiveEvidenceId.MEMORY_RECALL:
            if self.expected_recall_source_conversation_id != self.expected_destination_id:
                raise ValueError("same-channel memory recall must cite its exact destination")
        if self.evidence_id is LiveEvidenceId.DELEGATED_REPLANNING:
            if self.plan_expectation != "required":
                raise ValueError("delegated/replanning proof requires a durable plan")
        elif self.plan_expectation != "forbidden":
            raise ValueError("non-delegated live proof cases must forbid durable plans")
        return self

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class LiveProofRequest(ContractModel):
    """Exact M5 cohort; completed bindings and pending IDs must partition it."""

    profile: Literal["m5-live-v1"] = LIVE_PROOF_PROFILE
    bindings: tuple[LiveProofBinding, ...] = ()
    pending_evidence_ids: tuple[LiveEvidenceId, ...] = ()

    @model_validator(mode="after")
    def exact_partition(self) -> LiveProofRequest:
        binding_ids = tuple(item.evidence_id for item in self.bindings)
        if binding_ids != tuple(sorted(binding_ids, key=str)) or len(binding_ids) != len(
            set(binding_ids)
        ):
            raise ValueError("live proof bindings must be sorted and unique")
        if self.pending_evidence_ids != tuple(sorted(set(self.pending_evidence_ids), key=str)):
            raise ValueError("pending live proof IDs must be sorted and unique")
        if set(binding_ids) & set(self.pending_evidence_ids):
            raise ValueError("collected and pending live proof IDs must be disjoint")
        if set(binding_ids) | set(self.pending_evidence_ids) != set(M5_LIVE_EVIDENCE_IDS):
            raise ValueError("live proof request must exactly partition the M5 live cohort")
        return self


class LiveProofAuthority(ContractModel):
    """Server/operator-derived scope and post-restore boundary, never model supplied."""

    organization_id: NonEmptyStr
    team_id: NonEmptyStr
    actor_id: NonEmptyStr
    not_before_received_at: datetime
    not_before_message_ts: SlackMessageTs
    allowed_bindings: tuple[LiveProofBinding, ...] = ()

    @model_validator(mode="after")
    def exact_fresh_authority(self) -> LiveProofAuthority:
        if self.not_before_received_at.tzinfo is None:
            raise ValueError("live proof restore boundary must be timezone-aware")
        ids = tuple(item.evidence_id for item in self.allowed_bindings)
        if ids != tuple(sorted(ids, key=str)) or len(ids) != len(set(ids)):
            raise ValueError("live proof authority bindings must be sorted and unique")
        if any(
            _slack_ts_value(item.message_ts) <= _slack_ts_value(self.not_before_message_ts)
            for item in self.allowed_bindings
        ):
            raise ValueError("live proof binding predates the post-restore Slack boundary")
        return self

    @property
    def access_digest(self) -> str:
        return _digest(self.model_dump(mode="json"))

    def permits(self, binding: LiveProofBinding) -> bool:
        return binding in self.allowed_bindings


class LiveProofCase(ContractModel):
    """Content-free reconciliation summary safe to embed in a proof manifest."""

    evidence_id: LiveEvidenceId
    binding_digest: Sha256
    message_ts: SlackMessageTs
    run_id: RunId
    task_id: NonEmptyStr
    channel_id: NonEmptyStr
    conversation_kind: NonEmptyStr
    slack_response_ts: SlackMessageTs
    objective_digest: Sha256
    final_output_digest: Sha256
    task_terminal_state: Literal["completed"]
    run_terminal_state: Literal["completed"]
    event_terminal_state: Literal["run_completed"]
    event_count: int = Field(ge=1)
    last_event_sequence: int = Field(ge=1)
    event_timeline_digest: Sha256
    context_access_hash: Sha256
    context_projection_source: NonEmptyStr
    context_conversation_count: int = Field(ge=1)
    conversation_source_set_digest: Sha256
    context_snapshot_digest: Sha256
    current_membership_count: int = Field(ge=0)
    current_membership_digest: Sha256
    context_manifest_digest: Sha256
    message_plane_digest: Sha256
    memory_recall_verified: bool
    memory_recall_source_digest: Sha256
    memory_recall_observation_digest: Sha256
    grounded_memory_claim_digest: Sha256
    case_invariant_digest: Sha256
    observed_evidence_count: int = Field(ge=0)
    grounded_claim_count: int = Field(ge=0)
    memory_mutation_record_count: int = Field(ge=0)
    memory_mutation_revision_count: int = Field(ge=0)
    memory_mutation_source_count: int = Field(ge=0)
    memory_mutation_digest: Sha256
    delegated_child_count: int = Field(ge=0)
    delegated_overlap_verified: bool
    delegated_evidence_digest: Sha256
    plan_present: bool
    plan_terminal_state: Literal["absent", "completed"]
    plan_revision_digest: Sha256
    plan_snapshot_digest: Sha256
    plan_node_count: int = Field(ge=0)
    delegation_count: int = Field(ge=0)
    delivery_state: Literal["delivered"]
    outbox_count: int = Field(ge=1)
    outbox_digest: Sha256
    ingress_digest: Sha256
    task_run_digest: Sha256
    row_snapshot_digest: Sha256

    @model_validator(mode="after")
    def plan_shape(self) -> LiveProofCase:
        if self.plan_present != (self.plan_terminal_state == "completed"):
            raise ValueError("live proof plan presence and terminal state diverge")
        if not self.plan_present and (self.plan_node_count or self.delegation_count):
            raise ValueError("absent live proof plan cannot have nodes or delegations")
        recall_required = self.evidence_id in {
            LiveEvidenceId.MEMORY_RECALL,
            LiveEvidenceId.DM_MEMBERSHIP_UNION,
        }
        empty_digest = _digest([])
        if self.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION:
            if (
                self.current_membership_count != self.context_conversation_count
                or self.current_membership_count < 2
                or self.current_membership_digest == empty_digest
            ):
                raise ValueError("DM-union proof lacks its exact current membership projection")
        elif self.current_membership_count or self.current_membership_digest != empty_digest:
            raise ValueError("non-DM-union proof cannot claim a membership projection")
        if self.plan_present:
            if (
                self.plan_revision_digest == empty_digest
                or self.plan_snapshot_digest == empty_digest
            ):
                raise ValueError("present live proof plan requires durable digests")
        elif self.plan_revision_digest != empty_digest or self.plan_snapshot_digest != empty_digest:
            raise ValueError("absent live proof plan cannot claim durable digests")
        if self.evidence_id is LiveEvidenceId.DELEGATED_REPLANNING:
            if not self.plan_present or self.plan_node_count < 2 or self.delegation_count < 2:
                raise ValueError("delegated live proof requires its durable parent plan")
        elif self.plan_present:
            raise ValueError("non-delegated live proof cannot claim a plan")
        if self.evidence_id is LiveEvidenceId.MEMORY_WRITE:
            expected_observations = 1
            claims_required = False
        elif self.evidence_id in {
            LiveEvidenceId.PRIVATE_CHANNEL,
            LiveEvidenceId.GROUP_DM,
        }:
            expected_observations = 0
            claims_required = False
        else:
            expected_observations = 1
            claims_required = True
        claims_invalid = (
            self.grounded_claim_count < 1 if claims_required else self.grounded_claim_count != 0
        )
        if self.observed_evidence_count != expected_observations or claims_invalid:
            raise ValueError("live proof case evidence counts do not match its contract")
        if self.memory_recall_verified != recall_required:
            raise ValueError("live proof memory recall state does not match its evidence case")
        recall_digests = (
            self.memory_recall_source_digest,
            self.memory_recall_observation_digest,
            self.grounded_memory_claim_digest,
        )
        if recall_required and any(item == empty_digest for item in recall_digests):
            raise ValueError("verified live proof memory recall requires grounded digests")
        if not recall_required and any(item != empty_digest for item in recall_digests):
            raise ValueError("non-memory live proof case cannot claim a recall")
        if self.case_invariant_digest == empty_digest:
            raise ValueError("live proof case lacks an observed invariant digest")
        if self.evidence_id is LiveEvidenceId.MEMORY_WRITE:
            if (
                self.memory_mutation_record_count != 1
                or self.memory_mutation_revision_count != 1
                or self.memory_mutation_source_count != 3
                or self.memory_mutation_digest == empty_digest
            ):
                raise ValueError("memory-write proof lacks one exact durable mutation")
        elif (
            self.memory_mutation_record_count
            or self.memory_mutation_revision_count
            or self.memory_mutation_source_count
            or self.memory_mutation_digest != empty_digest
        ):
            raise ValueError("non-write live proof case cannot claim a memory mutation")
        if self.evidence_id is LiveEvidenceId.DELEGATED_REPLANNING:
            if (
                self.delegated_child_count < 2
                or self.delegated_child_count != self.plan_node_count
                or not self.delegated_overlap_verified
                or self.delegated_evidence_digest == empty_digest
            ):
                raise ValueError("delegated live proof lacks parallel verified child evidence")
        elif (
            self.delegated_child_count
            or self.delegated_overlap_verified
            or self.delegated_evidence_digest != empty_digest
        ):
            raise ValueError("non-delegated live proof case cannot claim child evidence")
        expected = _digest(self.model_dump(mode="json", exclude={"row_snapshot_digest"}))
        if self.row_snapshot_digest != expected:
            raise ValueError("live proof row snapshot digest mismatch")
        return self


class LiveProofCollection(ContractModel):
    version: Literal["live-proof-v1"] = "live-proof-v1"
    profile: Literal["m5-live-v1"] = LIVE_PROOF_PROFILE
    authority_digest: Sha256
    cases: tuple[LiveProofCase, ...] = ()
    pending_evidence_ids: tuple[LiveEvidenceId, ...] = ()
    status: Literal["partial", "complete"]
    digest: Sha256

    @model_validator(mode="after")
    def reconcile(self) -> LiveProofCollection:
        case_ids = tuple(item.evidence_id for item in self.cases)
        if case_ids != tuple(sorted(case_ids, key=str)) or len(case_ids) != len(set(case_ids)):
            raise ValueError("live proof cases must be sorted and unique")
        if self.pending_evidence_ids != tuple(sorted(set(self.pending_evidence_ids), key=str)):
            raise ValueError("pending live proof IDs must be sorted and unique")
        if set(case_ids) & set(self.pending_evidence_ids):
            raise ValueError("live proof cases and pending IDs overlap")
        if set(case_ids) | set(self.pending_evidence_ids) != set(M5_LIVE_EVIDENCE_IDS):
            raise ValueError("live proof collection does not cover the exact M5 cohort")
        expected_status = "complete" if not self.pending_evidence_ids else "partial"
        if self.status != expected_status:
            raise ValueError("live proof collection status does not reconcile")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("live proof collection digest mismatch")
        return self


class AsyncLiveProofSource(Protocol):
    async def load(
        self,
        *,
        authority: LiveProofAuthority,
        binding: LiveProofBinding,
    ) -> LiveProofCase: ...


async def collect_live_proof(
    source: AsyncLiveProofSource,
    *,
    authority: LiveProofAuthority,
    request: LiveProofRequest,
) -> LiveProofCollection:
    """Collect every admitted binding or fail without emitting a partial self-attestation."""

    if request.bindings != authority.allowed_bindings:
        raise LiveProofNotFound
    cases: list[LiveProofCase] = []
    for binding in request.bindings:
        observed = await source.load(authority=authority, binding=binding)
        # Revalidate serialized state so a fake/custom source cannot bypass model
        # invariants with ``model_construct`` and attest its own row digest.
        case = LiveProofCase.model_validate(observed.model_dump(mode="json"))
        if (
            case.evidence_id is not binding.evidence_id
            or case.message_ts != binding.message_ts
            or case.run_id != binding.run_id
            or case.binding_digest != binding.digest
            or case.conversation_kind != binding.expected_conversation_kind
            or (
                binding.expected_response_ts is not None
                and case.slack_response_ts != binding.expected_response_ts
            )
        ):
            raise LiveProofIntegrityError("live proof source returned an unmatched trace")
        cases.append(case)
    payload: dict[str, object] = {
        "version": "live-proof-v1",
        "profile": request.profile,
        "authority_digest": authority.access_digest,
        "cases": [item.model_dump(mode="json") for item in cases],
        "pending_evidence_ids": [str(item) for item in request.pending_evidence_ids],
        "status": "complete" if not request.pending_evidence_ids else "partial",
    }
    return LiveProofCollection.model_validate({**payload, "digest": _digest(payload)})


def attach_live_collection(
    manifest: ProofManifest,
    collection: LiveProofCollection,
) -> ProofManifest:
    """Add/replace the sanitized live artifact without weakening offline proof requirements."""

    if not collection.cases:
        raise ValueError("cannot attach an empty live proof collection")
    case_payloads: list[JsonValue] = [
        cast(JsonValue, item.model_dump(mode="json")) for item in collection.cases
    ]
    artifact = make_proof_artifact(
        artifact_id=LIVE_PROOF_ARTIFACT_ID,
        kind=(
            "live_slack_supabase_reconciliation"
            if collection.status == "complete"
            else "live_slack_supabase_reconciliation_partial"
        ),
        command="python -m leo.evals.live_proof_operator",
        scenario_ids=tuple(str(item.evidence_id) for item in collection.cases),
        fixture_digests=tuple(item.binding_digest for item in collection.cases),
        provider_label="slack-supabase-live",
        sanitized_run_ids=tuple(item.run_id for item in collection.cases),
        metadata={
            "authority_digest": collection.authority_digest,
            "case_summaries": case_payloads,
            "collection_digest": collection.digest,
            "pending_evidence_ids": [str(item) for item in collection.pending_evidence_ids],
            "profile": collection.profile,
            "status": collection.status,
            "version": collection.version,
        },
    )
    artifacts = (
        *(item for item in manifest.artifacts if item.id != LIVE_PROOF_ARTIFACT_ID),
        artifact,
    )
    output = manifest.model_copy(update={"artifacts": artifacts})
    validate_proof_manifest(output)
    return output


def require_complete_live_proof(manifest: ProofManifest) -> None:
    artifacts = [item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID]
    if len(artifacts) != 1 or artifacts[0].metadata.get("status") != "complete":
        raise ValueError("proof manifest lacks complete live Slack/Supabase reconciliation")
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
    if (
        artifact.kind != "live_slack_supabase_reconciliation"
        or artifact.provider_label != "slack-supabase-live"
        or artifact.command != "python -m leo.evals.live_proof_operator"
        or artifact.scenario_ids != tuple(str(item.evidence_id) for item in collection.cases)
        or artifact.fixture_digests != tuple(item.binding_digest for item in collection.cases)
        or artifact.sanitized_run_ids != tuple(item.run_id for item in collection.cases)
    ):
        raise ValueError("live proof artifact diverges from its reconciled collection")
    if set(artifact.scenario_ids) != {str(item) for item in M5_LIVE_EVIDENCE_IDS}:
        raise ValueError("live proof artifact does not cover the exact M5 cohort")
    if collection.pending_evidence_ids:
        raise ValueError("live proof artifact still has pending evidence")


class PostgresLiveProofSource:
    """SELECT-only durable source; the session is always closed without commit."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def load(
        self,
        *,
        authority: LiveProofAuthority,
        binding: LiveProofBinding,
    ) -> LiveProofCase:
        if not authority.permits(binding):
            raise LiveProofNotFound
        async with self._sessions() as session:
            ingress = _require_one(
                (
                    await session.scalars(
                        select(SlackIngressEventRow).where(
                            SlackIngressEventRow.team_id == authority.team_id,
                            SlackIngressEventRow.organization_id == authority.organization_id,
                            SlackIngressEventRow.message_ts == binding.message_ts,
                        )
                    )
                ).all()
            )
            if (
                ingress.received_at < authority.not_before_received_at
                or _slack_ts_value(ingress.message_ts)
                <= _slack_ts_value(authority.not_before_message_ts)
                or ingress.task_id is None
                or ingress.launch_status != "queued"
            ):
                raise LiveProofNotFound
            task = await session.scalar(
                select(TaskRow).where(
                    TaskRow.id == ingress.task_id,
                    TaskRow.organization_id == authority.organization_id,
                )
            )
            run = await session.scalar(
                select(RunRow).where(
                    RunRow.id == binding.run_id,
                    RunRow.task_id == ingress.task_id,
                    RunRow.organization_id == authority.organization_id,
                )
            )
            if task is None or run is None:
                raise LiveProofNotFound
            thread = await session.scalar(
                select(ThreadRow).where(
                    ThreadRow.id == task.thread_id,
                    ThreadRow.organization_id == authority.organization_id,
                )
            )
            conversation = await session.scalar(
                select(ConversationRow).where(
                    ConversationRow.id == ingress.conversation_id,
                    ConversationRow.team_id == authority.team_id,
                    ConversationRow.external_id == ingress.channel_id,
                )
            )
            if thread is None or conversation is None:
                raise LiveProofNotFound
            context_rows = tuple(
                (
                    await session.scalars(
                        select(ConversationAccessSnapshotRow)
                        .where(
                            ConversationAccessSnapshotRow.ingress_event_id == ingress.event_id,
                            ConversationAccessSnapshotRow.organization_id
                            == authority.organization_id,
                            ConversationAccessSnapshotRow.team_id == authority.team_id,
                        )
                        .order_by(ConversationAccessSnapshotRow.position)
                    )
                ).all()
            )
            membership_rows = (
                tuple(
                    (
                        await session.scalars(
                            select(ConversationActorMembershipRow)
                            .where(
                                ConversationActorMembershipRow.organization_id
                                == authority.organization_id,
                                ConversationActorMembershipRow.team_id == authority.team_id,
                                ConversationActorMembershipRow.actor_id == ingress.user_id,
                            )
                            .order_by(ConversationActorMembershipRow.conversation_external_id)
                        )
                    ).all()
                )
                if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION
                else ()
            )
            event_rows = tuple(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(
                            RunEventRow.run_id == binding.run_id,
                            RunEventRow.task_id == task.id,
                        )
                        .order_by(RunEventRow.sequence)
                    )
                ).all()
            )
            observation_rows = tuple(
                (
                    await session.scalars(
                        select(ObservationRow)
                        .where(
                            ObservationRow.run_id == binding.run_id,
                            ObservationRow.organization_id == authority.organization_id,
                        )
                        .order_by(ObservationRow.id)
                    )
                ).all()
            )
            claim_rows = tuple(
                (
                    await session.scalars(
                        select(ClaimRow)
                        .where(
                            ClaimRow.run_id == binding.run_id,
                            ClaimRow.organization_id == authority.organization_id,
                        )
                        .order_by(ClaimRow.id)
                    )
                ).all()
            )
            remembered_record_ids = {
                str(item.data.get("record_id"))
                for item in observation_rows
                if item.kind == "memory.remember" and isinstance(item.data.get("record_id"), str)
            }
            memory_record_rows = tuple(
                (
                    await session.scalars(
                        select(MemoryRecordRow)
                        .where(
                            MemoryRecordRow.id.in_(remembered_record_ids),
                            MemoryRecordRow.organization_id == authority.organization_id,
                            MemoryRecordRow.strategy_id == run.strategy_id,
                        )
                        .order_by(MemoryRecordRow.id)
                    )
                ).all()
            )
            memory_revision_rows = tuple(
                (
                    await session.scalars(
                        select(MemoryRevisionRow)
                        .where(
                            MemoryRevisionRow.record_id.in_(remembered_record_ids),
                            MemoryRevisionRow.organization_id == authority.organization_id,
                            MemoryRevisionRow.strategy_id == run.strategy_id,
                        )
                        .order_by(MemoryRevisionRow.record_id, MemoryRevisionRow.number)
                    )
                ).all()
            )
            memory_source_ids = {
                str(source_id) for item in memory_revision_rows for source_id in item.source_ids
            }
            memory_source_rows = tuple(
                (
                    await session.scalars(
                        select(MemorySourceRow)
                        .where(
                            MemorySourceRow.id.in_(memory_source_ids),
                            MemorySourceRow.organization_id == authority.organization_id,
                            MemorySourceRow.strategy_id == run.strategy_id,
                        )
                        .order_by(MemorySourceRow.id)
                    )
                ).all()
            )
            outbox_rows = tuple(
                (
                    await session.scalars(
                        select(DeliveryOutboxRow)
                        .where(
                            DeliveryOutboxRow.ingress_event_id == ingress.event_id,
                            DeliveryOutboxRow.task_id == task.id,
                            DeliveryOutboxRow.run_id == binding.run_id,
                            DeliveryOutboxRow.organization_id == authority.organization_id,
                        )
                        .order_by(DeliveryOutboxRow.kind, DeliveryOutboxRow.payload_version)
                    )
                ).all()
            )
            final_outbox = _require_one([item for item in outbox_rows if item.kind == "final"])
            if final_outbox.receipt_message_ts is None:
                raise LiveProofIntegrityError("delivered live proof lacks a Slack receipt")
            message_external_event_ids = {
                ingress.event_id,
                *(f"slack-delivery:{item.id}" for item in outbox_rows),
            }
            message_rows = tuple(
                (
                    await session.scalars(
                        select(SanitizedMessageRow)
                        .where(
                            SanitizedMessageRow.organization_id == authority.organization_id,
                            SanitizedMessageRow.harness_thread_id == thread.id,
                            SanitizedMessageRow.external_event_id.in_(message_external_event_ids),
                        )
                        .order_by(
                            SanitizedMessageRow.provider_message_ts,
                            SanitizedMessageRow.role,
                            SanitizedMessageRow.external_event_id,
                        )
                    )
                ).all()
            )
            plan_rows = tuple(
                (
                    await session.scalars(
                        select(PlanRow).where(
                            PlanRow.parent_task_id == task.id,
                            PlanRow.parent_run_id == binding.run_id,
                            PlanRow.organization_id == authority.organization_id,
                        )
                    )
                ).all()
            )
            plan_snapshot = await _load_plan_snapshot(
                session,
                plan_rows=plan_rows,
                organization_id=authority.organization_id,
                parent_task_id=task.id,
                parent_run_id=binding.run_id,
            )
            child_run_ids = (
                {item.child_run_id for item in plan_snapshot.nodes if item.child_run_id is not None}
                if plan_snapshot is not None
                else set()
            )
            child_task_ids = (
                {
                    item.child_task_id
                    for item in plan_snapshot.nodes
                    if item.child_task_id is not None
                }
                if plan_snapshot is not None
                else set()
            )
            child_task_rows = tuple(
                (
                    await session.scalars(
                        select(TaskRow)
                        .where(
                            TaskRow.id.in_(child_task_ids),
                            TaskRow.organization_id == authority.organization_id,
                        )
                        .order_by(TaskRow.id)
                    )
                ).all()
            )
            child_run_rows = tuple(
                (
                    await session.scalars(
                        select(RunRow)
                        .where(
                            RunRow.id.in_(child_run_ids),
                            RunRow.organization_id == authority.organization_id,
                        )
                        .order_by(RunRow.id)
                    )
                ).all()
            )
            child_event_rows = tuple(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(RunEventRow.run_id.in_(child_run_ids))
                        .order_by(RunEventRow.run_id, RunEventRow.sequence)
                    )
                ).all()
            )
            child_observation_rows = tuple(
                (
                    await session.scalars(
                        select(ObservationRow)
                        .where(
                            ObservationRow.run_id.in_(child_run_ids),
                            ObservationRow.organization_id == authority.organization_id,
                        )
                        .order_by(ObservationRow.run_id, ObservationRow.id)
                    )
                ).all()
            )
            child_claim_rows = tuple(
                (
                    await session.scalars(
                        select(ClaimRow)
                        .where(
                            ClaimRow.run_id.in_(child_run_ids),
                            ClaimRow.organization_id == authority.organization_id,
                        )
                        .order_by(ClaimRow.run_id, ClaimRow.id)
                    )
                ).all()
            )
            child_outbox_rows = tuple(
                (
                    await session.scalars(
                        select(DeliveryOutboxRow)
                        .where(
                            DeliveryOutboxRow.run_id.in_(child_run_ids),
                            DeliveryOutboxRow.organization_id == authority.organization_id,
                        )
                        .order_by(DeliveryOutboxRow.run_id, DeliveryOutboxRow.id)
                    )
                ).all()
            )
        return _build_case(
            authority=authority,
            binding=binding,
            ingress=ingress,
            conversation=conversation,
            thread=thread,
            task=task,
            run=run,
            context_rows=context_rows,
            membership_rows=membership_rows,
            event_rows=event_rows,
            observation_rows=observation_rows,
            claim_rows=claim_rows,
            memory_record_rows=memory_record_rows,
            memory_revision_rows=memory_revision_rows,
            memory_source_rows=memory_source_rows,
            outbox_rows=outbox_rows,
            final_outbox=final_outbox,
            message_rows=message_rows,
            plan_snapshot=plan_snapshot,
            child_task_rows=child_task_rows,
            child_run_rows=child_run_rows,
            child_event_rows=child_event_rows,
            child_observation_rows=child_observation_rows,
            child_claim_rows=child_claim_rows,
            child_outbox_rows=child_outbox_rows,
        )


async def _load_plan_snapshot(
    session: AsyncSession,
    *,
    plan_rows: Sequence[PlanRow],
    organization_id: str,
    parent_task_id: str,
    parent_run_id: str,
) -> PlanSnapshot | None:
    if not plan_rows:
        return None
    plan_row = _require_one(plan_rows)
    revisions = tuple(
        (
            await session.scalars(
                select(PlanRevisionRow)
                .where(
                    PlanRevisionRow.plan_id == plan_row.id,
                    PlanRevisionRow.organization_id == organization_id,
                )
                .order_by(PlanRevisionRow.number)
            )
        ).all()
    )
    nodes = tuple(
        (
            await session.scalars(
                select(PlanNodeRow)
                .where(
                    PlanNodeRow.plan_id == plan_row.id,
                    PlanNodeRow.organization_id == organization_id,
                )
                .order_by(PlanNodeRow.revision_number, PlanNodeRow.node_key)
            )
        ).all()
    )
    delegations = tuple(
        (
            await session.scalars(
                select(DelegationRow)
                .where(
                    DelegationRow.plan_id == plan_row.id,
                    DelegationRow.organization_id == organization_id,
                    DelegationRow.parent_task_id == parent_task_id,
                    DelegationRow.parent_run_id == parent_run_id,
                )
                .order_by(DelegationRow.node_id, DelegationRow.attempt)
            )
        ).all()
    )
    snapshot = PlanSnapshot(
        plan=Plan(
            id=plan_row.id,
            scope=ScopeKey(
                organization_id=plan_row.organization_id,
                strategy_id=plan_row.strategy_id,
            ),
            parent_task_id=plan_row.parent_task_id,
            parent_run_id=plan_row.parent_run_id,
            idempotency_key=plan_row.idempotency_key,
            initial_digest=plan_row.initial_digest,
            status=PlanStatus(plan_row.status),
            current_revision=plan_row.current_revision,
            max_revisions=plan_row.max_revisions,
            output=plan_row.output,
            error=plan_row.error,
            version=plan_row.version,
            created_at=plan_row.created_at,
            updated_at=plan_row.updated_at,
        ),
        revisions=tuple(
            PlanRevision(
                id=item.id,
                plan_id=item.plan_id,
                number=item.number,
                goal=item.goal,
                nodes=tuple(PlanNodeDefinition.model_validate(node) for node in item.definition),
                digest=item.digest,
                parent_revision_id=item.parent_revision_id,
                parent_digest=item.parent_digest,
                reason=item.reason,
                created_at=item.created_at,
            )
            for item in revisions
        ),
        nodes=tuple(
            PlanNode(
                id=item.id,
                plan_id=item.plan_id,
                revision_id=item.revision_id,
                revision_number=item.revision_number,
                definition=PlanNodeDefinition(
                    key=item.node_key,
                    objective=item.objective,
                    depends_on=tuple(item.depends_on),
                    max_attempts=item.max_attempts,
                ),
                status=PlanNodeStatus(item.status),
                attempt=item.attempt,
                claim_owner=item.claim_owner,
                claim_token=item.claim_token,
                lease_expires_at=item.lease_expires_at,
                child_task_id=item.child_task_id,
                child_run_id=item.child_run_id,
                output=item.output,
                error=item.error,
                created_at=item.created_at,
                updated_at=item.updated_at,
            )
            for item in nodes
        ),
        delegations=tuple(
            Delegation(
                id=item.id,
                plan_id=item.plan_id,
                revision_id=item.revision_id,
                node_id=item.node_id,
                parent_task_id=item.parent_task_id,
                parent_run_id=item.parent_run_id,
                attempt=item.attempt,
                owner=item.owner,
                claim_token=item.claim_token,
                status=DelegationStatus(item.status),
                child_task_id=item.child_task_id,
                child_run_id=item.child_run_id,
                output=item.output,
                error=item.error,
                created_at=item.created_at,
                finished_at=item.finished_at,
            )
            for item in delegations
        ),
    )
    if snapshot.plan.status is not PlanStatus.COMPLETED:
        raise LiveProofIntegrityError("live proof plan is not completed")
    child_pairs = {
        (item.child_task_id, item.child_run_id)
        for item in snapshot.nodes
        if item.child_task_id is not None and item.child_run_id is not None
    }
    child_task_ids = {task_id for task_id, _ in child_pairs}
    child_run_ids = {run_id for _, run_id in child_pairs}
    if child_pairs:
        child_tasks = tuple(
            (
                await session.scalars(
                    select(TaskRow).where(
                        TaskRow.id.in_(child_task_ids),
                        TaskRow.organization_id == organization_id,
                    )
                )
            ).all()
        )
        child_runs = tuple(
            (
                await session.scalars(
                    select(RunRow).where(
                        RunRow.id.in_(child_run_ids),
                        RunRow.organization_id == organization_id,
                    )
                )
            ).all()
        )
        tasks_by_id = {item.id: item for item in child_tasks}
        runs_by_id = {item.id: item for item in child_runs}
        if set(tasks_by_id) != child_task_ids or set(runs_by_id) != child_run_ids:
            raise LiveProofIntegrityError("live proof plan child scope is incomplete")
        if any(
            tasks_by_id[task_id].parent_task_id != parent_task_id
            or runs_by_id[run_id].task_id != task_id
            or tasks_by_id[task_id].strategy_id != snapshot.plan.scope.strategy_id
            or runs_by_id[run_id].strategy_id != snapshot.plan.scope.strategy_id
            or (
                TaskStatus(tasks_by_id[task_id].status),
                RunStatus(runs_by_id[run_id].status),
            )
            not in LEGAL_TASK_RUN_PAIRS
            for task_id, run_id in child_pairs
        ):
            raise LiveProofIntegrityError("live proof plan child authority diverges")
        current_child_pairs = {
            (item.child_task_id, item.child_run_id)
            for item in snapshot.current_nodes
            if item.child_task_id is not None and item.child_run_id is not None
        }
        if any(
            tasks_by_id[task_id].status != "completed" or runs_by_id[run_id].status != "completed"
            for task_id, run_id in current_child_pairs
        ):
            raise LiveProofIntegrityError("live proof current plan child did not complete")
    return snapshot


def _build_case(
    *,
    authority: LiveProofAuthority,
    binding: LiveProofBinding,
    ingress: SlackIngressEventRow,
    conversation: ConversationRow,
    thread: ThreadRow,
    task: TaskRow,
    run: RunRow,
    context_rows: tuple[ConversationAccessSnapshotRow, ...],
    membership_rows: tuple[ConversationActorMembershipRow, ...],
    event_rows: tuple[RunEventRow, ...],
    observation_rows: tuple[ObservationRow, ...],
    claim_rows: tuple[ClaimRow, ...],
    memory_record_rows: tuple[MemoryRecordRow, ...],
    memory_revision_rows: tuple[MemoryRevisionRow, ...],
    memory_source_rows: tuple[MemorySourceRow, ...],
    outbox_rows: tuple[DeliveryOutboxRow, ...],
    final_outbox: DeliveryOutboxRow,
    message_rows: tuple[SanitizedMessageRow, ...],
    plan_snapshot: PlanSnapshot | None,
    child_task_rows: tuple[TaskRow, ...],
    child_run_rows: tuple[RunRow, ...],
    child_event_rows: tuple[RunEventRow, ...],
    child_observation_rows: tuple[ObservationRow, ...],
    child_claim_rows: tuple[ClaimRow, ...],
    child_outbox_rows: tuple[DeliveryOutboxRow, ...],
) -> LiveProofCase:
    _validate_scope_and_terminal(
        authority=authority,
        binding=binding,
        ingress=ingress,
        conversation=conversation,
        thread=thread,
        task=task,
        run=run,
    )
    context_summary = _context_summary(binding, ingress, context_rows, membership_rows)
    event_summary = _event_summary(task, run, event_rows)
    outbox_summary = _outbox_summary(ingress, task, run, outbox_rows, final_outbox)
    message_plane_digest = _message_plane_digest(
        ingress,
        thread,
        run,
        final_outbox,
        message_rows,
        outbox_rows=outbox_rows,
    )
    plan_summary = _plan_summary(binding, plan_snapshot)
    recall_summary = _memory_recall_summary(
        binding,
        run,
        observation_rows,
        claim_rows,
    )
    case_summary = _case_contract_summary(
        authority=authority,
        binding=binding,
        ingress=ingress,
        task=task,
        run=run,
        event_rows=event_rows,
        observations=observation_rows,
        claims=claim_rows,
        memory_records=memory_record_rows,
        memory_revisions=memory_revision_rows,
        memory_sources=memory_source_rows,
        plan_snapshot=plan_snapshot,
        child_tasks=child_task_rows,
        child_runs=child_run_rows,
        child_events=child_event_rows,
        child_observations=child_observation_rows,
        child_claims=child_claim_rows,
        child_outbox=child_outbox_rows,
    )
    final_output_digest = _digest_text(cast(str, run.final_output))
    task_run_digest = _digest(
        {
            "task_id": task.id,
            "task_status": task.status,
            "task_version": task.version,
            "run_id": run.id,
            "run_status": run.status,
            "run_phase": run.phase,
            "run_iteration": run.iteration,
            "run_event_sequence": run.event_sequence,
            "run_version": run.version,
            "terminal_reason": run.terminal_reason,
            "final_output_digest": final_output_digest,
        }
    )
    payload: dict[str, object] = {
        "evidence_id": binding.evidence_id,
        "binding_digest": binding.digest,
        "message_ts": binding.message_ts,
        "run_id": binding.run_id,
        "task_id": task.id,
        "channel_id": ingress.channel_id,
        "conversation_kind": ingress.conversation_kind,
        "slack_response_ts": final_outbox.receipt_message_ts,
        "objective_digest": _digest_text(task.objective),
        "final_output_digest": final_output_digest,
        "task_terminal_state": task.status,
        "run_terminal_state": run.status,
        **event_summary,
        **context_summary,
        "message_plane_digest": message_plane_digest,
        **recall_summary,
        **case_summary,
        **plan_summary,
        **outbox_summary,
        "ingress_digest": _digest(
            {
                "event_id": ingress.event_id,
                "team_id": ingress.team_id,
                "channel_id": ingress.channel_id,
                "message_ts": ingress.message_ts,
                "thread_root_ts": ingress.thread_root_ts,
                "conversation_id": ingress.conversation_id,
                "conversation_kind": ingress.conversation_kind,
                "trigger_kind": ingress.trigger_kind,
                "organization_id": ingress.organization_id,
                "task_id": ingress.task_id,
                "launch_status": ingress.launch_status,
                "context_access_hash": ingress.context_access_hash,
            }
        ),
        "task_run_digest": task_run_digest,
    }
    payload["row_snapshot_digest"] = _digest(payload)
    sanitize_payload(cast(dict[str, JsonValue], payload))
    return LiveProofCase.model_validate(payload)


def _validate_scope_and_terminal(
    *,
    authority: LiveProofAuthority,
    binding: LiveProofBinding,
    ingress: SlackIngressEventRow,
    conversation: ConversationRow,
    thread: ThreadRow,
    task: TaskRow,
    run: RunRow,
) -> None:
    if (
        ingress.message_ts != binding.message_ts
        or run.id != binding.run_id
        or ingress.conversation_kind != binding.expected_conversation_kind
        or ingress.channel_id != binding.expected_destination_id
        or ingress.channel_id != conversation.external_id
        or conversation.kind
        != {
            "ordinary_internal": "channel",
            "dm": "dm",
            "mpim": "group_dm",
            "shared": "shared",
            "external": "external",
        }[ingress.conversation_kind]
        or conversation.bot_presence != "present"
        or conversation.lifecycle != "active"
        or ingress.bot_presence != "present"
        or ingress.conversation_lifecycle != "active"
        or ingress.conversation_id != thread.conversation_id
        or task.thread_id != thread.id
        or thread.external_channel_id != ingress.channel_id
        or task.organization_id != authority.organization_id
        or run.organization_id != authority.organization_id
        or ingress.strategy_id != task.strategy_id
        or task.strategy_id != run.strategy_id
        or task.status != "completed"
        or run.status != "completed"
        or run.terminal_reason != "verified_completion"
        or not task.final_output
        or task.final_output != run.final_output
    ):
        raise LiveProofIntegrityError("live proof scope or terminal state diverges")


def _context_summary(
    binding: LiveProofBinding,
    ingress: SlackIngressEventRow,
    rows: tuple[ConversationAccessSnapshotRow, ...],
    membership_rows: tuple[ConversationActorMembershipRow, ...],
) -> dict[str, object]:
    conversation_ids = tuple(str(item) for item in ingress.context_conversation_ids)
    expected_projection_source = (
        "dm_membership_intersection"
        if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION
        else "exact_destination"
    )
    if (
        not conversation_ids
        or conversation_ids != tuple(sorted(set(conversation_ids)))
        or conversation_ids != binding.expected_context_conversation_ids
        or ingress.context_access_hash != binding.expected_context_access_hash
        or ingress.context_projection_source != expected_projection_source
        or (ingress.conversation_kind != "dm" and conversation_ids != (ingress.channel_id,))
        or tuple(item.position for item in rows) != tuple(range(len(conversation_ids)))
        or tuple(item.conversation_external_id for item in rows) != conversation_ids
        or any(
            item.team_id != ingress.team_id
            or item.context_access_hash != ingress.context_access_hash
            or item.destination_external_id != ingress.channel_id
            or item.actor_id != ingress.user_id
            or item.source_kind != expected_projection_source
            for item in rows
        )
    ):
        raise LiveProofIntegrityError("live proof context projection is incomplete or widened")
    current_memberships = tuple(item for item in membership_rows if item.status == "active")
    if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION:
        if tuple(
            item.conversation_external_id for item in current_memberships
        ) != conversation_ids or any(
            item.team_id != ingress.team_id or item.actor_id != ingress.user_id
            for item in current_memberships
        ):
            raise LiveProofIntegrityError(
                "live proof DM membership union is no longer the exact active source set"
            )
        membership_digest = _digest(
            [
                {
                    "conversation_external_id": item.conversation_external_id,
                    "context_access_hash": item.context_access_hash,
                    "source_kind": item.source_kind,
                    "status": item.status,
                    "version": item.version,
                    "observed_at": item.observed_at,
                }
                for item in current_memberships
            ]
        )
    elif membership_rows:
        raise LiveProofIntegrityError("non-DM-union proof received membership authority rows")
    else:
        membership_digest = _digest([])
    return {
        "context_access_hash": ingress.context_access_hash,
        "context_projection_source": ingress.context_projection_source,
        "context_conversation_count": len(conversation_ids),
        "conversation_source_set_digest": _digest(conversation_ids),
        "context_snapshot_digest": _digest(
            [
                {
                    "position": item.position,
                    "conversation_external_id": item.conversation_external_id,
                    "source_kind": item.source_kind,
                    "context_access_hash": item.context_access_hash,
                }
                for item in rows
            ]
        ),
        "current_membership_count": len(current_memberships),
        "current_membership_digest": membership_digest,
    }


def _event_summary(
    task: TaskRow,
    run: RunRow,
    rows: tuple[RunEventRow, ...],
) -> dict[str, object]:
    if not rows or tuple(item.sequence for item in rows) != tuple(range(1, len(rows) + 1)):
        raise LiveProofIntegrityError("live proof event timeline is absent or non-contiguous")
    if run.event_sequence != rows[-1].sequence or rows[-1].type != EventType.RUN_COMPLETED.value:
        raise LiveProofIntegrityError("live proof terminal event does not match the run")
    events = tuple(
        RunEvent(
            id=item.id,
            run_id=item.run_id,
            task_id=item.task_id,
            sequence=item.sequence,
            type=EventType(item.type),
            occurred_at=item.occurred_at,
            iteration=item.iteration,
            schema_version=item.schema_version,
            payload=item.payload,
        )
        for item in rows
    )
    normalized = normalize_run_timeline(
        events,
        ScopeKey(organization_id=run.organization_id, strategy_id=run.strategy_id),
    )
    context_events = [item for item in normalized if item.kind.value == "context_built"]
    raw_context_events = [item for item in rows if item.type == EventType.CONTEXT_BUILT.value]
    if not context_events or len(context_events) != len(raw_context_events):
        raise LiveProofIntegrityError("live proof timeline lacks a context manifest")
    # The universal event envelope intentionally projects CONTEXT_BUILT into the
    # compact ContextPayload contract.  The complete source manifest remains in
    # the authoritative durable v1 event and must be validated there.
    source_manifest = raw_context_events[-1].payload.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise LiveProofIntegrityError("live proof context manifest is malformed")
    manifest_digest = source_manifest.get("manifest_digest")
    if not _is_sha256(manifest_digest):
        raise LiveProofIntegrityError("live proof context manifest digest is malformed")
    envelopes = [
        {
            "event_id": item.event_id,
            "run_id": item.run_id,
            "task_id": item.task_id,
            "sequence": item.sequence,
            "kind": item.kind.value,
            "schema_version": item.schema_version,
            "correlation_id": item.correlation_id,
            "causation_id": item.causation_id,
            "payload_digest": _digest(sanitize_payload(item.payload)),
        }
        for item in normalized
    ]
    if any(item["task_id"] != task.id or item["run_id"] != run.id for item in envelopes):
        raise LiveProofIntegrityError("live proof event scope diverges")
    return {
        "event_terminal_state": "run_completed",
        "event_count": len(rows),
        "last_event_sequence": rows[-1].sequence,
        "event_timeline_digest": _digest(envelopes),
        "context_manifest_digest": manifest_digest,
    }


def _outbox_summary(
    ingress: SlackIngressEventRow,
    task: TaskRow,
    run: RunRow,
    rows: tuple[DeliveryOutboxRow, ...],
    final: DeliveryOutboxRow,
) -> dict[str, object]:
    if (
        not rows
        or final.state != "delivered"
        or final.receipt_message_ts is None
        or final.destination_channel_id != ingress.channel_id
        or final.destination_thread_ts != ingress.thread_root_ts
        or any(
            item.task_id != task.id
            or item.run_id != run.id
            or item.ingress_event_id != ingress.event_id
            or item.destination_channel_id != ingress.channel_id
            or item.state != "delivered"
            or item.payload_hash != _digest_text(item.payload)
            for item in rows
        )
    ):
        raise LiveProofIntegrityError("live proof outbox is incomplete or mismatched")
    return {
        "delivery_state": "delivered",
        "outbox_count": len(rows),
        "outbox_digest": _digest(
            [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "payload_version": item.payload_version,
                    "payload_hash": item.payload_hash,
                    "state": item.state,
                    "attempt_count": item.attempt_count,
                    "receipt_message_ts": item.receipt_message_ts,
                    "destination_channel_id": item.destination_channel_id,
                    "destination_thread_ts": item.destination_thread_ts,
                }
                for item in rows
            ]
        ),
    }


def _message_plane_digest(
    ingress: SlackIngressEventRow,
    thread: ThreadRow,
    run: RunRow,
    final: DeliveryOutboxRow,
    rows: tuple[SanitizedMessageRow, ...],
    *,
    outbox_rows: tuple[DeliveryOutboxRow, ...] = (),
) -> str:
    if final.receipt_message_ts is None or run.final_output is None:
        raise LiveProofIntegrityError("live proof Slack receipt is absent")
    user_rows = tuple(item for item in rows if item.role == "user")
    assistant_rows = tuple(item for item in rows if item.role == "assistant")
    if len(user_rows) != 1 or not assistant_rows or len(rows) != 1 + len(assistant_rows):
        raise LiveProofIntegrityError("live proof sanitized message plane is incomplete")
    user = user_rows[0]
    delivery_by_event_id = {
        f"slack-delivery:{item.id}": item for item in outbox_rows if item.id is not None
    }
    delivered_assistants = {
        item.external_event_id: item
        for item in assistant_rows
        if item.external_event_id in delivery_by_event_id
    }
    legacy_assistants = tuple(
        item for item in assistant_rows if item.external_event_id == ingress.event_id
    )

    if delivered_assistants:
        if (
            len(delivered_assistants) != len(assistant_rows)
            or set(delivered_assistants) != set(delivery_by_event_id)
            or any(
                outbox.receipt_message_ts is None
                or message.provider_message_ts != outbox.receipt_message_ts
                or message.content_hash != outbox.payload_hash
                or message.content_hash != _digest_text(outbox.payload)
                for event_id, outbox in delivery_by_event_id.items()
                for message in (delivered_assistants[event_id],)
            )
        ):
            raise LiveProofIntegrityError("live proof sanitized message plane is incomplete")
        final_assistant = delivered_assistants.get(f"slack-delivery:{final.id}")
        final_content_hash = final.payload_hash
    elif len(legacy_assistants) == 1 and len(assistant_rows) == 1:
        # Historical live cases persisted the verified final alongside the
        # ingress identity before delivery receipts became message-plane rows.
        final_assistant = legacy_assistants[0]
        if final_assistant.provider_message_ts not in {
            ingress.message_ts,
            final.receipt_message_ts,
        }:
            raise LiveProofIntegrityError("live proof sanitized message plane is incomplete")
        final_content_hash = _digest_text(run.final_output)
    else:
        raise LiveProofIntegrityError("live proof sanitized message plane is incomplete")

    allowed_external_event_ids = {ingress.event_id, *delivery_by_event_id}
    if (
        user.provider_message_ts != ingress.message_ts
        or user.external_event_id != ingress.event_id
        or user.actor_id != ingress.user_id
        or user.content_hash != _digest_text(ingress.prompt)
        or final_assistant is None
        or final_assistant.provider_message_ts not in {ingress.message_ts, final.receipt_message_ts}
        or final_assistant.content_hash != final_content_hash
        or any(
            item.external_event_id not in allowed_external_event_ids
            or item.conversation_id != ingress.conversation_id
            or item.harness_thread_id != thread.id
            or item.destination_id != ingress.channel_id
            or item.context_access_hash != ingress.context_access_hash
            or item.content_hash != _digest_text(item.text)
            or (item.role == "assistant" and item.actor_id != "leo")
            for item in rows
        )
    ):
        raise LiveProofIntegrityError("live proof sanitized message plane is incomplete")
    return _digest(
        [
            {
                "id": item.id,
                "role": item.role,
                "external_event_id": item.external_event_id,
                "provider_message_ts": item.provider_message_ts,
                "content_hash": item.content_hash,
                "context_access_hash": item.context_access_hash,
            }
            for item in rows
        ]
    )


def _memory_recall_summary(
    binding: LiveProofBinding,
    run: RunRow,
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
) -> dict[str, object]:
    expected_source = binding.expected_recall_source_conversation_id
    empty_digest = _digest([])
    if expected_source is None:
        return {
            "memory_recall_verified": False,
            "memory_recall_source_digest": empty_digest,
            "memory_recall_observation_digest": empty_digest,
            "grounded_memory_claim_digest": empty_digest,
        }
    eligible: list[tuple[ObservationRow, ProgressiveMemorySearchResult]] = []
    for row in observations:
        if (
            row.kind != "memory.search"
            or row.status != "retrieved"
            or row.quality != "internal_context"
            or row.schema_version != "observation-v2"
            or row.rejection_code is not None
            or row.strategy_id != run.strategy_id
            or row.source.get("provider") != "leo_memory"
        ):
            continue
        try:
            result = ProgressiveMemorySearchResult.model_validate(row.data)
        except ValueError:
            continue
        if result.selected_count < 1 or not any(
            item.source_conversation == expected_source for item in result.items
        ):
            continue
        eligible.append((row, result))
    eligible_ids = {row.id for row, _ in eligible}
    grounded_claims = [
        item
        for item in claims
        if item.kind == "inference"
        and item.strategy_id == run.strategy_id
        and bool(eligible_ids.intersection(str(value) for value in item.observation_ids))
    ]
    if not eligible or not grounded_claims or not run.final_output:
        raise LiveProofIntegrityError(
            "live proof memory recall lacks a positive grounded channel-memory result"
        )
    return {
        "memory_recall_verified": True,
        "memory_recall_source_digest": _digest(expected_source),
        "memory_recall_observation_digest": _digest(
            [
                {
                    "id": row.id,
                    "raw_hash": row.raw_hash,
                    "source_reference": row.source.get("reference"),
                    "query_hash": result.query_hash,
                    "selected_count": result.selected_count,
                    "items": [
                        {
                            "kind": item.kind.value,
                            "reference": item.reference,
                            "source_conversation": item.source_conversation,
                            "content_digest": _digest_text(item.content)
                            if item.content is not None
                            else None,
                            "excerpt_digest": _digest_text(item.excerpt)
                            if item.excerpt is not None
                            else None,
                        }
                        for item in result.items
                        if item.source_conversation == expected_source
                    ],
                }
                for row, result in eligible
            ]
        ),
        "grounded_memory_claim_digest": _digest(
            [
                {
                    "id": item.id,
                    "kind": item.kind,
                    "statement_digest": _digest_text(item.statement),
                    "observation_ids": sorted(str(value) for value in item.observation_ids),
                }
                for item in grounded_claims
            ]
        ),
    }


def _case_contract_summary(
    *,
    authority: LiveProofAuthority,
    binding: LiveProofBinding,
    ingress: SlackIngressEventRow,
    task: TaskRow,
    run: RunRow,
    event_rows: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    memory_records: tuple[MemoryRecordRow, ...],
    memory_revisions: tuple[MemoryRevisionRow, ...],
    memory_sources: tuple[MemorySourceRow, ...],
    plan_snapshot: PlanSnapshot | None,
    child_tasks: tuple[TaskRow, ...],
    child_runs: tuple[RunRow, ...],
    child_events: tuple[RunEventRow, ...],
    child_observations: tuple[ObservationRow, ...],
    child_claims: tuple[ClaimRow, ...],
    child_outbox: tuple[DeliveryOutboxRow, ...],
) -> dict[str, object]:
    evidence_id = binding.evidence_id
    if evidence_id is LiveEvidenceId.MEMORY_WRITE:
        invariant, mutation = _memory_write_contract(
            authority=authority,
            ingress=ingress,
            task=task,
            run=run,
            events=event_rows,
            observations=observations,
            claims=claims,
            records=memory_records,
            revisions=memory_revisions,
            sources=memory_sources,
        )
    elif evidence_id is LiveEvidenceId.CROSS_CHANNEL_NEGATIVE:
        invariant = _cross_channel_negative_contract(run, event_rows, observations, claims)
        mutation = None
    elif evidence_id is LiveEvidenceId.QUOTE:
        invariant = _provider_contract(
            run,
            event_rows,
            observations,
            claims,
            kind="market.get_quote",
            provider="finnhub",
            quality="provider_reported",
            subject_key="symbol",
        )
        mutation = None
    elif evidence_id is LiveEvidenceId.SEC:
        invariant = _provider_contract(
            run,
            event_rows,
            observations,
            claims,
            kind="sec.get_recent_filings",
            provider="sec-edgar",
            quality="primary_source",
            subject_key="ticker",
        )
        mutation = None
    elif evidence_id in {LiveEvidenceId.PRIVATE_CHANNEL, LiveEvidenceId.GROUP_DM}:
        invariant = _direct_conversation_contract(
            run,
            event_rows,
            observations,
            claims,
            plan_snapshot,
        )
        mutation = None
    elif evidence_id is LiveEvidenceId.DELEGATED_REPLANNING:
        invariant, delegated = _delegated_contract(
            run=run,
            parent_events=event_rows,
            observations=observations,
            claims=claims,
            snapshot=plan_snapshot,
            child_tasks=child_tasks,
            child_runs=child_runs,
            child_events=child_events,
            child_observations=child_observations,
            child_claims=child_claims,
            child_outbox=child_outbox,
        )
        mutation = None
    else:
        invariant = _positive_memory_case_contract(
            binding,
            run,
            event_rows,
            observations,
            claims,
        )
        mutation = None
    empty_digest = _digest([])
    delegated_values = (
        delegated
        if evidence_id is LiveEvidenceId.DELEGATED_REPLANNING
        else {
            "delegated_child_count": 0,
            "delegated_overlap_verified": False,
            "delegated_evidence_digest": empty_digest,
        }
    )
    return {
        "case_invariant_digest": invariant,
        "observed_evidence_count": len(observations),
        "grounded_claim_count": len(claims),
        "memory_mutation_record_count": len(memory_records) if mutation is not None else 0,
        "memory_mutation_revision_count": len(memory_revisions) if mutation is not None else 0,
        "memory_mutation_source_count": len(memory_sources) if mutation is not None else 0,
        "memory_mutation_digest": mutation or empty_digest,
        **delegated_values,
    }


def _memory_write_contract(
    *,
    authority: LiveProofAuthority,
    ingress: SlackIngressEventRow,
    task: TaskRow,
    run: RunRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    records: tuple[MemoryRecordRow, ...],
    revisions: tuple[MemoryRevisionRow, ...],
    sources: tuple[MemorySourceRow, ...],
) -> tuple[str, str]:
    if len(observations) != 1 or claims:
        raise LiveProofIntegrityError("memory-write proof requires only its remember observation")
    observation = observations[0]
    if (
        observation.kind != "memory.remember"
        or observation.status != "retrieved"
        or observation.quality != "internal_context"
        or observation.schema_version != "observation-v2"
        or observation.normalization_version != "normalization-v1"
        or observation.rejection_code is not None
        or observation.source.get("provider") != "leo_memory"
        or observation.data.get("operation") != "remember"
        or observation.data.get("status") != "active"
    ):
        raise LiveProofIntegrityError("memory-write proof lacks a trusted remember observation")
    record_id = observation.data.get("record_id")
    revision_number = observation.data.get("revision")
    if not isinstance(record_id, str) or not isinstance(revision_number, int):
        raise LiveProofIntegrityError("memory-write observation identity is malformed")
    if (
        len(records) != 1
        or len(revisions) != 1
        or len(sources) != 3
        or records[0].id != record_id
        or observation.source.get("reference") != record_id
    ):
        raise LiveProofIntegrityError("memory-write durable row set is incomplete or ambiguous")
    record = records[0]
    revision = revisions[0]
    expected_visibility = (
        "actor_private" if ingress.conversation_kind == "dm" else "conversation_local"
    )
    expected_namespace = (
        ingress.user_id if ingress.conversation_kind == "dm" else ingress.channel_id
    )
    if (
        record.status != "active"
        or record.kind != "note"
        or record.current_revision != revision_number
        or record.current_revision != revision.number
        or record.generation != 1
        or record.visibility != expected_visibility
        or record.namespace_id != expected_namespace
        or revision.record_id != record.id
        or revision.status != "active"
        or revision.visibility != expected_visibility
        or revision.namespace_id != expected_namespace
        or revision.actor_id != ingress.user_id
        or revision.reason != "explicit Slack remember"
        or revision.valid_until is not None
        or (revision.expires_at is not None and revision.expires_at <= run.updated_at)
        or run.started_at is None
        or record.created_at != revision.recorded_at
        or not (run.started_at <= revision.recorded_at <= observation.observed_at <= run.updated_at)
        or revision.content_hash != _digest_text(revision.content)
        or record.organization_id != authority.organization_id
        or revision.organization_id != authority.organization_id
        or record.strategy_id != run.strategy_id
        or revision.strategy_id != run.strategy_id
    ):
        raise LiveProofIntegrityError("memory-write revision is foreign, widened, or forgotten")
    sources_by_kind = {item.source_kind: item for item in sources}
    expected_references = {
        "slack_event": ingress.event_id,
        "leo_task": task.id,
        "slack_message": ingress.message_ts,
    }
    if (
        set(sources_by_kind) != set(expected_references)
        or set(revision.source_ids) != {item.id for item in sources}
        or len(revision.source_ids) != len(set(revision.source_ids))
        or any(
            item.reference != expected_references[kind]
            or item.visibility != expected_visibility
            or item.namespace_id != expected_namespace
            or item.organization_id != authority.organization_id
            or item.strategy_id != run.strategy_id
            for kind, item in sources_by_kind.items()
        )
    ):
        raise LiveProofIntegrityError("memory-write provenance is foreign or widened")
    _require_tool_path(events, observation, "memory.remember")
    mutation_payload = {
        "record_id": record.id,
        "revision_id": revision.id,
        "revision": revision.number,
        "generation": record.generation,
        "content_hash": revision.content_hash,
        "visibility": revision.visibility,
        "namespace_digest": _digest(revision.namespace_id),
        "actor_digest": _digest(revision.actor_id),
        "source_rows": [
            {
                "id": item.id,
                "kind": item.source_kind,
                "reference_digest": _digest(item.reference),
                "visibility": item.visibility,
                "namespace_digest": _digest(item.namespace_id),
            }
            for item in sorted(sources, key=lambda row: row.source_kind)
        ],
        "observation_id": observation.id,
        "tool_call_id": observation.tool_call_id,
        "raw_hash": observation.raw_hash,
    }
    mutation_digest = _digest(mutation_payload)
    return _digest(
        {"contract": "memory-write-v1", "mutation_digest": mutation_digest}
    ), mutation_digest


def _cross_channel_negative_contract(
    run: RunRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
) -> str:
    if len(observations) != 1 or len(claims) != 1:
        raise LiveProofIntegrityError(
            "cross-channel negative requires one empty search and inference"
        )
    observation = observations[0]
    claim = claims[0]
    try:
        result = ProgressiveMemorySearchResult.model_validate(observation.data)
    except ValueError as exc:
        raise LiveProofIntegrityError("cross-channel negative search result is malformed") from exc
    if (
        observation.kind != "memory.search"
        or observation.status != "retrieved"
        or observation.quality != "internal_context"
        or observation.schema_version != "observation-v2"
        or observation.normalization_version != "normalization-v1"
        or observation.rejection_code is not None
        or observation.strategy_id != run.strategy_id
        or observation.source.get("provider") != "leo_memory"
        or result.items
        or result.selected_count != 0
        or claim.kind != "inference"
        or tuple(str(item) for item in claim.observation_ids) != (observation.id,)
        or claim.statement != _EMPTY_MEMORY_SCOPE_INFERENCE
        or run.final_output != _EMPTY_MEMORY_SCOPE_INFERENCE
    ):
        raise LiveProofIntegrityError(
            "cross-channel negative is not the exact scoped-empty inference"
        )
    _require_tool_path(events, observation, "memory.search")
    return _digest(
        {
            "contract": "cross-channel-negative-v1",
            "observation_id": observation.id,
            "raw_hash": observation.raw_hash,
            "query_hash": result.query_hash,
            "claim_id": claim.id,
            "statement_digest": _digest_text(claim.statement),
        }
    )


def _provider_contract(
    run: RunRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    *,
    kind: str,
    provider: str,
    quality: str,
    subject_key: str,
) -> str:
    if len(observations) != 1 or not claims:
        raise LiveProofIntegrityError(
            "provider proof requires one exact observation and grounded claim"
        )
    observation = observations[0]
    subject = observation.data.get(subject_key)
    if (
        observation.kind != kind
        or observation.source.get("provider") != provider
        or observation.status != "retrieved"
        or observation.quality != quality
        or observation.schema_version != "observation-v2"
        or observation.normalization_version != "normalization-v1"
        or observation.rejection_code is not None
        or observation.strategy_id != run.strategy_id
        or observation.observed_at > run.updated_at
        or observation.expires_at is None
        or observation.expires_at <= run.updated_at
        or not isinstance(observation.source.get("reference"), str)
        or not cast(str, observation.source["reference"])
        or not isinstance(subject, str)
        or subject.upper() != "NVDA"
        or any(
            item.kind != "source_claim"
            or tuple(str(value) for value in item.observation_ids) != (observation.id,)
            or not _contains_statement(cast(str, run.final_output), item.statement)
            for item in claims
        )
    ):
        raise LiveProofIntegrityError("provider proof is stale, ungrounded, or has extra evidence")
    _require_tool_path(events, observation, kind)
    if kind == "market.get_quote" and (
        not isinstance(observation.data.get("price"), (int, float))
        or not isinstance(observation.data.get("as_of"), str)
    ):
        raise LiveProofIntegrityError("quote proof lacks exact numeric/as-of provider data")
    if kind == "sec.get_recent_filings":
        filings = observation.data.get("filings")
        if not isinstance(filings, list) or not filings or not isinstance(filings[0], dict):
            raise LiveProofIntegrityError("SEC proof lacks a primary filing result")
        if not {
            "form",
            "filing_date",
            "accession",
            "primary_document",
            "filing_url",
        } <= set(filings[0]):
            raise LiveProofIntegrityError("SEC proof filing result is incomplete")
    return _digest(
        {
            "contract": f"provider:{kind}:v1",
            "observation_id": observation.id,
            "raw_hash": observation.raw_hash,
            "source_reference_digest": _digest(observation.source.get("reference")),
            "subject": subject.upper(),
            "claim_rows": [
                {
                    "id": item.id,
                    "statement_digest": _digest_text(item.statement),
                    "observation_ids": list(item.observation_ids),
                }
                for item in claims
            ],
        }
    )


def _direct_conversation_contract(
    run: RunRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    snapshot: PlanSnapshot | None,
) -> str:
    tool_events = [
        item
        for item in events
        if item.type
        in {
            EventType.TOOL_STARTED.value,
            EventType.TOOL_COMPLETED.value,
            EventType.TOOL_FAILED.value,
        }
    ]
    usage_tool_calls = run.usage.get("tool_calls")
    if (
        snapshot is not None
        or observations
        or claims
        or tool_events
        or usage_tool_calls != 0
        or not run.final_output
    ):
        raise LiveProofIntegrityError("direct conversation proof used a tool, claim, or plan")
    return _digest(
        {
            "contract": "direct-conversation-no-tool-v1",
            "final_output_digest": _digest_text(run.final_output),
            "event_count": len(events),
        }
    )


def _positive_memory_case_contract(
    binding: LiveProofBinding,
    run: RunRow,
    events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
) -> str:
    expected_source = binding.expected_recall_source_conversation_id
    if expected_source is None or len(observations) != 1 or len(claims) != 1:
        raise LiveProofIntegrityError("positive memory proof requires one search and inference")
    observation = observations[0]
    claim = claims[0]
    try:
        result = ProgressiveMemorySearchResult.model_validate(observation.data)
    except ValueError as exc:
        raise LiveProofIntegrityError("positive memory proof search is malformed") from exc
    if (
        observation.kind != "memory.search"
        or observation.status != "retrieved"
        or observation.quality != "internal_context"
        or observation.schema_version != "observation-v2"
        or observation.normalization_version != "normalization-v1"
        or observation.rejection_code is not None
        or observation.strategy_id != run.strategy_id
        or observation.source.get("provider") != "leo_memory"
        or result.selected_count < 1
        or result.selected_count != len(result.items)
        or not any(item.source_conversation == expected_source for item in result.items)
        or any(item.source_conversation != expected_source for item in result.items)
        or claim.kind != "inference"
        or tuple(str(item) for item in claim.observation_ids) != (observation.id,)
        or not run.final_output
        or not _contains_statement(run.final_output, claim.statement)
    ):
        raise LiveProofIntegrityError(
            "positive memory proof is ungrounded or from the wrong source"
        )
    _require_tool_path(events, observation, "memory.search")
    return _digest(
        {
            "contract": "positive-memory-recall-v1",
            "observation_id": observation.id,
            "raw_hash": observation.raw_hash,
            "source_digest": _digest(expected_source),
            "claim_id": claim.id,
            "statement_digest": _digest_text(claim.statement),
        }
    )


def _delegated_contract(
    *,
    run: RunRow,
    parent_events: tuple[RunEventRow, ...],
    observations: tuple[ObservationRow, ...],
    claims: tuple[ClaimRow, ...],
    snapshot: PlanSnapshot | None,
    child_tasks: tuple[TaskRow, ...],
    child_runs: tuple[RunRow, ...],
    child_events: tuple[RunEventRow, ...],
    child_observations: tuple[ObservationRow, ...],
    child_claims: tuple[ClaimRow, ...],
    child_outbox: tuple[DeliveryOutboxRow, ...],
) -> tuple[str, dict[str, object]]:
    if snapshot is None or len(observations) != 1 or not claims:
        raise LiveProofIntegrityError(
            "delegated proof lacks its parent plan observation and claims"
        )
    source_claims = tuple(item for item in claims if item.kind == "source_claim")
    epistemic_claims = tuple(
        item for item in claims if item.kind in {"affected_assumption", "uncertainty"}
    )
    if (
        not source_claims
        or len(source_claims) + len(epistemic_claims) != len(claims)
        or any(
            sum(item.kind == kind for item in epistemic_claims) > 1
            for kind in ("affected_assumption", "uncertainty")
        )
    ):
        raise LiveProofIntegrityError("delegated proof claim taxonomy is invalid")
    current_nodes = snapshot.current_nodes
    completed_delegations = tuple(
        item
        for item in snapshot.delegations
        if item.revision_id == snapshot.revisions[-1].id
        and item.status is DelegationStatus.COMPLETED
    )
    if (
        len(current_nodes) < 2
        or any(item.status is not PlanNodeStatus.COMPLETED for item in current_nodes)
        or len(completed_delegations) < 2
        or child_outbox
    ):
        raise LiveProofIntegrityError("delegated proof lacks two parent-owned completed children")
    parent_observation = observations[0]
    if (
        parent_observation.kind != "agent.execute_research_plan"
        or parent_observation.status != "retrieved"
        or parent_observation.quality != "verified_child"
        or parent_observation.schema_version != "observation-v2"
        or parent_observation.normalization_version != "normalization-v1"
        or parent_observation.rejection_code is not None
        or parent_observation.strategy_id != run.strategy_id
        or parent_observation.observed_at > run.updated_at
        or parent_observation.source.get("provider") != "leo-subagent-plan"
        or parent_observation.source.get("reference") != snapshot.plan.id
        or parent_observation.data.get("status") != "completed"
        or not isinstance(parent_observation.data.get("completed_count"), int)
        or cast(int, parent_observation.data["completed_count"]) < 2
        or parent_observation.data.get("failed_count") != 0
        or parent_observation.data.get("blocked_count") != 0
        or any(
            tuple(str(value) for value in item.observation_ids) != (parent_observation.id,)
            or not _contains_statement(cast(str, run.final_output), item.statement)
            for item in source_claims
        )
        or any(
            tuple(str(value) for value in item.observation_ids) != (parent_observation.id,)
            or not item.statement.strip()
            for item in epistemic_claims
        )
    ):
        raise LiveProofIntegrityError("delegated parent synthesis is not grounded in its plan")
    _require_tool_path(parent_events, parent_observation, "agent.execute_research_plan")
    tasks_by_id = {item.id: item for item in child_tasks}
    runs_by_id = {item.id: item for item in child_runs}
    observations_by_run: dict[str, list[ObservationRow]] = {}
    claims_by_run: dict[str, list[ClaimRow]] = {}
    events_by_run: dict[str, list[RunEventRow]] = {}
    for observation_row in child_observations:
        observations_by_run.setdefault(observation_row.run_id, []).append(observation_row)
    for claim_row in child_claims:
        claims_by_run.setdefault(claim_row.run_id, []).append(claim_row)
    for event_row in child_events:
        events_by_run.setdefault(event_row.run_id, []).append(event_row)
    provider_run_ids: dict[str, str] = {}
    envelope_digests: list[str] = []
    verified_child_statements: set[str] = set()
    child_terminal_times: dict[str, datetime] = {}
    delegations_by_node = {item.node_id: item for item in completed_delegations}
    if set(delegations_by_node) != {item.id for item in current_nodes}:
        raise LiveProofIntegrityError("delegated nodes lack exact completed delegation rows")
    for node in current_nodes:
        if node.child_task_id is None or node.child_run_id is None or node.output is None:
            raise LiveProofIntegrityError("delegated node lacks its durable child identity")
        child_task = tasks_by_id.get(node.child_task_id)
        child_run = runs_by_id.get(node.child_run_id)
        delegation = delegations_by_node.get(node.id)
        if (
            child_task is None
            or child_run is None
            or delegation is None
            or delegation.child_task_id != node.child_task_id
            or delegation.child_run_id != node.child_run_id
            or child_task.parent_task_id != snapshot.plan.parent_task_id
            or child_task.continuation_kind != "subagent"
            or child_task.status != "completed"
            or child_run.task_id != child_task.id
            or child_run.status != "completed"
            or child_run.terminal_reason != "verified_completion"
            or not child_run.final_output
        ):
            raise LiveProofIntegrityError("delegated child task/run authority is incomplete")
        try:
            envelope = parse_child_evidence_envelope(node.output)
        except ChildEvidenceError as exc:
            raise LiveProofIntegrityError(
                "delegated node output is not a verified envelope"
            ) from exc
        if envelope.child_run_id != child_run.id or envelope.answer != child_run.final_output:
            raise LiveProofIntegrityError("delegated child envelope diverges from its durable run")
        child_obs = observations_by_run.get(child_run.id, [])
        child_claim_rows = claims_by_run.get(child_run.id, [])
        child_timeline = sorted(events_by_run.get(child_run.id, []), key=lambda item: item.sequence)
        child_obs_by_id = {item.id: item for item in child_obs}
        child_claims_by_id = {item.id: item for item in child_claim_rows}
        envelope_source_ids: set[str] = set()
        envelope_claim_ids: set[str] = set()
        for verified_claim in envelope.verified_source_claims:
            persisted_claim = child_claims_by_id.get(verified_claim.claim_id)
            if (
                persisted_claim is None
                or persisted_claim.kind != "source_claim"
                or persisted_claim.statement != verified_claim.statement
            ):
                raise LiveProofIntegrityError("delegated envelope claim was not durably verified")
            envelope_claim_ids.add(verified_claim.claim_id)
            verified_child_statements.add(_normalize_statement(verified_claim.statement))
            source_ids = tuple(item.observation_id for item in verified_claim.sources)
            if tuple(str(item) for item in persisted_claim.observation_ids) != source_ids:
                raise LiveProofIntegrityError("delegated envelope claim sources diverge")
            for source in verified_claim.sources:
                persisted = child_obs_by_id.get(source.observation_id)
                if (
                    persisted is None
                    or persisted.kind != source.kind
                    or persisted.source.get("provider") != source.provider
                    or persisted.source.get("reference") != source.reference
                    or persisted.source.get("url") != source.url
                    or persisted.observed_at != source.observed_at
                    or persisted.expires_at != source.expires_at
                    or persisted.raw_hash != source.raw_hash
                    or persisted.status != "retrieved"
                    or persisted.schema_version != "observation-v2"
                    or persisted.normalization_version != "normalization-v1"
                    or persisted.rejection_code is not None
                    or persisted.strategy_id != child_run.strategy_id
                    or persisted.observed_at > child_run.updated_at
                    or persisted.expires_at is None
                    or persisted.expires_at <= child_run.updated_at
                ):
                    raise LiveProofIntegrityError(
                        "delegated envelope source is stale or fabricated"
                    )
                if (source.kind, source.provider) == ("market.get_quote", "finnhub"):
                    if persisted.quality != "provider_reported":
                        raise LiveProofIntegrityError("delegated quote evidence quality is invalid")
                    provider_run_ids["market"] = child_run.id
                elif (source.kind, source.provider) == (
                    "sec.get_recent_filings",
                    "sec-edgar",
                ):
                    if persisted.quality != "primary_source":
                        raise LiveProofIntegrityError("delegated SEC evidence quality is invalid")
                    provider_run_ids["sec"] = child_run.id
                else:
                    raise LiveProofIntegrityError("delegated envelope carries forbidden evidence")
                envelope_source_ids.add(source.observation_id)
                _require_tool_path(tuple(child_timeline), persisted, source.kind)
        persisted_source_claim_ids = {
            item.id for item in child_claim_rows if item.kind == "source_claim"
        }
        if (
            envelope_source_ids != set(child_obs_by_id)
            or envelope_claim_ids != persisted_source_claim_ids
        ):
            raise LiveProofIntegrityError("delegated child has uncited or absent provider evidence")
        if (
            not child_timeline
            or child_timeline[-1].type != EventType.RUN_COMPLETED.value
            or not any(item.type == EventType.VERIFICATION_PASSED.value for item in child_timeline)
        ):
            raise LiveProofIntegrityError("delegated child lacks a verified terminal timeline")
        child_terminal_times[child_run.id] = child_timeline[-1].occurred_at
        envelope_digests.append(envelope.digest)
    parent_claim_statements = {_normalize_statement(item.statement) for item in source_claims}
    if not verified_child_statements or parent_claim_statements != verified_child_statements:
        raise LiveProofIntegrityError("delegated parent claims diverge from verified child claims")
    if set(provider_run_ids) != {"market", "sec"} or len(set(provider_run_ids.values())) != 2:
        raise LiveProofIntegrityError("delegated proof lacks distinct market and SEC children")
    market_run = runs_by_id[provider_run_ids["market"]]
    sec_run = runs_by_id[provider_run_ids["sec"]]
    if (
        market_run.started_at is None
        or sec_run.started_at is None
        or max(market_run.started_at, sec_run.started_at)
        >= min(
            child_terminal_times[market_run.id],
            child_terminal_times[sec_run.id],
        )
    ):
        raise LiveProofIntegrityError("delegated market and SEC child intervals did not overlap")
    parent_nodes = parent_observation.data.get("nodes")
    if not isinstance(parent_nodes, list):
        raise LiveProofIntegrityError("delegated parent observation lacks node envelopes")
    observed_envelope_digests = {
        evidence.get("digest")
        for item in parent_nodes
        if isinstance(item, dict) and isinstance((evidence := item.get("child_evidence")), dict)
    }
    if observed_envelope_digests != set(envelope_digests):
        raise LiveProofIntegrityError("delegated parent observation diverges from plan nodes")
    child_expiries = tuple(
        source.expires_at
        for node in current_nodes
        for claim in parse_child_evidence_envelope(cast(str, node.output)).verified_source_claims
        for source in claim.sources
        if source.expires_at is not None
    )
    if (
        not child_expiries
        or parent_observation.expires_at != min(child_expiries)
        or parent_observation.expires_at <= run.updated_at
    ):
        raise LiveProofIntegrityError("delegated parent evidence freshness diverges from children")
    evidence_digest = _digest(
        {
            "plan_id": snapshot.plan.id,
            "revision_digest": snapshot.revisions[-1].digest,
            "parent_observation_id": parent_observation.id,
            "parent_claim_ids": [item.id for item in claims],
            "child_run_ids": sorted(item.id for item in child_runs),
            "child_envelope_digests": sorted(envelope_digests),
            "providers": provider_run_ids,
            "overlap": True,
        }
    )
    return (
        _digest({"contract": "delegated-parallel-evidence-v1", "digest": evidence_digest}),
        {
            "delegated_child_count": len(current_nodes),
            "delegated_overlap_verified": True,
            "delegated_evidence_digest": evidence_digest,
        },
    )


def _require_tool_path(
    events: tuple[RunEventRow, ...],
    observation: ObservationRow,
    tool_name: str,
) -> None:
    matching = [
        item for item in events if item.payload.get("tool_call_id") == observation.tool_call_id
    ]
    tool_events = [item for item in matching if item.payload.get("tool") == tool_name]
    started = [item for item in tool_events if item.type == EventType.TOOL_STARTED.value]
    completed = [item for item in tool_events if item.type == EventType.TOOL_COMPLETED.value]
    failed = [item for item in tool_events if item.type == EventType.TOOL_FAILED.value]
    observation_events = [
        item
        for item in matching
        if item.type == EventType.OBSERVATION_CREATED.value
        and item.payload.get("observation_id") == observation.id
    ]
    if (
        len(started) != 1
        or len(completed) != 1
        or failed
        or len(observation_events) != 1
        or not (started[0].sequence < completed[0].sequence < observation_events[0].sequence)
    ):
        raise LiveProofIntegrityError("live proof observation lacks its exact completed tool path")


def _contains_statement(text: str, statement: str) -> bool:
    normalized_text = _normalize_statement(text)
    normalized_statement = _normalize_statement(statement)
    return bool(normalized_statement) and normalized_statement in normalized_text


def _normalize_statement(value: str) -> str:
    return " ".join(value.split()).casefold()


def _plan_summary(
    binding: LiveProofBinding,
    snapshot: PlanSnapshot | None,
) -> dict[str, object]:
    if binding.plan_expectation == "required" and snapshot is None:
        raise LiveProofIntegrityError("live proof requires a durable plan")
    if binding.plan_expectation == "forbidden" and snapshot is not None:
        raise LiveProofIntegrityError("live proof unexpectedly created a durable plan")
    if snapshot is None:
        empty_digest = _digest([])
        return {
            "plan_present": False,
            "plan_terminal_state": "absent",
            "plan_revision_digest": empty_digest,
            "plan_snapshot_digest": empty_digest,
            "plan_node_count": 0,
            "delegation_count": 0,
        }
    current = snapshot.revisions[-1]
    return {
        "plan_present": True,
        "plan_terminal_state": snapshot.plan.status.value,
        "plan_revision_digest": current.digest,
        "plan_snapshot_digest": _digest(
            {
                "plan_id": snapshot.plan.id,
                "parent_task_id": snapshot.plan.parent_task_id,
                "parent_run_id": snapshot.plan.parent_run_id,
                "status": snapshot.plan.status.value,
                "current_revision": snapshot.plan.current_revision,
                "revision_digests": [item.digest for item in snapshot.revisions],
                "nodes": [
                    {
                        "id": item.id,
                        "revision_number": item.revision_number,
                        "node_key": item.definition.key,
                        "depends_on": list(item.definition.depends_on),
                        "status": item.status.value,
                        "attempt": item.attempt,
                        "child_task_id": item.child_task_id,
                        "child_run_id": item.child_run_id,
                        "output_digest": _digest_text(item.output) if item.output else None,
                        "error_digest": _digest_text(item.error) if item.error else None,
                    }
                    for item in snapshot.nodes
                ],
                "delegations": [
                    {
                        "id": item.id,
                        "node_id": item.node_id,
                        "attempt": item.attempt,
                        "status": item.status.value,
                        "child_task_id": item.child_task_id,
                        "child_run_id": item.child_run_id,
                    }
                    for item in snapshot.delegations
                ],
            }
        ),
        "plan_node_count": len(snapshot.nodes),
        "delegation_count": len(snapshot.delegations),
    }


def _require_one[T](items: Sequence[T]) -> T:
    if len(items) != 1:
        raise LiveProofNotFound
    return items[0]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _slack_ts_value(value: str) -> int:
    seconds, micros = value.split(".", maxsplit=1)
    return int(seconds) * 1_000_000 + int(micros)


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"unsupported proof value: {type(value).__name__}")


def main() -> int:
    """Fail closed when invoked without a trusted operator composition root."""

    print(
        json.dumps(
            {
                "code": "trusted_live_proof_composition_required",
                "status": "unavailable",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 2


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
