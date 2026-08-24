"""Versioned, redacted event envelopes for replay and evaluation traces."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import datetime
from enum import StrEnum
from itertools import pairwise

from pydantic import Field, JsonValue, model_validator

from leo.harness.models import (
    ClaimKind,
    ContractModel,
    EventType,
    EvidenceQuality,
    NonEmptyStr,
    ObservationStatus,
    RunEvent,
    ScopeKey,
)


class EventKind(StrEnum):
    INGRESS_ADMITTED = "ingress_admitted"
    CONVERSATION_RESOLVED = "conversation_resolved"
    MEMBERSHIP_SOURCE_SET = "membership_source_set"
    AUTHORITY_RESOLVED = "authority_resolved"
    LIFECYCLE = "lifecycle"
    CONTEXT_RETRIEVED = "context_retrieved"
    RETRIEVAL = "context_retrieved"  # Compatibility alias.
    CONTEXT_BUILT = "context_built"
    MEMORY_RETRIEVED = "memory_retrieved"
    MODEL_CALLED = "model_called"
    PLAN_REVISION = "plan_revision"
    PLAN_VALIDATED = "plan_revision"  # Compatibility alias.
    PLAN_NODE = "plan_node"
    DELEGATION = "delegation"
    TOOL_CALL = "tool_call"
    TOOL_COMPLETED = "tool_call"  # Compatibility alias.
    EVIDENCE_NORMALIZED = "evidence_normalized"
    CONFLICT_DETECTED = "conflict_detected"
    SYNTHESIS = "synthesis"
    MEMORY_COMMITTED = "memory_committed"
    VERIFICATION = "verification"
    DELIVERY = "delivery"
    USAGE = "usage"
    TERMINAL = "terminal"


class _PayloadBase(ContractModel):
    status: str | None = None
    code: str | None = None
    reason: str | None = None
    version: str | None = None
    count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def has_observable_field(self) -> _PayloadBase:
        if not self.model_fields_set:
            raise ValueError("typed event payload cannot be empty")
        return self


class IngressPayload(_PayloadBase):
    ingress_id: str | None = None
    destination_id: str | None = None
    destination_kind: str | None = None
    external_event_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class ConversationPayload(_PayloadBase):
    destination_id: str | None = None
    destination_kind: str | None = None
    conversation_id_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    thread_id_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    external_provenance: str | None = None


class MembershipPayload(_PayloadBase):
    source_ids: tuple[str, ...] = ()
    source_set_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    access_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    actor_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AuthorityPayload(_PayloadBase):
    actor_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    roles_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    organization_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LifecyclePayload(_PayloadBase):
    task_status: str | None = None
    run_status: str | None = None
    terminal_reason: str | None = None
    iteration: int | None = Field(default=None, ge=0)


class RetrievalPayload(_PayloadBase):
    source_ids: tuple[str, ...] = ()
    source_set_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    query_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    access_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_count: int | None = Field(default=None, ge=0)
    excluded_count: int | None = Field(default=None, ge=0)


class ContextPayload(_PayloadBase):
    manifest_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_ids: tuple[str, ...] = ()
    included_tokens: int | None = Field(default=None, ge=0)
    excluded_tokens: int | None = Field(default=None, ge=0)


class ModelPayload(_PayloadBase):
    provider: str | None = None
    model: str | None = None
    request_id: str | None = None
    finish_reason: str | None = None
    parent_run_id: str | None = None
    node_id: str | None = None
    iteration: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    text: JsonValue | None = None


class PlanRevisionPayload(_PayloadBase):
    plan_id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    node_count: int | None = Field(default=None, ge=0)
    replan_count: int | None = Field(default=None, ge=0)


class PlanNodePayload(_PayloadBase):
    plan_id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    node_id: str | None = None
    dependency_ids: tuple[str, ...] = ()
    attempt: int | None = Field(default=None, ge=0)
    child_task_id: str | None = None
    child_run_id: str | None = None


class DelegationPayload(_PayloadBase):
    plan_id: str | None = None
    node_id: str | None = None
    parent_task_id: str | None = None
    parent_run_id: str | None = None
    child_task_id: str | None = None
    child_run_id: str | None = None
    effect: str | None = None


class ToolPayload(_PayloadBase):
    tool_id: str | None = None
    tool_call_id: str | None = None
    attempt: int | None = Field(default=None, ge=0)
    effect: str | None = None
    observation_id: str | None = None
    normalization_code: str | None = None


class EvidencePayload(_PayloadBase):
    observation_id: str | None = None
    observation_kind: str | None = None
    observation_schema_version: str | None = Field(default=None, pattern=r"^observation-v[0-9]+$")
    observation_status: ObservationStatus | None = None
    evidence_quality: EvidenceQuality | None = None
    normalization_version: str | None = None
    rejection_code: str | None = None
    quality_status: str | None = None  # Compatibility field for pre-v2 producers.
    raw_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_ids: tuple[str, ...] = ()
    claim_kinds: tuple[ClaimKind, ...] = ()
    affected_assumption: bool | None = None
    uncertainty: bool | None = None

    @model_validator(mode="after")
    def observation_state_is_compatible(self) -> EvidencePayload:
        if self.observation_schema_version not in {None, "observation-v1", "observation-v2"}:
            raise ValueError("observation schema version is unsupported")
        if self.observation_status is ObservationStatus.REJECTED and self.rejection_code is None:
            raise ValueError("rejected evidence requires a rejection code")
        if (
            self.observation_status is not ObservationStatus.REJECTED
            and self.rejection_code is not None
        ):
            raise ValueError("only rejected evidence may carry a rejection code")
        return self


class ConflictPayload(_PayloadBase):
    conflict_id: str | None = None
    source_ids: tuple[str, ...] = ()
    affected_assumption: bool | None = None
    uncertainty: bool | None = None


class SynthesisPayload(_PayloadBase):
    source_ids: tuple[str, ...] = ()
    claim_kinds: tuple[ClaimKind, ...] = ()
    answer_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_authority: bool | None = None


class MemoryPayload(_PayloadBase):
    memory_id: str | None = None
    revision: int | None = Field(default=None, ge=1)
    content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_ids: tuple[str, ...] = ()
    visibility: str | None = None


class VerificationPayload(_PayloadBase):
    verifier_version: str | None = None
    check_count: int | None = Field(default=None, ge=0)
    claim_kinds: tuple[ClaimKind, ...] = ()
    affected_assumption: bool | None = None
    uncertainty: bool | None = None


class DeliveryPayload(_PayloadBase):
    outbox_id: str | None = None
    destination_id: str | None = None
    idempotency_key_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    attempt: int | None = Field(default=None, ge=0)
    unknown_effect: bool | None = None


class UsagePayload(_PayloadBase):
    model_calls: int | None = Field(default=None, ge=0)
    tool_calls: int | None = Field(default=None, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)


class TerminalPayload(_PayloadBase):
    task_status: str | None = None
    run_status: str | None = None
    terminal_reason: str | None = None
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    parent_authority: bool | None = None


_PAYLOAD_MODELS: dict[EventKind, type[_PayloadBase]] = {
    EventKind.INGRESS_ADMITTED: IngressPayload,
    EventKind.CONVERSATION_RESOLVED: ConversationPayload,
    EventKind.MEMBERSHIP_SOURCE_SET: MembershipPayload,
    EventKind.AUTHORITY_RESOLVED: AuthorityPayload,
    EventKind.LIFECYCLE: LifecyclePayload,
    EventKind.CONTEXT_RETRIEVED: RetrievalPayload,
    EventKind.CONTEXT_BUILT: ContextPayload,
    EventKind.MEMORY_RETRIEVED: RetrievalPayload,
    EventKind.MODEL_CALLED: ModelPayload,
    EventKind.PLAN_REVISION: PlanRevisionPayload,
    EventKind.PLAN_NODE: PlanNodePayload,
    EventKind.DELEGATION: DelegationPayload,
    EventKind.TOOL_CALL: ToolPayload,
    EventKind.EVIDENCE_NORMALIZED: EvidencePayload,
    EventKind.CONFLICT_DETECTED: ConflictPayload,
    EventKind.SYNTHESIS: SynthesisPayload,
    EventKind.MEMORY_COMMITTED: MemoryPayload,
    EventKind.VERIFICATION: VerificationPayload,
    EventKind.DELIVERY: DeliveryPayload,
    EventKind.USAGE: UsagePayload,
    EventKind.TERMINAL: TerminalPayload,
}
EVENT_SCHEMA_VERSION = "v2"
SUPPORTED_EVENT_SCHEMA_VERSIONS = frozenset({"v1", EVENT_SCHEMA_VERSION})
SUPPORTED_RUN_EVENT_SCHEMA_VERSIONS = frozenset({1})


class EventEnvelope(ContractModel):
    event_id: NonEmptyStr
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    scope: ScopeKey
    sequence: int = Field(ge=0)
    occurred_at: datetime
    kind: EventKind
    schema_version: str = Field(pattern=r"^v[0-9]+$")
    correlation_id: NonEmptyStr
    causation_id: str | None = None
    payload: dict[str, JsonValue]

    @model_validator(mode="after")
    def bounded_redacted_payload(self) -> EventEnvelope:
        if self.schema_version not in SUPPORTED_EVENT_SCHEMA_VERSIONS:
            raise ValueError("event schema version is unsupported")
        encoded = json.dumps(self.payload, ensure_ascii=False, sort_keys=True)
        if len(encoded.encode("utf-8")) > 8192:
            raise ValueError("event payload exceeds 8192 bytes")
        if any(_looks_like_secret(str(value)) for value in _walk_values(self.payload)):
            raise ValueError("event payload contains a secret-like value")
        if self.schema_version == EVENT_SCHEMA_VERSION:
            try:
                _PAYLOAD_MODELS[self.kind].model_validate(self.payload)
            except ValueError as exc:
                raise ValueError("event payload does not match its typed kind") from exc
        elif set(self.payload) - _V1_ALLOWED_FIELDS:
            raise ValueError("legacy event payload contains an unknown field")
        return self


class EventSchemaError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def build_event(
    *,
    event_id: str,
    run_id: str,
    task_id: str,
    scope: ScopeKey,
    sequence: int,
    occurred_at: datetime,
    kind: EventKind,
    correlation_id: str,
    payload: dict[str, JsonValue],
    causation_id: str | None = None,
) -> EventEnvelope:
    clean_payload = redact_payload(payload)
    try:
        typed_payload = _PAYLOAD_MODELS[kind].model_validate(clean_payload)
    except ValueError as exc:
        raise EventSchemaError("event_payload_field_not_allowed") from exc
    return EventEnvelope(
        event_id=event_id,
        run_id=run_id,
        task_id=task_id,
        scope=scope,
        sequence=sequence,
        occurred_at=occurred_at,
        kind=kind,
        schema_version=EVENT_SCHEMA_VERSION,
        correlation_id=correlation_id,
        causation_id=causation_id,
        payload=typed_payload.model_dump(mode="json", exclude_none=True),
    )


def parse_event(value: EventEnvelope | dict[str, object]) -> EventEnvelope:
    """Read current events and migrate the bounded legacy v1 envelope to v2."""

    event = value if isinstance(value, EventEnvelope) else EventEnvelope.model_validate(value)
    if event.schema_version == EVENT_SCHEMA_VERSION:
        return event
    model = _PAYLOAD_MODELS[event.kind]
    fields = set(model.model_fields)
    migrated = {key: item for key, item in event.payload.items() if key in fields}
    if not migrated:
        migrated = {"status": "legacy_event"}
    try:
        payload = model.model_validate(migrated).model_dump(mode="json", exclude_none=True)
    except ValueError as exc:
        raise EventSchemaError("event_legacy_payload_incompatible") from exc
    return event.model_copy(update={"schema_version": EVENT_SCHEMA_VERSION, "payload": payload})


def normalize_run_event(
    event: RunEvent,
    scope: ScopeKey,
    *,
    correlation_id: str | None = None,
    causation_id: str | None = None,
) -> EventEnvelope:
    """Project one durable v1 RunEvent into the strict universal v2 envelope."""

    if event.schema_version not in SUPPORTED_RUN_EVENT_SCHEMA_VERSIONS:
        raise EventSchemaError("run_event_schema_version_unsupported")
    kind = RUN_EVENT_KIND[event.type]
    payload = _normalize_run_payload(event, kind)
    return build_event(
        event_id=event.id,
        run_id=event.run_id,
        task_id=event.task_id,
        scope=scope,
        sequence=event.sequence,
        occurred_at=event.occurred_at,
        kind=kind,
        correlation_id=correlation_id or event.run_id,
        causation_id=causation_id,
        payload=payload,
    )


def normalize_run_timeline(
    events: Iterable[RunEvent],
    scope: ScopeKey,
) -> tuple[EventEnvelope, ...]:
    """Normalize a single exact, contiguous durable timeline for replay/export."""

    ordered = tuple(events)
    if not ordered:
        return ()
    if len({(item.run_id, item.task_id) for item in ordered}) != 1:
        raise EventSchemaError("run_event_timeline_identity_mismatch")
    if tuple(item.sequence for item in ordered) != tuple(range(1, len(ordered) + 1)):
        raise EventSchemaError("run_event_timeline_sequence_gap")
    if any(current.occurred_at < previous.occurred_at for previous, current in pairwise(ordered)):
        raise EventSchemaError("run_event_timeline_time_reversed")
    output: list[EventEnvelope] = []
    for index, event in enumerate(ordered):
        output.append(
            normalize_run_event(
                event,
                scope,
                correlation_id=ordered[0].run_id,
                causation_id=ordered[index - 1].id if index else None,
            )
        )
    return tuple(output)


def redact_payload(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for key, value in payload.items():
        if key.lower() in {"prompt", "content", "body", "text", "raw", "headers"}:
            output[key] = {
                "sha256": hashlib.sha256(str(value).encode()).hexdigest(),
                "length": len(str(value)),
            }
        else:
            output[key] = _redact_value(value)
    return output


_V1_ALLOWED_FIELDS = frozenset(
    {
        "code",
        "reason",
        "count",
        "version",
        "status",
        "source_ids",
        "tool_id",
        "cost",
        "text",
    }
)
RUN_EVENT_KIND: dict[EventType, EventKind] = {
    EventType.TASK_STARTED: EventKind.LIFECYCLE,
    EventType.CONTEXT_BUILT: EventKind.CONTEXT_BUILT,
    EventType.MODEL_CALLED: EventKind.MODEL_CALLED,
    EventType.MODEL_BUDGET_RESERVED: EventKind.USAGE,
    EventType.TOOL_STARTED: EventKind.TOOL_CALL,
    EventType.TOOL_COMPLETED: EventKind.TOOL_CALL,
    EventType.TOOL_FAILED: EventKind.TOOL_CALL,
    EventType.OBSERVATION_CREATED: EventKind.EVIDENCE_NORMALIZED,
    EventType.VERIFICATION_STARTED: EventKind.VERIFICATION,
    EventType.VERIFICATION_FAILED: EventKind.VERIFICATION,
    EventType.VERIFICATION_PASSED: EventKind.VERIFICATION,
    # Answering with committed steps still undone is a planning outcome, not a
    # verifier judgement: the answer was never verified, the run just continued.
    EventType.PLAN_STEP_OUTSTANDING: EventKind.PLAN_REVISION,
    EventType.RUN_COMPLETED: EventKind.TERMINAL,
    EventType.RUN_REQUIRES_ACTION: EventKind.LIFECYCLE,
    EventType.RUN_RESUMED: EventKind.LIFECYCLE,
    EventType.RUN_REQUEUED: EventKind.LIFECYCLE,
    EventType.RUN_CANCELLED: EventKind.TERMINAL,
    EventType.RUN_FAILED: EventKind.TERMINAL,
    EventType.RUN_TIMED_OUT: EventKind.TERMINAL,
    EventType.BUDGET_EXHAUSTED: EventKind.TERMINAL,
}
UNIVERSAL_EVENT_INVENTORY: tuple[EventKind, ...] = tuple(EventKind)


def validate_event_inventory() -> None:
    if set(_PAYLOAD_MODELS) != set(EventKind):
        raise EventSchemaError("event_kind_missing_typed_payload")
    if set(RUN_EVENT_KIND) != set(EventType):
        raise EventSchemaError("run_event_type_missing_inventory_mapping")
    claim_kinds = set(ClaimKind)
    if not {
        ClaimKind.SOURCE_CLAIM,
        ClaimKind.INFERENCE,
        ClaimKind.AFFECTED_ASSUMPTION,
        ClaimKind.UNCERTAINTY,
    }.issubset(claim_kinds):
        raise EventSchemaError("claim_kind_missing_from_event_inventory")


def event_contract_digest() -> str:
    validate_event_inventory()
    payload = {
        "schema_version": EVENT_SCHEMA_VERSION,
        "kinds": {kind.value: sorted(_PAYLOAD_MODELS[kind].model_fields) for kind in EventKind},
        "run_event_mapping": {
            event_type.value: kind.value
            for event_type, kind in sorted(RUN_EVENT_KIND.items(), key=lambda item: item[0].value)
        },
        "claim_kinds": sorted(kind.value for kind in ClaimKind),
        "observation_statuses": sorted(item.value for item in ObservationStatus),
        "evidence_qualities": sorted(item.value for item in EvidenceQuality),
        "supported_run_event_schema_versions": sorted(SUPPORTED_RUN_EVENT_SCHEMA_VERSIONS),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


_SECRET_PATTERN = re.compile(r"(?i)\b(?:bearer|token|secret|password|api[_-]?key)\s*[:=]\s*\S+")


def _redact_value(value: JsonValue) -> JsonValue:
    if isinstance(value, str):
        return _SECRET_PATTERN.sub("[REDACTED]", value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_value(item) for key, item in value.items()}
    return value


def _walk_values(value: JsonValue) -> Iterator[JsonValue]:
    if isinstance(value, dict):
        for nested in value.values():
            yield from _walk_values(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested)
    else:
        yield value


def _looks_like_secret(value: str) -> bool:
    return bool(re.search(r"(?i)\b(?:xox[baprs]-|sk-|gh[pousr]_)\S+", value))


def _normalize_run_payload(event: RunEvent, kind: EventKind) -> dict[str, JsonValue]:
    raw = dict(event.payload)
    payload: dict[str, JsonValue] = {"status": event.type.value}
    model_fields = set(_PAYLOAD_MODELS[kind].model_fields)
    for key, value in raw.items():
        if key in model_fields:
            payload[key] = value
    if "tool" in raw and "tool_id" in model_fields:
        payload["tool_id"] = raw["tool"]
    if "estimated_cost" in raw and "cost" in model_fields:
        payload["cost"] = raw["estimated_cost"]
    if "segments" in raw and "count" in model_fields:
        segments = raw["segments"]
        if isinstance(segments, int) and segments >= 0:
            payload["count"] = segments
        elif isinstance(segments, list):
            payload["count"] = len(segments)
    if "claim_count" in raw and "count" in model_fields:
        payload["count"] = raw["claim_count"]
    if "failed_checks" in raw and "check_count" in model_fields:
        failed_checks = raw["failed_checks"]
        if isinstance(failed_checks, list):
            payload["check_count"] = len(failed_checks)
    return payload
