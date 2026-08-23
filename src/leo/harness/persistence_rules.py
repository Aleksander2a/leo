"""Lifecycle invariants shared by every run-store adapter."""

from __future__ import annotations

import hashlib
import json
import re
from typing import cast

from pydantic import JsonValue

from leo.harness.models import (
    LEGAL_TASK_RUN_PAIRS,
    BudgetUsage,
    ClaimKind,
    ContractModel,
    EventDraft,
    EventType,
    Observation,
    Run,
    RunStatus,
    Task,
    TaskStatus,
    Thread,
    VerifiedCompletion,
    VerifierStatus,
)
from leo.harness.store_errors import StoreError

EVENT_PAYLOAD_MAX_BYTES = 8192
CONTEXT_BUILT_PROJECTION_VERSION = "context-built-v1"
VERIFICATION_CHECK_PROJECTION_VERSION = "verification-checks-v1"
_PROJECTED_CHECK_NAME_MAX_BYTES = 256
_CORRELATION_FIELDS = frozenset(
    {
        "run_id",
        "task_id",
        "sequence",
        "scope",
        "organization_id",
        "strategy_id",
        "occurred_at",
        "schema_version",
    }
)
_SENSITIVE_KEY_MARKERS = (
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "cookie",
    "header",
    "connection_string",
    "dsn",
)
_SENSITIVE_KEY_EXACT = frozenset({"prompt"})
_SENSITIVE_VALUE_PATTERNS = (
    re.compile(r"(?i)\b(?:authorization|proxy-authorization)\s*:\s*(?:bearer|basic)\s+\S+"),
    re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+"),
    re.compile(r"(?i:\btraceback\b)|\b[A-Z][A-Za-z0-9_.]*(?:Error|Exception)\b"),
    re.compile(r"\b(?:xox[baprs]-|sk-)[A-Za-z0-9_-]+"),
)
_EVENT_PAYLOAD_FIELDS: dict[EventType, frozenset[str]] = {
    EventType.TASK_STARTED: frozenset({"phase"}),
    EventType.CONTEXT_BUILT: frozenset(
        {
            "segments",
            "tool_count",
            "tool_choice",
            "required_tool",
            "required_arguments",
            "completion_contract",
            "source_manifest",
            "catalog_version",
            "catalog_fingerprint",
            "selection_fingerprint",
            "selection_mode",
            "selection_reason",
            "capability_candidates",
            "capability_selected",
            "skill_selected",
            "capability_query_hash",
            "eligible_capability_count",
            "projection",
        }
    ),
    EventType.MODEL_CALLED: frozenset(
        {
            "decision",
            "provider",
            "model",
            "request_id",
            "finish_reason",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost",
        }
    ),
    EventType.MODEL_BUDGET_RESERVED: frozenset({"reservation_id", "estimated_cost"}),
    EventType.TOOL_STARTED: frozenset(
        {"tool_call_id", "tool", "arguments", "parallel_batch"}
    ),
    EventType.TOOL_COMPLETED: frozenset({"tool_call_id", "tool"}),
    EventType.TOOL_FAILED: frozenset(
        {"tool_call_id", "tool", "code", "retryable", "arguments"}
    ),
    EventType.OBSERVATION_CREATED: frozenset({"observation_id", "tool_call_id"}),
    EventType.VERIFICATION_STARTED: frozenset(),
    EventType.VERIFICATION_FAILED: frozenset({"failed_checks", "retryable", "checks"}),
    EventType.VERIFICATION_PASSED: frozenset(
        {"claim_count", "check_count", "checks", "projection"}
    ),
    EventType.RUN_COMPLETED: frozenset({"reason"}),
    EventType.RUN_REQUIRES_ACTION: frozenset({"reason"}),
    EventType.RUN_RESUMED: frozenset(),
    EventType.RUN_REQUEUED: frozenset(),
    EventType.RUN_CANCELLED: frozenset({"reason"}),
    EventType.RUN_FAILED: frozenset({"reason", "detail"}),
    EventType.RUN_TIMED_OUT: frozenset({"reason"}),
    EventType.BUDGET_EXHAUSTED: frozenset({"reason"}),
}


def validate_seed(thread: Thread, task: Task, run: Run) -> None:
    if task.thread_id != thread.id or thread.scope != task.scope:
        raise StoreError("thread and task identity mismatch")
    if run.task_id != task.id or run.scope != task.scope:
        raise StoreError("task and run identity mismatch")
    if task.status is not TaskStatus.QUEUED or run.status is not RunStatus.QUEUED:
        raise StoreError("new task and run must be queued")


def validate_commit(
    current_task: Task,
    current_run: Run,
    task: Task,
    run: Run,
    events: tuple[EventDraft, ...],
    observations: tuple[Observation, ...] = (),
) -> tuple[EventDraft, ...]:
    """Enforce non-completion lifecycle invariants at the atomic write boundary."""

    events = sanitize_event_drafts(events, run)
    if (
        task.id != current_task.id
        or task.thread_id != current_task.thread_id
        or task.scope != current_task.scope
        or task.objective != current_task.objective
        or task.parent_task_id != current_task.parent_task_id
        or task.continuation_kind != current_task.continuation_kind
        or task.mapping_version != current_task.mapping_version
    ):
        raise StoreError("task identity and objective are immutable")
    if (
        run.id != current_run.id
        or run.task_id != current_run.task_id
        or run.scope != current_run.scope
        or run.limits != current_run.limits
    ):
        raise StoreError("run identity and limits are immutable")
    if (
        current_run.started_at is not None
        and run.started_at != current_run.started_at
        and not (
            current_task.status is TaskStatus.REQUIRES_ACTION
            and current_run.status is RunStatus.REQUIRES_ACTION
            and task.status is TaskStatus.QUEUED
            and run.status is RunStatus.QUEUED
            and run.started_at is None
        )
    ):
        raise StoreError("run start time is immutable")
    if current_run.deadline_at is not None and run.deadline_at != current_run.deadline_at:
        raise StoreError("run deadline is immutable")
    if run.iteration < current_run.iteration:
        raise StoreError("run iteration cannot decrease")
    if (
        run.usage.model_calls < current_run.usage.model_calls
        or run.usage.tool_calls < current_run.usage.tool_calls
    ):
        raise StoreError("run usage cannot decrease")
    if (
        current_run.usage.cost is not None
        and run.usage.cost is not None
        and run.usage.cost < current_run.usage.cost
    ):
        raise StoreError("run cost usage cannot decrease")
    if (
        run.iteration > run.limits.max_iterations
        or run.usage.model_calls > run.limits.max_model_calls
        or run.usage.tool_calls > run.limits.max_tool_calls
    ):
        raise StoreError("run state exceeds its configured budget")
    if (
        run.status is not RunStatus.BUDGET_EXHAUSTED
        and run.limits.max_cost is not None
        and run.usage.cost is not None
    ):
        if run.usage.cost > run.limits.max_cost:
            raise StoreError("run state exceeds its configured cost budget")
    if (task.status, run.status) not in LEGAL_TASK_RUN_PAIRS:
        raise StoreError("invalid task/run lifecycle pair")
    if (
        run.status
        in {
            RunStatus.REQUIRES_ACTION,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
            RunStatus.BUDGET_EXHAUSTED,
        }
        and not run.terminal_reason
    ):
        raise StoreError("non-running run state requires a reason")
    if run.status in {RunStatus.QUEUED, RunStatus.RUNNING} and run.terminal_reason is not None:
        raise StoreError("queued/running run cannot have a terminal reason")
    if task.observation_ids[: len(current_task.observation_ids)] != current_task.observation_ids:
        raise StoreError("task observations are append-only")
    if (
        task.verifier_feedback[: len(current_task.verifier_feedback)]
        != current_task.verifier_feedback
    ):
        raise StoreError("verifier feedback is append-only")

    if current_task.status in {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
    } or current_run.status in {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
        RunStatus.BUDGET_EXHAUSTED,
    }:
        raise StoreError("terminal task or run is immutable")

    if current_task.status is TaskStatus.QUEUED and current_run.status is RunStatus.QUEUED:
        if (task.status, run.status) not in {
            (TaskStatus.ACTIVE, RunStatus.RUNNING),
            (TaskStatus.CANCELLED, RunStatus.CANCELLED),
        }:
            raise StoreError("queued task/run may only transition to active/running or cancelled")
    elif current_task.status is TaskStatus.ACTIVE and current_run.status is RunStatus.RUNNING:
        allowed: frozenset[tuple[TaskStatus, RunStatus]] = frozenset(
            {
                (TaskStatus.ACTIVE, RunStatus.RUNNING),
                (TaskStatus.REQUIRES_ACTION, RunStatus.REQUIRES_ACTION),
                (TaskStatus.FAILED, RunStatus.FAILED),
                (TaskStatus.FAILED, RunStatus.TIMED_OUT),
                (TaskStatus.FAILED, RunStatus.BUDGET_EXHAUSTED),
                (TaskStatus.CANCELLED, RunStatus.CANCELLED),
            }
        )
        if (task.status, run.status) not in allowed:
            raise StoreError("illegal active task/run transition")
    elif (
        current_task.status is TaskStatus.REQUIRES_ACTION
        and current_run.status is RunStatus.REQUIRES_ACTION
    ):
        allowed = frozenset(
            {
                (TaskStatus.REQUIRES_ACTION, RunStatus.REQUIRES_ACTION),
                (TaskStatus.ACTIVE, RunStatus.RUNNING),
                (TaskStatus.QUEUED, RunStatus.QUEUED),
                (TaskStatus.CANCELLED, RunStatus.CANCELLED),
            }
        )
        if (task.status, run.status) not in allowed:
            raise StoreError("illegal requires-action task/run transition")
    else:
        raise StoreError("unsupported task/run lifecycle pair")

    if not events and (
        _snapshot_changed(current_task, task) or _snapshot_changed(current_run, run) or observations
    ):
        raise StoreError("state or effect changes require at least one event")

    event_types = {event.type for event in events}
    reserved_completion_events = {EventType.VERIFICATION_PASSED, EventType.RUN_COMPLETED}
    if event_types.intersection(reserved_completion_events):
        raise StoreError("completion events require the verified-completion boundary")
    return events


def sanitize_event_drafts(
    events: tuple[EventDraft, ...],
    run: Run,
) -> tuple[EventDraft, ...]:
    sanitized: list[EventDraft] = []
    for draft in events:
        if draft.iteration > run.iteration:
            raise StoreError("event iteration cannot exceed run iteration")
        allowed_fields = _EVENT_PAYLOAD_FIELDS[draft.type]
        for field in draft.payload:
            if field in _CORRELATION_FIELDS:
                raise StoreError(f"event payload cannot provide correlation field: {field}")
            if field not in allowed_fields:
                raise StoreError(f"event payload field is not allowlisted: {field}")
        payload = {key: _sanitize_event_value(value, key) for key, value in draft.payload.items()}
        encoded_size = len(_canonical_json_bytes(payload))
        if encoded_size > EVENT_PAYLOAD_MAX_BYTES and draft.type is EventType.CONTEXT_BUILT:
            payload = _project_context_built_payload(payload)
            encoded_size = len(_canonical_json_bytes(payload))
        if encoded_size > EVENT_PAYLOAD_MAX_BYTES:
            raise StoreError("event payload exceeds the maximum size")
        sanitized.append(draft.model_copy(update={"payload": payload}))
    return tuple(sanitized)


def build_verification_passed_event(
    completion: VerifiedCompletion,
    run: Run,
) -> EventDraft:
    """Build the bounded store-owned audit event for a verified completion.

    The answer and claims remain canonical task/run/claim records. Small verifier
    results retain their complete sanitized checks. Oversized results use a
    deterministic, digested status-only projection so audit persistence cannot
    prevent an otherwise valid terminal transition.
    """

    payload: dict[str, JsonValue] = {
        "claim_count": len(completion.claims),
        "check_count": len(completion.verifier_result.checks),
        "checks": [check.model_dump(mode="json") for check in completion.verifier_result.checks],
    }
    safe_payload = {key: _sanitize_event_value(value, key) for key, value in payload.items()}
    if len(_canonical_json_bytes(safe_payload)) > EVENT_PAYLOAD_MAX_BYTES:
        safe_payload = _project_verification_checks(safe_payload)
    draft = EventDraft(
        type=EventType.VERIFICATION_PASSED,
        iteration=run.iteration,
        payload=safe_payload,
    )
    return sanitize_event_drafts((draft,), run)[0]


def _snapshot_changed(current: ContractModel, candidate: ContractModel) -> bool:
    current_dump = current.model_dump(mode="json", exclude={"version"})
    candidate_dump = candidate.model_dump(mode="json", exclude={"version"})
    return current_dump != candidate_dump


def _sanitize_event_value(value: JsonValue, key: str) -> JsonValue:
    key_lower = key.lower()
    if key_lower in _SENSITIVE_KEY_EXACT or any(
        marker in key_lower for marker in _SENSITIVE_KEY_MARKERS
    ):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            child_key: _sanitize_event_value(child_value, child_key)
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_event_value(item, key) for item in value]
    if isinstance(value, str):
        sanitized = value
        for pattern in _SENSITIVE_VALUE_PATTERNS:
            sanitized = pattern.sub("[REDACTED]", sanitized)
        if len(sanitized) > 2048:
            return "[REDACTED_OVERSIZE]"
        return sanitized
    return value


def _project_verification_checks(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, list):
        raise StoreError("verification checks must be a list")
    digest = hashlib.sha256(_canonical_json_bytes(raw_checks)).hexdigest()
    compact_checks: list[JsonValue] = []
    for index, raw_check in enumerate(raw_checks):
        if not isinstance(raw_check, dict):
            raise StoreError("verification check must be an object")
        compact_checks.append(
            {
                "name": _projected_check_name(raw_check.get("name"), index),
                "passed": raw_check.get("passed") is True,
                "detail": "passed",
            }
        )

    included: list[JsonValue] = []
    for check in compact_checks:
        candidate = _verification_projection_payload(
            claim_count=payload.get("claim_count", 0),
            checks=[*included, check],
            total_check_count=len(compact_checks),
            checks_digest=digest,
        )
        if len(_canonical_json_bytes(candidate)) > EVENT_PAYLOAD_MAX_BYTES:
            break
        included.append(check)

    projected = _verification_projection_payload(
        claim_count=payload.get("claim_count", 0),
        checks=included,
        total_check_count=len(compact_checks),
        checks_digest=digest,
    )
    if len(_canonical_json_bytes(projected)) > EVENT_PAYLOAD_MAX_BYTES:
        raise StoreError("verification event projection exceeds the maximum size")
    return projected


def _project_context_built_payload(payload: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Replace an oversized context audit event with content-free authority evidence.

    Context content remains in the authorized request/manifest plane.  The durable
    event retains the full manifest digest and accounting, the trusted authority-set
    count/digest markers, and hashes/counts for every omitted decision collection.
    """

    source_manifest = payload.get("source_manifest")
    if not isinstance(source_manifest, dict):
        raise StoreError("context event source manifest must be an object")
    included_source_ids = _string_list(source_manifest.get("included_source_ids"))
    excluded_source_ids = _string_list(source_manifest.get("excluded_source_ids"))
    authority_markers = sorted(
        source_id
        for source_id in included_source_ids
        if source_id.startswith("authority-source-set-count:")
    )
    omitted_before = _nonnegative_int(source_manifest.get("omitted_source_id_count"))
    compact_source_manifest: dict[str, JsonValue] = {
        "schema_version": _nonnegative_int(source_manifest.get("schema_version"), default=1),
        "manifest_digest": _projected_scalar(
            source_manifest.get("manifest_digest"), "context-manifest"
        ),
        "budget_profile": _projected_scalar(
            source_manifest.get("budget_profile"), "budget-profile"
        ),
        "estimator_version": _projected_scalar(
            source_manifest.get("estimator_version"), "estimator-version"
        ),
        "included_source_ids": cast(list[JsonValue], authority_markers),
        "excluded_source_ids": [],
        "omitted_source_id_count": (
            omitted_before
            + len(included_source_ids)
            + len(excluded_source_ids)
            - len(authority_markers)
        ),
        "included_estimated_tokens": _nonnegative_int(
            source_manifest.get("included_estimated_tokens")
        ),
        "excluded_estimated_tokens": _nonnegative_int(
            source_manifest.get("excluded_estimated_tokens")
        ),
        "included_estimated_bytes": _nonnegative_int(
            source_manifest.get("included_estimated_bytes")
        ),
        "excluded_estimated_bytes": _nonnegative_int(
            source_manifest.get("excluded_estimated_bytes")
        ),
    }
    collection_fields = (
        "segments",
        "required_arguments",
        "capability_candidates",
        "capability_selected",
        "skill_selected",
    )
    projection: dict[str, JsonValue] = {
        "version": CONTEXT_BUILT_PROJECTION_VERSION,
        "detail_mode": "counts_and_sha256",
        "original_payload_bytes": len(_canonical_json_bytes(payload)),
        "original_payload_sha256": hashlib.sha256(_canonical_json_bytes(payload)).hexdigest(),
        "completion_contract_sha256": _json_digest(payload.get("completion_contract")),
        "selection_reason_sha256": _json_digest(payload.get("selection_reason")),
        "source_ids_sha256": _json_digest(
            {
                "included": cast(list[JsonValue], included_source_ids),
                "excluded": cast(list[JsonValue], excluded_source_ids),
            }
        ),
        "source_id_count": (
            len(included_source_ids)
            + len(excluded_source_ids)
            + omitted_before
            - len(authority_markers)
        ),
    }
    for field in collection_fields:
        values = payload.get(field)
        projection[f"{field}_count"] = len(values) if isinstance(values, list) else 0
        projection[f"{field}_sha256"] = _json_digest(values)

    return {
        "segments": projection["segments_count"],
        "tool_count": _nonnegative_int(payload.get("tool_count")),
        "tool_choice": _projected_scalar(payload.get("tool_choice"), "tool-choice"),
        "required_tool": _projected_optional_scalar(payload.get("required_tool")),
        "required_arguments": [],
        "completion_contract": {
            "sha256": projection["completion_contract_sha256"],
        },
        "source_manifest": compact_source_manifest,
        "catalog_version": _projected_scalar(payload.get("catalog_version"), "catalog-version"),
        "catalog_fingerprint": _projected_scalar(
            payload.get("catalog_fingerprint"), "catalog-fingerprint"
        ),
        "selection_fingerprint": _projected_scalar(
            payload.get("selection_fingerprint"), "selection-fingerprint"
        ),
        "selection_mode": _projected_scalar(payload.get("selection_mode"), "selection-mode"),
        "selection_reason": "bounded_content_free_projection",
        "capability_candidates": [],
        "capability_selected": [],
        "skill_selected": [],
        "capability_query_hash": _projected_scalar(
            payload.get("capability_query_hash"), "capability-query"
        ),
        "eligible_capability_count": _nonnegative_int(payload.get("eligible_capability_count")),
        "projection": projection,
    }


def _string_list(value: JsonValue | None) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise StoreError("context event source IDs must be a string list")
    return cast(list[str], value)


def _nonnegative_int(value: JsonValue | None, *, default: int = 0) -> int:
    return (
        value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else default
    )


def _projected_optional_scalar(value: JsonValue | None) -> JsonValue:
    return None if value is None else _projected_scalar(value, "value")


def _projected_scalar(value: JsonValue | None, label: str) -> str:
    if isinstance(value, str) and len(_canonical_json_bytes(value)) <= 256:
        return value
    return f"{label}-sha256:{_json_digest(value)}"


def _json_digest(value: JsonValue | None) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _verification_projection_payload(
    *,
    claim_count: JsonValue,
    checks: list[JsonValue],
    total_check_count: int,
    checks_digest: str,
) -> dict[str, JsonValue]:
    included_check_count = len(checks)
    return {
        "claim_count": claim_count,
        "check_count": total_check_count,
        "checks": checks,
        "projection": {
            "version": VERIFICATION_CHECK_PROJECTION_VERSION,
            "ordering": "verifier_order",
            "detail_mode": "status_only",
            "total_check_count": total_check_count,
            "included_check_count": included_check_count,
            "omitted_check_count": total_check_count - included_check_count,
            "checks_sha256": checks_digest,
        },
    }


def _projected_check_name(value: JsonValue | None, index: int) -> str:
    if (
        isinstance(value, str)
        and len(_canonical_json_bytes(value)) <= _PROJECTED_CHECK_NAME_MAX_BYTES
    ):
        return value
    digest = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()[:16]
    return f"check_{index}_{digest}"


def _canonical_json_bytes(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def validate_verified_completion(
    current_task: Task,
    current_run: Run,
    usage: BudgetUsage,
    completion: VerifiedCompletion,
    available_observation_ids: frozenset[str],
    preceding_events: tuple[EventDraft, ...],
) -> None:
    """Validate verifier output before a store constructs terminal state itself."""

    if current_task.status is not TaskStatus.ACTIVE or current_run.status is not RunStatus.RUNNING:
        raise StoreError("verified completion requires an active task and running run")
    if completion.verifier_result.status is not VerifierStatus.PASS:
        raise StoreError("verified completion requires a passing verifier result")
    if (
        usage.model_calls < current_run.usage.model_calls
        or usage.tool_calls < current_run.usage.tool_calls
    ):
        raise StoreError("run usage cannot decrease")
    if (
        current_run.usage.cost is not None
        and usage.cost is not None
        and usage.cost < current_run.usage.cost
    ):
        raise StoreError("run cost usage cannot decrease")
    if (
        current_run.iteration + 1 > current_run.limits.max_iterations
        or usage.model_calls > current_run.limits.max_model_calls
        or usage.tool_calls > current_run.limits.max_tool_calls
    ):
        raise StoreError("verified completion exceeds its configured budget")
    if current_run.limits.max_cost is not None and usage.cost is not None:
        if usage.cost > current_run.limits.max_cost:
            raise StoreError("verified completion exceeds its configured cost budget")

    reserved_completion_events = {EventType.VERIFICATION_PASSED, EventType.RUN_COMPLETED}
    if any(event.type in reserved_completion_events for event in preceding_events):
        raise StoreError("caller cannot supply authoritative completion events")
    has_source_claim = any(
        claim.kind is ClaimKind.SOURCE_CLAIM and claim.observation_ids
        for claim in completion.claims
    )
    if not has_source_claim and not completion.verifier_result.allow_unsourced_completion:
        raise StoreError("verified completion requires a source-backed claim")
    if len({claim.id for claim in completion.claims}) != len(completion.claims):
        raise StoreError("verified completion contains duplicate claim ids")
    if any(item not in available_observation_ids for item in current_task.observation_ids):
        raise StoreError("task references an unavailable observation")
    for claim in completion.claims:
        if claim.run_id != current_run.id or claim.scope != current_run.scope:
            raise StoreError("verified claim is outside the run scope")
        if any(item not in current_task.observation_ids for item in claim.observation_ids):
            raise StoreError("verified claim references an observation outside the task")
        if any(item not in available_observation_ids for item in claim.observation_ids):
            raise StoreError("verified claim references an unavailable observation")
