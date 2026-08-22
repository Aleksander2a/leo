"""Typed failure taxonomy, sanitized bundles, and regression closure gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import Field, JsonValue, model_validator

from leo.evals.recordings import RecordingSanitizationError, sanitize_payload
from leo.harness.models import ContractModel, NonEmptyStr


class FailureClass(StrEnum):
    POLICY = "policy_rejection"
    INVALID_DECISION = "invalid_conversation_or_plan_decision"
    MODEL = "invalid_conversation_or_plan_decision"  # Compatibility alias.
    MEMBERSHIP_SOURCE = "membership_source_change"
    PROVIDER_TRANSIENT = "provider_transient"
    PROVIDER_PERMANENT = "provider_permanent"
    CHILD_TRANSIENT = "child_transient"
    CHILD_PERMANENT = "child_permanent"
    ORPHAN = "orphan"
    DUPLICATE = "duplicate"
    DEADLOCK = "deadlock"
    NO_PROGRESS = "no_progress"
    CONCURRENCY = "concurrency"
    BUDGET = "budget_deadline_cancel"
    SYNTHESIS = "synthesis"
    INVARIANT = "invariant_data"
    UNKNOWN_EFFECT = "unknown_effect"
    DELIVERY = "delivery"


class FailureRecord(ContractModel):
    version: NonEmptyStr = "failure-v2"
    run_id: NonEmptyStr
    root_code: NonEmptyStr
    failure_class: FailureClass
    boundary: str | None = None
    terminal_reason: str | None = None
    event_ids: tuple[NonEmptyStr, ...] = ()
    recording_ids: tuple[NonEmptyStr, ...] = ()
    reproduction_command: NonEmptyStr

    @model_validator(mode="after")
    def reproduction_is_safe(self) -> FailureRecord:
        _validate_reproduction_command(self.reproduction_command)
        sanitize_payload(
            {
                "run_id": self.run_id,
                "root_code": self.root_code,
                "boundary": self.boundary,
                "terminal_reason": self.terminal_reason,
                "event_ids": list(self.event_ids),
                "recording_ids": list(self.recording_ids),
            }
        )
        return self


class FailureBundle(ContractModel):
    version: NonEmptyStr = "bundle-v2"
    failure: FailureRecord
    sanitized_config: dict[str, JsonValue]
    sanitized_events: tuple[dict[str, JsonValue], ...] = ()
    fixture_id: NonEmptyStr
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches_payload(self) -> FailureBundle:
        if self.digest != _bundle_digest(
            self.failure,
            fixture_id=self.fixture_id,
            sanitized_config=self.sanitized_config,
            sanitized_events=self.sanitized_events,
        ):
            raise ValueError("failure bundle digest mismatch")
        return self


class RegressionClosure(ContractModel):
    failure_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    fixture_id: NonEmptyStr
    focused_tests_passed: bool
    aggregate_gate_passed: bool
    focused_evidence_ids: tuple[NonEmptyStr, ...] = ()
    aggregate_evidence_id: str | None = None

    @property
    def closed(self) -> bool:
        return (
            self.focused_tests_passed
            and self.aggregate_gate_passed
            and bool(self.focused_evidence_ids)
            and self.aggregate_evidence_id is not None
        )


class FailureExportAuthority(ContractModel):
    """Trusted operator projection supplied by the export composition, not a bundle."""

    organization_id: NonEmptyStr
    actor_id: NonEmptyStr
    allowed_run_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def exact_run_projection(self) -> FailureExportAuthority:
        if tuple(sorted(set(self.allowed_run_ids))) != self.allowed_run_ids:
            raise ValueError("failure export run IDs must be sorted and unique")
        return self

    @property
    def access_digest(self) -> str:
        return _digest_json(
            {
                "organization_id": self.organization_id,
                "actor_id": self.actor_id,
                "allowed_run_ids": list(self.allowed_run_ids),
            }
        )


class FailureExportNotFound(LookupError):
    def __init__(self) -> None:
        super().__init__("failure_bundle_not_found")
        self.safe_code = "failure_bundle_not_found"


class FailureExportReceipt(ContractModel):
    version: NonEmptyStr = "failure-export-v1"
    run_id: NonEmptyStr
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    export_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    destination_name: NonEmptyStr


class ScopedFailureBundleStore:
    """Minimal exact-scope source used by the deterministic exporter workflow."""

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, str], FailureBundle] = {}

    def put(self, *, organization_id: str, bundle: FailureBundle) -> None:
        key = (organization_id, bundle.failure.run_id)
        existing = self._bundles.get(key)
        if existing is not None and existing.digest != bundle.digest:
            raise ValueError("failure bundle run already has different content")
        self._bundles[key] = bundle

    def load(
        self,
        *,
        authority: FailureExportAuthority,
        run_id: str,
    ) -> FailureBundle:
        if run_id not in authority.allowed_run_ids:
            raise FailureExportNotFound
        try:
            return self._bundles[(authority.organization_id, run_id)]
        except KeyError as exc:
            raise FailureExportNotFound from exc


class AsyncFailureBundleSource(Protocol):
    async def load(
        self,
        *,
        authority: FailureExportAuthority,
        run_id: str,
    ) -> FailureBundle: ...


def classify_failure(
    run_id: str,
    root_code: str,
    *,
    reproduction_command: str,
    boundary: str | None = None,
    terminal_reason: str | None = None,
    event_ids: tuple[str, ...] = (),
    recording_ids: tuple[str, ...] = (),
) -> FailureRecord:
    lowered = " ".join(item.lower() for item in (root_code, boundary or "", terminal_reason or ""))
    if any(token in lowered for token in ("scope", "policy", "not_allowed", "eligibility")):
        category = FailureClass.POLICY
    elif any(token in lowered for token in ("membership", "source_set", "source-set")):
        category = FailureClass.MEMBERSHIP_SOURCE
    elif any(
        token in lowered
        for token in (
            "invalid_decision",
            "invalid_model",
            "invalid_conversation",
            "invalid_plan",
            "malformed_plan",
        )
    ):
        category = FailureClass.INVALID_DECISION
    elif "orphan" in lowered:
        category = FailureClass.ORPHAN
    elif "duplicate" in lowered:
        category = FailureClass.DUPLICATE
    elif "deadlock" in lowered or "cycle" in lowered:
        category = FailureClass.DEADLOCK
    elif "no_progress" in lowered or "stalled" in lowered:
        category = FailureClass.NO_PROGRESS
    elif "child" in lowered and any(
        token in lowered for token in ("timeout", "rate", "transport", "unavailable", "retry")
    ):
        category = FailureClass.CHILD_TRANSIENT
    elif "child" in lowered:
        category = FailureClass.CHILD_PERMANENT
    elif any(token in lowered for token in ("race", "cas", "lease", "concurrency")):
        category = FailureClass.CONCURRENCY
    elif any(token in lowered for token in ("timeout", "budget", "deadline", "cancel")):
        category = FailureClass.BUDGET
    elif any(token in lowered for token in ("rate", "transport", "unavailable", "retry")):
        category = FailureClass.PROVIDER_TRANSIENT
    elif any(token in lowered for token in ("provider", "authentication", "permission_denied")):
        category = FailureClass.PROVIDER_PERMANENT
    elif "synthesis" in lowered:
        category = FailureClass.SYNTHESIS
    elif "delivery" in lowered or "slack" in lowered or "outbox" in lowered:
        category = FailureClass.DELIVERY
    elif "effect" in lowered:
        category = FailureClass.UNKNOWN_EFFECT
    else:
        category = FailureClass.INVARIANT
    return FailureRecord(
        run_id=run_id,
        root_code=root_code,
        failure_class=category,
        boundary=boundary,
        terminal_reason=terminal_reason,
        event_ids=event_ids,
        recording_ids=recording_ids,
        reproduction_command=reproduction_command,
    )


def make_bundle(
    failure: FailureRecord,
    *,
    fixture_id: str,
    sanitized_config: Mapping[str, JsonValue | str],
    events: tuple[Mapping[str, JsonValue], ...] = (),
) -> FailureBundle:
    try:
        clean_config = sanitize_payload(sanitized_config)
        clean_events = tuple(sanitize_payload(event) for event in events)
    except RecordingSanitizationError:
        raise
    digest = _bundle_digest(
        failure,
        fixture_id=fixture_id,
        sanitized_config=clean_config,
        sanitized_events=clean_events,
    )
    return FailureBundle(
        failure=failure,
        sanitized_config=clean_config,
        sanitized_events=clean_events,
        fixture_id=fixture_id,
        digest=digest,
    )


def validate_failure_bundle(bundle: FailureBundle) -> None:
    expected = _bundle_digest(
        bundle.failure,
        fixture_id=bundle.fixture_id,
        sanitized_config=bundle.sanitized_config,
        sanitized_events=bundle.sanitized_events,
    )
    if bundle.digest != expected:
        raise ValueError("failure bundle digest mismatch")


def validate_regression_closure(
    bundle: FailureBundle,
    closure: RegressionClosure,
) -> None:
    if closure.failure_digest != bundle.digest or closure.fixture_id != bundle.fixture_id:
        raise ValueError("regression closure does not match the failure bundle")
    if not closure.closed:
        raise ValueError("regression closure requires focused and aggregate evidence")


def export_failure_bundle(
    store: ScopedFailureBundleStore,
    *,
    authority: FailureExportAuthority,
    run_id: str,
    destination: Path,
) -> FailureExportReceipt:
    """Write one sanitized exact-scope bundle atomically with deterministic bytes."""

    bundle = store.load(authority=authority, run_id=run_id)
    return _write_failure_bundle(bundle, run_id=run_id, destination=destination)


async def export_failure_bundle_async(
    source: AsyncFailureBundleSource,
    *,
    authority: FailureExportAuthority,
    run_id: str,
    destination: Path,
) -> FailureExportReceipt:
    """Load a durable authorized run source, then use the same atomic exporter."""

    bundle = await source.load(authority=authority, run_id=run_id)
    return _write_failure_bundle(bundle, run_id=run_id, destination=destination)


def _write_failure_bundle(
    bundle: FailureBundle,
    *,
    run_id: str,
    destination: Path,
) -> FailureExportReceipt:
    validate_failure_bundle(bundle)
    if destination.suffix.casefold() != ".json" or destination.name in {".json", "..json"}:
        raise ValueError("failure export destination must be a named JSON file")
    if not destination.parent.is_dir():
        raise ValueError("failure export destination directory does not exist")
    payload = {
        "version": "failure-export-v1",
        "bundle": bundle.model_dump(mode="json"),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    export_digest = hashlib.sha256(encoded).hexdigest()
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return FailureExportReceipt(
        run_id=run_id,
        bundle_digest=bundle.digest,
        export_digest=export_digest,
        destination_name=destination.name,
    )


def import_failure_bundle(source: Path) -> FailureBundle:
    """Validate a deterministic export before using its reproduction metadata."""

    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("failure export is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {"version", "bundle"}:
        raise ValueError("failure export envelope is invalid")
    if payload["version"] != "failure-export-v1":
        raise ValueError("failure export version is unsupported")
    bundle = FailureBundle.model_validate(payload["bundle"])
    validate_failure_bundle(bundle)
    return bundle


def _bundle_digest(
    failure: FailureRecord,
    *,
    fixture_id: str,
    sanitized_config: Mapping[str, JsonValue],
    sanitized_events: tuple[dict[str, JsonValue], ...],
) -> str:
    payload = {
        "failure": failure.model_dump(mode="json"),
        "fixture_id": fixture_id,
        "sanitized_config": dict(sanitized_config),
        "sanitized_events": list(sanitized_events),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _digest_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_reproduction_command(command: str) -> None:
    if "\n" in command or "\r" in command:
        raise ValueError("reproduction command must be one line")
    if re.search(r"(?:&&|\|\||[;|<>]|\.\.[\\/])", command):
        raise ValueError("reproduction command contains unsafe shell syntax")
    sanitize_payload({"reproduction_command": command})
