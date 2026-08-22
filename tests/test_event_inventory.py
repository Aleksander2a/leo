from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.harness.events import (
    EVENT_SCHEMA_VERSION,
    RUN_EVENT_KIND,
    EventEnvelope,
    EventKind,
    EventSchemaError,
    build_event,
    event_contract_digest,
    normalize_run_event,
    normalize_run_timeline,
    parse_event,
    validate_event_inventory,
)
from leo.harness.models import (
    ClaimKind,
    EventType,
    EvidenceQuality,
    ObservationStatus,
    RunEvent,
    ScopeKey,
)

NOW = datetime(2026, 8, 21, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org-eval", strategy_id="metadata-only")


def _event_payload(kind: EventKind, *, version: str = EVENT_SCHEMA_VERSION) -> dict[str, object]:
    return {
        "event_id": f"event-{kind.value}",
        "run_id": "run-eval",
        "task_id": "task-eval",
        "scope": SCOPE.model_dump(mode="json"),
        "sequence": 1,
        "occurred_at": NOW.isoformat(),
        "kind": kind.value,
        "schema_version": version,
        "correlation_id": "correlation-eval",
        "payload": {"status": "observed"},
    }


def test_every_event_kind_and_run_event_type_has_a_typed_versioned_contract() -> None:
    validate_event_inventory()
    assert set(RUN_EVENT_KIND) == set(EventType)
    assert set(RUN_EVENT_KIND.values()) <= set(EventKind)
    assert len(event_contract_digest()) == 64
    for sequence, kind in enumerate(EventKind, start=1):
        event = build_event(
            event_id=f"event-{sequence}",
            run_id="run-eval",
            task_id="task-eval",
            scope=SCOPE,
            sequence=sequence,
            occurred_at=NOW,
            kind=kind,
            correlation_id="correlation-eval",
            payload={"status": "observed"},
        )
        assert event.schema_version == "v2"
        assert event.kind is kind


def test_supported_v1_event_migrates_and_unknown_version_fails_closed() -> None:
    legacy = EventEnvelope.model_validate(_event_payload(EventKind.TOOL_CALL, version="v1"))
    migrated = parse_event(legacy)
    assert migrated.schema_version == "v2"
    assert migrated.payload == {"status": "observed"}
    unknown = _event_payload(EventKind.TOOL_CALL, version="v99")
    with pytest.raises(ValidationError, match="schema version is unsupported"):
        EventEnvelope.model_validate(unknown)


def test_evidence_event_carries_m4_quality_schema_and_harness_claim_kinds() -> None:
    event = build_event(
        event_id="event-evidence",
        run_id="run-eval",
        task_id="task-eval",
        scope=SCOPE,
        sequence=1,
        occurred_at=NOW,
        kind=EventKind.EVIDENCE_NORMALIZED,
        correlation_id="correlation-eval",
        payload={
            "status": "normalized",
            "observation_id": "observation-1",
            "observation_kind": "sec.get_recent_filings",
            "observation_schema_version": "observation-v2",
            "observation_status": ObservationStatus.RETRIEVED.value,
            "evidence_quality": EvidenceQuality.PRIMARY_SOURCE.value,
            "normalization_version": "normalization-v1",
            "claim_kinds": [
                ClaimKind.SOURCE_CLAIM.value,
                ClaimKind.AFFECTED_ASSUMPTION.value,
                ClaimKind.UNCERTAINTY.value,
            ],
            "affected_assumption": True,
            "uncertainty": True,
        },
    )
    assert event.payload["observation_status"] == "retrieved"
    assert event.payload["evidence_quality"] == "primary_source"
    assert event.payload["claim_kinds"] == [
        "source_claim",
        "affected_assumption",
        "uncertainty",
    ]


def test_typed_events_reject_wrong_fields_oversize_payloads_and_secret_values() -> None:
    with pytest.raises(EventSchemaError, match="field_not_allowed"):
        build_event(
            event_id="event-forged",
            run_id="run-eval",
            task_id="task-eval",
            scope=SCOPE,
            sequence=1,
            occurred_at=NOW,
            kind=EventKind.TERMINAL,
            correlation_id="correlation-eval",
            payload={"status": "completed", "membership_source_ids": ["forged"]},
        )
    oversized = _event_payload(EventKind.TERMINAL)
    oversized["payload"] = {"reason": "x" * 9_000}
    with pytest.raises(ValidationError, match="exceeds 8192 bytes"):
        EventEnvelope.model_validate(oversized)
    secret = _event_payload(EventKind.TERMINAL)
    secret["payload"] = {"reason": "sk-" + "a" * 30}
    with pytest.raises(ValidationError, match="secret-like"):
        EventEnvelope.model_validate(secret)


def test_durable_v1_run_timeline_normalizes_to_strict_causal_v2() -> None:
    events = (
        RunEvent(
            id="event-1",
            run_id="run-eval",
            task_id="task-eval",
            sequence=1,
            type=EventType.TASK_STARTED,
            occurred_at=NOW,
            iteration=0,
            schema_version=1,
            payload={"phase": "research"},
        ),
        RunEvent(
            id="event-2",
            run_id="run-eval",
            task_id="task-eval",
            sequence=2,
            type=EventType.RUN_FAILED,
            occurred_at=NOW,
            iteration=1,
            schema_version=1,
            payload={"reason": "synthetic_failure"},
        ),
    )
    normalized = normalize_run_timeline(events, SCOPE)
    assert [item.schema_version for item in normalized] == ["v2", "v2"]
    assert [item.sequence for item in normalized] == [1, 2]
    assert normalized[0].correlation_id == normalized[1].correlation_id == "run-eval"
    assert normalized[0].causation_id is None
    assert normalized[1].causation_id == "event-1"
    assert normalized[0].payload == {"status": "task_started"}
    assert normalized[1].payload == {
        "status": "run_failed",
        "reason": "synthetic_failure",
    }

    with pytest.raises(EventSchemaError, match="sequence_gap"):
        normalize_run_timeline((events[0], events[1].model_copy(update={"sequence": 3})), SCOPE)
    with pytest.raises(EventSchemaError, match="schema_version_unsupported"):
        normalize_run_event(events[0].model_copy(update={"schema_version": 99}), SCOPE)
