"""Bridge from durable ``run_events`` rows to the harness's typed event envelope.

Reuses the harness's own replay-grade normalization (``leo.harness.events``) instead of
inventing a parallel event model. Falls back to raw, un-normalized entries if a run's event
sequence is not perfectly contiguous (e.g. an older/partially-recovered demo run) so the
dashboard never 500s on historical data.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import JsonValue

from leo.harness.events import RUN_EVENT_KIND, EventSchemaError, normalize_run_timeline
from leo.harness.models import EventType, RunEvent, ScopeKey
from leo.persistence.schema import RunEventRow


def row_to_run_event(row: RunEventRow) -> RunEvent:
    return RunEvent(
        id=row.id,
        run_id=row.run_id,
        task_id=row.task_id,
        sequence=row.sequence,
        type=EventType(row.type),
        occurred_at=row.occurred_at,
        iteration=row.iteration,
        schema_version=row.schema_version,
        payload=cast(dict[str, JsonValue], row.payload),
    )


def build_timeline(rows: list[RunEventRow], scope: ScopeKey) -> list[dict[str, Any]]:
    events = [row_to_run_event(row) for row in rows]
    raw_by_id = {row.id: row.payload for row in rows}

    try:
        envelopes = normalize_run_timeline(events, scope)
    except EventSchemaError:
        return [
            {
                "sequence": event.sequence,
                "kind": RUN_EVENT_KIND[event.type].value,
                "occurred_at": event.occurred_at.isoformat(),
                "envelope": None,
                "raw_payload": raw_by_id.get(event.id, {}),
                "normalization_error": True,
            }
            for event in sorted(events, key=lambda item: item.sequence)
        ]

    return [
        {
            "sequence": envelope.sequence,
            "kind": envelope.kind.value,
            "occurred_at": envelope.occurred_at.isoformat(),
            "envelope": envelope.model_dump(mode="json"),
            "raw_payload": raw_by_id.get(envelope.event_id, {}),
            "normalization_error": False,
        }
        for envelope in envelopes
    ]
