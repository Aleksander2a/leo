"""Bridge from durable ``run_events`` rows to the harness's typed event envelope.

Reuses the harness's own replay-grade normalization (``leo.harness.events``) instead of
inventing a parallel event model. Falls back to raw, un-normalized entries if a run's event
sequence is not perfectly contiguous (e.g. an older/partially-recovered demo run) so the
dashboard never 500s on historical data.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import JsonValue

from leo.api.dashboard.provenance import classify_call
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


def build_timeline(
    rows: list[RunEventRow],
    scope: ScopeKey,
    *,
    transcripts_by_request_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    events = [row_to_run_event(row) for row in rows]
    raw_by_id = {row.id: row.payload for row in rows}
    transcripts = transcripts_by_request_id or {}

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
                **_call_provenance(raw_by_id.get(event.id, {})),
                "transcript": _transcript_for(raw_by_id.get(event.id, {}), transcripts),
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
            **_call_provenance(raw_by_id.get(envelope.event_id, {})),
            "transcript": _transcript_for(raw_by_id.get(envelope.event_id, {}), transcripts),
        }
        for envelope in envelopes
    ]


def _transcript_for(
    raw_payload: dict[str, Any], transcripts: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    """Attach the exact request/response for a model_called entry, when recorded.

    Older runs (before this feature existed) or runs where the transcript sink
    failed have no matching row; the caller falls back to a reconstructed view.
    """

    request_id = raw_payload.get("request_id")
    if not isinstance(request_id, str):
        return None
    return transcripts.get(request_id)


def _call_provenance(raw_payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort MCP/REST/internal classification for one timeline entry.

    Only tool_started/tool_completed/tool_failed payloads carry a ``tool``
    name; every other event kind (model calls, verification, delivery, ...)
    has neither field and gets no provenance badge.
    """

    tool_name = raw_payload.get("tool")
    if not isinstance(tool_name, str):
        return {"call_kind": None, "integration": None}
    return classify_call(tool_name=tool_name, provider=None)
