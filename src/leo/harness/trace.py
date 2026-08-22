"""Ordered trace container over the redacted event envelope."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from leo.harness.events import EventEnvelope
from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class TraceError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class EventTrace(ContractModel):
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    scope: ScopeKey
    events: tuple[EventEnvelope, ...] = ()
    digest: NonEmptyStr = Field(min_length=64, max_length=64)


def append_trace(trace: EventTrace, event: EventEnvelope) -> EventTrace:
    if event.run_id != trace.run_id or event.task_id != trace.task_id or event.scope != trace.scope:
        raise TraceError("trace_identity_mismatch")
    if event.sequence != len(trace.events):
        raise TraceError("trace_sequence_mismatch")
    events = (*trace.events, event)
    return EventTrace(
        run_id=trace.run_id,
        task_id=trace.task_id,
        scope=trace.scope,
        events=events,
        digest=_trace_digest(events),
    )


def new_trace(run_id: str, task_id: str, scope: ScopeKey) -> EventTrace:
    return EventTrace(run_id=run_id, task_id=task_id, scope=scope, digest=_trace_digest(()))


def _trace_digest(events: tuple[EventEnvelope, ...]) -> str:
    payload = [event.model_dump(mode="json") for event in events]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
