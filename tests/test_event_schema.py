from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.harness.events import EventKind, EventSchemaError, build_event
from leo.harness.models import ScopeKey

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_event_registry_redacts_content_and_preserves_replay_metadata() -> None:
    event = build_event(
        event_id="event-1",
        run_id="run-1",
        task_id="task-1",
        scope=SCOPE,
        sequence=3,
        occurred_at=NOW,
        kind=EventKind.MODEL_CALLED,
        correlation_id="corr-1",
        payload={"status": "ok", "text": "private synthetic content", "cost": 0.01},
    )
    assert event.payload["text"] != "private synthetic content"
    assert event.sequence == 3
    with pytest.raises(EventSchemaError, match="field_not_allowed"):
        build_event(
            event_id="event-2",
            run_id="run-1",
            task_id="task-1",
            scope=SCOPE,
            sequence=4,
            occurred_at=NOW,
            kind=EventKind.TERMINAL,
            correlation_id="corr-1",
            payload={"completion_authority": True},
        )


def test_secret_like_event_payload_fails_closed() -> None:
    with pytest.raises(ValueError, match="secret-like"):
        build_event(
            event_id="event-1",
            run_id="run-1",
            task_id="task-1",
            scope=SCOPE,
            sequence=0,
            occurred_at=NOW,
            kind=EventKind.USAGE,
            correlation_id="corr-1",
            payload={"reason": "xoxb-" + "123456789012345"},
        )
