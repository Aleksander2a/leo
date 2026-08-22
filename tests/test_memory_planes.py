from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from leo.harness.models import ScopeKey
from leo.memory.planes import (
    InMemoryDerivedPlaneStore,
    SanitizedMessage,
    SummaryRevision,
    sanitize_message_text,
)

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def test_sanitized_message_and_derived_summary_keep_source_boundary() -> None:
    message = SanitizedMessage.from_text(
        id="message-1",
        scope=SCOPE,
        destination_id="channel-1",
        external_event_id="event-1",
        text="token: do-not-store ask about NVDA",
        recorded_at=NOW,
    )
    assert "do-not-store" not in message.text
    store = InMemoryDerivedPlaneStore()
    store.add_message(message)
    summary_text = "NVDA question recorded."
    summary = SummaryRevision(
        id="summary-1",
        scope=SCOPE,
        thread_id="thread-1",
        source_message_ids=(message.id,),
        revision=1,
        content=summary_text,
        content_hash=hashlib.sha256(summary_text.encode()).hexdigest(),
        created_at=NOW,
    )
    store.add_summary(summary)
    store.drop_summary(summary.id)
    assert message.id in store.messages


def test_derived_rows_require_source_hashes_and_scope() -> None:
    with pytest.raises(ValueError):
        sanitize_message_text("\x00\x01")
    with pytest.raises(ValueError, match="unknown source"):
        InMemoryDerivedPlaneStore().add_summary(
            SummaryRevision(
                id="summary-1",
                scope=SCOPE,
                thread_id="thread-1",
                source_message_ids=("missing",),
                revision=1,
                content="summary",
                content_hash=hashlib.sha256(b"summary").hexdigest(),
                created_at=NOW,
            )
        )
