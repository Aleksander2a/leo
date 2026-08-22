from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from leo.harness.models import ScopeKey
from leo.persistence.conversation_plane import (
    ConversationMessageRole,
    ConversationPlaneMessage,
    build_conversation_plane_message,
    canonical_slack_conversation_id,
)


def _build(**updates: object) -> ConversationPlaneMessage:
    values: dict[str, object] = {
        "scope": ScopeKey(organization_id="org-1", strategy_id="default"),
        "conversation_id": "slack-conversation-1",
        "harness_thread_id": "thread-1",
        "destination_id": "C123",
        "external_event_id": "Ev123",
        "actor_id": "U123",
        "role": ConversationMessageRole.USER,
        "provider_message_ts": "100.200",
        "context_access_hash": "a" * 64,
        "text": "Remember this token=secret-value please",
        "recorded_at": datetime(2026, 8, 21, tzinfo=UTC),
    }
    values.update(updates)
    return build_conversation_plane_message(**values)  # type: ignore[arg-type]


def test_message_is_sanitized_hashed_and_deterministic() -> None:
    first = _build()
    second = _build(text="Remember this token=another-secret please")

    assert first.id == second.id
    assert "secret-value" not in first.text
    assert "[REDACTED]" in first.text
    assert first.content_hash == hashlib.sha256(first.text.encode()).hexdigest()


def test_user_and_assistant_have_distinct_replay_identities() -> None:
    user = _build(role=ConversationMessageRole.USER)
    assistant = _build(role=ConversationMessageRole.ASSISTANT, actor_id="leo")

    assert user.id != assistant.id


def test_canonical_slack_conversation_id_matches_migration_shape() -> None:
    identifier = canonical_slack_conversation_id("T123", "C123")

    assert identifier.startswith("slack-")
    assert len(identifier) == 62
    assert identifier == canonical_slack_conversation_id("T123", "C123")


def test_slack_length_is_bounded_before_persistence() -> None:
    message = _build(text="x" * 9000)

    assert len(message.text) == 8192


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "z" * 64])
def test_access_hash_fails_closed(digest: str) -> None:
    with pytest.raises(ValueError, match="context_access_hash"):
        _build(context_access_hash=digest)
