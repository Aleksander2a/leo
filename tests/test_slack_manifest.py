"""The Slack app manifest: the scopes granted, and the ones that must never appear."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "slack" / "manifest.yml"
RUNBOOK = ROOT / "docs" / "slack-local.md"

#: The workspace grant. Leo's code exercises `app_mentions:read` and `im:history`
#: to receive its two events and `chat:write` to reply; the remaining read scopes
#: are granted but unused, and exist so the install does not have to change when
#: a conversation surface is added.
GRANTED_BOT_SCOPES = {
    "app_mentions:read",
    "chat:write",
    "channels:read",
    "channels:history",
    "groups:read",
    "groups:history",
    "mpim:read",
    "mpim:history",
    "im:read",
    "im:history",
}
REQUIRED_BOT_EVENTS = {"app_mention", "message.im"}

SECRET_PATTERN = re.compile(r"\b(?:xox[a-z]|xapp)-[A-Za-z0-9-]{8,}\b", re.IGNORECASE)


def _manifest() -> dict[str, object]:
    loaded = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_the_manifest_is_socket_mode_with_the_two_events_leo_handles() -> None:
    manifest = _manifest()
    settings = manifest["settings"]
    assert isinstance(settings, dict)
    subscriptions = settings["event_subscriptions"]
    assert isinstance(subscriptions, dict)
    assert set(subscriptions["bot_events"]) == REQUIRED_BOT_EVENTS
    assert settings["socket_mode_enabled"] is True
    assert settings["org_deploy_enabled"] is False


def test_the_granted_scopes_are_read_only_and_bot_only() -> None:
    oauth = _manifest()["oauth_config"]
    assert isinstance(oauth, dict)
    scopes = oauth["scopes"]
    assert isinstance(scopes, dict)
    assert set(scopes["bot"]) == GRANTED_BOT_SCOPES
    # A user token would let Leo read beyond what the bot is a member of.
    assert "user" not in scopes


def test_no_write_or_administrative_scope_creeps_in() -> None:
    """`chat:write` is the only capability Leo has to change anything in Slack."""

    forbidden = {
        "chat:write.public",
        "chat:write.customize",
        "files:write",
        "channels:manage",
        "groups:write",
        "im:write",
        "users:read",
        "users:read.email",
        "search:read",
        "admin",
    }
    oauth = _manifest()["oauth_config"]
    assert isinstance(oauth, dict)
    scopes = oauth["scopes"]
    assert isinstance(scopes, dict)
    assert not set(scopes["bot"]) & forbidden


def test_neither_manifest_nor_runbook_carries_a_token() -> None:
    assert SECRET_PATTERN.search(MANIFEST.read_text(encoding="utf-8")) is None
    assert SECRET_PATTERN.search(RUNBOOK.read_text(encoding="utf-8")) is None


def test_the_runbook_covers_install_operation_and_recovery() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    for phrase in (
        "Install the app",
        "auth.test",
        "Rotate the app-level token",
        "Rotate the bot token",
        "Suspected compromise",
        "Remove Leo from one conversation",
        "Uninstall",
        "leo health",
        "leo slack",
    ):
        assert phrase in runbook, f"runbook is missing: {phrase}"
