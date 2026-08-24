"""The Slack app manifest must request exactly what the code uses -- no more."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "slack" / "manifest.yml"
RUNBOOK = ROOT / "docs" / "slack-local.md"

#: Every scope Leo actually exercises. `app_mentions:read` and `im:history`
#: deliver the two events it subscribes to; `chat:write` posts and updates the
#: reply. Leo reconstructs conversation history from its own tables, so it needs
#: no bulk-read scope -- and must not hold one.
REQUIRED_BOT_SCOPES = {"app_mentions:read", "chat:write", "im:history"}
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


def test_scopes_are_exactly_what_the_code_uses() -> None:
    oauth = _manifest()["oauth_config"]
    assert isinstance(oauth, dict)
    scopes = oauth["scopes"]
    assert isinstance(scopes, dict)
    assert set(scopes["bot"]) == REQUIRED_BOT_SCOPES
    # A user token would grant Leo reach beyond the conversation it is in.
    assert "user" not in scopes


def test_no_write_or_broad_read_scope_creeps_in() -> None:
    forbidden = {
        "channels:read",
        "channels:history",
        "groups:history",
        "mpim:history",
        "chat:write.public",
        "files:write",
        "users:read",
        "admin",
        "search:read",
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
