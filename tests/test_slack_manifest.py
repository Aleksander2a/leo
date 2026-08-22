from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "slack" / "manifest.yml"
RUNBOOK = ROOT / "docs" / "slack-local.md"

REQUIRED_BOT_SCOPES = {
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
REQUIRED_BOT_EVENTS = {
    "app_mention",
    "message.channels",
    "message.groups",
    "message.im",
    "message.mpim",
}
REQUIRED_USER_SCOPES = {"channels:history", "groups:history"}
SECRET_PATTERN = re.compile(r"\b(?:xox[a-z]|xapp)-[A-Za-z0-9-]{8,}\b", re.IGNORECASE)


def _manifest() -> dict[str, object]:
    loaded = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_manifest_is_socket_mode_and_requests_exact_required_surface() -> None:
    manifest = _manifest()
    oauth = manifest["oauth_config"]
    settings = manifest["settings"]
    assert isinstance(oauth, dict)
    assert isinstance(settings, dict)
    scopes = oauth["scopes"]
    subscriptions = settings["event_subscriptions"]
    assert isinstance(scopes, dict)
    assert isinstance(subscriptions, dict)

    assert set(scopes["bot"]) == REQUIRED_BOT_SCOPES
    assert set(scopes["user"]) == REQUIRED_USER_SCOPES
    assert set(subscriptions["bot_events"]) == REQUIRED_BOT_EVENTS
    assert settings["socket_mode_enabled"] is True
    assert settings["org_deploy_enabled"] is False
    assert settings["is_hosted"] is False


def test_manifest_excludes_unrelated_broad_or_write_scopes_and_secrets() -> None:
    manifest_text = MANIFEST.read_text(encoding="utf-8")
    scopes = REQUIRED_BOT_SCOPES | REQUIRED_USER_SCOPES

    assert not scopes.intersection(
        {
            "admin",
            "channels:manage",
            "channels:write",
            "files:read",
            "files:write",
            "groups:write",
            "im:write",
            "mpim:write",
            "search:read",
            "users:read",
        }
    )
    assert SECRET_PATTERN.search(manifest_text) is None


def test_runbook_covers_install_authority_recovery_and_all_conversation_semantics() -> None:
    runbook = RUNBOOK.read_text(encoding="utf-8")
    required_phrases = (
        "Install or refresh the app",
        "auth.test",
        "no strategy mapping",
        "Public/private/shared/external channel: exact destination only",
        "Group DM/MPIM: exact group only",
        "1:1 DM: DM-local plus",
        "Rotate an app-level token",
        "Refresh bot scopes or rotate the bot token",
        "Suspected compromise",
        "Remove Leo from one conversation",
        "Uninstall",
        "Reinstall/rollback",
        "unknown Slack delivery effects",
    )

    assert all(phrase in runbook for phrase in required_phrases)
    assert SECRET_PATTERN.search(runbook) is None
