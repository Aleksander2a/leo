import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from leo.cli import _parse_projection_namespaces, app
from leo.config import Settings
from leo.memory.models import MemoryVisibility


@pytest.fixture(autouse=True)
def _isolate_cli_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    isolated = Settings(_env_file=None)
    monkeypatch.setattr("leo.cli.Settings", lambda: isolated)


def test_cli_help_renders_on_windows_compatible_console() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "slack-live" in result.output


def test_live_quote_uses_the_documented_optional_positional_symbol() -> None:
    result = CliRunner().invoke(app, ["live-quote", "--help"])

    assert result.exit_code == 0
    assert "[symbol]" in result.output
    assert "--symbol" not in result.output


def test_check_config_reports_optional_thread_history_without_gating_slack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(
        _env_file=None,
        slack_app_token="xapp-test",
        slack_bot_token="xoxb-test",
        leo_slack_team_id="T1",
    )
    monkeypatch.setattr("leo.cli.Settings", lambda: settings)

    result = CliRunner().invoke(app, ["check-config"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["live_slack"] == {"ready": True, "missing": []}
    assert payload["thread_history"] == {
        "ready": False,
        "required": False,
        "mode": "persisted_coverage_fallback_required",
        "missing_optional": ["SLACK_USER_TOKEN"],
    }

    configured = Settings(
        _env_file=None,
        slack_app_token="xapp-test",
        slack_bot_token="xoxb-test",
        slack_user_token="xoxp-test",
        leo_slack_team_id="T1",
    )
    monkeypatch.setattr("leo.cli.Settings", lambda: configured)
    configured_result = CliRunner().invoke(app, ["check-config"])
    configured_payload = json.loads(configured_result.output)
    assert configured_payload["thread_history"] == {
        "ready": True,
        "required": False,
        "mode": "direct_user_history",
        "missing_optional": [],
    }


def test_projection_namespace_parser_is_exact_and_rejects_wildcards() -> None:
    parsed = _parse_projection_namespaces(
        [
            "conversation_local=C-demo",
            "actor_private=U-demo",
            "thread_local=C-demo:100.1",
        ]
    )
    assert {(item.visibility, item.namespace_id) for item in parsed} == {
        (MemoryVisibility.CONVERSATION_LOCAL, "C-demo"),
        (MemoryVisibility.ACTOR_PRIVATE, "U-demo"),
        (MemoryVisibility.THREAD_LOCAL, "C-demo:100.1"),
    }
    with pytest.raises(ValueError, match="wildcards"):
        _parse_projection_namespaces(["conversation_local=C-*"])
    with pytest.raises(ValueError, match="conversation-native"):
        _parse_projection_namespaces(["organization_shared=demo-org"])


def test_memory_projection_cli_emits_escaped_page_and_machine_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_project(**kwargs: Any) -> dict[str, object]:
        assert kwargs == {
            "namespaces": ["conversation_local=C-demo"],
            "after": None,
            "page_size": 2,
        }
        return {
            "markdown": "# Safe projection\n",
            "digest": "a" * 64,
            "item_count": 1,
            "next_cursor": "cursor-next",
            "source_revisions": [["memory-demo", 2]],
        }

    monkeypatch.setattr("leo.cli._memory_project_command", fake_project)
    result = CliRunner().invoke(
        app,
        [
            "memory-project",
            "--namespace",
            "conversation_local=C-demo",
            "--page-size",
            "2",
        ],
    )
    assert result.exit_code == 0
    assert "# Safe projection" in result.output
    assert '"next_cursor": "cursor-next"' in result.output
    assert "memory-demo" in result.output


def test_memory_purge_cli_requires_exact_ids_and_forwards_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_purge(**kwargs: Any) -> dict[str, object]:
        assert kwargs == {
            "record_ids": ("memory-a", "memory-b"),
            "confirmation_token": "confirm:0123456789abcdef",
        }
        return {
            "manifest_hash": "0" * 64,
            "purged_record_ids": ["memory-a", "memory-b"],
        }

    monkeypatch.setattr("leo.cli._memory_purge_command", fake_purge)
    result = CliRunner().invoke(
        app,
        [
            "memory-purge",
            "memory-a",
            "memory-b",
            "--confirm",
            "confirm:0123456789abcdef",
        ],
    )
    assert result.exit_code == 0
    assert '"purged_record_ids"' in result.output
    assert "memory-a" in result.output


def test_replay_cli_forwards_bounded_normalized_export_without_scope_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    async def fake_replay(run_id: str, **kwargs: object) -> None:
        captured["run_id"] = run_id
        captured.update(kwargs)

    monkeypatch.setattr("leo.cli._replay_command", fake_replay)
    destination = tmp_path / "durable-replay.json"
    result = CliRunner().invoke(
        app,
        [
            "replay",
            "run-parent",
            "--format",
            "json",
            "--output",
            str(destination),
            "--max-entries",
            "17",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "run_id": "run-parent",
        "output_format": "json",
        "output": destination,
        "max_entries": 17,
    }
    help_result = CliRunner().invoke(app, ["replay", "--help"])
    assert help_result.exit_code == 0
    assert "organization" not in help_result.output.lower()
    assert "strategy" not in help_result.output.lower()
