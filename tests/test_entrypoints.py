"""The process entry points a deployment can be configured to run.

A hosting platform pins its start command in its own settings, outside this
repository, so a CLI rename can leave a deployment invoking a command that no
longer exists. The build stays green, the container exits immediately, and
nothing in CI notices.

These tests close that gap: every command name this repository ships or has
shipped in a deploy configuration must still resolve.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from leo.cli import app

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"

runner = CliRunner()


def _command_names() -> set[str]:
    from typer.main import get_command

    return set(get_command(app).commands)  # type: ignore[attr-defined]


def _dockerfile_cmd() -> list[str]:
    """The argv from the Dockerfile's exec-form CMD."""

    match = re.search(r"^CMD\s+\[(.+)\]\s*$", DOCKERFILE.read_text(encoding="utf-8"), re.M)
    assert match is not None, "Dockerfile must declare an exec-form CMD"
    return [part.strip().strip('"') for part in match.group(1).split(",")]


def test_the_dockerfile_cmd_invokes_a_real_cli_command() -> None:
    argv = _dockerfile_cmd()
    assert argv[:3] == ["python", "-m", "leo"], argv
    assert argv[3] in _command_names(), (
        f"Dockerfile CMD runs `leo {argv[3]}`, which is not a command this CLI defines. "
        f"Available: {sorted(_command_names())}"
    )


@pytest.mark.parametrize("name", ["slack", "slack-live", "ask", "chat", "health", "memory"])
def test_every_shipped_command_name_resolves(name: str) -> None:
    """Includes `slack-live`, which existing deployments may still be pinned to."""

    assert name in _command_names()


def test_slack_live_is_hidden_from_help_but_still_runs() -> None:
    """The alias exists for old deploy configs; new users should only see `slack`."""

    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "slack-live" not in result.stdout
    assert "slack" in result.stdout


@pytest.mark.parametrize("command", ["slack", "slack-live"])
def test_both_slack_entry_points_reach_the_same_configuration_gate(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither name may fail with `no such command`; both must reach real config checks."""

    for variable in ("SLACK_BOT_TOKEN", "SLACK_APP_TOKEN"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr("leo.cli.Settings", _unconfigured_settings)

    result = runner.invoke(app, [command])

    assert result.exit_code == 2
    assert "missing Slack configuration" in result.stdout
    assert "No such command" not in result.stdout


def _unconfigured_settings():  # type: ignore[no-untyped-def]
    from leo.config import Settings

    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        slack_bot_token=None,
        slack_app_token=None,
    )
