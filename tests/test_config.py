"""Settings: what is required, what is optional, and what must never be logged."""

from __future__ import annotations

import pytest
from pydantic import SecretStr

from leo.config import Environment, Settings, has_value, is_configured_secret


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> Settings:
    for name in (
        "LEO_ENV",
        "LEO_MODEL",
        "OPENROUTER_API_KEY",
        "DATABASE_URL",
        "SLACK_BOT_TOKEN",
        "SLACK_APP_TOKEN",
        "FINNHUB_API_KEY",
        "TAVILY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    return Settings(_env_file=None)


def test_defaults_are_development_and_unconfigured(isolated: Settings) -> None:
    assert isolated.leo_env is Environment.DEVELOPMENT
    assert isolated.leo_model is None
    assert isolated.openrouter_api_key is None


def test_slack_readiness_names_only_what_is_missing(isolated: Settings) -> None:
    partial = isolated.model_copy(update={"slack_bot_token": SecretStr("bot-token")})
    assert partial.missing_for_live_slack() == ("SLACK_APP_TOKEN",)


def test_a_blank_secret_counts_as_missing(isolated: Settings) -> None:
    blank = isolated.model_copy(
        update={"slack_bot_token": SecretStr("   "), "slack_app_token": SecretStr("app-token")}
    )
    assert blank.missing_for_live_slack() == ("SLACK_BOT_TOKEN",)


def test_has_value_accepts_secrets_and_plain_strings() -> None:
    assert has_value(SecretStr("value")) is True
    assert has_value(SecretStr("  ")) is False
    assert has_value("model/name") is True
    assert has_value("") is False
    assert has_value(None) is False


def test_is_configured_secret_matches_has_value_for_secrets() -> None:
    assert is_configured_secret(SecretStr("k")) is True
    assert is_configured_secret(SecretStr(" ")) is False
    assert is_configured_secret(None) is False


def test_secrets_are_collected_for_log_redaction(isolated: Settings) -> None:
    configured = isolated.model_copy(
        update={
            "openrouter_api_key": SecretStr("sk-secret-value"),
            "slack_bot_token": SecretStr("bot-secret-value"),
        }
    )
    values = configured.sensitive_values_for_logging()
    assert "sk-secret-value" in values
    assert "bot-secret-value" in values


def test_secrets_do_not_appear_in_a_repr(isolated: Settings) -> None:
    configured = isolated.model_copy(update={"openrouter_api_key": SecretStr("sk-do-not-log")})
    assert "sk-do-not-log" not in repr(configured)


def test_budgets_are_bounded(isolated: Settings) -> None:
    with pytest.raises(ValueError):
        isolated.model_copy(update={"leo_max_model_turns": 0}).model_validate(
            {"leo_max_model_turns": 0}
        )
