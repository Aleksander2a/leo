from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from leo.config import Settings


def test_config_reports_only_missing_names() -> None:
    settings = Settings(_env_file=None)
    assert settings.missing_for_deterministic_smoke() == ()
    assert "SLACK_BOT_TOKEN" in settings.missing_for_live_slack()
    assert "OPENROUTER_API_KEY" in settings.missing_for_live_harness()


def test_secret_values_are_redacted_from_repr() -> None:
    tavily_endpoint = "https://mcp.tavily.example/mcp?token=never-print-tavily-endpoint"
    coingecko_endpoint = (
        "https://mcp.api.coingecko.com/mcp?x_cg_demo_api_key=never-print-cg-endpoint"
    )
    settings = Settings(
        _env_file=None,
        openrouter_api_key=SecretStr("never-print-me"),
        slack_user_token=SecretStr("never-print-user-history-token"),
        tavily_endpoint=tavily_endpoint,
        coingecko_endpoint=coingecko_endpoint,
    )
    assert "never-print-me" not in repr(settings)
    assert "never-print-user-history-token" not in repr(settings)
    assert "never-print-tavily-endpoint" not in repr(settings)
    assert "never-print-cg-endpoint" not in repr(settings)
    assert "never-print-tavily-endpoint" not in settings.model_dump_json()
    assert "never-print-cg-endpoint" not in settings.model_dump_json()


def test_blank_values_are_still_reported_missing() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key="   ",
        leo_model="",
        finnhub_api_key="",
    )
    assert settings.missing_for_live_providers() == (
        "OPENROUTER_API_KEY",
        "LEO_MODEL",
        "FINNHUB_API_KEY",
    )


def test_conversation_readiness_does_not_require_an_optional_tool() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key="model-key",
        leo_model="provider/model",
        finnhub_api_key=None,
    )

    assert settings.missing_for_conversation_providers() == ()
    assert settings.missing_for_live_providers() == ("FINNHUB_API_KEY",)


def test_optional_user_history_token_never_gates_slack_availability() -> None:
    settings = Settings(
        _env_file=None,
        slack_bot_token="bot-token",
        slack_app_token="app-token",
        slack_user_token=None,
        leo_slack_team_id="T1",
    )

    assert settings.missing_for_live_slack() == ()


def test_settings_are_immutable_after_process_configuration() -> None:
    settings = Settings(_env_file=None)

    with pytest.raises(ValidationError, match="frozen"):
        settings.leo_model = "forged/provider"  # type: ignore[misc]
