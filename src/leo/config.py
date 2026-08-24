"""Runtime configuration loaded from environment variables or a local .env file."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEVELOPMENT = "development"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Leo settings.

    Secrets remain wrapped as ``SecretStr`` so normal representations do not reveal them.
    All live settings are optional because deterministic smoke/eval mode must run offline.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    leo_env: Environment = Environment.DEVELOPMENT
    leo_log_level: str = "INFO"
    leo_model: str | None = None
    leo_max_model_turns: int = Field(default=12, ge=1, le=32)
    leo_max_tool_calls: int = Field(default=24, ge=0, le=64)
    leo_max_run_seconds: float = Field(default=600.0, ge=10.0, le=3600.0)
    leo_max_output_tokens: int = Field(default=4_000, ge=256, le=16_384)
    leo_slack_worker_concurrency: int = Field(default=4, ge=1, le=32)
    leo_dashboard_cors_origins: str = ""

    slack_bot_token: SecretStr | None = None
    slack_app_token: SecretStr | None = None
    leo_slack_team_id: str | None = None

    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    database_url: SecretStr | None = None

    finnhub_api_key: SecretStr | None = None
    finnhub_base_url: str = "https://finnhub.io/api/v1"

    tavily_api_key: SecretStr | None = None
    # TAVILY_ENDPOINT may contain MCP transport query credentials. Native Tavily
    # REST remains pinned to its official fixed endpoint and never consumes it.
    tavily_endpoint: SecretStr | None = None
    tavily_max_calls_per_minute: int = Field(default=10, ge=1, le=100)
    # Tavily's free tier is 1,000 monthly credits and advanced search costs two.
    # Capping calls at 500 remains safe even if every search is advanced.
    tavily_max_calls_per_month: int = Field(default=500, ge=1, le=500)

    exa_api_key: SecretStr | None = None

    # ALPHA_VANTAGE_ENDPOINT and its legacy alias may describe an MCP transport.
    # Native REST composition accepts them only when they are the exact official,
    # credential-free /query URL and otherwise falls back to the fixed REST URL.
    alpha_vantage_api_key: SecretStr | None = None
    alpha_vantage_endpoint: SecretStr | None = None
    alpha_vantage_endpoint_legacy: SecretStr | None = None
    alpha_vantage_max_calls_per_minute: int = Field(default=5, ge=1, le=25)
    alpha_vantage_max_calls_per_day: int = Field(default=25, ge=1, le=25)

    # MASSIVE_ENDPOINT may likewise be an MCP endpoint; native requests remain
    # pinned to Massive's official REST origin.
    massive_api_key: SecretStr | None = None
    massive_endpoint: SecretStr | None = None
    massive_max_calls_per_minute: int = Field(default=10, ge=1, le=120)

    # TickerLayer's REST origin is intentionally not configurable. The default
    # process-local allowance mirrors its documented UTC-calendar-month free quota.
    ticker_layer_api_key: SecretStr | None = None
    ticker_layer_max_calls_per_minute: int = Field(default=60, ge=1, le=600)
    ticker_layer_max_calls_per_month: int = Field(default=3_000, ge=1, le=1_000_000)

    equity_quote_agreement_threshold_percent: float = Field(default=1.0, ge=0, le=100)
    equity_quote_max_corroboration_skew_seconds: int = Field(default=900, ge=0, le=604_800)

    # The existing COINGECKO_ENDPOINT may describe CoinGecko's MCP transport.  The
    # native REST adapter has a separate explicit base so an MCP URL is never treated
    # as a REST root or sent an incompatible request.
    coingecko_endpoint: SecretStr | None = None
    coingecko_api_key: SecretStr | None = None
    coingecko_base_url: str = "https://api.coingecko.com/api/v3"
    coingecko_max_calls_per_minute: int = Field(default=20, ge=1, le=120)

    coin_market_cap_api_key: SecretStr | None = None
    coin_market_cap_base_url: str = "https://pro-api.coinmarketcap.com"
    coin_market_cap_max_calls_per_minute: int = Field(default=20, ge=1, le=120)
    crypto_agreement_threshold_bps: float = Field(default=250.0, ge=0, le=10_000)
    crypto_max_corroboration_skew_seconds: int = Field(default=120, ge=0, le=86_400)

    sec_user_agent: str | None = None
    sec_edgar_base_url: str = "https://data.sec.gov/submissions"

    def missing_for_live_slack(self) -> tuple[str, ...]:
        required: dict[str, object | None] = {
            "SLACK_BOT_TOKEN": self.slack_bot_token,
            "SLACK_APP_TOKEN": self.slack_app_token,
        }
        return tuple(name for name, value in required.items() if _is_missing(value))

    def sensitive_values_for_logging(self) -> tuple[str, ...]:
        """Return configured secret values solely for process-local log redaction."""

        values: list[str] = []
        for field_name in type(self).model_fields:
            value = getattr(self, field_name)
            if isinstance(value, SecretStr):
                secret = value.get_secret_value()
                if secret:
                    values.append(secret)
        return tuple(values)


def _is_missing(value: object | None) -> bool:
    if value is None:
        return True
    if isinstance(value, SecretStr):
        return not value.get_secret_value().strip()
    if isinstance(value, str):
        return not value.strip()
    return False


def is_configured_secret(value: SecretStr | None) -> bool:
    """Return whether an optional secret contains a non-whitespace value."""

    return value is not None and bool(value.get_secret_value().strip())


def has_value(value: SecretStr | str | None) -> bool:
    """Whether an optional setting -- secret or plain -- actually carries a value."""

    if value is None:
        return False
    if isinstance(value, SecretStr):
        return bool(value.get_secret_value().strip())
    return bool(str(value).strip())
