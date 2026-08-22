"""Composition for optional, provider-neutral equity market reads."""

from __future__ import annotations

import httpx
from pydantic import JsonValue, SecretStr

from leo.config import Settings, is_configured_secret
from leo.harness.equity_market import (
    canonical_equity_profile_statements,
    canonical_equity_quote_statement,
    valid_equity_observed_at,
    valid_equity_profile_provenance,
    valid_equity_quote_provenance,
)
from leo.harness.models import (
    RunPhase,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRetryPolicy,
    ToolSpec,
)
from leo.harness.ports import Clock, Tool
from leo.harness.provider_health import ProviderHealthSnapshot
from leo.integrations.alpha_vantage import (
    AlphaVantageCompanyProfileTool,
    AlphaVantageQuoteTool,
    AlphaVantageSymbolSearchTool,
)
from leo.integrations.equity_market import (
    EquityProfileArguments,
    EquityProfileRoute,
    EquityQuoteRoute,
    EquitySearchRoute,
    RedundantEquityProfileTool,
    RedundantEquityQuoteTool,
    RedundantEquitySymbolSearchTool,
)
from leo.integrations.finnhub import FinnhubCompanyProfileTool, FinnhubQuoteTool
from leo.integrations.massive import (
    MassiveCompanyProfileTool,
    MassiveStockSnapshotTool,
    MassiveSymbolSearchTool,
)
from leo.integrations.provider_runtime import ProviderCallGate, ProviderGateRegistry
from leo.integrations.tickerlayer import (
    TickerLayerCompanyProfileTool,
    TickerLayerStockSnapshotTool,
    TickerLayerSymbolSearchTool,
)

_ALPHA_REST_URL = "https://www.alphavantage.co/query"
_MASSIVE_REST_URL = "https://api.massive.com"


def build_equity_market_tools(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    clock: Clock,
    provider_gates: ProviderGateRegistry | None = None,
) -> tuple[Tool, ...]:
    """Build raw provider tools plus bounded provider-neutral routes.

    The stable quote ID is always a provider-neutral route, including a one-provider
    Finnhub deployment. Direct provider IDs remain available for explicit diagnosis.
    """

    tools: list[Tool] = []
    quotes: dict[str, Tool] = {}
    searches: dict[str, Tool] = {}
    profiles: dict[str, Tool] = {}

    if is_configured_secret(settings.alpha_vantage_api_key):
        assert settings.alpha_vantage_api_key is not None
        gate = _provider_gate(
            provider_gates,
            provider="alpha_vantage",
            clock=clock,
            max_concurrency=1,
            max_calls_per_minute=settings.alpha_vantage_max_calls_per_minute,
            max_calls_per_day=settings.alpha_vantage_max_calls_per_day,
        )
        api_key = settings.alpha_vantage_api_key.get_secret_value()
        base_url = resolve_alpha_vantage_rest_base_url(
            configured_endpoint=settings.alpha_vantage_endpoint,
            configured_legacy_endpoint=settings.alpha_vantage_endpoint_legacy,
        )
        alpha_quote = AlphaVantageQuoteTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        alpha_search = AlphaVantageSymbolSearchTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        alpha_profile = AlphaVantageCompanyProfileTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        tools.extend((alpha_quote, alpha_search, alpha_profile))
        quotes["alpha-vantage"] = alpha_quote
        searches["alpha-vantage"] = alpha_search
        profiles["alpha-vantage"] = alpha_profile

    if is_configured_secret(settings.massive_api_key):
        assert settings.massive_api_key is not None
        gate = _provider_gate(
            provider_gates,
            provider="massive",
            clock=clock,
            max_concurrency=2,
            max_calls_per_minute=settings.massive_max_calls_per_minute,
        )
        api_key = settings.massive_api_key.get_secret_value()
        base_url = resolve_massive_rest_base_url(configured_endpoint=settings.massive_endpoint)
        massive_quote = MassiveStockSnapshotTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        massive_search = MassiveSymbolSearchTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        massive_profile = MassiveCompanyProfileTool(
            client=client, api_key=api_key, clock=clock, base_url=base_url, gate=gate
        )
        tools.extend((massive_quote, massive_search, massive_profile))
        quotes["massive"] = massive_quote
        searches["massive"] = massive_search
        profiles["massive"] = massive_profile

    if is_configured_secret(settings.ticker_layer_api_key):
        assert settings.ticker_layer_api_key is not None
        gate = _provider_gate(
            provider_gates,
            provider="ticker_layer",
            clock=clock,
            max_concurrency=2,
            max_calls_per_minute=settings.ticker_layer_max_calls_per_minute,
            max_calls_per_month=settings.ticker_layer_max_calls_per_month,
        )
        api_key = settings.ticker_layer_api_key.get_secret_value()
        ticker_quote = TickerLayerStockSnapshotTool(
            client=client, api_key=api_key, clock=clock, gate=gate
        )
        ticker_search = TickerLayerSymbolSearchTool(
            client=client, api_key=api_key, clock=clock, gate=gate
        )
        ticker_profile = TickerLayerCompanyProfileTool(
            client=client, api_key=api_key, clock=clock, gate=gate
        )
        tools.extend((ticker_quote, ticker_search, ticker_profile))
        quotes["ticker-layer"] = ticker_quote
        searches["ticker-layer"] = ticker_search
        profiles["ticker-layer"] = ticker_profile

    peer_configured = bool(quotes)
    if is_configured_secret(settings.finnhub_api_key):
        assert settings.finnhub_api_key is not None
        finnhub_gate = _provider_gate(
            provider_gates,
            provider="finnhub",
            clock=clock,
            max_concurrency=4,
            max_calls_per_minute=60,
        )
        finnhub_quote = FinnhubQuoteTool(
            client=client,
            api_key=settings.finnhub_api_key.get_secret_value(),
            clock=clock,
            base_url=settings.finnhub_base_url,
            gate=finnhub_gate,
        )
        finnhub_profile = FinnhubCompanyProfileTool(
            client=client,
            api_key=settings.finnhub_api_key.get_secret_value(),
            clock=clock,
            base_url=settings.finnhub_base_url,
            gate=finnhub_gate,
        )
        normalized_quote = _NormalizedFinnhubQuoteTool(
            finnhub_quote,
            tool_name="market.get_quote_finnhub",
        )
        tools.append(normalized_quote)
        quotes["finnhub"] = normalized_quote
        if peer_configured:
            normalized_profile = _NormalizedFinnhubProfileTool(finnhub_profile)
            tools.append(normalized_profile)
            profiles["finnhub"] = normalized_profile
        else:
            # Compatibility: keep the historical profile ID when Finnhub is the only
            # configured equity provider; quotes still use the truthful aggregate ID.
            tools.append(finnhub_profile)

    if quotes:
        quote_order = ("finnhub", "massive", "ticker-layer", "alpha-vantage")
        tools.append(
            RedundantEquityQuoteTool(
                routes=tuple(
                    EquityQuoteRoute(provider, quotes[provider])
                    for provider in quote_order
                    if provider in quotes
                ),
                clock=clock,
                agreement_threshold_percent=(settings.equity_quote_agreement_threshold_percent),
                max_corroboration_skew_seconds=(
                    settings.equity_quote_max_corroboration_skew_seconds
                ),
            )
        )
    if searches:
        search_order = ("massive", "ticker-layer", "alpha-vantage")
        tools.append(
            RedundantEquitySymbolSearchTool(
                routes=tuple(
                    EquitySearchRoute(provider, searches[provider])
                    for provider in search_order
                    if provider in searches
                ),
                clock=clock,
            )
        )
    if peer_configured:
        profile_order = ("finnhub", "massive", "ticker-layer", "alpha-vantage")
        tools.append(
            RedundantEquityProfileTool(
                routes=tuple(
                    EquityProfileRoute(provider, profiles[provider])
                    for provider in profile_order
                    if provider in profiles
                ),
                clock=clock,
            )
        )

    return tuple(tools)


class _NormalizedFinnhubQuoteTool:
    """Give the raw Finnhub route a unique ID and the common canonical payload."""

    def __init__(self, delegate: FinnhubQuoteTool, *, tool_name: str) -> None:
        self._delegate = delegate
        self._spec = delegate.spec.model_copy(
            update={
                "name": tool_name,
                "version": "2.0.0",
                "description": "Return one normalized Finnhub equity quote directly.",
                "retry": ToolRetryPolicy(max_attempts=1),
            }
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return self._delegate.validate(arguments)

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._delegate.provider_health()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        outcome = await self._delegate.execute(arguments, context)
        if isinstance(outcome, ToolFailure):
            return outcome
        symbol = outcome.data.get("symbol")
        if not (
            isinstance(symbol, str)
            and valid_equity_quote_provenance(
                provider="finnhub",
                reference=outcome.source.reference,
                symbol=symbol,
                observed_at=outcome.observed_at,
            )
            and valid_equity_observed_at(outcome.data, outcome.observed_at)
        ):
            return ToolFailure(
                code="FINNHUB_QUOTE_CONTRACT_VIOLATION",
                safe_message="Finnhub returned a quote outside Leo's normalized contract.",
            )
        data: dict[str, JsonValue] = dict(outcome.data)
        data["provider"] = "finnhub"
        canonical = canonical_equity_quote_statement(data)
        if canonical is None:
            return ToolFailure(
                code="FINNHUB_QUOTE_CONTRACT_VIOLATION",
                safe_message="Finnhub returned a quote outside Leo's normalized contract.",
            )
        data["statements"] = [canonical]
        return outcome.model_copy(update={"data": data})


class _NormalizedFinnhubProfileTool:
    """Translate Company Profile 2 into the common profile evidence contract."""

    def __init__(self, delegate: FinnhubCompanyProfileTool) -> None:
        self._delegate = delegate
        self._spec = ToolSpec(
            name="market.get_company_profile_finnhub",
            version="2.0.0",
            description="Return one normalized Finnhub company profile directly.",
            domain="MARKET",
            input_schema=EquityProfileArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=8_192,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return EquityProfileArguments.model_validate(arguments).model_dump(mode="json")

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._delegate.provider_health()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = EquityProfileArguments.model_validate(arguments)
        outcome = await self._delegate.execute({"symbol": parsed.symbol}, context)
        if isinstance(outcome, ToolFailure):
            return outcome
        data: dict[str, JsonValue] = dict(outcome.data)
        data.update(
            {
                "provider": "finnhub",
                "provider_symbol": parsed.symbol,
                "as_of": outcome.observed_at.isoformat(),
            }
        )
        canonical = canonical_equity_profile_statements(data)
        if not (
            outcome.source.provider == "finnhub"
            and valid_equity_profile_provenance(
                provider="finnhub",
                reference=outcome.source.reference,
                provider_symbol=parsed.symbol,
            )
            and canonical is not None
        ):
            return ToolFailure(
                code="FINNHUB_PROFILE_CONTRACT_VIOLATION",
                safe_message="Finnhub returned a profile outside Leo's normalized contract.",
            )
        data["statements"] = list(canonical)
        return outcome.model_copy(update={"data": data})


def resolve_alpha_vantage_rest_base_url(
    *,
    configured_endpoint: SecretStr | str | None,
    configured_legacy_endpoint: SecretStr | str | None,
) -> str:
    """Use a generic endpoint only when it is exactly the official REST query URL."""

    for configured in (configured_endpoint, configured_legacy_endpoint):
        candidate = _secret_value(configured)
        if candidate and _is_official_url(
            candidate,
            host="www.alphavantage.co",
            path="/query",
        ):
            return candidate.rstrip("/")
    return _ALPHA_REST_URL


def resolve_massive_rest_base_url(
    *,
    configured_endpoint: SecretStr | str | None,
) -> str:
    """Use MASSIVE_ENDPOINT only when it is the credential-free official REST root."""

    candidate = _secret_value(configured_endpoint)
    if candidate and _is_official_url(candidate, host="api.massive.com", path=""):
        return candidate.rstrip("/")
    return _MASSIVE_REST_URL


def _secret_value(value: SecretStr | str | None) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value().strip()
    return value.strip() if isinstance(value, str) else ""


def _is_official_url(value: str, *, host: str, path: str) -> bool:
    try:
        parsed = httpx.URL(value)
    except (TypeError, ValueError):
        return False
    return bool(
        parsed.scheme == "https"
        and parsed.host == host
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") == path
        and not parsed.username
        and not parsed.password
        and not parsed.query
        and not parsed.fragment
    )


def _provider_gate(
    registry: ProviderGateRegistry | None,
    *,
    provider: str,
    clock: Clock,
    max_concurrency: int,
    max_calls_per_minute: int,
    max_calls_per_day: int | None = None,
    max_calls_per_month: int | None = None,
) -> ProviderCallGate:
    if registry is not None:
        return registry.get(
            provider=provider,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
            max_calls_per_day=max_calls_per_day,
            max_calls_per_month=max_calls_per_month,
        )
    return ProviderCallGate(
        provider=provider,
        clock=clock,
        max_concurrency=max_concurrency,
        max_calls_per_minute=max_calls_per_minute,
        max_calls_per_day=max_calls_per_day,
        max_calls_per_month=max_calls_per_month,
    )


__all__ = (
    "build_equity_market_tools",
    "resolve_alpha_vantage_rest_base_url",
    "resolve_massive_rest_base_url",
)
