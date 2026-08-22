"""Composition helper for optional cryptocurrency providers."""

from __future__ import annotations

from urllib.parse import urlsplit

import httpx
from pydantic import SecretStr

from leo.config import Settings, is_configured_secret
from leo.harness.ports import Clock, Tool
from leo.integrations.crypto_market import (
    CoinGeckoMarketSnapshotTool,
    CoinMarketCapMarketSnapshotTool,
    CryptoMarketSnapshotTool,
    CryptoSnapshotProvider,
)
from leo.integrations.provider_runtime import ProviderCallGate, ProviderGateRegistry

_DEFAULT_COINGECKO_REST_BASE = "https://api.coingecko.com/api/v3"
_COINGECKO_REST_HOSTS = frozenset({"api.coingecko.com", "pro-api.coingecko.com"})


def build_crypto_market_tools(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    clock: Clock,
    provider_gates: ProviderGateRegistry | None = None,
) -> tuple[Tool, ...]:
    """Build configured provider tools and one common resilient aggregate.

    Provider credentials are optional.  Missing credentials remove only that provider;
    they never gate the conversation harness or another integration.
    """

    providers: list[CryptoSnapshotProvider] = []
    tools: list[Tool] = []
    if is_configured_secret(settings.coingecko_api_key):
        assert settings.coingecko_api_key is not None
        try:
            coingecko_gate = _provider_gate(
                provider_gates,
                provider="coingecko",
                clock=clock,
                max_calls_per_minute=settings.coingecko_max_calls_per_minute,
            )
            coingecko = CoinGeckoMarketSnapshotTool(
                client=client,
                api_key=settings.coingecko_api_key.get_secret_value(),
                clock=clock,
                base_url=resolve_coingecko_rest_base_url(
                    configured_base=settings.coingecko_base_url,
                    configured_endpoint=settings.coingecko_endpoint,
                ),
                gate=coingecko_gate,
            )
        except (ValueError, httpx.InvalidURL):
            pass
        else:
            providers.append(coingecko)
            tools.append(coingecko)
    if is_configured_secret(settings.coin_market_cap_api_key):
        assert settings.coin_market_cap_api_key is not None
        try:
            coinmarketcap_gate = _provider_gate(
                provider_gates,
                provider="coinmarketcap",
                clock=clock,
                max_calls_per_minute=settings.coin_market_cap_max_calls_per_minute,
            )
            coinmarketcap = CoinMarketCapMarketSnapshotTool(
                client=client,
                api_key=settings.coin_market_cap_api_key.get_secret_value(),
                clock=clock,
                base_url=settings.coin_market_cap_base_url,
                gate=coinmarketcap_gate,
            )
        except (ValueError, httpx.InvalidURL):
            pass
        else:
            providers.append(coinmarketcap)
            tools.append(coinmarketcap)
    if providers:
        tools.append(
            CryptoMarketSnapshotTool(
                providers,
                agreement_threshold_bps=settings.crypto_agreement_threshold_bps,
                max_corroboration_skew_seconds=(settings.crypto_max_corroboration_skew_seconds),
            )
        )
    return tuple(tools)


def _provider_gate(
    registry: ProviderGateRegistry | None,
    *,
    provider: str,
    clock: Clock,
    max_calls_per_minute: int,
) -> ProviderCallGate:
    if registry is not None:
        return registry.get(
            provider=provider,
            max_concurrency=2,
            max_calls_per_minute=max_calls_per_minute,
        )
    return ProviderCallGate(
        provider=provider,
        clock=clock,
        max_concurrency=2,
        max_calls_per_minute=max_calls_per_minute,
    )


def resolve_coingecko_rest_base_url(
    *,
    configured_base: str,
    configured_endpoint: SecretStr | str | None,
) -> str:
    """Validate only the native REST base; MCP endpoint authority stays separate."""

    # COINGECKO_ENDPOINT belongs to a separate MCP transport and may carry query
    # credentials. It must never influence native REST destination selection.
    del configured_endpoint
    base = configured_base.strip() or _DEFAULT_COINGECKO_REST_BASE
    parsed_base = urlsplit(base)
    if (
        parsed_base.scheme != "https"
        or parsed_base.hostname not in _COINGECKO_REST_HOSTS
        or parsed_base.path.rstrip("/") != "/api/v3"
        or parsed_base.port not in {None, 443}
        or parsed_base.query
        or parsed_base.fragment
        or parsed_base.username is not None
        or parsed_base.password is not None
    ):
        raise ValueError("CoinGecko REST base must use an official /api/v3 endpoint")
    return base.rstrip("/")


__all__ = ["build_crypto_market_tools", "resolve_coingecko_rest_base_url"]
