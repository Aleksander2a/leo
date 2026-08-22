"""Discoverable metadata for native equity market provider tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from leo.capabilities.catalog import CapabilityLatency
from leo.capabilities.provider_descriptors import ProviderCapabilityDescriptor

_QUOTE_TAGS = frozenset(
    {
        "agreement",
        "compare",
        "corroborate",
        "current",
        "equity",
        "failover",
        "latest",
        "market",
        "market-data",
        "price",
        "quote",
        "redundancy",
        "resilient",
        "stock",
    }
)
_PROFILE_TAGS = frozenset(
    {
        "company",
        "equity",
        "exchange",
        "identity",
        "industry",
        "listing",
        "market",
        "profile",
        "stock",
    }
)
_SEARCH_TAGS = frozenset(
    {
        "company",
        "discover",
        "equity",
        "find",
        "lookup",
        "market",
        "search",
        "stock",
        "symbol",
        "ticker",
    }
)
_CANONICAL_VERIFICATION = frozenset(
    {"canonical_statement", "exact_provenance", "provider_reported"}
)

EQUITY_CAPABILITY_DESCRIPTORS: Final[Mapping[str, ProviderCapabilityDescriptor]] = MappingProxyType(
    {
        "market.get_quote": ProviderCapabilityDescriptor(
            provider="equity-corroboration",
            runtime_health_providers=frozenset(
                {"alpha_vantage", "finnhub", "massive", "ticker_layer"}
            ),
            tags=_QUOTE_TAGS,
            freshness_seconds=900,
            latency=CapabilityLatency.MEDIUM,
            verification_expectations=_CANONICAL_VERIFICATION
            | {
                "as_of_freshness",
                "deterministic_failover",
                "exact_numeric_grounding",
                "partial_failure_accounting",
                "provider_disagreement_measurement",
                "timestamp_skew_accounting",
            },
        ),
        "market.search_equity_symbols": ProviderCapabilityDescriptor(
            provider="equity-search-failover",
            runtime_health_providers=frozenset({"alpha_vantage", "massive", "ticker_layer"}),
            tags=_SEARCH_TAGS | {"deterministic", "failover", "resilient"},
            freshness_seconds=3_600,
            latency=CapabilityLatency.MEDIUM,
            verification_expectations=_CANONICAL_VERIFICATION
            | {"deterministic_failover", "partial_failure_accounting"},
        ),
        "market.get_equity_profile": ProviderCapabilityDescriptor(
            provider="equity-profile-failover",
            runtime_health_providers=frozenset(
                {"alpha_vantage", "finnhub", "massive", "ticker_layer"}
            ),
            tags=_PROFILE_TAGS | {"deterministic", "failover", "resilient"},
            freshness_seconds=86_400,
            latency=CapabilityLatency.MEDIUM,
            verification_expectations=_CANONICAL_VERIFICATION
            | {"deterministic_failover", "partial_failure_accounting"},
        ),
        "market.get_quote_alpha_vantage": ProviderCapabilityDescriptor(
            provider="alpha-vantage",
            runtime_health_providers=frozenset({"alpha_vantage"}),
            tags=_QUOTE_TAGS | {"alpha-vantage", "end-of-day"},
            freshness_seconds=900,
            rate_limit_per_minute=5,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION
            | {"end_of_day_timestamp", "exact_numeric_grounding"},
        ),
        "market.get_quote_finnhub": ProviderCapabilityDescriptor(
            provider="finnhub",
            runtime_health_providers=frozenset({"finnhub"}),
            tags=_QUOTE_TAGS | {"finnhub"},
            freshness_seconds=900,
            rate_limit_per_minute=60,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION
            | {"as_of_freshness", "exact_numeric_grounding"},
        ),
        "market.get_quote_massive": ProviderCapabilityDescriptor(
            provider="massive",
            runtime_health_providers=frozenset({"massive"}),
            tags=_QUOTE_TAGS | {"entitlement-dependent", "massive", "snapshot"},
            freshness_seconds=900,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION
            | {"as_of_freshness", "exact_numeric_grounding"},
        ),
        "market.get_quote_ticker_layer": ProviderCapabilityDescriptor(
            provider="ticker-layer",
            runtime_health_providers=frozenset({"ticker_layer"}),
            tags=_QUOTE_TAGS | {"indicative", "non-exchange", "ticker-layer"},
            freshness_seconds=900,
            rate_limit_per_minute=60,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION
            | {
                "as_of_freshness",
                "exact_numeric_grounding",
                "non_exchange_indicative",
            },
        ),
        "market.search_symbols_alpha_vantage": ProviderCapabilityDescriptor(
            provider="alpha-vantage",
            runtime_health_providers=frozenset({"alpha_vantage"}),
            tags=_SEARCH_TAGS | {"alpha-vantage", "global"},
            freshness_seconds=3_600,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION,
        ),
        "market.get_company_profile_alpha_vantage": ProviderCapabilityDescriptor(
            provider="alpha-vantage",
            runtime_health_providers=frozenset({"alpha_vantage"}),
            tags=_PROFILE_TAGS | {"alpha-vantage", "fundamentals", "overview"},
            freshness_seconds=86_400,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION,
        ),
        "market.get_company_profile_finnhub": ProviderCapabilityDescriptor(
            provider="finnhub",
            runtime_health_providers=frozenset({"finnhub"}),
            tags=_PROFILE_TAGS | {"finnhub"},
            freshness_seconds=86_400,
            rate_limit_per_minute=60,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION,
        ),
        "market.search_symbols_massive": ProviderCapabilityDescriptor(
            provider="massive",
            runtime_health_providers=frozenset({"massive"}),
            tags=_SEARCH_TAGS | {"massive", "reference"},
            freshness_seconds=3_600,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION,
        ),
        "market.get_company_profile_massive": ProviderCapabilityDescriptor(
            provider="massive",
            runtime_health_providers=frozenset({"massive"}),
            tags=_PROFILE_TAGS | {"massive", "reference", "sic"},
            freshness_seconds=86_400,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION,
        ),
        "market.search_symbols_ticker_layer": ProviderCapabilityDescriptor(
            provider="ticker-layer",
            runtime_health_providers=frozenset({"ticker_layer"}),
            tags=_SEARCH_TAGS | {"enabled-market", "ticker-layer"},
            freshness_seconds=3_600,
            rate_limit_per_minute=60,
            latency=CapabilityLatency.MEDIUM,
            verification_expectations=_CANONICAL_VERIFICATION | {"non_exchange_indicative"},
        ),
        "market.get_company_profile_ticker_layer": ProviderCapabilityDescriptor(
            provider="ticker-layer",
            runtime_health_providers=frozenset({"ticker_layer"}),
            tags=_PROFILE_TAGS | {"fundamentals", "permission-dependent", "ticker-layer"},
            freshness_seconds=7_776_000,
            rate_limit_per_minute=60,
            latency=CapabilityLatency.LOW,
            verification_expectations=_CANONICAL_VERIFICATION | {"non_exchange_indicative"},
        ),
    }
)


def equity_capability_descriptor(tool_name: str) -> ProviderCapabilityDescriptor | None:
    return EQUITY_CAPABILITY_DESCRIPTORS.get(tool_name)


__all__ = ["EQUITY_CAPABILITY_DESCRIPTORS", "equity_capability_descriptor"]
