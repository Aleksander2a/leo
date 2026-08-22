"""Discoverable metadata for native cryptocurrency market tools."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from leo.capabilities.catalog import CapabilityLatency
from leo.capabilities.provider_descriptors import ProviderCapabilityDescriptor

_COMMON_TAGS = frozenset(
    {
        "bitcoin",
        "coin",
        "crypto",
        "cryptocurrency",
        "ethereum",
        "market",
        "market-data",
        "price",
        "quote",
        "token",
    }
)
_COMMON_VERIFICATION = frozenset(
    {
        "as_of_freshness",
        "canonical_statement",
        "exact_numeric_grounding",
        "provider_reported",
    }
)

CRYPTO_CAPABILITY_DESCRIPTORS: Final[Mapping[str, ProviderCapabilityDescriptor]] = MappingProxyType(
    {
        "market.get_crypto_snapshot": ProviderCapabilityDescriptor(
            provider="crypto-corroboration",
            runtime_health_providers=frozenset({"coingecko", "coinmarketcap"}),
            tags=_COMMON_TAGS
            | {
                "agreement",
                "compare",
                "corroborate",
                "cross-check",
                "divergence",
                "fallback",
                "redundancy",
                "resilient",
            },
            freshness_seconds=180,
            latency=CapabilityLatency.MEDIUM,
            verification_expectations=_COMMON_VERIFICATION
            | {
                "partial_failure_accounting",
                "provenance_digest",
                "provider_agreement_measurement",
                "timestamp_skew_accounting",
            },
        ),
        "market.get_crypto_snapshot_coingecko": ProviderCapabilityDescriptor(
            provider="coingecko",
            runtime_health_providers=frozenset({"coingecko"}),
            tags=_COMMON_TAGS | {"coingecko"},
            freshness_seconds=180,
            rate_limit_per_minute=20,
            latency=CapabilityLatency.LOW,
            verification_expectations=_COMMON_VERIFICATION,
        ),
        "market.get_crypto_snapshot_coinmarketcap": ProviderCapabilityDescriptor(
            provider="coinmarketcap",
            runtime_health_providers=frozenset({"coinmarketcap"}),
            tags=_COMMON_TAGS | {"coinmarketcap", "cmc"},
            freshness_seconds=180,
            rate_limit_per_minute=20,
            latency=CapabilityLatency.LOW,
            verification_expectations=_COMMON_VERIFICATION,
        ),
    }
)


def crypto_capability_descriptor(tool_name: str) -> ProviderCapabilityDescriptor | None:
    return CRYPTO_CAPABILITY_DESCRIPTORS.get(tool_name)


__all__ = ["CRYPTO_CAPABILITY_DESCRIPTORS", "crypto_capability_descriptor"]
