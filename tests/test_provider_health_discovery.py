"""Cross-turn provider health projection into bounded capability discovery."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import JsonValue

from leo.capabilities.catalog import CapabilityHealth, InMemoryToolCatalog
from leo.capabilities.equity_descriptors import EQUITY_CAPABILITY_DESCRIPTORS
from leo.harness.models import (
    RunPhase,
    ToolEffect,
    ToolExecutionContext,
    ToolOutcome,
    ToolSpec,
)
from leo.integrations.fake import FixedClock
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.live import _conversation_capability_catalog

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


class _CatalogOnlyTool:
    def __init__(self, name: str) -> None:
        self._spec = ToolSpec(
            name=name,
            version="1.0.0",
            description=f"Fixture discovery surface for {name}.",
            domain="MARKET" if name.startswith("market.") else "WEB",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return arguments

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del arguments, context
        raise AssertionError("catalog-only tools are never executed")


def _eligible_ids(catalog: InMemoryToolCatalog) -> frozenset[str]:
    records = catalog.eligible(
        phase=RunPhase.RESEARCH,
        profile="research",
        remaining_cost=100,
    )
    return frozenset(item.id for item in records)


@pytest.mark.asyncio
async def test_cross_turn_cooldown_removes_direct_tool_then_degraded_route_recovers() -> None:
    clock = FixedClock(NOW)
    registry = ProviderGateRegistry(clock)
    gate = registry.get(provider="coingecko", max_calls_per_minute=20)
    direct = _CatalogOnlyTool("market.get_crypto_snapshot_coingecko")

    first_catalog = _conversation_capability_catalog(
        [direct],
        provider_health=await registry.snapshot_all(),
    )
    assert first_catalog.get(direct.spec.name).health is CapabilityHealth.HEALTHY
    assert direct.spec.name in _eligible_ids(first_catalog)

    await gate.record_failure(
        "COINGECKO_RATE_LIMITED",
        rate_limited=True,
        retry_after_seconds=60,
    )
    next_turn_catalog = _conversation_capability_catalog(
        [direct],
        provider_health=await registry.snapshot_all(),
    )
    assert next_turn_catalog.get(direct.spec.name).health is CapabilityHealth.RATE_LIMITED
    assert direct.spec.name not in _eligible_ids(next_turn_catalog)

    clock.advance(seconds=61)
    recovery_turn_catalog = _conversation_capability_catalog(
        [direct],
        provider_health=await registry.snapshot_all(),
    )
    assert recovery_turn_catalog.get(direct.spec.name).health is CapabilityHealth.DEGRADED
    assert direct.spec.name in _eligible_ids(recovery_turn_catalog)


@pytest.mark.asyncio
async def test_provider_family_stays_eligible_when_one_explicit_peer_is_healthy() -> None:
    registry = ProviderGateRegistry(FixedClock(NOW))
    registry.get(provider="coingecko", max_calls_per_minute=20)
    coinmarketcap = registry.get(provider="coinmarketcap", max_calls_per_minute=20)
    await coinmarketcap.record_failure(
        "COINMARKETCAP_RATE_LIMITED",
        rate_limited=True,
        retry_after_seconds=60,
    )
    aggregate = _CatalogOnlyTool("market.get_crypto_snapshot")
    direct_healthy = _CatalogOnlyTool("market.get_crypto_snapshot_coingecko")
    direct_limited = _CatalogOnlyTool("market.get_crypto_snapshot_coinmarketcap")

    catalog = _conversation_capability_catalog(
        [aggregate, direct_healthy, direct_limited],
        provider_health=await registry.snapshot_all(),
    )
    assert catalog.get(aggregate.spec.name).health is CapabilityHealth.HEALTHY
    assert catalog.get(direct_healthy.spec.name).health is CapabilityHealth.HEALTHY
    assert catalog.get(direct_limited.spec.name).health is CapabilityHealth.RATE_LIMITED
    assert _eligible_ids(catalog) == frozenset({aggregate.spec.name, direct_healthy.spec.name})


def test_equity_health_authorities_are_explicit_registry_ids() -> None:
    assert EQUITY_CAPABILITY_DESCRIPTORS[
        "market.get_quote_alpha_vantage"
    ].runtime_health_providers == frozenset({"alpha_vantage"})
    assert EQUITY_CAPABILITY_DESCRIPTORS[
        "market.get_quote_ticker_layer"
    ].runtime_health_providers == frozenset({"ticker_layer"})
    assert EQUITY_CAPABILITY_DESCRIPTORS["market.get_quote"].runtime_health_providers == (
        frozenset({"alpha_vantage", "finnhub", "massive", "ticker_layer"})
    )
