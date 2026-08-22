"""Focused process-local provider health and quota contract tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.integrations.fake import FixedClock
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRegistry,
    ProviderGateRejected,
)


@pytest.mark.asyncio
async def test_registry_current_and_all_snapshots_are_bounded_and_content_free() -> None:
    clock = FixedClock(datetime(2026, 8, 22, 12, 0, tzinfo=UTC))
    registry = ProviderGateRegistry(clock)
    tavily = registry.get(provider="tavily", max_calls_per_minute=10)
    registry.get(provider="exa", max_calls_per_minute=20)

    async with tavily.slot():
        await tavily.record_failure(
            "TAVILY_RATE_LIMITED",
            rate_limited=True,
            retry_after_seconds=90,
        )

    current = await registry.snapshot_provider("TAVILY")
    assert current is not None
    assert current.provider == "tavily"
    assert current.status == "rate_limited"
    assert current.cooldown_active is True
    assert current.accounting_scope == "process_lifetime"
    assert await registry.snapshot_provider("unregistered") is None

    snapshots = await registry.snapshot_all()
    assert tuple(item.provider for item in snapshots) == ("exa", "tavily")
    encoded = "".join(item.model_dump_json() for item in snapshots)
    assert "TAVILY_RATE_LIMITED" not in encoded
    assert "https://" not in encoded
    assert "api_key" not in encoded
    assert set(current.model_dump()) == {
        "provider",
        "status",
        "minute_calls_available",
        "daily_calls_available",
        "monthly_calls_available",
        "monthly_provider_credits_available",
        "cooldown_active",
        "accounting_scope",
    }


@pytest.mark.asyncio
async def test_monthly_provider_credit_limit_blocks_after_recorded_cmc_credits_and_resets() -> None:
    clock = FixedClock(datetime(2026, 8, 31, 23, 59, tzinfo=UTC))
    gate = ProviderCallGate(
        provider="coinmarketcap",
        clock=clock,
        max_calls_per_minute=10,
        max_provider_credits_per_month=3,
    )

    async with gate.slot():
        await gate.record_success(provider_credits_used=3)

    exhausted = await gate.snapshot()
    assert exhausted.status == "rate_limited"
    assert exhausted.provider_credits_used == 3
    assert exhausted.provider_credits_used_in_month == 3
    assert exhausted.local_monthly_provider_credit_limit == 3
    assert exhausted.remaining_local_monthly_provider_credits == 0
    assert exhausted.accounting_scope == "process_lifetime"
    with pytest.raises(ProviderGateRejected) as blocked:
        async with gate.slot():
            pass
    assert blocked.value.code == "COINMARKETCAP_LOCAL_MONTHLY_PROVIDER_CREDIT_LIMIT"

    clock.advance(seconds=120)
    async with gate.slot():
        await gate.record_success(provider_credits_used=1)
    reset = await gate.snapshot()
    assert reset.status == "healthy"
    assert reset.provider_credits_used == 4
    assert reset.provider_credits_used_in_month == 1
    assert reset.remaining_local_monthly_provider_credits == 2


def test_registry_rejects_monthly_provider_credit_policy_drift() -> None:
    registry = ProviderGateRegistry(FixedClock())
    original = registry.get(
        provider="coinmarketcap",
        max_provider_credits_per_month=100,
    )
    assert (
        registry.get(
            provider="COINMARKETCAP",
            max_provider_credits_per_month=100,
        )
        is original
    )
    with pytest.raises(ValueError, match="policy changed"):
        registry.get(
            provider="coinmarketcap",
            max_provider_credits_per_month=101,
        )


@pytest.mark.asyncio
async def test_registry_health_is_explicitly_process_local_and_resets_on_restart() -> None:
    clock = FixedClock()
    first_registry = ProviderGateRegistry(clock)
    first_gate = first_registry.get(provider="exa", max_calls_per_minute=20)
    async with first_gate.slot():
        await first_gate.record_failure(
            "EXA_RATE_LIMITED",
            rate_limited=True,
            retry_after_seconds=60,
        )
    first = await first_registry.snapshot_provider("exa")
    assert first is not None
    assert first.status == "rate_limited"
    assert first.accounting_scope == "process_lifetime"

    restarted_registry = ProviderGateRegistry(clock)
    restarted_registry.get(provider="exa", max_calls_per_minute=20)
    restarted = await restarted_registry.snapshot_provider("exa")
    assert restarted is not None
    assert restarted.status == "healthy"
    assert restarted.accounting_scope == "process_lifetime"


def test_registry_has_a_bounded_provider_projection_set() -> None:
    registry = ProviderGateRegistry(FixedClock(), max_registered_providers=1)
    registry.get(provider="exa")
    with pytest.raises(ValueError, match="capacity exceeded"):
        registry.get(provider="tavily")
