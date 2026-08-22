"""Shared bounded provider-call health and local quota gates.

The gate is deliberately transport-agnostic.  One composition root creates one
instance per credential/provider and shares it across every adapter backed by that
credential.  It does not sleep inside an agent run: a provider cooldown fails fast so
another provider can be tried without spending the remaining run budget.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal

from leo.harness.ports import Clock
from leo.harness.provider_health import ProviderHealthProjection, ProviderHealthSnapshot


class ProviderGateRejected(RuntimeError):
    """A local provider boundary rejected a call before network I/O."""

    def __init__(self, *, code: str, safe_message: str) -> None:
        super().__init__(code)
        self.code = code
        self.safe_message = safe_message


class ProviderCallGate:
    """Concurrency, fixed-window call budget, cooldown, and health for one provider."""

    def __init__(
        self,
        *,
        provider: str,
        clock: Clock,
        max_concurrency: int = 2,
        max_calls_per_minute: int = 20,
        max_calls_per_day: int | None = None,
        max_calls_per_month: int | None = None,
        max_provider_credits_per_month: int | None = None,
        max_cooldown_seconds: int = 300,
    ) -> None:
        normalized_provider = _normalize_provider(provider)
        if (
            max_concurrency < 1
            or max_calls_per_minute < 1
            or (max_calls_per_day is not None and max_calls_per_day < 1)
            or (max_calls_per_month is not None and max_calls_per_month < 1)
            or (max_provider_credits_per_month is not None and max_provider_credits_per_month < 1)
            or max_cooldown_seconds < 1
        ):
            raise ValueError("provider call limits must be positive")
        self._provider = normalized_provider
        self._clock = clock
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._state_lock = asyncio.Lock()
        self._max_calls_per_minute = max_calls_per_minute
        self._max_calls_per_day = max_calls_per_day
        self._max_calls_per_month = max_calls_per_month
        self._max_provider_credits_per_month = max_provider_credits_per_month
        self._max_cooldown_seconds = max_cooldown_seconds
        initial_now = clock.now()
        if initial_now.tzinfo is None or initial_now.utcoffset() is None:
            raise ValueError("provider call gate clock must return timezone-aware timestamps")
        self._window_started_at = initial_now
        self._day_started_at = initial_now
        self._month_started_at = initial_now
        self._calls_in_window = 0
        self._calls_in_day = 0
        self._calls_in_month = 0
        self._successes = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._rate_limit_count = 0
        self._provider_credits_used = 0
        self._provider_credits_used_in_month = 0
        self._cooldown_until: datetime | None = None
        self._last_failure_code: str | None = None

    @property
    def provider(self) -> str:
        return self._provider

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Reserve one bounded call or fail fast before touching the network."""

        async with self._semaphore:
            async with self._state_lock:
                now = self._clock.now()
                self._refresh_window(now)
                if self._cooldown_until is not None and now < self._cooldown_until:
                    raise ProviderGateRejected(
                        code=f"{self._provider.upper()}_COOLDOWN_ACTIVE",
                        safe_message=(
                            f"{self._provider} is temporarily cooling down after rate limiting."
                        ),
                    )
                if self._calls_in_window >= self._max_calls_per_minute:
                    raise ProviderGateRejected(
                        code=f"{self._provider.upper()}_LOCAL_RATE_LIMIT",
                        safe_message=(
                            f"{self._provider} reached Leo's bounded local call allowance."
                        ),
                    )
                if (
                    self._max_calls_per_day is not None
                    and self._calls_in_day >= self._max_calls_per_day
                ):
                    raise ProviderGateRejected(
                        code=f"{self._provider.upper()}_LOCAL_DAILY_RATE_LIMIT",
                        safe_message=(
                            f"{self._provider} reached Leo's bounded local daily call allowance."
                        ),
                    )
                if (
                    self._max_calls_per_month is not None
                    and self._calls_in_month >= self._max_calls_per_month
                ):
                    raise ProviderGateRejected(
                        code=f"{self._provider.upper()}_LOCAL_MONTHLY_RATE_LIMIT",
                        safe_message=(
                            f"{self._provider} reached Leo's bounded local monthly call allowance."
                        ),
                    )
                if (
                    self._max_provider_credits_per_month is not None
                    and self._provider_credits_used_in_month >= self._max_provider_credits_per_month
                ):
                    raise ProviderGateRejected(
                        code=(f"{self._provider.upper()}_LOCAL_MONTHLY_PROVIDER_CREDIT_LIMIT"),
                        safe_message=(
                            f"{self._provider} reached Leo's bounded local monthly "
                            "provider-credit allowance."
                        ),
                    )
                self._calls_in_window += 1
                self._calls_in_day += 1
                self._calls_in_month += 1
            yield

    async def record_success(self, *, provider_credits_used: int = 0) -> None:
        if provider_credits_used < 0:
            raise ValueError("provider credits used cannot be negative")
        async with self._state_lock:
            now = self._clock.now()
            self._refresh_window(now)
            self._successes += 1
            self._consecutive_failures = 0
            self._provider_credits_used += provider_credits_used
            self._provider_credits_used_in_month += provider_credits_used
            self._last_failure_code = None
            if self._cooldown_until is not None and now >= self._cooldown_until:
                self._cooldown_until = None

    async def record_failure(
        self,
        code: str,
        *,
        rate_limited: bool = False,
        retry_after_seconds: int | None = None,
        provider_credits_used: int = 0,
    ) -> None:
        normalized_code = _normalize_failure_code(code)
        if provider_credits_used < 0:
            raise ValueError("provider credits used cannot be negative")
        async with self._state_lock:
            now = self._clock.now()
            self._refresh_window(now)
            self._failures += 1
            self._consecutive_failures += 1
            self._provider_credits_used += provider_credits_used
            self._provider_credits_used_in_month += provider_credits_used
            self._last_failure_code = normalized_code
            if rate_limited:
                self._rate_limit_count += 1
                retry_after = min(
                    max(retry_after_seconds or 60, 1),
                    self._max_cooldown_seconds,
                )
                candidate = now + timedelta(seconds=retry_after)
                if self._cooldown_until is None or candidate > self._cooldown_until:
                    self._cooldown_until = candidate

    async def snapshot(self) -> ProviderHealthSnapshot:
        async with self._state_lock:
            now = self._clock.now()
            self._refresh_window(now)
            cooldown_active = self._cooldown_until is not None and now < self._cooldown_until
            status: Literal["healthy", "degraded", "rate_limited"]
            daily_limit_reached = (
                self._max_calls_per_day is not None
                and self._calls_in_day >= self._max_calls_per_day
            )
            monthly_limit_reached = (
                self._max_calls_per_month is not None
                and self._calls_in_month >= self._max_calls_per_month
            )
            monthly_provider_credit_limit_reached = (
                self._max_provider_credits_per_month is not None
                and self._provider_credits_used_in_month >= self._max_provider_credits_per_month
            )
            if (
                cooldown_active
                or self._calls_in_window >= self._max_calls_per_minute
                or daily_limit_reached
                or monthly_limit_reached
                or monthly_provider_credit_limit_reached
            ):
                status = "rate_limited"
            elif self._consecutive_failures:
                status = "degraded"
            else:
                status = "healthy"
            return ProviderHealthSnapshot(
                provider=self._provider,
                status=status,
                calls_in_window=self._calls_in_window,
                local_call_limit=self._max_calls_per_minute,
                remaining_local_calls=max(
                    self._max_calls_per_minute - self._calls_in_window,
                    0,
                ),
                calls_in_day=self._calls_in_day,
                local_daily_call_limit=self._max_calls_per_day,
                remaining_local_daily_calls=(
                    None
                    if self._max_calls_per_day is None
                    else max(self._max_calls_per_day - self._calls_in_day, 0)
                ),
                calls_in_month=self._calls_in_month,
                local_monthly_call_limit=self._max_calls_per_month,
                remaining_local_monthly_calls=(
                    None
                    if self._max_calls_per_month is None
                    else max(self._max_calls_per_month - self._calls_in_month, 0)
                ),
                successes=self._successes,
                failures=self._failures,
                consecutive_failures=self._consecutive_failures,
                rate_limit_count=self._rate_limit_count,
                provider_credits_used=self._provider_credits_used,
                provider_credits_used_in_month=self._provider_credits_used_in_month,
                local_monthly_provider_credit_limit=(self._max_provider_credits_per_month),
                remaining_local_monthly_provider_credits=(
                    None
                    if self._max_provider_credits_per_month is None
                    else max(
                        self._max_provider_credits_per_month - self._provider_credits_used_in_month,
                        0,
                    )
                ),
                window_started_at=self._window_started_at,
                day_started_at=self._day_started_at,
                month_started_at=self._month_started_at,
                cooldown_until=self._cooldown_until if cooldown_active else None,
                last_failure_code=self._last_failure_code,
            )

    def _refresh_window(self, now: datetime) -> None:
        if now - self._window_started_at >= timedelta(minutes=1):
            self._window_started_at = now
            self._calls_in_window = 0
        if now.astimezone(UTC).date() != self._day_started_at.astimezone(UTC).date():
            self._day_started_at = now
            self._calls_in_day = 0
        now_utc = now.astimezone(UTC)
        month_started_utc = self._month_started_at.astimezone(UTC)
        if (now_utc.year, now_utc.month) != (
            month_started_utc.year,
            month_started_utc.month,
        ):
            self._month_started_at = now
            self._calls_in_month = 0
            self._provider_credits_used_in_month = 0
        if self._cooldown_until is not None and now >= self._cooldown_until:
            self._cooldown_until = None


@dataclass(frozen=True, slots=True)
class _ProviderGatePolicy:
    max_concurrency: int
    max_calls_per_minute: int
    max_calls_per_day: int | None
    max_calls_per_month: int | None
    max_provider_credits_per_month: int | None
    max_cooldown_seconds: int


class ProviderGateRegistry:
    """Process-owned provider health shared across runs in one runtime.

    The registry is intentionally not durable. Cooldowns and locally observed quota
    counters survive individual conversation turns only while this Python process is
    alive; a restart creates a fresh registry and must not be represented as provider
    account truth.
    """

    def __init__(self, clock: Clock, *, max_registered_providers: int = 64) -> None:
        if not 1 <= max_registered_providers <= 256:
            raise ValueError("provider registry bound must be between 1 and 256")
        self._clock = clock
        self._max_registered_providers = max_registered_providers
        self._gates: dict[str, tuple[_ProviderGatePolicy, ProviderCallGate]] = {}

    def get(
        self,
        *,
        provider: str,
        max_concurrency: int = 2,
        max_calls_per_minute: int = 20,
        max_calls_per_day: int | None = None,
        max_calls_per_month: int | None = None,
        max_provider_credits_per_month: int | None = None,
        max_cooldown_seconds: int = 300,
    ) -> ProviderCallGate:
        normalized_provider = _normalize_provider(provider)
        policy = _ProviderGatePolicy(
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
            max_calls_per_day=max_calls_per_day,
            max_calls_per_month=max_calls_per_month,
            max_provider_credits_per_month=max_provider_credits_per_month,
            max_cooldown_seconds=max_cooldown_seconds,
        )
        existing = self._gates.get(normalized_provider)
        if existing is not None:
            existing_policy, gate = existing
            if existing_policy != policy:
                raise ValueError("provider gate policy changed inside one runtime")
            return gate
        if len(self._gates) >= self._max_registered_providers:
            raise ValueError("provider registry capacity exceeded")
        gate = ProviderCallGate(
            provider=normalized_provider,
            clock=self._clock,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
            max_calls_per_day=max_calls_per_day,
            max_calls_per_month=max_calls_per_month,
            max_provider_credits_per_month=max_provider_credits_per_month,
            max_cooldown_seconds=max_cooldown_seconds,
        )
        self._gates[normalized_provider] = (policy, gate)
        return gate

    @property
    def registered_providers(self) -> tuple[str, ...]:
        return tuple(sorted(self._gates))

    async def snapshot_provider(self, provider: str) -> ProviderHealthProjection | None:
        """Return one content-free current projection, or ``None`` if unregistered."""

        normalized_provider = _normalize_provider(provider)
        existing = self._gates.get(normalized_provider)
        if existing is None:
            return None
        _policy, gate = existing
        return ProviderHealthProjection.from_snapshot(await gate.snapshot())

    async def snapshot_all(self) -> tuple[ProviderHealthProjection, ...]:
        """Return bounded, provider-sorted projections for this process registry."""

        gates = tuple(gate for _provider, (_policy, gate) in sorted(self._gates.items()))
        snapshots = await asyncio.gather(*(gate.snapshot() for gate in gates))
        return tuple(ProviderHealthProjection.from_snapshot(item) for item in snapshots)


def _normalize_provider(provider: str) -> str:
    normalized_provider = provider.strip().casefold()
    if (
        len(normalized_provider) > 64
        or re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", normalized_provider) is None
    ):
        raise ValueError("provider name must be a normalized identifier")
    return normalized_provider


def _normalize_failure_code(code: str) -> str:
    normalized_code = code.strip().upper()
    if (
        len(normalized_code) > 96
        or re.fullmatch(r"[A-Z0-9]+(?:_[A-Z0-9]+)*", normalized_code) is None
    ):
        raise ValueError("provider failure code must be a normalized identifier")
    return normalized_code


def bounded_retry_after(value: str | None, *, maximum_seconds: int = 300) -> int | None:
    """Parse only the integer Retry-After form and keep it inside a local bound."""

    if value is None or not value.strip().isdigit() or maximum_seconds < 1:
        return None
    return min(max(int(value.strip()), 1), maximum_seconds)
