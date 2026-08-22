"""Provider-neutral, content-safe health contracts exposed to the harness."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field

from leo.harness.models import ContractModel

ProviderName = Annotated[
    str,
    Field(
        min_length=1,
        max_length=64,
        pattern=r"^[a-z0-9]+(?:_[a-z0-9]+)*$",
    ),
]
ProviderFailureCode = Annotated[
    str,
    Field(min_length=1, max_length=96, pattern=r"^[A-Z0-9]+(?:_[A-Z0-9]+)*$"),
]


class ProviderHealthSnapshot(ContractModel):
    """Point-in-time process-local health; never provider account truth or secrets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    status: Literal["healthy", "degraded", "rate_limited"]
    calls_in_window: int = Field(ge=0)
    local_call_limit: int = Field(ge=1)
    remaining_local_calls: int = Field(ge=0)
    calls_in_day: int = Field(ge=0)
    local_daily_call_limit: int | None = Field(default=None, ge=1)
    remaining_local_daily_calls: int | None = Field(default=None, ge=0)
    calls_in_month: int = Field(ge=0)
    local_monthly_call_limit: int | None = Field(default=None, ge=1)
    remaining_local_monthly_calls: int | None = Field(default=None, ge=0)
    successes: int = Field(ge=0)
    failures: int = Field(ge=0)
    consecutive_failures: int = Field(ge=0)
    rate_limit_count: int = Field(ge=0)
    provider_credits_used: int = Field(ge=0)
    provider_credits_used_in_month: int = Field(ge=0)
    local_monthly_provider_credit_limit: int | None = Field(default=None, ge=1)
    remaining_local_monthly_provider_credits: int | None = Field(default=None, ge=0)
    window_started_at: datetime
    day_started_at: datetime
    month_started_at: datetime
    cooldown_until: datetime | None = None
    last_failure_code: ProviderFailureCode | None = None
    accounting_scope: Literal["process_lifetime"] = "process_lifetime"


class ProviderHealthProjection(ContractModel):
    """Bounded, content-free provider health used by capability discovery.

    This projection deliberately excludes URLs, request/response content, credentials,
    raw failure messages, and exact timestamps. Its accounting scope makes the runtime
    limitation explicit: a fresh process starts with a fresh local registry.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: ProviderName
    status: Literal["healthy", "degraded", "rate_limited"]
    minute_calls_available: bool
    daily_calls_available: bool | None = None
    monthly_calls_available: bool | None = None
    monthly_provider_credits_available: bool | None = None
    cooldown_active: bool
    accounting_scope: Literal["process_lifetime"] = "process_lifetime"

    @classmethod
    def from_snapshot(cls, snapshot: ProviderHealthSnapshot) -> ProviderHealthProjection:
        return cls(
            provider=snapshot.provider,
            status=snapshot.status,
            minute_calls_available=snapshot.remaining_local_calls > 0,
            daily_calls_available=(
                None
                if snapshot.remaining_local_daily_calls is None
                else snapshot.remaining_local_daily_calls > 0
            ),
            monthly_calls_available=(
                None
                if snapshot.remaining_local_monthly_calls is None
                else snapshot.remaining_local_monthly_calls > 0
            ),
            monthly_provider_credits_available=(
                None
                if snapshot.remaining_local_monthly_provider_credits is None
                else snapshot.remaining_local_monthly_provider_credits > 0
            ),
            cooldown_active=snapshot.cooldown_until is not None,
        )
