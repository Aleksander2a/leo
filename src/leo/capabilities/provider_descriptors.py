"""Reusable provider-family metadata kept outside any one composition root."""

from __future__ import annotations

from pydantic import Field

from leo.capabilities.catalog import (
    CapabilityLatency,
    CapabilitySensitivity,
)
from leo.harness.models import ContractModel, NonEmptyStr
from leo.harness.provider_health import ProviderName


class ProviderCapabilityDescriptor(ContractModel):
    provider: NonEmptyStr
    runtime_health_providers: frozenset[ProviderName] = Field(default_factory=frozenset)
    tags: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    sensitivity: CapabilitySensitivity = CapabilitySensitivity.PUBLIC
    freshness_seconds: int | None = Field(default=None, ge=0)
    rate_limit_per_minute: int | None = Field(default=None, ge=1)
    latency: CapabilityLatency = CapabilityLatency.MEDIUM
    verification_expectations: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
