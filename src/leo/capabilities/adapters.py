"""Compatibility projections from native tool specs into the versioned catalog."""

from leo.capabilities.catalog import (
    CapabilityHealth,
    CapabilityLatency,
    CapabilitySensitivity,
    CatalogTool,
)
from leo.harness.models import ToolSpec


def catalog_tool_from_spec(
    spec: ToolSpec,
    *,
    provider: str,
    tags: frozenset[str] = frozenset(),
    profiles: frozenset[str] = frozenset({"research"}),
    health: CapabilityHealth = CapabilityHealth.HEALTHY,
    sensitivity: CapabilitySensitivity = CapabilitySensitivity.PUBLIC,
    freshness_seconds: int | None = None,
    rate_limit_per_minute: int | None = None,
    latency: CapabilityLatency = CapabilityLatency.MEDIUM,
    observation_kind: str | None = None,
    verification_expectations: frozenset[str] = frozenset(),
) -> CatalogTool:
    parts = spec.version.split(".")
    if not 1 <= len(parts) <= 3 or any(not part.isdigit() for part in parts):
        raise ValueError("tool version must contain one to three numeric components")
    version = ".".join((*parts, *("0" for _ in range(3 - len(parts)))))
    return CatalogTool(
        id=spec.name,
        semantic_version=version,
        provider=provider,
        spec=spec,
        short_description=spec.description[:240],
        long_description=spec.description,
        tags=tags,
        capability_tags=tags,
        profiles=profiles,
        health=health,
        sensitivity=sensitivity,
        freshness_seconds=freshness_seconds,
        rate_limit_per_minute=rate_limit_per_minute,
        latency=latency,
        observation_kind=observation_kind or spec.name,
        verification_expectations=verification_expectations,
    )
