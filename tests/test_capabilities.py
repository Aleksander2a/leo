from __future__ import annotations

import pytest

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import (
    CapabilityHealth,
    CapabilityLatency,
    CapabilitySensitivity,
    CatalogTool,
    InMemoryToolCatalog,
    ToolCatalogError,
)
from leo.capabilities.discovery import DiscoveryBroker, DiscoveryQuery
from leo.harness.models import RunPhase, ScopeKey, ToolEffect, ToolSpec


def _tool(name: str = "market.quote", *, effect: ToolEffect = ToolEffect.READ) -> CatalogTool:
    spec = ToolSpec(
        name=name,
        description="Read a synthetic market quote.",
        domain="market",
        input_schema={"type": "object", "properties": {"symbol": {"type": "string"}}},
        effect=effect,
        allowed_phases=frozenset({RunPhase.RESEARCH}),
    )
    return CatalogTool(
        id=name,
        semantic_version="1.0.0",
        provider="fake",
        spec=spec,
        short_description="Read market quote",
        tags=frozenset({"market", "quote", "read"}),
        authorized_roles=frozenset({"researcher"}),
    )


def test_policy_filter_runs_before_discovery_and_describe() -> None:
    catalog = InMemoryToolCatalog()
    catalog.register(_tool())
    catalog.register(_tool("market.write", effect=ToolEffect.WRITE))
    catalog.register(
        _tool("market.down", effect=ToolEffect.READ).model_copy(
            update={"health": CapabilityHealth.UNHEALTHY}
        )
    )
    broker = DiscoveryBroker(catalog)
    summaries = broker.search(
        DiscoveryQuery(query="market quote"),
        phase=RunPhase.RESEARCH,
        profile="research",
        role="researcher",
        remaining_cost=1,
    )
    assert [summary.id for summary in summaries] == ["market.quote"]
    assert [
        record.id
        for record in broker.describe(
            ("market.quote",),
            phase=RunPhase.RESEARCH,
            profile="research",
            role="researcher",
            remaining_cost=1,
        )
    ] == ["market.quote"]
    with pytest.raises(ToolCatalogError, match="not_eligible"):
        broker.describe(
            ("market.write",),
            phase=RunPhase.RESEARCH,
            profile="research",
            role="researcher",
            remaining_cost=1,
        )


def test_complete_catalog_metadata_and_namespace_conversation_policy_fail_closed() -> None:
    record = _tool().model_copy(
        update={
            "long_description": "Provider-neutral long description.",
            "capability_tags": frozenset({"retrieval"}),
            "entity_tags": frozenset({"equity"}),
            "allowed_namespaces": frozenset({"org/domain"}),
            "allowed_conversation_kinds": frozenset({"channel"}),
            "sensitivity": CapabilitySensitivity.INTERNAL,
            "latency": CapabilityLatency.LOW,
            "observation_kind": "market.get_quote",
            "normalization_version": "normalization-v1",
            "freshness_seconds": 60,
            "rate_limit_per_minute": 30,
            "verification_expectations": frozenset({"exact_numeric_grounding"}),
        }
    )
    catalog = InMemoryToolCatalog(version="catalog-v4")
    catalog.register(record)

    assert (
        catalog.eligible(
            phase=RunPhase.RESEARCH,
            profile="research",
            role="researcher",
            remaining_cost=1,
        )
        == ()
    )
    namespace = ScopeKey(organization_id="org", strategy_id="domain")
    assert catalog.eligible(
        phase=RunPhase.RESEARCH,
        profile="research",
        role="researcher",
        remaining_cost=1,
        namespace=namespace,
        conversation_kind="channel",
    ) == (record,)
    denied = catalog.eligibility(
        record,
        phase=RunPhase.RESEARCH,
        profile="research",
        role="researcher",
        remaining_cost=1,
        namespace=namespace,
        conversation_kind="dm",
    )
    assert not denied.eligible
    assert denied.reason == "conversation_not_allowed"
    assert record.schema_fingerprint == catalog.get(record.id).schema_fingerprint


def test_native_projection_populates_observation_and_verification_metadata() -> None:
    source = _tool()
    record = catalog_tool_from_spec(
        source.spec,
        provider="fixture",
        tags=frozenset({"market", "quote"}),
        sensitivity=CapabilitySensitivity.INTERNAL,
        freshness_seconds=60,
        rate_limit_per_minute=30,
        latency=CapabilityLatency.LOW,
        verification_expectations=frozenset({"exact_numeric_grounding"}),
    )

    assert record.long_description == source.spec.description
    assert record.capability_tags == frozenset({"market", "quote"})
    assert record.observation_kind == source.spec.name
    assert record.normalization_version == "normalization-v1"
    assert record.sensitivity is CapabilitySensitivity.INTERNAL
    assert record.freshness_seconds == 60
    assert record.rate_limit_per_minute == 30
    assert record.latency is CapabilityLatency.LOW
    assert record.verification_expectations == frozenset({"exact_numeric_grounding"})
