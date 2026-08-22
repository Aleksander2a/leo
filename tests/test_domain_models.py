from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from leo.domain.models import (
    Asset,
    AssetKind,
    Mandate,
    Position,
    Strategy,
    Thesis,
    ThesisVersion,
)
from leo.harness.models import ScopeKey


def test_domain_contracts_reject_invalid_temporal_and_weight_values() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ValidationError):
        Mandate(
            id="mandate-invalid",
            scope=ScopeKey(organization_id="org", strategy_id="strategy"),
            statement="keep risk bounded",
            target_weight=1.1,
            effective_at=now,
        )
    with pytest.raises(ValidationError):
        Mandate(
            id="mandate-invalid-window",
            scope=ScopeKey(organization_id="org", strategy_id="strategy"),
            statement="keep risk bounded",
            effective_at=now,
            expires_at=now,
        )
    with pytest.raises(ValidationError):
        Position(
            id="position-invalid",
            portfolio_id="portfolio",
            asset_id="asset",
            quantity=1,
            weight=-0.1,
            as_of=now,
            source_ref="fixture",
        )


def test_domain_contracts_are_immutable_and_scope_explicit() -> None:
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    strategy = Strategy(
        id=scope.strategy_id,
        organization_id=scope.organization_id,
        name="Technology",
        slug="technology",
    )
    assert strategy.organization_id == scope.organization_id
    with pytest.raises(ValidationError):
        strategy.name = "mutated"  # type: ignore[misc]


def test_asset_and_thesis_fixture_fields_are_typed() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    assert Asset(id="asset-nvda", symbol="NVDA", kind=AssetKind.EQUITY, display_name="NVIDIA")
    thesis = Thesis(
        id="thesis-tech",
        scope=ScopeKey(organization_id="org", strategy_id="technology"),
        subject="NVDA",
        current_version=1,
    )
    version = ThesisVersion(
        id="thesis-tech-v1",
        thesis_id=thesis.id,
        number=1,
        summary="AI infrastructure demand remains durable.",
        created_at=now,
        actor_id="fixture-owner",
        source_ref="fixture:technology",
    )
    assert version.created_at + timedelta(days=1) > now
