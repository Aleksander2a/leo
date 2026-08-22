from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.domain.models import (
    Asset,
    AssetKind,
    Membership,
    MembershipRole,
    Organization,
    Portfolio,
    Position,
    Strategy,
    Thesis,
    ThesisVersion,
)
from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError
from leo.persistence.domain_store import InMemoryDomainStore


def _fixture(strategy_id: str, *, with_position: bool) -> dict[str, object]:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    scope = ScopeKey(organization_id="demo-org", strategy_id=strategy_id)
    asset = Asset(id="asset-nvda", symbol="NVDA", kind=AssetKind.EQUITY, display_name="NVIDIA")
    return {
        "scope": scope,
        "organization": Organization(id="demo-org", name="Leo Demo"),
        "strategy": Strategy(
            id=strategy_id,
            organization_id=scope.organization_id,
            name=strategy_id.title(),
            slug=strategy_id,
            description=f"Contradictory fixture for {strategy_id}",
        ),
        "membership": Membership(
            id=f"membership-{strategy_id}",
            organization_id=scope.organization_id,
            actor_id=f"actor-{strategy_id}",
            role=MembershipRole.RESEARCHER,
        ),
        "portfolio": Portfolio(id=f"portfolio-{strategy_id}", scope=scope, name="Demo portfolio"),
        "thesis": Thesis(
            id=f"thesis-{strategy_id}",
            scope=scope,
            subject="NVDA",
            current_version=1,
        ),
        "thesis_version": ThesisVersion(
            id=f"thesis-version-{strategy_id}",
            thesis_id=f"thesis-{strategy_id}",
            number=1,
            summary=(
                "AI infrastructure demand is durable."
                if strategy_id == "technology"
                else "Valuation and concentration risk outweigh the opportunity."
            ),
            created_at=now,
            actor_id=f"actor-{strategy_id}",
            source_ref=f"fixture:{strategy_id}",
        ),
        "assets": (asset,) if with_position else (),
        "positions": (
            Position(
                id=f"position-{strategy_id}",
                portfolio_id=f"portfolio-{strategy_id}",
                asset_id=asset.id,
                quantity=10,
                weight=0.2,
                as_of=now,
                source_ref=f"fixture:{strategy_id}",
            ),
        )
        if with_position
        else (),
    }


@pytest.mark.asyncio
async def test_domain_store_keeps_contradictory_strategy_state_scoped() -> None:
    store = InMemoryDomainStore()
    technology = _fixture("technology", with_position=True)
    conservative = _fixture("conservative", with_position=False)
    await store.seed(**technology)  # type: ignore[arg-type]
    await store.seed(**conservative)  # type: ignore[arg-type]

    tech_scope = ScopeKey(organization_id="demo-org", strategy_id="technology")
    conservative_scope = ScopeKey(organization_id="demo-org", strategy_id="conservative")
    assert (await store.get_thesis(tech_scope)).id == "thesis-technology"
    assert (await store.get_thesis(conservative_scope)).id == "thesis-conservative"
    assert len(await store.list_positions(tech_scope)) == 1
    assert await store.list_positions(conservative_scope) == ()
    with pytest.raises(NotFoundError):
        await store.get_strategy(ScopeKey(organization_id="other-org", strategy_id="technology"))


@pytest.mark.asyncio
async def test_domain_store_uses_optimistic_thesis_version() -> None:
    store = InMemoryDomainStore()
    fixture = _fixture("technology", with_position=False)
    await store.seed(**fixture)  # type: ignore[arg-type]
    scope = ScopeKey(organization_id="demo-org", strategy_id="technology")
    revision = ThesisVersion(
        id="thesis-version-technology-v2",
        thesis_id="thesis-technology",
        number=2,
        summary="Updated synthetic evidence changes the confidence band.",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        actor_id="actor-technology",
        source_ref="fixture:technology:v2",
    )
    updated = await store.append_thesis_version(scope, "thesis-technology", 1, revision)
    assert updated.current_version == 2
    with pytest.raises(ConcurrencyError):
        await store.append_thesis_version(scope, "thesis-technology", 1, revision)
