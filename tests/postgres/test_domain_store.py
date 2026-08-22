from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio

from leo.config import Settings
from leo.domain.models import (
    Membership,
    MembershipRole,
    Organization,
    Portfolio,
    Strategy,
    Thesis,
    ThesisVersion,
)
from leo.harness.models import ScopeKey
from leo.harness.store_errors import NotFoundError
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.domain_store import PostgresDomainStore


@pytest_asyncio.fixture
async def domain_store(postgres_store: object) -> PostgresDomainStore:
    del postgres_store
    database_url = Settings().database_url
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(database_url.get_secret_value())
    try:
        yield PostgresDomainStore(create_session_factory(engine))
    finally:
        await engine.dispose()


def _records(strategy_id: str) -> tuple[Organization, Strategy, Membership, Thesis, ThesisVersion]:
    scope = ScopeKey(organization_id="domain-org", strategy_id=strategy_id)
    organization = Organization(id=scope.organization_id, name="Domain Demo")
    strategy = Strategy(
        id=scope.strategy_id,
        organization_id=scope.organization_id,
        name=strategy_id.title(),
        slug=strategy_id,
    )
    membership = Membership(
        id=f"membership-{strategy_id}",
        organization_id=scope.organization_id,
        actor_id=f"actor-{strategy_id}",
        role=MembershipRole.RESEARCHER,
    )
    thesis = Thesis(id=f"thesis-{strategy_id}", scope=scope, subject="NVDA", current_version=1)
    revision = ThesisVersion(
        id=f"revision-{strategy_id}-1",
        thesis_id=thesis.id,
        number=1,
        summary=f"Synthetic {strategy_id} thesis.",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        actor_id=membership.actor_id,
        source_ref=f"fixture:{strategy_id}",
    )
    return organization, strategy, membership, thesis, revision


@pytest.mark.asyncio
async def test_postgres_domain_round_trip_and_scope_denial(
    domain_store: PostgresDomainStore,
) -> None:
    organization, strategy, membership, thesis, revision = _records("technology")
    await domain_store.seed(
        ScopeKey(organization_id=organization.id, strategy_id=strategy.id),
        organization,
        strategy,
        membership,
        _portfolio(strategy.id),
        thesis,
        revision,
    )

    scope = ScopeKey(organization_id="domain-org", strategy_id="technology")
    assert (await domain_store.get_strategy(scope)).id == strategy.id
    assert (await domain_store.get_thesis(scope)).current_version == 1
    with pytest.raises(NotFoundError):
        await domain_store.get_strategy(
            ScopeKey(organization_id="domain-org", strategy_id="conservative")
        )


@pytest.mark.asyncio
async def test_postgres_domain_can_seed_same_org_with_contradictory_strategy(
    domain_store: PostgresDomainStore,
) -> None:
    first = _records("technology")
    second = _records("conservative")
    await domain_store.seed(
        ScopeKey(organization_id=first[0].id, strategy_id=first[1].id),
        *first[:3],
        _portfolio(first[1].id),
        first[3],
        first[4],
    )
    await domain_store.seed(
        ScopeKey(organization_id=second[0].id, strategy_id=second[1].id),
        *second[:3],
        _portfolio(second[1].id),
        second[3],
        second[4],
    )
    assert (
        await domain_store.get_thesis(
            ScopeKey(organization_id="domain-org", strategy_id="conservative")
        )
    ).id == "thesis-conservative"


def _portfolio(strategy_id: str) -> Portfolio:
    return Portfolio(
        id=f"portfolio-{strategy_id}",
        scope=ScopeKey(organization_id="domain-org", strategy_id=strategy_id),
        name="Demo portfolio",
    )
