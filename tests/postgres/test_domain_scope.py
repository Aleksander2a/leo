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
from leo.harness.store_errors import ConcurrencyError, NotFoundError
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.domain_store import PostgresDomainStore


def _records(strategy_id: str) -> tuple[Organization, Strategy, Membership, Thesis, ThesisVersion]:
    scope = ScopeKey(organization_id="domain-scope-org", strategy_id=strategy_id)
    organization = Organization(id=scope.organization_id, name="Domain Scope Demo")
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


def _portfolio(strategy_id: str) -> Portfolio:
    return Portfolio(
        id=f"portfolio-{strategy_id}",
        scope=ScopeKey(organization_id="domain-scope-org", strategy_id=strategy_id),
        name="Demo portfolio",
    )


@pytest_asyncio.fixture
async def scoped_domain_store(postgres_store: object) -> PostgresDomainStore:
    del postgres_store
    database_url = Settings().database_url
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured")
    engine = create_database_engine(database_url.get_secret_value())
    try:
        yield PostgresDomainStore(create_session_factory(engine))
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_postgres_domain_wrong_scope_returns_not_found_or_empty(
    scoped_domain_store: PostgresDomainStore,
) -> None:
    organization, strategy, membership, thesis, revision = _records("technology")
    scope = ScopeKey(organization_id=organization.id, strategy_id=strategy.id)
    await scoped_domain_store.seed(
        scope, organization, strategy, membership, _portfolio(strategy.id), thesis, revision
    )

    wrong_org = ScopeKey(organization_id="other-org", strategy_id=strategy.id)
    wrong_strategy = ScopeKey(organization_id=organization.id, strategy_id="conservative")
    with pytest.raises(NotFoundError):
        await scoped_domain_store.get_strategy(wrong_org)
    with pytest.raises(NotFoundError):
        await scoped_domain_store.get_thesis(wrong_org)
    assert await scoped_domain_store.list_positions(wrong_org) == ()
    assert await scoped_domain_store.list_positions(wrong_strategy) == ()
    with pytest.raises(NotFoundError):
        await scoped_domain_store.append_thesis_version(
            wrong_org,
            thesis.id,
            1,
            ThesisVersion(
                id="foreign-revision",
                thesis_id=thesis.id,
                number=2,
                summary="Foreign scope must not append.",
                created_at=datetime(2026, 1, 2, tzinfo=UTC),
                actor_id=membership.actor_id,
                source_ref="fixture:foreign",
            ),
        )


@pytest.mark.asyncio
async def test_postgres_domain_scope_cas_rejects_stale_revision(
    scoped_domain_store: PostgresDomainStore,
) -> None:
    organization, strategy, membership, thesis, revision = _records("technology")
    scope = ScopeKey(organization_id=organization.id, strategy_id=strategy.id)
    await scoped_domain_store.seed(
        scope, organization, strategy, membership, _portfolio(strategy.id), thesis, revision
    )
    candidate = ThesisVersion(
        id="technology-revision-2",
        thesis_id=thesis.id,
        number=2,
        summary="Synthetic revised thesis.",
        created_at=datetime(2026, 1, 2, tzinfo=UTC),
        actor_id=membership.actor_id,
        source_ref="fixture:technology:v2",
    )
    await scoped_domain_store.append_thesis_version(scope, thesis.id, 1, candidate)
    with pytest.raises(ConcurrencyError):
        await scoped_domain_store.append_thesis_version(scope, thesis.id, 1, candidate)
