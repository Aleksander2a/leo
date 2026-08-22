from __future__ import annotations

import inspect

import pytest

from leo.domain.ports import DomainStore
from leo.harness.models import ScopeKey
from leo.harness.store_errors import NotFoundError, StoreError
from leo.persistence.domain_store import InMemoryDomainStore
from tests.test_domain_store import _fixture


def test_domain_port_has_no_unscoped_operation() -> None:
    expected_methods = (
        "seed",
        "get_strategy",
        "get_thesis",
        "list_positions",
        "append_thesis_version",
    )
    for name in expected_methods:
        parameters = inspect.signature(getattr(DomainStore, name)).parameters
        assert "scope" in parameters


@pytest.mark.asyncio
async def test_wrong_scope_is_not_an_existence_probe() -> None:
    store = InMemoryDomainStore()
    fixture = _fixture("technology", with_position=True)
    await store.seed(**fixture)  # type: ignore[arg-type]
    wrong_org = ScopeKey(organization_id="other-org", strategy_id="technology")
    wrong_strategy = ScopeKey(organization_id="demo-org", strategy_id="conservative")

    with pytest.raises(NotFoundError):
        await store.get_strategy(wrong_org)
    with pytest.raises(NotFoundError):
        await store.get_thesis(wrong_org)
    assert await store.list_positions(wrong_org) == ()
    assert await store.list_positions(wrong_strategy) == ()
    with pytest.raises(NotFoundError):
        await store.append_thesis_version(
            wrong_org, "thesis-technology", 1, fixture["thesis_version"]
        )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_seed_scope_is_explicit_and_relationship_checked() -> None:
    store = InMemoryDomainStore()
    fixture = _fixture("technology", with_position=False)
    with pytest.raises(StoreError, match="seed scope"):
        await store.seed(
            ScopeKey(organization_id="other-org", strategy_id="technology"),
            **{key: value for key, value in fixture.items() if key != "scope"},
        )  # type: ignore[arg-type]
