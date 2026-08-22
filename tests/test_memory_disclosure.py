from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.harness.models import ScopeKey
from leo.memory.access import (
    DisclosureGrant,
    GrantTarget,
    InMemoryDisclosureGrantStore,
    intersect_grants,
)

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
SOURCE = GrantTarget(provider="slack", team_id="team", destination_id="channel-a")
DESTINATION = GrantTarget(provider="slack", team_id="team", destination_id="channel-b")
NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _grant(grant_id: str = "grant-1", *, expires_at: datetime | None = None) -> DisclosureGrant:
    return DisclosureGrant(
        id=grant_id,
        scope=SCOPE,
        source=SOURCE,
        destination=DESTINATION,
        sensitivity_ceiling=0.7,
        authorizing_actor_id="owner",
        authorizing_role="owner",
        expires_at=expires_at,
        reason="synthetic demo promotion",
        provenance="operator:demo",
        created_at=NOW,
        updated_at=NOW,
    )


def test_membership_overlap_without_explicit_grant_does_not_transfer_memory() -> None:
    store = InMemoryDisclosureGrantStore(now=lambda: NOW)
    assert (
        store.authorize(
            SCOPE,
            source=SOURCE,
            destination=DESTINATION,
            sensitivity=0.1,
            membership_valid=True,
        ).allowed
        is False
    )


def test_explicit_grant_allows_only_covered_destination_and_revocation_is_immediate() -> None:
    store = InMemoryDisclosureGrantStore(now=lambda: NOW)
    store.create(SCOPE, grant=_grant(), membership_valid=True)
    allowed = store.authorize(
        SCOPE,
        source=SOURCE,
        destination=DESTINATION,
        sensitivity=0.5,
        membership_valid=True,
    )
    assert allowed.allowed is True
    assert (
        store.authorize(
            SCOPE,
            source=SOURCE,
            destination=GrantTarget(provider="slack", team_id="team", destination_id="channel-c"),
            sensitivity=0.5,
            membership_valid=True,
        ).allowed
        is False
    )
    store.revoke(
        SCOPE,
        "grant-1",
        actor_id="owner",
        membership_valid=True,
        reason="demo revoke",
    )
    assert (
        store.authorize(
            SCOPE,
            source=SOURCE,
            destination=DESTINATION,
            sensitivity=0.5,
            membership_valid=True,
        ).allowed
        is False
    )


def test_expiry_membership_and_grant_intersection_fail_closed() -> None:
    expired = _grant(expires_at=NOW + timedelta(seconds=1))
    store = InMemoryDisclosureGrantStore(now=lambda: NOW + timedelta(seconds=2))
    store.create(SCOPE, grant=expired, membership_valid=True)
    assert (
        store.authorize(
            SCOPE,
            source=SOURCE,
            destination=DESTINATION,
            sensitivity=0.1,
            membership_valid=True,
        ).allowed
        is False
    )
    with pytest.raises(PermissionError, match="grant_membership_invalid"):
        InMemoryDisclosureGrantStore(now=lambda: NOW).create(
            SCOPE, grant=_grant("grant-denied"), membership_valid=False
        )
    assert (
        intersect_grants(
            [_grant()],
            source=SOURCE,
            destination=DESTINATION,
            sensitivity=0.8,
            now=NOW,
            membership_valid=True,
        ).allowed
        is False
    )
