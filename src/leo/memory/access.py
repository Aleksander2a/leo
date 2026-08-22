"""Explicit, auditable disclosure grants; membership overlap is never authority."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class GrantStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class GrantTarget(ContractModel):
    provider: NonEmptyStr
    team_id: NonEmptyStr
    destination_id: NonEmptyStr


class DisclosureGrant(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    source: GrantTarget
    destination: GrantTarget
    sensitivity_ceiling: float = Field(ge=0, le=1)
    authorizing_actor_id: NonEmptyStr
    authorizing_role: str = Field(pattern=r"^(owner|researcher)$")
    status: GrantStatus = GrantStatus.ACTIVE
    version: int = Field(default=1, ge=1)
    expires_at: datetime | None = None
    reason: NonEmptyStr
    provenance: NonEmptyStr
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_window(self) -> DisclosureGrant:
        for timestamp in (self.created_at, self.updated_at):
            if timestamp.tzinfo is None:
                raise ValueError("grant timestamps must be timezone-aware")
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("grant expiry must be after creation")
        if self.source == self.destination:
            raise ValueError("a disclosure grant must cross distinct destinations")
        return self

    def covers(
        self,
        *,
        source: GrantTarget,
        destination: GrantTarget,
        sensitivity: float,
        now: datetime,
        membership_valid: bool,
    ) -> bool:
        return (
            self.status is GrantStatus.ACTIVE
            and membership_valid
            and now.tzinfo is not None
            and (self.expires_at is None or now < self.expires_at)
            and self.source == source
            and self.destination == destination
            and sensitivity <= self.sensitivity_ceiling
        )


class GrantAudit(ContractModel):
    grant_id: NonEmptyStr
    scope: ScopeKey
    action: str = Field(pattern=r"^(created|revoked|denied)$")
    actor_id: NonEmptyStr
    version: int = Field(ge=1)
    reason: NonEmptyStr
    occurred_at: datetime


class DisclosureDecision(ContractModel):
    allowed: bool
    safe_code: NonEmptyStr
    grant_ids: tuple[str, ...] = ()


class InMemoryDisclosureGrantStore:
    def __init__(self, *, now: Callable[[], datetime]) -> None:
        self._now = now
        self._revisions: dict[str, list[DisclosureGrant]] = {}
        self._audit: list[GrantAudit] = []

    def create(
        self,
        scope: ScopeKey,
        *,
        grant: DisclosureGrant,
        membership_valid: bool,
    ) -> DisclosureGrant:
        _assert_scope(grant.scope, scope)
        if not membership_valid:
            self._record_denial(scope, grant.id, grant.authorizing_actor_id, "membership_invalid")
            raise PermissionError("grant_membership_invalid")
        if grant.id in self._revisions:
            raise ValueError("grant ID already exists")
        self._revisions[grant.id] = [grant]
        self._audit.append(
            GrantAudit(
                grant_id=grant.id,
                scope=scope,
                action="created",
                actor_id=grant.authorizing_actor_id,
                version=grant.version,
                reason=grant.reason,
                occurred_at=grant.created_at,
            )
        )
        return grant

    def revoke(
        self,
        scope: ScopeKey,
        grant_id: str,
        *,
        actor_id: str,
        membership_valid: bool,
        reason: str,
    ) -> DisclosureGrant:
        current = self.current(scope, grant_id)
        if not membership_valid:
            self._record_denial(scope, grant_id, actor_id, "membership_invalid")
            raise PermissionError("grant_membership_invalid")
        now = self._now()
        revoked = current.model_copy(
            update={
                "status": GrantStatus.REVOKED,
                "version": current.version + 1,
                "updated_at": now,
            }
        )
        self._revisions[grant_id].append(revoked)
        self._audit.append(
            GrantAudit(
                grant_id=grant_id,
                scope=scope,
                action="revoked",
                actor_id=actor_id,
                version=revoked.version,
                reason=reason,
                occurred_at=now,
            )
        )
        return revoked

    def current(self, scope: ScopeKey, grant_id: str) -> DisclosureGrant:
        revisions = self._revisions.get(grant_id)
        if not revisions or revisions[-1].scope != scope:
            raise KeyError("grant_not_found")
        return revisions[-1]

    def authorize(
        self,
        scope: ScopeKey,
        *,
        source: GrantTarget,
        destination: GrantTarget,
        sensitivity: float,
        membership_valid: bool,
    ) -> DisclosureDecision:
        if sensitivity < 0 or sensitivity > 1:
            raise ValueError("sensitivity must be between 0 and 1")
        matches = tuple(
            grant
            for revisions in self._revisions.values()
            for grant in (revisions[-1],)
            if grant.scope == scope
            and grant.covers(
                source=source,
                destination=destination,
                sensitivity=sensitivity,
                now=self._now(),
                membership_valid=membership_valid,
            )
        )
        if not matches:
            return DisclosureDecision(allowed=False, safe_code="disclosure_grant_required")
        return DisclosureDecision(
            allowed=True,
            safe_code="explicit_grant",
            grant_ids=tuple(grant.id for grant in matches),
        )

    def audit(self, scope: ScopeKey) -> tuple[GrantAudit, ...]:
        return tuple(item for item in self._audit if item.scope == scope)

    def _record_denial(self, scope: ScopeKey, grant_id: str, actor_id: str, reason: str) -> None:
        self._audit.append(
            GrantAudit(
                grant_id=grant_id,
                scope=scope,
                action="denied",
                actor_id=actor_id,
                version=1,
                reason=reason,
                occurred_at=self._now(),
            )
        )


def intersect_grants(
    grants: Iterable[DisclosureGrant],
    *,
    source: GrantTarget,
    destination: GrantTarget,
    sensitivity: float,
    now: datetime,
    membership_valid: bool,
) -> DisclosureDecision:
    """Derived content is authorized only when every source is covered."""

    selected = tuple(grants)
    if not selected:
        return DisclosureDecision(allowed=False, safe_code="disclosure_grant_required")
    if all(
        grant.covers(
            source=source,
            destination=destination,
            sensitivity=sensitivity,
            now=now,
            membership_valid=membership_valid,
        )
        for grant in selected
    ):
        return DisclosureDecision(
            allowed=True,
            safe_code="grant_intersection",
            grant_ids=tuple(grant.id for grant in selected),
        )
    return DisclosureDecision(allowed=False, safe_code="disclosure_intersection_denied")


def _assert_scope(actual: ScopeKey, expected: ScopeKey) -> None:
    if actual != expected:
        raise PermissionError("scope_mismatch")
