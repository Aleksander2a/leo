"""Immutable strategy-domain state owned by Leo's harness."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class MembershipRole(StrEnum):
    OWNER = "owner"
    RESEARCHER = "researcher"
    VIEWER = "viewer"


class MembershipStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class AssetKind(StrEnum):
    EQUITY = "equity"
    ETF = "etf"
    CRYPTO = "crypto"
    CASH = "cash"


class ThesisStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CLOSED = "closed"


class DecisionKind(StrEnum):
    INVEST = "invest"
    AVOID = "avoid"
    REVIEW = "review"


class RiskConstraintKind(StrEnum):
    MAX_POSITION_WEIGHT = "max_position_weight"
    MAX_DRAWDOWN = "max_drawdown"
    MIN_CASH_WEIGHT = "min_cash_weight"


class Organization(ContractModel):
    id: NonEmptyStr
    name: NonEmptyStr
    version: int = Field(default=1, ge=1)


class Strategy(ContractModel):
    id: NonEmptyStr
    organization_id: NonEmptyStr
    name: NonEmptyStr
    slug: NonEmptyStr
    description: str = ""
    version: int = Field(default=1, ge=1)


class Membership(ContractModel):
    id: NonEmptyStr
    organization_id: NonEmptyStr
    actor_id: NonEmptyStr
    role: MembershipRole
    status: MembershipStatus = MembershipStatus.ACTIVE
    version: int = Field(default=1, ge=1)


class Asset(ContractModel):
    id: NonEmptyStr
    symbol: NonEmptyStr
    kind: AssetKind
    display_name: NonEmptyStr


class Portfolio(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    name: NonEmptyStr
    base_currency: NonEmptyStr = "USD"
    version: int = Field(default=1, ge=1)


class Position(ContractModel):
    id: NonEmptyStr
    portfolio_id: NonEmptyStr
    asset_id: NonEmptyStr
    quantity: float = Field(ge=0)
    weight: float = Field(ge=0, le=1)
    as_of: datetime
    source_ref: NonEmptyStr
    version: int = Field(default=1, ge=1)


class Mandate(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    statement: NonEmptyStr
    target_weight: float | None = Field(default=None, ge=0, le=1)
    effective_at: datetime
    expires_at: datetime | None = None
    actor_id: NonEmptyStr
    source_ref: NonEmptyStr
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_window(self) -> Mandate:
        if self.expires_at is not None and self.expires_at <= self.effective_at:
            raise ValueError("mandate expiry must be after effective time")
        return self


class Thesis(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    subject: NonEmptyStr
    current_version: int = Field(default=0, ge=0)
    status: ThesisStatus = ThesisStatus.ACTIVE
    version: int = Field(default=1, ge=1)


class ThesisVersion(ContractModel):
    id: NonEmptyStr
    thesis_id: NonEmptyStr
    number: int = Field(ge=1)
    summary: NonEmptyStr
    status: ThesisStatus = ThesisStatus.ACTIVE
    created_at: datetime
    actor_id: NonEmptyStr
    source_ref: NonEmptyStr


class Assumption(ContractModel):
    id: NonEmptyStr
    thesis_version_id: NonEmptyStr
    statement: NonEmptyStr
    status: ThesisStatus = ThesisStatus.ACTIVE
    version: int = Field(default=1, ge=1)


class Decision(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    thesis_version_id: str | None = None
    kind: DecisionKind
    statement: NonEmptyStr
    decided_at: datetime
    actor_id: NonEmptyStr
    source_ref: NonEmptyStr
    version: int = Field(default=1, ge=1)


class RiskConstraint(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    kind: RiskConstraintKind
    value: float = Field(ge=0, le=1)
    effective_at: datetime
    expires_at: datetime | None = None
    actor_id: NonEmptyStr
    source_ref: NonEmptyStr
    version: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def valid_window(self) -> RiskConstraint:
        if self.expires_at is not None and self.expires_at <= self.effective_at:
            raise ValueError("risk constraint expiry must be after effective time")
        return self
