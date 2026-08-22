"""Scope-first repository ports for normalized strategy state."""

from __future__ import annotations

from typing import Protocol

from leo.domain.models import (
    Asset,
    Assumption,
    Decision,
    Mandate,
    Membership,
    Organization,
    Portfolio,
    Position,
    RiskConstraint,
    Strategy,
    Thesis,
    ThesisVersion,
)
from leo.harness.models import ScopeKey


class DomainStore(Protocol):
    async def seed(
        self,
        scope: ScopeKey,
        organization: Organization,
        strategy: Strategy,
        membership: Membership,
        portfolio: Portfolio,
        thesis: Thesis,
        thesis_version: ThesisVersion,
        *,
        assets: tuple[Asset, ...] = (),
        mandate: Mandate | None = None,
        positions: tuple[Position, ...] = (),
        assumptions: tuple[Assumption, ...] = (),
        decisions: tuple[Decision, ...] = (),
        risk_constraints: tuple[RiskConstraint, ...] = (),
    ) -> None: ...

    async def get_strategy(self, scope: ScopeKey) -> Strategy: ...

    async def get_thesis(self, scope: ScopeKey) -> Thesis: ...

    async def list_positions(self, scope: ScopeKey) -> tuple[Position, ...]: ...

    async def append_thesis_version(
        self,
        scope: ScopeKey,
        thesis_id: str,
        expected_version: int,
        revision: ThesisVersion,
    ) -> Thesis: ...
