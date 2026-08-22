"""Scope-first in-memory and Postgres repositories for strategy-domain state."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

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
from leo.domain.ports import DomainStore
from leo.harness.models import ScopeKey
from leo.harness.store_errors import ConcurrencyError, NotFoundError, StoreError
from leo.persistence.schema import (
    AssetRow,
    AssumptionRow,
    DecisionRow,
    MandateRow,
    MembershipRow,
    OrganizationRow,
    PortfolioRow,
    PositionRow,
    RiskConstraintRow,
    StrategyRow,
    ThesisRow,
    ThesisVersionRow,
)


class InMemoryDomainStore(DomainStore):
    def __init__(self) -> None:
        self._organizations: dict[str, Organization] = {}
        self._strategies: dict[str, Strategy] = {}
        self._memberships: dict[str, Membership] = {}
        self._assets: dict[str, Asset] = {}
        self._portfolios: dict[str, Portfolio] = {}
        self._positions: dict[str, Position] = {}
        self._mandates: dict[str, Mandate] = {}
        self._theses: dict[str, Thesis] = {}
        self._thesis_versions: dict[str, ThesisVersion] = {}
        self._assumptions: dict[str, Assumption] = {}
        self._decisions: dict[str, Decision] = {}
        self._risk_constraints: dict[str, RiskConstraint] = {}
        self._lock = asyncio.Lock()

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
    ) -> None:
        _validate_seed(
            scope,
            organization,
            strategy,
            membership,
            portfolio,
            thesis,
            thesis_version,
            assets,
            mandate,
            positions,
            assumptions,
            decisions,
            risk_constraints,
        )
        async with self._lock:
            mandate_records = ((mandate.id, mandate),) if mandate is not None else ()
            records: Iterable[tuple[str, object]] = (
                (strategy.id, strategy),
                (membership.id, membership),
                (portfolio.id, portfolio),
                (thesis.id, thesis),
                (thesis_version.id, thesis_version),
                *((asset.id, asset) for asset in assets),
                *((item.id, item) for item in positions),
                *((item.id, item) for item in assumptions),
                *((item.id, item) for item in decisions),
                *((item.id, item) for item in risk_constraints),
                *mandate_records,
            )
            all_ids = [item_id for item_id, _item in records]
            if (
                organization.id in self._organizations
                and self._organizations[organization.id] != organization
            ):
                raise ConcurrencyError("organization identity already exists with different data")
            if len(set(all_ids)) != len(all_ids) or any(
                item_id in self._all_ids() for item_id in all_ids
            ):
                raise ConcurrencyError("domain record already exists")
            self._organizations.setdefault(organization.id, organization)
            self._strategies[strategy.id] = strategy
            self._memberships[membership.id] = membership
            self._portfolios[portfolio.id] = portfolio
            self._theses[thesis.id] = thesis
            self._thesis_versions[thesis_version.id] = thesis_version
            for asset in assets:
                self._assets[asset.id] = asset
            for position in positions:
                self._positions[position.id] = position
            for assumption in assumptions:
                self._assumptions[assumption.id] = assumption
            for decision in decisions:
                self._decisions[decision.id] = decision
            for constraint in risk_constraints:
                self._risk_constraints[constraint.id] = constraint
            if mandate is not None:
                self._mandates[mandate.id] = mandate

    async def get_strategy(self, scope: ScopeKey) -> Strategy:
        async with self._lock:
            for strategy in self._strategies.values():
                if (
                    strategy.organization_id == scope.organization_id
                    and strategy.id == scope.strategy_id
                ):
                    return strategy
        raise NotFoundError("strategy not found")

    async def get_thesis(self, scope: ScopeKey) -> Thesis:
        async with self._lock:
            for thesis in self._theses.values():
                if thesis.scope == scope:
                    return thesis
        raise NotFoundError("thesis not found")

    async def list_positions(self, scope: ScopeKey) -> tuple[Position, ...]:
        async with self._lock:
            portfolio_ids = {
                portfolio.id for portfolio in self._portfolios.values() if portfolio.scope == scope
            }
            return tuple(
                item for item in self._positions.values() if item.portfolio_id in portfolio_ids
            )

    async def append_thesis_version(
        self,
        scope: ScopeKey,
        thesis_id: str,
        expected_version: int,
        revision: ThesisVersion,
    ) -> Thesis:
        async with self._lock:
            current = self._theses.get(thesis_id)
            if current is None or current.scope != scope:
                raise NotFoundError("thesis not found")
            if current.current_version != expected_version:
                raise ConcurrencyError("stale thesis version")
            if revision.thesis_id != thesis_id or revision.number != expected_version + 1:
                raise StoreError("thesis revision does not match the expected version")
            if any(
                item.thesis_id == thesis_id and item.number == revision.number
                for item in self._thesis_versions.values()
            ):
                raise ConcurrencyError("thesis revision already exists")
            self._thesis_versions[revision.id] = revision
            updated = current.model_copy(
                update={"current_version": revision.number, "version": current.version + 1}
            )
            self._theses[thesis_id] = updated
            return updated

    def _all_ids(self) -> frozenset[str]:
        return frozenset(
            item_id
            for records in (
                self._organizations,
                self._strategies,
                self._memberships,
                self._assets,
                self._portfolios,
                self._positions,
                self._mandates,
                self._theses,
                self._thesis_versions,
                self._assumptions,
                self._decisions,
                self._risk_constraints,
            )
            for item_id in records
        )


class PostgresDomainStore(DomainStore):
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

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
    ) -> None:
        _validate_seed(
            scope,
            organization,
            strategy,
            membership,
            portfolio,
            thesis,
            thesis_version,
            assets,
            mandate,
            positions,
            assumptions,
            decisions,
            risk_constraints,
        )
        try:
            async with self._sessions() as session, session.begin():
                await session.execute(
                    postgres_insert(OrganizationRow)
                    .values(
                        id=organization.id,
                        name=organization.name,
                        version=organization.version,
                    )
                    .on_conflict_do_nothing(index_elements=[OrganizationRow.id])
                )
                existing_organization = await session.scalar(
                    select(OrganizationRow).where(OrganizationRow.id == organization.id)
                )
                if existing_organization is None or existing_organization.name != organization.name:
                    raise StoreError("organization identity already exists with different data")
                session.add(_strategy_row(strategy))
                session.add(_membership_row(membership))
                await session.flush()
                for asset in assets:
                    session.add(_asset_row(asset))
                if assets:
                    await session.flush()
                session.add(_portfolio_row(portfolio))
                await session.flush()
                for position in positions:
                    session.add(_position_row(position))
                if mandate is not None:
                    session.add(_mandate_row(mandate))
                session.add(_thesis_row(thesis))
                await session.flush()
                session.add(_thesis_version_row(thesis_version))
                for assumption in assumptions:
                    session.add(_assumption_row(assumption))
                for decision in decisions:
                    session.add(_decision_row(decision))
                for constraint in risk_constraints:
                    session.add(_risk_constraint_row(constraint))
                await session.flush()
        except IntegrityError as exc:
            raise ConcurrencyError("domain record already exists or violates a relation") from exc

    async def get_strategy(self, scope: ScopeKey) -> Strategy:
        async with self._sessions() as session:
            row = await session.scalar(
                select(StrategyRow).where(
                    StrategyRow.organization_id == scope.organization_id,
                    StrategyRow.id == scope.strategy_id,
                )
            )
        if row is None:
            raise NotFoundError("strategy not found")
        return _strategy_model(row)

    async def get_thesis(self, scope: ScopeKey) -> Thesis:
        async with self._sessions() as session:
            row = await session.scalar(
                select(ThesisRow).where(
                    ThesisRow.organization_id == scope.organization_id,
                    ThesisRow.strategy_id == scope.strategy_id,
                )
            )
        if row is None:
            raise NotFoundError("thesis not found")
        return _thesis_model(row)

    async def list_positions(self, scope: ScopeKey) -> tuple[Position, ...]:
        async with self._sessions() as session:
            rows = await session.scalars(
                select(PositionRow)
                .join(PortfolioRow, PortfolioRow.id == PositionRow.portfolio_id)
                .where(
                    PortfolioRow.organization_id == scope.organization_id,
                    PortfolioRow.strategy_id == scope.strategy_id,
                )
                .order_by(PositionRow.as_of, PositionRow.id)
            )
        return tuple(_position_model(row) for row in rows)

    async def append_thesis_version(
        self,
        scope: ScopeKey,
        thesis_id: str,
        expected_version: int,
        revision: ThesisVersion,
    ) -> Thesis:
        if revision.thesis_id != thesis_id or revision.number != expected_version + 1:
            raise StoreError("thesis revision does not match the expected version")
        try:
            async with self._sessions() as session, session.begin():
                thesis = await session.scalar(
                    select(ThesisRow)
                    .where(
                        ThesisRow.id == thesis_id,
                        ThesisRow.organization_id == scope.organization_id,
                        ThesisRow.strategy_id == scope.strategy_id,
                    )
                    .with_for_update()
                )
                if thesis is None:
                    raise NotFoundError("thesis not found")
                if thesis.current_version != expected_version:
                    raise ConcurrencyError("stale thesis version")
                session.add(_thesis_version_row(revision))
                thesis.current_version = revision.number
                thesis.version += 1
                await session.flush()
                return _thesis_model(thesis)
        except IntegrityError as exc:
            raise ConcurrencyError("thesis revision already exists") from exc


def _validate_seed(
    scope: ScopeKey,
    organization: Organization,
    strategy: Strategy,
    membership: Membership,
    portfolio: Portfolio,
    thesis: Thesis,
    thesis_version: ThesisVersion,
    assets: tuple[Asset, ...],
    mandate: Mandate | None,
    positions: tuple[Position, ...],
    assumptions: tuple[Assumption, ...],
    decisions: tuple[Decision, ...],
    risk_constraints: tuple[RiskConstraint, ...],
) -> None:
    expected_scope = ScopeKey(organization_id=organization.id, strategy_id=strategy.id)
    if scope != expected_scope:
        raise StoreError("seed scope does not match the organization and strategy")
    if strategy.organization_id != organization.id or membership.organization_id != organization.id:
        raise StoreError("organization and strategy scope mismatch")
    if portfolio.scope != scope or thesis.scope != scope:
        raise StoreError("domain aggregate scope mismatch")
    if thesis_version.thesis_id != thesis.id or thesis_version.number != 1:
        raise StoreError("initial thesis version mismatch")
    if mandate is not None and mandate.scope != scope:
        raise StoreError("mandate scope mismatch")
    if any(item.portfolio_id != portfolio.id for item in positions):
        raise StoreError("position portfolio mismatch")
    asset_ids = {item.id for item in assets}
    if any(item.asset_id not in asset_ids for item in positions):
        raise StoreError("position asset is not part of the seed")
    if any(item.thesis_version_id != thesis_version.id for item in assumptions):
        raise StoreError("assumption thesis mismatch")
    if any(item.scope != scope for item in decisions + risk_constraints):
        raise StoreError("decision or risk scope mismatch")


def _organization_row(item: Organization) -> OrganizationRow:
    return OrganizationRow(id=item.id, name=item.name, version=item.version)


def _strategy_row(item: Strategy) -> StrategyRow:
    return StrategyRow(
        id=item.id,
        organization_id=item.organization_id,
        name=item.name,
        slug=item.slug,
        description=item.description,
        version=item.version,
    )


def _membership_row(item: Membership) -> MembershipRow:
    return MembershipRow(
        id=item.id,
        organization_id=item.organization_id,
        actor_id=item.actor_id,
        role=item.role.value,
        status=item.status.value,
        version=item.version,
    )


def _asset_row(item: Asset) -> AssetRow:
    return AssetRow(
        id=item.id, symbol=item.symbol, kind=item.kind.value, display_name=item.display_name
    )


def _portfolio_row(item: Portfolio) -> PortfolioRow:
    return PortfolioRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        name=item.name,
        base_currency=item.base_currency,
        version=item.version,
    )


def _position_row(item: Position) -> PositionRow:
    return PositionRow(
        id=item.id,
        portfolio_id=item.portfolio_id,
        asset_id=item.asset_id,
        quantity=item.quantity,
        weight=item.weight,
        as_of=item.as_of,
        source_ref=item.source_ref,
        version=item.version,
    )


def _mandate_row(item: Mandate) -> MandateRow:
    return MandateRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        statement=item.statement,
        target_weight=item.target_weight,
        effective_at=item.effective_at,
        expires_at=item.expires_at,
        actor_id=item.actor_id,
        source_ref=item.source_ref,
        version=item.version,
    )


def _thesis_row(item: Thesis) -> ThesisRow:
    return ThesisRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        subject=item.subject,
        current_version=item.current_version,
        status=item.status.value,
        version=item.version,
    )


def _thesis_version_row(item: ThesisVersion) -> ThesisVersionRow:
    return ThesisVersionRow(
        id=item.id,
        thesis_id=item.thesis_id,
        number=item.number,
        summary=item.summary,
        status=item.status.value,
        created_at=item.created_at,
        actor_id=item.actor_id,
        source_ref=item.source_ref,
    )


def _assumption_row(item: Assumption) -> AssumptionRow:
    return AssumptionRow(
        id=item.id,
        thesis_version_id=item.thesis_version_id,
        statement=item.statement,
        status=item.status.value,
        version=item.version,
    )


def _decision_row(item: Decision) -> DecisionRow:
    return DecisionRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        thesis_version_id=item.thesis_version_id,
        kind=item.kind.value,
        statement=item.statement,
        decided_at=item.decided_at,
        actor_id=item.actor_id,
        source_ref=item.source_ref,
        version=item.version,
    )


def _risk_constraint_row(item: RiskConstraint) -> RiskConstraintRow:
    return RiskConstraintRow(
        id=item.id,
        organization_id=item.scope.organization_id,
        strategy_id=item.scope.strategy_id,
        kind=item.kind.value,
        value=item.value,
        effective_at=item.effective_at,
        expires_at=item.expires_at,
        actor_id=item.actor_id,
        source_ref=item.source_ref,
        version=item.version,
    )


def _strategy_model(row: StrategyRow) -> Strategy:
    return Strategy(
        id=row.id,
        organization_id=row.organization_id,
        name=row.name,
        slug=row.slug,
        description=row.description,
        version=row.version,
    )


def _thesis_model(row: ThesisRow) -> Thesis:
    return Thesis(
        id=row.id,
        scope=ScopeKey(organization_id=row.organization_id, strategy_id=row.strategy_id),
        subject=row.subject,
        current_version=row.current_version,
        status=row.status,
        version=row.version,
    )


def _position_model(row: PositionRow) -> Position:
    return Position(
        id=row.id,
        portfolio_id=row.portfolio_id,
        asset_id=row.asset_id,
        quantity=row.quantity,
        weight=row.weight,
        as_of=row.as_of,
        source_ref=row.source_ref,
        version=row.version,
    )
