"""Add normalized, scoped strategy-domain state."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0009"
down_revision: str | None = "20260821_0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = (
    "organizations",
    "strategies",
    "organization_memberships",
    "assets",
    "portfolios",
    "positions",
    "mandates",
    "theses",
    "thesis_versions",
    "thesis_assumptions",
    "strategy_decisions",
    "risk_constraints",
)


def _protect(table: str) -> None:
    op.execute(sa.text(f"REVOKE ALL PRIVILEGES ON TABLE public.{table} FROM anon, authenticated"))
    op.execute(sa.text(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            f"CREATE POLICY leo_client_deny ON public.{table} AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def upgrade() -> None:
    op.create_table(
        "organizations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint("version >= 1", name="ck_organizations_version"),
    )
    op.create_table(
        "strategies",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("organization_id", "slug", name="uq_strategies_org_slug"),
        sa.CheckConstraint("version >= 1", name="ck_strategies_version"),
    )
    op.create_table(
        "organization_memberships",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("role", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint("organization_id", "actor_id", name="uq_membership_org_actor"),
        sa.CheckConstraint("role IN ('owner', 'researcher', 'viewer')", name="ck_membership_role"),
        sa.CheckConstraint("status IN ('active', 'revoked')", name="ck_membership_status"),
        sa.CheckConstraint("version >= 1", name="ck_membership_version"),
    )
    op.create_table(
        "assets",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("symbol", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("display_name", sa.String(255), nullable=False),
        sa.UniqueConstraint("symbol", name="uq_assets_symbol"),
    )
    op.create_table(
        "portfolios",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("base_currency", sa.String(16), nullable=False, server_default="USD"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "organization_id", "strategy_id", "name", name="uq_portfolio_scope_name"
        ),
        sa.CheckConstraint("version >= 1", name="ck_portfolios_version"),
    )
    op.create_table(
        "positions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("portfolio_id", sa.String(64), sa.ForeignKey("portfolios.id"), nullable=False),
        sa.Column("asset_id", sa.String(64), sa.ForeignKey("assets.id"), nullable=False),
        sa.Column("quantity", sa.Float, nullable=False),
        sa.Column("weight", sa.Float, nullable=False),
        sa.Column("as_of", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint("quantity >= 0", name="ck_positions_quantity"),
        sa.CheckConstraint("weight >= 0 AND weight <= 1", name="ck_positions_weight"),
        sa.CheckConstraint("version >= 1", name="ck_positions_version"),
    )
    op.create_table(
        "mandates",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("statement", sa.Text, nullable=False),
        sa.Column("target_weight", sa.Float),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint(
            "target_weight IS NULL OR (target_weight >= 0 AND target_weight <= 1)",
            name="ck_mandates_target_weight",
        ),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > effective_at", name="ck_mandates_window"
        ),
        sa.CheckConstraint("version >= 1", name="ck_mandates_version"),
    )
    op.create_table(
        "theses",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("subject", sa.String(255), nullable=False),
        sa.Column("current_version", sa.Integer, nullable=False, server_default="0"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.UniqueConstraint(
            "organization_id", "strategy_id", "subject", name="uq_theses_scope_subject"
        ),
        sa.CheckConstraint("status IN ('active', 'superseded', 'closed')", name="ck_theses_status"),
        sa.CheckConstraint("current_version >= 0", name="ck_theses_current_version"),
        sa.CheckConstraint("version >= 1", name="ck_theses_version"),
    )
    op.create_table(
        "thesis_versions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("thesis_id", sa.String(64), sa.ForeignKey("theses.id"), nullable=False),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column("summary", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=False),
        sa.UniqueConstraint("thesis_id", "number", name="uq_thesis_versions_number"),
        sa.CheckConstraint("number >= 1", name="ck_thesis_versions_number"),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'closed')", name="ck_thesis_versions_status"
        ),
    )
    op.create_table(
        "thesis_assumptions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "thesis_version_id", sa.String(64), sa.ForeignKey("thesis_versions.id"), nullable=False
        ),
        sa.Column("statement", sa.Text, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint(
            "status IN ('active', 'superseded', 'closed')", name="ck_assumptions_status"
        ),
        sa.CheckConstraint("version >= 1", name="ck_assumptions_version"),
    )
    op.create_table(
        "strategy_decisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("thesis_version_id", sa.String(64), sa.ForeignKey("thesis_versions.id")),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("statement", sa.Text, nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint("kind IN ('invest', 'avoid', 'review')", name="ck_decisions_kind"),
        sa.CheckConstraint("version >= 1", name="ck_decisions_version"),
    )
    op.create_table(
        "risk_constraints",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False),
        sa.Column("value", sa.Float, nullable=False),
        sa.Column("effective_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("source_ref", sa.String(255), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.CheckConstraint(
            "kind IN ('max_position_weight', 'max_drawdown', 'min_cash_weight')",
            name="ck_risk_constraints_kind",
        ),
        sa.CheckConstraint("value >= 0 AND value <= 1", name="ck_risk_constraints_value"),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > effective_at", name="ck_risk_constraints_window"
        ),
        sa.CheckConstraint("version >= 1", name="ck_risk_constraints_version"),
    )

    indexes = (
        ("ix_strategies_scope", "strategies", ["organization_id", "id"]),
        ("ix_membership_org_status", "organization_memberships", ["organization_id", "status"]),
        ("ix_portfolios_scope", "portfolios", ["organization_id", "strategy_id"]),
        ("ix_positions_portfolio_as_of", "positions", ["portfolio_id", "as_of"]),
        (
            "ix_mandates_scope_effective",
            "mandates",
            ["organization_id", "strategy_id", "effective_at"],
        ),
        ("ix_theses_scope_status", "theses", ["organization_id", "strategy_id", "status"]),
        ("ix_thesis_versions_thesis_number", "thesis_versions", ["thesis_id", "number"]),
        ("ix_assumptions_thesis_version", "thesis_assumptions", ["thesis_version_id"]),
        (
            "ix_decisions_scope_time",
            "strategy_decisions",
            ["organization_id", "strategy_id", "decided_at"],
        ),
        (
            "ix_risk_constraints_scope_effective",
            "risk_constraints",
            ["organization_id", "strategy_id", "effective_at"],
        ),
    )
    for name, table, columns in indexes:
        op.create_index(name, table, columns)
    for table in _TABLES:
        _protect(table)


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))
    for name, table, _columns in reversed(
        (
            ("ix_strategies_scope", "strategies", []),
            ("ix_membership_org_status", "organization_memberships", []),
            ("ix_portfolios_scope", "portfolios", []),
            ("ix_positions_portfolio_as_of", "positions", []),
            ("ix_mandates_scope_effective", "mandates", []),
            ("ix_theses_scope_status", "theses", []),
            ("ix_thesis_versions_thesis_number", "thesis_versions", []),
            ("ix_assumptions_thesis_version", "thesis_assumptions", []),
            ("ix_decisions_scope_time", "strategy_decisions", []),
            ("ix_risk_constraints_scope_effective", "risk_constraints", []),
        )
    ):
        op.drop_index(name, table_name=table)
    for table in reversed(_TABLES):
        op.drop_table(table)
