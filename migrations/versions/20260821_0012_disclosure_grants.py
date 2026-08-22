"""Add explicit, auditable cross-destination disclosure grants."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0012"
down_revision: str | None = "20260821_0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


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
        "memory_disclosure_grants",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("source_provider", sa.String(32), nullable=False),
        sa.Column("source_team_id", sa.String(64), nullable=False),
        sa.Column("source_destination_id", sa.String(128), nullable=False),
        sa.Column("destination_provider", sa.String(32), nullable=False),
        sa.Column("destination_team_id", sa.String(64), nullable=False),
        sa.Column("destination_destination_id", sa.String(128), nullable=False),
        sa.Column("sensitivity_ceiling", sa.Float, nullable=False),
        sa.Column("authorizing_actor_id", sa.String(128), nullable=False),
        sa.Column("authorizing_role", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column("provenance", sa.String(255), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "sensitivity_ceiling >= 0 AND sensitivity_ceiling <= 1",
            name="ck_disclosure_grants_sensitivity",
        ),
        sa.CheckConstraint(
            "authorizing_role IN ('owner', 'researcher')", name="ck_disclosure_grants_role"
        ),
        sa.CheckConstraint("status IN ('active', 'revoked')", name="ck_disclosure_grants_status"),
        sa.CheckConstraint("version >= 1", name="ck_disclosure_grants_version"),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > created_at", name="ck_disclosure_grants_expiry"
        ),
    )
    op.create_index(
        "ix_disclosure_grants_source_destination",
        "memory_disclosure_grants",
        [
            "organization_id",
            "strategy_id",
            "source_provider",
            "source_team_id",
            "source_destination_id",
            "status",
        ],
    )
    op.create_index(
        "ix_disclosure_grants_destination",
        "memory_disclosure_grants",
        [
            "organization_id",
            "strategy_id",
            "destination_provider",
            "destination_team_id",
            "destination_destination_id",
            "status",
        ],
    )
    op.create_table(
        "memory_disclosure_grant_events",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "grant_id", sa.String(64), sa.ForeignKey("memory_disclosure_grants.id"), nullable=False
        ),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("action", sa.String(16), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("version", sa.Integer, nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column(
            "occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "action IN ('created', 'revoked', 'denied')", name="ck_disclosure_grant_events_action"
        ),
        sa.CheckConstraint("version >= 1", name="ck_disclosure_grant_events_version"),
    )
    op.create_index(
        "ix_disclosure_grant_events_scope",
        "memory_disclosure_grant_events",
        ["organization_id", "strategy_id", "occurred_at"],
    )
    op.create_index(
        "ix_disclosure_grant_events_grant",
        "memory_disclosure_grant_events",
        ["grant_id", "version"],
    )
    for table in ("memory_disclosure_grants", "memory_disclosure_grant_events"):
        _protect(table)


def downgrade() -> None:
    for table in ("memory_disclosure_grant_events", "memory_disclosure_grants"):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.drop_table(table)
