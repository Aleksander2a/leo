"""Add durable Slack channel scopes and ingress authority snapshots.

Revision ID: 20260820_0002
Revises: 20260820_0001
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260820_0002"
down_revision: str | None = "20260820_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "slack_channel_scopes",
        sa.Column("team_id", sa.String(length=32), nullable=False),
        sa.Column("channel_id", sa.String(length=32), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("provisioned_by_user_id", sa.String(length=32), nullable=False),
        sa.Column("provisioned_via", sa.String(length=32), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint(
            "status IN ('active', 'pending', 'revoked', 'conflict')",
            name="ck_slack_channel_scopes_status",
        ),
        sa.CheckConstraint("version >= 1", name="ck_slack_channel_scopes_version"),
        sa.PrimaryKeyConstraint("team_id", "channel_id"),
    )
    op.create_index(
        "ix_slack_channel_scopes_scope_status",
        "slack_channel_scopes",
        ["organization_id", "strategy_id", "status"],
    )

    # Legacy admissions remain readable with an empty snapshot. Atomic admission in 0002+
    # always writes all three fields together before its transaction commits.
    op.add_column(
        "slack_ingress_events",
        sa.Column("organization_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("strategy_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("mapping_version", sa.Integer(), nullable=True),
    )
    op.create_check_constraint(
        "ck_slack_ingress_mapping_version",
        "slack_ingress_events",
        "mapping_version IS NULL OR mapping_version >= 1",
    )
    op.create_check_constraint(
        "ck_slack_ingress_scope_snapshot",
        "slack_ingress_events",
        "(organization_id IS NULL AND strategy_id IS NULL AND mapping_version IS NULL) "
        "OR (organization_id IS NOT NULL AND strategy_id IS NOT NULL "
        "AND mapping_version IS NOT NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_slack_ingress_scope_snapshot",
        "slack_ingress_events",
        type_="check",
    )
    op.drop_constraint(
        "ck_slack_ingress_mapping_version",
        "slack_ingress_events",
        type_="check",
    )
    op.drop_column("slack_ingress_events", "mapping_version")
    op.drop_column("slack_ingress_events", "strategy_id")
    op.drop_column("slack_ingress_events", "organization_id")
    op.drop_index("ix_slack_channel_scopes_scope_status", table_name="slack_channel_scopes")
    op.drop_table("slack_channel_scopes")
