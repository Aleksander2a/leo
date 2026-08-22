"""Add recoverable admission-to-task launch intent state.

Revision ID: 20260821_0004
Revises: 20260821_0003
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0004"
down_revision: str | None = "20260821_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "slack_ingress_events",
        sa.Column("launch_status", sa.String(length=16), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("launch_attempt_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("launch_error", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("launch_updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE slack_ingress_events
            SET launch_status = CASE
                    WHEN task_id IS NOT NULL THEN 'queued'
                    WHEN status = 'policy_rejected' THEN 'rejected'
                    WHEN status = 'runtime_failed' THEN 'failed'
                    ELSE 'unlaunched'
                END,
                launch_attempt_count = COALESCE(attempt_count, 0),
                launch_updated_at = COALESCE(updated_at, received_at, now())
            WHERE launch_status IS NULL
            """
        )
    )
    op.alter_column(
        "slack_ingress_events",
        "launch_status",
        nullable=False,
        server_default="unlaunched",
    )
    op.alter_column(
        "slack_ingress_events",
        "launch_attempt_count",
        nullable=False,
        server_default="0",
    )
    op.alter_column(
        "slack_ingress_events",
        "launch_updated_at",
        nullable=False,
        server_default=sa.func.now(),
    )
    op.create_check_constraint(
        "ck_slack_ingress_launch_status",
        "slack_ingress_events",
        "launch_status IN ("
        "'admitting', 'unlaunched', 'materializing', 'queued', 'failed', 'rejected'"
        ")",
    )
    op.create_check_constraint(
        "ck_slack_ingress_launch_attempt_count",
        "slack_ingress_events",
        "launch_attempt_count >= 0",
    )
    op.create_check_constraint(
        "ck_slack_ingress_launch_link",
        "slack_ingress_events",
        "(launch_status = 'queued' AND task_id IS NOT NULL) OR "
        "(launch_status IN ('admitting', 'unlaunched', 'materializing', 'failed', 'rejected') "
        "AND task_id IS NULL)",
    )
    op.create_check_constraint(
        "ck_slack_ingress_launch_payload",
        "slack_ingress_events",
        "launch_status NOT IN ('unlaunched', 'materializing') OR "
        "(organization_id IS NOT NULL AND strategy_id IS NOT NULL "
        "AND mapping_version IS NOT NULL AND prompt <> '')",
    )
    op.create_index(
        "uq_slack_ingress_task_id",
        "slack_ingress_events",
        ["task_id"],
        unique=True,
        postgresql_where=sa.text("task_id IS NOT NULL"),
    )
    op.create_index(
        "ix_slack_ingress_launch_status",
        "slack_ingress_events",
        ["launch_status", "launch_updated_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_slack_ingress_launch_status", table_name="slack_ingress_events")
    op.drop_index("uq_slack_ingress_task_id", table_name="slack_ingress_events")
    op.drop_constraint("ck_slack_ingress_launch_payload", "slack_ingress_events", type_="check")
    op.drop_constraint("ck_slack_ingress_launch_link", "slack_ingress_events", type_="check")
    op.drop_constraint(
        "ck_slack_ingress_launch_attempt_count", "slack_ingress_events", type_="check"
    )
    op.drop_constraint("ck_slack_ingress_launch_status", "slack_ingress_events", type_="check")
    op.drop_column("slack_ingress_events", "launch_updated_at")
    op.drop_column("slack_ingress_events", "launch_error")
    op.drop_column("slack_ingress_events", "launch_attempt_count")
    op.drop_column("slack_ingress_events", "launch_status")
