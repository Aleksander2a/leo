"""Add durable Task lease and retry state for the local worker queue.

Revision ID: 20260821_0005
Revises: 20260821_0004
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0005"
down_revision: str | None = "20260821_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("tasks", sa.Column("lease_owner", sa.String(length=128), nullable=True))
    op.add_column("tasks", sa.Column("lease_token", sa.String(length=128), nullable=True))
    op.add_column("tasks", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column("tasks", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "tasks",
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("tasks", sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True))
    op.add_column("tasks", sa.Column("last_error", sa.String(length=255), nullable=True))
    op.create_check_constraint(
        "ck_tasks_attempt_count",
        "tasks",
        "attempt_count >= 0",
    )
    op.create_check_constraint(
        "ck_tasks_lease_fields",
        "tasks",
        "(lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL "
        "AND heartbeat_at IS NULL) OR "
        "(lease_owner IS NOT NULL AND lease_token IS NOT NULL "
        "AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_tasks_terminal_no_lease",
        "tasks",
        "status NOT IN ('completed', 'failed', 'cancelled') OR "
        "(lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL "
        "AND heartbeat_at IS NULL)",
    )
    op.create_index(
        "ix_tasks_claim_eligibility",
        "tasks",
        ["status", "retry_after", "lease_expires_at", "created_at", "id"],
    )


def downgrade() -> None:
    op.drop_index("ix_tasks_claim_eligibility", table_name="tasks")
    op.drop_constraint("ck_tasks_terminal_no_lease", "tasks", type_="check")
    op.drop_constraint("ck_tasks_lease_fields", "tasks", type_="check")
    op.drop_constraint("ck_tasks_attempt_count", "tasks", type_="check")
    op.drop_column("tasks", "last_error")
    op.drop_column("tasks", "retry_after")
    op.drop_column("tasks", "attempt_count")
    op.drop_column("tasks", "heartbeat_at")
    op.drop_column("tasks", "lease_expires_at")
    op.drop_column("tasks", "lease_token")
    op.drop_column("tasks", "lease_owner")
