"""Add immutable Slack follow-up links and one-active-task-per-thread enforcement."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0008"
down_revision: str | None = "20260821_0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("threads", sa.Column("mapping_version", sa.Integer(), nullable=True))
    op.add_column("tasks", sa.Column("parent_task_id", sa.String(length=64), nullable=True))
    op.add_column(
        "tasks",
        sa.Column("continuation_kind", sa.String(length=32), nullable=True),
    )
    op.add_column("tasks", sa.Column("mapping_version", sa.Integer(), nullable=True))
    op.execute(
        sa.text("UPDATE tasks SET continuation_kind = 'root' WHERE continuation_kind IS NULL")
    )
    op.alter_column("tasks", "continuation_kind", nullable=False, server_default="root")
    op.create_foreign_key("tasks_parent_task_id_fkey", "tasks", "tasks", ["parent_task_id"], ["id"])
    op.create_check_constraint(
        "ck_tasks_continuation_kind",
        "tasks",
        "continuation_kind IN ('root', 'follow_up')",
    )
    op.create_check_constraint(
        "ck_tasks_mapping_version",
        "tasks",
        "mapping_version IS NULL OR mapping_version >= 1",
    )
    op.create_check_constraint(
        "ck_threads_mapping_version",
        "threads",
        "mapping_version IS NULL OR mapping_version >= 1",
    )
    op.create_index(
        "uq_tasks_one_active_per_thread",
        "tasks",
        ["thread_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'active', 'requires_action')"),
    )
    op.create_index("ix_tasks_parent_task_id", "tasks", ["parent_task_id"])


def downgrade() -> None:
    op.drop_index("ix_tasks_parent_task_id", table_name="tasks")
    op.drop_index("uq_tasks_one_active_per_thread", table_name="tasks")
    op.drop_constraint("ck_threads_mapping_version", "threads", type_="check")
    op.drop_constraint("ck_tasks_mapping_version", "tasks", type_="check")
    op.drop_constraint("ck_tasks_continuation_kind", "tasks", type_="check")
    op.drop_constraint("tasks_parent_task_id_fkey", "tasks", type_="foreignkey")
    op.drop_column("tasks", "mapping_version")
    op.drop_column("tasks", "continuation_kind")
    op.drop_column("tasks", "parent_task_id")
    op.drop_column("threads", "mapping_version")
