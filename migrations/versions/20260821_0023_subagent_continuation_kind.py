"""Allow durable child-agent tasks in the task continuation contract."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260821_0023"
down_revision: str | None = "20260821_0022"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint("ck_tasks_continuation_kind", "tasks", type_="check")
    op.create_check_constraint(
        "ck_tasks_continuation_kind",
        "tasks",
        "continuation_kind IN ('root', 'follow_up', 'subagent')",
    )


def downgrade() -> None:
    op.execute("UPDATE tasks SET continuation_kind = 'root' WHERE continuation_kind = 'subagent'")
    op.drop_constraint("ck_tasks_continuation_kind", "tasks", type_="check")
    op.create_check_constraint(
        "ck_tasks_continuation_kind",
        "tasks",
        "continuation_kind IN ('root', 'follow_up')",
    )
