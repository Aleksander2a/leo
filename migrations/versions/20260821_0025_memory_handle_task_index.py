"""Cover memory-capability-handle task foreign-key lookups."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260821_0025"
down_revision: str | None = "20260821_0024"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_memory_capability_handles_task",
        "memory_capability_handles",
        ["task_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_memory_capability_handles_task",
        table_name="memory_capability_handles",
    )
