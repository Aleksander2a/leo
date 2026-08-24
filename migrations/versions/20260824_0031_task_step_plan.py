"""Add tasks.step_plan, the model's committed plan for the run.

The scratchpad records what already happened. Nothing recorded what the model
said it was going to do, so the coordinator could not tell a finished answer
from an abandoned one: a turn that narrated "I'm pulling the earnings data now"
and then stopped was indistinguishable from a turn that was actually done, and
shipped to Slack as a final answer.

This column stores an ordered list of {key, intent, tool, status, note} steps.
Completion is gated on it -- a run cannot finish while a step is pending -- and
a step naming a tool is discharged only by a real retrieved observation from
that tool, never by the model asserting it is complete.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260824_0031"
down_revision: str | None = "20260824_0030"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column(
            "step_plan",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("tasks", "step_plan")
