"""Add tasks.scratchpad, Leo's durable ReAct working memory.

Each model iteration previously rebuilt a stateless prompt from the objective,
the observation set, and the verifier's complaints. Nothing carried the model's
own reasoning forward, so by iteration four it could not tell which tools it had
already called, with what arguments, or what it had been trying to establish.
Genuine multi-step work ("I have the quote, now I need earnings, then I compare")
was impossible, and near-identical prompts produced near-identical decisions that
tripped the no-progress guard.

This column stores a bounded list of {iteration, plan, action, outcome} steps.
The plan is model-authored; the outcome is always written by the harness from
what actually happened, so a model cannot record its own success. The trace
carries no authority: it cannot grant a capability, cite evidence, or satisfy a
verifier check.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260824_0030"
down_revision: str | None = "20260823_0029"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "tasks",
        sa.Column(
            "scratchpad",
            JSONB,
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_column("tasks", "scratchpad")
