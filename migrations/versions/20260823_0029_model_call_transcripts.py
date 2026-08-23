"""Add model_call_transcripts, a dashboard-only durable record of the exact
request/response for each model call.

Deliberately separate from run_events: that table is capped at 8KB and
field-allowlisted per event type to keep the durable event log bounded and
replayable, which a full transcript cannot respect. This table carries no
replay authority and follows the same demo exposure boundary as every other
Leo table: RLS enabled, anon/authenticated denied, no client policies.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260823_0029"
down_revision: str | None = "20260823_0028"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_call_transcripts",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("task_id", sa.String(64), nullable=False),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("request_id", sa.String(128), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("raw_request", JSONB, nullable=False),
        sa.Column("raw_response", JSONB, nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint("run_id", "request_id", name="uq_model_call_transcripts_identity"),
    )
    op.create_index("ix_model_call_transcripts_run", "model_call_transcripts", ["run_id"])

    op.execute(
        sa.text(
            "REVOKE ALL PRIVILEGES ON TABLE public.model_call_transcripts FROM anon, authenticated"
        )
    )
    op.execute(sa.text("ALTER TABLE public.model_call_transcripts ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            "CREATE POLICY leo_client_deny ON public.model_call_transcripts AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP POLICY IF EXISTS leo_client_deny ON public.model_call_transcripts"))
    op.execute(sa.text("ALTER TABLE public.model_call_transcripts DISABLE ROW LEVEL SECURITY"))
    op.drop_table("model_call_transcripts")
