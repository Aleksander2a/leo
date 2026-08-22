"""Add durable run-bound progressive-memory capability handles."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260821_0021"
down_revision: str | None = "20260821_0020"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "memory_capability_handles",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("handle_hash", sa.String(64), nullable=False),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column(
            "task_id",
            sa.String(64),
            sa.ForeignKey("tasks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "run_id",
            sa.String(64),
            sa.ForeignKey("runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("team_id", sa.String(32), nullable=False),
        sa.Column("destination_id", sa.String(128), nullable=False),
        sa.Column("destination_kind", sa.String(16), nullable=False),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("access_hash", sa.String(64), nullable=False),
        sa.Column("membership_hash", sa.String(64), nullable=False),
        sa.Column(
            "source_conversation_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("current_thread_namespace_id", sa.String(255), nullable=False),
        sa.Column(
            "record_id",
            sa.String(64),
            sa.ForeignKey("memory_records.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("visibility", sa.String(32), nullable=False),
        sa.Column("namespace_id", sa.String(128), nullable=False),
        sa.Column("policy_version", sa.String(64), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("max_opens", sa.Integer, nullable=False, server_default="8"),
        sa.Column("open_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("invalidated_at", sa.DateTime(timezone=True)),
        sa.Column("invalidation_reason", sa.String(128)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("handle_hash", name="uq_memory_capability_handle_hash"),
        sa.CheckConstraint(
            "destination_kind IN ('channel', 'dm', 'group_dm', 'shared', 'external')",
            name="ck_memory_capability_handle_destination_kind",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(source_conversation_ids) = 'array' "
            "AND jsonb_array_length(source_conversation_ids) >= 1",
            name="ck_memory_capability_handle_sources",
        ),
        sa.CheckConstraint(
            "source_conversation_ids ? destination_id",
            name="ck_memory_capability_handle_destination_source",
        ),
        sa.CheckConstraint("revision >= 1", name="ck_memory_capability_handle_revision"),
        sa.CheckConstraint(
            "max_opens BETWEEN 1 AND 64 AND open_count BETWEEN 0 AND max_opens",
            name="ck_memory_capability_handle_open_budget",
        ),
        sa.CheckConstraint(
            "access_hash ~ '^[0-9a-f]{64}$' AND membership_hash ~ '^[0-9a-f]{64}$'",
            name="ck_memory_capability_handle_authority_hashes",
        ),
        sa.CheckConstraint(
            "handle_hash ~ '^[0-9a-f]{64}$'",
            name="ck_memory_capability_handle_hash",
        ),
        sa.CheckConstraint(
            "(invalidated_at IS NULL AND invalidation_reason IS NULL) OR "
            "(invalidated_at IS NOT NULL AND invalidation_reason IS NOT NULL)",
            name="ck_memory_capability_handle_invalidation",
        ),
    )
    op.create_index(
        "ix_memory_capability_handles_run",
        "memory_capability_handles",
        ["run_id", "expires_at"],
    )
    op.create_index(
        "ix_memory_capability_handles_actor",
        "memory_capability_handles",
        ["organization_id", "team_id", "actor_id", "invalidated_at"],
    )
    op.create_index(
        "ix_memory_capability_handles_record",
        "memory_capability_handles",
        ["record_id", "revision"],
    )
    op.execute(
        sa.text(
            "REVOKE ALL PRIVILEGES ON TABLE public.memory_capability_handles "
            "FROM anon, authenticated"
        )
    )
    op.execute(sa.text("ALTER TABLE public.memory_capability_handles ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            "CREATE POLICY leo_client_deny ON public.memory_capability_handles AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def downgrade() -> None:
    op.drop_index("ix_memory_capability_handles_record", table_name="memory_capability_handles")
    op.drop_index("ix_memory_capability_handles_actor", table_name="memory_capability_handles")
    op.drop_index("ix_memory_capability_handles_run", table_name="memory_capability_handles")
    op.drop_table("memory_capability_handles")
