"""Separate sanitized messages and derived memory-plane metadata."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0013"
down_revision: str | None = "20260821_0012"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _protect(table: str) -> None:
    op.execute(sa.text(f"REVOKE ALL PRIVILEGES ON TABLE public.{table} FROM anon, authenticated"))
    op.execute(sa.text(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            f"CREATE POLICY leo_client_deny ON public.{table} AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def upgrade() -> None:
    op.create_table(
        "sanitized_messages",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("destination_id", sa.String(128), nullable=False),
        sa.Column("external_event_id", sa.String(128), nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["organization_id"], ["organizations.id"]),
        sa.ForeignKeyConstraint(["strategy_id"], ["strategies.id"]),
        sa.CheckConstraint(
            "char_length(text) BETWEEN 1 AND 8192", name="ck_sanitized_messages_text"
        ),
    )
    op.create_index(
        "ix_sanitized_messages_scope_time",
        "sanitized_messages",
        ["organization_id", "strategy_id", "recorded_at"],
    )
    op.create_index(
        "ix_sanitized_messages_destination", "sanitized_messages", ["destination_id", "recorded_at"]
    )

    op.create_table(
        "thread_summary_revisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("thread_id", sa.String(64), nullable=False),
        sa.Column("source_message_ids", sa.JSON, nullable=False),
        sa.Column("revision", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("revision >= 1", name="ck_thread_summary_revision"),
        sa.CheckConstraint(
            "char_length(content) BETWEEN 1 AND 8192", name="ck_thread_summary_content"
        ),
        sa.UniqueConstraint("thread_id", "revision", name="uq_thread_summary_revision"),
    )
    op.create_index(
        "ix_thread_summary_scope",
        "thread_summary_revisions",
        ["organization_id", "strategy_id", "thread_id"],
    )

    op.create_table(
        "memory_embedding_jobs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("source_plane", sa.String(32), nullable=False),
        sa.Column("source_id", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("dimensions", sa.Integer, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued"),
        sa.Column("attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("dimensions >= 1", name="ck_memory_embedding_dimensions"),
        sa.CheckConstraint("attempts >= 0", name="ck_memory_embedding_attempts"),
        sa.CheckConstraint(
            "status IN ('queued', 'retry', 'succeeded', 'dead')", name="ck_memory_embedding_status"
        ),
        sa.UniqueConstraint("source_id", "content_hash", "model", name="uq_memory_embedding_work"),
    )
    op.create_index(
        "ix_memory_embedding_jobs_scope_status",
        "memory_embedding_jobs",
        ["organization_id", "strategy_id", "status"],
    )

    op.create_table(
        "memory_retrieval_cache",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("key_hash", sa.String(64), nullable=False),
        sa.Column("generation", sa.Integer, nullable=False),
        sa.Column("result_ids", sa.JSON, nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint("generation >= 1", name="ck_memory_retrieval_cache_generation"),
        sa.UniqueConstraint(
            "organization_id",
            "strategy_id",
            "key_hash",
            "generation",
            name="uq_memory_retrieval_cache_key",
        ),
    )
    op.create_index(
        "ix_memory_retrieval_cache_scope",
        "memory_retrieval_cache",
        ["organization_id", "strategy_id", "generation"],
    )
    for table in (
        "sanitized_messages",
        "thread_summary_revisions",
        "memory_embedding_jobs",
        "memory_retrieval_cache",
    ):
        _protect(table)


def downgrade() -> None:
    for table in (
        "memory_retrieval_cache",
        "memory_embedding_jobs",
        "thread_summary_revisions",
        "sanitized_messages",
    ):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.drop_table(table)
