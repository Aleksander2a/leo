"""Add pgvector-backed embedding tables for capability discovery and memory recall.

Enables the ``vector`` extension and adds two disposable semantic indexes
(capability_embeddings, memory_embeddings) plus a harness-set provenance
column (memory_revisions.source_type) distinguishing an explicit user command
from an autonomous model-proposed capture. Both new tables follow the same
demo exposure boundary as every other Leo table: RLS enabled, anon/
authenticated denied, no client policies.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "20260823_0028"
down_revision: str | None = "20260823_0027"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EMBEDDING_DIMENSION = 1536
_NEW_TABLES = ("memory_embeddings", "capability_embeddings")


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    op.add_column(
        "memory_revisions",
        sa.Column(
            "source_type",
            sa.String(16),
            nullable=False,
            server_default="explicit",
        ),
    )
    op.create_check_constraint(
        "ck_memory_revisions_source_type",
        "memory_revisions",
        "source_type IN ('explicit', 'autonomous')",
    )

    op.create_table(
        "memory_embeddings",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "revision_id",
            sa.String(64),
            sa.ForeignKey("memory_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "record_id",
            sa.String(64),
            sa.ForeignKey("memory_records.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIMENSION), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "revision_id", "content_hash", "model", name="uq_memory_embeddings_identity"
        ),
    )
    op.create_index(
        "ix_memory_embeddings_scope",
        "memory_embeddings",
        ["organization_id", "strategy_id"],
    )
    op.create_index("ix_memory_embeddings_record", "memory_embeddings", ["record_id"])

    op.create_table(
        "capability_embeddings",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("capability_id", sa.String(128), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("model", sa.String(128), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIMENSION), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "capability_id", "content_hash", "model", name="uq_capability_embeddings_identity"
        ),
    )

    for table in _NEW_TABLES:
        op.execute(
            sa.text(f"REVOKE ALL PRIVILEGES ON TABLE public.{table} FROM anon, authenticated")
        )
        op.execute(sa.text(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"))
        op.execute(
            sa.text(
                f"CREATE POLICY leo_client_deny ON public.{table} AS RESTRICTIVE "
                "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
            )
        )


def downgrade() -> None:
    for table in reversed(_NEW_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))

    op.drop_table("capability_embeddings")
    op.drop_table("memory_embeddings")
    op.drop_constraint("ck_memory_revisions_source_type", "memory_revisions", type_="check")
    op.drop_column("memory_revisions", "source_type")
    # The vector extension is left installed on downgrade -- other tables/sessions
    # may depend on it, and dropping it is a privileged, destructive operation out
    # of scope for a schema rollback.
