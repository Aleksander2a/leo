"""Add a deterministic Postgres full-text retrieval index."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0014"
down_revision: str | None = "20260821_0013"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            "ALTER TABLE public.memory_revisions ADD COLUMN search_vector tsvector "
            "GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED"
        )
    )
    op.create_index(
        "ix_memory_revisions_search_vector",
        "memory_revisions",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_memory_revisions_search_vector", table_name="memory_revisions")
    op.drop_column("memory_revisions", "search_vector")
