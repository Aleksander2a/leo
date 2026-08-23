"""Switch memory full-text search from the 'simple' to the 'english' configuration.

English stemming lets a query for "cancel" match stored content containing
"cancelled"/"cancelling" and similar morphological variants; the 'simple'
configuration performed no stemming at all.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260823_0027"
down_revision: str | None = "20260822_0026"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_memory_revisions_search_vector", table_name="memory_revisions")
    op.drop_column("memory_revisions", "search_vector")
    op.execute(
        sa.text(
            "ALTER TABLE public.memory_revisions ADD COLUMN search_vector tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
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
