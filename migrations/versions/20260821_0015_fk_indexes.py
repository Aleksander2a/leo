"""Add covering indexes for domain, conversation, and memory foreign keys."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260821_0015"
down_revision: str | None = "20260821_0014"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_INDEXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("ix_conversation_threads_strategy", "conversation_threads", ("strategy_id",)),
    (
        "ix_conversation_scope_selections_strategy",
        "conversation_scope_selections",
        ("strategy_id",),
    ),
    ("ix_mandates_strategy", "mandates", ("strategy_id",)),
    ("ix_memory_disclosure_grants_strategy", "memory_disclosure_grants", ("strategy_id",)),
    (
        "ix_memory_disclosure_grant_events_strategy",
        "memory_disclosure_grant_events",
        ("strategy_id",),
    ),
    ("ix_portfolios_strategy", "portfolios", ("strategy_id",)),
    ("ix_positions_asset", "positions", ("asset_id",)),
    ("ix_risk_constraints_strategy", "risk_constraints", ("strategy_id",)),
    ("ix_sanitized_messages_strategy", "sanitized_messages", ("strategy_id",)),
    ("ix_strategy_decisions_strategy", "strategy_decisions", ("strategy_id",)),
    (
        "ix_strategy_decisions_thesis_version",
        "strategy_decisions",
        ("thesis_version_id",),
    ),
    ("ix_theses_strategy", "theses", ("strategy_id",)),
)


def upgrade() -> None:
    for index_name, table_name, columns in _INDEXES:
        op.create_index(index_name, table_name, columns)


def downgrade() -> None:
    for index_name, table_name, _ in reversed(_INDEXES):
        op.drop_index(index_name, table_name=table_name)
