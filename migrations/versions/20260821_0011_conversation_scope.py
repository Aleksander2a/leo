"""Add server-derived conversation identities and pinned thread scopes."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0011"
down_revision: str | None = "20260821_0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_TABLES = ("conversations", "conversation_threads", "conversation_scope_selections")


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
        "conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("provider", sa.String(32), nullable=False),
        sa.Column("team_id", sa.String(64), nullable=False),
        sa.Column("external_id", sa.String(128), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("actor_id", sa.String(128)),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "provider", "team_id", "external_id", name="uq_conversations_provider_external"
        ),
        sa.CheckConstraint(
            "kind IN ('channel', 'dm', 'group_dm', 'shared', 'external')",
            name="ck_conversations_kind",
        ),
        sa.CheckConstraint(
            "(kind = 'dm' AND actor_id IS NOT NULL) OR (kind <> 'dm' AND actor_id IS NULL)",
            name="ck_conversations_actor_shape",
        ),
    )
    op.create_index("ix_conversations_team_kind", "conversations", ["team_id", "kind"])

    op.create_table(
        "conversation_threads",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("root_ts", sa.String(64), nullable=False),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("mapping_version", sa.Integer, nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("conversation_id", "root_ts", name="uq_conversation_threads_root"),
        sa.CheckConstraint("mapping_version >= 1", name="ck_conversation_threads_mapping_version"),
        sa.CheckConstraint("version >= 1", name="ck_conversation_threads_version"),
    )
    op.create_index(
        "ix_conversation_threads_scope", "conversation_threads", ["organization_id", "strategy_id"]
    )
    op.create_index(
        "ix_conversation_threads_conversation",
        "conversation_threads",
        ["conversation_id", "root_ts"],
    )

    op.create_table(
        "conversation_scope_selections",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column(
            "organization_id", sa.String(64), sa.ForeignKey("organizations.id"), nullable=False
        ),
        sa.Column("strategy_id", sa.String(64), sa.ForeignKey("strategies.id"), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked')", name="ck_conversation_selection_status"
        ),
        sa.CheckConstraint("version >= 1", name="ck_conversation_selection_version"),
    )
    op.create_index(
        "ix_conversation_selection_actor",
        "conversation_scope_selections",
        ["conversation_id", "actor_id", "status"],
    )
    op.create_index(
        "ix_conversation_selection_scope",
        "conversation_scope_selections",
        ["organization_id", "strategy_id", "status"],
    )
    for table in _TABLES:
        _protect(table)


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.drop_table(table)
