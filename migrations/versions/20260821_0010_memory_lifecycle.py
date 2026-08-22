"""Add append-only, scope-first memory lifecycle records."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "20260821_0010"
down_revision: str | None = "20260821_0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TABLES = ("memory_records", "memory_sources", "memory_revisions")
_VISIBILITY = (
    "'thread_local', 'channel_local', 'actor_private', 'strategy_shared', 'organization_shared'"
)
_STATUS = "'active', 'superseded', 'contested', 'retracted'"


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
        "memory_records",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("kind", sa.String(64), nullable=False),
        sa.Column("visibility", sa.String(32), nullable=False),
        sa.Column("namespace_id", sa.String(128), nullable=False),
        sa.Column("current_revision", sa.Integer, nullable=False, server_default="1"),
        sa.Column("generation", sa.Integer, nullable=False, server_default="1"),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(f"visibility IN ({_VISIBILITY})", name="ck_memory_records_visibility"),
        sa.CheckConstraint(f"status IN ({_STATUS})", name="ck_memory_records_status"),
        sa.CheckConstraint("current_revision >= 1", name="ck_memory_records_revision"),
        sa.CheckConstraint("generation >= 1", name="ck_memory_records_generation"),
    )
    op.create_table(
        "memory_sources",
        sa.Column("id", sa.String(128), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("source_kind", sa.String(64), nullable=False),
        sa.Column("reference", sa.String(255), nullable=False),
        sa.Column("visibility", sa.String(32), nullable=False),
        sa.Column("namespace_id", sa.String(128), nullable=False),
        sa.CheckConstraint(f"visibility IN ({_VISIBILITY})", name="ck_memory_sources_visibility"),
    )
    op.create_table(
        "memory_revisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("record_id", sa.String(64), sa.ForeignKey("memory_records.id"), nullable=False),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("source_ids", JSONB, nullable=False),
        sa.Column("visibility", sa.String(32), nullable=False),
        sa.Column("namespace_id", sa.String(128), nullable=False),
        sa.Column("sensitivity", sa.Float, nullable=False),
        sa.Column("valid_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("valid_until", sa.DateTime(timezone=True)),
        sa.Column("recorded_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("actor_id", sa.String(128), nullable=False),
        sa.Column("reason", sa.String(128), nullable=False),
        sa.Column("supersedes_revision", sa.Integer),
        sa.UniqueConstraint("record_id", "number", name="uq_memory_revisions_number"),
        sa.CheckConstraint("number >= 1", name="ck_memory_revisions_number"),
        sa.CheckConstraint(
            "char_length(content) <= 16384", name="ck_memory_revisions_content_size"
        ),
        sa.CheckConstraint(
            "sensitivity >= 0 AND sensitivity <= 1", name="ck_memory_revisions_sensitivity"
        ),
        sa.CheckConstraint(f"visibility IN ({_VISIBILITY})", name="ck_memory_revisions_visibility"),
        sa.CheckConstraint(f"status IN ({_STATUS})", name="ck_memory_revisions_status"),
        sa.CheckConstraint(
            "valid_until IS NULL OR valid_until > valid_from",
            name="ck_memory_revisions_valid_window",
        ),
        sa.CheckConstraint(
            "expires_at IS NULL OR expires_at > recorded_at", name="ck_memory_revisions_expiry"
        ),
        sa.CheckConstraint(
            "status <> 'superseded' OR supersedes_revision IS NOT NULL",
            name="ck_memory_revisions_superseded_parent",
        ),
    )
    indexes = (
        (
            "ix_memory_records_scope_status",
            "memory_records",
            ["organization_id", "strategy_id", "status"],
        ),
        ("ix_memory_records_namespace", "memory_records", ["visibility", "namespace_id", "status"]),
        ("ix_memory_sources_scope", "memory_sources", ["organization_id", "strategy_id"]),
        ("ix_memory_sources_namespace", "memory_sources", ["visibility", "namespace_id"]),
        ("ix_memory_revisions_record_number", "memory_revisions", ["record_id", "number"]),
        (
            "ix_memory_revisions_scope_status",
            "memory_revisions",
            ["organization_id", "strategy_id", "status"],
        ),
        (
            "ix_memory_revisions_validity",
            "memory_revisions",
            ["valid_from", "valid_until", "expires_at"],
        ),
    )
    for name, table, columns in indexes:
        op.create_index(name, table, columns)
    for table in _TABLES:
        _protect(table)


def downgrade() -> None:
    for table in reversed(_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))
    indexes = (
        ("ix_memory_records_scope_status", "memory_records"),
        ("ix_memory_records_namespace", "memory_records"),
        ("ix_memory_sources_scope", "memory_sources"),
        ("ix_memory_sources_namespace", "memory_sources"),
        ("ix_memory_revisions_record_number", "memory_revisions"),
        ("ix_memory_revisions_scope_status", "memory_revisions"),
        ("ix_memory_revisions_validity", "memory_revisions"),
    )
    for name, table in reversed(indexes):
        op.drop_index(name, table_name=table)
    for table in reversed(_TABLES):
        op.drop_table(table)
