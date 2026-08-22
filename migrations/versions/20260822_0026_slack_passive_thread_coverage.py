"""Persist passive Slack thread identity and authoritative coverage metadata."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260822_0026"
down_revision: str | None = "20260821_0025"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "sanitized_messages",
        sa.Column("provider_thread_root_ts", sa.String(64)),
    )
    op.create_check_constraint(
        "ck_sanitized_messages_provider_thread_root_ts",
        "sanitized_messages",
        "provider_thread_root_ts IS NULL OR provider_thread_root_ts ~ '^[0-9]+[.][0-9]+$'",
    )
    op.create_index(
        "ix_sanitized_messages_provider_thread",
        "sanitized_messages",
        ["conversation_id", "provider_thread_root_ts", "provider_message_ts"],
        postgresql_where=sa.text("provider_thread_root_ts IS NOT NULL"),
    )
    op.create_table(
        "slack_thread_coverage",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("team_id", sa.String(64), nullable=False),
        sa.Column("channel_id", sa.String(128), nullable=False),
        sa.Column("thread_root_ts", sa.String(64), nullable=False),
        sa.Column("authoritative_reply_count", sa.Integer, nullable=False),
        sa.Column("authoritative_latest_reply_ts", sa.String(64)),
        sa.Column("authority_source", sa.String(64), nullable=False),
        sa.Column("authority_snapshot_hash", sa.String(64), nullable=False),
        sa.Column("metadata_observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "team_id",
            "channel_id",
            "thread_root_ts",
            name="uq_slack_thread_coverage_root",
        ),
        sa.CheckConstraint(
            "authoritative_reply_count >= 0",
            name="ck_slack_thread_coverage_reply_count",
        ),
        sa.CheckConstraint(
            "(authoritative_reply_count = 0 AND authoritative_latest_reply_ts IS NULL) OR "
            "(authoritative_reply_count > 0 AND authoritative_latest_reply_ts IS NOT NULL)",
            name="ck_slack_thread_coverage_reply_shape",
        ),
        sa.CheckConstraint(
            "thread_root_ts ~ '^[0-9]+[.][0-9]+$'",
            name="ck_slack_thread_coverage_root_ts",
        ),
        sa.CheckConstraint(
            "CASE WHEN authoritative_latest_reply_ts IS NULL THEN true "
            "WHEN authoritative_latest_reply_ts ~ '^[0-9]+[.][0-9]+$' "
            "AND thread_root_ts ~ '^[0-9]+[.][0-9]+$' "
            "THEN authoritative_latest_reply_ts::numeric > thread_root_ts::numeric "
            "ELSE false END",
            name="ck_slack_thread_coverage_latest_ts",
        ),
        sa.CheckConstraint(
            "authority_source IN ('slack_conversations_history_bot', "
            "'slack_conversations_history_user')",
            name="ck_slack_thread_coverage_authority_source",
        ),
        sa.CheckConstraint(
            "authority_snapshot_hash ~ '^[0-9a-f]{64}$'",
            name="ck_slack_thread_coverage_snapshot_hash",
        ),
    )
    op.create_index(
        "ix_slack_thread_coverage_conversation",
        "slack_thread_coverage",
        ["conversation_id", "thread_root_ts"],
    )
    op.execute(
        sa.text(
            "REVOKE ALL PRIVILEGES ON TABLE public.slack_thread_coverage FROM anon, authenticated"
        )
    )
    op.execute(sa.text("ALTER TABLE public.slack_thread_coverage ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            "CREATE POLICY leo_client_deny ON public.slack_thread_coverage AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP POLICY IF EXISTS leo_client_deny ON public.slack_thread_coverage"))
    op.drop_index(
        "ix_slack_thread_coverage_conversation",
        table_name="slack_thread_coverage",
    )
    op.drop_table("slack_thread_coverage")
    op.drop_index(
        "ix_sanitized_messages_provider_thread",
        table_name="sanitized_messages",
    )
    op.drop_constraint(
        "ck_sanitized_messages_provider_thread_root_ts",
        "sanitized_messages",
        type_="check",
    )
    op.drop_column("sanitized_messages", "provider_thread_root_ts")
