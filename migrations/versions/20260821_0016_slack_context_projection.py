"""Persist Slack conversation authority and exact context projections."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260821_0016"
down_revision: str | None = "20260821_0015"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Add nullable columns first so the existing ingress table can be backfilled without a
    # long table rewrite or an invalid intermediate state.
    op.add_column(
        "slack_ingress_events",
        sa.Column("conversation_kind", sa.String(24), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("trigger_kind", sa.String(24), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("context_conversation_ids", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("context_access_hash", sa.String(64), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE public.slack_ingress_events
            SET conversation_kind = CASE
                    WHEN channel_id LIKE 'D%' THEN 'dm'
                    WHEN channel_id LIKE 'G%' THEN 'mpim'
                    ELSE 'ordinary_internal'
                END,
                trigger_kind = 'app_mention',
                context_conversation_ids = jsonb_build_array(channel_id),
                context_access_hash = encode(
                    sha256(
                        convert_to(
                            concat_ws(
                                chr(31),
                                'slack-context-v1',
                                team_id,
                                user_id,
                                channel_id,
                                channel_id
                            ),
                            'UTF8'
                        )
                    ),
                    'hex'
                )
            WHERE conversation_kind IS NULL
               OR trigger_kind IS NULL
               OR context_conversation_ids IS NULL
               OR context_access_hash IS NULL
            """
        )
    )
    op.alter_column("slack_ingress_events", "conversation_kind", nullable=False)
    op.alter_column("slack_ingress_events", "trigger_kind", nullable=False)
    op.alter_column("slack_ingress_events", "context_conversation_ids", nullable=False)
    op.alter_column("slack_ingress_events", "context_access_hash", nullable=False)

    op.create_check_constraint(
        "ck_slack_ingress_conversation_kind",
        "slack_ingress_events",
        "conversation_kind IN ('ordinary_internal', 'dm', 'mpim', 'shared', 'external')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_trigger_kind",
        "slack_ingress_events",
        "trigger_kind IN ('app_mention', 'message_im')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_context_shape",
        "slack_ingress_events",
        "jsonb_typeof(context_conversation_ids) = 'array' "
        "AND jsonb_array_length(context_conversation_ids) >= 1",
    )
    op.create_check_constraint(
        "ck_slack_ingress_context_current",
        "slack_ingress_events",
        "context_conversation_ids ? channel_id",
    )
    op.create_check_constraint(
        "ck_slack_ingress_context_isolation",
        "slack_ingress_events",
        "conversation_kind = 'dm' OR "
        "(jsonb_array_length(context_conversation_ids) = 1 "
        "AND context_conversation_ids ->> 0 = channel_id)",
    )
    op.create_check_constraint(
        "ck_slack_ingress_message_im_kind",
        "slack_ingress_events",
        "trigger_kind <> 'message_im' OR conversation_kind = 'dm'",
    )
    op.create_check_constraint(
        "ck_slack_ingress_context_access_hash",
        "slack_ingress_events",
        "context_access_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_index(
        "ix_slack_ingress_actor_context",
        "slack_ingress_events",
        ["team_id", "user_id", "conversation_kind", "received_at"],
    )
    op.create_index(
        "ix_slack_ingress_context_conversations",
        "slack_ingress_events",
        ["context_conversation_ids"],
        postgresql_using="gin",
    )
    op.create_index(
        "ix_slack_ingress_context_access_hash",
        "slack_ingress_events",
        ["context_access_hash"],
    )

    # The table was protected in revision 0003. Reassert the boundary and create the same
    # restrictive policy only if an operator removed it between revisions.
    op.execute(
        sa.text(
            "REVOKE ALL PRIVILEGES ON TABLE public.slack_ingress_events FROM anon, authenticated"
        )
    )
    op.execute(sa.text("ALTER TABLE public.slack_ingress_events ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_policies
                    WHERE schemaname = 'public'
                      AND tablename = 'slack_ingress_events'
                      AND policyname = 'leo_client_deny'
                ) THEN
                    EXECUTE 'CREATE POLICY leo_client_deny ON public.slack_ingress_events '
                            'AS RESTRICTIVE FOR ALL TO anon, authenticated '
                            'USING (false) WITH CHECK (false)';
                END IF;
            END
            $$
            """
        )
    )


def downgrade() -> None:
    op.drop_index(
        "ix_slack_ingress_context_access_hash",
        table_name="slack_ingress_events",
    )
    op.drop_index(
        "ix_slack_ingress_context_conversations",
        table_name="slack_ingress_events",
    )
    op.drop_index("ix_slack_ingress_actor_context", table_name="slack_ingress_events")
    for constraint in (
        "ck_slack_ingress_context_access_hash",
        "ck_slack_ingress_message_im_kind",
        "ck_slack_ingress_context_isolation",
        "ck_slack_ingress_context_current",
        "ck_slack_ingress_context_shape",
        "ck_slack_ingress_trigger_kind",
        "ck_slack_ingress_conversation_kind",
    ):
        op.drop_constraint(constraint, "slack_ingress_events", type_="check")
    op.drop_column("slack_ingress_events", "context_access_hash")
    op.drop_column("slack_ingress_events", "context_conversation_ids")
    op.drop_column("slack_ingress_events", "trigger_kind")
    op.drop_column("slack_ingress_events", "conversation_kind")
