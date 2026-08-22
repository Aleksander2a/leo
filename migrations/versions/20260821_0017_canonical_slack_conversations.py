"""Link Slack ingress and harness threads to canonical conversations."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0017"
down_revision: str | None = "20260821_0016"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            INSERT INTO public.conversations (
                id,
                provider,
                team_id,
                external_id,
                kind,
                actor_id,
                version,
                created_at,
                updated_at
            )
            SELECT DISTINCT ON (team_id, channel_id)
                'slack-' || substr(
                    encode(sha256(convert_to(team_id || chr(31) || channel_id, 'UTF8')), 'hex'),
                    1,
                    56
                ),
                'slack',
                team_id,
                channel_id,
                CASE conversation_kind
                    WHEN 'ordinary_internal' THEN 'channel'
                    WHEN 'mpim' THEN 'group_dm'
                    ELSE conversation_kind
                END,
                CASE WHEN conversation_kind = 'dm' THEN user_id ELSE NULL END,
                1,
                received_at,
                updated_at
            FROM public.slack_ingress_events
            ORDER BY team_id, channel_id, received_at DESC, event_id DESC
            ON CONFLICT (provider, team_id, external_id) DO NOTHING
            """
        )
    )

    op.add_column(
        "slack_ingress_events",
        sa.Column("conversation_id", sa.String(64), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE public.slack_ingress_events AS ingress
            SET conversation_id = conversation.id
            FROM public.conversations AS conversation
            WHERE conversation.provider = 'slack'
              AND conversation.team_id = ingress.team_id
              AND conversation.external_id = ingress.channel_id
            """
        )
    )
    op.alter_column("slack_ingress_events", "conversation_id", nullable=False)
    op.create_foreign_key(
        "fk_slack_ingress_conversation",
        "slack_ingress_events",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_index(
        "ix_slack_ingress_conversation_id",
        "slack_ingress_events",
        ["conversation_id", "received_at"],
    )

    op.add_column("threads", sa.Column("conversation_id", sa.String(64), nullable=True))
    op.create_foreign_key(
        "fk_threads_conversation",
        "threads",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.execute(
        sa.text(
            """
            UPDATE public.threads AS thread
            SET conversation_id = ingress.conversation_id
            FROM public.tasks AS task
            JOIN public.slack_ingress_events AS ingress ON ingress.task_id = task.id
            WHERE task.thread_id = thread.id
              AND thread.origin_provider = 'slack'
            """
        )
    )
    op.create_index(
        "ix_threads_conversation",
        "threads",
        ["conversation_id", "created_at"],
    )

    op.add_column(
        "conversation_threads",
        sa.Column("harness_thread_id", sa.String(64), nullable=True),
    )
    op.create_foreign_key(
        "fk_conversation_threads_harness_thread",
        "conversation_threads",
        "threads",
        ["harness_thread_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.execute(
        sa.text(
            """
            UPDATE public.conversation_threads AS conversation_thread
            SET harness_thread_id = task.thread_id
            FROM public.slack_ingress_events AS ingress
            JOIN public.tasks AS task ON task.id = ingress.task_id
            WHERE conversation_thread.conversation_id = ingress.conversation_id
              AND conversation_thread.root_ts = ingress.thread_root_ts
              AND conversation_thread.harness_thread_id IS NULL
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO public.conversation_threads (
                id,
                conversation_id,
                harness_thread_id,
                root_ts,
                organization_id,
                strategy_id,
                mapping_version,
                version,
                created_at,
                updated_at
            )
            SELECT DISTINCT ON (ingress.conversation_id, ingress.thread_root_ts)
                'conversation-thread-' || substr(
                    encode(
                        sha256(
                            convert_to(
                                ingress.conversation_id || chr(31) || ingress.thread_root_ts,
                                'UTF8'
                            )
                        ),
                        'hex'
                    ),
                    1,
                    44
                ),
                ingress.conversation_id,
                task.thread_id,
                ingress.thread_root_ts,
                ingress.organization_id,
                ingress.strategy_id,
                coalesce(ingress.mapping_version, 1),
                1,
                ingress.received_at,
                ingress.updated_at
            FROM public.slack_ingress_events AS ingress
            JOIN public.tasks AS task ON task.id = ingress.task_id
            JOIN public.organizations AS organization
              ON organization.id = ingress.organization_id
            JOIN public.strategies AS strategy
              ON strategy.id = ingress.strategy_id
             AND strategy.organization_id = ingress.organization_id
            WHERE ingress.conversation_id IS NOT NULL
              AND ingress.organization_id IS NOT NULL
              AND ingress.strategy_id IS NOT NULL
            ORDER BY
                ingress.conversation_id,
                ingress.thread_root_ts,
                ingress.received_at,
                ingress.event_id
            ON CONFLICT (conversation_id, root_ts) DO UPDATE
            SET harness_thread_id = EXCLUDED.harness_thread_id,
                updated_at = EXCLUDED.updated_at
            """
        )
    )
    op.create_unique_constraint(
        "uq_conversation_threads_harness_thread",
        "conversation_threads",
        ["harness_thread_id"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_conversation_threads_harness_thread",
        "conversation_threads",
        type_="unique",
    )
    op.drop_constraint(
        "fk_conversation_threads_harness_thread",
        "conversation_threads",
        type_="foreignkey",
    )
    op.drop_column("conversation_threads", "harness_thread_id")

    op.drop_index("ix_threads_conversation", table_name="threads")
    op.drop_constraint("fk_threads_conversation", "threads", type_="foreignkey")
    op.drop_column("threads", "conversation_id")

    op.drop_index("ix_slack_ingress_conversation_id", table_name="slack_ingress_events")
    op.drop_constraint(
        "fk_slack_ingress_conversation",
        "slack_ingress_events",
        type_="foreignkey",
    )
    op.drop_column("slack_ingress_events", "conversation_id")
