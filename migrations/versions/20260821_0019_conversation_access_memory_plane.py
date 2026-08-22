"""Persist conversation access snapshots and the sanitized Slack message plane."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0019"
down_revision: str | None = "20260821_0018"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ACCESS_TABLES = ("conversation_access_snapshots", "conversation_actor_memberships")


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
    op.add_column(
        "slack_ingress_events",
        sa.Column(
            "context_projection_source",
            sa.String(32),
            nullable=True,
            server_default="exact_destination",
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE public.slack_ingress_events
            SET context_projection_source = CASE
                WHEN conversation_kind = 'dm' THEN 'dm_only_fallback'
                ELSE 'exact_destination'
            END
            WHERE context_projection_source IS NULL
               OR (conversation_kind = 'dm' AND context_projection_source = 'exact_destination')
            """
        )
    )
    op.alter_column("slack_ingress_events", "context_projection_source", nullable=False)
    op.create_check_constraint(
        "ck_slack_ingress_context_projection_source",
        "slack_ingress_events",
        "context_projection_source IN "
        "('exact_destination', 'dm_membership_intersection', 'dm_only_fallback')",
    )

    op.create_table(
        "conversation_access_snapshots",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "ingress_event_id",
            sa.String(64),
            sa.ForeignKey("slack_ingress_events.event_id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("team_id", sa.String(32), nullable=False),
        sa.Column("actor_id", sa.String(32), nullable=False),
        sa.Column("destination_external_id", sa.String(32), nullable=False),
        sa.Column("conversation_external_id", sa.String(32), nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("context_access_hash", sa.String(64), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "ingress_event_id",
            "conversation_external_id",
            name="uq_conversation_access_snapshot_source",
        ),
        sa.CheckConstraint("position >= 0", name="ck_conversation_access_snapshot_position"),
        sa.CheckConstraint(
            "source_kind IN ('exact_destination', 'dm_membership_intersection', "
            "'dm_only_fallback', 'historical_snapshot')",
            name="ck_conversation_access_snapshot_source_kind",
        ),
        sa.CheckConstraint(
            "context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_conversation_access_snapshot_hash",
        ),
    )
    op.create_index(
        "ix_conversation_access_snapshot_actor",
        "conversation_access_snapshots",
        ["team_id", "actor_id", "observed_at"],
    )
    op.create_index(
        "ix_conversation_access_snapshot_source",
        "conversation_access_snapshots",
        ["team_id", "conversation_external_id", "observed_at"],
    )

    op.create_table(
        "conversation_actor_memberships",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("team_id", sa.String(32), nullable=False),
        sa.Column("actor_id", sa.String(32), nullable=False),
        sa.Column("conversation_external_id", sa.String(32), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("source_kind", sa.String(32), nullable=False),
        sa.Column("context_access_hash", sa.String(64), nullable=False),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "team_id",
            "actor_id",
            "conversation_external_id",
            name="uq_conversation_actor_membership",
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked')",
            name="ck_conversation_actor_membership_status",
        ),
        sa.CheckConstraint("version >= 1", name="ck_conversation_actor_membership_version"),
        sa.CheckConstraint(
            "context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_conversation_actor_membership_hash",
        ),
    )
    op.create_index(
        "ix_conversation_actor_membership_actor",
        "conversation_actor_memberships",
        ["team_id", "actor_id", "status", "observed_at"],
    )
    op.create_index(
        "ix_conversation_actor_membership_source",
        "conversation_actor_memberships",
        ["team_id", "conversation_external_id", "status"],
    )

    # Normalize every historical admission before using the mutable membership projection.
    op.execute(
        sa.text(
            """
            INSERT INTO public.conversation_access_snapshots (
                id,
                ingress_event_id,
                organization_id,
                team_id,
                actor_id,
                destination_external_id,
                conversation_external_id,
                position,
                source_kind,
                context_access_hash,
                observed_at
            )
            SELECT
                'access-' || substr(
                    encode(
                        sha256(
                            convert_to(
                                ingress.event_id || chr(31) || source.conversation_external_id,
                                'UTF8'
                            )
                        ),
                        'hex'
                    ),
                    1,
                    57
                ),
                ingress.event_id,
                coalesce(ingress.organization_id, 'demo-org'),
                ingress.team_id,
                ingress.user_id,
                ingress.channel_id,
                source.conversation_external_id,
                source.position - 1,
                CASE
                    WHEN ingress.context_projection_source = 'dm_membership_intersection'
                        THEN 'dm_membership_intersection'
                    WHEN ingress.context_projection_source = 'dm_only_fallback'
                        THEN 'dm_only_fallback'
                    ELSE 'exact_destination'
                END,
                ingress.context_access_hash,
                ingress.received_at
            FROM public.slack_ingress_events AS ingress
            CROSS JOIN LATERAL jsonb_array_elements_text(
                ingress.context_conversation_ids
            ) WITH ORDINALITY AS source(conversation_external_id, position)
            ON CONFLICT (ingress_event_id, conversation_external_id) DO NOTHING
            """
        )
    )
    op.execute(
        sa.text(
            """
            INSERT INTO public.conversation_actor_memberships (
                id,
                organization_id,
                team_id,
                actor_id,
                conversation_external_id,
                status,
                source_kind,
                context_access_hash,
                version,
                observed_at,
                created_at,
                updated_at
            )
            SELECT DISTINCT ON (
                snapshot.team_id,
                snapshot.actor_id,
                snapshot.conversation_external_id
            )
                'membership-' || substr(
                    encode(
                        sha256(
                            convert_to(
                                snapshot.team_id || chr(31) || snapshot.actor_id || chr(31)
                                    || snapshot.conversation_external_id,
                                'UTF8'
                            )
                        ),
                        'hex'
                    ),
                    1,
                    53
                ),
                snapshot.organization_id,
                snapshot.team_id,
                snapshot.actor_id,
                snapshot.conversation_external_id,
                'active',
                'historical_snapshot',
                snapshot.context_access_hash,
                1,
                snapshot.observed_at,
                snapshot.created_at,
                snapshot.created_at
            FROM public.conversation_access_snapshots AS snapshot
            ORDER BY
                snapshot.team_id,
                snapshot.actor_id,
                snapshot.conversation_external_id,
                snapshot.observed_at DESC,
                snapshot.ingress_event_id DESC
            ON CONFLICT (team_id, actor_id, conversation_external_id) DO NOTHING
            """
        )
    )

    # Conversation identity is now the authority boundary. The old domain FKs made the
    # sanitized transport plane depend on optional strategy catalog seed rows.
    op.drop_constraint(
        "sanitized_messages_organization_id_fkey", "sanitized_messages", type_="foreignkey"
    )
    op.drop_constraint(
        "sanitized_messages_strategy_id_fkey", "sanitized_messages", type_="foreignkey"
    )
    op.add_column("sanitized_messages", sa.Column("conversation_id", sa.String(64)))
    op.add_column("sanitized_messages", sa.Column("harness_thread_id", sa.String(64)))
    op.add_column("sanitized_messages", sa.Column("actor_id", sa.String(128)))
    op.add_column(
        "sanitized_messages",
        sa.Column("role", sa.String(16), nullable=False, server_default="user"),
    )
    op.add_column("sanitized_messages", sa.Column("provider_message_ts", sa.String(64)))
    op.add_column("sanitized_messages", sa.Column("context_access_hash", sa.String(64)))
    op.create_foreign_key(
        "fk_sanitized_messages_conversation",
        "sanitized_messages",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    op.create_foreign_key(
        "fk_sanitized_messages_harness_thread",
        "sanitized_messages",
        "threads",
        ["harness_thread_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        "ck_sanitized_messages_role",
        "sanitized_messages",
        "role IN ('user', 'assistant')",
    )
    op.create_check_constraint(
        "ck_sanitized_messages_context_access_hash",
        "sanitized_messages",
        "context_access_hash IS NULL OR context_access_hash ~ '^[0-9a-f]{64}$'",
    )
    op.create_index(
        "uq_sanitized_messages_conversation_event_role",
        "sanitized_messages",
        ["conversation_id", "external_event_id", "role"],
        unique=True,
        postgresql_where=sa.text("conversation_id IS NOT NULL"),
    )
    op.create_index(
        "ix_sanitized_messages_conversation_time",
        "sanitized_messages",
        ["conversation_id", "recorded_at"],
    )
    op.create_index(
        "ix_sanitized_messages_thread_time",
        "sanitized_messages",
        ["harness_thread_id", "recorded_at"],
    )

    for table in _ACCESS_TABLES:
        _protect(table)
    op.execute(
        sa.text("REVOKE ALL PRIVILEGES ON TABLE public.sanitized_messages FROM anon, authenticated")
    )
    op.execute(sa.text("ALTER TABLE public.sanitized_messages ENABLE ROW LEVEL SECURITY"))


def downgrade() -> None:
    op.drop_index("ix_sanitized_messages_thread_time", table_name="sanitized_messages")
    op.drop_index("ix_sanitized_messages_conversation_time", table_name="sanitized_messages")
    op.drop_index("uq_sanitized_messages_conversation_event_role", table_name="sanitized_messages")
    op.drop_constraint(
        "ck_sanitized_messages_context_access_hash", "sanitized_messages", type_="check"
    )
    op.drop_constraint("ck_sanitized_messages_role", "sanitized_messages", type_="check")
    op.drop_constraint(
        "fk_sanitized_messages_harness_thread", "sanitized_messages", type_="foreignkey"
    )
    op.drop_constraint(
        "fk_sanitized_messages_conversation", "sanitized_messages", type_="foreignkey"
    )
    for column in (
        "context_access_hash",
        "provider_message_ts",
        "role",
        "actor_id",
        "harness_thread_id",
        "conversation_id",
    ):
        op.drop_column("sanitized_messages", column)
    op.execute(
        sa.text(
            """
            DELETE FROM public.sanitized_messages AS message
            WHERE NOT EXISTS (
                    SELECT 1 FROM public.organizations AS organization
                    WHERE organization.id = message.organization_id
                )
               OR NOT EXISTS (
                    SELECT 1 FROM public.strategies AS strategy
                    WHERE strategy.id = message.strategy_id
                )
            """
        )
    )
    op.create_foreign_key(
        "sanitized_messages_strategy_id_fkey",
        "sanitized_messages",
        "strategies",
        ["strategy_id"],
        ["id"],
    )
    op.create_foreign_key(
        "sanitized_messages_organization_id_fkey",
        "sanitized_messages",
        "organizations",
        ["organization_id"],
        ["id"],
    )

    for table in reversed(_ACCESS_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))
    op.drop_index(
        "ix_conversation_actor_membership_source",
        table_name="conversation_actor_memberships",
    )
    op.drop_index(
        "ix_conversation_actor_membership_actor",
        table_name="conversation_actor_memberships",
    )
    op.drop_table("conversation_actor_memberships")
    op.drop_index(
        "ix_conversation_access_snapshot_source",
        table_name="conversation_access_snapshots",
    )
    op.drop_index(
        "ix_conversation_access_snapshot_actor",
        table_name="conversation_access_snapshots",
    )
    op.drop_table("conversation_access_snapshots")

    op.drop_constraint(
        "ck_slack_ingress_context_projection_source",
        "slack_ingress_events",
        type_="check",
    )
    op.drop_column("slack_ingress_events", "context_projection_source")
