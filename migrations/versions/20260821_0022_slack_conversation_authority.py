"""Persist Slack ConversationRef and ingress authority provenance."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0022"
down_revision: str | None = "20260821_0021"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "conversations",
        sa.Column(
            "authority_source",
            sa.String(32),
            nullable=False,
            server_default="historical_snapshot",
        ),
    )
    op.add_column(
        "conversations",
        sa.Column("bot_presence", sa.String(16), nullable=False, server_default="present"),
    )
    op.add_column(
        "conversations",
        sa.Column("lifecycle", sa.String(16), nullable=False, server_default="active"),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "external_provenance",
            sa.String(24),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "conversations",
        sa.Column(
            "membership_policy_version",
            sa.Integer,
            nullable=False,
            server_default="1",
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE public.conversations
            SET external_provenance = CASE kind
                WHEN 'channel' THEN 'internal'
                WHEN 'shared' THEN 'shared'
                WHEN 'external' THEN 'external'
                WHEN 'dm' THEN 'not_applicable'
                WHEN 'group_dm' THEN 'not_applicable'
                ELSE 'unknown'
            END
            """
        )
    )
    op.create_check_constraint(
        "ck_conversations_authority_source",
        "conversations",
        "authority_source IN ('slack_conversations_info', 'slack_event', 'historical_snapshot')",
    )
    op.create_check_constraint(
        "ck_conversations_bot_presence",
        "conversations",
        "bot_presence IN ('present', 'absent', 'unknown')",
    )
    op.create_check_constraint(
        "ck_conversations_lifecycle",
        "conversations",
        "lifecycle IN ('active', 'archived', 'left', 'unknown')",
    )
    op.create_check_constraint(
        "ck_conversations_external_provenance",
        "conversations",
        "external_provenance IN ('internal', 'shared', 'external', 'not_applicable', 'unknown')",
    )
    op.create_check_constraint(
        "ck_conversations_membership_policy_version",
        "conversations",
        "membership_policy_version >= 1",
    )
    op.create_index(
        "ix_conversations_team_authority",
        "conversations",
        ["team_id", "bot_presence", "lifecycle", "updated_at"],
    )

    op.add_column(
        "slack_ingress_events",
        sa.Column(
            "conversation_authority_source",
            sa.String(32),
            nullable=False,
            server_default="slack_event",
        ),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column("bot_presence", sa.String(16), nullable=False, server_default="present"),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column(
            "conversation_lifecycle",
            sa.String(16),
            nullable=False,
            server_default="active",
        ),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column(
            "external_provenance",
            sa.String(24),
            nullable=False,
            server_default="unknown",
        ),
    )
    op.add_column(
        "slack_ingress_events",
        sa.Column(
            "membership_policy_version",
            sa.Integer,
            nullable=False,
            server_default="1",
        ),
    )
    op.execute(
        sa.text(
            """
            UPDATE public.slack_ingress_events
            SET external_provenance = CASE conversation_kind
                WHEN 'ordinary_internal' THEN 'internal'
                WHEN 'shared' THEN 'shared'
                WHEN 'external' THEN 'external'
                WHEN 'dm' THEN 'not_applicable'
                WHEN 'mpim' THEN 'not_applicable'
                ELSE 'unknown'
            END
            """
        )
    )
    op.create_check_constraint(
        "ck_slack_ingress_conversation_authority_source",
        "slack_ingress_events",
        "conversation_authority_source IN ('slack_conversations_info', 'slack_event')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_bot_presence",
        "slack_ingress_events",
        "bot_presence IN ('present', 'absent', 'unknown')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_conversation_lifecycle",
        "slack_ingress_events",
        "conversation_lifecycle IN ('active', 'archived', 'left', 'unknown')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_external_provenance",
        "slack_ingress_events",
        "external_provenance IN ('internal', 'shared', 'external', 'not_applicable', 'unknown')",
    )
    op.create_check_constraint(
        "ck_slack_ingress_membership_policy_version",
        "slack_ingress_events",
        "membership_policy_version >= 1",
    )


def downgrade() -> None:
    for constraint in (
        "ck_slack_ingress_membership_policy_version",
        "ck_slack_ingress_external_provenance",
        "ck_slack_ingress_conversation_lifecycle",
        "ck_slack_ingress_bot_presence",
        "ck_slack_ingress_conversation_authority_source",
    ):
        op.drop_constraint(constraint, "slack_ingress_events", type_="check")
    for column in (
        "membership_policy_version",
        "external_provenance",
        "conversation_lifecycle",
        "bot_presence",
        "conversation_authority_source",
    ):
        op.drop_column("slack_ingress_events", column)

    op.drop_index("ix_conversations_team_authority", table_name="conversations")
    for constraint in (
        "ck_conversations_membership_policy_version",
        "ck_conversations_external_provenance",
        "ck_conversations_lifecycle",
        "ck_conversations_bot_presence",
        "ck_conversations_authority_source",
    ):
        op.drop_constraint(constraint, "conversations", type_="check")
    for column in (
        "membership_policy_version",
        "external_provenance",
        "lifecycle",
        "bot_presence",
        "authority_source",
    ):
        op.drop_column("conversations", column)
