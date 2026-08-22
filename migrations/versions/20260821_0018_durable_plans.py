"""Add bounded durable plan, revision, node, and delegation journals."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260821_0018"
down_revision: str | None = "20260821_0017"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PLAN_TABLES = ("plans", "plan_revisions", "plan_nodes", "delegations")
_MEMORY_CONSTRAINTS = {
    "memory_records": "ck_memory_records_visibility",
    "memory_sources": "ck_memory_sources_visibility",
    "memory_revisions": "ck_memory_revisions_visibility",
}
_OLD_VISIBILITY = (
    "'thread_local', 'channel_local', 'actor_private', 'strategy_shared', 'organization_shared'"
)
_NEW_VISIBILITY = (
    "'thread_local', 'conversation_local', 'channel_local', 'actor_private', "
    "'strategy_shared', 'organization_shared'"
)


def _protect(table: str) -> None:
    op.execute(sa.text(f"REVOKE ALL PRIVILEGES ON TABLE public.{table} FROM anon, authenticated"))
    op.execute(sa.text(f"ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            f"CREATE POLICY leo_client_deny ON public.{table} AS RESTRICTIVE "
            "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)"
        )
    )


def _replace_memory_visibility(values: str) -> None:
    for table, constraint in _MEMORY_CONSTRAINTS.items():
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(constraint, table, f"visibility IN ({values})")


def upgrade() -> None:
    _replace_memory_visibility(_NEW_VISIBILITY)

    op.create_table(
        "plans",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("strategy_id", sa.String(64), nullable=False),
        sa.Column(
            "parent_task_id",
            sa.String(64),
            sa.ForeignKey("tasks.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "parent_run_id",
            sa.String(64),
            sa.ForeignKey("runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("initial_digest", sa.String(64), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="active"),
        sa.Column("current_revision", sa.Integer, nullable=False, server_default="1"),
        sa.Column("max_revisions", sa.Integer, nullable=False, server_default="4"),
        sa.Column("output", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("version", sa.Integer, nullable=False, server_default="1"),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint(
            "organization_id", "idempotency_key", name="uq_plans_org_idempotency_key"
        ),
        sa.CheckConstraint("status IN ('active', 'completed', 'failed')", name="ck_plans_status"),
        sa.CheckConstraint(
            "current_revision >= 1 AND current_revision <= max_revisions",
            name="ck_plans_current_revision",
        ),
        sa.CheckConstraint(
            "max_revisions >= 1 AND max_revisions <= 8", name="ck_plans_max_revisions"
        ),
        sa.CheckConstraint("version >= 1", name="ck_plans_version"),
        sa.CheckConstraint("initial_digest ~ '^[0-9a-f]{64}$'", name="ck_plans_initial_digest"),
        sa.CheckConstraint("char_length(idempotency_key) >= 1", name="ck_plans_idempotency_key"),
        sa.CheckConstraint(
            "(status = 'active' AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND output IS NOT NULL AND error IS NULL) OR "
            "(status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name="ck_plans_terminal_result",
        ),
    )
    op.create_index("ix_plans_parent", "plans", ["parent_task_id", "parent_run_id"])
    op.create_index("ix_plans_scope_status", "plans", ["organization_id", "status"])

    op.create_table(
        "plan_revisions",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "plan_id",
            sa.String(64),
            sa.ForeignKey("plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("number", sa.Integer, nullable=False),
        sa.Column(
            "parent_revision_id",
            sa.String(64),
            sa.ForeignKey("plan_revisions.id", ondelete="RESTRICT"),
        ),
        sa.Column("parent_digest", sa.String(64)),
        sa.Column("digest", sa.String(64), nullable=False),
        sa.Column("goal", sa.Text, nullable=False),
        sa.Column("definition", postgresql.JSONB, nullable=False),
        sa.Column("reason", sa.String(1000), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("plan_id", "number", name="uq_plan_revisions_number"),
        sa.UniqueConstraint("plan_id", "digest", name="uq_plan_revisions_digest"),
        sa.CheckConstraint("number >= 1 AND number <= 8", name="ck_plan_revisions_number"),
        sa.CheckConstraint(
            "digest ~ '^[0-9a-f]{64}$' AND "
            "(parent_digest IS NULL OR parent_digest ~ '^[0-9a-f]{64}$')",
            name="ck_plan_revisions_digests",
        ),
        sa.CheckConstraint(
            "(number = 1 AND parent_revision_id IS NULL AND parent_digest IS NULL) OR "
            "(number > 1 AND parent_revision_id IS NOT NULL AND parent_digest IS NOT NULL)",
            name="ck_plan_revisions_parent",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(definition) = 'array' AND jsonb_array_length(definition) >= 1 "
            "AND jsonb_array_length(definition) <= 64",
            name="ck_plan_revisions_definition",
        ),
    )
    op.create_index(
        "ix_plan_revisions_scope",
        "plan_revisions",
        ["organization_id", "plan_id", "number"],
    )

    op.create_table(
        "plan_nodes",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "plan_id",
            sa.String(64),
            sa.ForeignKey("plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "revision_id",
            sa.String(64),
            sa.ForeignKey("plan_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column("revision_number", sa.Integer, nullable=False),
        sa.Column("node_key", sa.String(64), nullable=False),
        sa.Column("objective", sa.Text, nullable=False),
        sa.Column("depends_on", postgresql.JSONB, nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="pending"),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.Integer, nullable=False, server_default="3"),
        sa.Column("claim_owner", sa.String(128)),
        sa.Column("claim_token", sa.String(128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("child_task_id", sa.String(64), sa.ForeignKey("tasks.id", ondelete="RESTRICT")),
        sa.Column("child_run_id", sa.String(64), sa.ForeignKey("runs.id", ondelete="RESTRICT")),
        sa.Column("output", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.UniqueConstraint("revision_id", "node_key", name="uq_plan_nodes_revision_key"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_plan_nodes_status",
        ),
        sa.CheckConstraint(
            "attempt >= 0 AND max_attempts >= 1 AND max_attempts <= 8 AND attempt <= max_attempts",
            name="ck_plan_nodes_attempt",
        ),
        sa.CheckConstraint(
            "jsonb_typeof(depends_on) = 'array' AND jsonb_array_length(depends_on) <= 16",
            name="ck_plan_nodes_dependencies",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND claim_owner IS NOT NULL AND claim_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND attempt >= 1) OR "
            "(status <> 'running' AND claim_owner IS NULL AND claim_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_plan_nodes_claim_state",
        ),
        sa.CheckConstraint(
            "(child_task_id IS NULL AND child_run_id IS NULL) OR "
            "(child_task_id IS NOT NULL AND child_run_id IS NOT NULL)",
            name="ck_plan_nodes_child_identity",
        ),
        sa.CheckConstraint(
            "(status IN ('pending', 'running') AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND output IS NOT NULL AND error IS NULL) OR "
            "(status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name="ck_plan_nodes_result",
        ),
    )
    op.create_index(
        "ix_plan_nodes_plan_revision",
        "plan_nodes",
        ["plan_id", "revision_id", "node_key"],
    )
    op.create_index(
        "ix_plan_nodes_claim_eligibility",
        "plan_nodes",
        ["plan_id", "revision_number", "status", "lease_expires_at", "node_key"],
    )

    op.create_table(
        "delegations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column(
            "plan_id",
            sa.String(64),
            sa.ForeignKey("plans.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "revision_id",
            sa.String(64),
            sa.ForeignKey("plan_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "node_id",
            sa.String(64),
            sa.ForeignKey("plan_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("organization_id", sa.String(64), nullable=False),
        sa.Column(
            "parent_task_id",
            sa.String(64),
            sa.ForeignKey("tasks.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "parent_run_id",
            sa.String(64),
            sa.ForeignKey("runs.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("attempt", sa.Integer, nullable=False),
        sa.Column("owner", sa.String(128), nullable=False),
        sa.Column("claim_token", sa.String(128), nullable=False),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("child_task_id", sa.String(64), sa.ForeignKey("tasks.id", ondelete="RESTRICT")),
        sa.Column("child_run_id", sa.String(64), sa.ForeignKey("runs.id", ondelete="RESTRICT")),
        sa.Column("output", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("node_id", "attempt", name="uq_delegations_node_attempt"),
        sa.UniqueConstraint("node_id", "claim_token", name="uq_delegations_node_claim_token"),
        sa.CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'superseded')",
            name="ck_delegations_status",
        ),
        sa.CheckConstraint("attempt >= 1 AND attempt <= 8", name="ck_delegations_attempt"),
        sa.CheckConstraint(
            "(child_task_id IS NULL AND child_run_id IS NULL) OR "
            "(child_task_id IS NOT NULL AND child_run_id IS NOT NULL)",
            name="ck_delegations_child_identity",
        ),
        sa.CheckConstraint(
            "(status = 'running' AND finished_at IS NULL AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND finished_at IS NOT NULL AND output IS NOT NULL "
            "AND error IS NULL) OR "
            "(status IN ('failed', 'superseded') AND finished_at IS NOT NULL "
            "AND output IS NULL AND error IS NOT NULL)",
            name="ck_delegations_result",
        ),
    )
    op.create_index(
        "ix_delegations_plan_revision",
        "delegations",
        ["plan_id", "revision_id", "node_id"],
    )
    op.create_index("ix_delegations_parent", "delegations", ["parent_task_id", "parent_run_id"])
    op.create_index("ix_delegations_scope_status", "delegations", ["organization_id", "status"])

    for table in _PLAN_TABLES:
        _protect(table)


def downgrade() -> None:
    for table in reversed(_PLAN_TABLES):
        op.execute(sa.text(f"DROP POLICY IF EXISTS leo_client_deny ON public.{table}"))
        op.execute(sa.text(f"ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))

    op.drop_index("ix_delegations_scope_status", table_name="delegations")
    op.drop_index("ix_delegations_parent", table_name="delegations")
    op.drop_index("ix_delegations_plan_revision", table_name="delegations")
    op.drop_table("delegations")
    op.drop_index("ix_plan_nodes_claim_eligibility", table_name="plan_nodes")
    op.drop_index("ix_plan_nodes_plan_revision", table_name="plan_nodes")
    op.drop_table("plan_nodes")
    op.drop_index("ix_plan_revisions_scope", table_name="plan_revisions")
    op.drop_table("plan_revisions")
    op.drop_index("ix_plans_scope_status", table_name="plans")
    op.drop_index("ix_plans_parent", table_name="plans")
    op.drop_table("plans")

    # The compatibility alias preserves data when rolling back the constraint vocabulary.
    for table, constraint in _MEMORY_CONSTRAINTS.items():
        op.drop_constraint(constraint, table, type_="check")
        op.execute(
            sa.text(
                f"UPDATE public.{table} SET visibility = 'channel_local' "
                "WHERE visibility = 'conversation_local'"
            )
        )
        op.create_check_constraint(constraint, table, f"visibility IN ({_OLD_VISIBILITY})")
