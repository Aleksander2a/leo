"""Initial trusted ingress and harness state schema.

Revision ID: 20260820_0001
Revises: None
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20260820_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "threads",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("origin_provider", sa.String(length=32), nullable=False),
        sa.Column("external_thread_id", sa.String(length=255), nullable=False),
        sa.Column("external_event_id", sa.String(length=64)),
        sa.Column("external_channel_id", sa.String(length=64)),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.UniqueConstraint("origin_provider", "external_thread_id", name="uq_thread_origin"),
    )
    op.create_index("ix_threads_scope", "threads", ["organization_id", "strategy_id"])

    op.create_table(
        "tasks",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("thread_id", sa.String(length=64), sa.ForeignKey("threads.id"), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("objective", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("observation_ids", postgresql.JSONB(), nullable=False),
        sa.Column("verifier_feedback", postgresql.JSONB(), nullable=False),
        sa.Column("final_output", sa.Text()),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_tasks_scope_status", "tasks", ["organization_id", "strategy_id", "status"])

    op.create_table(
        "runs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("phase", sa.String(length=32), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("limits", postgresql.JSONB(), nullable=False),
        sa.Column("usage", postgresql.JSONB(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("final_output", sa.Text()),
        sa.Column("terminal_reason", sa.String(length=255)),
        sa.Column("event_sequence", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index("ix_runs_task_status", "runs", ["task_id", "status"])

    op.create_table(
        "observations",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("tool_call_id", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=128), nullable=False),
        sa.Column("data", postgresql.JSONB(), nullable=False),
        sa.Column("source", postgresql.JSONB(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("raw_hash", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_observations_run", "observations", ["run_id"])
    op.create_index("ix_observations_scope", "observations", ["organization_id", "strategy_id"])

    op.create_table(
        "claims",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("observation_ids", postgresql.JSONB(), nullable=False),
    )
    op.create_index("ix_claims_run", "claims", ["run_id"])
    op.create_index("ix_claims_scope", "claims", ["organization_id", "strategy_id"])

    op.create_table(
        "run_events",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("run_id", sa.String(length=64), sa.ForeignKey("runs.id"), nullable=False),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("tasks.id"), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("type", sa.String(length=64), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("iteration", sa.Integer(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),
    )
    op.create_index("ix_run_events_task", "run_events", ["task_id"])

    op.create_table(
        "slack_ingress_events",
        sa.Column("event_id", sa.String(length=64), primary_key=True),
        sa.Column("team_id", sa.String(length=32), nullable=False),
        sa.Column("channel_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.String(length=32), nullable=False),
        sa.Column("message_ts", sa.String(length=32), nullable=False),
        sa.Column("thread_root_ts", sa.String(length=32), nullable=False),
        sa.Column("conversation_key", sa.String(length=255), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("tasks.id")),
        sa.Column("lease_owner", sa.String(length=128)),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("last_error", sa.Text()),
        sa.Column(
            "received_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_slack_ingress_thread",
        "slack_ingress_events",
        ["team_id", "channel_id", "thread_root_ts"],
    )


def downgrade() -> None:
    op.drop_index("ix_slack_ingress_thread", table_name="slack_ingress_events")
    op.drop_table("slack_ingress_events")
    op.drop_index("ix_run_events_task", table_name="run_events")
    op.drop_table("run_events")
    op.drop_index("ix_claims_scope", table_name="claims")
    op.drop_index("ix_claims_run", table_name="claims")
    op.drop_table("claims")
    op.drop_index("ix_observations_scope", table_name="observations")
    op.drop_index("ix_observations_run", table_name="observations")
    op.drop_table("observations")
    op.drop_index("ix_runs_task_status", table_name="runs")
    op.drop_table("runs")
    op.drop_index("ix_tasks_scope_status", table_name="tasks")
    op.drop_table("tasks")
    op.drop_index("ix_threads_scope", table_name="threads")
    op.drop_table("threads")
