"""Harden public Leo tables against Supabase client access.

Revision ID: 20260821_0003
Revises: 20260820_0002
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0003"
down_revision: str | None = "20260820_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_LEO_TABLES = (
    "threads",
    "tasks",
    "runs",
    "observations",
    "claims",
    "run_events",
    "slack_ingress_events",
    "slack_channel_scopes",
    "alembic_version",
)


def _table_sql(table: str, statement: str) -> str:
    return statement.format(table=table)


def upgrade() -> None:
    op.create_index("ix_tasks_thread_id", "tasks", ["thread_id"])
    op.create_index(
        "ix_slack_ingress_events_task_id",
        "slack_ingress_events",
        ["task_id"],
    )

    for table in _LEO_TABLES:
        op.execute(
            sa.text(
                _table_sql(
                    table,
                    "REVOKE ALL PRIVILEGES ON TABLE public.{table} FROM anon, authenticated",
                )
            )
        )
        op.execute(
            sa.text(_table_sql(table, "ALTER TABLE public.{table} ENABLE ROW LEVEL SECURITY"))
        )
        op.execute(
            sa.text(
                _table_sql(
                    table,
                    "CREATE POLICY leo_client_deny ON public.{table} AS RESTRICTIVE "
                    "FOR ALL TO anon, authenticated USING (false) WITH CHECK (false)",
                )
            )
        )


def downgrade() -> None:
    # Downgrade removes only this revision's objects. It deliberately does not restore
    # client privileges, so an operational rollback cannot silently re-expose the demo.
    for table in reversed(_LEO_TABLES):
        op.execute(
            sa.text(
                _table_sql(
                    table,
                    "DROP POLICY IF EXISTS leo_client_deny ON public.{table}",
                )
            )
        )
        op.execute(
            sa.text(_table_sql(table, "ALTER TABLE public.{table} DISABLE ROW LEVEL SECURITY"))
        )

    op.drop_index("ix_slack_ingress_events_task_id", table_name="slack_ingress_events")
    op.drop_index("ix_tasks_thread_id", table_name="tasks")
