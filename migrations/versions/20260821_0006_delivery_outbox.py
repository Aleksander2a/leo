"""Add the durable Slack delivery outbox."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0006"
down_revision: str | None = "20260821_0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "delivery_outbox",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("ingress_event_id", sa.String(length=64), nullable=False),
        sa.Column("organization_id", sa.String(length=64), nullable=False),
        sa.Column("strategy_id", sa.String(length=64), nullable=False),
        sa.Column("destination_channel_id", sa.String(length=64), nullable=False),
        sa.Column("destination_thread_ts", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("payload_version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("payload_hash", sa.String(length=64), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("state", sa.String(length=24), nullable=False, server_default="pending"),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_token", sa.String(length=128), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_after", sa.DateTime(timezone=True), nullable=True),
        sa.Column("receipt_message_ts", sa.String(length=64), nullable=True),
        sa.Column("last_error", sa.String(length=255), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(["task_id"], ["tasks.id"]),
        sa.ForeignKeyConstraint(["run_id"], ["runs.id"]),
        sa.ForeignKeyConstraint(["ingress_event_id"], ["slack_ingress_events.event_id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "task_id",
            "run_id",
            "kind",
            "payload_version",
            name="uq_delivery_outbox_logical_key",
        ),
        sa.CheckConstraint(
            "kind IN ('progress', 'final')",
            name="ck_delivery_outbox_kind",
        ),
        sa.CheckConstraint(
            "state IN ('pending', 'leased', 'retry', 'delivered', 'dead', 'unknown_effect')",
            name="ck_delivery_outbox_state",
        ),
        sa.CheckConstraint("payload_version >= 1", name="ck_delivery_outbox_payload_version"),
        sa.CheckConstraint("attempt_count >= 0", name="ck_delivery_outbox_attempt_count"),
        sa.CheckConstraint(
            "(state = 'leased' AND lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL) OR "
            "(state <> 'leased' AND lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_delivery_outbox_lease_state",
        ),
        sa.CheckConstraint(
            "state NOT IN ('delivered', 'dead', 'unknown_effect') OR "
            "(lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)",
            name="ck_delivery_outbox_terminal_no_lease",
        ),
    )
    op.create_index(
        "ix_delivery_outbox_claim_eligibility",
        "delivery_outbox",
        ["state", "retry_after", "lease_expires_at", "created_at", "id"],
    )
    op.create_index(
        "ix_delivery_outbox_scope",
        "delivery_outbox",
        ["organization_id", "strategy_id", "state"],
    )
    op.create_index("ix_delivery_outbox_task", "delivery_outbox", ["task_id"])
    op.create_index("ix_delivery_outbox_run", "delivery_outbox", ["run_id"])
    op.create_index("ix_delivery_outbox_ingress", "delivery_outbox", ["ingress_event_id"])
    op.execute(
        sa.text("REVOKE ALL PRIVILEGES ON TABLE public.delivery_outbox FROM anon, authenticated")
    )
    op.execute(sa.text("ALTER TABLE public.delivery_outbox ENABLE ROW LEVEL SECURITY"))
    op.execute(
        sa.text(
            "CREATE POLICY leo_client_deny ON public.delivery_outbox "
            "AS RESTRICTIVE FOR ALL TO anon, authenticated "
            "USING (false) WITH CHECK (false)"
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP POLICY IF EXISTS leo_client_deny ON public.delivery_outbox"))
    op.execute(sa.text("ALTER TABLE public.delivery_outbox DISABLE ROW LEVEL SECURITY"))
    op.drop_index("ix_delivery_outbox_ingress", table_name="delivery_outbox")
    op.drop_index("ix_delivery_outbox_run", table_name="delivery_outbox")
    op.drop_index("ix_delivery_outbox_task", table_name="delivery_outbox")
    op.drop_index("ix_delivery_outbox_scope", table_name="delivery_outbox")
    op.drop_index("ix_delivery_outbox_claim_eligibility", table_name="delivery_outbox")
    op.drop_table("delivery_outbox")
