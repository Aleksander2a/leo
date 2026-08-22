"""Add covering indexes for durable-plan foreign keys."""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "20260821_0020"
down_revision: str | None = "20260821_0019"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_plans_parent_run", "plans", ["parent_run_id"])
    op.create_index(
        "ix_plan_revisions_parent_revision",
        "plan_revisions",
        ["parent_revision_id"],
    )
    op.create_index("ix_plan_nodes_child_task", "plan_nodes", ["child_task_id"])
    op.create_index("ix_plan_nodes_child_run", "plan_nodes", ["child_run_id"])
    op.create_index("ix_delegations_revision", "delegations", ["revision_id"])
    op.create_index("ix_delegations_parent_run", "delegations", ["parent_run_id"])
    op.create_index("ix_delegations_child_task", "delegations", ["child_task_id"])
    op.create_index("ix_delegations_child_run", "delegations", ["child_run_id"])


def downgrade() -> None:
    op.drop_index("ix_delegations_child_run", table_name="delegations")
    op.drop_index("ix_delegations_child_task", table_name="delegations")
    op.drop_index("ix_delegations_parent_run", table_name="delegations")
    op.drop_index("ix_delegations_revision", table_name="delegations")
    op.drop_index("ix_plan_nodes_child_run", table_name="plan_nodes")
    op.drop_index("ix_plan_nodes_child_task", table_name="plan_nodes")
    op.drop_index("ix_plan_revisions_parent_revision", table_name="plan_revisions")
    op.drop_index("ix_plans_parent_run", table_name="plans")
