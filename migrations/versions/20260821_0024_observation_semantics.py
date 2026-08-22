"""Persist explicit observation status, quality, and schema provenance."""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "20260821_0024"
down_revision: str | None = "20260821_0023"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("observations", sa.Column("status", sa.String(24), nullable=True))
    op.add_column("observations", sa.Column("quality", sa.String(32), nullable=True))
    op.add_column("observations", sa.Column("schema_version", sa.String(32), nullable=True))
    op.add_column("observations", sa.Column("normalization_version", sa.String(32), nullable=True))
    op.add_column("observations", sa.Column("rejection_code", sa.String(64), nullable=True))
    op.execute(
        sa.text(
            """
            UPDATE public.observations
            SET status = 'retrieved',
                quality = 'provider_reported',
                schema_version = 'observation-v1',
                normalization_version = 'legacy-v1'
            WHERE status IS NULL
            """
        )
    )
    op.alter_column("observations", "status", nullable=False, server_default=sa.text("'retrieved'"))
    op.alter_column(
        "observations",
        "quality",
        nullable=False,
        server_default=sa.text("'provider_reported'"),
    )
    op.alter_column(
        "observations",
        "schema_version",
        nullable=False,
        server_default=sa.text("'observation-v2'"),
    )
    op.alter_column(
        "observations",
        "normalization_version",
        nullable=False,
        server_default=sa.text("'normalization-v1'"),
    )
    op.create_check_constraint(
        "ck_observations_status",
        "observations",
        "status IN ('retrieved', 'stale', 'rejected')",
    )
    op.create_check_constraint(
        "ck_observations_quality",
        "observations",
        "quality IN ('primary_source', 'provider_reported', 'verified_child', "
        "'internal_context', 'untrusted_retrieval', 'discovery_only')",
    )
    op.create_check_constraint(
        "ck_observations_schema_version",
        "observations",
        "schema_version IN ('observation-v1', 'observation-v2')",
    )
    op.create_check_constraint(
        "ck_observations_rejection_state",
        "observations",
        "(status = 'rejected' AND rejection_code IS NOT NULL) OR "
        "(status <> 'rejected' AND rejection_code IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint("ck_observations_rejection_state", "observations", type_="check")
    op.drop_constraint("ck_observations_schema_version", "observations", type_="check")
    op.drop_constraint("ck_observations_quality", "observations", type_="check")
    op.drop_constraint("ck_observations_status", "observations", type_="check")
    op.drop_column("observations", "rejection_code")
    op.drop_column("observations", "normalization_version")
    op.drop_column("observations", "schema_version")
    op.drop_column("observations", "quality")
    op.drop_column("observations", "status")
