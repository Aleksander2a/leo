"""The agent schema, as a single baseline.

Leo's previous runtime needed forty-five tables -- plans, plan revisions, plan
nodes, delegations, observations, run events, claims, delivery outbox, seven
memory tables, six Slack-projection tables, and the rest -- to support a loop
that could not reliably answer a question. The loop that replaced it needs six.

Everything a conversation owns hangs off one ``scope_key``, so isolation
between a channel and a DM is a WHERE clause on every read rather than a policy
layer sitting above the query.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0001_agent_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSIONS = 1536

#: Tables created by the runtime this schema replaces. Dropped here so a
#: database carrying the old shape converges on the new one.
LEGACY_TABLES = (
    "conversation_access_snapshots",
    "conversation_actor_memberships",
    "conversation_scope_selections",
    "conversation_threads",
    "conversations",
    "sanitized_messages",
    "slack_thread_coverage",
    "thread_summary_revisions",
    "slack_ingress_events",
    "slack_channel_scopes",
    "delivery_outbox",
    "model_call_transcripts",
    "claims",
    "run_events",
    "observations",
    "delegations",
    "plan_nodes",
    "plan_revisions",
    "plans",
    "runs",
    "tasks",
    "threads",
    "memory_retrieval_cache",
    "memory_embedding_jobs",
    "memory_capability_handles",
    "memory_disclosure_grant_events",
    "memory_disclosure_grants",
    "memory_embeddings",
    "memory_revisions",
    "memory_sources",
    "memory_records",
    "capability_embeddings",
    "strategy_decisions",
    "risk_constraints",
    "thesis_assumptions",
    "thesis_versions",
    "theses",
    "mandates",
    "positions",
    "portfolios",
    "assets",
    "organization_memberships",
    "strategies",
    "organizations",
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    for table in LEGACY_TABLES:
        op.execute(f'DROP TABLE IF EXISTS "{table}" CASCADE')

    op.create_table(
        "agent_conversations",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scope_key", sa.String(255), nullable=False, unique=True),
        sa.Column("provider", sa.String(32), nullable=False, server_default="slack"),
        sa.Column("team_id", sa.String(64)),
        sa.Column("channel_id", sa.String(64)),
        sa.Column("kind", sa.String(16), nullable=False, server_default="channel"),
        sa.Column("title", sa.String(255)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("last_active_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        "agent_messages",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("scope_key", sa.String(255), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("agent_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(64)),
        sa.Column("thread_key", sa.String(128)),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("author_id", sa.String(64)),
        sa.Column("external_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_messages_scope_created", "agent_messages", ["scope_key", "created_at"])
    op.create_index(
        "ix_agent_messages_thread", "agent_messages", ["scope_key", "thread_key", "created_at"]
    )
    # Slack redelivers events; a partial unique index makes ingest idempotent
    # without a lock or a separate dedupe table.
    op.create_index(
        "uq_agent_messages_external",
        "agent_messages",
        ["scope_key", "external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL"),
    )

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scope_key", sa.String(255), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(64),
            sa.ForeignKey("agent_conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("actor_id", sa.String(64)),
        sa.Column("thread_key", sa.String(128)),
        sa.Column("question", sa.Text, nullable=False, server_default=""),
        sa.Column("answer", sa.Text),
        sa.Column("status", sa.String(16), nullable=False, server_default="running"),
        sa.Column("error", sa.Text),
        sa.Column("model", sa.String(128)),
        sa.Column("turns", sa.Integer, nullable=False, server_default="0"),
        sa.Column("tool_calls", sa.Integer, nullable=False, server_default="0"),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("cost", sa.Float, nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_agent_runs_scope_started", "agent_runs", ["scope_key", "started_at"])

    op.create_table(
        "agent_steps",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.String(64),
            sa.ForeignKey("agent_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("name", sa.String(128), nullable=False, server_default=""),
        sa.Column("arguments", JSONB),
        sa.Column("result", JSONB),
        sa.Column("ok", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("duration_ms", sa.Integer, nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_steps_run_seq", "agent_steps", ["run_id", "seq"])

    op.create_table(
        "agent_memories",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("scope_key", sa.String(255), nullable=False),
        sa.Column("kind", sa.String(32), nullable=False, server_default="fact"),
        sa.Column("subject", sa.String(255), nullable=False, server_default=""),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("importance", sa.Integer, nullable=False, server_default="3"),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS)),
        sa.Column("source_run_id", sa.String(64)),
        sa.Column("author_id", sa.String(64)),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("superseded_by", sa.String(64)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_agent_memories_scope_active", "agent_memories", ["scope_key", "active"])
    op.execute(
        "CREATE INDEX ix_agent_memories_embedding ON agent_memories "
        "USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)"
    )

    op.create_table(
        "agent_tool_index",
        sa.Column("name", sa.String(128), primary_key=True),
        sa.Column("fingerprint", sa.String(64), nullable=False),
        sa.Column("description", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSIONS)),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("agent_tool_index")
    op.drop_index("ix_agent_memories_embedding", table_name="agent_memories")
    op.drop_index("ix_agent_memories_scope_active", table_name="agent_memories")
    op.drop_table("agent_memories")
    op.drop_index("ix_agent_steps_run_seq", table_name="agent_steps")
    op.drop_table("agent_steps")
    op.drop_index("ix_agent_runs_scope_started", table_name="agent_runs")
    op.drop_table("agent_runs")
    op.drop_index("uq_agent_messages_external", table_name="agent_messages")
    op.drop_index("ix_agent_messages_thread", table_name="agent_messages")
    op.drop_index("ix_agent_messages_scope_created", table_name="agent_messages")
    op.drop_table("agent_messages")
    op.drop_table("agent_conversations")
