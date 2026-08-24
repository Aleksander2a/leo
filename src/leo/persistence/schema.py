"""Initial relational schema for trusted Slack ingress and harness state."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    CheckConstraint,
    Computed,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# openai/text-embedding-3-small via OpenRouter; a model/dimension change requires a
# new migration (the column width is fixed, unlike TSVECTOR).
_EMBEDDING_DIMENSION = 1536


class Base(DeclarativeBase):
    pass


class ThreadRow(Base):
    __tablename__ = "threads"
    __table_args__ = (
        UniqueConstraint("origin_provider", "external_thread_id", name="uq_thread_origin"),
        Index("ix_threads_scope", "organization_id", "strategy_id"),
        Index("ix_threads_conversation", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    origin_provider: Mapped[str] = mapped_column(String(32), nullable=False)
    external_thread_id: Mapped[str] = mapped_column(String(255), nullable=False)
    external_event_id: Mapped[str | None] = mapped_column(String(64))
    external_channel_id: Mapped[str | None] = mapped_column(String(64))
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="RESTRICT")
    )
    mapping_version: Mapped[int | None] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class TaskRow(Base):
    __tablename__ = "tasks"
    __table_args__ = (
        CheckConstraint(
            "continuation_kind IN ('root', 'follow_up', 'subagent')",
            name="ck_tasks_continuation_kind",
        ),
        Index("ix_tasks_scope_status", "organization_id", "strategy_id", "status"),
        Index("ix_tasks_thread_id", "thread_id"),
        Index(
            "uq_tasks_one_active_per_thread",
            "thread_id",
            unique=True,
            postgresql_where=text("status IN ('queued', 'active', 'requires_action')"),
        ),
        Index(
            "ix_tasks_claim_eligibility",
            "status",
            "retry_after",
            "lease_expires_at",
            "created_at",
            "id",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("threads.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    parent_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"))
    continuation_kind: Mapped[str] = mapped_column(
        String(32), nullable=False, default="root", server_default="root"
    )
    mapping_version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    observation_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    verifier_feedback: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    # Leo's ReAct working memory: the plan/action/result trace of this task's own
    # earlier iterations, so a resumed or multi-turn run does not cold-start.
    scratchpad: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    # The committed step plan. Durable because completion is gated on it: a
    # resumed run must still know what it promised to do.
    step_plan: Mapped[list[dict[str, object]]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb")
    )
    final_output: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    retry_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(String(255))


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (Index("ix_runs_task_status", "task_id", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    phase: Mapped[str] = mapped_column(String(32), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    limits: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    usage: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deadline_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    final_output: Mapped[str | None] = mapped_column(Text)
    terminal_reason: Mapped[str | None] = mapped_column(String(255))
    event_sequence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class PlanRow(Base):
    """Stable parent-owned identity for a durable multi-step plan."""

    __tablename__ = "plans"
    __table_args__ = (
        UniqueConstraint("organization_id", "idempotency_key", name="uq_plans_org_idempotency_key"),
        CheckConstraint("status IN ('active', 'completed', 'failed')", name="ck_plans_status"),
        CheckConstraint(
            "current_revision >= 1 AND current_revision <= max_revisions",
            name="ck_plans_current_revision",
        ),
        CheckConstraint("max_revisions >= 1 AND max_revisions <= 8", name="ck_plans_max_revisions"),
        CheckConstraint("version >= 1", name="ck_plans_version"),
        CheckConstraint("initial_digest ~ '^[0-9a-f]{64}$'", name="ck_plans_initial_digest"),
        CheckConstraint("char_length(idempotency_key) >= 1", name="ck_plans_idempotency_key"),
        CheckConstraint(
            "(status = 'active' AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND output IS NOT NULL AND error IS NULL) OR "
            "(status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name="ck_plans_terminal_result",
        ),
        Index("ix_plans_parent", "parent_task_id", "parent_run_id"),
        Index("ix_plans_parent_run", "parent_run_id"),
        Index("ix_plans_scope_status", "organization_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=False
    )
    parent_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False
    )
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    initial_digest: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    current_revision: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    max_revisions: Mapped[int] = mapped_column(
        Integer, nullable=False, default=4, server_default="4"
    )
    output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class PlanRevisionRow(Base):
    """Append-only immutable revision DAG and its canonical digest."""

    __tablename__ = "plan_revisions"
    __table_args__ = (
        UniqueConstraint("plan_id", "number", name="uq_plan_revisions_number"),
        UniqueConstraint("plan_id", "digest", name="uq_plan_revisions_digest"),
        CheckConstraint("number >= 1 AND number <= 8", name="ck_plan_revisions_number"),
        CheckConstraint(
            "digest ~ '^[0-9a-f]{64}$' AND "
            "(parent_digest IS NULL OR parent_digest ~ '^[0-9a-f]{64}$')",
            name="ck_plan_revisions_digests",
        ),
        CheckConstraint(
            "(number = 1 AND parent_revision_id IS NULL AND parent_digest IS NULL) OR "
            "(number > 1 AND parent_revision_id IS NOT NULL AND parent_digest IS NOT NULL)",
            name="ck_plan_revisions_parent",
        ),
        CheckConstraint(
            "jsonb_typeof(definition) = 'array' AND jsonb_array_length(definition) >= 1 "
            "AND jsonb_array_length(definition) <= 64",
            name="ck_plan_revisions_definition",
        ),
        Index("ix_plan_revisions_scope", "organization_id", "plan_id", "number"),
        Index("ix_plan_revisions_parent_revision", "parent_revision_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("plan_revisions.id", ondelete="RESTRICT")
    )
    parent_digest: Mapped[str | None] = mapped_column(String(64))
    digest: Mapped[str] = mapped_column(String(64), nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    definition: Mapped[list[dict[str, object]]] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str] = mapped_column(String(1000), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PlanNodeRow(Base):
    """Mutable, fenced execution state for one immutable revision node."""

    __tablename__ = "plan_nodes"
    __table_args__ = (
        UniqueConstraint("revision_id", "node_key", name="uq_plan_nodes_revision_key"),
        CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'failed')",
            name="ck_plan_nodes_status",
        ),
        CheckConstraint(
            "attempt >= 0 AND max_attempts >= 1 AND max_attempts <= 8 AND attempt <= max_attempts",
            name="ck_plan_nodes_attempt",
        ),
        CheckConstraint(
            "jsonb_typeof(depends_on) = 'array' AND jsonb_array_length(depends_on) <= 16",
            name="ck_plan_nodes_dependencies",
        ),
        CheckConstraint(
            "(status = 'running' AND claim_owner IS NOT NULL AND claim_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL AND attempt >= 1) OR "
            "(status <> 'running' AND claim_owner IS NULL AND claim_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_plan_nodes_claim_state",
        ),
        CheckConstraint(
            "(child_task_id IS NULL AND child_run_id IS NULL) OR "
            "(child_task_id IS NOT NULL AND child_run_id IS NOT NULL)",
            name="ck_plan_nodes_child_identity",
        ),
        CheckConstraint(
            "(status IN ('pending', 'running') AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND output IS NOT NULL AND error IS NULL) OR "
            "(status = 'failed' AND output IS NULL AND error IS NOT NULL)",
            name="ck_plan_nodes_result",
        ),
        Index("ix_plan_nodes_plan_revision", "plan_id", "revision_id", "node_key"),
        Index(
            "ix_plan_nodes_claim_eligibility",
            "plan_id",
            "revision_number",
            "status",
            "lease_expires_at",
            "node_key",
        ),
        Index("ix_plan_nodes_child_task", "child_task_id"),
        Index("ix_plan_nodes_child_run", "child_run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), nullable=False)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("plan_revisions.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    revision_number: Mapped[int] = mapped_column(Integer, nullable=False)
    node_key: Mapped[str] = mapped_column(String(64), nullable=False)
    objective: Mapped[str] = mapped_column(Text, nullable=False)
    depends_on: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    max_attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3, server_default="3"
    )
    claim_owner: Mapped[str | None] = mapped_column(String(128))
    claim_token: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    child_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id", ondelete="RESTRICT"))
    child_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="RESTRICT"))
    output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class DelegationRow(Base):
    """Append-only child-attempt record for audit and restart replay."""

    __tablename__ = "delegations"
    __table_args__ = (
        UniqueConstraint("node_id", "attempt", name="uq_delegations_node_attempt"),
        UniqueConstraint("node_id", "claim_token", name="uq_delegations_node_claim_token"),
        CheckConstraint(
            "status IN ('running', 'completed', 'failed', 'superseded')",
            name="ck_delegations_status",
        ),
        CheckConstraint("attempt >= 1 AND attempt <= 8", name="ck_delegations_attempt"),
        CheckConstraint(
            "(child_task_id IS NULL AND child_run_id IS NULL) OR "
            "(child_task_id IS NOT NULL AND child_run_id IS NOT NULL)",
            name="ck_delegations_child_identity",
        ),
        CheckConstraint(
            "(status = 'running' AND finished_at IS NULL AND output IS NULL AND error IS NULL) OR "
            "(status = 'completed' AND finished_at IS NOT NULL AND output IS NOT NULL "
            "AND error IS NULL) OR "
            "(status IN ('failed', 'superseded') AND finished_at IS NOT NULL "
            "AND output IS NULL AND error IS NOT NULL)",
            name="ck_delegations_result",
        ),
        Index("ix_delegations_plan_revision", "plan_id", "revision_id", "node_id"),
        Index("ix_delegations_parent", "parent_task_id", "parent_run_id"),
        Index("ix_delegations_revision", "revision_id"),
        Index("ix_delegations_parent_run", "parent_run_id"),
        Index("ix_delegations_child_task", "child_task_id"),
        Index("ix_delegations_child_run", "child_run_id"),
        Index("ix_delegations_scope_status", "organization_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    plan_id: Mapped[str] = mapped_column(ForeignKey("plans.id", ondelete="CASCADE"), nullable=False)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("plan_revisions.id", ondelete="CASCADE"), nullable=False
    )
    node_id: Mapped[str] = mapped_column(
        ForeignKey("plan_nodes.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_task_id: Mapped[str] = mapped_column(
        ForeignKey("tasks.id", ondelete="RESTRICT"), nullable=False
    )
    parent_run_id: Mapped[str] = mapped_column(
        ForeignKey("runs.id", ondelete="RESTRICT"), nullable=False
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False)
    owner: Mapped[str] = mapped_column(String(128), nullable=False)
    claim_token: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    child_task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id", ondelete="RESTRICT"))
    child_run_id: Mapped[str | None] = mapped_column(ForeignKey("runs.id", ondelete="RESTRICT"))
    output: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class ObservationRow(Base):
    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_run", "run_id"),
        Index("ix_observations_scope", "organization_id", "strategy_id"),
        CheckConstraint(
            "status IN ('retrieved', 'stale', 'rejected')",
            name="ck_observations_status",
        ),
        CheckConstraint(
            "quality IN ('primary_source', 'provider_reported', 'verified_child', "
            "'internal_context', 'untrusted_retrieval', 'discovery_only')",
            name="ck_observations_quality",
        ),
        CheckConstraint(
            "schema_version IN ('observation-v1', 'observation-v2')",
            name="ck_observations_schema_version",
        ),
        CheckConstraint(
            "(status = 'rejected' AND rejection_code IS NOT NULL) OR "
            "(status <> 'rejected' AND rejection_code IS NULL)",
            name="ck_observations_rejection_state",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    tool_call_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(128), nullable=False)
    data: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    source: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="retrieved", server_default="retrieved"
    )
    quality: Mapped[str] = mapped_column(
        String(32), nullable=False, default="provider_reported", server_default="provider_reported"
    )
    schema_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="observation-v2", server_default="observation-v2"
    )
    normalization_version: Mapped[str] = mapped_column(
        String(32), nullable=False, default="normalization-v1", server_default="normalization-v1"
    )
    rejection_code: Mapped[str | None] = mapped_column(String(64))


class RunEventRow(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence", name="uq_run_event_sequence"),
        Index("ix_run_events_task", "task_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False, default=dict)


class OrganizationRow(Base):
    __tablename__ = "organizations"
    __table_args__ = (CheckConstraint("version >= 1", name="ck_organizations_version"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class StrategyRow(Base):
    __tablename__ = "strategies"
    __table_args__ = (
        UniqueConstraint("organization_id", "slug", name="uq_strategies_org_slug"),
        CheckConstraint("version >= 1", name="ck_strategies_version"),
        Index("ix_strategies_scope", "organization_id", "id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="", server_default="")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MembershipRow(Base):
    __tablename__ = "organization_memberships"
    __table_args__ = (
        UniqueConstraint("organization_id", "actor_id", name="uq_membership_org_actor"),
        CheckConstraint("role IN ('owner', 'researcher', 'viewer')", name="ck_membership_role"),
        CheckConstraint("status IN ('active', 'revoked')", name="ck_membership_status"),
        CheckConstraint("version >= 1", name="ck_membership_version"),
        Index("ix_membership_org_status", "organization_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class AssetRow(Base):
    __tablename__ = "assets"
    __table_args__ = (UniqueConstraint("symbol", name="uq_assets_symbol"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)


class PortfolioRow(Base):
    __tablename__ = "portfolios"
    __table_args__ = (
        UniqueConstraint("organization_id", "strategy_id", "name", name="uq_portfolio_scope_name"),
        CheckConstraint("version >= 1", name="ck_portfolios_version"),
        Index("ix_portfolios_scope", "organization_id", "strategy_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="USD")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class PositionRow(Base):
    __tablename__ = "positions"
    __table_args__ = (
        CheckConstraint("quantity >= 0", name="ck_positions_quantity"),
        CheckConstraint("weight >= 0 AND weight <= 1", name="ck_positions_weight"),
        CheckConstraint("version >= 1", name="ck_positions_version"),
        Index("ix_positions_portfolio_as_of", "portfolio_id", "as_of"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    asset_id: Mapped[str] = mapped_column(ForeignKey("assets.id"), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class MandateRow(Base):
    __tablename__ = "mandates"
    __table_args__ = (
        CheckConstraint(
            "target_weight IS NULL OR (target_weight >= 0 AND target_weight <= 1)",
            name="ck_mandates_target_weight",
        ),
        CheckConstraint(
            "expires_at IS NULL OR expires_at > effective_at", name="ck_mandates_window"
        ),
        CheckConstraint("version >= 1", name="ck_mandates_version"),
        Index("ix_mandates_scope_effective", "organization_id", "strategy_id", "effective_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    target_weight: Mapped[float | None] = mapped_column(Float)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class ThesisRow(Base):
    __tablename__ = "theses"
    __table_args__ = (
        UniqueConstraint(
            "organization_id", "strategy_id", "subject", name="uq_theses_scope_subject"
        ),
        CheckConstraint("status IN ('active', 'superseded', 'closed')", name="ck_theses_status"),
        CheckConstraint("current_version >= 0", name="ck_theses_current_version"),
        CheckConstraint("version >= 1", name="ck_theses_version"),
        Index("ix_theses_scope_status", "organization_id", "strategy_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    current_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class ThesisVersionRow(Base):
    __tablename__ = "thesis_versions"
    __table_args__ = (
        UniqueConstraint("thesis_id", "number", name="uq_thesis_versions_number"),
        CheckConstraint("number >= 1", name="ck_thesis_versions_number"),
        CheckConstraint(
            "status IN ('active', 'superseded', 'closed')", name="ck_thesis_versions_status"
        ),
        Index("ix_thesis_versions_thesis_number", "thesis_id", "number"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thesis_id: Mapped[str] = mapped_column(ForeignKey("theses.id"), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)


class AssumptionRow(Base):
    __tablename__ = "thesis_assumptions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'superseded', 'closed')", name="ck_assumptions_status"
        ),
        CheckConstraint("version >= 1", name="ck_assumptions_version"),
        Index("ix_assumptions_thesis_version", "thesis_version_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thesis_version_id: Mapped[str] = mapped_column(ForeignKey("thesis_versions.id"), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class DecisionRow(Base):
    __tablename__ = "strategy_decisions"
    __table_args__ = (
        CheckConstraint("kind IN ('invest', 'avoid', 'review')", name="ck_decisions_kind"),
        CheckConstraint("version >= 1", name="ck_decisions_version"),
        Index("ix_decisions_scope_time", "organization_id", "strategy_id", "decided_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    thesis_version_id: Mapped[str | None] = mapped_column(ForeignKey("thesis_versions.id"))
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class RiskConstraintRow(Base):
    __tablename__ = "risk_constraints"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('max_position_weight', 'max_drawdown', 'min_cash_weight')",
            name="ck_risk_constraints_kind",
        ),
        CheckConstraint("value >= 0 AND value <= 1", name="ck_risk_constraints_value"),
        CheckConstraint(
            "expires_at IS NULL OR expires_at > effective_at", name="ck_risk_constraints_window"
        ),
        CheckConstraint("version >= 1", name="ck_risk_constraints_version"),
        Index(
            "ix_risk_constraints_scope_effective", "organization_id", "strategy_id", "effective_at"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    effective_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    source_ref: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")


class MemoryRecordRow(Base):
    __tablename__ = "memory_records"
    __table_args__ = (
        CheckConstraint(
            "visibility IN ('thread_local', 'conversation_local', 'channel_local', "
            "'actor_private', 'strategy_shared', 'organization_shared')",
            name="ck_memory_records_visibility",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'contested', 'retracted')",
            name="ck_memory_records_status",
        ),
        CheckConstraint("current_revision >= 1", name="ck_memory_records_revision"),
        CheckConstraint("generation >= 1", name="ck_memory_records_generation"),
        Index("ix_memory_records_scope_status", "organization_id", "strategy_id", "status"),
        Index("ix_memory_records_namespace", "visibility", "namespace_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    current_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class MemorySourceRow(Base):
    __tablename__ = "memory_sources"
    __table_args__ = (
        CheckConstraint(
            "visibility IN ('thread_local', 'conversation_local', 'channel_local', "
            "'actor_private', 'strategy_shared', 'organization_shared')",
            name="ck_memory_sources_visibility",
        ),
        Index("ix_memory_sources_scope", "organization_id", "strategy_id"),
        Index("ix_memory_sources_namespace", "visibility", "namespace_id"),
    )

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    reference: Mapped[str] = mapped_column(String(255), nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)


class MemoryRevisionRow(Base):
    __tablename__ = "memory_revisions"
    __table_args__ = (
        UniqueConstraint("record_id", "number", name="uq_memory_revisions_number"),
        CheckConstraint("number >= 1", name="ck_memory_revisions_number"),
        CheckConstraint("char_length(content) <= 16384", name="ck_memory_revisions_content_size"),
        CheckConstraint(
            "sensitivity >= 0 AND sensitivity <= 1", name="ck_memory_revisions_sensitivity"
        ),
        CheckConstraint(
            "visibility IN ('thread_local', 'conversation_local', 'channel_local', "
            "'actor_private', 'strategy_shared', 'organization_shared')",
            name="ck_memory_revisions_visibility",
        ),
        CheckConstraint(
            "status IN ('active', 'superseded', 'contested', 'retracted')",
            name="ck_memory_revisions_status",
        ),
        CheckConstraint(
            "valid_until IS NULL OR valid_until > valid_from",
            name="ck_memory_revisions_valid_window",
        ),
        CheckConstraint(
            "expires_at IS NULL OR expires_at > recorded_at",
            name="ck_memory_revisions_expiry",
        ),
        CheckConstraint(
            "status <> 'superseded' OR supersedes_revision IS NOT NULL",
            name="ck_memory_revisions_superseded_parent",
        ),
        CheckConstraint(
            "source_type IN ('explicit', 'autonomous')",
            name="ck_memory_revisions_source_type",
        ),
        Index("ix_memory_revisions_record_number", "record_id", "number"),
        Index("ix_memory_revisions_scope_status", "organization_id", "strategy_id", "status"),
        Index("ix_memory_revisions_validity", "valid_from", "valid_until", "expires_at"),
        Index(
            "ix_memory_revisions_search_vector",
            "search_vector",
            postgresql_using="gin",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("memory_records.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    number: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    sensitivity: Mapped[float] = mapped_column(Float, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    valid_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    reason: Mapped[str] = mapped_column(String(128), nullable=False)
    supersedes_revision: Mapped[int | None] = mapped_column(Integer)
    search_vector: Mapped[object] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )
    # Harness-set provenance, never model/candidate-supplied: 'explicit' for a
    # user-issued remember/correct command, 'autonomous' for a model-proposed
    # memory.note capture that passed duplicate/contradiction governance.
    source_type: Mapped[str] = mapped_column(String(16), nullable=False, server_default="explicit")


class MemoryEmbeddingRow(Base):
    """Disposable semantic index for one memory revision's canonical content.

    Keyed by (revision_id, content_hash, model) so a content change or model
    rotation creates a new row rather than silently reusing a stale vector;
    old rows are cache-invalidated the same way memory_retrieval_cache is.
    """

    __tablename__ = "memory_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "revision_id", "content_hash", "model", name="uq_memory_embeddings_identity"
        ),
        Index("ix_memory_embeddings_scope", "organization_id", "strategy_id"),
        Index("ix_memory_embeddings_record", "record_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision_id: Mapped[str] = mapped_column(
        ForeignKey("memory_revisions.id", ondelete="CASCADE"), nullable=False
    )
    record_id: Mapped[str] = mapped_column(
        ForeignKey("memory_records.id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class CapabilityEmbeddingRow(Base):
    """Disposable semantic index for one tool/capability's catalog summary.

    Process-global (not organization/strategy-scoped): the tool catalog is
    static configuration, not tenant data, matching how capability eligibility
    is already evaluated per-request rather than per-tenant-stored.
    """

    __tablename__ = "capability_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "capability_id", "content_hash", "model", name="uq_capability_embeddings_identity"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    capability_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(_EMBEDDING_DIMENSION), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ModelCallTranscriptRow(Base):
    """Full request/response for one model call -- dashboard inspection only.

    Deliberately separate from run_events: that table is capped at 8KB and
    field-allowlisted per event type (see leo.harness.persistence_rules) to keep the
    durable event log bounded and replayable, which a full transcript cannot respect.
    This table carries no such bound and no replay authority; it exists purely so an
    operator can see the exact message a model call sent and received. raw_request
    never contains the Authorization header (that's set on the HTTP client, not the
    request body), so it is safe to store without redaction.
    """

    __tablename__ = "model_call_transcripts"
    __table_args__ = (
        UniqueConstraint("run_id", "request_id", name="uq_model_call_transcripts_identity"),
        Index("ix_model_call_transcripts_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(String(64), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    iteration: Mapped[int] = mapped_column(Integer, nullable=False)
    raw_request: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    raw_response: Mapped[dict[str, object]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class DeliveryOutboxRow(Base):
    """One immutable Slack delivery intent and its retry/ambiguity state."""

    __tablename__ = "delivery_outbox"
    __table_args__ = (
        UniqueConstraint(
            "task_id",
            "run_id",
            "kind",
            "payload_version",
            name="uq_delivery_outbox_logical_key",
        ),
        CheckConstraint(
            "kind IN ('progress', 'final')",
            name="ck_delivery_outbox_kind",
        ),
        CheckConstraint(
            "state IN ('pending', 'leased', 'retry', 'delivered', 'dead', 'unknown_effect')",
            name="ck_delivery_outbox_state",
        ),
        CheckConstraint("payload_version >= 1", name="ck_delivery_outbox_payload_version"),
        CheckConstraint("attempt_count >= 0", name="ck_delivery_outbox_attempt_count"),
        CheckConstraint(
            "(state = 'leased' AND lease_owner IS NOT NULL AND lease_token IS NOT NULL "
            "AND lease_expires_at IS NOT NULL) OR "
            "(state <> 'leased' AND lease_owner IS NULL AND lease_token IS NULL "
            "AND lease_expires_at IS NULL)",
            name="ck_delivery_outbox_lease_state",
        ),
        CheckConstraint(
            "state NOT IN ('delivered', 'dead', 'unknown_effect') OR "
            "(lease_owner IS NULL AND lease_token IS NULL AND lease_expires_at IS NULL)",
            name="ck_delivery_outbox_terminal_no_lease",
        ),
        Index(
            "ix_delivery_outbox_claim_eligibility",
            "state",
            "retry_after",
            "lease_expires_at",
            "created_at",
            "id",
        ),
        Index("ix_delivery_outbox_scope", "organization_id", "strategy_id", "state"),
        Index("ix_delivery_outbox_task", "task_id"),
        Index("ix_delivery_outbox_run", "run_id"),
        Index("ix_delivery_outbox_ingress", "ingress_event_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    ingress_event_id: Mapped[str] = mapped_column(
        ForeignKey("slack_ingress_events.event_id"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_channel_id: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    payload_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    payload_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(String(24), nullable=False, default="pending")
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_token: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    retry_after: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    receipt_message_ts: Mapped[str | None] = mapped_column(String(64))
    last_error: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ClaimRow(Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index("ix_claims_run", "run_id"),
        Index("ix_claims_scope", "organization_id", "strategy_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    observation_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)


class SlackChannelScopeRow(Base):
    """Durable trust mapping from a Slack channel to one Leo namespace."""

    __tablename__ = "slack_channel_scopes"
    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'pending', 'revoked', 'conflict')",
            name="ck_slack_channel_scopes_status",
        ),
        CheckConstraint("version >= 1", name="ck_slack_channel_scopes_version"),
        Index(
            "ix_slack_channel_scopes_scope_status",
            "organization_id",
            "strategy_id",
            "status",
        ),
    )

    team_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    channel_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    provisioned_by_user_id: Mapped[str] = mapped_column(String(32), nullable=False)
    provisioned_via: Mapped[str] = mapped_column(String(32), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ConversationRow(Base):
    __tablename__ = "conversations"
    __table_args__ = (
        UniqueConstraint(
            "provider", "team_id", "external_id", name="uq_conversations_provider_external"
        ),
        CheckConstraint(
            "kind IN ('channel', 'dm', 'group_dm', 'shared', 'external')",
            name="ck_conversations_kind",
        ),
        CheckConstraint(
            "(kind = 'dm' AND actor_id IS NOT NULL) OR (kind <> 'dm' AND actor_id IS NULL)",
            name="ck_conversations_actor_shape",
        ),
        CheckConstraint(
            "authority_source IN ('slack_conversations_info', 'slack_event', "
            "'historical_snapshot')",
            name="ck_conversations_authority_source",
        ),
        CheckConstraint(
            "bot_presence IN ('present', 'absent', 'unknown')",
            name="ck_conversations_bot_presence",
        ),
        CheckConstraint(
            "lifecycle IN ('active', 'archived', 'left', 'unknown')",
            name="ck_conversations_lifecycle",
        ),
        CheckConstraint(
            "external_provenance IN ('internal', 'shared', 'external', "
            "'not_applicable', 'unknown')",
            name="ck_conversations_external_provenance",
        ),
        CheckConstraint(
            "membership_policy_version >= 1",
            name="ck_conversations_membership_policy_version",
        ),
        Index("ix_conversations_team_kind", "team_id", "kind"),
        Index(
            "ix_conversations_team_authority",
            "team_id",
            "bot_presence",
            "lifecycle",
            "updated_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(32), nullable=False)
    team_id: Mapped[str] = mapped_column(String(64), nullable=False)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_id: Mapped[str | None] = mapped_column(String(128))
    authority_source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="historical_snapshot",
        server_default="historical_snapshot",
    )
    bot_presence: Mapped[str] = mapped_column(
        String(16), nullable=False, default="present", server_default="present"
    )
    lifecycle: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active"
    )
    external_provenance: Mapped[str] = mapped_column(
        String(24), nullable=False, default="unknown", server_default="unknown"
    )
    membership_policy_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ConversationThreadRow(Base):
    __tablename__ = "conversation_threads"
    __table_args__ = (
        UniqueConstraint("conversation_id", "root_ts", name="uq_conversation_threads_root"),
        CheckConstraint("mapping_version >= 1", name="ck_conversation_threads_mapping_version"),
        CheckConstraint("version >= 1", name="ck_conversation_threads_version"),
        Index("ix_conversation_threads_scope", "organization_id", "strategy_id"),
        Index("ix_conversation_threads_conversation", "conversation_id", "root_ts"),
        UniqueConstraint(
            "harness_thread_id",
            name="uq_conversation_threads_harness_thread",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    harness_thread_id: Mapped[str | None] = mapped_column(
        ForeignKey("threads.id", ondelete="CASCADE")
    )
    root_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    mapping_version: Mapped[int] = mapped_column(Integer, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ConversationSelectionRow(Base):
    __tablename__ = "conversation_scope_selections"
    __table_args__ = (
        CheckConstraint("status IN ('active', 'revoked')", name="ck_conversation_selection_status"),
        CheckConstraint("version >= 1", name="ck_conversation_selection_version"),
        Index("ix_conversation_selection_actor", "conversation_id", "actor_id", "status"),
        Index("ix_conversation_selection_scope", "organization_id", "strategy_id", "status"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    organization_id: Mapped[str] = mapped_column(ForeignKey("organizations.id"), nullable=False)
    strategy_id: Mapped[str] = mapped_column(ForeignKey("strategies.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SlackIngressEventRow(Base):
    __tablename__ = "slack_ingress_events"
    __table_args__ = (
        CheckConstraint(
            "mapping_version IS NULL OR mapping_version >= 1",
            name="ck_slack_ingress_mapping_version",
        ),
        CheckConstraint(
            "(organization_id IS NULL AND strategy_id IS NULL AND mapping_version IS NULL) "
            "OR (organization_id IS NOT NULL AND strategy_id IS NOT NULL "
            "AND mapping_version IS NOT NULL)",
            name="ck_slack_ingress_scope_snapshot",
        ),
        Index("ix_slack_ingress_thread", "team_id", "channel_id", "thread_root_ts"),
        Index(
            "ix_slack_ingress_actor_context",
            "team_id",
            "user_id",
            "conversation_kind",
            "received_at",
        ),
        Index(
            "ix_slack_ingress_context_conversations",
            "context_conversation_ids",
            postgresql_using="gin",
        ),
        Index("ix_slack_ingress_context_access_hash", "context_access_hash"),
        Index("ix_slack_ingress_conversation_id", "conversation_id", "received_at"),
        Index("ix_slack_ingress_events_task_id", "task_id"),
        Index(
            "uq_slack_ingress_task_id",
            "task_id",
            unique=True,
            postgresql_where=text("task_id IS NOT NULL"),
        ),
        CheckConstraint(
            "launch_status IN ("
            "'admitting', 'unlaunched', 'materializing', 'queued', 'failed', 'rejected'"
            ")",
            name="ck_slack_ingress_launch_status",
        ),
        CheckConstraint(
            "launch_attempt_count >= 0",
            name="ck_slack_ingress_launch_attempt_count",
        ),
        CheckConstraint(
            "(launch_status = 'queued' AND task_id IS NOT NULL) OR "
            "(launch_status IN ('admitting', 'unlaunched', 'materializing', 'failed', 'rejected') "
            "AND task_id IS NULL)",
            name="ck_slack_ingress_launch_link",
        ),
        CheckConstraint(
            "launch_status NOT IN ('unlaunched', 'materializing') OR "
            "(organization_id IS NOT NULL AND strategy_id IS NOT NULL "
            "AND mapping_version IS NOT NULL AND prompt <> '')",
            name="ck_slack_ingress_launch_payload",
        ),
        CheckConstraint(
            "conversation_kind IN ('ordinary_internal', 'dm', 'mpim', 'shared', 'external')",
            name="ck_slack_ingress_conversation_kind",
        ),
        CheckConstraint(
            "trigger_kind IN ('app_mention', 'message_im')",
            name="ck_slack_ingress_trigger_kind",
        ),
        CheckConstraint(
            "jsonb_typeof(context_conversation_ids) = 'array' "
            "AND jsonb_array_length(context_conversation_ids) >= 1",
            name="ck_slack_ingress_context_shape",
        ),
        CheckConstraint(
            "context_conversation_ids ? channel_id",
            name="ck_slack_ingress_context_current",
        ),
        CheckConstraint(
            "conversation_kind = 'dm' OR "
            "(jsonb_array_length(context_conversation_ids) = 1 "
            "AND context_conversation_ids ->> 0 = channel_id)",
            name="ck_slack_ingress_context_isolation",
        ),
        CheckConstraint(
            "trigger_kind <> 'message_im' OR conversation_kind = 'dm'",
            name="ck_slack_ingress_message_im_kind",
        ),
        CheckConstraint(
            "context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_slack_ingress_context_access_hash",
        ),
        CheckConstraint(
            "context_projection_source IN ('exact_destination', "
            "'dm_membership_intersection', 'dm_only_fallback')",
            name="ck_slack_ingress_context_projection_source",
        ),
        CheckConstraint(
            "conversation_authority_source IN ('slack_conversations_info', 'slack_event')",
            name="ck_slack_ingress_conversation_authority_source",
        ),
        CheckConstraint(
            "bot_presence IN ('present', 'absent', 'unknown')",
            name="ck_slack_ingress_bot_presence",
        ),
        CheckConstraint(
            "conversation_lifecycle IN ('active', 'archived', 'left', 'unknown')",
            name="ck_slack_ingress_conversation_lifecycle",
        ),
        CheckConstraint(
            "external_provenance IN ('internal', 'shared', 'external', "
            "'not_applicable', 'unknown')",
            name="ck_slack_ingress_external_provenance",
        ),
        CheckConstraint(
            "membership_policy_version >= 1",
            name="ck_slack_ingress_membership_policy_version",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    team_id: Mapped[str] = mapped_column(String(32), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[str] = mapped_column(String(32), nullable=False)
    message_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    thread_root_ts: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_key: Mapped[str] = mapped_column(String(255), nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    conversation_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    trigger_kind: Mapped[str] = mapped_column(String(24), nullable=False)
    context_conversation_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    context_access_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    context_projection_source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="exact_destination", server_default="exact_destination"
    )
    conversation_authority_source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="slack_event", server_default="slack_event"
    )
    bot_presence: Mapped[str] = mapped_column(
        String(16), nullable=False, default="present", server_default="present"
    )
    conversation_lifecycle: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", server_default="active"
    )
    external_provenance: Mapped[str] = mapped_column(
        String(24), nullable=False, default="unknown", server_default="unknown"
    )
    membership_policy_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="RESTRICT"), nullable=False
    )
    # Nullable only so pre-0002 admissions remain readable. Every new claim must snapshot these.
    organization_id: Mapped[str | None] = mapped_column(String(64))
    strategy_id: Mapped[str | None] = mapped_column(String(64))
    mapping_version: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="received")
    task_id: Mapped[str | None] = mapped_column(ForeignKey("tasks.id"))
    launch_status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="unlaunched", server_default="unlaunched"
    )
    launch_attempt_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0"
    )
    launch_error: Mapped[str | None] = mapped_column(String(255))
    launch_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    lease_owner: Mapped[str | None] = mapped_column(String(128))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ConversationAccessSnapshotRow(Base):
    """One immutable, normalized source in an admitted turn's exact context projection."""

    __tablename__ = "conversation_access_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "ingress_event_id",
            "conversation_external_id",
            name="uq_conversation_access_snapshot_source",
        ),
        CheckConstraint("position >= 0", name="ck_conversation_access_snapshot_position"),
        CheckConstraint(
            "source_kind IN ('exact_destination', 'dm_membership_intersection', "
            "'dm_only_fallback', 'historical_snapshot')",
            name="ck_conversation_access_snapshot_source_kind",
        ),
        CheckConstraint(
            "context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_conversation_access_snapshot_hash",
        ),
        Index(
            "ix_conversation_access_snapshot_actor",
            "team_id",
            "actor_id",
            "observed_at",
        ),
        Index(
            "ix_conversation_access_snapshot_source",
            "team_id",
            "conversation_external_id",
            "observed_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ingress_event_id: Mapped[str] = mapped_column(
        ForeignKey("slack_ingress_events.event_id", ondelete="CASCADE"), nullable=False
    )
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    team_id: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(32), nullable=False)
    destination_external_id: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_external_id: Mapped[str] = mapped_column(String(32), nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    context_access_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class ConversationActorMembershipRow(Base):
    """Latest observed actor-and-Leo access to an exact Slack conversation."""

    __tablename__ = "conversation_actor_memberships"
    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "actor_id",
            "conversation_external_id",
            name="uq_conversation_actor_membership",
        ),
        CheckConstraint(
            "status IN ('active', 'revoked')", name="ck_conversation_actor_membership_status"
        ),
        CheckConstraint("version >= 1", name="ck_conversation_actor_membership_version"),
        CheckConstraint(
            "context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_conversation_actor_membership_hash",
        ),
        Index(
            "ix_conversation_actor_membership_actor",
            "team_id",
            "actor_id",
            "status",
            "observed_at",
        ),
        Index(
            "ix_conversation_actor_membership_source",
            "team_id",
            "conversation_external_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    team_id: Mapped[str] = mapped_column(String(32), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(32), nullable=False)
    conversation_external_id: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    source_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    context_access_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1, server_default="1")
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class SanitizedMessageRow(Base):
    """Sanitized raw-message plane linked to canonical conversation authority."""

    __tablename__ = "sanitized_messages"
    __table_args__ = (
        CheckConstraint("char_length(text) BETWEEN 1 AND 8192", name="ck_sanitized_messages_text"),
        CheckConstraint("role IN ('user', 'assistant')", name="ck_sanitized_messages_role"),
        CheckConstraint(
            "context_access_hash IS NULL OR context_access_hash ~ '^[0-9a-f]{64}$'",
            name="ck_sanitized_messages_context_access_hash",
        ),
        CheckConstraint(
            "provider_thread_root_ts IS NULL OR provider_thread_root_ts ~ '^[0-9]+[.][0-9]+$'",
            name="ck_sanitized_messages_provider_thread_root_ts",
        ),
        Index(
            "ix_sanitized_messages_scope_time",
            "organization_id",
            "strategy_id",
            "recorded_at",
        ),
        Index("ix_sanitized_messages_destination", "destination_id", "recorded_at"),
        Index(
            "uq_sanitized_messages_conversation_event_role",
            "conversation_id",
            "external_event_id",
            "role",
            unique=True,
            postgresql_where=text("conversation_id IS NOT NULL"),
        ),
        Index("ix_sanitized_messages_conversation_time", "conversation_id", "recorded_at"),
        Index(
            "ix_sanitized_messages_provider_thread",
            "conversation_id",
            "provider_thread_root_ts",
            "provider_message_ts",
            postgresql_where=text("provider_thread_root_ts IS NOT NULL"),
        ),
        Index("ix_sanitized_messages_thread_time", "harness_thread_id", "recorded_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    destination_id: Mapped[str] = mapped_column(String(128), nullable=False)
    external_event_id: Mapped[str] = mapped_column(String(128), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    conversation_id: Mapped[str | None] = mapped_column(
        ForeignKey("conversations.id", ondelete="RESTRICT")
    )
    harness_thread_id: Mapped[str | None] = mapped_column(
        ForeignKey("threads.id", ondelete="SET NULL")
    )
    actor_id: Mapped[str | None] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(
        String(16), nullable=False, default="user", server_default="user"
    )
    provider_message_ts: Mapped[str | None] = mapped_column(String(64))
    provider_thread_root_ts: Mapped[str | None] = mapped_column(String(64))
    context_access_hash: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SlackThreadCoverageRow(Base):
    """Authoritative Slack root metadata used to prove persisted-thread completeness."""

    __tablename__ = "slack_thread_coverage"
    __table_args__ = (
        UniqueConstraint(
            "team_id",
            "channel_id",
            "thread_root_ts",
            name="uq_slack_thread_coverage_root",
        ),
        CheckConstraint(
            "authoritative_reply_count >= 0",
            name="ck_slack_thread_coverage_reply_count",
        ),
        CheckConstraint(
            "(authoritative_reply_count = 0 AND authoritative_latest_reply_ts IS NULL) OR "
            "(authoritative_reply_count > 0 AND authoritative_latest_reply_ts IS NOT NULL)",
            name="ck_slack_thread_coverage_reply_shape",
        ),
        CheckConstraint(
            "thread_root_ts ~ '^[0-9]+[.][0-9]+$'",
            name="ck_slack_thread_coverage_root_ts",
        ),
        CheckConstraint(
            "CASE WHEN authoritative_latest_reply_ts IS NULL THEN true "
            "WHEN authoritative_latest_reply_ts ~ '^[0-9]+[.][0-9]+$' "
            "AND thread_root_ts ~ '^[0-9]+[.][0-9]+$' "
            "THEN authoritative_latest_reply_ts::numeric > thread_root_ts::numeric "
            "ELSE false END",
            name="ck_slack_thread_coverage_latest_ts",
        ),
        CheckConstraint(
            "authority_source IN ('slack_conversations_history_bot', "
            "'slack_conversations_history_user')",
            name="ck_slack_thread_coverage_authority_source",
        ),
        CheckConstraint(
            "authority_snapshot_hash ~ '^[0-9a-f]{64}$'",
            name="ck_slack_thread_coverage_snapshot_hash",
        ),
        Index(
            "ix_slack_thread_coverage_conversation",
            "conversation_id",
            "thread_root_ts",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    team_id: Mapped[str] = mapped_column(String(64), nullable=False)
    channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    thread_root_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    authoritative_reply_count: Mapped[int] = mapped_column(Integer, nullable=False)
    authoritative_latest_reply_ts: Mapped[str | None] = mapped_column(String(64))
    authority_source: Mapped[str] = mapped_column(String(64), nullable=False)
    authority_snapshot_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    metadata_observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class ThreadSummaryRevisionRow(Base):
    __tablename__ = "thread_summary_revisions"
    __table_args__ = (
        CheckConstraint("revision >= 1", name="ck_thread_summary_revision"),
        CheckConstraint(
            "char_length(content) BETWEEN 1 AND 8192", name="ck_thread_summary_content"
        ),
        UniqueConstraint("thread_id", "revision", name="uq_thread_summary_revision"),
        Index("ix_thread_summary_scope", "organization_id", "strategy_id", "thread_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    thread_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_message_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryEmbeddingJobRow(Base):
    __tablename__ = "memory_embedding_jobs"
    __table_args__ = (
        CheckConstraint("dimensions >= 1", name="ck_memory_embedding_dimensions"),
        CheckConstraint("attempts >= 0", name="ck_memory_embedding_attempts"),
        CheckConstraint(
            "status IN ('queued', 'retry', 'succeeded', 'dead')",
            name="ck_memory_embedding_status",
        ),
        UniqueConstraint("source_id", "content_hash", "model", name="uq_memory_embedding_work"),
        Index(
            "ix_memory_embedding_jobs_scope_status",
            "organization_id",
            "strategy_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    source_plane: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(64), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued", server_default="queued"
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class MemoryRetrievalCacheRow(Base):
    __tablename__ = "memory_retrieval_cache"
    __table_args__ = (
        CheckConstraint("generation >= 1", name="ck_memory_retrieval_cache_generation"),
        UniqueConstraint(
            "organization_id",
            "strategy_id",
            "key_hash",
            "generation",
            name="uq_memory_retrieval_cache_key",
        ),
        Index(
            "ix_memory_retrieval_cache_scope",
            "organization_id",
            "strategy_id",
            "generation",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    key_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    result_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class MemoryCapabilityHandleRow(Base):
    """Opaque progressive-memory capability bound to one run and access snapshot."""

    __tablename__ = "memory_capability_handles"
    __table_args__ = (
        UniqueConstraint("handle_hash", name="uq_memory_capability_handle_hash"),
        CheckConstraint(
            "destination_kind IN ('channel', 'dm', 'group_dm', 'shared', 'external')",
            name="ck_memory_capability_handle_destination_kind",
        ),
        CheckConstraint(
            "jsonb_typeof(source_conversation_ids) = 'array' "
            "AND jsonb_array_length(source_conversation_ids) >= 1",
            name="ck_memory_capability_handle_sources",
        ),
        CheckConstraint(
            "source_conversation_ids ? destination_id",
            name="ck_memory_capability_handle_destination_source",
        ),
        CheckConstraint("revision >= 1", name="ck_memory_capability_handle_revision"),
        CheckConstraint(
            "max_opens BETWEEN 1 AND 64 AND open_count BETWEEN 0 AND max_opens",
            name="ck_memory_capability_handle_open_budget",
        ),
        CheckConstraint(
            "access_hash ~ '^[0-9a-f]{64}$' AND membership_hash ~ '^[0-9a-f]{64}$'",
            name="ck_memory_capability_handle_authority_hashes",
        ),
        CheckConstraint(
            "handle_hash ~ '^[0-9a-f]{64}$'",
            name="ck_memory_capability_handle_hash",
        ),
        CheckConstraint(
            "(invalidated_at IS NULL AND invalidation_reason IS NULL) OR "
            "(invalidated_at IS NOT NULL AND invalidation_reason IS NOT NULL)",
            name="ck_memory_capability_handle_invalidation",
        ),
        Index(
            "ix_memory_capability_handles_run",
            "run_id",
            "expires_at",
        ),
        Index(
            "ix_memory_capability_handles_task",
            "task_id",
        ),
        Index(
            "ix_memory_capability_handles_actor",
            "organization_id",
            "team_id",
            "actor_id",
            "invalidated_at",
        ),
        Index(
            "ix_memory_capability_handles_record",
            "record_id",
            "revision",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    handle_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    organization_id: Mapped[str] = mapped_column(String(64), nullable=False)
    strategy_id: Mapped[str] = mapped_column(String(64), nullable=False)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    team_id: Mapped[str] = mapped_column(String(32), nullable=False)
    destination_id: Mapped[str] = mapped_column(String(128), nullable=False)
    destination_kind: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_id: Mapped[str] = mapped_column(String(128), nullable=False)
    access_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    membership_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_conversation_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    current_thread_namespace_id: Mapped[str] = mapped_column(String(255), nullable=False)
    record_id: Mapped[str] = mapped_column(
        ForeignKey("memory_records.id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    visibility: Mapped[str] = mapped_column(String(32), nullable=False)
    namespace_id: Mapped[str] = mapped_column(String(128), nullable=False)
    policy_version: Mapped[str] = mapped_column(String(64), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    max_opens: Mapped[int] = mapped_column(Integer, nullable=False, default=8, server_default="8")
    open_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    invalidated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    invalidation_reason: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )
