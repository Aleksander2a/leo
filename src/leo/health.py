"""Safe, read-only health contracts for the local demo operator boundary."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, JsonValue
from sqlalchemy import exists, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import aliased

from leo.config import Settings
from leo.persistence.schema import (
    ConversationActorMembershipRow,
    ConversationRow,
    DelegationRow,
    DeliveryOutboxRow,
    PlanNodeRow,
    PlanRow,
    RunEventRow,
    RunRow,
    SlackIngressEventRow,
    TaskRow,
)


class HealthState(StrEnum):
    OK = "ok"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    NOT_CONFIGURED = "not_configured"


class HealthComponent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    state: HealthState
    observed_at: datetime
    age_seconds: float | None = Field(default=None, ge=0)
    reason: str = Field(min_length=1, max_length=96)
    details: dict[str, JsonValue] = Field(default_factory=dict)


class HealthSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=2, ge=1)
    observed_at: datetime
    status: HealthState
    components: tuple[HealthComponent, ...]


class SlackSocketReadinessRegistry:
    """Process-local socket signal backed by actual connection probes, never config alone."""

    def __init__(self) -> None:
        self._connected: bool | None = None
        self._ever_connected = False
        self._observed_at: datetime | None = None
        self._reason = "socket_state_not_registered"

    def record_starting(self, *, observed_at: datetime | None = None) -> None:
        self._connected = None
        self._observed_at = observed_at or datetime.now(UTC)
        self._reason = "socket_connecting"

    def record_probe(self, connected: bool, *, observed_at: datetime | None = None) -> None:
        self._connected = connected
        self._ever_connected = self._ever_connected or connected
        self._observed_at = observed_at or datetime.now(UTC)
        self._reason = (
            "socket_connected"
            if connected
            else "socket_disconnected"
            if self._ever_connected
            else "socket_connecting"
        )

    def record_probe_failure(self, *, observed_at: datetime | None = None) -> None:
        self._connected = None
        self._observed_at = observed_at or datetime.now(UTC)
        self._reason = "socket_probe_failed"

    def record_stopped(self, *, observed_at: datetime | None = None) -> None:
        self._connected = False
        self._observed_at = observed_at or datetime.now(UTC)
        self._reason = "socket_stopped"

    def component(
        self,
        *,
        configured: bool,
        observed_at: datetime | None = None,
    ) -> HealthComponent:
        now = observed_at or datetime.now(UTC)
        if not configured:
            return HealthComponent(
                name="slack_socket",
                state=HealthState.NOT_CONFIGURED,
                observed_at=now,
                reason="slack_configuration_missing",
            )
        state = (
            HealthState.OK
            if self._connected is True
            else HealthState.UNHEALTHY
            if self._connected is False
            and (self._ever_connected or self._reason == "socket_stopped")
            else HealthState.UNKNOWN
        )
        return HealthComponent(
            name="slack_socket",
            state=state,
            observed_at=now,
            age_seconds=_age(now, self._observed_at),
            reason=self._reason,
        )


SLACK_SOCKET_READINESS = SlackSocketReadinessRegistry()


def config_snapshot(
    settings: Settings,
    *,
    observed_at: datetime | None = None,
    database: HealthComponent | None = None,
    socket: HealthComponent | None = None,
    conversation_metadata: HealthComponent | None = None,
    dm_membership_sync: HealthComponent | None = None,
    model: HealthComponent | None = None,
    orchestration: HealthComponent | None = None,
    queue: HealthComponent | None = None,
    outbox: HealthComponent | None = None,
    last_success: HealthComponent | None = None,
) -> HealthSnapshot:
    now = observed_at or datetime.now(UTC)
    components = (
        HealthComponent(
            name="process",
            state=HealthState.OK,
            observed_at=now,
            reason="process_alive",
        ),
        database
        or HealthComponent(
            name="database",
            state=(
                HealthState.UNKNOWN
                if settings.database_url is not None
                else HealthState.NOT_CONFIGURED
            ),
            observed_at=now,
            reason=(
                "database_probe_not_run"
                if settings.database_url is not None
                else "database_not_configured"
            ),
        ),
        socket
        or SLACK_SOCKET_READINESS.component(
            configured=not settings.missing_for_live_slack(),
            observed_at=now,
        ),
        conversation_metadata
        or HealthComponent(
            name="conversation_metadata",
            state=(
                HealthState.UNKNOWN
                if not settings.missing_for_live_slack()
                else HealthState.NOT_CONFIGURED
            ),
            observed_at=now,
            reason=(
                "conversation_probe_not_run"
                if not settings.missing_for_live_slack()
                else "slack_configuration_missing"
            ),
        ),
        dm_membership_sync
        or HealthComponent(
            name="dm_membership_sync",
            state=(
                HealthState.UNKNOWN
                if not settings.missing_for_live_slack()
                else HealthState.NOT_CONFIGURED
            ),
            observed_at=now,
            reason=(
                "membership_probe_not_run"
                if not settings.missing_for_live_slack()
                else "slack_configuration_missing"
            ),
        ),
        model
        or HealthComponent(
            name="model",
            state=(
                HealthState.UNKNOWN
                if not settings.missing_for_conversation_providers()
                else HealthState.NOT_CONFIGURED
            ),
            observed_at=now,
            reason=(
                "model_result_not_registered"
                if not settings.missing_for_conversation_providers()
                else "provider_configuration_missing"
            ),
        ),
        orchestration
        or HealthComponent(
            name="parent_child_orchestration",
            state=HealthState.UNKNOWN,
            observed_at=now,
            reason="orchestration_probe_not_run",
        ),
        queue
        or HealthComponent(
            name="task_queue",
            state=HealthState.UNKNOWN,
            observed_at=now,
            reason="queue_probe_not_run",
        ),
        outbox
        or HealthComponent(
            name="delivery_outbox",
            state=HealthState.UNKNOWN,
            observed_at=now,
            reason="outbox_probe_not_run",
        ),
        last_success
        or HealthComponent(
            name="last_success",
            state=HealthState.UNKNOWN,
            observed_at=now,
            reason="no_last_success_probe",
        ),
    )
    return HealthSnapshot(
        observed_at=now,
        status=aggregate_status(components),
        components=components,
    )


def aggregate_status(components: tuple[HealthComponent, ...]) -> HealthState:
    states = {component.state for component in components}
    if HealthState.UNHEALTHY in states:
        return HealthState.UNHEALTHY
    if states.intersection({HealthState.DEGRADED, HealthState.UNKNOWN, HealthState.NOT_CONFIGURED}):
        return HealthState.DEGRADED
    return HealthState.OK


async def deep_health_snapshot(
    settings: Settings,
    sessions: async_sessionmaker[AsyncSession],
    *,
    observed_at: datetime | None = None,
    timeout_seconds: float = 2.0,
) -> HealthSnapshot:
    """Build the CLI/API deep snapshot from the same bounded read-only probes."""

    now = observed_at or datetime.now(UTC)
    database_result, metadata_result = await asyncio.gather(
        probe_database(
            sessions,
            observed_at=now,
            timeout_seconds=timeout_seconds,
        ),
        probe_operational_metadata(
            sessions,
            observed_at=now,
            timeout_seconds=timeout_seconds,
        ),
    )
    database, queue, outbox, last_success = database_result
    conversation, membership, orchestration, model = metadata_result
    return config_snapshot(
        settings,
        observed_at=now,
        database=database,
        conversation_metadata=conversation,
        dm_membership_sync=membership,
        model=model,
        orchestration=orchestration,
        queue=queue,
        outbox=outbox,
        last_success=last_success,
    )


async def probe_database(
    sessions: async_sessionmaker[AsyncSession],
    *,
    observed_at: datetime | None = None,
    timeout_seconds: float = 2.0,
) -> tuple[HealthComponent, HealthComponent, HealthComponent, HealthComponent]:
    """Run the read-only database probe within a hard operator-health deadline."""

    if timeout_seconds <= 0:
        raise ValueError("health probe timeout must be positive")
    now = observed_at or datetime.now(UTC)
    try:
        return await asyncio.wait_for(
            _probe_database_unbounded(sessions, observed_at=now),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        failed = HealthComponent(
            name="database",
            state=HealthState.DEGRADED,
            observed_at=now,
            reason="database_probe_timeout",
        )
        return (
            failed,
            _unknown("task_queue", now, "database_probe_timeout"),
            _unknown("delivery_outbox", now, "database_probe_timeout"),
            _unknown("last_success", now, "database_probe_timeout"),
        )


async def _probe_database_unbounded(
    sessions: async_sessionmaker[AsyncSession],
    *,
    observed_at: datetime | None = None,
) -> tuple[HealthComponent, HealthComponent, HealthComponent, HealthComponent]:
    """Read queue, outbox, and last-success aggregates without mutating leases."""

    now = observed_at or datetime.now(UTC)
    try:
        async with sessions() as session:
            await session.execute(text("SELECT 1"))
            queue_row = (
                await session.execute(
                    select(
                        func.count(TaskRow.id).filter(TaskRow.status == "queued"),
                        func.count(TaskRow.id).filter(TaskRow.status == "active"),
                        func.count(TaskRow.id).filter(
                            TaskRow.status == "active",
                            TaskRow.lease_expires_at <= func.now(),
                        ),
                        func.min(TaskRow.created_at).filter(TaskRow.status == "queued"),
                    )
                )
            ).one()
            launch_row = (
                await session.execute(
                    select(
                        func.count(SlackIngressEventRow.event_id).filter(
                            SlackIngressEventRow.launch_status.in_(
                                ("unlaunched", "materializing", "failed")
                            )
                        ),
                        func.min(SlackIngressEventRow.launch_updated_at).filter(
                            SlackIngressEventRow.launch_status.in_(
                                ("unlaunched", "materializing", "failed")
                            )
                        ),
                    )
                )
            ).one()
            outbox_row = (
                await session.execute(
                    select(
                        func.count(DeliveryOutboxRow.id).filter(
                            DeliveryOutboxRow.state.in_(("pending", "retry", "leased"))
                        ),
                        func.count(DeliveryOutboxRow.id).filter(
                            DeliveryOutboxRow.state == "unknown_effect"
                        ),
                        func.count(DeliveryOutboxRow.id).filter(DeliveryOutboxRow.state == "dead"),
                        func.min(DeliveryOutboxRow.created_at).filter(
                            DeliveryOutboxRow.state.in_(
                                ("pending", "retry", "leased", "unknown_effect", "dead")
                            )
                        ),
                    )
                )
            ).one()
            last_success_at = await session.scalar(
                select(func.max(RunRow.updated_at))
                .join(TaskRow, TaskRow.id == RunRow.task_id)
                .where(
                    RunRow.status == "completed",
                    TaskRow.parent_task_id.is_(None),
                )
            )
    except Exception:
        failed = HealthComponent(
            name="database",
            state=HealthState.DEGRADED,
            observed_at=now,
            reason="database_probe_failed",
        )
        return (
            failed,
            _unknown("task_queue", now, "database_probe_failed"),
            _unknown("delivery_outbox", now, "database_probe_failed"),
            _unknown("last_success", now, "database_probe_failed"),
        )

    queued_tasks, active_tasks, expired_leases, oldest_queued = queue_row
    unmaterialized_launches, oldest_unmaterialized = launch_row
    pending_outbox, unknown_effects, dead_outbox, oldest_outbox = outbox_row
    database = HealthComponent(
        name="database",
        state=HealthState.OK,
        observed_at=now,
        reason="database_query_succeeded",
    )
    queue = HealthComponent(
        name="task_queue",
        state=(
            HealthState.UNHEALTHY
            if expired_leases
            else HealthState.DEGRADED
            if unmaterialized_launches
            else HealthState.DEGRADED
            if queued_tasks or active_tasks
            else HealthState.OK
        ),
        observed_at=now,
        reason=(
            "expired_task_leases"
            if expired_leases
            else "unmaterialized_slack_launches"
            if unmaterialized_launches
            else "queued_or_active_tasks"
            if queued_tasks or active_tasks
            else "queue_empty"
        ),
        details={
            "queued": int(queued_tasks),
            "active": int(active_tasks),
            "expired_leases": int(expired_leases),
            "oldest_queued": _safe_time(oldest_queued),
            "unmaterialized_launches": int(unmaterialized_launches),
            "oldest_unmaterialized": _safe_time(oldest_unmaterialized),
        },
    )
    outbox = HealthComponent(
        name="delivery_outbox",
        state=(
            HealthState.UNHEALTHY
            if dead_outbox
            else HealthState.DEGRADED
            if unknown_effects or pending_outbox
            else HealthState.OK
        ),
        observed_at=now,
        reason=(
            "dead_delivery_intents"
            if dead_outbox
            else "unknown_delivery_effect"
            if unknown_effects
            else "pending_delivery_intents"
            if pending_outbox
            else "outbox_empty"
        ),
        details={
            "pending_or_leased": int(pending_outbox),
            "unknown_effect": int(unknown_effects),
            "dead": int(dead_outbox),
            "oldest_pending": _safe_time(oldest_outbox),
        },
    )
    last_success = HealthComponent(
        name="last_success",
        state=HealthState.OK if last_success_at is not None else HealthState.UNKNOWN,
        observed_at=now,
        age_seconds=(
            max(0.0, (now - last_success_at).total_seconds())
            if last_success_at is not None
            else None
        ),
        reason="completed_run_present" if last_success_at is not None else "no_completed_run",
    )
    return database, queue, outbox, last_success


async def probe_operational_metadata(
    sessions: async_sessionmaker[AsyncSession],
    *,
    observed_at: datetime | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    conversation_stale_seconds: float = 3_600,
    membership_stale_seconds: float = 3_600,
    model_stale_seconds: float = 86_400,
    timeout_seconds: float = 2.0,
) -> tuple[HealthComponent, HealthComponent, HealthComponent, HealthComponent]:
    """Run read-only metadata probes within a hard operator-health deadline."""

    if timeout_seconds <= 0:
        raise ValueError("health probe timeout must be positive")
    now = observed_at or datetime.now(UTC)
    try:
        return await asyncio.wait_for(
            _probe_operational_metadata_unbounded(
                sessions,
                observed_at=now,
                organization_id=organization_id,
                team_id=team_id,
                conversation_stale_seconds=conversation_stale_seconds,
                membership_stale_seconds=membership_stale_seconds,
                model_stale_seconds=model_stale_seconds,
            ),
            timeout=timeout_seconds,
        )
    except TimeoutError:
        return (
            _unknown("conversation_metadata", now, "metadata_probe_timeout"),
            _unknown("dm_membership_sync", now, "metadata_probe_timeout"),
            _unknown("parent_child_orchestration", now, "metadata_probe_timeout"),
            _unknown("model", now, "metadata_probe_timeout"),
        )


async def _probe_operational_metadata_unbounded(
    sessions: async_sessionmaker[AsyncSession],
    *,
    observed_at: datetime | None = None,
    organization_id: str | None = None,
    team_id: str | None = None,
    conversation_stale_seconds: float = 3_600,
    membership_stale_seconds: float = 3_600,
    model_stale_seconds: float = 86_400,
) -> tuple[HealthComponent, HealthComponent, HealthComponent, HealthComponent]:
    """Read conversation, membership, parent/child, and last-model-result signals."""

    if conversation_stale_seconds <= 0 or membership_stale_seconds <= 0 or model_stale_seconds <= 0:
        raise ValueError("health staleness thresholds must be positive")
    now = observed_at or datetime.now(UTC)
    failed_dependency = aliased(PlanNodeRow)
    matching_delegation = (
        select(DelegationRow.id)
        .where(
            DelegationRow.node_id == PlanNodeRow.id,
            DelegationRow.claim_token == PlanNodeRow.claim_token,
            DelegationRow.status == "running",
        )
        .correlate(PlanNodeRow)
    )
    matching_running_node = (
        select(PlanNodeRow.id)
        .where(
            PlanNodeRow.id == DelegationRow.node_id,
            PlanNodeRow.claim_token == DelegationRow.claim_token,
            PlanNodeRow.status == "running",
        )
        .correlate(DelegationRow)
    )
    failed_dependency_exists = (
        select(failed_dependency.id)
        .where(
            failed_dependency.plan_id == PlanNodeRow.plan_id,
            failed_dependency.revision_id == PlanNodeRow.revision_id,
            failed_dependency.status == "failed",
            PlanNodeRow.depends_on.op("?")(failed_dependency.node_key),
        )
        .correlate(PlanNodeRow)
    )
    conversation_query = select(
        func.count(ConversationRow.id),
        func.max(ConversationRow.updated_at),
        func.count(ConversationRow.id).filter(
            or_(
                ConversationRow.bot_presence != "present",
                ConversationRow.lifecycle != "active",
            )
        ),
    )
    membership_query = select(
        func.count(ConversationActorMembershipRow.id).filter(
            ConversationActorMembershipRow.status == "active"
        ),
        func.max(ConversationActorMembershipRow.observed_at).filter(
            ConversationActorMembershipRow.status == "active"
        ),
    )
    if team_id is not None:
        conversation_query = conversation_query.where(ConversationRow.team_id == team_id)
        membership_query = membership_query.where(ConversationActorMembershipRow.team_id == team_id)
    if organization_id is not None:
        membership_query = membership_query.where(
            ConversationActorMembershipRow.organization_id == organization_id
        )

    active_plans_query = select(func.count(PlanRow.id)).where(PlanRow.status == "active")
    running_nodes_query = (
        select(func.count(PlanNodeRow.id))
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(PlanNodeRow.status == "running")
    )
    expired_nodes_query = (
        select(func.count(PlanNodeRow.id))
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(
            PlanNodeRow.status == "running",
            PlanNodeRow.lease_expires_at <= func.now(),
        )
    )
    running_delegations_query = (
        select(func.count(DelegationRow.id))
        .join(PlanNodeRow, PlanNodeRow.id == DelegationRow.node_id)
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(DelegationRow.status == "running")
    )
    orphaned_nodes_query = (
        select(func.count(PlanNodeRow.id))
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(
            PlanNodeRow.status == "running",
            ~exists(matching_delegation),
        )
    )
    orphaned_delegations_query = (
        select(func.count(DelegationRow.id))
        .join(PlanNodeRow, PlanNodeRow.id == DelegationRow.node_id)
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(
            DelegationRow.status == "running",
            ~exists(matching_running_node),
        )
    )
    blocked_nodes_query = (
        select(func.count(PlanNodeRow.id))
        .join(PlanRow, PlanRow.id == PlanNodeRow.plan_id)
        .where(
            PlanRow.status == "active",
            PlanNodeRow.revision_number == PlanRow.current_revision,
            PlanNodeRow.status == "pending",
            exists(failed_dependency_exists),
        )
    )
    if organization_id is not None:
        active_plans_query = active_plans_query.where(PlanRow.organization_id == organization_id)
        running_nodes_query = running_nodes_query.where(PlanRow.organization_id == organization_id)
        expired_nodes_query = expired_nodes_query.where(PlanRow.organization_id == organization_id)
        running_delegations_query = running_delegations_query.where(
            PlanRow.organization_id == organization_id
        )
        orphaned_nodes_query = orphaned_nodes_query.where(
            PlanRow.organization_id == organization_id
        )
        orphaned_delegations_query = orphaned_delegations_query.where(
            PlanRow.organization_id == organization_id
        )
        blocked_nodes_query = blocked_nodes_query.where(PlanRow.organization_id == organization_id)

    model_query = (
        select(func.max(RunEventRow.occurred_at))
        .select_from(RunEventRow)
        .join(RunRow, RunRow.id == RunEventRow.run_id)
        .where(RunEventRow.type == "model_called")
    )
    if organization_id is not None:
        model_query = model_query.where(RunRow.organization_id == organization_id)
    try:
        async with sessions() as session:
            conversation_count, latest_conversation, unavailable_conversations = (
                await session.execute(conversation_query)
            ).one()
            active_memberships, latest_membership = (await session.execute(membership_query)).one()
            orchestration_row = (
                await session.execute(
                    select(
                        active_plans_query.scalar_subquery(),
                        running_nodes_query.scalar_subquery(),
                        expired_nodes_query.scalar_subquery(),
                        running_delegations_query.scalar_subquery(),
                        orphaned_nodes_query.scalar_subquery(),
                        orphaned_delegations_query.scalar_subquery(),
                        blocked_nodes_query.scalar_subquery(),
                    )
                )
            ).one()
            latest_model_result = await session.scalar(model_query)
    except Exception:
        return (
            _unknown("conversation_metadata", now, "database_probe_failed"),
            _unknown("dm_membership_sync", now, "database_probe_failed"),
            _unknown("parent_child_orchestration", now, "database_probe_failed"),
            _unknown("model", now, "database_probe_failed"),
        )

    conversation_age = _age(now, latest_conversation)
    conversation = HealthComponent(
        name="conversation_metadata",
        state=(
            HealthState.UNKNOWN
            if not conversation_count or conversation_age is None
            else HealthState.DEGRADED
            if unavailable_conversations or conversation_age > conversation_stale_seconds
            else HealthState.OK
        ),
        observed_at=now,
        age_seconds=conversation_age,
        reason=(
            "no_conversation_seen"
            if not conversation_count or conversation_age is None
            else "conversation_authority_unavailable"
            if unavailable_conversations
            else "conversation_snapshot_stale"
            if conversation_age > conversation_stale_seconds
            else "conversation_snapshot_fresh"
        ),
        details={
            "conversation_count": int(conversation_count),
            "unavailable_conversations": int(unavailable_conversations),
            "stale_after_seconds": conversation_stale_seconds,
        },
    )
    membership_age = _age(now, latest_membership)
    membership = HealthComponent(
        name="dm_membership_sync",
        state=(
            HealthState.UNKNOWN
            if not active_memberships or membership_age is None
            else HealthState.DEGRADED
            if membership_age > membership_stale_seconds
            else HealthState.OK
        ),
        observed_at=now,
        age_seconds=membership_age,
        reason=(
            "no_active_membership_snapshot"
            if not active_memberships or membership_age is None
            else "membership_snapshot_stale"
            if membership_age > membership_stale_seconds
            else "membership_snapshot_fresh"
        ),
        details={
            "active_memberships": int(active_memberships),
            "stale_after_seconds": membership_stale_seconds,
        },
    )
    (
        active_plans,
        running_nodes,
        expired_nodes,
        running_delegations,
        orphaned_nodes,
        orphaned_delegations,
        blocked_nodes,
    ) = orchestration_row
    orchestration_problem = bool(
        expired_nodes or orphaned_nodes or orphaned_delegations or blocked_nodes
    )
    orchestration = HealthComponent(
        name="parent_child_orchestration",
        state=HealthState.UNHEALTHY if orchestration_problem else HealthState.OK,
        observed_at=now,
        reason=(
            "expired_plan_node_leases"
            if expired_nodes
            else "orphaned_running_plan_nodes"
            if orphaned_nodes
            else "orphaned_running_delegations"
            if orphaned_delegations
            else "blocked_plan_dependencies"
            if blocked_nodes
            else "orchestration_consistent"
        ),
        details={
            "active_plans": int(active_plans),
            "running_nodes": int(running_nodes),
            "expired_node_leases": int(expired_nodes),
            "running_delegations": int(running_delegations),
            "orphaned_running_nodes": int(orphaned_nodes),
            "orphaned_running_delegations": int(orphaned_delegations),
            "blocked_dependency_nodes": int(blocked_nodes),
        },
    )
    model_age = _age(now, latest_model_result)
    model = HealthComponent(
        name="model",
        state=(
            HealthState.UNKNOWN
            if model_age is None
            else HealthState.DEGRADED
            if model_age > model_stale_seconds
            else HealthState.OK
        ),
        observed_at=now,
        age_seconds=model_age,
        reason=(
            "no_recorded_model_result"
            if model_age is None
            else "model_result_stale"
            if model_age > model_stale_seconds
            else "model_result_recent"
        ),
        details={"stale_after_seconds": model_stale_seconds},
    )
    return conversation, membership, orchestration, model


def _unknown(name: str, observed_at: datetime, reason: str) -> HealthComponent:
    return HealthComponent(
        name=name, state=HealthState.UNKNOWN, observed_at=observed_at, reason=reason
    )


def _safe_time(value: object) -> str | None:
    return value.isoformat() if isinstance(value, datetime) else None


def _age(now: datetime, value: object) -> float | None:
    return max(0.0, (now - value).total_seconds()) if isinstance(value, datetime) else None
