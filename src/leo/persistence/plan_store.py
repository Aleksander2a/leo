"""Postgres persistence for bounded, restart-safe plan and delegation execution."""

from __future__ import annotations

import math
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import EventType, RunStatus, ScopeKey, TaskStatus
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeClaim,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    PlanStatus,
    revision_digest,
)
from leo.harness.ports import Clock, IdGenerator
from leo.harness.store_errors import NotFoundError, StoreError
from leo.persistence.schema import (
    DelegationRow,
    PlanNodeRow,
    PlanRevisionRow,
    PlanRow,
    RunEventRow,
    RunRow,
    TaskRow,
)


class PlanConflictError(StoreError):
    """An idempotency, revision, or lifecycle precondition conflicted."""


class PlanScopeMismatchError(StoreError):
    """The trusted organization does not own the requested aggregate."""


class PlanClaimConflictError(StoreError):
    """A node completion used an expired, superseded, or forged claim."""


class PlanNoProgressError(StoreError):
    """An active plan has unfinished work but no possible next claim."""


class PlanRevisionLimitError(StoreError):
    """The immutable revision/replan bound has been exhausted."""


class PlanTerminalError(StoreError):
    """Only the stable parent may perform a valid terminal transition."""


class PostgresPlanStore:
    """Durable plan journal with one transactional serialization point per plan."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        clock: Clock,
        ids: IdGenerator,
    ) -> None:
        self._sessions = sessions
        self._clock = clock
        self._ids = ids

    async def create_or_load(
        self,
        *,
        scope: ScopeKey,
        parent_task_id: str,
        parent_run_id: str,
        idempotency_key: str,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        max_revisions: int = 4,
    ) -> PlanSnapshot:
        """Create the initial journal or replay the exact idempotent request."""

        _validate_identity(idempotency_key, "idempotency_key", maximum=128)
        if not 1 <= max_revisions <= 8:
            raise ValueError("max_revisions must be between 1 and 8")
        digest = revision_digest(goal, nodes)
        now = self._clock.now()
        try:
            async with self._sessions() as session, session.begin():
                await self._require_active_parent(
                    session,
                    scope,
                    parent_task_id,
                    parent_run_id,
                )
                existing = await session.scalar(
                    select(PlanRow)
                    .where(
                        PlanRow.organization_id == scope.organization_id,
                        PlanRow.idempotency_key == idempotency_key,
                    )
                    .with_for_update()
                )
                if existing is not None:
                    self._validate_idempotent_request(
                        existing,
                        parent_task_id=parent_task_id,
                        parent_run_id=parent_run_id,
                        digest=digest,
                        max_revisions=max_revisions,
                    )
                    return await self._snapshot(session, existing)

                plan_id = self._ids.new("plan")
                revision = self._new_revision(
                    plan_id=plan_id,
                    number=1,
                    goal=goal,
                    nodes=nodes,
                    reason="initial_plan",
                    parent=None,
                    now=now,
                )
                plan_row = PlanRow(
                    id=plan_id,
                    organization_id=scope.organization_id,
                    strategy_id=scope.strategy_id,
                    parent_task_id=parent_task_id,
                    parent_run_id=parent_run_id,
                    idempotency_key=idempotency_key,
                    initial_digest=revision.digest,
                    status=PlanStatus.ACTIVE.value,
                    current_revision=1,
                    max_revisions=max_revisions,
                    output=None,
                    error=None,
                    version=1,
                    created_at=now,
                    updated_at=now,
                )
                session.add(plan_row)
                await session.flush()
                await self._add_revision_rows(session, scope.organization_id, revision)
                await session.flush()
                return await self._snapshot(session, plan_row)
        except IntegrityError as exc:
            # A concurrent creator may have won the unique organization/idempotency key.
            async with self._sessions() as session, session.begin():
                await self._require_active_parent(
                    session,
                    scope,
                    parent_task_id,
                    parent_run_id,
                )
                existing = await session.scalar(
                    select(PlanRow)
                    .where(
                        PlanRow.organization_id == scope.organization_id,
                        PlanRow.idempotency_key == idempotency_key,
                    )
                    .with_for_update()
                )
                if existing is None:
                    raise PlanConflictError("plan creation conflicted with durable state") from exc
                self._validate_idempotent_request(
                    existing,
                    parent_task_id=parent_task_id,
                    parent_run_id=parent_run_id,
                    digest=digest,
                    max_revisions=max_revisions,
                )
                return await self._snapshot(session, existing)

    async def append_revision(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        reason: str,
    ) -> PlanSnapshot:
        """Append one bounded replan while preserving every prior attempt."""

        _validate_identity(reason, "reason", maximum=1_000)
        digest = revision_digest(goal, nodes)
        now = self._clock.now()
        try:
            async with self._sessions() as session, session.begin():
                await self._require_active_parent(
                    session,
                    scope,
                    parent_task_id,
                    parent_run_id,
                )
                plan_row = await self._lock_plan(session, scope, plan_id)
                self._require_parent_authority(plan_row, parent_task_id, parent_run_id)
                if plan_row.status != PlanStatus.ACTIVE.value:
                    raise PlanTerminalError("terminal plan cannot be replanned")
                parent_revision = await session.scalar(
                    select(PlanRevisionRow).where(
                        PlanRevisionRow.plan_id == plan_row.id,
                        PlanRevisionRow.number == plan_row.current_revision,
                    )
                )
                if parent_revision is None:
                    raise PlanConflictError("current plan revision is missing")
                if parent_revision.digest == digest:
                    return await self._snapshot(session, plan_row)
                if plan_row.current_revision >= plan_row.max_revisions:
                    raise PlanRevisionLimitError("plan revision bound exhausted")
                running = await session.scalar(
                    select(PlanNodeRow.id)
                    .where(
                        PlanNodeRow.plan_id == plan_row.id,
                        PlanNodeRow.revision_number == plan_row.current_revision,
                        PlanNodeRow.status == PlanNodeStatus.RUNNING.value,
                    )
                    .limit(1)
                )
                if running is not None:
                    raise PlanConflictError("cannot replan while a child claim is running")

                revision = self._new_revision(
                    plan_id=plan_row.id,
                    number=plan_row.current_revision + 1,
                    goal=goal,
                    nodes=nodes,
                    reason=reason,
                    parent=parent_revision,
                    now=now,
                )
                await self._add_revision_rows(session, plan_row.organization_id, revision)
                plan_row.current_revision = revision.number
                plan_row.version += 1
                plan_row.updated_at = now
                await session.flush()
                return await self._snapshot(session, plan_row)
        except IntegrityError as exc:
            raise PlanConflictError("plan revision conflicted with durable state") from exc

    async def claim_ready_node(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        owner: str,
        lease_seconds: float = 60.0,
        now: datetime | None = None,
    ) -> PlanNodeClaim | None:
        """Atomically claim one ready node, reclaiming stale attempts when bounded."""

        _validate_identity(owner, "owner", maximum=128)
        _validate_lease_seconds(lease_seconds)
        current_time = now if now is not None else self._clock.now()
        no_progress = False
        claim_result: PlanNodeClaim | None = None
        async with self._sessions() as session, session.begin():
            plan_authority = await self._get_plan(session, scope, plan_id)
            await self._require_active_parent(
                session,
                scope,
                plan_authority.parent_task_id,
                plan_authority.parent_run_id,
            )
            plan_row = await self._lock_plan(session, scope, plan_id)
            if plan_row.status != PlanStatus.ACTIVE.value:
                raise PlanTerminalError("terminal plan cannot issue child claims")
            nodes = list(
                (
                    await session.scalars(
                        select(PlanNodeRow)
                        .where(
                            PlanNodeRow.plan_id == plan_row.id,
                            PlanNodeRow.revision_number == plan_row.current_revision,
                        )
                        .order_by(PlanNodeRow.node_key)
                        .with_for_update()
                    )
                ).all()
            )
            if not nodes:
                raise PlanNoProgressError("active plan has no current revision nodes")

            # A worker that crashed on its last allowed attempt cannot leave a permanent
            # pseudo-running node. Record the stale attempt and fail the node before deciding
            # whether another independent branch can progress.
            for node in nodes:
                if (
                    node.status == PlanNodeStatus.RUNNING.value
                    and node.lease_expires_at is not None
                    and node.lease_expires_at <= current_time
                    and node.attempt >= node.max_attempts
                ):
                    await self._supersede_delegation(session, node, current_time)
                    node.status = PlanNodeStatus.FAILED.value
                    node.claim_owner = None
                    node.claim_token = None
                    node.lease_expires_at = None
                    node.error = "plan_node_attempts_exhausted"
                    node.updated_at = current_time

            completed = {
                node.node_key for node in nodes if node.status == PlanNodeStatus.COMPLETED.value
            }
            candidate = next(
                (node for node in nodes if self._claimable(node, completed, current_time)),
                None,
            )
            if candidate is None:
                if all(node.status == PlanNodeStatus.COMPLETED.value for node in nodes):
                    return None
                if any(
                    node.status == PlanNodeStatus.RUNNING.value
                    and node.lease_expires_at is not None
                    and node.lease_expires_at > current_time
                    for node in nodes
                ):
                    return None
                no_progress = True
            else:
                if candidate.status == PlanNodeStatus.RUNNING.value:
                    await self._supersede_delegation(session, candidate, current_time)
                token = self._ids.new("plan-claim")
                delegation_id = self._ids.new("delegation")
                candidate.status = PlanNodeStatus.RUNNING.value
                candidate.attempt += 1
                candidate.claim_owner = owner
                candidate.claim_token = token
                candidate.lease_expires_at = current_time + timedelta(seconds=lease_seconds)
                candidate.child_task_id = None
                candidate.child_run_id = None
                candidate.output = None
                candidate.error = None
                candidate.updated_at = current_time
                session.add(
                    DelegationRow(
                        id=delegation_id,
                        plan_id=plan_row.id,
                        revision_id=candidate.revision_id,
                        node_id=candidate.id,
                        organization_id=plan_row.organization_id,
                        parent_task_id=plan_row.parent_task_id,
                        parent_run_id=plan_row.parent_run_id,
                        attempt=candidate.attempt,
                        owner=owner,
                        claim_token=token,
                        status=DelegationStatus.RUNNING.value,
                        child_task_id=None,
                        child_run_id=None,
                        output=None,
                        error=None,
                        created_at=current_time,
                        finished_at=None,
                    )
                )
                await session.flush()
                if candidate.lease_expires_at is None:
                    raise PlanClaimConflictError("claimed node has no lease expiry")
                claim_result = PlanNodeClaim(
                    scope=_scope_model(plan_row),
                    plan_id=plan_row.id,
                    revision_id=candidate.revision_id,
                    node_id=candidate.id,
                    node_key=candidate.node_key,
                    parent_task_id=plan_row.parent_task_id,
                    parent_run_id=plan_row.parent_run_id,
                    objective=candidate.objective,
                    depends_on=tuple(candidate.depends_on),
                    owner=owner,
                    token=token,
                    attempt=candidate.attempt,
                    expires_at=candidate.lease_expires_at,
                )
        if no_progress:
            raise PlanNoProgressError(
                "plan has unfinished work but no dependency-ready node or live claim"
            )
        return claim_result

    async def complete_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        output: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
        now: datetime | None = None,
    ) -> PlanSnapshot:
        _validate_identity(output, "output", maximum=65_536)
        return await self._settle_node(
            scope=scope,
            claim=claim,
            status=PlanNodeStatus.COMPLETED,
            result=output,
            child_task_id=child_task_id,
            child_run_id=child_run_id,
            now=now,
        )

    async def attach_child(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        child_task_id: str,
        child_run_id: str,
        now: datetime | None = None,
    ) -> PlanSnapshot:
        """Persist a child identity under the current fence before child execution."""

        _validate_identity(child_task_id, "child_task_id", maximum=64)
        _validate_identity(child_run_id, "child_run_id", maximum=64)
        if scope.organization_id != claim.scope.organization_id:
            raise PlanScopeMismatchError("claim is outside the trusted organization")
        current_time = now if now is not None else self._clock.now()
        async with self._sessions() as session, session.begin():
            await self._require_active_parent(
                session,
                scope,
                claim.parent_task_id,
                claim.parent_run_id,
            )
            plan_row = await self._lock_plan(session, scope, claim.plan_id)
            if plan_row.status != PlanStatus.ACTIVE.value:
                raise PlanClaimConflictError("terminal plan cannot attach a child")
            if (
                claim.parent_task_id != plan_row.parent_task_id
                or claim.parent_run_id != plan_row.parent_run_id
            ):
                raise PlanClaimConflictError("claim parent authority is invalid")
            node = await session.scalar(
                select(PlanNodeRow)
                .where(PlanNodeRow.id == claim.node_id, PlanNodeRow.plan_id == plan_row.id)
                .with_for_update()
            )
            if not _is_current_claim(node, plan_row, claim, current_time):
                raise PlanClaimConflictError("node claim is stale, expired, or owned elsewhere")
            if node is None:  # pragma: no cover - narrowed by the fenced predicate above
                raise PlanClaimConflictError("node claim is missing")
            delegation = await session.scalar(
                select(DelegationRow)
                .where(
                    DelegationRow.node_id == node.id,
                    DelegationRow.attempt == node.attempt,
                    DelegationRow.claim_token == claim.token,
                    DelegationRow.status == DelegationStatus.RUNNING.value,
                )
                .with_for_update()
            )
            if delegation is None:
                raise PlanClaimConflictError("current delegation attempt is missing")
            requested = (child_task_id, child_run_id)
            node_child = (node.child_task_id, node.child_run_id)
            delegation_child = (delegation.child_task_id, delegation.child_run_id)
            empty = (None, None)
            if node_child not in {empty, requested} or delegation_child not in {
                empty,
                requested,
            }:
                raise PlanClaimConflictError("claim already has a different child identity")
            if node_child != delegation_child:
                raise PlanClaimConflictError("node and delegation child identities diverged")
            await self._require_child(session, scope, child_task_id, child_run_id)
            node.child_task_id = child_task_id
            node.child_run_id = child_run_id
            node.updated_at = current_time
            delegation.child_task_id = child_task_id
            delegation.child_run_id = child_run_id
            await session.flush()
            return await self._snapshot(session, plan_row)

    async def fail_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        error: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
        now: datetime | None = None,
    ) -> PlanSnapshot:
        _validate_identity(error, "error", maximum=4_000)
        return await self._settle_node(
            scope=scope,
            claim=claim,
            status=PlanNodeStatus.FAILED,
            result=error,
            child_task_id=child_task_id,
            child_run_id=child_run_id,
            now=now,
        )

    async def finalize(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        status: PlanStatus,
        result: str,
    ) -> PlanSnapshot:
        """Apply a terminal state only under the stable parent task/run authority."""

        if status not in {PlanStatus.COMPLETED, PlanStatus.FAILED}:
            raise ValueError("final status must be completed or failed")
        _validate_identity(result, "result", maximum=65_536)
        now = self._clock.now()
        async with self._sessions() as session, session.begin():
            await self._require_active_parent(
                session,
                scope,
                parent_task_id,
                parent_run_id,
            )
            plan_row = await self._lock_plan(session, scope, plan_id)
            self._require_parent_authority(plan_row, parent_task_id, parent_run_id)
            if plan_row.status != PlanStatus.ACTIVE.value:
                existing_result = (
                    plan_row.output if status is PlanStatus.COMPLETED else plan_row.error
                )
                if plan_row.status == status.value and existing_result == result:
                    return await self._snapshot(session, plan_row)
                raise PlanTerminalError("plan already has a different terminal result")
            nodes = list(
                (
                    await session.scalars(
                        select(PlanNodeRow)
                        .where(
                            PlanNodeRow.plan_id == plan_row.id,
                            PlanNodeRow.revision_number == plan_row.current_revision,
                        )
                        .with_for_update()
                    )
                ).all()
            )
            if not nodes:
                raise PlanTerminalError("plan cannot finalize without current revision nodes")
            if status is PlanStatus.COMPLETED:
                if any(node.status != PlanNodeStatus.COMPLETED.value for node in nodes):
                    raise PlanTerminalError(
                        "completed plan requires every current node to complete"
                    )
                plan_row.output = result
                plan_row.error = None
            else:
                if any(node.status == PlanNodeStatus.RUNNING.value for node in nodes):
                    raise PlanTerminalError("failed plan cannot finalize while a child is running")
                if not any(node.status == PlanNodeStatus.FAILED.value for node in nodes):
                    raise PlanTerminalError("failed plan requires at least one failed node")
                completed = {
                    node.node_key for node in nodes if node.status == PlanNodeStatus.COMPLETED.value
                }
                if any(self._claimable(node, completed, now) for node in nodes):
                    raise PlanTerminalError("failed plan still has dependency-ready work")
                plan_row.output = None
                plan_row.error = result
            plan_row.status = status.value
            plan_row.version += 1
            plan_row.updated_at = now
            await session.flush()
            return await self._snapshot(session, plan_row)

    async def reload(self, *, scope: ScopeKey, plan_id: str) -> PlanSnapshot:
        async with self._sessions() as session:
            plan_row = await self._get_plan(session, scope, plan_id)
            return await self._snapshot(session, plan_row)

    async def cancel(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        reason: str,
    ) -> PlanSnapshot:
        """Propagate an already-durable parent cancellation through child work."""

        return await self.terminate_for_parent(
            scope=scope,
            plan_id=plan_id,
            parent_task_id=parent_task_id,
            parent_run_id=parent_run_id,
            parent_status=RunStatus.CANCELLED,
            reason=reason,
            child_terminal_reason="parent_plan_cancelled",
        )

    async def terminate_for_parent(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        parent_status: RunStatus,
        reason: str,
        child_terminal_reason: str,
    ) -> PlanSnapshot:
        """Propagate one durable non-success parent terminal through plan/children."""

        if parent_status not in {
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
            RunStatus.BUDGET_EXHAUSTED,
        }:
            raise ValueError("parent plan termination requires a non-success terminal status")
        _validate_identity(reason, "reason", maximum=4_000)
        _validate_identity(child_terminal_reason, "child_terminal_reason", maximum=255)
        error = f"parent_{parent_status.value}:{reason.strip()}"
        now = self._clock.now()
        async with self._sessions() as session, session.begin():
            await self._require_terminal_parent(
                session,
                scope,
                parent_task_id,
                parent_run_id,
                parent_status,
            )
            plan_row = await self._lock_plan(session, scope, plan_id)
            self._require_parent_authority(plan_row, parent_task_id, parent_run_id)
            if plan_row.status != PlanStatus.ACTIVE.value:
                if plan_row.status == PlanStatus.FAILED.value and plan_row.error == error:
                    return await self._snapshot(session, plan_row)
                raise PlanTerminalError("plan already has a different terminal result")
            nodes = tuple(
                (
                    await session.scalars(
                        select(PlanNodeRow)
                        .where(PlanNodeRow.plan_id == plan_row.id)
                        .with_for_update()
                    )
                ).all()
            )
            child_identities = tuple(
                sorted(
                    {
                        (node.child_task_id, node.child_run_id)
                        for node in nodes
                        if node.child_task_id is not None and node.child_run_id is not None
                    }
                )
            )
            for child_task_id, child_run_id in child_identities:
                await self._cancel_attached_child(
                    session,
                    scope=scope,
                    child_task_id=child_task_id,
                    child_run_id=child_run_id,
                    terminal_reason=child_terminal_reason,
                    now=now,
                )
            for node in nodes:
                if node.status == PlanNodeStatus.RUNNING.value:
                    await self._supersede_delegation(
                        session,
                        node,
                        now,
                        error=error,
                    )
                if node.status in {
                    PlanNodeStatus.PENDING.value,
                    PlanNodeStatus.RUNNING.value,
                }:
                    node.status = PlanNodeStatus.FAILED.value
                    node.claim_owner = None
                    node.claim_token = None
                    node.lease_expires_at = None
                    node.output = None
                    node.error = error
                    node.updated_at = now
            plan_row.status = PlanStatus.FAILED.value
            plan_row.output = None
            plan_row.error = error
            plan_row.version += 1
            plan_row.updated_at = now
            await session.flush()
            return await self._snapshot(session, plan_row)

    async def _cancel_attached_child(
        self,
        session: AsyncSession,
        *,
        scope: ScopeKey,
        child_task_id: str,
        child_run_id: str,
        terminal_reason: str,
        now: datetime,
    ) -> None:
        """Atomically stop nonterminal durable child work during parent cancellation."""

        task = await session.scalar(
            select(TaskRow).where(TaskRow.id == child_task_id).with_for_update()
        )
        run = await session.scalar(
            select(RunRow).where(RunRow.id == child_run_id).with_for_update()
        )
        if task is None or run is None or run.task_id != child_task_id:
            raise NotFoundError("attached child task or run not found")
        if (
            task.organization_id != scope.organization_id
            or run.organization_id != scope.organization_id
        ):
            raise PlanScopeMismatchError("attached child is outside the trusted organization")

        current = (task.status, run.status)
        cancellable = {
            (TaskStatus.QUEUED.value, RunStatus.QUEUED.value),
            (TaskStatus.ACTIVE.value, RunStatus.RUNNING.value),
            (TaskStatus.REQUIRES_ACTION.value, RunStatus.REQUIRES_ACTION.value),
        }
        terminal = {
            (TaskStatus.COMPLETED.value, RunStatus.COMPLETED.value),
            (TaskStatus.FAILED.value, RunStatus.FAILED.value),
            (TaskStatus.FAILED.value, RunStatus.TIMED_OUT.value),
            (TaskStatus.FAILED.value, RunStatus.BUDGET_EXHAUSTED.value),
            (TaskStatus.CANCELLED.value, RunStatus.CANCELLED.value),
        }
        if current in terminal:
            return
        if current not in cancellable:
            raise PlanConflictError("attached child has an invalid task/run lifecycle pair")

        task.status = TaskStatus.CANCELLED.value
        task.version += 1
        task.updated_at = now
        task.lease_owner = None
        task.lease_token = None
        task.lease_expires_at = None
        task.heartbeat_at = None
        task.retry_after = None
        task.last_error = terminal_reason
        run.status = RunStatus.CANCELLED.value
        run.terminal_reason = terminal_reason
        run.version += 1
        run.updated_at = now
        run.event_sequence += 1
        session.add(
            RunEventRow(
                id=self._ids.new("evt"),
                run_id=run.id,
                task_id=task.id,
                sequence=run.event_sequence,
                type=EventType.RUN_CANCELLED.value,
                occurred_at=now,
                iteration=run.iteration,
                schema_version=1,
                payload={"reason": terminal_reason},
            )
        )

    async def replay(self, *, scope: ScopeKey, plan_id: str) -> PlanSnapshot:
        """Rebuild and validate the aggregate entirely from its durable journal."""

        return await self.reload(scope=scope, plan_id=plan_id)

    async def _settle_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        status: PlanNodeStatus,
        result: str,
        child_task_id: str | None,
        child_run_id: str | None,
        now: datetime | None,
    ) -> PlanSnapshot:
        if scope.organization_id != claim.scope.organization_id:
            raise PlanScopeMismatchError("claim is outside the trusted organization")
        if (child_task_id is None) != (child_run_id is None):
            raise ValueError("child_task_id and child_run_id must be supplied together")
        current_time = now if now is not None else self._clock.now()
        async with self._sessions() as session, session.begin():
            await self._require_active_parent(
                session,
                scope,
                claim.parent_task_id,
                claim.parent_run_id,
            )
            plan_row = await self._lock_plan(session, scope, claim.plan_id)
            if plan_row.status != PlanStatus.ACTIVE.value:
                raise PlanClaimConflictError("terminal plan cannot accept child results")
            if (
                claim.parent_task_id != plan_row.parent_task_id
                or claim.parent_run_id != plan_row.parent_run_id
            ):
                raise PlanClaimConflictError("claim parent authority is invalid")
            node = await session.scalar(
                select(PlanNodeRow)
                .where(PlanNodeRow.id == claim.node_id, PlanNodeRow.plan_id == plan_row.id)
                .with_for_update()
            )
            if not _is_current_claim(node, plan_row, claim, current_time):
                raise PlanClaimConflictError("node claim is stale, expired, or owned elsewhere")
            if node is None:  # pragma: no cover - narrowed by the fenced predicate above
                raise PlanClaimConflictError("node claim is missing")

            delegation = await session.scalar(
                select(DelegationRow)
                .where(
                    DelegationRow.node_id == node.id,
                    DelegationRow.attempt == node.attempt,
                    DelegationRow.claim_token == claim.token,
                    DelegationRow.status == DelegationStatus.RUNNING.value,
                )
                .with_for_update()
            )
            if delegation is None:
                raise PlanClaimConflictError("current delegation attempt is missing")
            attached_child = (node.child_task_id, node.child_run_id)
            delegation_child = (delegation.child_task_id, delegation.child_run_id)
            requested_child = (child_task_id, child_run_id)
            empty_child = (None, None)
            if attached_child != delegation_child:
                raise PlanClaimConflictError("node and delegation child identities diverged")
            if requested_child != empty_child:
                if attached_child not in {empty_child, requested_child}:
                    raise PlanClaimConflictError("claim already has a different child identity")
                effective_child = requested_child
            else:
                effective_child = attached_child
            effective_task_id, effective_run_id = effective_child
            if effective_task_id is not None and effective_run_id is not None:
                await self._require_child(
                    session,
                    scope,
                    effective_task_id,
                    effective_run_id,
                )
            node.status = status.value
            node.claim_owner = None
            node.claim_token = None
            node.lease_expires_at = None
            node.child_task_id = effective_task_id
            node.child_run_id = effective_run_id
            node.output = result if status is PlanNodeStatus.COMPLETED else None
            node.error = result if status is PlanNodeStatus.FAILED else None
            node.updated_at = current_time
            delegation.status = (
                DelegationStatus.COMPLETED.value
                if status is PlanNodeStatus.COMPLETED
                else DelegationStatus.FAILED.value
            )
            delegation.child_task_id = effective_task_id
            delegation.child_run_id = effective_run_id
            delegation.output = node.output
            delegation.error = node.error
            delegation.finished_at = current_time
            await session.flush()
            return await self._snapshot(session, plan_row)

    async def _require_active_parent(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        parent_task_id: str,
        parent_run_id: str,
    ) -> None:
        """Lock and require live parent authority before creating delegated work.

        Task then Run is the repository-wide lock order.  In particular this makes
        plan creation serialize with cancellation: either the plan commits first and
        cancellation observes/terminates it, or cancellation commits first and the
        stale plan request is rejected.
        """

        task = await session.scalar(
            select(TaskRow).where(TaskRow.id == parent_task_id).with_for_update()
        )
        run = await session.scalar(
            select(RunRow).where(RunRow.id == parent_run_id).with_for_update()
        )
        if task is None or run is None or run.task_id != parent_task_id:
            raise NotFoundError("parent task or run not found")
        if (
            task.organization_id != scope.organization_id
            or run.organization_id != scope.organization_id
        ):
            raise PlanScopeMismatchError("parent task or run is outside the trusted organization")
        if task.status != TaskStatus.ACTIVE.value or run.status != RunStatus.RUNNING.value:
            raise PlanTerminalError(
                "terminal parent task/run cannot create, claim, attach, or settle delegated work"
            )

    async def _require_child(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        child_task_id: str,
        child_run_id: str,
    ) -> None:
        task = await session.scalar(select(TaskRow).where(TaskRow.id == child_task_id))
        run = await session.scalar(select(RunRow).where(RunRow.id == child_run_id))
        if task is None or run is None or run.task_id != child_task_id:
            raise NotFoundError("child task or run not found")
        if (
            task.organization_id != scope.organization_id
            or run.organization_id != scope.organization_id
        ):
            raise PlanScopeMismatchError("child task or run is outside the trusted organization")

    async def _require_terminal_parent(
        self,
        session: AsyncSession,
        scope: ScopeKey,
        parent_task_id: str,
        parent_run_id: str,
        expected_run_status: RunStatus,
    ) -> None:
        task = await session.scalar(
            select(TaskRow).where(TaskRow.id == parent_task_id).with_for_update()
        )
        run = await session.scalar(
            select(RunRow).where(RunRow.id == parent_run_id).with_for_update()
        )
        if task is None or run is None or run.task_id != parent_task_id:
            raise NotFoundError("parent task or run not found")
        if (
            task.organization_id != scope.organization_id
            or run.organization_id != scope.organization_id
        ):
            raise PlanScopeMismatchError("parent task or run is outside the trusted organization")
        expected_task_status = (
            TaskStatus.CANCELLED
            if expected_run_status is RunStatus.CANCELLED
            else TaskStatus.FAILED
        )
        if task.status != expected_task_status.value or run.status != expected_run_status.value:
            raise PlanTerminalError(
                "matching parent terminal state must be durable before plan termination"
            )

    async def _lock_plan(self, session: AsyncSession, scope: ScopeKey, plan_id: str) -> PlanRow:
        row = await session.scalar(select(PlanRow).where(PlanRow.id == plan_id).with_for_update())
        if row is None:
            raise NotFoundError("plan not found")
        self._authorize_scope(row, scope)
        return row

    async def _get_plan(self, session: AsyncSession, scope: ScopeKey, plan_id: str) -> PlanRow:
        row = await session.scalar(select(PlanRow).where(PlanRow.id == plan_id))
        if row is None:
            raise NotFoundError("plan not found")
        self._authorize_scope(row, scope)
        return row

    @staticmethod
    def _authorize_scope(row: PlanRow, scope: ScopeKey) -> None:
        # Strategy is provenance only. The exact organization is the durable authority.
        if row.organization_id != scope.organization_id:
            raise PlanScopeMismatchError("plan is outside the trusted organization")

    @staticmethod
    def _require_parent_authority(row: PlanRow, parent_task_id: str, parent_run_id: str) -> None:
        if row.parent_task_id != parent_task_id or row.parent_run_id != parent_run_id:
            raise PlanTerminalError("only the stable parent task and run may mutate the plan")

    @staticmethod
    def _validate_idempotent_request(
        row: PlanRow,
        *,
        parent_task_id: str,
        parent_run_id: str,
        digest: str,
        max_revisions: int,
    ) -> None:
        if (
            row.parent_task_id != parent_task_id
            or row.parent_run_id != parent_run_id
            or row.initial_digest != digest
            or row.max_revisions != max_revisions
        ):
            raise PlanConflictError("idempotency key was already used for a different plan")

    def _new_revision(
        self,
        *,
        plan_id: str,
        number: int,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        reason: str,
        parent: PlanRevisionRow | None,
        now: datetime,
    ) -> PlanRevision:
        return PlanRevision(
            id=self._ids.new("plan-revision"),
            plan_id=plan_id,
            number=number,
            goal=goal,
            nodes=nodes,
            digest=revision_digest(goal, nodes),
            parent_revision_id=parent.id if parent is not None else None,
            parent_digest=parent.digest if parent is not None else None,
            reason=reason,
            created_at=now,
        )

    async def _add_revision_rows(
        self, session: AsyncSession, organization_id: str, revision: PlanRevision
    ) -> None:
        session.add(
            PlanRevisionRow(
                id=revision.id,
                plan_id=revision.plan_id,
                organization_id=organization_id,
                number=revision.number,
                parent_revision_id=revision.parent_revision_id,
                parent_digest=revision.parent_digest,
                digest=revision.digest,
                goal=revision.goal,
                definition=[node.model_dump(mode="json") for node in revision.nodes],
                reason=revision.reason,
                created_at=revision.created_at,
            )
        )
        # Flush the immutable revision before its nodes. SQLAlchemy has no ORM
        # relationships on these intentionally lean row types, so relying on unit-
        # of-work ordering can issue plan_nodes first despite the revision_id FK.
        await session.flush()
        for definition in revision.nodes:
            session.add(
                PlanNodeRow(
                    id=self._ids.new("plan-node"),
                    plan_id=revision.plan_id,
                    revision_id=revision.id,
                    organization_id=organization_id,
                    revision_number=revision.number,
                    node_key=definition.key,
                    objective=definition.objective,
                    depends_on=list(definition.depends_on),
                    status=PlanNodeStatus.PENDING.value,
                    attempt=0,
                    max_attempts=definition.max_attempts,
                    claim_owner=None,
                    claim_token=None,
                    lease_expires_at=None,
                    child_task_id=None,
                    child_run_id=None,
                    output=None,
                    error=None,
                    created_at=revision.created_at,
                    updated_at=revision.created_at,
                )
            )

    @staticmethod
    def _claimable(node: PlanNodeRow, completed: set[str], current_time: datetime) -> bool:
        dependencies_ready = set(node.depends_on) <= completed
        if not dependencies_ready or node.attempt >= node.max_attempts:
            return False
        if node.status == PlanNodeStatus.PENDING.value:
            return True
        return (
            node.status == PlanNodeStatus.RUNNING.value
            and node.lease_expires_at is not None
            and node.lease_expires_at <= current_time
        )

    @staticmethod
    async def _supersede_delegation(
        session: AsyncSession,
        node: PlanNodeRow,
        current_time: datetime,
        *,
        error: str = "stale_lease_reclaimed",
    ) -> None:
        if node.claim_token is None:
            raise PlanClaimConflictError("stale running node has no claim token")
        delegation = await session.scalar(
            select(DelegationRow)
            .where(
                DelegationRow.node_id == node.id,
                DelegationRow.claim_token == node.claim_token,
                DelegationRow.status == DelegationStatus.RUNNING.value,
            )
            .with_for_update()
        )
        if delegation is None:
            raise PlanClaimConflictError("stale running node has no delegation record")
        delegation.status = DelegationStatus.SUPERSEDED.value
        delegation.error = error
        delegation.finished_at = current_time

    async def _snapshot(self, session: AsyncSession, row: PlanRow) -> PlanSnapshot:
        # Flush expires server/on-update columns. Refresh within the active async
        # greenlet so model construction never triggers implicit database IO.
        await session.refresh(row)
        revision_rows = tuple(
            (
                await session.scalars(
                    select(PlanRevisionRow)
                    .where(
                        PlanRevisionRow.plan_id == row.id,
                        PlanRevisionRow.organization_id == row.organization_id,
                    )
                    .order_by(PlanRevisionRow.number)
                )
            ).all()
        )
        node_rows = tuple(
            (
                await session.scalars(
                    select(PlanNodeRow)
                    .where(
                        PlanNodeRow.plan_id == row.id,
                        PlanNodeRow.organization_id == row.organization_id,
                    )
                    .order_by(PlanNodeRow.revision_number, PlanNodeRow.node_key)
                )
            ).all()
        )
        delegation_rows = tuple(
            (
                await session.scalars(
                    select(DelegationRow)
                    .where(
                        DelegationRow.plan_id == row.id,
                        DelegationRow.organization_id == row.organization_id,
                    )
                    .order_by(DelegationRow.created_at, DelegationRow.id)
                )
            ).all()
        )
        revisions = tuple(_revision_model(item) for item in revision_rows)
        return PlanSnapshot(
            plan=_plan_model(row),
            revisions=revisions,
            nodes=tuple(_node_model(item) for item in node_rows),
            delegations=tuple(_delegation_model(item) for item in delegation_rows),
        )


def _scope_model(row: PlanRow) -> ScopeKey:
    return ScopeKey(organization_id=row.organization_id, strategy_id=row.strategy_id)


def _is_current_claim(
    node: PlanNodeRow | None,
    plan: PlanRow,
    claim: PlanNodeClaim,
    current_time: datetime,
) -> bool:
    return bool(
        node is not None
        and node.revision_id == claim.revision_id
        and node.revision_number == plan.current_revision
        and node.status == PlanNodeStatus.RUNNING.value
        and node.claim_owner == claim.owner
        and node.claim_token == claim.token
        and node.attempt == claim.attempt
        and node.lease_expires_at is not None
        and node.lease_expires_at > current_time
    )


def _plan_model(row: PlanRow) -> Plan:
    return Plan(
        id=row.id,
        scope=_scope_model(row),
        parent_task_id=row.parent_task_id,
        parent_run_id=row.parent_run_id,
        idempotency_key=row.idempotency_key,
        initial_digest=row.initial_digest,
        status=PlanStatus(row.status),
        current_revision=row.current_revision,
        max_revisions=row.max_revisions,
        output=row.output,
        error=row.error,
        version=row.version,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _revision_model(row: PlanRevisionRow) -> PlanRevision:
    definitions = tuple(PlanNodeDefinition.model_validate(item) for item in row.definition)
    return PlanRevision(
        id=row.id,
        plan_id=row.plan_id,
        number=row.number,
        goal=row.goal,
        nodes=definitions,
        digest=row.digest,
        parent_revision_id=row.parent_revision_id,
        parent_digest=row.parent_digest,
        reason=row.reason,
        created_at=row.created_at,
    )


def _node_model(row: PlanNodeRow) -> PlanNode:
    return PlanNode(
        id=row.id,
        plan_id=row.plan_id,
        revision_id=row.revision_id,
        revision_number=row.revision_number,
        definition=PlanNodeDefinition(
            key=row.node_key,
            objective=row.objective,
            depends_on=tuple(row.depends_on),
            max_attempts=row.max_attempts,
        ),
        status=PlanNodeStatus(row.status),
        attempt=row.attempt,
        claim_owner=row.claim_owner,
        claim_token=row.claim_token,
        lease_expires_at=row.lease_expires_at,
        child_task_id=row.child_task_id,
        child_run_id=row.child_run_id,
        output=row.output,
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _delegation_model(row: DelegationRow) -> Delegation:
    return Delegation(
        id=row.id,
        plan_id=row.plan_id,
        revision_id=row.revision_id,
        node_id=row.node_id,
        parent_task_id=row.parent_task_id,
        parent_run_id=row.parent_run_id,
        attempt=row.attempt,
        owner=row.owner,
        claim_token=row.claim_token,
        status=DelegationStatus(row.status),
        child_task_id=row.child_task_id,
        child_run_id=row.child_run_id,
        output=row.output,
        error=row.error,
        created_at=row.created_at,
        finished_at=row.finished_at,
    )


def _validate_identity(value: str, name: str, *, maximum: int) -> None:
    if not value or value != value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a non-empty value of at most {maximum} characters")


def _validate_lease_seconds(value: float) -> None:
    if not math.isfinite(value) or value <= 0 or value > 86_400:
        raise ValueError("lease_seconds must be finite and between 0 and 86400")
