"""Durable, thread-scoped Slack cancellation control service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import EventDraft, EventType, RunStatus, ScopeKey
from leo.harness.ports import Clock, IdGenerator
from leo.harness.store_errors import ConcurrencyError
from leo.harness.transitions import cancel_task_and_run
from leo.integrations.slack.cancellation import (
    SlackCancellationOutcome,
    SlackCancellationResult,
    cancellation_message,
    is_slack_cancellation_request,
)
from leo.integrations.slack.events import AdmittedSlackMention
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import PlanRow, RunRow, SlackIngressEventRow, TaskRow, ThreadRow
from leo.persistence.slack_ingress import PostgresSlackIngressAdmission

_TARGET_REASON = "slack_user_cancelled"
_CONTROL_REASON_PREFIX = "slack_cancel_control_"


class SlackLaunchPreparerPort(Protocol):
    async def prepare(self, admitted: AdmittedSlackMention) -> AdmittedSlackMention: ...


@dataclass(frozen=True, slots=True)
class _Target:
    task_id: str
    run_id: str
    scope: ScopeKey
    lease_owner: str | None
    lease_token: str | None


class PostgresSlackCancellationService:
    """Cancel only the initiating actor's active parent in the exact Slack thread.

    A cancellation event is admitted before this service runs.  Its own control
    Task is then materialized and terminalized so the acknowledgement uses the
    same durable outbox boundary as every other final Slack response.
    """

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        clock: Clock,
        ids: IdGenerator,
        ingress: PostgresSlackIngressAdmission,
    ) -> None:
        self._sessions = sessions
        self._clock = clock
        self._ids = ids
        self._ingress = ingress
        self._runs = PostgresRunStore(sessions, clock, ids)
        self._plans = PostgresPlanStore(sessions, clock, ids)

    def accepts(self, prompt: str) -> bool:
        return is_slack_cancellation_request(prompt)

    async def handle(
        self,
        admitted: AdmittedSlackMention,
        launch_preparer: SlackLaunchPreparerPort,
    ) -> SlackCancellationResult:
        if not self.accepts(admitted.job.prompt):
            raise ValueError("Slack cancellation service received a non-control prompt")
        if admitted.launch is None:
            outcome = await self._cancel_active_parent(admitted)
            if outcome is SlackCancellationOutcome.NOT_AUTHORIZED:
                # The one-active-Task invariant prevents creating a durable control
                # Task while another actor owns the active parent. Persist the denial
                # as a terminal admission policy result; do not bypass the outbox with
                # an untracked Slack response.
                await self._ingress.mark_followup_rejected(
                    admitted.job.event_id,
                    "cancel_actor_not_authorized",
                )
                return SlackCancellationResult(
                    admitted=admitted,
                    outcome=outcome,
                    message=cancellation_message(outcome),
                )
            admitted = await launch_preparer.prepare(admitted)
        else:
            outcome = await self._recover_outcome(admitted)
        return await self._terminalize_control(admitted, outcome)

    async def recover(
        self,
        launch_preparer: SlackLaunchPreparerPort,
        *,
        limit: int = 100,
    ) -> tuple[SlackCancellationResult, ...]:
        requests = await self._ingress.recover_control_requests(self.accepts, limit=limit)
        results: list[SlackCancellationResult] = []
        for admitted in requests:
            results.append(await self.handle(admitted, launch_preparer))
        return tuple(results)

    async def _cancel_active_parent(
        self,
        admitted: AdmittedSlackMention,
    ) -> SlackCancellationOutcome:
        target = await self._active_target(admitted)
        if target is None:
            return await self._prior_cancel_outcome(admitted)
        if not await self._actor_owns_target(admitted, target.task_id):
            return SlackCancellationOutcome.NOT_AUTHORIZED

        bundle = await self._runs.load(
            target.task_id,
            target.run_id,
            target.scope,
        )
        try:
            task, run = cancel_task_and_run(
                bundle.task,
                bundle.run,
                _TARGET_REASON,
                usage=bundle.run.usage,
            )
            cancelled = await self._runs.commit(
                expected_task_version=bundle.task.version,
                expected_run_version=bundle.run.version,
                task=task,
                run=run,
                events=(
                    EventDraft(
                        type=EventType.RUN_CANCELLED,
                        iteration=run.iteration,
                        payload={"reason": _TARGET_REASON},
                    ),
                ),
                lease_owner=target.lease_owner,
                lease_token=target.lease_token,
            )
        except ConcurrencyError:
            winner = await self._runs.load(
                target.task_id,
                target.run_id,
                target.scope,
            )
            if (
                winner.run.status is RunStatus.CANCELLED
                and winner.run.terminal_reason == _TARGET_REASON
            ):
                cancelled = winner
            elif winner.run.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
                RunStatus.BUDGET_EXHAUSTED,
            }:
                return SlackCancellationOutcome.TERMINAL_RACE
            else:
                raise

        await self._terminate_active_plans(
            scope=target.scope,
            task_id=cancelled.task.id,
            run_id=cancelled.run.id,
            reason=_TARGET_REASON,
        )
        return SlackCancellationOutcome.APPLIED

    async def _active_target(self, admitted: AdmittedSlackMention) -> _Target | None:
        job = admitted.job
        scope = admitted.resolution.scope
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(TaskRow, RunRow)
                    .join(ThreadRow, ThreadRow.id == TaskRow.thread_id)
                    .join(RunRow, RunRow.task_id == TaskRow.id)
                    .where(
                        ThreadRow.origin_provider == "slack",
                        ThreadRow.external_thread_id == job.conversation_key,
                        ThreadRow.external_channel_id == job.channel_id,
                        ThreadRow.organization_id == scope.organization_id,
                        TaskRow.organization_id == scope.organization_id,
                        TaskRow.status.in_(("queued", "active", "requires_action")),
                        RunRow.status.in_(("queued", "running", "requires_action")),
                    )
                    .order_by(TaskRow.created_at.desc(), RunRow.created_at.desc())
                    .limit(1)
                )
            ).one_or_none()
        if row is None:
            return None
        task, run = row
        return _Target(
            task_id=task.id,
            run_id=run.id,
            scope=ScopeKey(
                organization_id=task.organization_id,
                strategy_id=task.strategy_id,
            ),
            lease_owner=task.lease_owner,
            lease_token=task.lease_token,
        )

    async def _actor_owns_target(
        self,
        admitted: AdmittedSlackMention,
        task_id: str,
    ) -> bool:
        job = admitted.job
        async with self._sessions() as session:
            actor = await session.scalar(
                select(SlackIngressEventRow.user_id).where(
                    SlackIngressEventRow.task_id == task_id,
                    SlackIngressEventRow.team_id == job.team_id,
                    SlackIngressEventRow.channel_id == job.channel_id,
                    SlackIngressEventRow.thread_root_ts == job.thread_root_ts,
                )
            )
        return actor == job.user_id

    async def _prior_cancel_outcome(
        self,
        admitted: AdmittedSlackMention,
    ) -> SlackCancellationOutcome:
        job = admitted.job
        scope = admitted.resolution.scope
        async with self._sessions() as session:
            row = (
                await session.execute(
                    select(
                        TaskRow.id,
                        RunRow.id,
                        TaskRow.organization_id,
                        TaskRow.strategy_id,
                        RunRow.terminal_reason,
                        SlackIngressEventRow.user_id,
                    )
                    .join(TaskRow, TaskRow.id == RunRow.task_id)
                    .join(ThreadRow, ThreadRow.id == TaskRow.thread_id)
                    .join(SlackIngressEventRow, SlackIngressEventRow.task_id == TaskRow.id)
                    .where(
                        ThreadRow.origin_provider == "slack",
                        ThreadRow.external_thread_id == job.conversation_key,
                        ThreadRow.external_channel_id == job.channel_id,
                        ThreadRow.organization_id == scope.organization_id,
                        TaskRow.organization_id == scope.organization_id,
                    )
                    .order_by(TaskRow.created_at.desc(), RunRow.created_at.desc())
                    .limit(1)
                )
            ).one_or_none()
        if row is not None and row.terminal_reason == _TARGET_REASON:
            target_scope = ScopeKey(
                organization_id=row.organization_id,
                strategy_id=row.strategy_id,
            )
            await self._terminate_active_plans(
                scope=target_scope,
                task_id=row[0],
                run_id=row[1],
                reason=_TARGET_REASON,
            )
            return (
                SlackCancellationOutcome.APPLIED
                if row.user_id == job.user_id
                else SlackCancellationOutcome.NOT_AUTHORIZED
            )
        return SlackCancellationOutcome.NO_ACTIVE_TASK

    async def _recover_outcome(
        self,
        admitted: AdmittedSlackMention,
    ) -> SlackCancellationOutcome:
        assert admitted.launch is not None
        bundle = await self._runs.load(
            admitted.launch.task_id,
            admitted.launch.run_id,
            admitted.resolution.scope,
        )
        if bundle.run.terminal_reason and bundle.run.terminal_reason.startswith(
            _CONTROL_REASON_PREFIX
        ):
            value = bundle.run.terminal_reason.removeprefix(_CONTROL_REASON_PREFIX)
            return SlackCancellationOutcome(value)
        async with self._sessions() as session:
            parent_task_id = await session.scalar(
                select(TaskRow.parent_task_id).where(TaskRow.id == admitted.launch.task_id)
            )
            if parent_task_id is None:
                return SlackCancellationOutcome.NO_ACTIVE_TASK
            parent_reason = await session.scalar(
                select(RunRow.terminal_reason)
                .where(RunRow.task_id == parent_task_id)
                .order_by(RunRow.created_at.desc())
                .limit(1)
            )
        return (
            SlackCancellationOutcome.APPLIED
            if parent_reason == _TARGET_REASON
            else SlackCancellationOutcome.TERMINAL_RACE
        )

    async def _terminalize_control(
        self,
        admitted: AdmittedSlackMention,
        outcome: SlackCancellationOutcome,
    ) -> SlackCancellationResult:
        if admitted.launch is None:
            raise RuntimeError("Slack cancellation control has no durable launch")
        launch = admitted.launch
        bundle = await self._runs.load(
            launch.task_id,
            launch.run_id,
            admitted.resolution.scope,
        )
        reason = f"{_CONTROL_REASON_PREFIX}{outcome.value}"
        if bundle.run.status is not RunStatus.CANCELLED:
            task, run = cancel_task_and_run(
                bundle.task,
                bundle.run,
                reason,
                usage=bundle.run.usage,
            )
            async with self._sessions() as session:
                task_row = await session.scalar(select(TaskRow).where(TaskRow.id == launch.task_id))
            if task_row is None:
                raise RuntimeError("Slack cancellation control Task disappeared")
            bundle = await self._runs.commit(
                expected_task_version=bundle.task.version,
                expected_run_version=bundle.run.version,
                task=task,
                run=run,
                events=(
                    EventDraft(
                        type=EventType.RUN_CANCELLED,
                        iteration=run.iteration,
                        payload={"reason": reason},
                    ),
                ),
                lease_owner=task_row.lease_owner,
                lease_token=task_row.lease_token,
            )
        elif bundle.run.terminal_reason != reason:
            outcome = SlackCancellationOutcome.TERMINAL_RACE
        expected_status = f"cancel_control_{outcome.value}"
        async with self._sessions() as session:
            current_status = await session.scalar(
                select(SlackIngressEventRow.status).where(
                    SlackIngressEventRow.event_id == admitted.job.event_id
                )
            )
        if current_status != expected_status:
            await self._ingress.mark_linked_status(
                admitted.job.event_id,
                expected_status,
                None,
            )
        return SlackCancellationResult(
            admitted=admitted,
            outcome=outcome,
            message=cancellation_message(outcome),
        )

    async def _terminate_active_plans(
        self,
        *,
        scope: ScopeKey,
        task_id: str,
        run_id: str,
        reason: str,
    ) -> None:
        async with self._sessions() as session:
            plan_ids = tuple(
                (
                    await session.scalars(
                        select(PlanRow.id)
                        .where(
                            PlanRow.organization_id == scope.organization_id,
                            PlanRow.strategy_id == scope.strategy_id,
                            PlanRow.parent_task_id == task_id,
                            PlanRow.parent_run_id == run_id,
                            PlanRow.status == "active",
                        )
                        .order_by(PlanRow.id)
                    )
                ).all()
            )
        for plan_id in plan_ids:
            await self._plans.terminate_for_parent(
                scope=scope,
                plan_id=plan_id,
                parent_task_id=task_id,
                parent_run_id=run_id,
                parent_status=RunStatus.CANCELLED,
                reason=reason,
                child_terminal_reason="parent_cancelled",
            )
