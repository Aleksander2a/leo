"""Postgres-backed worker entry point for queued Slack launches."""

from __future__ import annotations

import asyncio

import httpx
from slack_sdk.web.async_client import AsyncWebClient

from leo.config import Environment, Settings
from leo.harness.models import (
    BudgetLimits,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
)
from leo.integrations.slack.context import (
    SlackHistoryContextLoader,
    slack_history_authority_ids,
)
from leo.integrations.slack.events import AdmittedSlackMention, SlackMentionJob
from leo.integrations.slack.render import (
    RENDERER_VERSION,
    RenderedSlackText,
    SlackTerminalResult,
    render_terminal_result,
    render_verified_result,
    verified_result_from_coordinator,
)
from leo.integrations.system import SystemClock, UuidIdGenerator
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.outbox import DeliveryKind, PostgresDeliveryOutbox, SlackOutboxDispatcher
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.schema import RunRow, TaskRow
from leo.persistence.slack_ingress import PostgresSlackIngressAdmission
from leo.persistence.task_leases import PostgresTaskLeaseStore, TaskLease
from leo.worker.runtime import DurableTaskWorker
from leo.worker.slack_conversation import run_admitted_slack_conversation
from leo.worker.terminal import (
    MAX_TASK_ATTEMPTS,
    RETRY_ATTEMPTS_EXHAUSTED,
    persist_safe_failure,
)


class _DurableSlackWorker:
    def __init__(self, settings: Settings) -> None:
        if settings.leo_env is not Environment.DEVELOPMENT:
            raise RuntimeError(
                "automatic Slack worker execution is limited to the development demo"
            )
        missing = tuple(
            dict.fromkeys(settings.missing_for_live_harness() + _missing_worker_slack(settings))
        )
        if missing:
            raise RuntimeError(f"missing durable worker configuration names: {', '.join(missing)}")
        assert settings.database_url is not None
        assert settings.slack_bot_token is not None
        self.settings = settings
        self.engine = create_database_engine(settings.database_url.get_secret_value())
        self.sessions = create_session_factory(self.engine)
        self.ids = UuidIdGenerator()
        self.clock = SystemClock()
        self.run_store = PostgresRunStore(self.sessions, self.clock, self.ids)
        self.ingress = PostgresSlackIngressAdmission(self.sessions)
        self.leases = PostgresTaskLeaseStore(self.sessions, self.ids)
        self.outbox = PostgresDeliveryOutbox(self.sessions, self.ids)
        self.dispatcher = SlackOutboxDispatcher(
            self.outbox,
            owner=f"worker-dispatcher-{self.ids.new('worker')}",
        )
        self.slack = AsyncWebClient(token=settings.slack_bot_token.get_secret_value())
        self.user_history_slack = (
            AsyncWebClient(token=settings.slack_user_token.get_secret_value())
            if settings.slack_user_token is not None
            else None
        )
        self.history = SlackHistoryContextLoader(
            self.slack,
            user_history_client=self.user_history_slack,
            thread_fallback=self.ingress,
        )
        self.owner = f"durable-worker-{self.ids.new('worker')}"

    async def close(self) -> None:
        await self.engine.dispose()

    async def run_once(self) -> bool:
        # The Socket Mode process treats its asyncio queue as a wake-up hint.  A
        # worker may therefore be the first process to observe an admitted event
        # after a crash or queue-pressure miss; materialize those durable launches
        # before claiming Tasks. Repair terminal runs and drain the durable outbox
        # here too, because the standalone worker may be the only surviving process.
        await self.ingress.recover_startup_launches(self._seed)
        await self.outbox.reconcile_terminal(
            _terminal_delivery_payload,
            payload_version=RENDERER_VERSION * 1000,
        )
        await self.dispatcher.dispatch_available(self.slack)
        lease_seconds = max(60.0, min(300.0, self.settings.leo_max_run_seconds))
        exhausted = await self.leases.claim_exhausted(
            self.owner,
            lease_seconds=lease_seconds,
            max_attempts=MAX_TASK_ATTEMPTS,
        )
        if exhausted is not None:
            await self._finalize_exhausted(exhausted)
            return True
        worker = DurableTaskWorker(
            leases=self.leases,
            owner=self.owner,
            handler=self.handle,
            lease_seconds=lease_seconds,
            max_attempts=MAX_TASK_ATTEMPTS,
            idle_wait_seconds=0.1,
        )
        return await worker.run_once()

    def _seed(self, job: SlackMentionJob, scope: ScopeKey) -> tuple[Thread, Task, Run]:
        thread = Thread(
            id=self.ids.new("thread"),
            scope=scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=job.conversation_key,
                external_event_id=job.event_id,
                external_channel_id=job.channel_id,
            ),
        )
        task = Task(
            id=self.ids.new("task"),
            thread_id=thread.id,
            scope=scope,
            objective=job.prompt,
        )
        run = Run(
            id=self.ids.new("run"),
            task_id=task.id,
            scope=scope,
            limits=BudgetLimits(
                max_iterations=self.settings.leo_max_model_turns,
                max_model_calls=self.settings.leo_max_model_turns,
                max_tool_calls=self.settings.leo_max_tool_calls,
                max_elapsed_seconds=self.settings.leo_max_run_seconds,
            ),
        )
        return thread, task, run

    async def handle(self, lease: TaskLease) -> None:
        admitted = await self.ingress.load_linked_mention(lease.task_id)
        if admitted.launch is None:
            raise RuntimeError("durable admission has no launch")
        history = await self.history.load(admitted.job)
        timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                result = await run_admitted_slack_conversation(
                    settings=self.settings,
                    client=client,
                    sessions=self.sessions,
                    admitted=admitted,
                    lease=lease,
                    additional_context=history.items,
                    thread_context_ranges=history.reopen_ranges,
                    additional_authority_ids=slack_history_authority_ids(history.manifest),
                )
        except Exception:
            if lease.attempt < MAX_TASK_ATTEMPTS:
                raise
            await self._finalize_exhausted(lease)
            return
        await self.ingress.mark_linked_status(
            admitted.job.event_id,
            f"run_{result.run.status.value}",
            None,
        )
        if result.run.status is RunStatus.COMPLETED and result.run.final_output:
            rendered = render_verified_result(
                verified_result_from_coordinator(result, include_evidence_details=False)
            )
        else:
            rendered = render_terminal_result(
                SlackTerminalResult(
                    run_id=result.run.id,
                    status=result.run.status,
                    terminal_reason=result.run.terminal_reason,
                    completed_output=result.run.final_output,
                )
            )
        await self._ensure_final(admitted, rendered)

    async def _finalize_exhausted(self, lease: TaskLease) -> None:
        admitted = await self.ingress.load_linked_mention(lease.task_id)
        if admitted.launch is None:
            raise RuntimeError("exhausted Slack admission has no launch")
        launch = admitted.launch
        bundle = await persist_safe_failure(
            LeaseBoundRunStore(self.run_store, lease),
            task_id=launch.task_id,
            run_id=launch.run_id,
            scope=admitted.resolution.scope,
            reason=RETRY_ATTEMPTS_EXHAUSTED,
            clock=self.clock,
        )
        await self.ingress.mark_linked_status(
            admitted.job.event_id,
            f"run_{bundle.run.status.value}",
            None,
        )
        rendered = render_terminal_result(
            SlackTerminalResult(
                run_id=bundle.run.id,
                status=bundle.run.status,
                terminal_reason=bundle.run.terminal_reason,
                completed_output=bundle.run.final_output,
            )
        )
        await self._ensure_final(admitted, rendered)

    async def _ensure_final(
        self,
        admitted: AdmittedSlackMention,
        rendered: RenderedSlackText,
    ) -> None:
        if admitted.launch is None:
            raise RuntimeError("durable admission has no launch")
        for index, chunk in enumerate(rendered.chunks):
            intent = await self.outbox.ensure_intent(
                task_id=admitted.launch.task_id,
                run_id=admitted.launch.run_id,
                ingress_event_id=admitted.job.event_id,
                kind=DeliveryKind.FINAL,
                payload_version=rendered.version * 1000 + index,
                payload=chunk,
            )
            await self.dispatcher.dispatch_once(self.slack, intent_id=intent.id)


async def run_once() -> bool:
    """Claim and process one durable Slack Task, returning whether work was found."""

    worker = _DurableSlackWorker(Settings())
    try:
        return await worker.run_once()
    finally:
        await worker.close()


def _missing_worker_slack(settings: Settings) -> tuple[str, ...]:
    return () if settings.slack_bot_token is not None else ("SLACK_BOT_TOKEN",)


def _terminal_delivery_payload(task: TaskRow, run: RunRow) -> str:
    del task
    return "\n".join(
        render_terminal_result(
            SlackTerminalResult(
                run_id=run.id,
                status=run.status,
                terminal_reason=run.terminal_reason,
                completed_output=run.final_output,
            )
        ).chunks
    )


def main() -> None:
    asyncio.run(run_once())


if __name__ == "__main__":
    main()
