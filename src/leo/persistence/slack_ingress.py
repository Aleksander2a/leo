"""Durable Slack event admission for duplicate suppression across process restarts."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql.elements import ColumnElement

from leo.harness.models import OriginRef, Run, ScopeKey, Task, Thread
from leo.harness.persistence_rules import validate_seed
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackContextProjectionSource,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationPolicyRejected,
    SlackLaunchRef,
    SlackMentionJob,
    SlackPassiveMessage,
    SlackScopeResolution,
)
from leo.persistence.conversation_plane import (
    ConversationMessageRole,
    build_conversation_plane_message,
    canonical_slack_conversation_id,
    persist_conversation_plane_message,
)
from leo.persistence.schema import (
    ConversationAccessSnapshotRow,
    ConversationActorMembershipRow,
    ConversationRow,
    MemoryCapabilityHandleRow,
    MemoryRetrievalCacheRow,
    RunRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    SlackThreadCoverageRow,
    TaskRow,
    ThreadRow,
)
from leo.persistence.slack_messages import (
    PersistedSlackThreadSnapshot,
    PostgresSlackMessagePlane,
    SlackThreadCoverageSource,
)
from leo.persistence.slack_scope import (
    SlackScopeStoreInvariantError,
    resolve_or_provision_in_session,
)


class PostgresSlackIngressAdmission:
    """Atomically admit one Slack event with an exact conversation-access snapshot."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        fault_hook: Callable[[str], None] | None = None,
    ) -> None:
        self._sessions = sessions
        self._fault_hook = fault_hook

    def _fault(self, stage: str) -> None:
        if self._fault_hook is not None:
            self._fault_hook(stage)

    async def preflight(self) -> None:
        """Verify connectivity and required schema before Socket Mode starts accepting events."""

        async with self._sessions() as session:
            await session.execute(
                select(
                    SlackIngressEventRow.event_id,
                    SlackIngressEventRow.organization_id,
                    SlackIngressEventRow.strategy_id,
                    SlackIngressEventRow.mapping_version,
                    SlackIngressEventRow.conversation_kind,
                    SlackIngressEventRow.trigger_kind,
                    SlackIngressEventRow.context_conversation_ids,
                    SlackIngressEventRow.context_access_hash,
                    SlackIngressEventRow.context_projection_source,
                    SlackIngressEventRow.conversation_authority_source,
                    SlackIngressEventRow.bot_presence,
                    SlackIngressEventRow.conversation_lifecycle,
                    SlackIngressEventRow.external_provenance,
                    SlackIngressEventRow.membership_policy_version,
                    SlackIngressEventRow.conversation_id,
                ).limit(1)
            )
            await session.execute(select(ConversationAccessSnapshotRow.id).limit(1))
            await session.execute(select(ConversationActorMembershipRow.id).limit(1))
            await session.execute(
                select(
                    SanitizedMessageRow.conversation_id,
                    SanitizedMessageRow.harness_thread_id,
                    SanitizedMessageRow.role,
                    SanitizedMessageRow.provider_thread_root_ts,
                    SanitizedMessageRow.context_access_hash,
                ).limit(1)
            )
            await session.execute(select(SlackThreadCoverageRow.id).limit(1))

    async def record_passive_message(
        self,
        message: SlackPassiveMessage,
        default_scope: ScopeKey,
    ) -> None:
        await PostgresSlackMessagePlane(self._sessions).record_passive_message(
            message,
            default_scope,
        )

    async def load_complete_thread(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        current_actor_id: str,
        current_event_id: str,
        max_messages: int = 500,
    ) -> PersistedSlackThreadSnapshot:
        return await PostgresSlackMessagePlane(self._sessions).load_complete_thread(
            team_id=team_id,
            channel_id=channel_id,
            thread_root_ts=thread_root_ts,
            current_message_ts=current_message_ts,
            current_actor_id=current_actor_id,
            current_event_id=current_event_id,
            max_messages=max_messages,
        )

    async def record_root_coverage(
        self,
        *,
        team_id: str,
        channel_id: str,
        thread_root_ts: str,
        current_message_ts: str,
        raw_root: Mapping[str, object],
        source: SlackThreadCoverageSource,
    ) -> bool:
        return await PostgresSlackMessagePlane(self._sessions).record_root_coverage(
            team_id=team_id,
            channel_id=channel_id,
            thread_root_ts=thread_root_ts,
            current_message_ts=current_message_ts,
            raw_root=raw_root,
            source=source,
        )

    async def admit(
        self,
        job: SlackMentionJob,
        default_scope: ScopeKey,
        *,
        eligibility: SlackConversationEligibility,
    ) -> AdmittedSlackMention | None:
        """Return a committed scoped mention, or ``None`` for a repeated event ID.

        Legacy scope columns record configured domain defaults only.  No channel-mapping row
        or mapping status participates in this decision.
        """

        if (
            not eligibility.admissible
            or eligibility.kind is not job.conversation_kind
            or eligibility.provenance != job.conversation_authority_source
            or eligibility.bot_presence is not job.bot_presence
            or eligibility.lifecycle is not job.conversation_lifecycle
            or eligibility.external_provenance is not job.external_provenance
            or eligibility.membership_policy_version != job.membership_policy_version
        ):
            raise SlackConversationPolicyRejected(eligibility)
        async with self._sessions() as session, session.begin():
            # Serialize the immutable event identity before touching mutable
            # conversation authority. Without this transaction-scoped fence, a
            # concurrent duplicate can lose the ingress INSERT but still increment
            # ConversationRow.version through _ensure_conversation(). Hash
            # collisions only over-serialize unrelated events; they never broaden
            # admission authority.
            await session.execute(
                select(
                    func.pg_advisory_xact_lock(
                        func.hashtext("leo-slack-ingress-event-v1"),
                        func.hashtext(job.event_id),
                    )
                )
            )
            existing = await session.scalar(
                select(SlackIngressEventRow)
                .where(SlackIngressEventRow.event_id == job.event_id)
                .with_for_update()
            )
            if existing is not None:
                if not _same_event_envelope(existing, job):
                    raise SlackScopeStoreInvariantError(
                        "Slack ingress event ID was reused with a different envelope"
                    )
                return None

            conversation = await _ensure_conversation(session, job)
            reservation = (
                postgres_insert(SlackIngressEventRow)
                .values(
                    event_id=job.event_id,
                    team_id=job.team_id,
                    channel_id=job.channel_id,
                    user_id=job.user_id,
                    message_ts=job.message_ts,
                    thread_root_ts=job.thread_root_ts,
                    conversation_key=job.conversation_key,
                    prompt=job.prompt,
                    conversation_kind=job.conversation_kind.value,
                    trigger_kind=job.trigger_kind.value,
                    context_conversation_ids=list(job.context_conversation_ids),
                    context_access_hash=job.context_access_hash,
                    context_projection_source=job.context_projection_source.value,
                    conversation_authority_source=job.conversation_authority_source,
                    bot_presence=job.bot_presence.value,
                    conversation_lifecycle=job.conversation_lifecycle.value,
                    external_provenance=job.external_provenance.value,
                    membership_policy_version=job.membership_policy_version,
                    conversation_id=conversation.id,
                    status="admitting",
                    launch_status="admitting",
                    attempt_count=0,
                )
                .on_conflict_do_nothing(index_elements=[SlackIngressEventRow.event_id])
                .returning(SlackIngressEventRow.event_id)
            )
            if (await session.execute(reservation)).scalar_one_or_none() is None:
                # Defensive for a legacy writer that does not yet take the fence.
                existing = await session.scalar(
                    select(SlackIngressEventRow)
                    .where(SlackIngressEventRow.event_id == job.event_id)
                    .with_for_update()
                )
                if existing is None:
                    raise SlackScopeStoreInvariantError(
                        "Slack ingress event disappeared after an ID conflict"
                    )
                if not _same_event_envelope(existing, job):
                    raise SlackScopeStoreInvariantError(
                        "Slack ingress event ID was reused with a different envelope"
                    )
                return None
            self._fault("admission_after_reserve")

            resolution = await resolve_or_provision_in_session(
                session,
                team_id=job.team_id,
                channel_id=job.channel_id,
                user_id=job.user_id,
                default_scope=default_scope,
                eligibility=eligibility,
            )
            self._fault("admission_after_scope")
            observed_at = datetime.now(UTC)
            await _persist_context_authority(
                session,
                job=job,
                scope=resolution.scope,
                observed_at=observed_at,
            )
            await persist_conversation_plane_message(
                session,
                build_conversation_plane_message(
                    scope=resolution.scope,
                    conversation_id=conversation.id,
                    harness_thread_id=None,
                    destination_id=job.channel_id,
                    external_event_id=job.event_id,
                    actor_id=job.user_id,
                    role=ConversationMessageRole.USER,
                    provider_message_ts=job.message_ts,
                    provider_thread_root_ts=job.thread_root_ts,
                    context_access_hash=job.context_access_hash,
                    text=job.prompt,
                    recorded_at=observed_at,
                ),
            )
            await session.execute(
                update(SlackIngressEventRow)
                .where(SlackIngressEventRow.event_id == job.event_id)
                .values(
                    organization_id=resolution.scope.organization_id,
                    strategy_id=resolution.scope.strategy_id,
                    mapping_version=resolution.mapping_version,
                    status="received",
                    launch_status="unlaunched",
                    last_error=None,
                    launch_error=None,
                    launch_updated_at=func.now(),
                )
            )
            self._fault("admission_before_commit")

        self._fault("admission_after_commit")
        return AdmittedSlackMention(job=job, resolution=resolution)

    async def release(self, event_id: str) -> None:
        """Release only an unstarted admission when the local queue rejects it."""

        async with self._sessions() as session, session.begin():
            await session.execute(
                delete(SlackIngressEventRow).where(
                    SlackIngressEventRow.event_id == event_id,
                    SlackIngressEventRow.status == "received",
                    SlackIngressEventRow.task_id.is_(None),
                )
            )

    async def materialize_initial_launch(
        self,
        *,
        event_id: str,
        thread: Thread,
        task: Task,
        run: Run,
    ) -> SlackLaunchMaterialization:
        """Persist one unambiguous initial launch before any volatile wake-up."""

        validate_seed(thread, task, run)
        async with self._sessions() as session, session.begin():
            ingress = await session.scalar(
                select(SlackIngressEventRow)
                .where(SlackIngressEventRow.event_id == event_id)
                .with_for_update()
            )
            if ingress is None:
                raise SlackLaunchInvariantError("Slack ingress event not found")
            if ingress.launch_status == "queued":
                if ingress.task_id is None:
                    raise SlackLaunchInvariantError("queued launch has no task link")
                existing_task = await session.scalar(
                    select(TaskRow).where(TaskRow.id == ingress.task_id)
                )
                existing_thread = None
                if existing_task is not None:
                    existing_thread = await session.scalar(
                        select(ThreadRow).where(ThreadRow.id == existing_task.thread_id)
                    )
                existing_run = await session.scalar(
                    select(RunRow)
                    .where(RunRow.task_id == ingress.task_id)
                    .order_by(RunRow.created_at, RunRow.id)
                    .limit(1)
                )
                if (
                    existing_task is None
                    or existing_thread is None
                    or existing_run is None
                    or not _matches_persisted_launch(
                        ingress, event_id, existing_thread, existing_task, existing_run
                    )
                ):
                    raise SlackLaunchInvariantError("queued launch is incomplete or inconsistent")
                return SlackLaunchMaterialization(
                    thread_id=existing_thread.id,
                    task_id=existing_task.id,
                    run_id=existing_run.id,
                    created=False,
                )
            if ingress.launch_status not in {"unlaunched", "failed"}:
                raise SlackLaunchInvariantError(
                    f"launch is not materializable from {ingress.launch_status}"
                )
            if ingress.task_id is not None:
                raise SlackLaunchInvariantError("unqueued launch already has a task link")
            if not _matches_initial_launch(ingress, event_id, thread, task, run):
                raise SlackLaunchInvariantError("launch identity or scope does not match ingress")
            if ingress.mapping_version is None:
                raise SlackLaunchInvariantError("launch has no mapping-version snapshot")

            existing_thread = await session.scalar(
                select(ThreadRow).where(
                    ThreadRow.origin_provider == thread.origin.provider,
                    ThreadRow.external_thread_id == thread.origin.external_thread_id,
                )
            )
            effective_thread = thread
            effective_task = task
            if existing_thread is not None:
                if existing_thread.organization_id != ingress.organization_id:
                    raise SlackLaunchInvariantError("thread organization changed")
                if existing_thread.conversation_id not in {None, ingress.conversation_id}:
                    raise SlackLaunchInvariantError("thread conversation changed")
                existing_thread.conversation_id = ingress.conversation_id
                # Optional strategy/domain metadata follows the existing conversation thread;
                # a default-strategy change must never make Leo unavailable.
                ingress.strategy_id = existing_thread.strategy_id
                ingress.mapping_version = existing_thread.mapping_version or 1
                effective_thread = Thread(
                    id=existing_thread.id,
                    scope=ScopeKey(
                        organization_id=existing_thread.organization_id,
                        strategy_id=existing_thread.strategy_id,
                    ),
                    origin=OriginRef(
                        provider=existing_thread.origin_provider,
                        external_thread_id=existing_thread.external_thread_id,
                        external_event_id=existing_thread.external_event_id,
                        external_channel_id=existing_thread.external_channel_id,
                    ),
                    mapping_version=existing_thread.mapping_version,
                    version=existing_thread.version,
                )
                effective_task = task.model_copy(
                    update={
                        "thread_id": effective_thread.id,
                        "scope": effective_thread.scope,
                    }
                )
                if ingress.thread_root_ts != ingress.message_ts:
                    active_task = await session.scalar(
                        select(TaskRow)
                        .where(
                            TaskRow.thread_id == effective_thread.id,
                            TaskRow.status.in_(("queued", "active", "requires_action")),
                        )
                        .limit(1)
                    )
                    if active_task is not None:
                        raise SlackLaunchInvariantError("thread has an active Task")
                    parent = await session.scalar(
                        select(TaskRow)
                        .where(TaskRow.thread_id == effective_thread.id)
                        .order_by(TaskRow.created_at.desc(), TaskRow.id.desc())
                        .limit(1)
                    )
                    if parent is not None:
                        effective_task = effective_task.model_copy(
                            update={
                                "parent_task_id": parent.id,
                                "continuation_kind": "follow_up",
                            }
                        )
            effective_thread = effective_thread.model_copy(
                update={"mapping_version": ingress.mapping_version}
            )
            effective_task = effective_task.model_copy(
                update={"mapping_version": ingress.mapping_version}
            )
            effective_run = run.model_copy(
                update={
                    "task_id": effective_task.id,
                    "scope": effective_task.scope,
                }
            )

            transitioned = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.task_id.is_(None),
                        SlackIngressEventRow.launch_status.in_(("unlaunched", "failed")),
                    )
                    .values(
                        launch_status="materializing",
                        launch_attempt_count=SlackIngressEventRow.launch_attempt_count + 1,
                        launch_error=None,
                        launch_updated_at=func.now(),
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if transitioned is None:
                raise SlackLaunchInvariantError("launch state changed before materialization")
            self._fault("launch_after_materializing")

            if existing_thread is None:
                session.add(
                    ThreadRow(
                        id=effective_thread.id,
                        organization_id=effective_thread.scope.organization_id,
                        strategy_id=effective_thread.scope.strategy_id,
                        origin_provider=effective_thread.origin.provider,
                        external_thread_id=effective_thread.origin.external_thread_id,
                        external_event_id=effective_thread.origin.external_event_id,
                        external_channel_id=effective_thread.origin.external_channel_id,
                        conversation_id=ingress.conversation_id,
                        mapping_version=effective_thread.mapping_version,
                        version=effective_thread.version,
                    )
                )
            await session.flush()
            # The harness thread's canonical conversation FK is the live link.  The legacy
            # domain conversation-thread table retains organization/strategy FKs, so writing it
            # here would incorrectly make optional domain setup an availability gate.
            await session.execute(
                update(SanitizedMessageRow)
                .where(
                    SanitizedMessageRow.conversation_id == ingress.conversation_id,
                    SanitizedMessageRow.external_event_id == event_id,
                    SanitizedMessageRow.role == ConversationMessageRole.USER.value,
                )
                .values(harness_thread_id=effective_thread.id)
            )
            session.add(
                TaskRow(
                    id=effective_task.id,
                    thread_id=effective_task.thread_id,
                    organization_id=effective_task.scope.organization_id,
                    strategy_id=effective_task.scope.strategy_id,
                    objective=effective_task.objective,
                    parent_task_id=effective_task.parent_task_id,
                    continuation_kind=effective_task.continuation_kind,
                    mapping_version=effective_task.mapping_version,
                    status=effective_task.status.value,
                    observation_ids=list(effective_task.observation_ids),
                    verifier_feedback=list(effective_task.verifier_feedback),
                    final_output=effective_task.final_output,
                    version=effective_task.version,
                )
            )
            await session.flush()
            session.add(
                RunRow(
                    id=effective_run.id,
                    task_id=effective_run.task_id,
                    organization_id=effective_run.scope.organization_id,
                    strategy_id=effective_run.scope.strategy_id,
                    status=effective_run.status.value,
                    phase=effective_run.phase.value,
                    iteration=effective_run.iteration,
                    limits=cast(dict[str, object], effective_run.limits.model_dump(mode="json")),
                    usage=cast(dict[str, object], effective_run.usage.model_dump(mode="json")),
                    started_at=effective_run.started_at,
                    deadline_at=effective_run.deadline_at,
                    final_output=effective_run.final_output,
                    terminal_reason=effective_run.terminal_reason,
                    event_sequence=0,
                    version=effective_run.version,
                )
            )
            await session.flush()
            await session.execute(
                update(SlackIngressEventRow)
                .where(SlackIngressEventRow.event_id == event_id)
                .values(
                    task_id=effective_task.id,
                    status="queued",
                    launch_status="queued",
                    launch_error=None,
                    launch_updated_at=func.now(),
                )
            )
            self._fault("launch_before_commit")

        self._fault("launch_after_commit")
        return SlackLaunchMaterialization(
            thread_id=effective_thread.id,
            task_id=effective_task.id,
            run_id=effective_run.id,
            created=True,
        )

    async def recover_startup_launches(
        self,
        seed_factory: Callable[[SlackMentionJob, ScopeKey], tuple[Thread, Task, Run]],
        *,
        limit: int = 100,
        include_queued: bool = True,
        max_attempts: int = 3,
        event_ids: tuple[str, ...] | None = None,
    ) -> tuple[AdmittedSlackMention, ...]:
        """Materialize or re-signal committed launches after process restart."""

        if limit < 1:
            raise ValueError("limit must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if event_ids is not None and (
            not event_ids
            or len(event_ids) != len(set(event_ids))
            or any(not event_id.strip() for event_id in event_ids)
        ):
            raise ValueError("event_ids must be non-empty, unique event identities")
        launch_predicate = SlackIngressEventRow.launch_status.in_(
            ("unlaunched", "failed")
        ) & SlackIngressEventRow.task_id.is_(None)
        if include_queued:
            launch_predicate |= _recoverable_linked_launch_predicate(max_attempts)
        recovery_filters: list[ColumnElement[bool]] = [launch_predicate]
        if event_ids is not None:
            recovery_filters.append(SlackIngressEventRow.event_id.in_(event_ids))
        async with self._sessions() as session:
            rows = list(
                (
                    await session.scalars(
                        select(SlackIngressEventRow)
                        .where(*recovery_filters)
                        .order_by(
                            # Immutable admission order keeps same-thread busy follow-ups
                            # FIFO even though each recovery attempt updates launch metadata.
                            SlackIngressEventRow.received_at,
                            SlackIngressEventRow.event_id,
                        )
                        .limit(limit)
                    )
                ).all()
            )

        recovered: list[AdmittedSlackMention] = []
        for row in rows:
            job = _job_from_row(row)
            scope = _ingress_scope(row)
            if row.mapping_version is None:
                raise SlackLaunchInvariantError("ingress has no committed mapping version")
            if row.launch_status == "queued":
                materialized = await self._queued_materialization(row)
            else:
                thread, task, run = seed_factory(job, scope)
                try:
                    materialized = await self.materialize_initial_launch(
                        event_id=job.event_id,
                        thread=thread,
                        task=task,
                        run=run,
                    )
                except SlackLaunchInvariantError as exc:
                    if str(exc) != "thread has an active Task":
                        raise
                    if row.status != "followup_pending":
                        await self.mark_followup_pending(
                            job.event_id,
                            "thread_task_active",
                        )
                    continue
                if not materialized.created:
                    continue
            recovered.append(await self.load_linked_mention(materialized.task_id))
        return tuple(recovered)

    async def load_linked_mention(self, task_id: str) -> AdmittedSlackMention:
        """Reload the canonical Slack admission behind a claimed Task ID."""

        if not task_id:
            raise ValueError("task_id must be non-empty")
        async with self._sessions() as session:
            row = await session.scalar(
                select(SlackIngressEventRow).where(
                    SlackIngressEventRow.task_id == task_id,
                    SlackIngressEventRow.launch_status == "queued",
                )
            )
        if row is None:
            raise SlackLaunchInvariantError("claimed task has no queued Slack admission")
        scope = _ingress_scope(row)
        if row.mapping_version is None:
            raise SlackLaunchInvariantError("ingress has no committed mapping version")
        materialized = await self._queued_materialization(row)
        return AdmittedSlackMention(
            job=_job_from_row(row),
            resolution=SlackScopeResolution(
                scope=scope,
                mapping_version=row.mapping_version,
                provisioned=False,
            ),
            launch=SlackLaunchRef(
                thread_id=materialized.thread_id,
                task_id=materialized.task_id,
                run_id=materialized.run_id,
            ),
        )

    async def recover_control_requests(
        self,
        predicate: Callable[[str], bool],
        *,
        limit: int = 100,
    ) -> tuple[AdmittedSlackMention, ...]:
        """Reload bounded durable control requests before ordinary launch recovery.

        The predicate classifies content only; every identity, scope and launch
        reference in the returned contracts is reconstructed from persisted
        admission authority.  Queued control Tasks are included so a crash after
        materialization but before terminalization can be repaired.
        """

        if limit < 1:
            raise ValueError("limit must be positive")
        async with self._sessions() as session:
            rows = tuple(
                (
                    await session.scalars(
                        select(SlackIngressEventRow)
                        .where(
                            SlackIngressEventRow.launch_status.in_(
                                ("unlaunched", "failed", "queued")
                            )
                        )
                        .order_by(
                            SlackIngressEventRow.received_at,
                            SlackIngressEventRow.event_id,
                        )
                        .limit(limit)
                    )
                ).all()
            )
        recovered: list[AdmittedSlackMention] = []
        for row in rows:
            if not predicate(row.prompt):
                continue
            if row.task_id is not None:
                recovered.append(await self.load_linked_mention(row.task_id))
                continue
            if row.mapping_version is None:
                raise SlackLaunchInvariantError("control ingress has no committed mapping version")
            recovered.append(
                AdmittedSlackMention(
                    job=_job_from_row(row),
                    resolution=SlackScopeResolution(
                        scope=_ingress_scope(row),
                        mapping_version=row.mapping_version,
                        provisioned=False,
                    ),
                )
            )
        return tuple(recovered)

    async def _queued_materialization(
        self, ingress: SlackIngressEventRow
    ) -> SlackLaunchMaterialization:
        if ingress.task_id is None:
            raise SlackLaunchInvariantError("queued launch has no task link")
        async with self._sessions() as session:
            task = await session.scalar(select(TaskRow).where(TaskRow.id == ingress.task_id))
            run = await session.scalar(
                select(RunRow)
                .where(RunRow.task_id == ingress.task_id)
                .order_by(RunRow.created_at, RunRow.id)
                .limit(1)
            )
            thread = (
                None
                if task is None
                else await session.scalar(select(ThreadRow).where(ThreadRow.id == task.thread_id))
            )
        if task is None or run is None or thread is None:
            raise SlackLaunchInvariantError("queued launch is incomplete")
        if not _matches_persisted_launch(ingress, ingress.event_id, thread, task, run):
            raise SlackLaunchInvariantError(
                "queued launch identity or scope does not match ingress"
            )
        return SlackLaunchMaterialization(
            thread_id=thread.id,
            task_id=task.id,
            run_id=run.id,
            created=False,
        )

    async def attach_task(self, event_id: str, task_id: str, status: str) -> None:
        if not event_id or not task_id or not status:
            raise ValueError("event_id, task_id, and status must be non-empty")
        async with self._sessions() as session, session.begin():
            attached = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.status == "received",
                        SlackIngressEventRow.task_id.is_(None),
                    )
                    .values(
                        task_id=task_id,
                        status=status,
                        attempt_count=SlackIngressEventRow.attempt_count + 1,
                        last_error=None,
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if attached is None:
                raise SlackScopeStoreInvariantError(
                    "Slack ingress task attachment did not match one received event"
                )

    async def mark_linked_status(self, event_id: str, status: str, safe_error: str | None) -> None:
        if not event_id or not status:
            raise ValueError("event_id and status must be non-empty")
        async with self._sessions() as session, session.begin():
            marked = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.task_id.is_not(None),
                        SlackIngressEventRow.launch_status == "queued",
                    )
                    .values(
                        status=status,
                        attempt_count=SlackIngressEventRow.attempt_count + 1,
                        last_error=safe_error,
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if marked is None:
                raise SlackScopeStoreInvariantError(
                    "Slack ingress linked status update did not match one queued launch"
                )

    async def mark_failed(self, event_id: str, safe_error: str) -> None:
        if not event_id or not safe_error:
            raise ValueError("event_id and safe_error must be non-empty")
        async with self._sessions() as session, session.begin():
            marked = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.status == "received",
                        SlackIngressEventRow.task_id.is_(None),
                    )
                    .values(
                        status="runtime_failed",
                        attempt_count=SlackIngressEventRow.attempt_count + 1,
                        last_error=safe_error,
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if marked is None:
                raise SlackScopeStoreInvariantError(
                    "Slack ingress failure update did not match one received event"
                )

    async def mark_followup_rejected(self, event_id: str, safe_error: str) -> None:
        """Persist a safe terminal policy outcome without creating a child Task."""

        if not event_id or not safe_error:
            raise ValueError("event_id and safe_error must be non-empty")
        async with self._sessions() as session, session.begin():
            marked = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.task_id.is_(None),
                        SlackIngressEventRow.launch_status.in_(("unlaunched", "failed")),
                    )
                    .values(
                        status="followup_rejected",
                        launch_status="rejected",
                        launch_error=safe_error,
                        launch_attempt_count=SlackIngressEventRow.launch_attempt_count + 1,
                        launch_updated_at=func.now(),
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if marked is None:
                raise SlackScopeStoreInvariantError(
                    "follow-up rejection did not match one unlaunched event"
                )

    async def mark_followup_pending(self, event_id: str, safe_error: str) -> None:
        """Keep a busy follow-up FIFO-eligible without creating a concurrent Task."""

        if not event_id or not safe_error:
            raise ValueError("event_id and safe_error must be non-empty")
        async with self._sessions() as session, session.begin():
            marked = (
                await session.execute(
                    update(SlackIngressEventRow)
                    .where(
                        SlackIngressEventRow.event_id == event_id,
                        SlackIngressEventRow.task_id.is_(None),
                        SlackIngressEventRow.launch_status.in_(("unlaunched", "failed")),
                    )
                    .values(
                        status="followup_pending",
                        launch_status="unlaunched",
                        launch_error=safe_error,
                        launch_attempt_count=SlackIngressEventRow.launch_attempt_count + 1,
                        launch_updated_at=func.now(),
                    )
                    .returning(SlackIngressEventRow.event_id)
                )
            ).scalar_one_or_none()
            if marked is None:
                raise SlackScopeStoreInvariantError(
                    "follow-up pending update did not match one unlaunched event"
                )


def _same_event_envelope(row: SlackIngressEventRow, job: SlackMentionJob) -> bool:
    return (
        row.team_id == job.team_id
        and row.channel_id == job.channel_id
        and row.user_id == job.user_id
        and row.message_ts == job.message_ts
        and row.thread_root_ts == job.thread_root_ts
        and row.conversation_key == job.conversation_key
        and row.prompt == job.prompt
        and row.conversation_kind == job.conversation_kind.value
        and row.trigger_kind == job.trigger_kind.value
        and tuple(row.context_conversation_ids) == job.context_conversation_ids
        and row.context_access_hash == job.context_access_hash
        and row.context_projection_source == job.context_projection_source.value
        and row.conversation_authority_source == job.conversation_authority_source
        and row.bot_presence == job.bot_presence.value
        and row.conversation_lifecycle == job.conversation_lifecycle.value
        and row.external_provenance == job.external_provenance.value
        and row.membership_policy_version == job.membership_policy_version
    )


@dataclass(frozen=True, slots=True)
class SlackLaunchMaterialization:
    thread_id: str
    task_id: str
    run_id: str
    created: bool = True


class SlackLaunchInvariantError(RuntimeError):
    """Persisted launch state or proposed seed violated the ingress contract."""


class SlackFollowupBusyError(SlackLaunchInvariantError):
    """A follow-up was rejected because its Thread already has active work."""

    safe_code = "thread_task_active"


async def _persist_context_authority(
    session: AsyncSession,
    *,
    job: SlackMentionJob,
    scope: ScopeKey,
    observed_at: datetime,
) -> None:
    """Persist immutable turn authority plus the latest actor-and-Leo membership view."""

    snapshot_values = [
        {
            "id": _access_snapshot_id(job.event_id, conversation_id),
            "ingress_event_id": job.event_id,
            "organization_id": scope.organization_id,
            "team_id": job.team_id,
            "actor_id": job.user_id,
            "destination_external_id": job.channel_id,
            "conversation_external_id": conversation_id,
            "position": position,
            "source_kind": job.context_projection_source.value,
            "context_access_hash": job.context_access_hash,
            "observed_at": observed_at,
        }
        for position, conversation_id in enumerate(job.context_conversation_ids)
    ]
    await session.execute(
        postgres_insert(ConversationAccessSnapshotRow)
        .values(snapshot_values)
        .on_conflict_do_nothing(
            constraint="uq_conversation_access_snapshot_source",
        )
    )

    membership_values = [
        {
            "id": _actor_membership_id(job.team_id, job.user_id, conversation_id),
            "organization_id": scope.organization_id,
            "team_id": job.team_id,
            "actor_id": job.user_id,
            "conversation_external_id": conversation_id,
            "status": "active",
            "source_kind": job.context_projection_source.value,
            "context_access_hash": job.context_access_hash,
            "version": 1,
            "observed_at": observed_at,
        }
        for conversation_id in job.context_conversation_ids
    ]
    membership_insert = postgres_insert(ConversationActorMembershipRow).values(membership_values)
    await session.execute(
        membership_insert.on_conflict_do_update(
            constraint="uq_conversation_actor_membership",
            set_={
                "organization_id": membership_insert.excluded.organization_id,
                "status": "active",
                "source_kind": membership_insert.excluded.source_kind,
                "context_access_hash": membership_insert.excluded.context_access_hash,
                "version": ConversationActorMembershipRow.version + 1,
                "observed_at": membership_insert.excluded.observed_at,
                "updated_at": func.now(),
            },
        )
    )
    if (
        job.conversation_kind is SlackConversationKind.DM
        and job.context_projection_source is SlackContextProjectionSource.DM_MEMBERSHIP_INTERSECTION
    ):
        await session.execute(
            update(ConversationActorMembershipRow)
            .where(
                ConversationActorMembershipRow.team_id == job.team_id,
                ConversationActorMembershipRow.actor_id == job.user_id,
                ConversationActorMembershipRow.status == "active",
                ConversationActorMembershipRow.conversation_external_id.not_in(
                    job.context_conversation_ids
                ),
            )
            .values(
                status="revoked",
                source_kind=job.context_projection_source.value,
                context_access_hash=job.context_access_hash,
                version=ConversationActorMembershipRow.version + 1,
                observed_at=observed_at,
                updated_at=func.now(),
            )
        )
    # A refreshed or revoked actor/Leo projection must not reuse results or capabilities
    # issued under the prior source-set snapshot. Safe workspace/actor over-invalidation
    # happens in the same transaction as the new membership authority.
    await session.execute(
        delete(MemoryRetrievalCacheRow).where(
            MemoryRetrievalCacheRow.organization_id == scope.organization_id,
        )
    )
    await session.execute(
        update(MemoryCapabilityHandleRow)
        .where(
            MemoryCapabilityHandleRow.organization_id == scope.organization_id,
            MemoryCapabilityHandleRow.team_id == job.team_id,
            MemoryCapabilityHandleRow.actor_id == job.user_id,
            MemoryCapabilityHandleRow.invalidated_at.is_(None),
        )
        .values(
            invalidated_at=observed_at,
            invalidation_reason="conversation_authority_refreshed",
            updated_at=observed_at,
        )
    )


def _access_snapshot_id(event_id: str, conversation_id: str) -> str:
    digest = hashlib.sha256(f"{event_id}\x1f{conversation_id}".encode()).hexdigest()
    return f"access-{digest[:57]}"


def _actor_membership_id(team_id: str, actor_id: str, conversation_id: str) -> str:
    digest = hashlib.sha256(f"{team_id}\x1f{actor_id}\x1f{conversation_id}".encode()).hexdigest()
    return f"membership-{digest[:53]}"


async def _ensure_conversation(
    session: AsyncSession,
    job: SlackMentionJob,
) -> ConversationRow:
    kind = {
        SlackConversationKind.ORDINARY_INTERNAL: "channel",
        SlackConversationKind.DM: "dm",
        SlackConversationKind.MPIM: "group_dm",
        SlackConversationKind.SHARED: "shared",
        SlackConversationKind.EXTERNAL: "external",
    }[job.conversation_kind]
    actor_id = job.user_id if job.conversation_kind is SlackConversationKind.DM else None
    await session.execute(
        postgres_insert(ConversationRow)
        .values(
            id=canonical_slack_conversation_id(job.team_id, job.channel_id),
            provider="slack",
            team_id=job.team_id,
            external_id=job.channel_id,
            kind=kind,
            actor_id=actor_id,
            authority_source=job.conversation_authority_source,
            bot_presence=job.bot_presence.value,
            lifecycle=job.conversation_lifecycle.value,
            external_provenance=job.external_provenance.value,
            membership_policy_version=job.membership_policy_version,
            version=1,
        )
        .on_conflict_do_update(
            constraint="uq_conversations_provider_external",
            set_={
                "kind": kind,
                "actor_id": actor_id,
                "authority_source": job.conversation_authority_source,
                "bot_presence": job.bot_presence.value,
                "lifecycle": job.conversation_lifecycle.value,
                "external_provenance": job.external_provenance.value,
                "membership_policy_version": job.membership_policy_version,
                "version": ConversationRow.version + 1,
                "updated_at": func.now(),
            },
        )
    )
    conversation = await session.scalar(
        select(ConversationRow).where(
            ConversationRow.provider == "slack",
            ConversationRow.team_id == job.team_id,
            ConversationRow.external_id == job.channel_id,
        )
    )
    if conversation is None:
        raise SlackScopeStoreInvariantError("canonical Slack conversation was not persisted")
    if (
        conversation.kind != kind
        or conversation.actor_id != actor_id
        or conversation.authority_source != job.conversation_authority_source
        or conversation.bot_presence != job.bot_presence.value
        or conversation.lifecycle != job.conversation_lifecycle.value
        or conversation.external_provenance != job.external_provenance.value
        or conversation.membership_policy_version != job.membership_policy_version
    ):
        raise SlackScopeStoreInvariantError("canonical Slack conversation shape is inconsistent")
    return conversation


def _matches_initial_launch(
    ingress: SlackIngressEventRow,
    event_id: str,
    thread: Thread,
    task: Task,
    run: Run,
) -> bool:
    return (
        thread.scope == _ingress_scope(ingress)
        and task.scope == thread.scope
        and run.scope == task.scope
        and task.thread_id == thread.id
        and run.task_id == task.id
        and task.objective == ingress.prompt
        and thread.origin.provider == "slack"
        and thread.origin.external_thread_id == ingress.conversation_key
        and thread.origin.external_event_id == event_id
        and thread.origin.external_channel_id == ingress.channel_id
    )


def _recoverable_linked_launch_predicate(max_attempts: int) -> ColumnElement[bool]:
    """Select linked work that startup can either retry or safely terminalize.

    The ingress ``status`` is diagnostic and may record the process failure that
    abandoned the lease. Durable Task state and its lease are the recovery
    authority. Work below the attempt cap is re-signalled when its retry window is
    due; work at or above the cap is re-signalled so the runtime can fence it with
    ``claim_exhausted_task`` and persist one safe terminal result.
    """

    current_time = func.now()
    lease_available = or_(
        TaskRow.lease_expires_at.is_(None),
        TaskRow.lease_expires_at <= current_time,
    )
    retryable = (TaskRow.attempt_count < max_attempts) & or_(
        TaskRow.retry_after.is_(None),
        TaskRow.retry_after <= current_time,
    )
    exhausted = TaskRow.attempt_count >= max_attempts
    task_is_recoverable = (
        select(TaskRow.id)
        .where(
            TaskRow.id == SlackIngressEventRow.task_id,
            TaskRow.status.in_(("queued", "active")),
            lease_available,
            or_(retryable, exhausted),
        )
        .exists()
    )
    return (
        (SlackIngressEventRow.launch_status == "queued")
        & SlackIngressEventRow.task_id.is_not(None)
        & task_is_recoverable
    )


def _matches_persisted_launch(
    ingress: SlackIngressEventRow,
    event_id: str,
    thread: ThreadRow,
    task: TaskRow,
    run: RunRow,
) -> bool:
    return (
        thread.organization_id == ingress.organization_id
        and thread.strategy_id == ingress.strategy_id
        and thread.mapping_version == ingress.mapping_version
        and task.organization_id == ingress.organization_id
        and task.strategy_id == ingress.strategy_id
        and task.mapping_version == ingress.mapping_version
        and run.organization_id == ingress.organization_id
        and run.strategy_id == ingress.strategy_id
        and task.thread_id == thread.id
        and run.task_id == task.id
        and task.objective == ingress.prompt
        and thread.origin_provider == "slack"
        and thread.external_thread_id == ingress.conversation_key
        and (
            thread.external_event_id == event_id
            if ingress.thread_root_ts == ingress.message_ts
            else bool(thread.external_event_id)
        )
        and thread.external_channel_id == ingress.channel_id
        and thread.conversation_id == ingress.conversation_id
    )


def _job_from_row(row: SlackIngressEventRow) -> SlackMentionJob:
    return SlackMentionJob(
        event_id=row.event_id,
        team_id=row.team_id,
        channel_id=row.channel_id,
        user_id=row.user_id,
        message_ts=row.message_ts,
        thread_root_ts=row.thread_root_ts,
        conversation_key=row.conversation_key,
        prompt=row.prompt,
        conversation_kind=row.conversation_kind,
        trigger_kind=row.trigger_kind,
        context_conversation_ids=tuple(row.context_conversation_ids),
        context_access_hash=row.context_access_hash,
        context_projection_source=row.context_projection_source,
        conversation_authority_source=row.conversation_authority_source,
        bot_presence=row.bot_presence,
        conversation_lifecycle=row.conversation_lifecycle,
        external_provenance=row.external_provenance,
        membership_policy_version=row.membership_policy_version,
    )


def _ingress_scope(ingress: SlackIngressEventRow) -> ScopeKey:
    if (
        ingress.organization_id is None
        or ingress.strategy_id is None
        or ingress.mapping_version is None
    ):
        raise SlackLaunchInvariantError("ingress has no committed scope snapshot")
    return ScopeKey(
        organization_id=ingress.organization_id,
        strategy_id=ingress.strategy_id,
    )


# Transitional import compatibility; production wiring must call ``admit``, never a split claim.
PostgresSlackEventDeduplicator = PostgresSlackIngressAdmission
