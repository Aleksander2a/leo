"""Shared durable Slack-to-conversational-harness composition."""

from __future__ import annotations

import json

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.domain.conversation import normalize_conversation_kind
from leo.harness.context_budget import BudgetSegment, ContextBudget, assemble_budgeted_context
from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    CoordinatorResult,
    EventDraft,
    EventType,
    OriginRef,
    RunBundle,
    RunStatus,
    ScopeKey,
    TaskStatus,
    TrustedScope,
)
from leo.harness.ports import Clock, RunStore
from leo.harness.store_errors import ConcurrencyError
from leo.harness.thread_context import ThreadContextRange
from leo.harness.thread_context_tools import (
    ThreadContextAuthority,
    build_thread_context_tools,
)
from leo.harness.transitions import start_task_and_run, time_out_task_and_run
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.slack.events import (
    AdmittedSlackMention,
    SlackConversationKind,
)
from leo.integrations.system import SystemClock, UuidIdGenerator
from leo.live import run_live_conversation
from leo.memory.navigation import (
    MemoryNavigationAuthority,
    ProgressiveMemoryItem,
)
from leo.memory.tools import bind_memory_mutation_authority
from leo.persistence.context_loader import (
    ConversationContextRequest,
    PostgresConversationContextLoader,
)
from leo.persistence.memory_navigation import PostgresProgressiveMemoryService
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.schema import PlanRow, RunRow
from leo.persistence.task_leases import TaskLease

_TERMINAL_RUN_STATUSES = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
        RunStatus.BUDGET_EXHAUSTED,
    }
)


async def run_admitted_slack_conversation(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    sessions: async_sessionmaker[AsyncSession],
    admitted: AdmittedSlackMention,
    lease: TaskLease,
    additional_context: tuple[ContextItem, ...] = (),
    additional_authority_ids: tuple[str, ...] = (),
    thread_context_ranges: tuple[ThreadContextRange, ...] = (),
    provider_gates: ProviderGateRegistry | None = None,
) -> CoordinatorResult:
    """Execute arbitrary admitted Slack text with its trusted context projection."""

    if admitted.launch is None:
        raise RuntimeError("Slack conversation runtime requires a materialized launch")
    job = admitted.job
    request = ConversationContextRequest(
        team_id=job.team_id,
        destination_id=job.channel_id,
        destination_kind=_destination_kind(job.conversation_kind),
        actor_id=job.user_id,
        objective=job.prompt,
        current_task_id=admitted.launch.task_id,
        current_event_id=job.event_id,
        current_message_ts=job.message_ts,
        thread_root_ts=job.thread_root_ts,
        allowed_conversation_ids=job.context_conversation_ids,
        access_hash=job.context_access_hash,
        current_thread_namespace_id=job.conversation_key,
        max_memories=0,
    )
    authorized_context = await PostgresConversationContextLoader(sessions).load_authorized(
        admitted.resolution.scope,
        request,
    )
    memory_navigation_authority = MemoryNavigationAuthority(
        scope=admitted.resolution.scope,
        team_id=job.team_id,
        destination_id=job.channel_id,
        destination_kind=normalize_conversation_kind(job.conversation_kind.value),
        actor_id=job.user_id,
        task_id=admitted.launch.task_id,
        run_id=admitted.launch.run_id,
        allowed_conversation_ids=job.context_conversation_ids,
        access_hash=job.context_access_hash,
        membership_hash=authorized_context.manifest.membership_hash,
        current_thread_namespace_id=job.conversation_key,
    )
    combined_thread_ranges = _merge_thread_context_ranges(
        (*authorized_context.reopen_ranges, *thread_context_ranges),
        destination_id=job.channel_id,
    )
    thread_context_authority = ThreadContextAuthority(
        scope=admitted.resolution.scope,
        team_id=job.team_id,
        destination_id=job.channel_id,
        actor_id=job.user_id,
        task_id=admitted.launch.task_id,
        run_id=admitted.launch.run_id,
        thread_root_ts=job.thread_root_ts,
        current_message_ts=job.message_ts,
        allowed_conversation_ids=job.context_conversation_ids,
        access_hash=job.context_access_hash,
        membership_hash=authorized_context.manifest.membership_hash,
    )
    thread_tools = build_thread_context_tools(
        ranges=combined_thread_ranges,
        authority=thread_context_authority,
        clock=SystemClock(),
    )
    progressive_memory = await PostgresProgressiveMemoryService(sessions).search(
        memory_navigation_authority,
        query=job.prompt,
        now=SystemClock().now(),
    )
    memory_context = tuple(
        _memory_context_item(item, memory_navigation_authority) for item in progressive_memory.items
    )
    context_items = _merge_authorized_context(
        (*authorized_context.items, *memory_context, *additional_context),
        allowed_conversation_ids=frozenset(job.context_conversation_ids),
        destination_id=job.channel_id,
        team_id=job.team_id,
        thread_root_ts=job.thread_root_ts,
        actor_id=job.user_id,
    )
    trusted_scope = TrustedScope(
        namespace=admitted.resolution.scope,
        actor_id=job.user_id,
        roles=frozenset({"researcher"}),
    )
    memory_authority = bind_memory_mutation_authority(
        scope=admitted.resolution.scope,
        team_id=job.team_id,
        conversation_id=job.channel_id,
        conversation_kind=normalize_conversation_kind(job.conversation_kind.value),
        actor_id=job.user_id,
        event_id=job.event_id,
        task_id=admitted.launch.task_id,
        run_id=admitted.launch.run_id,
        message_reference=job.message_ts,
        objective=job.prompt,
    )
    result = await run_live_conversation(
        settings=settings,
        client=client,
        objective=job.prompt,
        context_items=context_items,
        context_authority_ids=(
            f"slack-access:{job.context_access_hash}",
            f"slack-membership:{authorized_context.manifest.membership_hash}",
            (f"slack-membership-policy:{authorized_context.manifest.membership_policy_version}"),
            f"slack-provenance:{authorized_context.manifest.external_provenance}",
            *additional_authority_ids,
        ),
        trusted_scope=trusted_scope,
        origin=OriginRef(
            provider="slack",
            external_thread_id=job.conversation_key,
            external_event_id=job.event_id,
            external_channel_id=job.channel_id,
        ),
        sessions=sessions,
        launch_ids=(
            admitted.launch.thread_id,
            admitted.launch.task_id,
            admitted.launch.run_id,
        ),
        lease=lease,
        memory_authority=memory_authority,
        memory_navigation_authority=memory_navigation_authority,
        thread_context_tools=thread_tools,
        provider_gates=provider_gates,
    )
    return result


def _merge_thread_context_ranges(
    ranges: tuple[ThreadContextRange, ...],
    *,
    destination_id: str,
) -> tuple[ThreadContextRange, ...]:
    unique: dict[str, ThreadContextRange] = {}
    for source_range in ranges:
        if any(item.conversation_id != destination_id for item in source_range.items):
            raise ValueError("thread context range escaped the exact Slack destination")
        prior = unique.get(source_range.handle)
        if prior is not None and prior != source_range:
            raise ValueError("thread context handle collision changed its source range")
        unique.setdefault(source_range.handle, source_range)
    return tuple(unique.values())


def _memory_context_item(
    item: ProgressiveMemoryItem,
    authority: MemoryNavigationAuthority,
) -> ContextItem:
    conversation_id = (
        item.source_conversation
        if item.source_conversation in authority.allowed_conversation_ids
        else authority.destination_id
    )
    return ContextItem(
        id=f"progressive-memory:{item.reference}",
        kind=ContextItemKind.MEMORY,
        content=json.dumps(item.model_dump(mode="json"), sort_keys=True),
        conversation_id=conversation_id,
        source_scope=authority.scope,
        source_actor_id=(
            authority.actor_id if item.source_conversation == "actor-private" else None
        ),
    )


async def reconcile_admitted_slack_timeout(
    *,
    sessions: async_sessionmaker[AsyncSession],
    admitted: AdmittedSlackMention,
    lease: TaskLease,
    reason: str = "slack_runtime_deadline_exceeded",
) -> RunBundle:
    """Persist the parent timeout winner before any timeout UX is rendered.

    The process timer is only a signal. This reloads the durable Task/Run, fences
    its transition with the current lease and versions, and becomes a no-op if a
    terminal coordinator commit already won the race.
    """

    if admitted.launch is None:
        raise RuntimeError("Slack timeout reconciliation requires a materialized launch")
    launch = admitted.launch
    clock = SystemClock()
    durable = PostgresRunStore(sessions, clock, UuidIdGenerator())
    store = LeaseBoundRunStore(durable, lease)
    bundle = await _reconcile_timeout_run_winner(
        durable=durable,
        fenced=store,
        task_id=launch.task_id,
        run_id=launch.run_id,
        scope=admitted.resolution.scope,
        clock=clock,
        reason=reason,
    )
    if bundle.run.status in _TERMINAL_RUN_STATUSES:
        await _terminate_parent_plans_after_terminal(
            sessions=sessions,
            admitted=admitted,
            parent_status=bundle.run.status,
            reason=bundle.run.terminal_reason or reason,
        )
    return bundle


async def _reconcile_timeout_run_winner(
    *,
    durable: RunStore,
    fenced: RunStore,
    task_id: str,
    run_id: str,
    scope: ScopeKey,
    clock: Clock,
    reason: str,
) -> RunBundle:
    """CAS a timeout, reloading a cancellation/terminal winner on conflict."""

    last_conflict: ConcurrencyError | None = None
    for _attempt in range(3):
        bundle = await durable.load(task_id, run_id, scope)
        if bundle.run.status in _TERMINAL_RUN_STATUSES:
            return bundle
        try:
            if (bundle.task.status, bundle.run.status) == (
                TaskStatus.QUEUED,
                RunStatus.QUEUED,
            ):
                task, run = start_task_and_run(bundle.task, bundle.run, started_at=clock.now())
                bundle = await fenced.commit(
                    expected_task_version=bundle.task.version,
                    expected_run_version=bundle.run.version,
                    task=task,
                    run=run,
                    events=(
                        EventDraft(
                            type=EventType.TASK_STARTED,
                            iteration=0,
                            payload={"phase": run.phase.value},
                        ),
                    ),
                )
            if (bundle.task.status, bundle.run.status) == (
                TaskStatus.ACTIVE,
                RunStatus.RUNNING,
            ):
                task, run = time_out_task_and_run(
                    bundle.task,
                    bundle.run,
                    reason,
                    usage=bundle.run.usage,
                )
                bundle = await fenced.commit(
                    expected_task_version=bundle.task.version,
                    expected_run_version=bundle.run.version,
                    task=task,
                    run=run,
                    events=(
                        EventDraft(
                            type=EventType.RUN_TIMED_OUT,
                            iteration=run.iteration,
                            payload={"reason": reason},
                        ),
                    ),
                )
            return bundle
        except ConcurrencyError as exc:
            # Cancellation, a terminal coordinator commit, or another timeout CAS
            # may have won after this snapshot. Reload durable truth and retry only
            # while it remains nonterminal.
            last_conflict = exc
    winner = await durable.load(task_id, run_id, scope)
    if winner.run.status in _TERMINAL_RUN_STATUSES:
        return winner
    if last_conflict is not None:
        raise last_conflict
    return winner


async def reconcile_admitted_slack_terminal_winner(
    *,
    sessions: async_sessionmaker[AsyncSession],
    admitted: AdmittedSlackMention,
    store: RunStore | None = None,
) -> RunBundle | None:
    """Reload and propagate an authoritative terminal winner after a stale runtime error."""

    if admitted.launch is None:
        raise RuntimeError("Slack terminal reconciliation requires a materialized launch")
    launch = admitted.launch
    durable = (
        store if store is not None else PostgresRunStore(sessions, SystemClock(), UuidIdGenerator())
    )
    bundle = await durable.load(
        launch.task_id,
        launch.run_id,
        admitted.resolution.scope,
    )
    if bundle.run.status not in _TERMINAL_RUN_STATUSES:
        return None
    await _terminate_parent_plans_after_terminal(
        sessions=sessions,
        admitted=admitted,
        parent_status=bundle.run.status,
        reason=bundle.run.terminal_reason or f"parent_{bundle.run.status.value}",
    )
    return bundle


async def _terminate_parent_plans_after_terminal(
    *,
    sessions: async_sessionmaker[AsyncSession],
    admitted: AdmittedSlackMention,
    parent_status: RunStatus,
    reason: str,
) -> None:
    """Idempotently close child work after a non-success parent is durable.

    Parent Task/Run CAS is deliberately the first transaction.  A crash between
    that commit and this propagation is recovered by terminal reload or the startup
    scanner. Already-terminated plans are excluded from the exact parent scan.
    """

    if parent_status is RunStatus.COMPLETED:
        return
    child_terminal_reason = _child_terminal_reason(parent_status)
    if admitted.launch is None:
        raise RuntimeError("Slack terminal plan propagation requires a materialized launch")
    launch = admitted.launch
    async with sessions() as session:
        plan_ids = tuple(
            (
                await session.scalars(
                    select(PlanRow.id)
                    .where(
                        PlanRow.organization_id == admitted.resolution.scope.organization_id,
                        PlanRow.strategy_id == admitted.resolution.scope.strategy_id,
                        PlanRow.parent_task_id == launch.task_id,
                        PlanRow.parent_run_id == launch.run_id,
                        PlanRow.status == "active",
                    )
                    .order_by(PlanRow.id)
                )
            ).all()
        )
    store = PostgresPlanStore(sessions, SystemClock(), UuidIdGenerator())
    for plan_id in plan_ids:
        await store.terminate_for_parent(
            scope=admitted.resolution.scope,
            plan_id=plan_id,
            parent_task_id=launch.task_id,
            parent_run_id=launch.run_id,
            parent_status=parent_status,
            reason=reason,
            child_terminal_reason=child_terminal_reason,
        )


async def reconcile_terminal_parent_plans(
    sessions: async_sessionmaker[AsyncSession],
    *,
    limit: int = 100,
) -> int:
    """Repair the parent-terminal -> child-termination crash window.

    The parent terminal state is authoritative and already committed.  This
    bounded startup scan only propagates non-success terminal truth to plans
    that are still active; each plan termination is independently idempotent.
    """

    if limit < 1:
        raise ValueError("limit must be positive")
    terminal_statuses = (
        RunStatus.FAILED,
        RunStatus.CANCELLED,
        RunStatus.TIMED_OUT,
        RunStatus.BUDGET_EXHAUSTED,
    )
    async with sessions() as session:
        rows = tuple(
            (
                await session.execute(
                    select(PlanRow, RunRow.status, RunRow.terminal_reason)
                    .join(
                        RunRow,
                        (RunRow.id == PlanRow.parent_run_id)
                        & (RunRow.task_id == PlanRow.parent_task_id)
                        & (RunRow.organization_id == PlanRow.organization_id)
                        & (RunRow.strategy_id == PlanRow.strategy_id),
                    )
                    .where(
                        PlanRow.status == "active",
                        RunRow.status.in_(tuple(item.value for item in terminal_statuses)),
                    )
                    .order_by(PlanRow.updated_at, PlanRow.id)
                    .limit(limit)
                )
            ).all()
        )
    store = PostgresPlanStore(sessions, SystemClock(), UuidIdGenerator())
    for plan, raw_status, terminal_reason in rows:
        parent_status = RunStatus(raw_status)
        reason = terminal_reason or f"parent_{parent_status.value}"
        child_terminal_reason = _child_terminal_reason(parent_status)
        await store.terminate_for_parent(
            scope=ScopeKey(
                organization_id=plan.organization_id,
                strategy_id=plan.strategy_id,
            ),
            plan_id=plan.id,
            parent_task_id=plan.parent_task_id,
            parent_run_id=plan.parent_run_id,
            parent_status=parent_status,
            reason=reason,
            child_terminal_reason=child_terminal_reason,
        )
    return len(rows)


def _child_terminal_reason(parent_status: RunStatus) -> str:
    try:
        return {
            RunStatus.CANCELLED: "parent_cancelled",
            RunStatus.TIMED_OUT: "parent_deadline_exceeded",
            RunStatus.BUDGET_EXHAUSTED: "parent_budget_exhausted",
            RunStatus.FAILED: "parent_failed",
        }[parent_status]
    except KeyError as exc:
        raise ValueError("parent status does not terminate child work") from exc


def _destination_kind(kind: SlackConversationKind) -> str:
    return {
        SlackConversationKind.ORDINARY_INTERNAL: "channel",
        SlackConversationKind.DM: "dm",
        SlackConversationKind.MPIM: "group_dm",
        SlackConversationKind.SHARED: "shared",
        SlackConversationKind.EXTERNAL: "external",
    }[kind]


def _merge_authorized_context(
    items: tuple[ContextItem, ...],
    *,
    allowed_conversation_ids: frozenset[str],
    destination_id: str,
    team_id: str,
    thread_root_ts: str,
    actor_id: str,
    max_tokens: int = 6_000,
) -> tuple[ContextItem, ...]:
    """Reauthorize and globally budget context after combining independent loaders."""

    if destination_id not in allowed_conversation_ids:
        raise ValueError("current destination is absent from the context authorization")
    unique: dict[str, ContextItem] = {}
    for item in items:
        if item.conversation_id not in allowed_conversation_ids:
            raise ValueError("context loader returned an unauthorized conversation")
        unique.setdefault(item.id, item)
    if not unique:
        return ()
    ordered = _reconcile_exact_thread_roots(
        tuple(unique.values()),
        destination_id=destination_id,
        expected_slack_root_id=(f"slack-thread:{team_id}:{destination_id}:{thread_root_ts}"),
        actor_id=actor_id,
    )
    budgeted = assemble_budgeted_context(
        tuple(
            BudgetSegment(
                name=item.id,
                text=item.content,
                priority=(
                    item.budget_priority
                    if item.budget_priority is not None
                    else 90
                    if item.kind is ContextItemKind.MEMORY
                    else 80
                    if item.kind is ContextItemKind.THREAD_SUMMARY
                    else 70
                ),
                pinned=item.retention.pinned,
                source_ids=(item.id,),
            )
            for item in ordered
        ),
        ContextBudget(max_tokens=max_tokens, max_bytes=max_tokens * 4),
    )
    selected = {segment.name for segment in budgeted.segments}
    return tuple(item for item in ordered if item.id in selected)


def _reconcile_exact_thread_roots(
    items: tuple[ContextItem, ...],
    *,
    destination_id: str,
    expected_slack_root_id: str,
    actor_id: str,
) -> tuple[ContextItem, ...]:
    """Collapse only the two equivalent authoritative projections of one Slack root.

    The durable loader emits ``thread-message:*`` with a server-normalized ``User:``
    envelope. The complete Slack loader emits ``slack-thread:*`` with provider metadata
    and may retain the leading app mention. Both sources are already authorized by the
    current ingress envelope. Their trusted retention class, destination, actor, and
    canonical user text must all agree; any other multiplicity or conflict fails closed.
    """

    roots = tuple(item for item in items if item.retention is ContextItemRetention.THREAD_ROOT)
    if len(roots) <= 1:
        return items
    durable = tuple(item for item in roots if item.id.startswith("thread-message:"))
    slack = tuple(item for item in roots if item.id.startswith("slack-thread:"))
    if len(roots) != 2 or len(durable) != 1 or len(slack) != 1:
        raise ValueError("unexpected authoritative thread root multiplicity")
    durable_root = durable[0]
    slack_root = slack[0]
    if (
        durable_root.conversation_id != destination_id
        or slack_root.conversation_id != destination_id
    ):
        raise ValueError("authoritative thread root escaped the exact destination")
    if slack_root.id != expected_slack_root_id:
        raise ValueError("Slack thread root identity mismatch")
    if durable_root.source_actor_id != actor_id or slack_root.source_actor_id != actor_id:
        raise ValueError("authoritative thread root actor mismatch")
    if _durable_root_text(durable_root) != _slack_root_text(slack_root):
        raise ValueError("authoritative thread root content mismatch")
    # Prefer the server-normalized ingress projection. It contains the exact admitted
    # prompt without the connector mention, while the Slack transcript remains covered
    # by the separately durable Slack thread authority proof.
    return tuple(item for item in items if item.id != slack_root.id)


def _durable_root_text(item: ContextItem) -> str:
    content = item.content.strip()
    if not content.startswith("User:"):
        raise ValueError("durable thread root envelope is malformed")
    return " ".join(content.removeprefix("User:").split())


def _slack_root_text(item: ContextItem) -> str:
    content = item.content.strip()
    header, separator, body = content.partition("\n")
    if not separator or not header.startswith("[Slack exact thread;") or not header.endswith("]"):
        raise ValueError("Slack thread root envelope is malformed")
    normalized = body.strip()
    if normalized.startswith("<@"):
        mention, separator, remainder = normalized.partition(">")
        if not separator or len(mention) < 4 or not mention[2:].isalnum():
            raise ValueError("Slack thread root mention is malformed")
        normalized = remainder.strip()
    return " ".join(normalized.split())
