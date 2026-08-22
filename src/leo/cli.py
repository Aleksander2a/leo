"""Local operator CLI for smoke runs, replay-oriented output, and config checks."""

from __future__ import annotations

import asyncio
import json
import logging
import selectors
import sys
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from slack_sdk.web.async_client import AsyncWebClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Environment, Settings
from leo.demo import run_quote_smoke
from leo.evals.loader import default_scenario_root, load_scenarios
from leo.evals.report import machine_report, markdown_report
from leo.evals.runner import run_scenarios
from leo.fixtures import FixtureNotFoundError, fixture_ids, run_fixture
from leo.harness.models import (
    BudgetLimits,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
)
from leo.health import config_snapshot, probe_database, probe_operational_metadata
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.slack.context import (
    SlackHistoryContextLoader,
    SlackThreadContextError,
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
from leo.integrations.slack.socket_mode import (
    RUNTIME_DEADLINE_CANCEL_MESSAGE,
    run_socket_mode,
)
from leo.integrations.system import SystemClock, UuidIdGenerator
from leo.live import run_live_quote
from leo.memory.benchmark import load_frozen_retrieval_fixture
from leo.memory.eval_report import validate_committed_m3_report
from leo.memory.maintenance import PurgePlan
from leo.memory.models import MemoryVisibility
from leo.memory.projection import ProjectionRequest
from leo.memory.retrieval import AuthorizedMemoryNamespace
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.derived_memory import PostgresMemoryMaintenance
from leo.persistence.memory_projection import PostgresMemoryProjectionService
from leo.persistence.outbox import PostgresDeliveryOutbox, SlackOutboxDispatcher
from leo.persistence.replay_store import PostgresReplayStore
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.schema import RunRow, TaskRow
from leo.persistence.slack_cancellation import PostgresSlackCancellationService
from leo.persistence.slack_ingress import (
    PostgresSlackIngressAdmission,
    SlackFollowupBusyError,
    SlackLaunchInvariantError,
)
from leo.persistence.task_leases import (
    PostgresTaskLeaseStore,
    TaskLease,
    TaskLeaseConflictError,
)
from leo.replay import (
    MAX_REPLAY_ENTRIES,
    ReplayFormat,
    export_replay,
    render_replay_json,
    render_replay_text,
)
from leo.safe_logging import configure_safe_logging
from leo.worker.runtime import LeaseHeartbeat
from leo.worker.slack_conversation import (
    reconcile_admitted_slack_terminal_winner,
    reconcile_admitted_slack_timeout,
    reconcile_terminal_parent_plans,
    run_admitted_slack_conversation,
)
from leo.worker.terminal import (
    MAX_TASK_ATTEMPTS,
    RETRY_ATTEMPTS_EXHAUSTED,
    persist_safe_failure,
)

app = typer.Typer(no_args_is_help=True, help="Leo custom harness operator commands.")
LOGGER = logging.getLogger(__name__)


def _run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run CLI coroutines on an event loop compatible with async Psycopg on Windows."""

    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        ) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


@app.callback()
def configure_cli_logging() -> None:
    """Configure safe operator logs without serializing settings or secret values."""

    settings = Settings()
    configure_safe_logging(
        level_name=settings.leo_log_level,
        sensitive_values=settings.sensitive_values_for_logging(),
    )


@app.command()
def smoke() -> None:
    """Run the deterministic two-turn quote fixture without credentials."""

    result = _run_async(run_quote_smoke())
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    if result.run.status is not RunStatus.COMPLETED:
        raise typer.Exit(code=1)


@app.command("eval")
def eval_command(
    scenario_id: Annotated[str | None, typer.Option(help="Run one scenario ID.")] = None,
) -> None:
    """Run the versioned offline evaluation fixtures without network/provider calls."""

    root = default_scenario_root()
    selected = None if scenario_id is None else frozenset({scenario_id})
    scenarios = load_scenarios(root, scenario_ids=selected)
    results = run_scenarios(scenarios)
    typer.echo(machine_report(results), nl=False)
    typer.echo(markdown_report(results), nl=False)
    if any(result.status.value != "passed" for result in results):
        raise typer.Exit(code=1)


@app.command("memory-eval")
def memory_eval() -> None:
    """Replay the frozen M3 memory benchmark and absolute safety report."""

    directory = Path(__file__).resolve().parents[2] / "evals/fixtures/memory-retrieval-v1"
    fixture = load_frozen_retrieval_fixture(directory)
    report = _run_async(validate_committed_m3_report(fixture, directory / "m3-report.json"))
    typer.echo(report.model_dump_json(indent=2))


@app.command("run-fixture")
def run_fixture_command(
    fixture_id: Annotated[str, typer.Argument(help="Versioned deterministic fixture ID.")],
    output_format: Annotated[
        ReplayFormat,
        typer.Option("--format", help="Render stable normalized JSON or text."),
    ] = ReplayFormat.JSON,
) -> None:
    """Run one named coordinator fixture and emit a sanitized replay timeline."""

    try:
        result = _run_async(run_fixture(fixture_id))
    except FixtureNotFoundError as exc:
        choices = ", ".join(fixture_ids())
        raise typer.BadParameter(
            f"unknown fixture ID; choose one of: {choices}",
            param_hint="fixture_id",
        ) from exc
    rendered = (
        render_replay_json(result.replay)
        if output_format is ReplayFormat.JSON
        else render_replay_text(result.replay)
    )
    typer.echo(rendered, nl=False)


@app.command("check-config")
def check_config() -> None:
    """Report missing variable names without printing any secret values."""

    settings = Settings()
    user_history_configured = settings.slack_user_token is not None and bool(
        settings.slack_user_token.get_secret_value().strip()
    )
    payload = {
        "deterministic_smoke": {"ready": True, "missing": []},
        "live_slack": {
            "ready": not settings.missing_for_live_slack(),
            "missing": settings.missing_for_live_slack(),
        },
        "thread_history": {
            "ready": user_history_configured,
            "required": False,
            "mode": (
                "direct_user_history"
                if user_history_configured
                else "persisted_coverage_fallback_required"
            ),
            "missing_optional": ([] if user_history_configured else ["SLACK_USER_TOKEN"]),
        },
        "live_harness": {
            "ready": not settings.missing_for_live_harness(),
            "missing": settings.missing_for_live_harness(),
        },
        "live_providers": {
            "ready": not settings.missing_for_live_providers(),
            "missing": settings.missing_for_live_providers(),
        },
    }
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


async def _health_command(deep: bool) -> dict[str, object]:
    settings = Settings()
    if not deep or settings.database_url is None:
        return config_snapshot(settings).model_dump(mode="json")
    engine = create_database_engine(settings.database_url.get_secret_value())
    sessions = create_session_factory(engine)
    try:
        database, queue, outbox, last_success = await probe_database(sessions)
        conversation, membership, orchestration, model = await probe_operational_metadata(sessions)
        return config_snapshot(
            settings,
            database=database,
            conversation_metadata=conversation,
            dm_membership_sync=membership,
            model=model,
            orchestration=orchestration,
            queue=queue,
            outbox=outbox,
            last_success=last_success,
        ).model_dump(mode="json")
    finally:
        await engine.dispose()


@app.command("health")
def health(
    deep: Annotated[bool, typer.Option(help="Run read-only database aggregate probes.")] = False,
) -> None:
    """Report safe process/config/queue/outbox health without provider calls."""

    payload = _run_async(_health_command(deep))
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command("slack-smoke")
def slack_smoke() -> None:
    """Run the blocking local Socket Mode transport smoke."""

    _run_async(run_socket_mode(Settings()))


class _DeterministicSlackHarnessRuntime:
    async def handle(self, admitted: AdmittedSlackMention) -> str | RenderedSlackText:
        job = admitted.job
        result = await run_quote_smoke(
            objective=job.prompt,
            scope=admitted.resolution.scope,
            actor_id=job.user_id,
            origin=OriginRef(
                provider="slack",
                external_thread_id=job.conversation_key,
                external_event_id=job.event_id,
                external_channel_id=job.channel_id,
            ),
        )
        if result.run.status is not RunStatus.COMPLETED or result.run.final_output is None:
            return render_terminal_result(
                SlackTerminalResult(
                    run_id=result.run.id,
                    status=result.run.status,
                    terminal_reason=result.run.terminal_reason,
                )
            )
        return f"{result.run.final_output}\n\nTrace events: {len(result.events)}"


@app.command("slack-harness-smoke")
def slack_harness_smoke() -> None:
    """Run Slack through Leo's deterministic custom loop (no live model/data yet)."""

    settings = Settings()
    _run_async(run_socket_mode(settings, runtime=_DeterministicSlackHarnessRuntime()))


async def _run_live_quote_command(symbol: str) -> None:
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        result = await run_live_quote(
            settings=Settings(),
            client=client,
            symbol=symbol,
            objective=f"Report the current {symbol} quote using an allowed market tool.",
        )
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    if result.run.status is not RunStatus.COMPLETED:
        raise typer.Exit(code=1)


async def _run_durable_quote_command(symbol: str) -> None:
    settings = Settings()
    missing = settings.missing_for_live_harness()
    if missing:
        raise RuntimeError(f"missing live harness configuration names: {', '.join(missing)}")
    assert settings.database_url is not None
    engine = create_database_engine(settings.database_url.get_secret_value())
    sessions = create_session_factory(engine)
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            result = await run_live_quote(
                settings=settings,
                client=client,
                symbol=symbol,
                objective=f"Report the current {symbol} quote using an allowed market tool.",
                sessions=sessions,
            )
    finally:
        await engine.dispose()
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    if result.run.status is not RunStatus.COMPLETED:
        raise typer.Exit(code=1)


@app.command("live-quote")
def live_quote(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to quote.")] = "NVDA",
) -> None:
    """Run the custom loop against real OpenRouter and Finnhub providers."""

    _run_async(_run_live_quote_command(symbol.upper()))


@app.command("durable-quote")
def durable_quote(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol to quote.")] = "NVDA",
) -> None:
    """Run OpenRouter + Finnhub with atomic state/events in configured Postgres."""

    _run_async(_run_durable_quote_command(symbol.upper()))


def _parse_projection_namespaces(
    namespaces: list[str],
) -> frozenset[AuthorizedMemoryNamespace]:
    """Parse explicit ``visibility=namespace`` pairs without wildcard expansion."""

    parsed: set[AuthorizedMemoryNamespace] = set()
    for raw in namespaces:
        visibility_value, separator, namespace_id = raw.partition("=")
        if not separator or not visibility_value.strip() or not namespace_id.strip():
            raise ValueError("each namespace must be an explicit visibility=namespace pair")
        if "*" in namespace_id or "?" in namespace_id:
            raise ValueError("projection namespaces cannot contain wildcards")
        try:
            visibility = MemoryVisibility(visibility_value.strip())
        except ValueError as exc:
            choices = ", ".join(
                item.value
                for item in MemoryVisibility
                if item
                not in {
                    MemoryVisibility.STRATEGY_SHARED,
                    MemoryVisibility.ORGANIZATION_SHARED,
                }
            )
            raise ValueError(f"unknown visibility; choose one of: {choices}") from exc
        parsed.add(
            AuthorizedMemoryNamespace(
                visibility=visibility,
                namespace_id=namespace_id.strip(),
            )
        )
    if not parsed:
        raise ValueError("at least one explicit namespace is required")
    return frozenset(parsed)


async def _memory_project_command(
    *,
    namespaces: list[str],
    after: str | None,
    page_size: int,
) -> dict[str, object]:
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("missing memory projection configuration name: DATABASE_URL")
    now = datetime.now(UTC)
    request = ProjectionRequest(
        scope=ScopeKey(
            organization_id=settings.leo_organization_id,
            strategy_id=settings.leo_strategy_id,
        ),
        authorized_namespaces=_parse_projection_namespaces(namespaces),
        generated_at=now.isoformat(),
        policy_version="projection-v1",
        page_size=page_size,
        after=after,
    )
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        page = await PostgresMemoryProjectionService(create_session_factory(engine)).render_page(
            request, as_of=now
        )
    finally:
        await engine.dispose()
    return page.model_dump(mode="json")


@app.command("memory-project")
def memory_project(
    namespace: Annotated[
        list[str],
        typer.Option(
            "--namespace",
            help=(
                "Repeat an exact visibility=namespace pair, for example "
                "conversation_local=C0123. Wildcards are rejected."
            ),
        ),
    ],
    after: Annotated[
        str | None,
        typer.Option(help="Opaque next cursor from the preceding projection page."),
    ] = None,
    page_size: Annotated[
        int,
        typer.Option(min=1, max=100, help="Number of authorized current records per page."),
    ] = 25,
) -> None:
    """Render one escaped, read-only page from explicitly authorized memory namespaces."""

    try:
        payload = _run_async(
            _memory_project_command(
                namespaces=namespace,
                after=after,
                page_size=page_size,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--namespace") from exc
    typer.echo(str(payload["markdown"]), nl=False)
    typer.echo(
        json.dumps(
            {
                "digest": payload["digest"],
                "item_count": payload["item_count"],
                "next_cursor": payload["next_cursor"],
                "source_revisions": payload["source_revisions"],
            },
            indent=2,
            sort_keys=True,
        )
    )


async def _memory_purge_command(
    *,
    record_ids: tuple[str, ...],
    confirmation_token: str | None,
) -> PurgePlan | dict[str, object]:
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("missing memory purge configuration name: DATABASE_URL")
    scope = ScopeKey(
        organization_id=settings.leo_organization_id,
        strategy_id=settings.leo_strategy_id,
    )
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        maintenance = PostgresMemoryMaintenance(create_session_factory(engine))
        # Always prepare against current, locked-at-execute generation/revision state. A
        # supplied token therefore becomes stale if any target changed after dry-run.
        plan = await maintenance.prepare_purge(scope, record_ids)
        if confirmation_token is None:
            return plan
        if confirmation_token != plan.confirmation_token:
            raise ValueError("purge confirmation is stale; run a new dry-run")
        result = await maintenance.execute_purge(
            plan,
            scope=scope,
            confirmation_token=confirmation_token,
        )
        return result.model_dump(mode="json")
    finally:
        await engine.dispose()


@app.command("memory-purge")
def memory_purge(
    record_ids: Annotated[
        list[str],
        typer.Argument(help="One to 100 exact, already-retracted memory record IDs."),
    ],
    confirm: Annotated[
        str | None,
        typer.Option(
            "--confirm",
            help="Execute using the exact token from a fresh dry-run manifest.",
        ),
    ] = None,
) -> None:
    """Dry-run or explicitly confirm bounded physical deletion of retracted demo memory."""

    try:
        payload = _run_async(
            _memory_purge_command(
                record_ids=tuple(record_ids),
                confirmation_token=confirm,
            )
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="record_ids/--confirm") from exc
    rendered = payload.model_dump(mode="json") if isinstance(payload, PurgePlan) else payload
    typer.echo(json.dumps(rendered, indent=2, sort_keys=True))


async def _replay_command(
    run_id: str,
    *,
    output_format: ReplayFormat,
    output: Path | None,
    max_entries: int,
) -> None:
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("missing replay configuration name: DATABASE_URL")
    scope = ScopeKey(
        organization_id=settings.leo_organization_id,
        strategy_id=settings.leo_strategy_id,
    )
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        store = PostgresReplayStore(
            create_session_factory(engine),
            SystemClock(),
            UuidIdGenerator(),
        )
        replay = await store.load(scope=scope, run_id=run_id, max_entries=max_entries)
    finally:
        await engine.dispose()
    if output is not None:
        destination = export_replay(replay, output, output_format=output_format)
        typer.echo(
            json.dumps(
                {
                    "digest": replay.digest,
                    "output": str(destination),
                    "run_id": replay.run_id,
                    "schema_version": replay.schema_version,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    rendered = (
        render_replay_json(replay)
        if output_format is ReplayFormat.JSON
        else render_replay_text(replay)
    )
    typer.echo(rendered, nl=False)


@app.command("replay")
def replay(
    run_id: Annotated[str, typer.Argument(help="Exact durable parent run ID.")],
    output_format: Annotated[
        ReplayFormat,
        typer.Option("--format", help="Render stable normalized JSON or text."),
    ] = ReplayFormat.JSON,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Atomically export the sanitized replay."),
    ] = None,
    max_entries: Annotated[
        int,
        typer.Option(min=1, max=MAX_REPLAY_ENTRIES, help="Maximum replay entries."),
    ] = MAX_REPLAY_ENTRIES,
) -> None:
    """Replay an exact-scope parent with durable plan, children, and source manifest."""

    _run_async(
        _replay_command(
            run_id,
            output_format=output_format,
            output=output,
            max_entries=max_entries,
        )
    )


class _LiveSlackHarnessRuntime:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient,
        sessions: async_sessionmaker[AsyncSession],
        ingress: PostgresSlackIngressAdmission,
        leases: PostgresTaskLeaseStore,
    ) -> None:
        self._settings = settings
        self._client = client
        self._sessions = sessions
        self._ingress = ingress
        self._leases = leases
        self._ids = UuidIdGenerator()
        self._clock = SystemClock()
        self._provider_gates = ProviderGateRegistry(self._clock)
        self._run_store = PostgresRunStore(sessions, self._clock, self._ids)
        self._owner = f"slack-runtime-{self._ids.new('worker')}"
        assert settings.slack_bot_token is not None
        self._history = SlackHistoryContextLoader(
            AsyncWebClient(token=settings.slack_bot_token.get_secret_value()),
            user_history_client=(
                AsyncWebClient(token=settings.slack_user_token.get_secret_value())
                if settings.slack_user_token is not None
                else None
            ),
            thread_fallback=ingress,
        )

    async def handle(self, admitted: AdmittedSlackMention) -> str | RenderedSlackText:
        if admitted.launch is None:
            raise RuntimeError("Slack runtime received an unmaterialized launch")
        admitted = await self._ingress.load_linked_mention(admitted.launch.task_id)
        if admitted.launch is None:
            raise RuntimeError("Slack admission reload returned no launch")
        launch = admitted.launch
        job = admitted.job
        lease_seconds = max(60.0, min(300.0, self._settings.leo_max_run_seconds))
        lease = await self._leases.claim_task(
            launch.task_id,
            self._owner,
            lease_seconds=lease_seconds,
            max_attempts=MAX_TASK_ATTEMPTS,
        )
        if lease is None:
            lease = await self._leases.claim_exhausted_task(
                launch.task_id,
                self._owner,
                lease_seconds=lease_seconds,
                max_attempts=MAX_TASK_ATTEMPTS,
            )
            if lease is None:
                winner = await reconcile_admitted_slack_terminal_winner(
                    sessions=self._sessions,
                    admitted=admitted,
                    store=self._run_store,
                )
                if winner is not None:
                    await self._mark_terminal_winner(admitted, winner)
                    return _render_durable_terminal_bundle(winner)
                return "Leo is already working on this request."
            return await self._finalize_exhausted(admitted, lease)
        try:
            history = await self._history.load(job)
            async with LeaseHeartbeat(
                self._leases,
                lease,
                lease_seconds,
            ):
                result = await run_admitted_slack_conversation(
                    settings=self._settings,
                    client=self._client,
                    sessions=self._sessions,
                    admitted=admitted,
                    lease=lease,
                    additional_context=history.items,
                    thread_context_ranges=history.reopen_ranges,
                    additional_authority_ids=slack_history_authority_ids(history.manifest),
                    provider_gates=self._provider_gates,
                )
        except asyncio.CancelledError as exc:
            if not exc.args or exc.args[0] != RUNTIME_DEADLINE_CANCEL_MESSAGE:
                raise
            # The processor timer is only a cancellation signal. Persist/reload the
            # fenced Task/Run winner before returning any timeout-shaped Slack text.
            bundle = await asyncio.shield(
                reconcile_admitted_slack_timeout(
                    sessions=self._sessions,
                    admitted=admitted,
                    lease=lease,
                )
            )
            await self._ingress.mark_linked_status(
                job.event_id,
                f"run_{bundle.run.status.value}",
                None,
            )
            return _render_durable_terminal_bundle(bundle)
        except Exception as exc:
            failure_reason = "runtime_error"
            if isinstance(exc, SlackThreadContextError):
                safe_code = str(exc).strip()
                if (
                    not safe_code
                    or len(safe_code) > 96
                    or safe_code != safe_code.casefold()
                    or not safe_code.replace("_", "").isalnum()
                ):
                    safe_code = "slack_thread_context_error"
                failure_reason = f"context_unavailable:{safe_code}"
            # Keep the durable/operator trail useful without logging prompt content,
            # credentials, provider payloads, or arbitrary exception messages.
            LOGGER.error(
                "Slack run failed before completion: exception_type=%s safe_reason=%s",
                type(exc).__name__,
                failure_reason,
                extra={"event_id": job.event_id},
            )
            winner = await asyncio.shield(
                reconcile_admitted_slack_terminal_winner(
                    sessions=self._sessions,
                    admitted=admitted,
                    store=self._run_store,
                )
            )
            if winner is not None:
                await self._mark_terminal_winner(admitted, winner)
                return _render_durable_terminal_bundle(winner)
            try:
                await asyncio.shield(
                    persist_safe_failure(
                        LeaseBoundRunStore(self._run_store, lease),
                        task_id=launch.task_id,
                        run_id=launch.run_id,
                        scope=admitted.resolution.scope,
                        reason=failure_reason,
                        clock=self._clock,
                    )
                )
            except Exception:
                # A cancellation/timeout may win the failure CAS. Reload it before
                # deciding whether this launch remains eligible for retry recovery.
                winner = await asyncio.shield(
                    reconcile_admitted_slack_terminal_winner(
                        sessions=self._sessions,
                        admitted=admitted,
                        store=self._run_store,
                    )
                )
                if winner is None:
                    if lease.attempt >= MAX_TASK_ATTEMPTS:
                        return await asyncio.shield(self._finalize_exhausted(admitted, lease))
                    try:
                        await self._leases.abandon(lease, safe_error=failure_reason)
                    except TaskLeaseConflictError:
                        # A terminal coordinator commit already cleared the lease.
                        pass
                    await self._ingress.mark_linked_status(
                        job.event_id, "runtime_failed", failure_reason
                    )
                    raise
            else:
                winner = await asyncio.shield(
                    reconcile_admitted_slack_terminal_winner(
                        sessions=self._sessions,
                        admitted=admitted,
                        store=self._run_store,
                    )
                )
                if winner is None:
                    raise RuntimeError("runtime failure did not persist a terminal run")
            await self._mark_terminal_winner(admitted, winner)
            return _render_durable_terminal_bundle(winner)
        if result.run.status is not RunStatus.COMPLETED or result.run.final_output is None:
            winner = await reconcile_admitted_slack_terminal_winner(
                sessions=self._sessions,
                admitted=admitted,
                store=self._run_store,
            )
            if winner is not None:
                await self._mark_terminal_winner(admitted, winner)
                return _render_durable_terminal_bundle(winner)
            await self._ingress.mark_linked_status(
                job.event_id,
                f"run_{result.run.status.value}",
                None,
            )
            return render_terminal_result(
                SlackTerminalResult(
                    run_id=result.run.id,
                    status=result.run.status,
                    terminal_reason=result.run.terminal_reason,
                    completed_output=result.run.final_output,
                )
            )
        await self._ingress.mark_linked_status(
            job.event_id,
            f"run_{result.run.status.value}",
            None,
        )
        return render_verified_result(
            verified_result_from_coordinator(result, include_evidence_details=False)
        )

    async def _finalize_exhausted(
        self,
        admitted: AdmittedSlackMention,
        lease: TaskLease,
    ) -> RenderedSlackText:
        if admitted.launch is None:
            raise RuntimeError("exhausted Slack admission has no launch")
        launch = admitted.launch
        try:
            await persist_safe_failure(
                LeaseBoundRunStore(self._run_store, lease),
                task_id=launch.task_id,
                run_id=launch.run_id,
                scope=admitted.resolution.scope,
                reason=RETRY_ATTEMPTS_EXHAUSTED,
                clock=self._clock,
            )
        except Exception:
            # A cancellation/timeout may win while the exhaustion failure is being
            # committed. Durable terminal truth still owns the Slack response.
            winner = await reconcile_admitted_slack_terminal_winner(
                sessions=self._sessions,
                admitted=admitted,
                store=self._run_store,
            )
            if winner is None:
                raise
        else:
            winner = await reconcile_admitted_slack_terminal_winner(
                sessions=self._sessions,
                admitted=admitted,
                store=self._run_store,
            )
            if winner is None:
                raise RuntimeError("retry exhaustion did not persist a terminal run")
        await self._mark_terminal_winner(admitted, winner)
        return _render_durable_terminal_bundle(winner)

    async def _mark_terminal_winner(
        self,
        admitted: AdmittedSlackMention,
        winner: RunBundle,
    ) -> None:
        await self._ingress.mark_linked_status(
            admitted.job.event_id,
            f"run_{winner.run.status.value}",
            None,
        )


class _LiveSlackLaunchPreparer:
    def __init__(self, settings: Settings, ingress: PostgresSlackIngressAdmission) -> None:
        self._settings = settings
        self._ingress = ingress
        self._ids = UuidIdGenerator()

    def _seed(self, slack_job: SlackMentionJob, scope: ScopeKey) -> tuple[Thread, Task, Run]:
        event_id = slack_job.event_id
        thread = Thread(
            id=self._ids.new("thread"),
            scope=scope,
            origin=OriginRef(
                provider="slack",
                external_thread_id=slack_job.conversation_key,
                external_event_id=event_id,
                external_channel_id=slack_job.channel_id,
            ),
        )
        task = Task(
            id=self._ids.new("task"),
            thread_id=thread.id,
            scope=scope,
            objective=slack_job.prompt,
        )
        run = Run(
            id=self._ids.new("run"),
            task_id=task.id,
            scope=scope,
            limits=BudgetLimits(
                max_iterations=self._settings.leo_max_model_turns,
                max_model_calls=self._settings.leo_max_model_turns,
                max_tool_calls=self._settings.leo_max_tool_calls,
                max_elapsed_seconds=self._settings.leo_max_run_seconds,
            ),
        )
        return thread, task, run

    async def prepare(self, admitted: AdmittedSlackMention) -> AdmittedSlackMention:
        job = admitted.job
        thread, task, run = self._seed(job, admitted.resolution.scope)
        try:
            materialized = await self._ingress.materialize_initial_launch(
                event_id=job.event_id,
                thread=thread,
                task=task,
                run=run,
            )
        except SlackLaunchInvariantError as exc:
            if str(exc) != "thread has an active Task":
                raise
            await self._ingress.mark_followup_pending(job.event_id, "thread_task_active")
            raise SlackFollowupBusyError from exc
        except IntegrityError as exc:
            # The partial unique index is the final authority when two follow-ups race.
            # Convert only threaded conflicts to the same safe policy outcome; root launches
            # must preserve their original database error for operator diagnosis.
            if job.thread_root_ts == job.message_ts:
                raise
            await self._ingress.mark_followup_pending(
                job.event_id,
                "thread_task_active_race",
            )
            raise SlackFollowupBusyError from exc
        # Materialization may bind a follow-up to an existing conversation thread whose
        # optional strategy metadata differs from today's configured default. Reload the
        # committed admission so runtime authority always matches the durable Task/Run.
        return await self._ingress.load_linked_mention(materialized.task_id)

    async def recover(self) -> tuple[AdmittedSlackMention, ...]:
        return await self._ingress.recover_startup_launches(
            self._seed,
            include_queued=True,
        )


async def _run_slack_live_command() -> None:
    settings = Settings()
    if settings.leo_env is not Environment.DEVELOPMENT:
        raise RuntimeError("automatic Slack channel onboarding is limited to the development demo")
    missing = tuple(
        dict.fromkeys(settings.missing_for_live_slack() + settings.missing_for_live_harness())
    )
    if missing:
        raise RuntimeError(f"missing live Slack configuration names: {', '.join(missing)}")
    assert settings.database_url is not None
    engine = create_database_engine(settings.database_url.get_secret_value())
    sessions = create_session_factory(engine)
    timeout = httpx.Timeout(connect=10.0, read=60.0, write=10.0, pool=10.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            ingress = PostgresSlackIngressAdmission(sessions)
            leases = PostgresTaskLeaseStore(sessions, UuidIdGenerator())
            runtime = _LiveSlackHarnessRuntime(settings, client, sessions, ingress, leases)
            launch_preparer = _LiveSlackLaunchPreparer(settings, ingress)
            cancellation = PostgresSlackCancellationService(
                sessions,
                SystemClock(),
                UuidIdGenerator(),
                ingress,
            )
            outbox = PostgresDeliveryOutbox(sessions, UuidIdGenerator())
            dispatcher = SlackOutboxDispatcher(
                outbox,
                owner=f"slack-dispatcher-{UuidIdGenerator().new('worker')}",
            )
            await reconcile_terminal_parent_plans(sessions)
            await outbox.reconcile_terminal(
                _terminal_delivery_payload,
                payload_version=RENDERER_VERSION * 1000,
            )
            await run_socket_mode(
                settings,
                runtime=runtime,
                admission=ingress,
                launch_preparer=launch_preparer,
                outbox=outbox,
                dispatcher=dispatcher,
                cancellation_handler=cancellation,
            )
    finally:
        await engine.dispose()


@app.command("slack-live")
def slack_live() -> None:
    """Run the real Slack -> Postgres -> OpenRouter/Finnhub local slice."""

    _run_async(_run_slack_live_command())


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


def _render_durable_terminal_bundle(bundle: RunBundle) -> RenderedSlackText:
    return render_terminal_result(
        SlackTerminalResult(
            run_id=bundle.run.id,
            status=bundle.run.status,
            terminal_reason=bundle.run.terminal_reason,
            completed_output=bundle.run.final_output,
        )
    )


if __name__ == "__main__":
    app()
