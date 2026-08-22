from __future__ import annotations

import asyncio
import os
import selectors
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest
import pytest_asyncio
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.harness.models import ScopeKey
from leo.harness.ports import RunStore
from leo.harness.storage import InMemoryRunStore
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.persistence.database import (
    create_database_engine,
    normalize_database_url,
)
from leo.persistence.run_store import PostgresRunStore

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def pytest_asyncio_loop_factories(
    config: pytest.Config,
    item: pytest.Item,
) -> dict[str, Callable[[], asyncio.AbstractEventLoop]]:
    if sys.platform == "win32":
        return {"selector": lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())}
    return {"default": asyncio.new_event_loop}


@dataclass(frozen=True)
class StoreHarness:
    store: RunStore
    clock: FixedClock


@dataclass(frozen=True)
class TwoConnectionPostgresHarness:
    """Two committed database sessions isolated by one UUID namespace."""

    sessions_a: async_sessionmaker[AsyncSession]
    sessions_b: async_sessionmaker[AsyncSession]
    scope: ScopeKey
    suffix: str
    team_id: str
    channel_id: str
    user_id: str
    backend_pids: tuple[int, int]


_TWO_CONNECTION_LOCK_NAME = "leo-m2-two-connection-contract-v1"
_TWO_CONNECTION_ACK_ENV = "LEO_SHARED_DEMO_RACE_ACK"
_TWO_CONNECTION_CLEANUP_SQL = (
    "DELETE FROM delivery_outbox WHERE organization_id = :organization_id",
    "DELETE FROM slack_thread_coverage WHERE team_id = :team_id",
    "DELETE FROM memory_capability_handles WHERE organization_id = :organization_id",
    "DELETE FROM memory_retrieval_cache WHERE organization_id = :organization_id",
    "DELETE FROM memory_embedding_jobs WHERE organization_id = :organization_id",
    "DELETE FROM plans WHERE organization_id = :organization_id",
    "DELETE FROM run_events WHERE task_id IN "
    "(SELECT id FROM tasks WHERE organization_id = :organization_id)",
    "DELETE FROM claims WHERE organization_id = :organization_id",
    "DELETE FROM observations WHERE organization_id = :organization_id",
    "DELETE FROM conversation_access_snapshots WHERE organization_id = :organization_id",
    "DELETE FROM sanitized_messages WHERE organization_id = :organization_id",
    "DELETE FROM conversation_threads WHERE organization_id = :organization_id",
    "DELETE FROM conversation_scope_selections WHERE organization_id = :organization_id",
    "DELETE FROM slack_ingress_events WHERE organization_id = :organization_id",
    "DELETE FROM runs WHERE organization_id = :organization_id",
    "DELETE FROM tasks WHERE organization_id = :organization_id",
    "DELETE FROM thread_summary_revisions WHERE organization_id = :organization_id",
    "DELETE FROM threads WHERE organization_id = :organization_id",
    "DELETE FROM conversation_actor_memberships WHERE organization_id = :organization_id",
    "DELETE FROM slack_channel_scopes WHERE team_id = :team_id",
    "DELETE FROM conversations WHERE team_id = :team_id",
)
_TWO_CONNECTION_REMAINING_SQL = (
    "SELECT "
    "(SELECT count(*) FROM delivery_outbox WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM slack_thread_coverage WHERE team_id = :team_id) + "
    "(SELECT count(*) FROM memory_capability_handles "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM memory_retrieval_cache "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM memory_embedding_jobs "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM plans WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM claims WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM observations WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM conversation_access_snapshots "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM sanitized_messages WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM conversation_threads WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM conversation_scope_selections "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM slack_ingress_events WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM runs WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM tasks WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM thread_summary_revisions "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM threads WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM conversation_actor_memberships "
    " WHERE organization_id = :organization_id) + "
    "(SELECT count(*) FROM slack_channel_scopes WHERE team_id = :team_id) + "
    "(SELECT count(*) FROM conversations WHERE team_id = :team_id)"
)


def _demo_database_url(monkeypatch: pytest.MonkeyPatch) -> str:
    configured = Settings().database_url
    if configured is None:
        pytest.skip("DATABASE_URL is not configured")

    candidate = configured.get_secret_value().strip()

    try:
        normalized_candidate = normalize_database_url(candidate)
    except ValueError as exc:
        pytest.fail(f"DATABASE_URL is invalid: {exc}")

    host = (make_url(normalized_candidate).host or "").lower()
    if "supabase" not in host:
        pytest.fail("DATABASE_URL must target the configured Supabase demo project")

    # Existing Postgres contract tests read DATABASE_URL through Settings/Alembic.
    # Installing the exact configured value is read/write authority for synthetic
    # rows only; it is never authority to downgrade/drop the shared schema.
    monkeypatch.setenv("DATABASE_URL", candidate)
    return candidate


async def _require_current_head(connection: AsyncConnection) -> None:
    config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPOSITORY_ROOT / "migrations"))
    expected = ScriptDirectory.from_config(config).get_current_head()
    if expected is None:
        raise RuntimeError("Alembic has no current head")
    actual = await connection.scalar(text("select version_num from alembic_version"))
    if actual != expected:
        raise RuntimeError(
            "Postgres contract tests require the current Alembic head; "
            "schema mutation is never performed by the shared-database fixture"
        )


@asynccontextmanager
async def _postgres_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[StoreHarness]:
    database_url = _demo_database_url(monkeypatch)
    engine = create_database_engine(database_url)
    async with engine.connect() as connection:
        outer_transaction = await connection.begin()
        await _require_current_head(connection)
        sessions = async_sessionmaker(
            bind=connection,
            class_=AsyncSession,
            autoflush=True,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        clock = FixedClock()
        harness = StoreHarness(
            store=PostgresRunStore(sessions, clock, SequentialIdGenerator()),
            clock=clock,
        )
        try:
            yield harness
        finally:
            if outer_transaction.is_active:
                await outer_transaction.rollback()
    await engine.dispose()


@pytest_asyncio.fixture
async def postgres_store(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[StoreHarness]:
    async with _postgres_harness(monkeypatch) as harness:
        yield harness


@pytest_asyncio.fixture
async def preserved_postgres_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    """Use the current demo schema while rolling every test mutation back.

    Sessions commit independent savepoints so repositories exercise their normal
    transaction boundaries. The outer connection transaction is never committed,
    which preserves unrelated demo rows and avoids a schema downgrade/reset.
    Tests using this fixture must not perform concurrent work on separate sessions;
    true multi-connection race tests use unique IDs plus targeted cleanup instead.
    """

    database_url = _demo_database_url(monkeypatch)
    engine = create_database_engine(database_url)
    async with engine.connect() as connection:
        outer_transaction = await connection.begin()
        await _require_current_head(connection)
        sessions = async_sessionmaker(
            bind=connection,
            class_=AsyncSession,
            autoflush=True,
            expire_on_commit=False,
            join_transaction_mode="create_savepoint",
        )
        try:
            yield sessions
        finally:
            if outer_transaction.is_active:
                await outer_transaction.rollback()
    await engine.dispose()


@pytest_asyncio.fixture
async def two_connection_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[TwoConnectionPostgresHarness]:
    """Expose two real committed connections with exact, failure-safe cleanup.

    A fixed transaction-level advisory lock serializes only this narrow contract-test
    cohort. Every row uses a UUID organization/team namespace, and teardown deletes
    only that namespace in foreign-key-safe order. This fixture never changes schema
    state and refuses to run unless the database is already at the current head.
    """

    if os.getenv(_TWO_CONNECTION_ACK_ENV) != "listener-stopped":
        pytest.fail(
            "committed two-connection contracts require "
            "LEO_SHARED_DEMO_RACE_ACK=listener-stopped; global listener/worker "
            "scanners must be stopped before synthetic rows are committed"
        )
    database_url = _demo_database_url(monkeypatch)
    engine = create_database_engine(database_url)
    suffix = uuid4().hex[:12]
    scope = ScopeKey(
        organization_id=f"org-m2-race-{suffix}",
        strategy_id=f"strategy-m2-race-{suffix}",
    )
    team_id = f"T{suffix.upper()}"
    channel_id = f"C{suffix[::-1].upper()}"
    user_id = f"U{suffix.upper()}"
    guard = await engine.connect()
    connection_a: AsyncConnection | None = None
    connection_b: AsyncConnection | None = None
    lock_acquired = False
    try:
        await _require_current_head(guard)
        await guard.rollback()
        await guard.begin()
        await guard.execute(text("SET LOCAL lock_timeout = '15s'"))
        await guard.execute(
            text("SELECT pg_advisory_xact_lock(hashtext('leo'), hashtext(:lock_name))"),
            {"lock_name": _TWO_CONNECTION_LOCK_NAME},
        )
        lock_acquired = True

        connection_a = await engine.connect()
        connection_b = await engine.connect()
        pid_a = int(await connection_a.scalar(text("SELECT pg_backend_pid()")))
        pid_b = int(await connection_b.scalar(text("SELECT pg_backend_pid()")))
        await connection_a.commit()
        await connection_b.commit()
        if pid_a == pid_b:
            raise RuntimeError("two-connection fixture did not obtain distinct backends")
        sessions_a = async_sessionmaker(
            bind=connection_a,
            class_=AsyncSession,
            autoflush=True,
            expire_on_commit=False,
        )
        sessions_b = async_sessionmaker(
            bind=connection_b,
            class_=AsyncSession,
            autoflush=True,
            expire_on_commit=False,
        )
        yield TwoConnectionPostgresHarness(
            sessions_a=sessions_a,
            sessions_b=sessions_b,
            scope=scope,
            suffix=suffix,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
            backend_pids=(pid_a, pid_b),
        )
    finally:
        try:
            actor_close_error: BaseException | None = None
            for connection in (connection_a, connection_b):
                if connection is None:
                    continue
                try:
                    if connection.in_transaction():
                        await connection.rollback()
                except BaseException as exc:
                    if actor_close_error is None:
                        actor_close_error = exc
                finally:
                    try:
                        await connection.close()
                    except BaseException as exc:
                        if actor_close_error is None:
                            actor_close_error = exc
            if guard.in_transaction():
                if not lock_acquired:
                    await guard.rollback()
            if lock_acquired:
                try:
                    cleanup_parameters = {
                        "organization_id": scope.organization_id,
                        "team_id": team_id,
                    }
                    for statement in _TWO_CONNECTION_CLEANUP_SQL:
                        await guard.execute(text(statement), cleanup_parameters)
                    remaining = int(
                        await guard.scalar(
                            text(_TWO_CONNECTION_REMAINING_SQL),
                            cleanup_parameters,
                        )
                    )
                    if remaining != 0:
                        raise RuntimeError("two-connection fixture cleanup left scoped rows")
                    await guard.commit()
                finally:
                    if guard.in_transaction():
                        await guard.rollback()
            if actor_close_error is not None:
                raise actor_close_error
        finally:
            await guard.close()
            await engine.dispose()


@pytest_asyncio.fixture(params=("memory", "postgres"))
async def store_harness(
    request: pytest.FixtureRequest,
) -> AsyncIterator[StoreHarness]:
    if request.param == "postgres":
        monkeypatch = request.getfixturevalue("monkeypatch")
        async with _postgres_harness(monkeypatch) as harness:
            yield harness
        return

    clock = FixedClock()
    yield StoreHarness(
        store=InMemoryRunStore(clock, SequentialIdGenerator()),
        clock=clock,
    )
