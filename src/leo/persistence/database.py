"""Supabase/Postgres connection helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession

logger = logging.getLogger(__name__)


def normalize_database_url(value: str) -> str:
    """Select SQLAlchemy's Psycopg 3 dialect without changing credentials/options."""

    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value.removeprefix("postgres://")
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value.removeprefix("postgresql://")
    if value.startswith("postgresql+psycopg://"):
        return value
    raise ValueError("DATABASE_URL must use a PostgreSQL URL")


def create_database_engine(value: str) -> AsyncEngine:
    return create_async_engine(
        normalize_database_url(value),
        pool_size=5,
        max_overflow=0,
        pool_pre_ping=True,
    )


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def _migrations_root() -> Path | None:
    """Locate the migration scripts, or report that this build does not ship them.

    Layout differs between a source checkout and an installed package. Deriving
    the root from this file's position works from `src/leo/persistence/` and is
    wrong everywhere else: installed into site-packages the same expression
    yields the interpreter's lib directory, and the worker crashed on startup
    with `Path doesn't exist: /usr/local/lib/python3.12/migrations`.

    So look for a directory that actually holds both files, starting with the
    working directory (the container copies them next to the app) and falling
    back to the source-checkout layout. Returning None means the scripts are not
    present -- which is not evidence of a stale schema, only of a build that does
    not carry them.
    """

    here = Path(__file__).resolve()
    candidates = (Path.cwd(), *here.parents)
    for candidate in candidates:
        if (candidate / "alembic.ini").is_file() and (candidate / "migrations").is_dir():
            return candidate
    return None


def build_alembic_head() -> str | None:
    """The Alembic revision this build's migrations declare as head, if shipped."""

    root = _migrations_root()
    if root is None:
        return None
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


class SchemaVersionError(RuntimeError):
    """The database schema is not at the revision this build requires."""


async def require_schema_at_head(sessions: async_sessionmaker[AsyncSession]) -> str | None:
    """Refuse to start a worker against a schema this build cannot use.

    Deployment applies migrations as a separate step from starting the process,
    and nothing ordered the two. A build carrying a new migration could therefore
    come up against the old schema and accept Slack traffic it could not serve:
    every task write fails on the missing column, each run ends in a generic
    terminal error, and the user is told a source was unavailable when the real
    problem is that the deploy is half-applied.

    Failing loudly at startup keeps that window closed. The message names both
    revisions so the fix -- run the migration -- is obvious from the logs.
    """

    expected = build_alembic_head()
    if expected is None:
        # The build does not ship migration scripts, so there is nothing to
        # compare against. Refusing to start here would take the worker down over
        # a packaging detail rather than a real schema problem.
        logger.warning("skipping the startup schema check: this build ships no migration scripts")
        return None

    async with sessions() as session:
        applied = await session.scalar(text("SELECT version_num FROM alembic_version"))
    if applied != expected:
        raise SchemaVersionError(
            "database schema is not at the revision this build requires "
            f"(database={applied!r}, build={expected!r}); run 'alembic upgrade head' "
            "before starting the worker"
        )
    return expected
