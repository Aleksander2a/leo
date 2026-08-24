"""Supabase/Postgres connection helpers."""

from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession


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


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def build_alembic_head() -> str | None:
    """The Alembic revision this build's migrations declare as head."""

    config = Config(str(_REPOSITORY_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_REPOSITORY_ROOT / "migrations"))
    return ScriptDirectory.from_config(config).get_current_head()


class SchemaVersionError(RuntimeError):
    """The database schema is not at the revision this build requires."""


async def require_schema_at_head(sessions: async_sessionmaker[AsyncSession]) -> str:
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
        raise SchemaVersionError("this build declares no Alembic head")

    async with sessions() as session:
        applied = await session.scalar(text("SELECT version_num FROM alembic_version"))
    if applied != expected:
        raise SchemaVersionError(
            "database schema is not at the revision this build requires "
            f"(database={applied!r}, build={expected!r}); run 'alembic upgrade head' "
            "before starting the worker"
        )
    return expected
