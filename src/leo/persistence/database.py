"""Supabase/Postgres connection helpers."""

from __future__ import annotations

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
