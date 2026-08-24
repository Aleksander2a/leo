"""Database connection helpers."""

from __future__ import annotations

import asyncio
import selectors
import sys
from collections.abc import Coroutine
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def normalize_database_url(value: str) -> str:
    """Select SQLAlchemy's psycopg 3 dialect without touching credentials or options."""

    for prefix in ("postgres://", "postgresql://"):
        if value.startswith(prefix):
            return "postgresql+psycopg://" + value.removeprefix(prefix)
    if value.startswith("postgresql+psycopg://"):
        return value
    raise ValueError("DATABASE_URL must be a PostgreSQL URL")


def create_engine(value: str) -> AsyncEngine:
    return create_async_engine(
        normalize_database_url(value),
        pool_size=5,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
    )


def create_sessions(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


def run[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine on a loop psycopg can actually use.

    Windows defaults to the Proactor loop, which psycopg's async mode refuses.
    Every process entry point goes through here so that is never a runtime
    surprise. Elsewhere the platform default (epoll/kqueue) is already correct
    and strictly better, so it is left alone.
    """

    if sys.platform == "win32":
        return asyncio.run(
            coroutine,
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector()),
        )
    return asyncio.run(coroutine)
