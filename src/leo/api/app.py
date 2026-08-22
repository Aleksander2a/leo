"""Minimal read-only API surface; Slack Socket Mode runs as a separate entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from leo.api.dashboard.router import dashboard_router
from leo.config import Settings
from leo.health import config_snapshot, deep_health_snapshot
from leo.persistence.database import create_database_engine, create_session_factory

_DASHBOARD_DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def create_app(
    settings: Settings | None = None,
    *,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    health_probe_timeout_seconds: float = 2.0,
) -> FastAPI:
    runtime_settings = settings or Settings()
    if health_probe_timeout_seconds <= 0:
        raise ValueError("health probe timeout must be positive")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Own one dashboard read session factory for the process lifetime.

        Reuses the caller-supplied ``sessions`` factory when present (tests, or a shared
        engine from an outer process) instead of opening a second pool against the same
        database.
        """

        dashboard_sessions = sessions
        engine: AsyncEngine | None = None
        if dashboard_sessions is None and runtime_settings.database_url is not None:
            engine = create_database_engine(runtime_settings.database_url.get_secret_value())
            dashboard_sessions = create_session_factory(engine)
        app.state.dashboard_sessions = dashboard_sessions
        try:
            yield
        finally:
            if engine is not None:
                await engine.dispose()

    application = FastAPI(title="Leo API", version="0.1.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(_DASHBOARD_DEV_ORIGINS),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    application.include_router(dashboard_router)

    @application.get("/health")
    async def health(deep: bool = False) -> dict[str, object]:
        if not deep or runtime_settings.database_url is None:
            return config_snapshot(runtime_settings).model_dump(mode="json")
        if sessions is not None:
            snapshot = await deep_health_snapshot(
                runtime_settings,
                sessions,
                timeout_seconds=health_probe_timeout_seconds,
            )
            return snapshot.model_dump(mode="json")

        engine = create_database_engine(runtime_settings.database_url.get_secret_value())
        transient_sessions = create_session_factory(engine)
        try:
            snapshot = await deep_health_snapshot(
                runtime_settings,
                transient_sessions,
                timeout_seconds=health_probe_timeout_seconds,
            )
            return snapshot.model_dump(mode="json")
        finally:
            await engine.dispose()

    return application


app = create_app()
