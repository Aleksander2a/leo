"""Minimal read-only API surface; Slack Socket Mode runs as a separate entry point."""

from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.health import config_snapshot, deep_health_snapshot
from leo.persistence.database import create_database_engine, create_session_factory


def create_app(
    settings: Settings | None = None,
    *,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    health_probe_timeout_seconds: float = 2.0,
) -> FastAPI:
    runtime_settings = settings or Settings()
    if health_probe_timeout_seconds <= 0:
        raise ValueError("health probe timeout must be positive")
    application = FastAPI(title="Leo API", version="0.1.0")

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
