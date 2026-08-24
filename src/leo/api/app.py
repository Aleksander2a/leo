"""The read-only HTTP surface: health, plus the observability dashboard API.

Slack runs as its own process (`leo slack`); this one only reads.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from leo.agent.db import create_engine, create_sessions
from leo.api.dashboard import router as dashboard_router
from leo.config import Settings

_DEV_ORIGINS = ("http://localhost:3000", "http://127.0.0.1:3000")


def _cors_origins(settings: Settings) -> list[str]:
    configured = (
        origin.strip()
        for origin in settings.leo_dashboard_cors_origins.split(",")
        if origin.strip()
    )
    return list(dict.fromkeys((*_DEV_ORIGINS, *configured)))


def create_app(
    settings: Settings | None = None,
    *,
    sessions: async_sessionmaker[AsyncSession] | None = None,
) -> FastAPI:
    runtime_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        owned: AsyncEngine | None = None
        factory = sessions
        if factory is None and runtime_settings.database_url is not None:
            owned = create_engine(runtime_settings.database_url.get_secret_value())
            factory = create_sessions(owned)
        application.state.sessions = factory
        try:
            yield
        finally:
            if owned is not None:
                await owned.dispose()

    application = FastAPI(title="Leo API", version="1.0.0", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(runtime_settings),
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    application.include_router(dashboard_router)

    @application.get("/health")
    async def health(deep: bool = False) -> dict[str, object]:
        configured = {
            "model": bool(runtime_settings.leo_model),
            "openrouter": runtime_settings.openrouter_api_key is not None,
            "database": runtime_settings.database_url is not None,
            "slack": runtime_settings.slack_bot_token is not None
            and runtime_settings.slack_app_token is not None,
        }
        payload: dict[str, object] = {
            "status": "ok" if all(configured.values()) else "degraded",
            "environment": runtime_settings.leo_env.value,
            "configured": configured,
        }
        if deep:
            factory = getattr(application.state, "sessions", None)
            if factory is None:
                payload["database_reachable"] = False
            else:
                try:
                    async with factory() as session:
                        await session.execute(text("select 1"))
                    payload["database_reachable"] = True
                except Exception as exc:
                    payload["database_reachable"] = False
                    payload["database_error"] = type(exc).__name__
        return payload

    return application


app = create_app()
