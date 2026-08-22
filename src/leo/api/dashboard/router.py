"""Top-level router for the read-only admin monitoring dashboard."""

from __future__ import annotations

from fastapi import APIRouter

from leo.api.dashboard.routers import (
    conversations,
    failures,
    integrations,
    memory,
    overview,
    plans,
    run_timeline,
    runs,
)

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])
dashboard_router.include_router(overview.router)
dashboard_router.include_router(runs.router)
dashboard_router.include_router(run_timeline.router)
dashboard_router.include_router(plans.router)
dashboard_router.include_router(memory.router)
dashboard_router.include_router(integrations.router)
dashboard_router.include_router(failures.router)
dashboard_router.include_router(conversations.router)
