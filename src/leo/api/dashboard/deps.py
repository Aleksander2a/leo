"""Shared FastAPI dependencies for the read-only dashboard API."""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import HTTPException, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessions = getattr(request.app.state, "dashboard_sessions", None)
    if sessions is None:
        raise HTTPException(status_code=503, detail="dashboard database is not configured")
    async with sessions() as session:
        yield session


class PageParams:
    """Bounded limit/offset pagination shared by every list endpoint."""

    def __init__(
        self,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset
