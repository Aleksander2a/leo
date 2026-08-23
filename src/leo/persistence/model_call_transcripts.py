"""Postgres-backed model_call_transcripts writer (dashboard inspection only)."""

from __future__ import annotations

from datetime import datetime

from pydantic import JsonValue
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.harness.ports import IdGenerator
from leo.persistence.schema import ModelCallTranscriptRow


class PostgresModelCallTranscriptSink:
    """Best-effort writer; leo.harness.coordinator already isolates failures here."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession], *, ids: IdGenerator) -> None:
        self._sessions = sessions
        self._ids = ids

    async def record(
        self,
        *,
        run_id: str,
        task_id: str,
        scope: ScopeKey,
        request_id: str,
        iteration: int,
        raw_request: dict[str, JsonValue],
        raw_response: dict[str, JsonValue],
        occurred_at: datetime,
    ) -> None:
        del occurred_at
        async with self._sessions() as session, session.begin():
            await session.execute(
                pg_insert(ModelCallTranscriptRow)
                .values(
                    id=self._ids.new("model-call-transcript"),
                    run_id=run_id,
                    task_id=task_id,
                    organization_id=scope.organization_id,
                    strategy_id=scope.strategy_id,
                    request_id=request_id,
                    iteration=iteration,
                    raw_request=raw_request,
                    raw_response=raw_response,
                )
                .on_conflict_do_nothing(index_elements=["run_id", "request_id"])
            )
