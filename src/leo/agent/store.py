"""Durable reads and writes for conversations, history, runs, and traces."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import desc, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.agent.contracts import Scope
from leo.agent.llm import Usage
from leo.agent.schema import Conversation, Message, Run, Step
from leo.agent.tools import ToolResult

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 24
HISTORY_CHAR_BUDGET = 24_000


@dataclass(frozen=True)
class Turn:
    role: str
    content: str
    author_id: str | None = None
    created_at: datetime | None = None


class AgentStore:
    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    # -- conversations ----------------------------------------------------

    async def ensure_conversation(
        self,
        scope: Scope,
        *,
        provider: str = "slack",
        team_id: str | None = None,
        channel_id: str | None = None,
        kind: str = "channel",
        title: str | None = None,
    ) -> str:
        now = datetime.now(UTC)
        async with self._sessions() as session, session.begin():
            stmt = (
                insert(Conversation)
                .values(
                    id=f"conv-{uuid.uuid4()}",
                    scope_key=scope.key,
                    provider=provider,
                    team_id=team_id,
                    channel_id=channel_id,
                    kind=kind,
                    title=title,
                    created_at=now,
                    last_active_at=now,
                )
                .on_conflict_do_update(
                    index_elements=[Conversation.scope_key],
                    set_={"last_active_at": now},
                )
                .returning(Conversation.id)
            )
            return str((await session.execute(stmt)).scalar_one())

    # -- history ----------------------------------------------------------

    async def record_message(
        self,
        scope: Scope,
        conversation_id: str,
        *,
        role: str,
        content: str,
        thread_key: str | None = None,
        run_id: str | None = None,
        author_id: str | None = None,
        external_id: str | None = None,
    ) -> None:
        """Append one turn. Idempotent when ``external_id`` is supplied."""

        async with self._sessions() as session, session.begin():
            stmt = insert(Message).values(
                scope_key=scope.key,
                conversation_id=conversation_id,
                run_id=run_id,
                thread_key=thread_key,
                role=role,
                content=content,
                author_id=author_id,
                external_id=external_id,
                created_at=datetime.now(UTC),
            )
            if external_id is not None:
                # The uniqueness index is partial, so Postgres will only infer it
                # when the predicate is restated here.
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=[Message.scope_key, Message.external_id],
                    index_where=Message.external_id.isnot(None),
                )
            await session.execute(stmt)

    async def history(
        self,
        scope: Scope,
        *,
        thread_key: str | None = None,
        limit: int = HISTORY_LIMIT,
        char_budget: int = HISTORY_CHAR_BUDGET,
        exclude_external_id: str | None = None,
    ) -> list[Turn]:
        """Recent conversation, oldest first, bounded by count and characters.

        A threaded request reads its own thread; an unthreaded one reads the
        channel's recent flow. Either way the read is confined to ``scope.key``.
        """

        async with self._sessions() as session:
            stmt = select(Message).where(Message.scope_key == scope.key)
            if thread_key:
                stmt = stmt.where(Message.thread_key == thread_key)
            if exclude_external_id:
                stmt = stmt.where(
                    (Message.external_id.is_(None)) | (Message.external_id != exclude_external_id)
                )
            rows = (
                (await session.execute(stmt.order_by(desc(Message.id)).limit(limit)))
                .scalars()
                .all()
            )
        newest_first: list[Turn] = []
        spent = 0
        for row in rows:
            content = (row.content or "").strip()
            if not content:
                continue
            spent += len(content)
            if spent > char_budget and newest_first:
                break
            newest_first.append(
                Turn(
                    role=row.role,
                    content=content,
                    author_id=row.author_id,
                    created_at=row.created_at,
                )
            )
        return list(reversed(newest_first))

    # -- runs -------------------------------------------------------------

    async def start_run(
        self,
        scope: Scope,
        conversation_id: str,
        *,
        question: str,
        model: str,
        thread_key: str | None = None,
    ) -> str:
        run_id = f"run-{uuid.uuid4()}"
        async with self._sessions() as session, session.begin():
            session.add(
                Run(
                    id=run_id,
                    scope_key=scope.key,
                    conversation_id=conversation_id,
                    actor_id=scope.actor_id,
                    thread_key=thread_key,
                    question=question,
                    status="running",
                    model=model,
                    started_at=datetime.now(UTC),
                )
            )
        return run_id

    async def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        answer: str | None = None,
        error: str | None = None,
        turns: int = 0,
        tool_calls: int = 0,
        usage: Usage | None = None,
    ) -> None:
        usage = usage or Usage()
        async with self._sessions() as session, session.begin():
            await session.execute(
                update(Run)
                .where(Run.id == run_id)
                .values(
                    status=status,
                    answer=answer,
                    error=error,
                    turns=turns,
                    tool_calls=tool_calls,
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens,
                    cost=usage.cost,
                    finished_at=datetime.now(UTC),
                )
            )

    async def record_model_step(
        self,
        run_id: str,
        *,
        seq: int,
        tool_names: list[str],
        content_preview: str,
        finish_reason: str,
        usage: Usage,
        duration_ms: int,
    ) -> None:
        await self._record_step(
            run_id,
            seq=seq,
            kind="model",
            name=finish_reason,
            arguments={"tools_offered": tool_names},
            result={
                "content_preview": content_preview[:1000],
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost": usage.cost,
            },
            ok=True,
            duration_ms=duration_ms,
        )

    async def record_tool_step(self, run_id: str, *, seq: int, result: ToolResult) -> None:
        await self._record_step(
            run_id,
            seq=seq,
            kind="tool",
            name=result.name,
            arguments=result.arguments,
            result=result.payload,
            ok=result.ok,
            duration_ms=result.duration_ms,
        )

    async def _record_step(
        self,
        run_id: str,
        *,
        seq: int,
        kind: str,
        name: str,
        arguments: dict[str, Any] | None,
        result: dict[str, Any] | None,
        ok: bool,
        duration_ms: int,
    ) -> None:
        try:
            async with self._sessions() as session, session.begin():
                session.add(
                    Step(
                        run_id=run_id,
                        seq=seq,
                        kind=kind,
                        name=name[:128],
                        arguments=arguments,
                        result=result,
                        ok=ok,
                        duration_ms=duration_ms,
                        created_at=datetime.now(UTC),
                    )
                )
        except Exception:
            logger.warning("could not record %s step for run %s", kind, run_id, exc_info=True)
