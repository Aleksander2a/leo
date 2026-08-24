"""Read-only observability API over the agent's own tables.

Six tables mean the dashboard needs no projection layer, no provenance
reconstruction, and no event-payload decoding: a run row is the run, and its
step rows are literally the ReAct trace in order.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import Integer, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.agent.schema import Conversation, Memory, Message, Run, Step, ToolIndex

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


async def session_dep(request: Request):  # type: ignore[no-untyped-def]
    sessions = getattr(request.app.state, "sessions", None)
    if sessions is None:
        raise HTTPException(status_code=503, detail="database is not configured")
    async with sessions() as session:
        yield session


class Page:
    def __init__(
        self,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> None:
        self.limit = limit
        self.offset = offset


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


@router.get("/overview")
async def overview(session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    status_rows = (
        await session.execute(select(Run.status, func.count()).group_by(Run.status))
    ).all()
    statuses = {str(row[0]): int(row[1]) for row in status_rows}

    totals = (
        await session.execute(
            select(
                func.count(Run.id),
                func.coalesce(func.sum(Run.prompt_tokens + Run.completion_tokens), 0),
                func.coalesce(func.sum(Run.cost), 0.0),
                func.coalesce(func.sum(Run.tool_calls), 0),
                func.avg(
                    func.extract("epoch", Run.finished_at) - func.extract("epoch", Run.started_at)
                ),
            )
        )
    ).one()

    tool_rows = (
        await session.execute(
            select(
                Step.name,
                func.count(),
                func.sum(cast(Step.ok, Integer)),
            )
            .where(Step.kind == "tool")
            .group_by(Step.name)
            .order_by(func.count().desc())
            .limit(20)
        )
    ).all()

    memories = (await session.execute(select(func.count(Memory.id)).where(Memory.active))).scalar()
    conversations = (await session.execute(select(func.count(Conversation.id)))).scalar()

    answered = statuses.get("answered", 0)
    total_runs = int(totals[0] or 0)
    return {
        "run_status_counts": statuses,
        "total_runs": total_runs,
        "answer_rate": (answered / total_runs) if total_runs else None,
        "total_tokens": int(totals[1] or 0),
        "total_cost": float(totals[2] or 0.0),
        "total_tool_calls": int(totals[3] or 0),
        "avg_run_seconds": float(totals[4]) if totals[4] is not None else None,
        "active_memories": int(memories or 0),
        "conversations": int(conversations or 0),
        "tool_usage": [
            {
                "name": str(row[0]),
                "calls": int(row[1]),
                "succeeded": int(row[2] or 0),
                "failed": int(row[1]) - int(row[2] or 0),
            }
            for row in tool_rows
        ],
    }


@router.get("/runs")
async def list_runs(
    status: str | None = None,
    scope_key: str | None = None,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Run)
    if status:
        query = query.where(Run.status == status)
    if scope_key:
        query = query.where(Run.scope_key == scope_key)
    total = (await session.execute(select(func.count()).select_from(query.subquery()))).scalar()
    rows = (
        (
            await session.execute(
                query.order_by(Run.started_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [_run_summary(row) for row in rows],
        "total": int(total or 0),
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/runs/{run_id}")
async def get_run(run_id: str, session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    steps = (
        (await session.execute(select(Step).where(Step.run_id == run_id).order_by(Step.seq)))
        .scalars()
        .all()
    )
    return {
        **_run_summary(run),
        "answer": run.answer,
        "error": run.error,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "steps": [
            {
                "seq": step.seq,
                "kind": step.kind,
                "name": step.name,
                "ok": step.ok,
                "duration_ms": step.duration_ms,
                "arguments": step.arguments,
                "result": step.result,
                "created_at": _iso(step.created_at),
            }
            for step in steps
        ],
    }


@router.get("/failures")
async def list_failures(
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Run).where(Run.status != "answered", Run.finished_at.isnot(None))
    total = (await session.execute(select(func.count()).select_from(query.subquery()))).scalar()
    rows = (
        (
            await session.execute(
                query.order_by(Run.started_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [{**_run_summary(row), "error": row.error} for row in rows],
        "total": int(total or 0),
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/conversations")
async def list_conversations(
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    total = (await session.execute(select(func.count(Conversation.id)))).scalar()
    rows = (
        (
            await session.execute(
                select(Conversation)
                .order_by(Conversation.last_active_at.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    counts: dict[str, int] = {
        str(row[0]): int(row[1])
        for row in (
            await session.execute(select(Run.scope_key, func.count()).group_by(Run.scope_key))
        ).all()
    }
    return {
        "items": [
            {
                "id": row.id,
                "scope_key": row.scope_key,
                "provider": row.provider,
                "kind": row.kind,
                "title": row.title,
                "team_id": row.team_id,
                "channel_id": row.channel_id,
                "runs": int(counts.get(row.scope_key, 0)),
                "created_at": _iso(row.created_at),
                "last_active_at": _iso(row.last_active_at),
            }
            for row in rows
        ],
        "total": int(total or 0),
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/conversations/{scope_key:path}/messages")
async def conversation_messages(
    scope_key: str,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    rows = (
        (
            await session.execute(
                select(Message)
                .where(Message.scope_key == scope_key)
                .order_by(Message.id.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [
            {
                "id": row.id,
                "role": row.role,
                "content": row.content,
                "author_id": row.author_id,
                "run_id": row.run_id,
                "thread_key": row.thread_key,
                "created_at": _iso(row.created_at),
            }
            for row in reversed(rows)
        ],
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/memory")
async def list_memory(
    scope_key: str | None = None,
    kind: str | None = None,
    include_inactive: bool = False,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Memory)
    if not include_inactive:
        query = query.where(Memory.active.is_(True))
    if scope_key:
        query = query.where(Memory.scope_key == scope_key)
    if kind:
        query = query.where(Memory.kind == kind)
    total = (await session.execute(select(func.count()).select_from(query.subquery()))).scalar()
    rows = (
        (
            await session.execute(
                query.order_by(Memory.updated_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [
            {
                "id": row.id,
                "scope_key": row.scope_key,
                "kind": row.kind,
                "subject": row.subject,
                "content": row.content,
                "importance": row.importance,
                "active": row.active,
                "superseded_by": row.superseded_by,
                "source_run_id": row.source_run_id,
                "created_at": _iso(row.created_at),
                "updated_at": _iso(row.updated_at),
            }
            for row in rows
        ],
        "total": int(total or 0),
        "limit": page.limit,
        "offset": page.offset,
    }


@router.get("/tools")
async def list_tools(session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    rows = (await session.execute(select(ToolIndex).order_by(ToolIndex.name))).scalars().all()
    usage: dict[str, int] = {
        str(row[0]): int(row[1])
        for row in (
            await session.execute(
                select(Step.name, func.count()).where(Step.kind == "tool").group_by(Step.name)
            )
        ).all()
    }
    return {
        "items": [
            {
                "name": row.name,
                "description": row.description,
                "indexed": row.embedding is not None,
                "calls": int(usage.get(row.name, 0)),
                "updated_at": _iso(row.updated_at),
            }
            for row in rows
        ]
    }


def _run_summary(run: Run) -> dict[str, Any]:
    return {
        "id": run.id,
        "scope_key": run.scope_key,
        "conversation_id": run.conversation_id,
        "actor_id": run.actor_id,
        "question": run.question,
        "status": run.status,
        "model": run.model,
        "turns": run.turns,
        "tool_calls": run.tool_calls,
        "total_tokens": run.prompt_tokens + run.completion_tokens,
        "cost": run.cost,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }
