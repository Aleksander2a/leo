"""Read-only observability API over the agent's own tables.

Six tables mean no projection layer and no event decoding: a run row *is* the
run, and its step rows are literally the ReAct trace in order. Every endpoint
here is a query over those rows, and every one of them is a GET.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import Integer, cast, func, or_, select, text
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


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _page(items: list[Any], total: int, page: Page) -> dict[str, Any]:
    return {"items": items, "total": total, "limit": page.limit, "offset": page.offset}


async def _count(session: AsyncSession, query: Any) -> int:
    total = (await session.execute(select(func.count()).select_from(query.subquery()))).scalar()
    return int(total or 0)


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@router.get("/overview")
async def overview(
    days: int = Query(default=14, ge=1, le=90),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
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
                func.coalesce(func.sum(Run.turns), 0),
            )
        )
    ).one()

    duration = func.extract("epoch", Run.finished_at) - func.extract("epoch", Run.started_at)
    latency = (
        await session.execute(
            select(
                func.avg(duration),
                func.percentile_cont(0.5).within_group(duration),
                func.percentile_cont(0.95).within_group(duration),
            ).where(Run.finished_at.isnot(None))
        )
    ).one()

    since = datetime.now(UTC) - timedelta(days=days)
    day = func.date_trunc("day", Run.started_at)
    activity_rows = (
        await session.execute(
            select(
                day.label("day"),
                func.count(),
                func.sum(cast(Run.status == "answered", Integer)),
                func.coalesce(func.sum(Run.cost), 0.0),
            )
            .where(Run.started_at >= since)
            .group_by(day)
            .order_by(day)
        )
    ).all()

    tool_rows = (
        await session.execute(
            select(
                Step.name,
                func.count(),
                func.sum(cast(Step.ok, Integer)),
                func.avg(Step.duration_ms),
            )
            .where(Step.kind == "tool")
            .group_by(Step.name)
            .order_by(func.count().desc())
            .limit(25)
        )
    ).all()

    # Inline the JSON key rather than binding it: two bound parameters are not
    # recognised as the same expression, and Postgres rejects the GROUP BY.
    error_code = text("result ->> 'error'")
    error_rows = (
        await session.execute(
            select(error_code, func.count())
            .select_from(Step)
            .where(Step.kind == "tool", Step.ok.is_(False))
            .group_by(error_code)
            .order_by(func.count().desc())
            .limit(10)
        )
    ).all()

    memories = (await session.execute(select(func.count(Memory.id)).where(Memory.active))).scalar()
    conversations = (await session.execute(select(func.count(Conversation.id)))).scalar()
    messages = (await session.execute(select(func.count(Message.id)))).scalar()

    total_runs = int(totals[0] or 0)
    answered = statuses.get("answered", 0)
    return {
        "run_status_counts": statuses,
        "total_runs": total_runs,
        "answered_runs": answered,
        "answer_rate": (answered / total_runs) if total_runs else None,
        "total_tokens": int(totals[1] or 0),
        "total_cost": float(totals[2] or 0.0),
        "total_tool_calls": int(totals[3] or 0),
        "total_model_turns": int(totals[4] or 0),
        "avg_run_seconds": float(latency[0]) if latency[0] is not None else None,
        "p50_run_seconds": float(latency[1]) if latency[1] is not None else None,
        "p95_run_seconds": float(latency[2]) if latency[2] is not None else None,
        "active_memories": int(memories or 0),
        "conversations": int(conversations or 0),
        "messages": int(messages or 0),
        "activity": [
            {
                "day": _iso(row[0]),
                "runs": int(row[1]),
                "answered": int(row[2] or 0),
                "cost": float(row[3] or 0.0),
            }
            for row in activity_rows
        ],
        "tool_usage": [
            {
                "name": str(row[0]),
                "calls": int(row[1]),
                "succeeded": int(row[2] or 0),
                "failed": int(row[1]) - int(row[2] or 0),
                "avg_ms": float(row[3]) if row[3] is not None else None,
            }
            for row in tool_rows
        ],
        "tool_errors": [
            {"code": str(row[0] or "unknown"), "count": int(row[1])} for row in error_rows
        ],
    }


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@router.get("/runs")
async def list_runs(
    status: str | None = None,
    scope_key: str | None = None,
    q: str | None = None,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Run)
    if status:
        query = query.where(Run.status == status)
    if scope_key:
        query = query.where(Run.scope_key == scope_key)
    if q:
        pattern = f"%{q}%"
        query = query.where(or_(Run.question.ilike(pattern), Run.answer.ilike(pattern)))
    total = await _count(session, query)
    rows = (
        (
            await session.execute(
                query.order_by(Run.started_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return _page([_run_summary(row) for row in rows], total, page)


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
    conversation = (
        await session.execute(select(Conversation).where(Conversation.id == run.conversation_id))
    ).scalar_one_or_none()
    memories_written = (
        await session.execute(select(func.count(Memory.id)).where(Memory.source_run_id == run_id))
    ).scalar()
    return {
        **_run_summary(run),
        "answer": run.answer,
        "error": run.error,
        "prompt_tokens": run.prompt_tokens,
        "completion_tokens": run.completion_tokens,
        "memories_written": int(memories_written or 0),
        "conversation": _conversation_summary(conversation) if conversation else None,
        "steps": [_step(step) for step in steps],
    }


@router.get("/failures")
async def list_failures(
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    """Runs that finished without an answer, plus the tool calls that failed inside them."""

    query = select(Run).where(Run.status != "answered", Run.finished_at.isnot(None))
    total = await _count(session, query)
    rows = (
        (
            await session.execute(
                query.order_by(Run.started_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    run_ids = [row.id for row in rows]
    failed_steps: dict[str, list[dict[str, Any]]] = {}
    if run_ids:
        step_rows = (
            (
                await session.execute(
                    select(Step)
                    .where(Step.run_id.in_(run_ids), Step.ok.is_(False))
                    .order_by(Step.run_id, Step.seq)
                )
            )
            .scalars()
            .all()
        )
        for step in step_rows:
            failed_steps.setdefault(step.run_id, []).append(
                {
                    "name": step.name,
                    "error": (step.result or {}).get("error"),
                    "message": (step.result or {}).get("message"),
                }
            )
    return _page(
        [
            {
                **_run_summary(row),
                "error": row.error,
                "failed_tool_calls": failed_steps.get(row.id, []),
            }
            for row in rows
        ],
        total,
        page,
    )


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.get("/conversations")
async def list_conversations(
    kind: str | None = None,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Conversation)
    if kind:
        query = query.where(Conversation.kind == kind)
    total = await _count(session, query)
    rows = (
        (
            await session.execute(
                query.order_by(Conversation.last_active_at.desc())
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    run_counts = await _counts_by_scope(session, Run.scope_key, Run.id)
    memory_counts = await _counts_by_scope(
        session, Memory.scope_key, Memory.id, where=Memory.active.is_(True)
    )
    message_counts = await _counts_by_scope(session, Message.scope_key, Message.id)
    return _page(
        [
            _conversation_summary(
                row,
                runs=run_counts.get(row.scope_key, 0),
                memories=memory_counts.get(row.scope_key, 0),
                messages=message_counts.get(row.scope_key, 0),
            )
            for row in rows
        ],
        total,
        page,
    )


@router.get("/conversations/{scope_key}")
async def get_conversation(
    scope_key: str,
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    conversation = (
        await session.execute(select(Conversation).where(Conversation.scope_key == scope_key))
    ).scalar_one_or_none()
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = (
        (
            await session.execute(
                select(Message)
                .where(Message.scope_key == scope_key)
                .order_by(Message.id.desc())
                .limit(100)
            )
        )
        .scalars()
        .all()
    )
    runs = (
        (
            await session.execute(
                select(Run)
                .where(Run.scope_key == scope_key)
                .order_by(Run.started_at.desc())
                .limit(25)
            )
        )
        .scalars()
        .all()
    )
    memories = (
        (
            await session.execute(
                select(Memory)
                .where(Memory.scope_key == scope_key, Memory.active.is_(True))
                .order_by(Memory.updated_at.desc())
                .limit(50)
            )
        )
        .scalars()
        .all()
    )
    # The summary's `runs`/`messages`/`memories` are counts; the detail's lists
    # get distinct names so neither shadows the other.
    return {
        **_conversation_summary(
            conversation,
            runs=len(runs),
            memories=len(memories),
            messages=len(messages),
        ),
        "recent_messages": [_message(row) for row in reversed(messages)],
        "recent_runs": [_run_summary(row) for row in runs],
        "recent_memories": [_memory(row) for row in memories],
    }


@router.get("/conversations/{scope_key}/messages")
async def conversation_messages(
    scope_key: str,
    page: Page = Depends(Page),
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    query = select(Message).where(Message.scope_key == scope_key)
    total = await _count(session, query)
    rows = (
        (
            await session.execute(
                query.order_by(Message.id.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return _page([_message(row) for row in reversed(rows)], total, page)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


@router.get("/memory")
async def list_memory(
    scope_key: str | None = None,
    kind: str | None = None,
    q: str | None = None,
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
    if q:
        pattern = f"%{q}%"
        query = query.where(or_(Memory.content.ilike(pattern), Memory.subject.ilike(pattern)))
    total = await _count(session, query)
    rows = (
        (
            await session.execute(
                query.order_by(Memory.updated_at.desc()).limit(page.limit).offset(page.offset)
            )
        )
        .scalars()
        .all()
    )
    return _page([_memory(row) for row in rows], total, page)


@router.get("/memory-kinds")
async def memory_kinds(session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    rows = (
        await session.execute(
            select(Memory.kind, func.count()).where(Memory.active).group_by(Memory.kind)
        )
    ).all()
    return {"items": [{"kind": str(row[0]), "count": int(row[1])} for row in rows]}


@router.get("/memory/{memory_id}")
async def get_memory(
    memory_id: str,
    session: AsyncSession = Depends(session_dep),
) -> dict[str, Any]:
    record = (
        await session.execute(select(Memory).where(Memory.id == memory_id))
    ).scalar_one_or_none()
    if record is None:
        raise HTTPException(status_code=404, detail="memory not found")

    # Walk both directions of the supersession chain: what this replaced, and
    # what replaced it. Updates are non-destructive, so the history is readable.
    later: list[dict[str, Any]] = []
    cursor = record
    seen = {record.id}
    while cursor.superseded_by:
        successor = (
            await session.execute(select(Memory).where(Memory.id == cursor.superseded_by))
        ).scalar_one_or_none()
        if successor is None or successor.id in seen:
            break
        seen.add(successor.id)
        later.append(_memory(successor))
        cursor = successor
    earlier = (
        (
            await session.execute(
                select(Memory).where(
                    Memory.superseded_by == memory_id, Memory.scope_key == record.scope_key
                )
            )
        )
        .scalars()
        .all()
    )
    source_run = None
    if record.source_run_id:
        run = (
            await session.execute(select(Run).where(Run.id == record.source_run_id))
        ).scalar_one_or_none()
        source_run = _run_summary(run) if run else None
    return {
        **_memory(record),
        "supersedes": [_memory(row) for row in earlier],
        "superseded_chain": later,
        "source_run": source_run,
    }


# ---------------------------------------------------------------------------
# Tools and scopes
# ---------------------------------------------------------------------------


@router.get("/tools")
async def list_tools(session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    """Every indexed tool, with how often it has actually been used."""

    indexed = (await session.execute(select(ToolIndex).order_by(ToolIndex.name))).scalars().all()
    usage_rows = (
        await session.execute(
            select(
                Step.name,
                func.count(),
                func.sum(cast(Step.ok, Integer)),
                func.avg(Step.duration_ms),
                func.max(Step.created_at),
            )
            .where(Step.kind == "tool")
            .group_by(Step.name)
        )
    ).all()
    usage: dict[str, dict[str, Any]] = {
        str(row[0]): {
            "calls": int(row[1]),
            "succeeded": int(row[2] or 0),
            "failed": int(row[1]) - int(row[2] or 0),
            "avg_ms": float(row[3]) if row[3] is not None else None,
            "last_used_at": _iso(row[4]),
        }
        for row in usage_rows
    }
    error_code = text("result ->> 'error'")
    error_rows = (
        await session.execute(
            select(Step.name, error_code, func.count())
            .select_from(Step)
            .where(Step.kind == "tool", Step.ok.is_(False))
            .group_by(Step.name, error_code)
            .order_by(func.count().desc())
        )
    ).all()
    errors: dict[str, list[dict[str, Any]]] = {}
    for row in error_rows:
        errors.setdefault(str(row[0]), []).append(
            {"code": str(row[1] or "unknown"), "count": int(row[2])}
        )

    by_name = {entry.name: entry for entry in indexed}
    items: list[dict[str, Any]] = []
    for name in sorted(set(by_name) | set(usage)):
        entry = by_name.get(name)
        stats = usage.get(name, {})
        items.append(
            {
                "name": name,
                "domain": name.split(".", 1)[0],
                "description": entry.description if entry else None,
                "indexed": bool(entry and entry.embedding is not None),
                "calls": stats.get("calls", 0),
                "succeeded": stats.get("succeeded", 0),
                "failed": stats.get("failed", 0),
                "avg_ms": stats.get("avg_ms"),
                "last_used_at": stats.get("last_used_at"),
                "errors": errors.get(name, []),
                "updated_at": _iso(entry.updated_at) if entry else None,
            }
        )
    return {"items": items}


@router.get("/scopes")
async def list_scopes(session: AsyncSession = Depends(session_dep)) -> dict[str, Any]:
    """Scope keys with a label, for filter controls."""

    rows = (
        (
            await session.execute(
                select(Conversation).order_by(Conversation.last_active_at.desc()).limit(200)
            )
        )
        .scalars()
        .all()
    )
    return {
        "items": [
            {"scope_key": row.scope_key, "label": row.title or row.scope_key, "kind": row.kind}
            for row in rows
        ]
    }


# ---------------------------------------------------------------------------
# Row shaping
# ---------------------------------------------------------------------------


async def _counts_by_scope(
    session: AsyncSession,
    scope_column: Any,
    id_column: Any,
    *,
    where: Any = None,
) -> dict[str, int]:
    query = select(scope_column, func.count(id_column)).group_by(scope_column)
    if where is not None:
        query = query.where(where)
    rows = (await session.execute(query)).all()
    return {str(row[0]): int(row[1]) for row in rows}


def _run_summary(run: Run) -> dict[str, Any]:
    seconds: float | None = None
    if run.finished_at is not None and run.started_at is not None:
        seconds = (run.finished_at - run.started_at).total_seconds()
    return {
        "id": run.id,
        "scope_key": run.scope_key,
        "conversation_id": run.conversation_id,
        "actor_id": run.actor_id,
        "thread_key": run.thread_key,
        "question": run.question,
        "status": run.status,
        "model": run.model,
        "turns": run.turns,
        "tool_calls": run.tool_calls,
        "total_tokens": run.prompt_tokens + run.completion_tokens,
        "cost": run.cost,
        "duration_seconds": seconds,
        "started_at": _iso(run.started_at),
        "finished_at": _iso(run.finished_at),
    }


def _step(step: Step) -> dict[str, Any]:
    return {
        "seq": step.seq,
        "kind": step.kind,
        "name": step.name,
        "ok": step.ok,
        "duration_ms": step.duration_ms,
        "arguments": step.arguments,
        "result": step.result,
        "created_at": _iso(step.created_at),
    }


def _message(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "author_id": message.author_id,
        "run_id": message.run_id,
        "thread_key": message.thread_key,
        "created_at": _iso(message.created_at),
    }


def _memory(record: Memory) -> dict[str, Any]:
    return {
        "id": record.id,
        "scope_key": record.scope_key,
        "kind": record.kind,
        "subject": record.subject,
        "content": record.content,
        "importance": record.importance,
        "active": record.active,
        "superseded_by": record.superseded_by,
        "source_run_id": record.source_run_id,
        "author_id": record.author_id,
        "embedded": record.embedding is not None,
        "created_at": _iso(record.created_at),
        "updated_at": _iso(record.updated_at),
    }


def _conversation_summary(
    conversation: Conversation,
    *,
    runs: int | None = None,
    memories: int | None = None,
    messages: int | None = None,
) -> dict[str, Any]:
    return {
        "id": conversation.id,
        "scope_key": conversation.scope_key,
        "provider": conversation.provider,
        "kind": conversation.kind,
        "title": conversation.title,
        "team_id": conversation.team_id,
        "channel_id": conversation.channel_id,
        "runs": runs,
        "memories": memories,
        "messages": messages,
        "created_at": _iso(conversation.created_at),
        "last_active_at": _iso(conversation.last_active_at),
    }
