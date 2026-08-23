"""Slack conversation / thread browser (lower-priority, best-effort view)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import PageParams, get_session
from leo.persistence.schema import ConversationRow, ThreadRow

router = APIRouter()


@router.get("/conversations")
async def list_conversations(
    page: PageParams = Depends(), session: AsyncSession = Depends(get_session)
) -> dict[str, Any]:
    total = await session.scalar(select(func.count()).select_from(ConversationRow))
    rows = (
        (
            await session.execute(
                select(ConversationRow)
                # `updated_at` alone is not a total order: conversations touched in
                # the same transaction share a timestamp, so paging could show one
                # twice and skip another. The id breaks the tie deterministically.
                .order_by(ConversationRow.updated_at.desc(), ConversationRow.id)
                .limit(page.limit)
                .offset(page.offset)
            )
        )
        .scalars()
        .all()
    )

    # One grouped count for the whole page instead of a query per row -- at the
    # 200-row page cap that was 200 round trips to render one list.
    # `conversation_threads` is a newer, still-unpopulated join table; the durable
    # thread<->conversation link in this demo dataset lives on `threads.conversation_id`.
    thread_counts: dict[str, int] = {}
    if rows:
        counted = await session.execute(
            select(ThreadRow.conversation_id, func.count())
            .where(ThreadRow.conversation_id.in_([item.id for item in rows]))
            .group_by(ThreadRow.conversation_id)
        )
        thread_counts = {
            conversation_id: count for conversation_id, count in counted.all() if conversation_id
        }

    items = []
    for conversation in rows:
        thread_count = thread_counts.get(conversation.id, 0)
        items.append(
            {
                "id": conversation.id,
                "provider": conversation.provider,
                "team_id": conversation.team_id,
                "kind": conversation.kind,
                "bot_presence": conversation.bot_presence,
                "lifecycle": conversation.lifecycle,
                "external_provenance": conversation.external_provenance,
                "thread_count": thread_count or 0,
                "created_at": conversation.created_at,
                "updated_at": conversation.updated_at,
            }
        )
    return {"items": items, "total": total or 0, "limit": page.limit, "offset": page.offset}
