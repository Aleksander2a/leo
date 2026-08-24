"""Scope-isolated memory: recall by meaning, write and revise by tool call.

Three things make this work where the previous memory system did not:

1. **Isolation is a WHERE clause.** Every read is ``scope_key = :scope``. A DM's
   memories are not filtered out of a channel's results -- they are never in
   them. There is no authority object to misconfigure and no plane to leak
   across.

2. **Recall is semantic, not lexical.** The question is embedded and compared to
   memory embeddings with pgvector's cosine distance, so "what did I say about
   risk?" finds "prefers max 20% drawdown" without sharing a single keyword.
   When embeddings are unavailable the query degrades to recency, which returns
   something useful rather than failing.

3. **Updates supersede, they do not overwrite.** Revising a memory writes a new
   row and marks the old one inactive, pointing at its replacement.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.agent.contracts import (
    RunPhase,
    Scope,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.agent.llm import LLM
from leo.agent.schema import Memory

logger = logging.getLogger(__name__)

MEMORY_KINDS = ("fact", "preference", "decision", "context", "task")
RECALL_LIMIT = 8


@dataclass(frozen=True)
class RecalledMemory:
    id: str
    kind: str
    subject: str
    content: str
    importance: int
    updated_at: datetime
    similarity: float | None

    def render(self) -> str:
        head = f"[{self.kind}] {self.subject}: " if self.subject else f"[{self.kind}] "
        return f"{head}{self.content}"


class MemoryService:
    """All memory reads and writes for one scope."""

    def __init__(
        self,
        *,
        sessions: async_sessionmaker[AsyncSession],
        llm: LLM | None = None,
    ) -> None:
        self._sessions = sessions
        self._llm = llm

    async def recall(
        self,
        scope: Scope,
        query: str,
        *,
        limit: int = RECALL_LIMIT,
    ) -> list[RecalledMemory]:
        """Return this scope's most relevant memories for ``query``."""

        vector = await self._embed_one(query)
        async with self._sessions() as session:
            if vector is not None:
                distance = Memory.embedding.cosine_distance(vector)
                stmt = (
                    select(Memory, distance.label("distance"))
                    .where(
                        Memory.scope_key == scope.key,
                        Memory.active.is_(True),
                        Memory.embedding.isnot(None),
                    )
                    .order_by(distance)
                    .limit(limit)
                )
                rows = (await session.execute(stmt)).all()
                # Cosine distance above ~0.75 is noise; surfacing it would push
                # unrelated facts into the prompt as if they were relevant.
                relevant = [
                    _recalled(row[0], similarity=1.0 - float(row[1]))
                    for row in rows
                    if float(row[1]) < 0.75
                ]
                if relevant:
                    return relevant
            # Either there are no vectors, or nothing cleared the relevance bar.
            # Fall through to what this scope holds most firmly rather than
            # reporting amnesia -- an empty result here is only correct when the
            # scope genuinely holds nothing.
            stmt_recent = (
                select(Memory)
                .where(Memory.scope_key == scope.key, Memory.active.is_(True))
                .order_by(Memory.importance.desc(), Memory.updated_at.desc())
                .limit(limit)
            )
            recent = (await session.execute(stmt_recent)).scalars().all()
            return [_recalled(item, similarity=None) for item in recent]

    async def list_all(self, scope: Scope, *, limit: int = 100) -> list[RecalledMemory]:
        async with self._sessions() as session:
            stmt = (
                select(Memory)
                .where(Memory.scope_key == scope.key, Memory.active.is_(True))
                .order_by(Memory.updated_at.desc())
                .limit(limit)
            )
            return [
                _recalled(item, similarity=None)
                for item in (await session.execute(stmt)).scalars().all()
            ]

    async def write(
        self,
        scope: Scope,
        *,
        content: str,
        kind: str = "fact",
        subject: str = "",
        importance: int = 3,
        run_id: str | None = None,
        supersedes: str | None = None,
    ) -> RecalledMemory:
        content = content.strip()
        if not content:
            raise ValueError("a memory needs content")
        kind = kind if kind in MEMORY_KINDS else "fact"
        importance = max(1, min(5, importance))
        vector = await self._embed_one(f"{subject}. {content}" if subject else content)
        now = datetime.now(UTC)
        record = Memory(
            id=f"mem-{uuid.uuid4()}",
            scope_key=scope.key,
            kind=kind,
            subject=subject.strip()[:255],
            content=content,
            importance=importance,
            embedding=vector,
            source_run_id=run_id,
            author_id=scope.actor_id,
            active=True,
            created_at=now,
            updated_at=now,
        )
        async with self._sessions() as session, session.begin():
            if supersedes:
                await session.execute(
                    update(Memory)
                    .where(Memory.id == supersedes, Memory.scope_key == scope.key)
                    .values(active=False, superseded_by=record.id, updated_at=now)
                )
            session.add(record)
        return _recalled(record, similarity=None)

    async def forget(self, scope: Scope, memory_id: str) -> bool:
        async with self._sessions() as session, session.begin():
            result = await session.execute(
                update(Memory)
                .where(
                    Memory.id == memory_id,
                    Memory.scope_key == scope.key,
                    Memory.active.is_(True),
                )
                .values(active=False, updated_at=datetime.now(UTC))
                .returning(Memory.id)
            )
            return result.scalar_one_or_none() is not None

    async def _embed_one(self, text: str) -> list[float] | None:
        if self._llm is None or not text.strip():
            return None
        vectors = await self._llm.embed([text[:8000]])
        return vectors[0] if vectors else None


def _recalled(record: Memory, *, similarity: float | None) -> RecalledMemory:
    return RecalledMemory(
        id=record.id,
        kind=record.kind,
        subject=record.subject or "",
        content=record.content,
        importance=record.importance,
        updated_at=record.updated_at,
        similarity=similarity,
    )


# --------------------------------------------------------------------------
# Memory as tools the model can call
# --------------------------------------------------------------------------


class _MemoryTool:
    """Base for the three memory tools; each is scoped to one conversation."""

    def __init__(self, service: MemoryService, scope: Scope, run_id: str) -> None:
        self._service = service
        self._scope = scope
        self._run_id = run_id

    @property
    def spec(self) -> ToolSpec:  # pragma: no cover - overridden
        raise NotImplementedError

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return dict(arguments)

    def _ok(self, data: dict[str, Any], reference: str) -> ToolSuccess:
        return ToolSuccess(
            data=data,
            source=SourceRef(provider="leo-memory", reference=reference),
            observed_at=datetime.now(UTC),
        )


class MemorySearchTool(_MemoryTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory.search",
            description=(
                "Search what you remember about this specific conversation: the user's "
                "stated preferences, constraints, holdings, prior decisions, and facts "
                "they told you before. Searches by meaning, so paraphrase freely. "
                "Only ever returns memories from this channel or DM."
            ),
            domain="memory",
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            max_result_bytes=8192,
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What you want to remember, in natural language.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 8},
                },
                "required": ["query"],
            },
        )

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolFailure(
                code="missing_query", safe_message="Provide a query describing what to recall."
            )
        raw_limit = arguments.get("limit")
        limit = int(raw_limit) if isinstance(raw_limit, (int, float)) else RECALL_LIMIT
        found = await self._service.recall(self._scope, query, limit=max(1, min(20, limit)))
        return self._ok(
            {
                "query": query,
                "count": len(found),
                "memories": [
                    {
                        "id": item.id,
                        "kind": item.kind,
                        "subject": item.subject,
                        "content": item.content,
                        "importance": item.importance,
                        "updated_at": item.updated_at.isoformat(),
                    }
                    for item in found
                ],
            },
            reference=f"search:{self._scope.key}",
        )


class MemoryWriteTool(_MemoryTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory.write",
            description=(
                "Remember something durable about this conversation for future turns: a "
                "stated preference, constraint, holding, decision, or personal fact. Use it "
                "when the user tells you something that should still be true next week. "
                "Do not use it for facts you looked up on the web -- those go stale. "
                "Pass 'supersedes' with an existing memory id to replace an outdated one."
            ),
            domain="memory",
            effect=ToolEffect.STATE_MUTATION,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            max_result_bytes=2048,
            input_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The fact, stated plainly and self-containedly.",
                    },
                    "kind": {"type": "string", "enum": list(MEMORY_KINDS), "default": "fact"},
                    "subject": {
                        "type": "string",
                        "description": "Short topic label, e.g. 'risk tolerance'.",
                    },
                    "importance": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                    "supersedes": {
                        "type": "string",
                        "description": "Id of a memory this one replaces, if any.",
                    },
                },
                "required": ["content"],
            },
        )

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        content = str(arguments.get("content") or "").strip()
        if not content:
            return ToolFailure(
                code="missing_content", safe_message="Provide the content to remember."
            )
        raw_importance = arguments.get("importance")
        try:
            stored = await self._service.write(
                self._scope,
                content=content,
                kind=str(arguments.get("kind") or "fact"),
                subject=str(arguments.get("subject") or ""),
                importance=(int(raw_importance) if isinstance(raw_importance, (int, float)) else 3),
                run_id=self._run_id,
                supersedes=(str(arguments["supersedes"]) if arguments.get("supersedes") else None),
            )
        except ValueError as exc:
            return ToolFailure(code="invalid_memory", safe_message=str(exc))
        return self._ok(
            {
                "stored": True,
                "id": stored.id,
                "kind": stored.kind,
                "subject": stored.subject,
                "content": stored.content,
            },
            reference=f"write:{stored.id}",
        )


class MemoryForgetTool(_MemoryTool):
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="memory.forget",
            description=(
                "Retire a memory that is wrong or no longer true. Find its id with "
                "memory.search first. Prefer memory.write with 'supersedes' when the fact "
                "has merely changed rather than become irrelevant."
            ),
            domain="memory",
            effect=ToolEffect.STATE_MUTATION,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            max_result_bytes=1024,
            input_schema={
                "type": "object",
                "properties": {"id": {"type": "string", "description": "The memory id."}},
                "required": ["id"],
            },
        )

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        memory_id = str(arguments.get("id") or "").strip()
        if not memory_id:
            return ToolFailure(code="missing_id", safe_message="Provide the memory id to forget.")
        removed = await self._service.forget(self._scope, memory_id)
        if not removed:
            return ToolFailure(
                code="not_found",
                safe_message=f"No active memory {memory_id!r} exists in this conversation.",
            )
        return self._ok({"forgotten": True, "id": memory_id}, reference=f"forget:{memory_id}")


def build_memory_tools(
    service: MemoryService,
    scope: Scope,
    run_id: str,
) -> list[Any]:
    return [
        MemorySearchTool(service, scope, run_id),
        MemoryWriteTool(service, scope, run_id),
        MemoryForgetTool(service, scope, run_id),
    ]
