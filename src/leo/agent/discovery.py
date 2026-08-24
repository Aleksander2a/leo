"""Semantic tool discovery.

Leo carries roughly thirty tools. Showing all of them on every turn is how the
old runtime produced its worst behaviour: asked for a *crypto strategy*, the
model called five different equity-quote tools in a row for the symbol "BTC",
because they were all in front of it and all looked plausible.

So the turn opens with the tools that are semantically close to what was
actually asked, ranked by embedding similarity over each tool's own
description -- no hand-maintained keyword tables, which is the other thing that
kept breaking. Ranking is not a gate: anything not shown up front is reachable
through ``tools.find``, which searches the same index and adds what it finds to
the live tool set for the rest of the run.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.agent.contracts import (
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.agent.llm import LLM
from leo.agent.schema import ToolIndex
from leo.agent.tools import ToolRegistry

logger = logging.getLogger(__name__)

#: Always available, whatever the question. Memory is how Leo stays coherent
#: across turns, and a general web route is the fallback for anything the
#: specialised providers do not cover.
ALWAYS_AVAILABLE = (
    "memory.search",
    "memory.write",
    "memory.forget",
    "tools.find",
    "web.search_tavily",
    "web.search_exa",
    "web.fetch_public_text",
    "web.search_public",
)

DEFAULT_TOOL_BUDGET = 14


def searchable_text(spec: ToolSpec) -> str:
    return f"{spec.name} ({spec.domain}): {spec.description}"


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True)
class ScoredTool:
    name: str
    score: float
    description: str


class ToolDiscovery:
    """Embeds tool descriptions once, then ranks them against a question."""

    def __init__(
        self,
        *,
        registry: ToolRegistry,
        llm: LLM | None,
        sessions: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._registry = registry
        self._llm = llm
        self._sessions = sessions
        self._vectors: dict[str, list[float]] = {}
        self._loaded = False

    async def prepare(self) -> None:
        """Populate the embedding index, reusing anything already cached."""

        if self._loaded or self._llm is None:
            self._loaded = True
            return
        self._loaded = True
        specs = self._registry.specs()
        # `wanted` is what gets embedded (name + domain + description, so a query
        # can match on any of them); `descriptions` is what a human reads. Storing
        # both keeps the index row honest without making the dashboard restate
        # the tool's own name back at itself.
        wanted = {spec.name: searchable_text(spec) for spec in specs}
        descriptions = {spec.name: spec.description for spec in specs}
        prints = {name: fingerprint(text) for name, text in wanted.items()}
        cached: dict[str, list[float]] = {}
        if self._sessions is not None:
            try:
                async with self._sessions() as session:
                    rows = (
                        (
                            await session.execute(
                                select(ToolIndex).where(ToolIndex.name.in_(list(wanted)))
                            )
                        )
                        .scalars()
                        .all()
                    )
                for row in rows:
                    if row.fingerprint == prints.get(row.name) and row.embedding is not None:
                        cached[row.name] = list(row.embedding)
            except Exception:
                logger.warning("tool embedding cache unavailable", exc_info=True)
        missing = [name for name in wanted if name not in cached]
        if missing:
            vectors = await self._llm.embed([wanted[name] for name in missing])
            fresh = {
                name: vector
                for name, vector in zip(missing, vectors, strict=True)
                if vector is not None
            }
            cached.update(fresh)
            if fresh and self._sessions is not None:
                await self._persist(fresh, descriptions, prints)
        self._vectors = cached

    async def _persist(
        self,
        fresh: dict[str, list[float]],
        descriptions: dict[str, str],
        prints: dict[str, str],
    ) -> None:
        try:
            async with self._sessions() as session, session.begin():  # type: ignore[misc]
                for name, vector in fresh.items():
                    stmt = insert(ToolIndex).values(
                        name=name,
                        fingerprint=prints[name],
                        description=descriptions[name],
                        embedding=vector,
                        updated_at=datetime.now(UTC),
                    )
                    await session.execute(
                        stmt.on_conflict_do_update(
                            index_elements=[ToolIndex.name],
                            set_={
                                "fingerprint": stmt.excluded.fingerprint,
                                "description": stmt.excluded.description,
                                "embedding": stmt.excluded.embedding,
                                "updated_at": stmt.excluded.updated_at,
                            },
                        )
                    )
        except Exception:
            logger.warning("could not cache tool embeddings", exc_info=True)

    async def rank(self, query: str, *, limit: int = 30) -> list[ScoredTool]:
        specs = {spec.name: spec for spec in self._registry.specs()}
        if not self._vectors or self._llm is None or not query.strip():
            return [
                ScoredTool(name=name, score=0.0, description=specs[name].description)
                for name in sorted(specs)
            ][:limit]
        vectors = await self._llm.embed([query[:4000]])
        probe = vectors[0] if vectors else None
        if probe is None:
            return [
                ScoredTool(name=name, score=0.0, description=specs[name].description)
                for name in sorted(specs)
            ][:limit]
        scored = [
            ScoredTool(
                name=name,
                score=_cosine(probe, vector),
                description=specs[name].description if name in specs else "",
            )
            for name, vector in self._vectors.items()
            if name in specs
        ]
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:limit]

    async def select(self, query: str, *, budget: int = DEFAULT_TOOL_BUDGET) -> tuple[str, ...]:
        """Choose the tool set a turn opens with."""

        available = set(self._registry.names)
        chosen: list[str] = [name for name in ALWAYS_AVAILABLE if name in available]
        for scored in await self.rank(query):
            if len(chosen) >= budget:
                break
            if scored.name not in chosen:
                chosen.append(scored.name)
        return tuple(chosen)


class ToolFinderTool:
    """Lets the model pull in a tool that was not on the opening roster."""

    def __init__(self, discovery: ToolDiscovery) -> None:
        self._discovery = discovery
        #: Names the model has discovered this run; the loop reads this and adds
        #: them to the schemas it advertises on the next turn.
        self.discovered: set[str] = set()

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="tools.find",
            description=(
                "Search Leo's full capability catalogue for a tool that is not currently "
                "in your tool list. Describe what you need to do ('historical crypto "
                "prices', 'SEC filings for a ticker', 'company earnings surprises') and "
                "you will get back matching tools. Anything returned becomes callable on "
                "your next turn."
            ),
            domain="meta",
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            max_result_bytes=6144,
            input_schema={
                "type": "object",
                "properties": {
                    "need": {
                        "type": "string",
                        "description": "What you are trying to do, in natural language.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
                },
                "required": ["need"],
            },
        )

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return dict(arguments)

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        need = str(arguments.get("need") or "").strip()
        if not need:
            return ToolFailure(
                code="missing_need", safe_message="Describe what capability you are looking for."
            )
        raw_limit = arguments.get("limit")
        limit = int(raw_limit) if isinstance(raw_limit, (int, float)) else 5
        matches = (await self._discovery.rank(need))[: max(1, min(10, limit))]
        self.discovered.update(match.name for match in matches)
        found: list[dict[str, Any]] = [
            {"name": match.name, "description": match.description} for match in matches
        ]
        return ToolSuccess(
            data={
                "need": need,
                "found": found,
                "note": "These tools are now callable. Call one directly on your next turn.",
            },
            source=SourceRef(provider="leo-tools", reference=f"find:{need[:60]}"),
            observed_at=datetime.now(UTC),
        )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
