"""Tool discovery: ranked by meaning, never gated by it."""

from __future__ import annotations

import pytest

from leo.agent.contracts import ToolExecutionContext
from leo.agent.discovery import (
    ALWAYS_AVAILABLE,
    ToolDiscovery,
    ToolFinderTool,
    _cosine,
    searchable_text,
)
from leo.agent.tools import ToolRegistry
from tests.conftest import SCOPE, FakeTool


class VectorLLM:
    """Embeds by keyword overlap, which is enough to test the ranking wiring."""

    model = "stub"
    embedding_model = "stub"

    def __init__(self, vocabulary: list[str]) -> None:
        self._vocabulary = vocabulary

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [
            [1.0 if word in text.lower() else 0.0 for word in self._vocabulary] or [1.0]
            for text in texts
        ]


def registry_of(*named: tuple[str, str]) -> ToolRegistry:
    return ToolRegistry([FakeTool(name, description=description) for name, description in named])


def test_searchable_text_includes_name_domain_and_description() -> None:
    tool = FakeTool("market.get_quote")
    text = searchable_text(tool.spec)
    assert "market.get_quote" in text and "test" in text


def test_cosine_handles_degenerate_vectors() -> None:
    assert _cosine([], [1.0]) == 0.0
    assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_selection_ranks_the_relevant_tool_first() -> None:
    registry = registry_of(
        ("crypto.snapshot", "Live cryptocurrency prices for bitcoin and ethereum"),
        ("filings.recent", "Recent SEC filings and annual reports for a company"),
    )
    llm = VectorLLM(["cryptocurrency", "bitcoin", "filings", "sec"])
    discovery = ToolDiscovery(registry=registry, llm=llm, sessions=None)  # type: ignore[arg-type]
    await discovery.prepare()
    ranked = await discovery.rank("what is bitcoin doing")
    assert ranked[0].name == "crypto.snapshot"


@pytest.mark.asyncio
async def test_the_core_tools_are_always_offered() -> None:
    registry = registry_of(
        ("memory.search", "recall"),
        ("web.search_tavily", "search the web"),
        ("market.obscure", "an obscure provider"),
    )
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    selected = await discovery.select("anything at all", budget=2)
    assert "memory.search" in selected
    assert "web.search_tavily" in selected


@pytest.mark.asyncio
async def test_selection_without_embeddings_still_returns_tools() -> None:
    """No embedding provider must degrade the ranking, not empty the tool list."""

    registry = registry_of(("a.one", "first"), ("b.two", "second"))
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    await discovery.prepare()
    assert set(await discovery.select("q")) == {"a.one", "b.two"}


@pytest.mark.asyncio
async def test_find_reports_matches_and_marks_them_discovered() -> None:
    registry = registry_of(("market.obscure", "an obscure market data provider"))
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    finder = ToolFinderTool(discovery)
    outcome = await finder.execute(
        {"need": "market data"},
        ToolExecutionContext(trusted_scope=SCOPE, run_id="run-1", tool_call_id="c1"),
    )
    assert outcome.kind == "success"
    assert outcome.data["found"][0]["name"] == "market.obscure"
    assert finder.discovered == {"market.obscure"}


@pytest.mark.asyncio
async def test_find_requires_a_description_of_the_need() -> None:
    registry = registry_of(("a.one", "first"))
    finder = ToolFinderTool(ToolDiscovery(registry=registry, llm=None, sessions=None))
    outcome = await finder.execute(
        {"need": "  "},
        ToolExecutionContext(trusted_scope=SCOPE, run_id="run-1", tool_call_id="c1"),
    )
    assert outcome.kind == "failure"


def test_the_always_available_set_covers_memory_and_a_web_route() -> None:
    assert "memory.search" in ALWAYS_AVAILABLE
    assert "memory.write" in ALWAYS_AVAILABLE
    assert "tools.find" in ALWAYS_AVAILABLE
    assert any(name.startswith("web.") for name in ALWAYS_AVAILABLE)
