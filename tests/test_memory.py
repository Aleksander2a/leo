"""Memory: what it recalls, and -- more importantly -- what it cannot.

Isolation is the property that matters most here. Every read is bounded by
``scope_key`` in SQL, so these tests assert that a DM's memories are absent from
a channel's result set rather than merely ranked below it.
"""

from __future__ import annotations

import uuid

import pytest

from leo.agent.contracts import Scope, ToolExecutionContext
from leo.agent.db import create_engine, create_sessions
from leo.agent.memory import MemoryService, build_memory_tools
from tests.conftest import database_url, requires_database

pytestmark = requires_database


class StubLLM:
    """Deterministic embeddings: same text, same vector; similar text, near vector."""

    model = "stub"
    embedding_model = "stub"

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [_vector(text) for text in texts]


def _vector(text: str) -> list[float]:
    from leo.agent.llm import EMBEDDING_DIMENSIONS

    vector = [0.0] * EMBEDDING_DIMENSIONS
    for token in text.lower().split():
        vector[hash(token) % EMBEDDING_DIMENSIONS] += 1.0
    if not any(vector):
        vector[0] = 1.0
    return vector


@pytest.fixture
async def service():  # type: ignore[no-untyped-def]
    engine = create_engine(database_url() or "")
    try:
        yield MemoryService(sessions=create_sessions(engine), llm=StubLLM())  # type: ignore[arg-type]
    finally:
        await engine.dispose()


def fresh(suffix: str) -> Scope:
    return Scope(key=f"test:{uuid.uuid4()}:{suffix}", actor_id="tester")


async def test_a_written_memory_comes_back(service) -> None:  # type: ignore[no-untyped-def]
    scope = fresh("dm")
    await service.write(scope, content="Holds 3 BTC.", kind="fact", subject="holdings")
    found = await service.list_all(scope)
    assert [item.content for item in found] == ["Holds 3 BTC."]
    assert found[0].kind == "fact"


async def test_recall_finds_a_memory_by_meaning(service) -> None:  # type: ignore[no-untyped-def]
    scope = fresh("dm")
    await service.write(scope, content="Never wants more than 15% drawdown.", subject="risk")
    recalled = await service.recall(scope, "drawdown tolerance")
    assert any("drawdown" in item.content for item in recalled)


async def test_a_channel_cannot_read_a_dms_memories(service) -> None:  # type: ignore[no-untyped-def]
    dm, channel = fresh("dm"), fresh("channel")
    await service.write(dm, content="Salary is 250k.", subject="compensation")
    assert await service.recall(channel, "salary") == []
    assert await service.list_all(channel) == []
    assert len(await service.list_all(dm)) == 1


async def test_an_update_supersedes_rather_than_overwrites(service) -> None:  # type: ignore[no-untyped-def]
    scope = fresh("dm")
    first = await service.write(scope, content="Holds 3 BTC.", subject="holdings")
    second = await service.write(
        scope, content="Holds 5 BTC.", subject="holdings", supersedes=first.id
    )
    active = await service.list_all(scope)
    assert [item.id for item in active] == [second.id]
    assert [item.content for item in active] == ["Holds 5 BTC."]


async def test_forgetting_is_scoped(service) -> None:  # type: ignore[no-untyped-def]
    mine, theirs = fresh("a"), fresh("b")
    stored = await service.write(mine, content="secret", subject="x")
    # The wrong scope cannot retire someone else's memory even knowing its id.
    assert await service.forget(theirs, stored.id) is False
    assert await service.forget(mine, stored.id) is True
    assert await service.list_all(mine) == []


async def test_empty_content_is_rejected(service) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        await service.write(fresh("dm"), content="   ")


async def test_importance_and_kind_are_clamped(service) -> None:  # type: ignore[no-untyped-def]
    scope = fresh("dm")
    stored = await service.write(scope, content="x", kind="not-a-kind", importance=99)
    assert stored.kind == "fact"
    assert stored.importance == 5


async def test_recall_without_embeddings_falls_back_to_recency(service) -> None:  # type: ignore[no-untyped-def]
    """An embedding outage must degrade the answer, not remove memory entirely."""

    scope = fresh("dm")
    await service.write(scope, content="older", importance=1)
    await service.write(scope, content="newer", importance=5)
    service._llm = None  # simulate the embedding provider being unavailable
    recalled = await service.recall(scope, "anything")
    assert next(item.content for item in recalled) == "newer"


async def test_the_memory_tools_only_touch_their_own_scope(service) -> None:  # type: ignore[no-untyped-def]
    dm, channel = fresh("dm"), fresh("channel")
    write, search = _tools(service, dm)
    context = ToolExecutionContext(trusted_scope=dm, run_id="run-1", tool_call_id="c1")
    stored = await write.execute(
        {"content": "Prefers weekly digests.", "kind": "preference"}, context
    )
    assert stored.data["stored"] is True

    hit = await search.execute({"query": "digest cadence"}, context)
    assert hit.data["count"] == 1

    _, other_search = _tools(service, channel)
    miss = await other_search.execute(
        {"query": "digest cadence"},
        ToolExecutionContext(trusted_scope=channel, run_id="run-2", tool_call_id="c2"),
    )
    assert miss.data["count"] == 0


async def test_forget_tool_reports_a_miss_instead_of_failing_silently(service) -> None:  # type: ignore[no-untyped-def]
    scope = fresh("dm")
    tools = build_memory_tools(service, scope, "run-1")
    forget = next(tool for tool in tools if tool.spec.name == "memory.forget")
    outcome = await forget.execute(
        {"id": "mem-does-not-exist"},
        ToolExecutionContext(trusted_scope=scope, run_id="run-1", tool_call_id="c1"),
    )
    assert outcome.kind == "failure"
    assert outcome.code == "not_found"


def _tools(service: MemoryService, scope: Scope):  # type: ignore[no-untyped-def]
    tools = build_memory_tools(service, scope, "run-1")
    write = next(tool for tool in tools if tool.spec.name == "memory.write")
    search = next(tool for tool in tools if tool.spec.name == "memory.search")
    return write, search
