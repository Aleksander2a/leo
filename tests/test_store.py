"""Durable conversation state: history, idempotent ingest, run and step records."""

from __future__ import annotations

import uuid

import pytest

from leo.agent.contracts import Scope
from leo.agent.db import create_engine, create_sessions
from leo.agent.llm import Usage
from leo.agent.store import AgentStore
from leo.agent.tools import ToolResult
from tests.conftest import database_url, requires_database

pytestmark = requires_database


@pytest.fixture
async def store():  # type: ignore[no-untyped-def]
    engine = create_engine(database_url() or "")
    try:
        yield AgentStore(create_sessions(engine))
    finally:
        await engine.dispose()


def fresh(suffix: str = "chan") -> Scope:
    return Scope(key=f"test:{uuid.uuid4()}:{suffix}", actor_id="tester")


async def test_ensuring_a_conversation_twice_returns_the_same_row(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    first = await store.ensure_conversation(scope, kind="dm")
    second = await store.ensure_conversation(scope, kind="dm")
    assert first == second


async def test_history_comes_back_oldest_first(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    await store.record_message(scope, conversation, role="user", content="first")
    await store.record_message(scope, conversation, role="assistant", content="second")
    history = await store.history(scope)
    assert [turn.content for turn in history] == ["first", "second"]


async def test_history_never_crosses_scopes(store) -> None:  # type: ignore[no-untyped-def]
    mine, theirs = fresh("a"), fresh("b")
    conversation = await store.ensure_conversation(mine)
    await store.ensure_conversation(theirs)
    await store.record_message(mine, conversation, role="user", content="private")
    assert await store.history(theirs) == []


async def test_a_redelivered_slack_event_is_stored_once(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    for _ in range(3):
        await store.record_message(
            scope, conversation, role="user", content="hello", external_id="evt-1"
        )
    assert len(await store.history(scope)) == 1


async def test_the_current_message_can_be_excluded_from_its_own_history(store) -> None:  # type: ignore[no-untyped-def]
    """The question is passed to the model separately; it must not appear twice."""

    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    await store.record_message(scope, conversation, role="user", content="older")
    await store.record_message(
        scope, conversation, role="user", content="current", external_id="evt-9"
    )
    history = await store.history(scope, exclude_external_id="evt-9")
    assert [turn.content for turn in history] == ["older"]


async def test_a_thread_reads_only_its_own_messages(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    await store.record_message(
        scope, conversation, role="user", content="in thread A", thread_key="A"
    )
    await store.record_message(
        scope, conversation, role="user", content="in thread B", thread_key="B"
    )
    history = await store.history(scope, thread_key="A")
    assert [turn.content for turn in history] == ["in thread A"]


async def test_history_respects_its_character_budget(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    for index in range(6):
        await store.record_message(scope, conversation, role="user", content="x" * 400 + str(index))
    history = await store.history(scope, char_budget=900)
    assert 1 <= len(history) <= 3


async def test_empty_messages_are_skipped(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    await store.record_message(scope, conversation, role="assistant", content="   ")
    await store.record_message(scope, conversation, role="user", content="real")
    assert [turn.content for turn in await store.history(scope)] == ["real"]


async def test_a_run_records_its_outcome_and_usage(store) -> None:  # type: ignore[no-untyped-def]
    scope = fresh()
    conversation = await store.ensure_conversation(scope)
    run_id = await store.start_run(scope, conversation, question="q", model="m")
    await store.finish_run(
        run_id,
        status="answered",
        answer="a",
        turns=2,
        tool_calls=3,
        usage=Usage(prompt_tokens=100, completion_tokens=50, cost=0.02),
    )
    from sqlalchemy import select

    from leo.agent.schema import Run

    async with store._sessions() as session:
        run = (await session.execute(select(Run).where(Run.id == run_id))).scalar_one()
    assert (run.status, run.answer, run.turns, run.tool_calls) == ("answered", "a", 2, 3)
    assert run.prompt_tokens == 100
    assert run.finished_at is not None


async def test_a_failing_trace_write_never_breaks_the_run(store) -> None:  # type: ignore[no-untyped-def]
    """The trace is diagnostics; losing it must not lose the answer."""

    result = ToolResult(call_id="c1", name="t", arguments={}, ok=True, payload={}, duration_ms=1)
    # No such run id, so the foreign key rejects the step -- and it is swallowed.
    await store.record_tool_step("run-does-not-exist", seq=1, result=result)
