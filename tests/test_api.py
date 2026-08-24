"""The dashboard API reads the agent's own tables directly."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from leo.agent.contracts import Scope
from leo.agent.db import create_engine, create_sessions
from leo.agent.llm import Usage
from leo.agent.store import AgentStore
from leo.agent.tools import ToolResult
from leo.api.app import create_app
from leo.config import Settings
from tests.conftest import database_url, requires_database

pytestmark = requires_database


@pytest.fixture
def seeded():  # type: ignore[no-untyped-def]
    """A client backed by the real database, plus one recorded run to read."""

    engine = create_engine(database_url() or "")
    sessions = create_sessions(engine)
    store = AgentStore(sessions)
    scope = Scope(key=f"test:{uuid.uuid4()}:api", actor_id="tester")

    async def seed() -> str:
        conversation = await store.ensure_conversation(scope, kind="dm", title="api test")
        await store.record_message(scope, conversation, role="user", content="hello")
        run_id = await store.start_run(scope, conversation, question="q", model="m")
        await store.record_tool_step(
            run_id,
            seq=1,
            result=ToolResult(
                call_id="c1",
                name="web.search_tavily",
                arguments={"query": "x"},
                ok=True,
                payload={"data": {"ok": True}},
                duration_ms=12,
            ),
        )
        await store.finish_run(
            run_id, status="answered", answer="a", turns=1, tool_calls=1, usage=Usage(5, 5, 10, 0.1)
        )
        return run_id

    run_id = _run(seed())
    with TestClient(create_app(Settings(), sessions=sessions)) as client:
        yield client, scope, run_id
    _run(engine.dispose())


def _run(coroutine):  # type: ignore[no-untyped-def]
    from leo.agent.db import run

    return run(coroutine)


def test_health_reports_configuration(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = seeded
    payload = client.get("/health").json()
    assert payload["configured"]["database"] is True
    assert payload["status"] in {"ok", "degraded"}


def test_overview_aggregates_runs_and_tools(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = seeded
    payload = client.get("/dashboard/overview").json()
    assert payload["total_runs"] >= 1
    assert payload["run_status_counts"].get("answered", 0) >= 1
    assert any(item["name"] == "web.search_tavily" for item in payload["tool_usage"])


def test_runs_can_be_filtered_by_scope(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, run_id = seeded
    payload = client.get("/dashboard/runs", params={"scope_key": scope.key}).json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == run_id


def test_a_run_detail_includes_its_step_trace(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, run_id = seeded
    payload = client.get(f"/dashboard/runs/{run_id}").json()
    assert payload["answer"] == "a"
    assert payload["steps"][0]["name"] == "web.search_tavily"
    assert payload["steps"][0]["arguments"] == {"query": "x"}


def test_an_unknown_run_is_a_404(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = seeded
    assert client.get("/dashboard/runs/run-nope").status_code == 404


def test_conversation_messages_are_scoped(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _ = seeded
    payload = client.get(f"/dashboard/conversations/{scope.key}/messages").json()
    assert [item["content"] for item in payload["items"]] == ["hello"]


def test_memory_listing_accepts_a_scope_filter(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _ = seeded
    payload = client.get("/dashboard/memory", params={"scope_key": scope.key}).json()
    assert payload["items"] == []


def test_tools_endpoint_lists_the_index(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _ = seeded
    payload = client.get("/dashboard/tools").json()
    assert "items" in payload
