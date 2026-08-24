"""The dashboard API reads the agent's own tables directly."""

from __future__ import annotations

import uuid
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from leo.agent.contracts import Scope
from leo.agent.db import create_engine, create_sessions
from leo.agent.llm import Usage
from leo.agent.memory import MemoryService
from leo.agent.store import AgentStore
from leo.agent.tools import ToolResult
from leo.api.app import create_app
from leo.config import Settings
from tests.conftest import database_url, requires_database

pytestmark = requires_database


@pytest.fixture
def seeded():  # type: ignore[no-untyped-def]
    """A client on the real database, plus one recorded run and a revised memory."""

    engine = create_engine(database_url() or "")
    sessions = create_sessions(engine)
    store = AgentStore(sessions)
    scope = Scope(key=f"test:{uuid.uuid4()}:api", actor_id="tester")

    async def seed() -> tuple[str, str]:
        conversation = await store.ensure_conversation(scope, kind="dm", title="api test")
        await store.record_message(scope, conversation, role="user", content="hello")
        run_id = await store.start_run(scope, conversation, question="q", model="m")
        await store.record_model_step(
            run_id,
            seq=1,
            tool_names=["web.search_tavily"],
            content_preview="thinking",
            finish_reason="tool_calls",
            usage=Usage(5, 5, 10, 0.1),
            duration_ms=900,
        )
        await store.record_tool_step(
            run_id,
            seq=2,
            result=ToolResult(
                call_id="c1",
                name="web.search_tavily",
                arguments={"query": "x"},
                ok=True,
                payload={"data": {"ok": True}, "source": "tavily", "reference": "s:1"},
                duration_ms=12,
            ),
        )
        # A failed tool call, so the JSON error-code aggregations are exercised.
        await store.record_tool_step(
            run_id,
            seq=3,
            result=ToolResult(
                call_id="c2",
                name="market.get_quote",
                arguments={"symbol": "NOPE"},
                ok=False,
                payload={"error": "SYMBOL_NOT_FOUND", "message": "no such symbol"},
                duration_ms=30,
            ),
        )
        await store.finish_run(
            run_id,
            status="answered",
            answer="a",
            turns=1,
            tool_calls=2,
            usage=Usage(5, 5, 10, 0.1),
        )
        memory = MemoryService(sessions=sessions, llm=None)
        first = await memory.write(scope, content="Holds 3 BTC.", subject="holdings", run_id=run_id)
        second = await memory.write(
            scope,
            content="Holds 5 BTC.",
            subject="holdings",
            supersedes=first.id,
            run_id=run_id,
        )
        return run_id, second.id

    run_id, memory_id = _run(seed())
    with TestClient(create_app(Settings(), sessions=sessions)) as client:
        yield client, scope, run_id, memory_id
    _run(engine.dispose())


def _run(coroutine):  # type: ignore[no-untyped-def]
    from leo.agent.db import run

    return run(coroutine)


def test_health_reports_configuration(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    payload = client.get("/health").json()
    assert payload["configured"]["database"] is True
    assert payload["status"] in {"ok", "degraded"}


def test_deep_health_probes_the_database(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    assert client.get("/health", params={"deep": True}).json()["database_reachable"] is True


def test_overview_aggregates_runs_tools_and_failures(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    payload = client.get("/dashboard/overview").json()
    assert payload["total_runs"] >= 1
    assert payload["run_status_counts"].get("answered", 0) >= 1
    assert payload["answer_rate"] is not None
    assert any(item["name"] == "web.search_tavily" for item in payload["tool_usage"])
    # Grouping on a JSON key is easy to get subtly wrong; assert it comes back.
    assert any(item["code"] == "SYMBOL_NOT_FOUND" for item in payload["tool_errors"])
    assert isinstance(payload["activity"], list)
    assert payload["p50_run_seconds"] is not None


def test_runs_can_be_filtered_by_scope(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, run_id, _ = seeded
    payload = client.get("/dashboard/runs", params={"scope_key": scope.key}).json()
    assert payload["total"] == 1
    assert payload["items"][0]["id"] == run_id


def test_runs_can_be_searched_by_question_text(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    hit = client.get("/dashboard/runs", params={"scope_key": scope.key, "q": "q"}).json()
    miss = client.get("/dashboard/runs", params={"scope_key": scope.key, "q": "zzzz"}).json()
    assert hit["total"] == 1
    assert miss["total"] == 0


def test_a_run_detail_separates_model_turns_from_tool_calls(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, run_id, _ = seeded
    detail = client.get(f"/dashboard/runs/{run_id}").json()
    assert detail["answer"] == "a"
    steps = detail["steps"]
    assert [step["kind"] for step in steps] == ["model", "tool", "tool"]
    assert steps[0]["arguments"]["tools_offered"] == ["web.search_tavily"]
    assert steps[1]["arguments"] == {"query": "x"}
    assert steps[2]["ok"] is False
    assert detail["memories_written"] == 2


def test_an_unknown_run_is_a_404(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    assert client.get("/dashboard/runs/run-nope").status_code == 404


def test_failures_list_excludes_runs_that_answered(seeded) -> None:  # type: ignore[no-untyped-def]
    """A failed tool call is not a failure; only a run with no answer is."""

    client, _, _, _ = seeded
    payload = client.get("/dashboard/failures").json()
    assert all(item["status"] != "answered" for item in payload["items"])


def test_conversation_detail_bundles_transcript_runs_and_memory(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    payload = client.get(f"/dashboard/conversations/{quote(scope.key, safe='')}").json()
    assert payload["scope_key"] == scope.key
    assert [m["content"] for m in payload["recent_messages"]] == ["hello"]
    assert len(payload["recent_runs"]) == 1
    assert [m["content"] for m in payload["recent_memories"]] == ["Holds 5 BTC."]


def test_an_unknown_conversation_is_a_404(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    assert client.get("/dashboard/conversations/nope%3Anope").status_code == 404


def test_conversation_messages_are_scoped(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    path = f"/dashboard/conversations/{quote(scope.key, safe='')}/messages"
    payload = client.get(path).json()
    assert [item["content"] for item in payload["items"]] == ["hello"]


def test_memory_listing_is_scoped_and_hides_superseded_rows(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    payload = client.get("/dashboard/memory", params={"scope_key": scope.key}).json()
    assert [item["content"] for item in payload["items"]] == ["Holds 5 BTC."]

    with_history = client.get(
        "/dashboard/memory", params={"scope_key": scope.key, "include_inactive": True}
    ).json()
    assert with_history["total"] == 2


def test_memory_detail_exposes_the_supersession_chain(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, run_id, memory_id = seeded
    payload = client.get(f"/dashboard/memory/{memory_id}").json()
    assert payload["content"] == "Holds 5 BTC."
    assert [row["content"] for row in payload["supersedes"]] == ["Holds 3 BTC."]
    assert payload["source_run"]["id"] == run_id


def test_memory_can_be_searched_by_content(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    hit = client.get("/dashboard/memory", params={"scope_key": scope.key, "q": "BTC"}).json()
    miss = client.get("/dashboard/memory", params={"scope_key": scope.key, "q": "zzzz"}).json()
    assert hit["total"] == 1
    assert miss["total"] == 0


def test_memory_kinds_are_counted(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    payload = client.get("/dashboard/memory-kinds").json()
    assert any(item["kind"] == "fact" for item in payload["items"])


def test_tools_endpoint_reports_usage_and_failure_codes(seeded) -> None:  # type: ignore[no-untyped-def]
    client, _, _, _ = seeded
    items = client.get("/dashboard/tools").json()["items"]
    by_name = {item["name"]: item for item in items}
    assert by_name["web.search_tavily"]["calls"] >= 1
    failing = by_name["market.get_quote"]
    assert failing["failed"] >= 1
    assert any(error["code"] == "SYMBOL_NOT_FOUND" for error in failing["errors"])


def test_scopes_endpoint_lists_conversations_for_filters(seeded) -> None:  # type: ignore[no-untyped-def]
    client, scope, _, _ = seeded
    items = client.get("/dashboard/scopes").json()["items"]
    assert any(item["scope_key"] == scope.key for item in items)
