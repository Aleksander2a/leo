"""The registry's contract: every outcome returns to the model as a message.

These tests exist because the previous runtime could turn a tool problem into a
dead run. Each case below asserts the opposite: the call comes back describing
what went wrong, and the loop is free to continue.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.agent.contracts import SourceRef, ToolSuccess
from leo.agent.tools import ToolRegistry, function_schema, shrink
from tests.conftest import SCOPE, FakeTool, failure


async def _run(registry: ToolRegistry, name: str, arguments: dict, **kwargs):
    return await registry.execute(
        call_id="c1", name=name, arguments=arguments, scope=SCOPE, run_id="run-1", **kwargs
    )


@pytest.mark.asyncio
async def test_successful_call_returns_source_and_data() -> None:
    registry = ToolRegistry([FakeTool()])
    result = await _run(registry, "test.echo", {"value": "hello"})
    assert result.ok
    assert result.payload["data"] == {"echo": "hello"}
    assert result.payload["source"] == "test"
    assert json.loads(result.as_message()["content"])["data"]["echo"] == "hello"


@pytest.mark.asyncio
async def test_unknown_tool_is_reported_not_raised() -> None:
    registry = ToolRegistry([FakeTool("market.get_quote")])
    result = await _run(registry, "market.get_quotes", {})
    assert not result.ok
    assert result.payload["error"] == "unknown_tool"
    # The model is told what it *could* have called, so the next turn can recover.
    assert "market.get_quote" in result.payload["available_tools"]


@pytest.mark.asyncio
async def test_invalid_arguments_return_the_schema() -> None:
    registry = ToolRegistry([FakeTool()])
    result = await _run(registry, "test.echo", {"wrong": 1})
    assert not result.ok
    assert result.payload["error"] == "invalid_arguments"
    assert result.payload["expected_schema"]["properties"]["value"]["type"] == "string"


@pytest.mark.asyncio
async def test_unparseable_arguments_become_a_correctable_message() -> None:
    registry = ToolRegistry([FakeTool()])
    result = await _run(registry, "test.echo", {}, parse_error="not JSON")
    assert not result.ok
    assert result.payload["message"] == "not JSON"


@pytest.mark.asyncio
async def test_tool_failure_carries_the_provider_code() -> None:
    registry = ToolRegistry([FakeTool(outcome=failure("RATE_LIMITED", "slow down"))])
    result = await _run(registry, "test.echo", {"value": "x"})
    assert not result.ok
    assert result.payload == {
        "error": "RATE_LIMITED",
        "message": "slow down",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_a_crashing_adapter_does_not_escape() -> None:
    registry = ToolRegistry([FakeTool(raises=RuntimeError("boom"))])
    result = await _run(registry, "test.echo", {"value": "x"})
    assert not result.ok
    assert result.payload["error"] == "tool_crashed"
    assert "RuntimeError" in result.payload["message"]


@pytest.mark.asyncio
async def test_a_hanging_tool_times_out_into_a_message() -> None:
    registry = ToolRegistry([FakeTool(delay=1.0, timeout_seconds=0.05)])
    result = await _run(registry, "test.echo", {"value": "x"})
    assert not result.ok
    assert result.payload["error"] == "timeout"


@pytest.mark.asyncio
async def test_cancellation_still_propagates() -> None:
    """Timeouts are handled; a real cancellation must not be swallowed."""

    registry = ToolRegistry([FakeTool(delay=5.0, timeout_seconds=30.0)])
    task = asyncio.create_task(_run(registry, "test.echo", {"value": "x"}))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_oversize_results_are_truncated_not_discarded() -> None:
    big = ToolSuccess(
        data={"body": "x" * 50_000},
        source=SourceRef(provider="test", reference="big"),
        observed_at=datetime.now(UTC),
    )
    registry = ToolRegistry([FakeTool(outcome=big, max_result_bytes=2048)])
    result = await _run(registry, "test.echo", {"value": "x"})
    assert result.ok
    body = result.payload["data"]["body"]
    assert len(body) < 50_000
    assert body.startswith("xxx")


def test_shrink_keeps_leading_text_and_halves_lists() -> None:
    shrunk = shrink({"text": "a" * 5000, "items": list(range(500))}, 512)
    assert len(json.dumps(shrunk).encode()) <= 512
    assert shrunk["text"].startswith("aaa")


def test_shrink_handles_non_json_values() -> None:
    shrunk = shrink({"when": datetime.now(UTC), "nan": float("nan")}, 4096)
    assert isinstance(shrunk["when"], str)
    assert shrunk["nan"] is None


def test_duplicate_tool_names_do_not_shadow_each_other() -> None:
    registry = ToolRegistry([FakeTool("dup"), FakeTool("dup")])
    assert registry.names == ("dup",)


def test_function_schema_is_always_an_object() -> None:
    tool = FakeTool(schema={"type": "string"})
    schema = function_schema(tool.spec)
    assert schema["function"]["parameters"]["type"] == "object"
    assert schema["function"]["name"] == "test.echo"


@pytest.mark.asyncio
async def test_result_shape_is_stable_for_a_pydantic_validation_error() -> None:
    class Strict(FakeTool):
        def validate(self, arguments):  # type: ignore[no-untyped-def]
            from pydantic import BaseModel

            class Model(BaseModel):
                value: int

            try:
                return Model.model_validate(arguments).model_dump()
            except ValidationError as exc:
                raise exc

    registry = ToolRegistry([Strict()])
    result = await _run(registry, "test.echo", {"value": "nope"})
    assert not result.ok
    assert "value" in result.payload["message"]
