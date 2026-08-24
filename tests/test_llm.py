"""Provider-transport behaviour: parse what arrives, recover what can be recovered."""

from __future__ import annotations

import httpx
import pytest

from leo.agent.contracts import ProviderError
from leo.agent.llm import LLM, Usage, _parse_completion, _repair_json


def build(handler) -> LLM:  # type: ignore[no-untyped-def]
    transport = httpx.MockTransport(handler)
    return LLM(
        client=httpx.AsyncClient(transport=transport),
        api_key="test-key",
        model="test/model",
        max_attempts=3,
    )


def body(**message) -> dict:  # type: ignore[no-untyped-def]
    return {
        "id": "gen-1",
        "model": "test/model",
        "choices": [{"message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10, "cost": 0.01},
    }


def test_plain_content_is_parsed_with_usage() -> None:
    completion = _parse_completion(body(role="assistant", content="  hello  "))
    assert completion.content == "hello"
    assert completion.usage == Usage(7, 3, 10, 0.01)
    assert not completion.wants_tools


def test_tool_calls_are_parsed_from_a_json_string() -> None:
    completion = _parse_completion(
        body(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "c1",
                    "function": {"name": "market.get_quote", "arguments": '{"symbol": "NVDA"}'},
                }
            ],
        )
    )
    assert completion.wants_tools
    assert completion.tool_calls[0].name == "market.get_quote"
    assert completion.tool_calls[0].arguments == {"symbol": "NVDA"}


def test_arguments_already_decoded_are_accepted() -> None:
    completion = _parse_completion(
        body(
            role="assistant",
            tool_calls=[{"id": "c1", "function": {"name": "t", "arguments": {"a": 1}}}],
        )
    )
    assert completion.tool_calls[0].arguments == {"a": 1}


def test_unrecoverable_arguments_become_a_parse_error_not_an_exception() -> None:
    completion = _parse_completion(
        body(
            role="assistant",
            tool_calls=[{"id": "c1", "function": {"name": "t", "arguments": "[1, 2, 3]"}}],
        )
    )
    call = completion.tool_calls[0]
    assert call.arguments == {}
    assert call.parse_error is not None


def test_content_and_tool_calls_can_coexist() -> None:
    completion = _parse_completion(
        body(
            role="assistant",
            content="Let me check.",
            tool_calls=[{"id": "c1", "function": {"name": "t", "arguments": "{}"}}],
        )
    )
    assert completion.content == "Let me check."
    assert completion.wants_tools


def test_a_choiceless_body_is_a_provider_error() -> None:
    with pytest.raises(ProviderError):
        _parse_completion({"choices": []})


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"symbol": "NVDA"}\n```',
        'here you go: {"symbol": "NVDA"}',
        "{'symbol': 'NVDA'}",
    ],
)
def test_repair_recovers_the_common_argument_mangles(raw: str) -> None:
    assert _repair_json(raw) == {"symbol": "NVDA"}


def test_repair_gives_up_cleanly() -> None:
    assert _repair_json("not json at all") is None


@pytest.mark.asyncio
async def test_a_rate_limit_is_retried_then_succeeds() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(200, json=body(role="assistant", content="ok"))

    completion = await build(handler).complete([{"role": "user", "content": "hi"}])
    assert completion.content == "ok"
    assert attempts["n"] == 2


@pytest.mark.asyncio
async def test_a_non_retryable_status_raises_immediately() -> None:
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        return httpx.Response(401, text="bad key")

    with pytest.raises(ProviderError) as excinfo:
        await build(handler).complete([{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "http_401"
    assert attempts["n"] == 1


@pytest.mark.asyncio
async def test_an_error_object_inside_a_200_is_still_an_error() -> None:
    """OpenRouter returns HTTP 200 with an error body; that must not parse as an answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"error": {"code": "invalid_model", "message": "nope"}})

    with pytest.raises(ProviderError) as excinfo:
        await build(handler).complete([{"role": "user", "content": "hi"}])
    assert excinfo.value.code == "invalid_model"


@pytest.mark.asyncio
async def test_tools_are_sent_only_when_supplied() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.append(json.loads(request.content))
        return httpx.Response(200, json=body(role="assistant", content="ok"))

    llm = build(handler)
    await llm.complete([{"role": "user", "content": "hi"}])
    await llm.complete(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "t", "parameters": {}}}],
    )
    assert "tools" not in seen[0]
    assert seen[1]["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_an_embedding_outage_degrades_to_none_rather_than_raising() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    assert await build(handler).embed(["a", "b"]) == [None, None]


@pytest.mark.asyncio
async def test_embeddings_are_returned_in_request_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"data": [{"index": 1, "embedding": [0.2]}, {"index": 0, "embedding": [0.1]}]},
        )

    assert await build(handler).embed(["a", "b"]) == [[0.1], [0.2]]
