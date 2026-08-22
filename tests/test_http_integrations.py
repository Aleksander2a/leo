from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from leo.harness.models import (
    ContextManifest,
    ContextSegment,
    ModelRequest,
    ModelTurnResult,
    RunPhase,
    ScopeKey,
    ToolArgumentConstraint,
    ToolChoiceMode,
    ToolChoicePolicy,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolRequests,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.integrations.fake import FixedClock, ScriptedQuoteModel
from leo.integrations.finnhub import FinnhubCompanyProfileTool, FinnhubQuoteTool
from leo.integrations.openrouter import OpenRouterError, OpenRouterGateway
from leo.integrations.provider_runtime import ProviderCallGate

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "openrouter"
PROVIDER_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "providers"


def _recorded_response(name: str) -> dict[str, object]:
    return json.loads((FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _provider_response(name: str) -> dict[str, object]:
    return json.loads((PROVIDER_FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _model_request(*, tool_choice: ToolChoicePolicy | None = None) -> ModelRequest:
    return ModelRequest(
        objective="Get NVDA quote",
        iteration=0,
        observations=(),
        verifier_feedback=(),
        tools=(
            ToolSpec(
                name="market.get_quote",
                description="Get quote",
                domain="MARKET",
                input_schema={
                    "type": "object",
                    "properties": {"symbol": {"type": "string"}},
                    "required": ["symbol"],
                },
                effect=ToolEffect.READ,
                allowed_phases=frozenset({RunPhase.RESEARCH}),
            ),
        ),
        tool_choice=tool_choice or ToolChoicePolicy(mode=ToolChoiceMode.AUTO),
        manifest=ContextManifest(
            segments=(ContextSegment(name="objective", priority=100, pinned=True),)
        ),
    )


@pytest.mark.asyncio
async def test_openrouter_gateway_returns_tool_request_without_executing_it() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["parallel_tool_calls"] is True
        assert payload["max_tokens"] == 2_000
        assert payload["tools"][0]["function"]["name"] == "market_get_quote"
        assert payload["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "model": "test/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-1",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_quote",
                                        "arguments": '{"symbol":"NVDA"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 3,
                    "total_tokens": 13,
                    "cost": 0.001,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(_model_request())

    assert isinstance(result, ModelTurnResult)
    assert isinstance(result.decision, ToolRequests)
    assert result.decision.calls[0].name == "market.get_quote"
    assert result.decision.calls[0].arguments == {"symbol": "NVDA"}
    assert result.provider == "openrouter"
    assert result.model == "test/model"
    assert result.request_id == "gen-1"
    assert result.finish_reason == "tool_calls"
    assert result.usage.total_tokens == 13
    assert result.usage.cost == 0.001


@pytest.mark.asyncio
async def test_openrouter_translates_required_named_tool_without_inferring_policy() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["parallel_tool_calls"] is False
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "market_get_quote"},
        }
        return httpx.Response(
            200,
            json={
                "id": "gen-required",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-required",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_quote",
                                        "arguments": '{"symbol":"NVDA"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    policy = ToolChoicePolicy(
        mode=ToolChoiceMode.REQUIRED,
        required_tool_name="market.get_quote",
        required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(_model_request(tool_choice=policy))

    assert isinstance(result.decision, ToolRequests)
    assert result.decision.calls[0].name == "market.get_quote"


@pytest.mark.asyncio
async def test_openrouter_gateway_parses_structured_completion() -> None:
    content = json.dumps(
        {
            "answer": "Answer",
            "source_claims": [
                {
                    "statement": "Answer",
                    "observation_ids": ["obs-1"],
                }
            ],
            "inferences": [],
            "affected_assumption": "Demand remains durable.",
            "uncertainty": "One evidence window is incomplete.",
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "gen-2",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(_model_request())

    assert result.decision.kind == "completion"
    assert result.decision.affected_assumption == "Demand remains durable."
    assert result.decision.uncertainty == "One evidence window is incomplete."
    assert result.finish_reason is None
    assert result.usage.prompt_tokens is None


@pytest.mark.asyncio
async def test_fake_and_recorded_openrouter_share_turn_result_contract() -> None:
    fake_result = await ScriptedQuoteModel().decide(_model_request())

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_recorded_response("tool_call"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        recorded_result = await gateway.decide(_model_request())

    for result in (fake_result, recorded_result):
        assert isinstance(result, ModelTurnResult)
        assert result.decision.kind == "tool_requests"
        assert result.provider
        assert result.model
        assert result.finish_reason == "tool_calls"

    assert fake_result.usage.prompt_tokens is None
    assert recorded_result.usage.prompt_tokens == 10


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture", "prompt_tokens", "completion_tokens", "total_tokens", "cost"),
    [
        ("missing_usage", None, None, None, None),
        ("partial_usage", 12, None, 12, None),
        ("completion", 11, 4, 15, 0.002),
    ],
)
async def test_recorded_usage_preserves_missing_metrics(
    fixture: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    total_tokens: int | None,
    cost: float | None,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(200, json=_recorded_response(fixture))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(_model_request())

    assert result.usage.prompt_tokens == prompt_tokens
    assert result.usage.completion_tokens == completion_tokens
    assert result.usage.total_tokens == total_tokens
    assert result.usage.cost == cost


@pytest.mark.asyncio
async def test_recorded_malformed_and_schema_drift_fail_safely() -> None:
    async def run_fixture(name: str, *, content: str | None = None) -> OpenRouterError:
        def handler(request: httpx.Request) -> httpx.Response:
            del request
            if content is not None:
                return httpx.Response(200, content=content)
            return httpx.Response(200, json=_recorded_response(name))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
            with pytest.raises(OpenRouterError) as captured:
                await gateway.decide(_model_request())
        return captured.value

    malformed = await run_fixture(
        "malformed", content=(FIXTURE_ROOT / "malformed.json").read_text(encoding="utf-8")
    )
    drift = await run_fixture("schema_drift")

    assert malformed.code == "malformed_response"
    assert drift.code == "empty_decision"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture", "status", "code"),
    [("rate_limit", 429, "http_429"), ("unsupported_parameter", 400, "http_400")],
)
async def test_recorded_provider_failures_have_stable_codes(
    fixture: str,
    status: int,
    code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(status, json=_recorded_response(fixture))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        with pytest.raises(OpenRouterError) as captured:
            await gateway.decide(_model_request())

    assert captured.value.code == code


@pytest.mark.asyncio
async def test_provider_timeout_maps_to_safe_transport_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("fixture timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        with pytest.raises(OpenRouterError) as captured:
            await gateway.decide(_model_request())

    assert captured.value.code == "transport_error"


@pytest.mark.asyncio
async def test_openrouter_marks_verifier_feedback_as_trusted_correction_guidance() -> None:
    feedback = "The answer price must copy 217.24 without rounding."
    content = json.dumps(
        {
            "answer": "NVDA is quoted at 217.24.",
            "source_claims": [
                {
                    "statement": "NVDA is quoted at 217.24.",
                    "observation_ids": ["obs-1"],
                }
            ],
            "inferences": [],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        assert "verifier_feedback is trusted correction guidance" in system
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["verifier_feedback"] == [feedback]
        schema = payload["response_format"]["json_schema"]["schema"]
        assert {"affected_assumption", "uncertainty"}.issubset(schema["required"])
        guidance = user_payload["completion_contract"]["guidance"]
        assert guidance in schema["properties"]["answer"]["description"]
        assert (
            guidance
            in schema["$defs"]["_SourceClaimPayload"]["properties"]["statement"]["description"]
        )
        return httpx.Response(
            200,
            json={
                "id": "gen-feedback",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    request = _model_request().model_copy(update={"verifier_feedback": (feedback,)})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(request)

    assert result.decision.kind == "completion"


@pytest.mark.asyncio
async def test_openrouter_rejects_source_claim_without_an_observation_id() -> None:
    content = json.dumps(
        {
            "answer": "Unsupported answer",
            "source_claims": [
                {
                    "statement": "Unsupported answer",
                    "observation_ids": [],
                }
            ],
            "inferences": [],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "id": "gen-invalid-source",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        with pytest.raises(OpenRouterError) as captured:
            await gateway.decide(_model_request())

    assert captured.value.code == "malformed_completion"


@pytest.mark.asyncio
async def test_openrouter_translates_explicit_none_with_advertised_tools() -> None:
    content = json.dumps(
        {
            "answer": "No tool call",
            "source_claims": [],
            "inferences": [{"statement": "No tool call", "observation_ids": []}],
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["tools"]
        assert payload["tool_choice"] == "none"
        schema = payload["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["source_claims"]["minItems"] == 0
        assert schema["properties"]["source_claims"]["maxItems"] >= 1
        assert schema["properties"]["inferences"]["minItems"] == 0
        assert schema["properties"]["inferences"]["maxItems"] >= 1
        return httpx.Response(
            200,
            json={
                "id": "gen-none",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    policy = ToolChoicePolicy(mode=ToolChoiceMode.NONE)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = OpenRouterGateway(client=client, api_key="test-key", model="test/model")
        result = await gateway.decide(_model_request(tool_choice=policy))

    assert result.decision.kind == "completion"


@pytest.mark.asyncio
async def test_finnhub_quote_normalizes_provider_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-Finnhub-Token"] == "test-key"
        assert request.url.params["symbol"] == "NVDA"
        return httpx.Response(
            200,
            json=_provider_response("finnhub_quote"),
            headers={"x-request-id": "finnhub-recorded-1"},
        )

    clock = FixedClock()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = FinnhubQuoteTool(client=client, api_key="test-key", clock=clock)
        result = await tool.execute(
            {"symbol": "NVDA"},
            ToolExecutionContext(
                trusted_scope=TrustedScope(
                    namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                    actor_id="user",
                ),
                run_id="run",
                tool_call_id="call",
            ),
        )

    assert isinstance(result, ToolSuccess)
    assert result.data["price"] == 181.25
    assert result.data["as_of"] == "2026-08-20T12:00:00+00:00"
    assert result.data["provider_request_id"] == "finnhub-recorded-1"
    assert result.source.provider == "finnhub"
    assert result.source.url is None
    assert result.expires_at == clock.now() + timedelta(minutes=15)
    clock.advance(seconds=600)
    assert result.expires_at > clock.now()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ({"c": float("nan")}, "FINNHUB_NON_FINITE_QUOTE"),
        ({"t": 1}, "FINNHUB_STALE_QUOTE"),
        ({"t": 1787227261}, "FINNHUB_FUTURE_QUOTE"),
        ({"c": 0}, "FINNHUB_EMPTY_QUOTE"),
    ],
)
async def test_finnhub_recorded_adversarial_payloads_fail_closed(
    mutation: dict[str, object],
    expected_code: str,
) -> None:
    payload = _provider_response("finnhub_quote")
    payload.update(mutation)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                content=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
        )
    ) as client:
        outcome = await FinnhubQuoteTool(
            client=client,
            api_key="test-key",
            clock=FixedClock(),
            max_quote_age_seconds=60,
            max_future_skew_seconds=60,
        ).execute(
            {"symbol": "NVDA"},
            ToolExecutionContext(
                trusted_scope=TrustedScope(
                    namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                    actor_id="user",
                ),
                run_id="run",
                tool_call_id="call",
            ),
        )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == expected_code


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_code", "retryable"),
    [
        (429, "FINNHUB_RATE_LIMITED", True),
        (503, "FINNHUB_UNAVAILABLE", True),
        (400, "FINNHUB_REQUEST_REJECTED", False),
    ],
)
async def test_finnhub_health_failures_are_typed(
    status: int,
    expected_code: str,
    retryable: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status))
    ) as client:
        outcome = await FinnhubQuoteTool(
            client=client,
            api_key="test-key",
            clock=FixedClock(),
        ).execute(
            {"symbol": "NVDA"},
            ToolExecutionContext(
                trusted_scope=TrustedScope(
                    namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                    actor_id="user",
                ),
                run_id="run",
                tool_call_id="call",
            ),
        )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == expected_code
    assert outcome.retryable is retryable


@pytest.mark.asyncio
async def test_finnhub_shared_gate_bounds_retry_after_and_tracks_all_endpoint_calls() -> None:
    calls = 0
    clock = FixedClock()
    gate = ProviderCallGate(
        provider="finnhub",
        clock=clock,
        max_concurrency=4,
        max_calls_per_minute=60,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "999999"})
        assert request.url.path == "/api/v1/stock/profile2"
        return httpx.Response(
            200,
            json={"ticker": "NVDA", "name": "NVIDIA", "exchange": "NASDAQ"},
        )

    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
            actor_id="user",
        ),
        run_id="run",
        tool_call_id="call",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        quote = FinnhubQuoteTool(client=client, api_key="key", clock=clock, gate=gate)
        profile = FinnhubCompanyProfileTool(
            client=client,
            api_key="key",
            clock=clock,
            gate=gate,
        )
        limited = await quote.execute({"symbol": "NVDA"}, context)
        clock.advance(seconds=299)
        cooling_down = await profile.execute({"symbol": "NVDA"}, context)
        clock.advance(seconds=2)
        recovered = await profile.execute({"symbol": "NVDA"}, context)

    health = await quote.provider_health()
    assert health == await profile.provider_health()
    assert isinstance(limited, ToolFailure) and limited.code == "FINNHUB_RATE_LIMITED"
    assert isinstance(cooling_down, ToolFailure)
    assert cooling_down.code == "FINNHUB_COOLDOWN_ACTIVE"
    assert isinstance(recovered, ToolSuccess)
    assert calls == 2
    assert health.calls_in_month == 2
    assert health.successes == 1
    assert health.failures == 1
    assert health.rate_limit_count == 1
    assert health.provider_credits_used == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_type", (FinnhubQuoteTool, FinnhubCompanyProfileTool))
async def test_finnhub_rejects_oversized_payload_before_json(
    tool_type: type[FinnhubQuoteTool] | type[FinnhubCompanyProfileTool],
) -> None:
    gate = ProviderCallGate(
        provider="finnhub",
        clock=FixedClock(),
        max_concurrency=4,
        max_calls_per_minute=60,
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=b"{" + (b"x" * 1_048_576),
                headers={"Content-Type": "application/json"},
            )
        )
    ) as client:
        tool = tool_type(client=client, api_key="key", clock=FixedClock(), gate=gate)
        outcome = await tool.execute(
            {"symbol": "NVDA"},
            ToolExecutionContext(
                trusted_scope=TrustedScope(
                    namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                    actor_id="user",
                ),
                run_id="run",
                tool_call_id="call",
            ),
        )

    health = await gate.snapshot()
    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "FINNHUB_RESPONSE_TOO_LARGE"
    assert health.calls_in_month == 1
    assert health.failures == 1
    assert health.provider_credits_used == 1


@pytest.mark.asyncio
async def test_finnhub_timeout_is_typed_retryable_failure() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("recorded timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        outcome = await FinnhubQuoteTool(
            client=client,
            api_key="test-key",
            clock=FixedClock(),
        ).execute(
            {"symbol": "NVDA"},
            ToolExecutionContext(
                trusted_scope=TrustedScope(
                    namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                    actor_id="user",
                ),
                run_id="run",
                tool_call_id="call",
            ),
        )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "FINNHUB_TIMEOUT"
    assert outcome.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    (
        "http://finnhub.io/api/v1",
        "https://attacker.example/api/v1",
        "https://finnhub.io:444/api/v1",
        "https://finnhub.io/not-api/v1",
        "https://finnhub.io/api/v1?next=https://attacker.example",
        "https://user:password@finnhub.io/api/v1",
    ),
)
async def test_finnhub_rejects_nonofficial_credential_origins(base_url: str) -> None:
    def unexpected_request(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("invalid Finnhub base URL reached the network")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected_request)) as client:
        for tool_type in (FinnhubQuoteTool, FinnhubCompanyProfileTool):
            with pytest.raises(ValueError, match="official credential-free HTTPS REST root"):
                tool_type(
                    client=client,
                    api_key="must-not-egress",
                    clock=FixedClock(),
                    base_url=base_url,
                )


@pytest.mark.asyncio
async def test_finnhub_tools_do_not_follow_redirects_from_credential_host() -> None:
    secret = "finnhub-secret-must-stay-on-official-host"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host != "finnhub.io":
            raise AssertionError("Finnhub credential-bearing request escaped its official host")
        assert request.headers["X-Finnhub-Token"] == secret
        return httpx.Response(
            302,
            headers={"location": "https://attacker.example/collect"},
        )

    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
            actor_id="user",
        ),
        run_id="run",
        tool_call_id="call",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        quote = await FinnhubQuoteTool(
            client=client,
            api_key=secret,
            clock=FixedClock(),
        ).execute({"symbol": "NVDA"}, context)
        profile = await FinnhubCompanyProfileTool(
            client=client,
            api_key=secret,
            clock=FixedClock(),
        ).execute({"symbol": "NVDA"}, context)

    assert isinstance(quote, ToolFailure) and quote.code == "FINNHUB_SCHEMA_DRIFT"
    assert isinstance(profile, ToolFailure) and profile.code == "FINNHUB_SCHEMA_DRIFT"
    assert [request.url.host for request in requests] == ["finnhub.io", "finnhub.io"]
    assert secret not in quote.model_dump_json()
    assert secret not in profile.model_dump_json()


@pytest.mark.asyncio
async def test_finnhub_bounds_provider_concurrency() -> None:
    active = 0
    maximum_active = 0

    async def handler(_: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        active -= 1
        return httpx.Response(200, json=_provider_response("finnhub_quote"))

    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
            actor_id="user",
        ),
        run_id="run",
        tool_call_id="call",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = FinnhubQuoteTool(
            client=client,
            api_key="test-key",
            clock=FixedClock(),
            max_concurrency=2,
        )
        outcomes = await asyncio.gather(
            *(tool.execute({"symbol": "NVDA"}, context) for _ in range(6))
        )

    assert all(isinstance(outcome, ToolSuccess) for outcome in outcomes)
    assert maximum_active == 2
