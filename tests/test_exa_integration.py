from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime

import httpx
import pytest

from leo.config import Settings
from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EvidenceQuality,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
    VerifierStatus,
)
from leo.harness.normalization import normalize_success
from leo.harness.exa_search import normalize_complete_exa_result
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.exa import ExaSearchTool
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.tavily import TavilySearchTool
from leo.integrations.verified_web import VerifiedWebResearchTool
from leo.integrations.web_fetch import PublicTextFetchTool
from leo.live import run_live_conversation

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
RESULT_URL = "https://docs.python.org/3/whatsnew/3.14.html"
TITLE = "What's New In Python 3.14"
HIGHLIGHT = "Python 3.14 adds deferred evaluation of annotations."
STATEMENT = f'Exa highlight from "{TITLE}" ({RESULT_URL}): {HIGHLIGHT}'


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="run",
        tool_call_id="call",
    )


def _observation(
    success: ToolSuccess,
    *,
    kind: str = "web.search_exa",
) -> Observation:
    return normalize_success(
        success,
        observation_id="obs",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call",
        observation_kind=kind,
    )


def _bundle(observation: Observation) -> RunBundle:
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective="Research Python 3.14")
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=(observation,))


def _verify(
    observation: Observation,
    statement: str,
    *,
    answer: str | None = None,
) -> VerifierStatus:
    proposal = CompletionProposal(
        answer=answer or statement,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(observation.id,),
            ),
        ),
    )
    return (
        DeterministicCompletionVerifier(
            SequentialIdGenerator(),
            FixedClock(NOW),
            require_source_claim=True,
        )
        .verify(proposal, _bundle(observation))
        .result.status
    )


def _complete_result() -> dict[str, object]:
    return {
        "title": TITLE,
        "url": RESULT_URL,
        "id": RESULT_URL,
        "highlights": [HIGHLIGHT],
        "highlightScores": [0.91],
    }


def test_exa_highlight_normalization_is_idempotent_at_text_limit() -> None:
    result = normalize_complete_exa_result(
        {
            "title": TITLE,
            "url": RESULT_URL,
            "highlights": ["word " * 300],
        }
    )

    assert result is not None
    assert normalize_complete_exa_result(result) == result


def _public_response(status_code: int, **kwargs: object) -> httpx.Response:
    response = httpx.Response(status_code, **kwargs)  # type: ignore[arg-type]
    response.extensions["leo_peer_ip"] = "93.184.216.34"
    return response


@pytest.mark.asyncio
async def test_exa_uses_exact_raw_search_shape_and_url_bound_highlight_grounding() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://api.exa.ai/search"
        assert request.headers["x-api-key"] == "exa-test-key"
        assert json.loads(request.content) == {
            "query": "Python 3.14 noteworthy change",
            "type": "auto",
            "contents": {"highlights": True},
        }
        return httpx.Response(
            200,
            json={
                "requestId": "exa-request-1",
                "results": [
                    {
                        "title": "Malformed first result",
                        "url": "https://example.org/incomplete",
                        "highlights": ["Missing structural score."],
                        "highlightScores": ["not-a-number"],
                    },
                    _complete_result(),
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = ExaSearchTool(client=client, api_key="exa-test-key", clock=FixedClock(NOW))
        outcome = await tool.execute({"query": "Python 3.14 noteworthy change"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.source.provider == "exa"
    assert outcome.source.url == RESULT_URL
    assert outcome.data["selected_result_rank"] == 2
    assert outcome.data["provider_request_id"] == "exa-request-1"
    assert outcome.data["statements"] == [STATEMENT]
    observation = _observation(outcome)
    assert observation.quality is EvidenceQuality.UNTRUSTED_RETRIEVAL
    assert _verify(observation, STATEMENT) is VerifierStatus.PASS
    assert _verify(observation, "Python 3.14 changed annotation evaluation.") is VerifierStatus.FAIL

    forged = observation.model_copy(
        update={
            "source": SourceRef(
                provider="exa",
                reference=observation.source.reference,
                url="https://attacker.example/forged",
            )
        }
    )
    assert _verify(forged, STATEMENT) is VerifierStatus.FAIL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_result", "expected_title", "expected_scores", "missing_fields", "statement"),
    [
        (
            {
                "url": RESULT_URL,
                "highlights": [HIGHLIGHT],
                "highlightScores": [0.91],
            },
            None,
            [0.91],
            ["title"],
            f'Exa highlight from "Untitled Exa result" ({RESULT_URL}): {HIGHLIGHT}',
        ),
        (
            {"title": TITLE, "url": RESULT_URL, "highlights": [HIGHLIGHT]},
            TITLE,
            None,
            ["highlight_scores"],
            STATEMENT,
        ),
        (
            {"url": RESULT_URL, "highlights": [HIGHLIGHT]},
            None,
            None,
            ["title", "highlight_scores"],
            f'Exa highlight from "Untitled Exa result" ({RESULT_URL}): {HIGHLIGHT}',
        ),
    ],
)
async def test_exa_accepts_missing_optional_metadata_with_deterministic_grounding(
    raw_result: dict[str, object],
    expected_title: str | None,
    expected_scores: list[float] | None,
    missing_fields: list[str],
    statement: str,
) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [raw_result]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await ExaSearchTool(
            client=client,
            api_key="exa-key",
            clock=FixedClock(NOW),
        ).execute({"query": "optional Exa metadata"}, _context())

    assert isinstance(outcome, ToolSuccess)
    result = outcome.data["result"]
    assert isinstance(result, dict)
    assert result["title"] == expected_title
    assert result["highlight_scores"] == expected_scores
    assert result["missing_fields"] == missing_fields
    expected_hash = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert outcome.data["result_hash"] == expected_hash
    assert outcome.source.reference.endswith(expected_hash)
    assert outcome.data["statements"] == [statement]

    observation = _observation(outcome)
    assert _verify(observation, statement) is VerifierStatus.PASS

    tampered_result = dict(result)
    tampered_result["missing_fields"] = []
    tampered_data = dict(observation.data)
    tampered_data["result"] = tampered_result
    tampered = observation.model_copy(update={"data": tampered_data})
    assert _verify(tampered, statement) is VerifierStatus.FAIL


@pytest.mark.asyncio
async def test_exa_reconstructed_tools_share_registry_quota_across_turns() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"results": [_complete_result()]})

    clock = FixedClock(NOW)
    registry = ProviderGateRegistry(clock)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = ExaSearchTool(
            client=client,
            api_key="exa-key",
            clock=clock,
            max_calls_per_minute=1,
            gate=registry.get(
                provider="exa",
                max_concurrency=4,
                max_calls_per_minute=1,
            ),
        )
        first_outcome = await first.execute({"query": "first turn query"}, _context())
        reconstructed = ExaSearchTool(
            client=client,
            api_key="exa-key",
            clock=clock,
            max_calls_per_minute=1,
            gate=registry.get(
                provider="exa",
                max_concurrency=4,
                max_calls_per_minute=1,
            ),
        )
        second_outcome = await reconstructed.execute(
            {"query": "second turn query"},
            _context(),
        )

    assert isinstance(first_outcome, ToolSuccess)
    assert isinstance(second_outcome, ToolFailure)
    assert second_outcome.code == "EXA_LOCAL_RATE_LIMIT"
    assert calls == 1


@pytest.mark.asyncio
async def test_exa_empty_or_private_highlights_fail_closed() -> None:
    responses = iter(
        (
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Private",
                            "url": "https://127.0.0.1/secret",
                            "highlights": ["private"],
                            "highlightScores": [1.0],
                        },
                        {
                            "title": "Incomplete",
                            "url": "https://example.org/incomplete",
                            "highlights": [],
                        },
                    ]
                },
            ),
            httpx.Response(200, json={"results": []}),
            httpx.Response(200, json={"results": "not-a-list"}),
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    ) as client:
        tool = ExaSearchTool(client=client, api_key="key", clock=FixedClock(NOW))
        incomplete = await tool.execute({"query": "incomplete highlights"}, _context())
        empty = await tool.execute({"query": "no results"}, _context())
        malformed = await tool.execute({"query": "bad schema"}, _context())

    assert isinstance(incomplete, ToolFailure)
    assert incomplete.code == "EXA_NO_COMPLETE_HIGHLIGHTS"
    assert isinstance(empty, ToolFailure) and empty.code == "EXA_NO_RESULTS"
    assert isinstance(malformed, ToolFailure) and malformed.code == "EXA_SCHEMA_DRIFT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_code", "retryable"),
    [
        (400, "EXA_REQUEST_REJECTED", False),
        (401, "EXA_AUTH_REJECTED", False),
        (402, "EXA_CREDITS_EXHAUSTED", False),
        (403, "EXA_ACCESS_DENIED", False),
        (422, "EXA_REQUEST_REJECTED", False),
        (429, "EXA_RATE_LIMITED", True),
        (500, "EXA_UNAVAILABLE", True),
        (503, "EXA_UNAVAILABLE", True),
    ],
)
async def test_exa_http_failures_are_safe_and_typed(
    status_code: int,
    expected_code: str,
    retryable: bool,
) -> None:
    secret = "exa-live-secret-never-persist"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == secret
        return httpx.Response(status_code, json={"error": f"provider echoed {secret}"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = ExaSearchTool(client=client, api_key=secret, clock=FixedClock(NOW))
        outcome = await tool.execute({"query": "safe failure"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == expected_code
    assert outcome.retryable is retryable
    assert secret not in outcome.model_dump_json()
    assert secret not in repr(tool)


@pytest.mark.asyncio
async def test_exa_transport_failure_does_not_leak_provider_or_api_secret() -> None:
    api_secret = "exa-api-secret-never-print"
    provider_secret = "provider-transport-secret-never-print"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout(provider_secret, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await ExaSearchTool(
            client=client,
            api_key=api_secret,
            clock=FixedClock(NOW),
        ).execute({"query": "timeout"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "EXA_TIMEOUT"
    assert outcome.retryable
    encoded = outcome.model_dump_json()
    assert api_secret not in encoded
    assert provider_secret not in encoded


@pytest.mark.asyncio
async def test_exa_does_not_follow_redirects_from_credential_host() -> None:
    secret = "exa-secret-must-stay-on-official-host"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host != "api.exa.ai":
            raise AssertionError("Exa credential-bearing request escaped its official host")
        assert request.headers["x-api-key"] == secret
        return httpx.Response(
            302,
            headers={"location": "https://attacker.example/collect"},
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        outcome = await ExaSearchTool(
            client=client,
            api_key=secret,
            clock=FixedClock(NOW),
        ).execute({"query": "redirect boundary"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "EXA_SCHEMA_DRIFT"
    assert [request.url.host for request in requests] == ["api.exa.ai"]
    assert secret not in outcome.model_dump_json()


@pytest.mark.asyncio
async def test_verified_web_family_recovers_exa_failure_and_candidate_local_fetch_failure() -> None:
    statement = "A complete alternate source says the noteworthy change is deferred annotations."

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.exa.ai":
            return httpx.Response(402, json={"error": "no credits"})
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "request_id": "tavily-fallback-1",
                    "results": [
                        {
                            "title": "Video result",
                            "url": "https://www.youtube.com/watch?v=bad",
                            "content": "Discovery only.",
                            "score": 1.0,
                        },
                        {
                            "title": "Empty first article",
                            "url": "https://93.184.216.34/empty",
                            "content": "Discovery only and not claim eligible.",
                            "score": 0.99,
                        },
                        {
                            "title": "Complete alternate article",
                            "url": "https://93.184.216.34/complete",
                            "content": "Discovery only and not claim eligible.",
                            "score": 0.95,
                        },
                    ],
                },
            )
        if request.url.path == "/empty":
            return _public_response(
                200,
                headers={"content-type": "text/html"},
                text="<html><script>no retained text</script></html>",
            )
        assert request.url.path == "/complete"
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text=statement,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        exa = ExaSearchTool(client=client, api_key="exa-key", clock=FixedClock(NOW))
        tavily = TavilySearchTool(client=client, api_key="tavily-key", clock=FixedClock(NOW))
        fetch = PublicTextFetchTool(client=client, clock=FixedClock(NOW))
        outcome = await VerifiedWebResearchTool(
            exa=exa,
            tavily=tavily,
            fetch=fetch,
        ).execute({"query": "noteworthy language change"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.source.provider == "public-web"
    assert outcome.source.url == "https://93.184.216.34/complete"
    assert outcome.data["selected_provider"] == "tavily_public_fetch"
    assert outcome.data["provider_attempts"] == [
        {
            "provider": "exa",
            "stage": "search",
            "status": "failed",
            "code": "EXA_CREDITS_EXHAUSTED",
        },
        {"provider": "tavily", "stage": "search", "status": "succeeded"},
        {"provider": "public-web", "stage": "fetch", "status": "succeeded"},
    ]
    assert outcome.data["failed_candidates"] == [
        {"url": "https://93.184.216.34/empty", "code": "fetch_empty_content"}
    ]
    observation = _observation(outcome, kind="web.research_verified")
    assert _verify(observation, statement) is VerifierStatus.PASS
    assert _verify(observation, "Discovery only and not claim eligible.") is VerifierStatus.FAIL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_code"),
    [
        ("rate", "EXA_RATE_LIMITED"),
        ("transport", "EXA_TRANSPORT_ERROR"),
        ("schema", "EXA_SCHEMA_DRIFT"),
        ("incomplete", "EXA_NO_COMPLETE_HIGHLIGHTS"),
    ],
)
async def test_verified_web_family_falls_back_for_each_exa_failure_class(
    failure_kind: str,
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.exa.ai":
            if failure_kind == "rate":
                return httpx.Response(429)
            if failure_kind == "transport":
                raise httpx.ConnectError("provider detail", request=request)
            if failure_kind == "schema":
                return httpx.Response(200, json={"unexpected": []})
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Incomplete",
                            "url": "https://example.org/incomplete",
                            "highlights": [],
                        }
                    ]
                },
            )
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Complete source",
                            "url": "https://93.184.216.34/complete",
                            "content": "Discovery only.",
                            "score": 0.9,
                        }
                    ]
                },
            )
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text="A complete retained source.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await VerifiedWebResearchTool(
            exa=ExaSearchTool(client=client, api_key="exa-key", clock=FixedClock(NOW)),
            tavily=TavilySearchTool(
                client=client,
                api_key="tavily-key",
                clock=FixedClock(NOW),
            ),
            fetch=PublicTextFetchTool(client=client, clock=FixedClock(NOW)),
        ).execute({"query": "bounded fallback"}, _context())

    assert isinstance(outcome, ToolSuccess)
    attempts = outcome.data["provider_attempts"]
    assert isinstance(attempts, list)
    assert attempts[0] == {
        "provider": "exa",
        "stage": "search",
        "status": "failed",
        "code": expected_code,
    }
    assert outcome.data["selected_provider"] == "tavily_public_fetch"


@pytest.mark.asyncio
async def test_verified_web_family_uses_shared_exa_cooldown_then_falls_back() -> None:
    exa_network_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exa_network_calls
        if request.url.host == "api.exa.ai":
            exa_network_calls += 1
            return httpx.Response(429, headers={"retry-after": "120"})
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Complete source",
                            "url": "https://93.184.216.34/complete",
                            "content": "Discovery only.",
                            "score": 0.9,
                        }
                    ]
                },
            )
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text="A complete retained source.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        exa = ExaSearchTool(client=client, api_key="exa-key", clock=FixedClock(NOW))
        first = await exa.execute({"query": "rate limited provider"}, _context())
        family = VerifiedWebResearchTool(
            exa=exa,
            tavily=TavilySearchTool(
                client=client,
                api_key="tavily-key",
                clock=FixedClock(NOW),
            ),
            fetch=PublicTextFetchTool(client=client, clock=FixedClock(NOW)),
        )
        fallback = await family.execute({"query": "rate limited provider"}, _context())

    assert isinstance(first, ToolFailure) and first.code == "EXA_RATE_LIMITED"
    assert isinstance(fallback, ToolSuccess)
    assert exa_network_calls == 1
    attempts = fallback.data["provider_attempts"]
    assert isinstance(attempts, list)
    assert attempts[0] == {
        "provider": "exa",
        "stage": "search",
        "status": "failed",
        "code": "EXA_COOLDOWN_ACTIVE",
    }


@pytest.mark.asyncio
async def test_verified_web_family_exhaustion_is_typed_bounded_and_secret_free() -> None:
    exa_secret = "exa-secret-never-leak"
    tavily_secret = "tavily-secret-never-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.exa.ai":
            return httpx.Response(402, json={"error": exa_secret})
        assert request.url.host == "api.tavily.com"
        return httpx.Response(429, json={"error": tavily_secret})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await VerifiedWebResearchTool(
            exa=ExaSearchTool(client=client, api_key=exa_secret, clock=FixedClock(NOW)),
            tavily=TavilySearchTool(
                client=client,
                api_key=tavily_secret,
                clock=FixedClock(NOW),
            ),
            fetch=PublicTextFetchTool(client=client, clock=FixedClock(NOW)),
        ).execute({"query": "provider exhaustion"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "VERIFIED_WEB_PROVIDERS_EXHAUSTED"
    assert not outcome.retryable
    assert "EXA_CREDITS_EXHAUSTED" in outcome.safe_message
    assert "TAVILY_RATE_LIMITED" in outcome.safe_message
    assert exa_secret not in outcome.model_dump_json()
    assert tavily_secret not in outcome.model_dump_json()


@pytest.mark.asyncio
async def test_verified_web_family_contains_unexpected_exa_exception_and_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exception_secret = "unexpected-provider-secret-never-leak"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.tavily.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Complete source",
                            "url": "https://93.184.216.34/complete",
                            "content": "Discovery only.",
                            "score": 0.9,
                        }
                    ]
                },
            )
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text="A complete retained source after provider-local recovery.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        exa = ExaSearchTool(client=client, api_key="exa-key", clock=FixedClock(NOW))

        async def explode(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError(exception_secret)

        monkeypatch.setattr(exa, "execute", explode)
        outcome = await VerifiedWebResearchTool(
            exa=exa,
            tavily=TavilySearchTool(
                client=client,
                api_key="tavily-key",
                clock=FixedClock(NOW),
            ),
            fetch=PublicTextFetchTool(client=client, clock=FixedClock(NOW)),
        ).execute({"query": "unexpected provider exception"}, _context())

    assert isinstance(outcome, ToolSuccess)
    attempts = outcome.data["provider_attempts"]
    assert isinstance(attempts, list)
    assert attempts[0] == {
        "provider": "exa",
        "stage": "search",
        "status": "failed",
        "code": "EXA_ADAPTER_EXCEPTION",
    }
    assert exception_secret not in outcome.model_dump_json()


def test_exa_setting_is_optional_secret_and_does_not_gate_conversation() -> None:
    secret = "exa-settings-secret-never-print"
    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        leo_model="fixture/model",
        exa_api_key=secret,
    )

    assert settings.missing_for_conversation_providers() == ()
    assert settings.exa_api_key is not None
    assert settings.exa_api_key.get_secret_value() == secret
    assert secret not in repr(settings)


@pytest.mark.asyncio
async def test_live_catalog_discovers_and_executes_exa_then_verifies_exact_highlight() -> None:
    model_calls = 0
    exa_calls = 0
    provider_gates = ProviderGateRegistry(FixedClock(NOW))

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, exa_calls
        if request.url.host == "api.exa.ai":
            exa_calls += 1
            assert request.headers["x-api-key"] == "exa-key"
            assert json.loads(request.content) == {
                "query": (
                    "Search the web for an official Python 3.14 noteworthy change "
                    "official documentation primary source"
                ),
                "type": "auto",
                "contents": {"highlights": True},
            }
            return httpx.Response(
                200,
                json={"requestId": "live-exa-1", "results": [_complete_result()]},
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        advertised = {item["function"]["name"] for item in payload["tools"]}
        assert "web_search_exa" in advertised
        assert "web_research_verified" not in advertised
        exa_observation = next(item for item in observations if item["kind"] == "web.search_exa")
        assert exa_observation["quality"] == "untrusted_retrieval"
        return httpx.Response(
            200,
            json={
                "id": "exa-completion-turn",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": STATEMENT,
                                    "source_claims": [
                                        {
                                            "statement": STATEMENT,
                                            "observation_ids": [exa_observation["id"]],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        },
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        exa_api_key="exa-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="Search the web for an official Python 3.14 noteworthy change",
            provider_gates=provider_gates,
        )

    exa_health = await provider_gates.get(
        provider="exa",
        max_concurrency=4,
        max_calls_per_minute=10,
    ).snapshot()
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == STATEMENT
    assert model_calls == 1
    assert exa_calls == 1
    assert result.run.usage.tool_calls == 1
    assert len(result.claims) == 1
    assert result.claims[0].statement == STATEMENT
    assert exa_health.calls_in_window == 1
    assert exa_health.successes == 1


@pytest.mark.asyncio
async def test_live_natural_version_prompt_preserves_tavily_then_fetch_with_exa_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "leo.integrations.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    objective = "What's one noteworthy change in Python 3.14?"
    retained_statement = "Python 3.14 adds deferred evaluation of annotations."
    model_calls = 0
    tavily_calls = 0
    fetch_calls = 0
    exa_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, tavily_calls, fetch_calls, exa_calls
        if request.url.host == "api.exa.ai":
            exa_calls += 1
            return httpx.Response(500)
        if request.url.host == "api.tavily.com":
            tavily_calls += 1
            payload = json.loads(request.content)
            assert payload["query"] == (
                "What's one noteworthy change in Python 3.14? official documentation primary source"
            )
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": TITLE,
                            "url": RESULT_URL,
                            "content": "Discovery metadata cannot support the claim.",
                            "score": 0.99,
                        }
                    ]
                },
            )
        if request.url.host == "docs.python.org":
            fetch_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=retained_statement,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        assert [item["kind"] for item in observations] == [
            "web.search_tavily",
            "web.fetch_public_text",
        ]
        advertised = {item["function"]["name"] for item in payload["tools"]}
        assert {"web_search_tavily", "web_fetch_public_text"}.issubset(advertised)
        fetch_observation = observations[-1]
        return httpx.Response(
            200,
            json={
                "id": "official-tavily-completion",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": retained_statement,
                                    "source_claims": [
                                        {
                                            "statement": retained_statement,
                                            "observation_ids": [fetch_observation["id"]],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        tavily_api_key="tavily-key",
        exa_api_key="exa-key",
        leo_max_model_turns=5,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == retained_statement
    assert tuple(item.kind for item in result.observations) == (
        "web.search_tavily",
        "web.fetch_public_text",
    )
    assert result.run.usage.tool_calls == 2
    assert model_calls == 1
    assert tavily_calls == 1
    assert fetch_calls == 1
    assert exa_calls == 0
