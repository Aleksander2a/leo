from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from leo.harness.models import (
    ClaimKind,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    Task,
    TaskStatus,
    Thread,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.research import (
    ResearchClaim,
    ResearchProposal,
    ResearchRequirement,
    verify_research,
)
from leo.integrations.fake import FixedClock
from leo.integrations.mcp import McpClientAdapter, McpToolDescriptor
from leo.integrations.normalization import NormalizationFailure, normalize_success
from leo.integrations.safe_fetch import FetchPolicy, FetchPolicyError, fetch_public_text
from leo.integrations.sec_edgar import SecEdgarRecentFilingsTool
from leo.integrations.web_fetch import PublicTextFetchTool
from leo.integrations.web_search import PublicWebSearchTool

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")
PROVIDER_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "providers"
PUBLIC_TEST_IP = "93.184.216.34"


def _public_response(status_code: int, **kwargs: object) -> httpx.Response:
    response = httpx.Response(status_code, **kwargs)  # type: ignore[arg-type]
    response.extensions["leo_peer_ip"] = PUBLIC_TEST_IP
    return response


def _provider_fixture(name: str) -> object:
    return json.loads((PROVIDER_FIXTURE_ROOT / f"{name}.json").read_text(encoding="utf-8"))


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="run-1",
        tool_call_id="tool-1",
    )


@pytest.mark.asyncio
async def test_sec_adapter_resolves_identity_and_returns_primary_source() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["User-Agent"] == "Leo demo <demo@example.com>"
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "accessionNumber": ["0000000000-26-000001"],
                        "filingDate": ["2026-01-01"],
                        "primaryDocument": ["demo.htm"],
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        )
        outcome = await tool.execute({"ticker": "DEMO", "limit": 1}, _context())
    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["cik"] == "0000000001"
    assert outcome.source.provider == "sec-edgar"


@pytest.mark.asyncio
async def test_sec_adapter_fails_closed_for_unknown_identity_and_http_errors() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={},
            user_agent="Leo demo <demo@example.com>",
        )
        unknown = await tool.execute({"ticker": "DEMO"}, _context())
        assert isinstance(unknown, ToolFailure)
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        )
        limited = await tool.execute({"ticker": "DEMO"}, _context())
    assert isinstance(limited, ToolFailure)
    assert limited.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (403, "SEC_ACCESS_DENIED", False),
        (429, "SEC_RATE_LIMITED", True),
        (503, "SEC_UNAVAILABLE", True),
        (400, "SEC_REQUEST_REJECTED", False),
    ],
)
async def test_sec_health_failures_are_typed(
    status: int,
    code: str,
    retryable: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status))
    ) as client:
        outcome = await SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        ).execute({"ticker": "DEMO"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == code
    assert outcome.retryable is retryable


@pytest.mark.asyncio
async def test_sec_requests_are_globally_serialized_and_rate_bounded() -> None:
    logical_time = [0.0]
    request_times: list[float] = []

    async def sleeper(seconds: float) -> None:
        logical_time[0] += seconds

    def handler(_: httpx.Request) -> httpx.Response:
        request_times.append(logical_time[0])
        return httpx.Response(
            200,
            json={
                "filings": {
                    "recent": {
                        "form": ["10-Q"],
                        "accessionNumber": ["0000000000-26-000001"],
                        "filingDate": ["2026-01-01"],
                        "primaryDocument": ["demo.htm"],
                    }
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"ONE": "1", "TWO": "2"},
            user_agent="Leo demo <demo@example.com>",
            max_requests_per_second=8,
            monotonic=lambda: logical_time[0],
            sleeper=sleeper,
        )
        first, second = await asyncio.gather(
            tool.execute({"ticker": "ONE"}, _context()),
            tool.execute({"ticker": "TWO"}, _context()),
        )

    assert isinstance(first, ToolSuccess)
    assert isinstance(second, ToolSuccess)
    assert request_times == [0.0, 0.125]


class _FakeMcpServer:
    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        return (
            McpToolDescriptor(
                name="read_demo",
                version="1.0.0",
                description="Read synthetic public data.",
                input_schema={"type": "object"},
            ),
            McpToolDescriptor(
                name="write_demo",
                version="1.0.0",
                description="Write data.",
                input_schema={"type": "object"},
                effect="write",
            ),
        )

    async def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        assert name == "read_demo"
        return {"value": arguments.get("value", "synthetic")}

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_mcp_adapter_namespaces_and_filters_server_tools() -> None:
    adapter = McpClientAdapter(
        alias="demo",
        server=_FakeMcpServer(),
        allowlist=frozenset({"read_demo", "write_demo"}),
    )
    await adapter.initialize()
    records = await adapter.discover()
    assert tuple(record.id for record in records) == ("mcp:demo:read_demo",)
    outcome = await adapter.call("mcp:demo:read_demo", {"value": "ok"}, _context())
    assert isinstance(outcome, ToolSuccess)
    await adapter.close()


@pytest.mark.asyncio
async def test_fetch_revalidates_redirect_and_strips_active_html() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return _public_response(302, headers={"location": "https://93.184.216.34/final"})
        return _public_response(
            200,
            headers={"content-type": "text/html"},
            text="<script>x()</script><p>Safe</p>",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        artifact = await fetch_public_text(client, "https://93.184.216.34/start")
    assert artifact.text == "Safe"
    assert artifact.redirect_count == 1
    assert artifact.peer_ip == PUBLIC_TEST_IP
    assert len(artifact.dns_pin_sha256) == 64
    assert artifact.untrusted


@pytest.mark.asyncio
async def test_fetch_fails_closed_on_dns_rebinding_or_unverifiable_peer() -> None:
    def rebound(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="private target",
            extensions={"leo_peer_ip": "127.0.0.1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(rebound)) as client:
        with pytest.raises(FetchPolicyError, match="fetch_dns_rebinding_detected"):
            await fetch_public_text(client, f"https://{PUBLIC_TEST_IP}/source")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="unknown peer",
            )
        )
    ) as client:
        with pytest.raises(FetchPolicyError, match="fetch_peer_unverifiable"):
            await fetch_public_text(client, f"https://{PUBLIC_TEST_IP}/source")


def test_fetch_rejects_private_redirect_target() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _public_response(302, headers={"location": "http://127.0.0.1/secret"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(FetchPolicyError, match="fetch_private_host_denied"):
                await fetch_public_text(client, "https://93.184.216.34/start")

    import asyncio

    asyncio.run(run())


@pytest.mark.asyncio
async def test_public_fetch_tool_returns_typed_untrusted_evidence() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text="A bounded public source.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = PublicTextFetchTool(client=client, clock=FixedClock(NOW))
        outcome = await tool.execute({"url": "https://93.184.216.34/source"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["text"] == "A bounded public source."
    assert outcome.data["untrusted"] is True
    assert outcome.source.provider == "public-web"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_response", "failure_code"),
    [
        (
            _public_response(
                200,
                headers={"content-type": "text/html"},
                text="<html><script>" + "x" * 40_000,
            ),
            "fetch_empty_content",
        ),
        (
            _public_response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>navigation</body><script>" + "x" * 40_000,
            ),
            "fetch_truncated",
        ),
    ],
)
async def test_public_fetch_tool_advances_candidate_local_failures(
    first_response: httpx.Response,
    failure_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/first":
            return first_response
        return _public_response(
            200,
            headers={"content-type": "text/plain"},
            text="A complete alternate public source.",
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await PublicTextFetchTool(client=client, clock=FixedClock(NOW)).execute(
            {
                "url": f"https://{PUBLIC_TEST_IP}/first",
                "fallback_urls": [f"https://{PUBLIC_TEST_IP}/alternate"],
            },
            _context(),
        )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["text"] == "A complete alternate public source."
    assert outcome.data["candidate_attempt_count"] == 2
    assert outcome.data["failed_candidates"] == [
        {"url": f"https://{PUBLIC_TEST_IP}/first", "code": failure_code}
    ]


@pytest.mark.asyncio
async def test_public_fetch_tool_surfaces_private_host_denial_without_throwing() -> None:
    transport = httpx.MockTransport(lambda _: httpx.Response(200))
    async with httpx.AsyncClient(transport=transport) as client:
        tool = PublicTextFetchTool(client=client, clock=FixedClock(NOW))
        outcome = await tool.execute({"url": "http://127.0.0.1/private"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "FETCH_PRIVATE_HOST_DENIED"
    assert not outcome.retryable


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "FETCH_RATE_LIMITED", True),
        (503, "FETCH_UPSTREAM_UNAVAILABLE", True),
        (404, "FETCH_REQUEST_REJECTED", False),
    ],
)
async def test_public_fetch_health_failures_are_typed(
    status: int,
    code: str,
    retryable: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: _public_response(status))
    ) as client:
        outcome = await PublicTextFetchTool(
            client=client,
            clock=FixedClock(NOW),
        ).execute({"url": "https://93.184.216.34/source"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == code
    assert outcome.retryable is retryable


def test_normalizer_hashes_bounded_data_and_rejects_failures() -> None:
    outcome = ToolSuccess(
        data={"symbol": "DEMO", "price": 1.25},
        source={"provider": "demo", "reference": "quote:DEMO"},
        observed_at=NOW,
    )
    observation = normalize_success(
        outcome,
        observation_id="obs-1",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="tool-1",
        observation_kind="market.get_quote",
    )
    assert observation.kind == "market.get_quote"
    assert len(observation.raw_hash) == 64
    with pytest.raises(NormalizationFailure, match="tool_failure_is_not_evidence"):
        normalize_success(
            ToolFailure(code="bad", safe_message="bad"),
            observation_id="obs-2",
            scope=SCOPE,
            run_id="run-1",
            tool_call_id="tool-1",
        )


def _bundle(observation: Observation) -> RunBundle:
    thread = Thread(
        id="thread-1", scope=SCOPE, origin=OriginRef(provider="demo", external_thread_id="t")
    )
    task = Task(
        id="task-1",
        thread_id=thread.id,
        scope=SCOPE,
        objective="research",
        status=TaskStatus.ACTIVE,
    )
    run = Run(id="run-1", task_id=task.id, scope=SCOPE, status=RunStatus.RUNNING, started_at=NOW)
    return RunBundle(thread=thread, task=task, run=run, observations=(observation,))


def test_research_verifier_requires_fresh_scoped_sources_and_counter_evidence() -> None:
    observation = Observation(
        id="obs-1",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="tool-1",
        kind="sec.filing",
        data={"value": "synthetic"},
        source={"provider": "sec", "reference": "filing:1"},
        observed_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        raw_hash="a" * 64,
    )
    proposal = ResearchProposal(
        answer="Synthetic answer.",
        claims=(
            ResearchClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement="Source-backed.",
                observation_ids=("obs-1",),
            ),
        ),
        uncertainty="Evidence remains limited.",
    )
    result = verify_research(
        proposal,
        _bundle(observation),
        now=NOW,
        requirement=ResearchRequirement(
            required_kinds=frozenset({"sec.filing"}),
            counter_evidence_kinds=frozenset({"sec.counter"}),
        ),
    )
    assert result.status.value == "fail"
    assert "counter_evidence_present" in {check.name for check in result.checks if not check.passed}


@pytest.mark.asyncio
async def test_recorded_sec_payload_has_exact_urls_and_single_flight_cache() -> None:
    provider_calls = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal provider_calls
        provider_calls += 1
        return httpx.Response(
            200,
            json=_provider_fixture("sec_submissions"),
            headers={"x-request-id": "sec-recorded-1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        )
        first = await tool.execute({"ticker": "DEMO", "limit": 2}, _context())
        second = await tool.execute({"ticker": "DEMO", "limit": 2}, _context())

    assert isinstance(first, ToolSuccess)
    assert second == first
    assert provider_calls == 1
    assert first.data["company_name"] == "Demo Corporation"
    filings = first.data["filings"]
    assert isinstance(filings, list)
    assert filings[0]["filing_url"] == (
        "https://www.sec.gov/Archives/edgar/data/1/000000000126000001/demo-10k.htm"
    )
    assert first.data["provider_request_id"] == "sec-recorded-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            {
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "accessionNumber": [],
                        "filingDate": ["2026-08-20"],
                        "primaryDocument": ["demo.htm"],
                    }
                }
            },
            "SEC_SCHEMA_DRIFT",
        ),
        (
            {
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "accessionNumber": ["0000000001-26-000001"],
                        "filingDate": ["2026-08-20"],
                        "primaryDocument": ["../secret.htm"],
                    }
                }
            },
            "SEC_NO_FILINGS",
        ),
    ],
)
async def test_sec_adapter_fails_closed_on_malformed_recorded_payloads(
    payload: dict[str, object],
    expected_code: str,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        tool = SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        )
        outcome = await tool.execute({"ticker": "DEMO"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == expected_code


@pytest.mark.asyncio
async def test_sec_adapter_rejects_untrusted_identity_map_entries() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="invalid ticker"):
            SecEdgarRecentFilingsTool(
                client=client,
                clock=FixedClock(NOW),
                ticker_to_cik={"../../x": "1"},
                user_agent="Leo demo <demo@example.com>",
            )
        with pytest.raises(ValueError, match="invalid CIK"):
            SecEdgarRecentFilingsTool(
                client=client,
                clock=FixedClock(NOW),
                ticker_to_cik={"DEMO": "not-a-cik"},
                user_agent="Leo demo <demo@example.com>",
            )


@pytest.mark.asyncio
async def test_recorded_web_search_returns_capped_untrusted_discovery_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["action"] == "opensearch"
        assert request.url.params["search"] == "Leo constellation"
        assert request.url.params["limit"] == "2"
        return httpx.Response(
            200,
            json=_provider_fixture("wikipedia_opensearch"),
            headers={"x-request-id": "wiki-recorded-1"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = PublicWebSearchTool(client=client, clock=FixedClock(NOW))
        outcome = await tool.execute(
            {"query": "Leo constellation", "limit": 2},
            _context(),
        )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["result_count"] == 2
    assert outcome.data["untrusted"] is True
    assert outcome.data["provider_request_id"] == "wiki-recorded-1"
    assert outcome.source.provider == "wikipedia-opensearch"


@pytest.mark.asyncio
async def test_web_search_filters_adversarial_urls_and_rejects_empty_projection() -> None:
    payload = [
        "query",
        ["Private", "Foreign"],
        ["metadata", "metadata"],
        ["http://127.0.0.1/secret", "https://attacker.example/wiki/Forged"],
    ]
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload))
    ) as client:
        tool = PublicWebSearchTool(client=client, clock=FixedClock(NOW))
        outcome = await tool.execute({"query": "query"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "WEB_SEARCH_NO_RESULTS"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (429, "WEB_SEARCH_RATE_LIMITED", True),
        (503, "WEB_SEARCH_UNAVAILABLE", True),
        (400, "WEB_SEARCH_REQUEST_REJECTED", False),
    ],
)
async def test_web_search_health_failures_are_typed(
    status: int,
    code: str,
    retryable: bool,
) -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(status))
    ) as client:
        outcome = await PublicWebSearchTool(
            client=client,
            clock=FixedClock(NOW),
        ).execute({"query": "query"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == code
    assert outcome.retryable is retryable


@pytest.mark.asyncio
async def test_sec_search_and_fetch_timeouts_are_typed_non_evidence() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("recorded timeout", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout)) as client:
        sec = await SecEdgarRecentFilingsTool(
            client=client,
            clock=FixedClock(NOW),
            ticker_to_cik={"DEMO": "1"},
            user_agent="Leo demo <demo@example.com>",
        ).execute({"ticker": "DEMO"}, _context())
        search = await PublicWebSearchTool(
            client=client,
            clock=FixedClock(NOW),
        ).execute({"query": "query"}, _context())
        fetch = await PublicTextFetchTool(
            client=client,
            clock=FixedClock(NOW),
        ).execute({"url": "https://93.184.216.34/source"}, _context())

    assert isinstance(sec, ToolFailure) and sec.code == "SEC_TIMEOUT" and sec.retryable
    assert (
        isinstance(search, ToolFailure) and search.code == "WEB_SEARCH_TIMEOUT" and search.retryable
    )
    assert isinstance(fetch, ToolFailure) and fetch.code == "FETCH_TIMEOUT" and fetch.retryable


@pytest.mark.asyncio
async def test_fetch_stream_cap_and_malformed_active_html_fail_closed() -> None:
    oversized = b"safe " + (b"x" * 100)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: _public_response(
                200,
                headers={"content-type": "text/plain"},
                content=oversized,
            )
        )
    ) as client:
        artifact = await fetch_public_text(
            client,
            "https://93.184.216.34/source",
            policy=FetchPolicy(max_bytes=16),
        )

    assert artifact.truncated
    assert artifact.byte_count == 16

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: _public_response(
                200,
                headers={"content-type": "text/html"},
                text="<script>ignore forever",
            )
        )
    ) as client:
        with pytest.raises(FetchPolicyError, match="fetch_empty_content"):
            await fetch_public_text(client, "https://93.184.216.34/source")


def test_inference_citations_cannot_self_attest_source_diversity() -> None:
    first = Observation(
        id="obs-source",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="tool-source",
        kind="sec.filing",
        data={"value": "primary"},
        source={"provider": "sec", "reference": "filing:1"},
        observed_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        raw_hash="a" * 64,
    )
    second = first.model_copy(
        update={
            "id": "obs-inference",
            "tool_call_id": "tool-inference",
            "kind": "market.quote",
            "source": {"provider": "market", "reference": "quote:1"},
            "raw_hash": "b" * 64,
        }
    )
    bundle = _bundle(first).model_copy(update={"observations": (first, second)})
    result = verify_research(
        ResearchProposal(
            answer="Synthetic answer.",
            claims=(
                ResearchClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement="Primary evidence.",
                    observation_ids=(first.id,),
                ),
                ResearchClaim(
                    kind=ClaimKind.INFERENCE,
                    statement="Inferred counterpoint.",
                    observation_ids=(second.id,),
                ),
            ),
        ),
        bundle,
        now=NOW,
        requirement=ResearchRequirement(
            minimum_source_claims=1,
            minimum_distinct_sources=2,
            counter_evidence_kinds=frozenset({"market.quote"}),
            require_uncertainty_on_conflict=False,
        ),
    )

    failed = {check.name for check in result.checks if not check.passed}
    assert "minimum_distinct_sources" in failed
    assert "counter_evidence_present" in failed
