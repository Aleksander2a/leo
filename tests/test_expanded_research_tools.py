from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from leo.config import Settings
from leo.harness.earnings import canonical_earnings_statements
from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EventType,
    EvidenceQuality,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
    VerifierStatus,
)
from leo.harness.normalization import normalize_success
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.harness.web_research import rank_tavily_result_urls
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.finnhub import (
    FinnhubBasicFinancialsTool,
    FinnhubCompanyNewsTool,
    FinnhubCompanyProfileTool,
    FinnhubEarningsSurprisesTool,
    FinnhubProviderLimiter,
)
from leo.integrations.provider_runtime import ProviderCallGate
from leo.integrations.tavily import TavilySearchTool
from leo.live import run_live_conversation
from leo.url_policy import is_public_https_url

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="run",
        tool_call_id="call",
    )


def _bundle(observation: Observation) -> RunBundle:
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective="Research")
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=(observation,))


def _observation(kind: str, success: ToolSuccess) -> Observation:
    return normalize_success(
        success,
        observation_id="obs",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call",
        observation_kind=kind,
    )


def test_thread_context_normalizes_as_internal_context() -> None:
    observation = _observation(
        "thread_context.open",
        ToolSuccess(
            data={"handle": "opaque", "chunks": [], "content_digest": "a" * 64},
            source={"provider": "slack-history", "reference": "opaque-handle"},
            observed_at=NOW,
        ),
    )

    assert observation.quality is EvidenceQuality.INTERNAL_CONTEXT


def _verify(
    observation: Observation,
    statement: str,
    *,
    claim_kind: ClaimKind = ClaimKind.SOURCE_CLAIM,
    answer: str | None = None,
) -> VerifierStatus:
    proposal = CompletionProposal(
        answer=answer or statement,
        claims=(
            CandidateClaim(
                kind=claim_kind,
                statement=statement,
                observation_ids=(observation.id,),
            ),
        ),
    )
    return (
        DeterministicCompletionVerifier(
            SequentialIdGenerator(),
            FixedClock(NOW),
            require_source_claim=claim_kind is ClaimKind.SOURCE_CLAIM,
        )
        .verify(proposal, _bundle(observation))
        .result.status
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/report",
        "https://localhost/report",
        "https://service.localhost/report",
        "https://127.0.0.1/report",
        "https://127.1/report",
        "https://10.0.0.8/report",
        "https://224.0.0.1/report",
        "https://[::1]/report",
        "https://[ff02::1]/report",
        "https://user:secret@example.org/report",
        "https://internal/report",
    ],
)
def test_literal_safe_public_https_policy_rejects_local_and_ambiguous_hosts(url: str) -> None:
    assert not is_public_https_url(url)


def test_literal_safe_public_https_policy_accepts_public_dns_without_resolving() -> None:
    assert is_public_https_url("https://example.org/public/report")


@pytest.mark.asyncio
async def test_tavily_search_is_bounded_discovery_and_never_requests_generated_answer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer test-tavily-key"
        payload = request.read().decode()
        assert '"include_answer":false' in payload
        assert '"include_raw_content":false' in payload
        return httpx.Response(
            200,
            json={
                "query": "official example",
                "results": [
                    {
                        "title": "Official source",
                        "url": "https://example.org/report",
                        "content": "A discovery snippet that must be fetched before citation.",
                        "score": 0.91,
                    }
                ],
                "request_id": "req-tavily-1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await TavilySearchTool(
            client=client,
            api_key="test-tavily-key",
            clock=FixedClock(NOW),
        ).execute({"query": "official example", "max_results": 2}, _context())

    assert isinstance(outcome, ToolSuccess)
    observation = _observation("web.search_tavily", outcome)
    assert observation.quality is EvidenceQuality.DISCOVERY_ONLY
    assert observation.expires_at == NOW + timedelta(minutes=10)
    assert observation.source.provider == "tavily"
    assert observation.data["provider_request_id"] == "req-tavily-1"

    statement = "Tavily returned 1 discovery results for query: official example"
    assert _verify(observation, statement, claim_kind=ClaimKind.INFERENCE) is VerifierStatus.PASS
    assert _verify(observation, statement) is VerifierStatus.FAIL
    assert (
        _verify(
            observation,
            "The discovery snippet proves the reported fact.",
            claim_kind=ClaimKind.INFERENCE,
        )
        is VerifierStatus.FAIL
    )


@pytest.mark.asyncio
async def test_tavily_skips_malformed_results_and_accepts_missing_optional_score() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    None,
                    "not-an-object",
                    {
                        "title": "Bad score",
                        "url": "https://example.org/bad-score",
                        "content": "Malformed metadata.",
                        "score": 2,
                    },
                    {
                        "title": "Private",
                        "url": "https://127.0.0.1/private",
                        "content": "Not public.",
                    },
                    {
                        "title": "Official docs without a score",
                        "url": "https://docs.example.org/guide",
                        "content": "Discovery text must still be fetched before citation.",
                    },
                    {
                        "title": "Complete result",
                        "url": "https://example.org/report",
                        "content": "Another discovery-only snippet.",
                        "score": 0.75,
                    },
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await TavilySearchTool(
            client=client,
            api_key="key",
            clock=FixedClock(NOW),
        ).execute({"query": "mixed Tavily results", "max_results": 2}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["rejected_result_count"] == 4
    results = outcome.data["results"]
    assert isinstance(results, list) and len(results) == 2
    assert results[0] == {
        "title": "Official docs without a score",
        "url": "https://docs.example.org/guide",
        "snippet": "Discovery text must still be fetched before citation.",
        "score": None,
        "missing_fields": ["score"],
    }

    observation = _observation("web.search_tavily", outcome)
    canonical = "Tavily returned 2 discovery results for query: mixed Tavily results"
    assert _verify(observation, canonical, claim_kind=ClaimKind.INFERENCE) is VerifierStatus.PASS
    snippet = "Discovery text must still be fetched before citation."
    assert _verify(observation, snippet) is VerifierStatus.FAIL
    assert _verify(observation, snippet, claim_kind=ClaimKind.INFERENCE) is VerifierStatus.FAIL

    malformed_results = [dict(item) for item in results if isinstance(item, dict)]
    malformed_results[0].pop("missing_fields")
    malformed_data = dict(observation.data)
    malformed_data["results"] = malformed_results
    malformed = observation.model_copy(update={"data": malformed_data})
    assert _verify(malformed, canonical, claim_kind=ClaimKind.INFERENCE) is VerifierStatus.FAIL


def test_tavily_ranking_treats_missing_optional_score_as_neutral() -> None:
    ranked = rank_tavily_result_urls(
        [
            {
                "url": "https://example.org/community/python",
                "score": 0.99,
            },
            {
                "url": "https://docs.python.org/3/whatsnew/3.14.html",
                "score": None,
                "missing_fields": ["score"],
            },
        ],
        "Python 3.14",
    )

    assert ranked == (
        "https://docs.python.org/3/whatsnew/3.14.html",
        "https://example.org/community/python",
    )


@pytest.mark.asyncio
async def test_tavily_registry_gate_bounds_retry_after_and_tracks_advanced_credit_cost() -> None:
    calls = 0
    clock = FixedClock(NOW)
    gate = ProviderCallGate(
        provider="tavily",
        clock=clock,
        max_calls_per_minute=10,
        max_calls_per_month=500,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "999999"})
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Official source",
                        "url": "https://example.org/report",
                        "content": "Complete discovery metadata.",
                        "score": 0.9,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = TavilySearchTool(
            client=client,
            api_key="key",
            clock=clock,
            gate=gate,
        )
        limited = await tool.execute(
            {"query": "bounded retry", "search_depth": "advanced"},
            _context(),
        )
        clock.advance(seconds=299)
        cooling_down = await tool.execute(
            {"query": "still cooling", "search_depth": "advanced"},
            _context(),
        )
        clock.advance(seconds=2)
        recovered = await tool.execute(
            {"query": "after cooldown", "search_depth": "advanced"},
            _context(),
        )

    health = await gate.snapshot()
    assert isinstance(limited, ToolFailure) and limited.code == "TAVILY_RATE_LIMITED"
    assert isinstance(cooling_down, ToolFailure)
    assert cooling_down.code == "TAVILY_COOLDOWN_ACTIVE"
    assert isinstance(recovered, ToolSuccess)
    assert calls == 2
    assert health.rate_limit_count == 1
    assert health.provider_credits_used == 2
    assert health.calls_in_month == 2


@pytest.mark.asyncio
async def test_tavily_free_tier_monthly_call_ceiling_is_conservatively_bounded() -> None:
    async with httpx.AsyncClient() as client:
        tool = TavilySearchTool(
            client=client,
            api_key="key",
            clock=FixedClock(NOW),
        )
        health = await tool.provider_health()
        assert health.local_monthly_provider_credit_limit == 1_000
        assert health.remaining_local_monthly_provider_credits == 1_000
        with pytest.raises(ValueError, match="free-tier limits"):
            TavilySearchTool(
                client=client,
                api_key="key",
                clock=FixedClock(NOW),
                max_calls_per_month=501,
            )
    with pytest.raises(ValueError):
        Settings(_env_file=None, tavily_max_calls_per_month=501)


@pytest.mark.asyncio
async def test_tavily_rejects_private_results_and_malformed_provider_payloads() -> None:
    responses = iter(
        (
            httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "title": "Private",
                            "url": "https://127.0.0.1/secret",
                            "content": "not public",
                            "score": 1.0,
                        },
                        {
                            "title": "Localhost",
                            "url": "https://localhost/secret",
                            "content": "not public",
                            "score": 1.0,
                        },
                        {
                            "title": "Private network",
                            "url": "https://10.0.0.8/secret",
                            "content": "not public",
                            "score": 1.0,
                        },
                    ]
                },
            ),
            httpx.Response(200, json={"results": "not-a-list"}),
            httpx.Response(429),
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: next(responses))
    ) as client:
        tool = TavilySearchTool(client=client, api_key="key", clock=FixedClock(NOW))
        private = await tool.execute({"query": "private result"}, _context())
        malformed = await tool.execute({"query": "bad shape"}, _context())
        limited = await tool.execute({"query": "rate limited"}, _context())

    assert isinstance(private, ToolFailure) and private.code == "TAVILY_NO_RESULTS"
    assert isinstance(malformed, ToolFailure) and malformed.code == "TAVILY_SCHEMA_DRIFT"
    assert isinstance(limited, ToolFailure) and limited.code == "TAVILY_RATE_LIMITED"
    assert limited.retryable


@pytest.mark.asyncio
async def test_tavily_rejects_oversized_payload_before_json_and_accounts_for_credit() -> None:
    calls = 0
    gate = ProviderCallGate(
        provider="tavily",
        clock=FixedClock(NOW),
        max_calls_per_minute=10,
        max_calls_per_month=500,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            content=b"{" + (b"x" * 1_048_576),
            headers={"Content-Type": "application/json"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await TavilySearchTool(
            client=client,
            api_key="key",
            clock=FixedClock(NOW),
            gate=gate,
        ).execute(
            {"query": "oversized response", "search_depth": "advanced"},
            _context(),
        )

    health = await gate.snapshot()
    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "TAVILY_RESPONSE_TOO_LARGE"
    assert calls == 1
    assert health.calls_in_month == 1
    assert health.provider_credits_used == 2
    assert health.failures == 1


def test_tavily_filter_contract_is_small_normalized_and_unambiguous() -> None:
    async def exercise() -> None:
        transport = httpx.MockTransport(lambda _: httpx.Response(500))
        async with httpx.AsyncClient(transport=transport) as client:
            tool = TavilySearchTool(client=client, api_key="key", clock=FixedClock(NOW))
            validated = tool.validate(
                {
                    "query": "bounded filters",
                    "include_domains": ["SEC.GOV", "sec.gov"],
                    "exclude_domains": ["example.com"],
                    "start_date": "2026-01-01",
                    "end_date": "2026-02-01",
                }
            )
            assert validated["include_domains"] == ["sec.gov"]
            with pytest.raises(ValueError):
                tool.validate(
                    {
                        "query": "overlap",
                        "include_domains": ["sec.gov"],
                        "exclude_domains": ["sec.gov"],
                    }
                )
            with pytest.raises(ValueError):
                tool.validate(
                    {
                        "query": "ambiguous dates",
                        "time_range": "week",
                        "start_date": "2026-01-01",
                        "end_date": "2026-02-01",
                    }
                )
            with pytest.raises(ValueError):
                tool.validate(
                    {
                        "query": "unsafe domain",
                        "include_domains": ["https://sec.gov/path"],
                    }
                )

    import asyncio

    asyncio.run(exercise())


@pytest.mark.asyncio
async def test_finnhub_company_profile_normalizes_one_groundable_statement() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/stock/profile2"
        assert request.url.params["symbol"] == "MSFT"
        assert request.headers["X-Finnhub-Token"] == "test-finnhub-key"
        return httpx.Response(
            200,
            json={
                "ticker": "MSFT",
                "name": "Microsoft Corp",
                "exchange": "NASDAQ NMS - GLOBAL MARKET",
                "finnhubIndustry": "Technology",
                "currency": "USD",
                "marketCapitalization": 3_800_000.5,
                "weburl": "https://localhost/internal",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubCompanyProfileTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT"}, _context())

    assert isinstance(outcome, ToolSuccess)
    statement = (
        "MSFT is Microsoft Corp, listed on NASDAQ NMS - GLOBAL MARKET, "
        "in Finnhub industry Technology."
    )
    observation = _observation("market.get_company_profile", outcome)
    assert observation.expires_at == NOW + timedelta(days=1)
    assert observation.data["statements"] == [statement]
    assert "web_url" not in observation.data
    assert observation.source.url is None
    assert _verify(observation, statement) is VerifierStatus.PASS
    assert _verify(observation, "MSFT has market capitalization 3800000.5.") is VerifierStatus.FAIL


@pytest.mark.asyncio
async def test_finnhub_company_news_uses_bounded_server_derived_date_window() -> None:
    timestamp = int(NOW.timestamp())

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/company-news"
        assert request.url.params["symbol"] == "MSFT"
        assert request.url.params["from"] == "2026-08-15"
        assert request.url.params["to"] == "2026-08-22"
        return httpx.Response(
            200,
            json=[
                {
                    "datetime": timestamp,
                    "headline": "Microsoft announces a bounded update",
                    "source": "Example Wire",
                    "summary": "Provider-reported summary.",
                    "url": "https://example.org/news/msft-update",
                },
                {
                    "datetime": timestamp + 86_400,
                    "headline": "Future-dated item",
                    "source": "Example Wire",
                    "url": "https://example.org/news/future",
                },
                {
                    "datetime": timestamp,
                    "headline": "Private literal",
                    "source": "Example Wire",
                    "url": "https://192.168.1.8/internal",
                },
                {
                    "datetime": timestamp,
                    "headline": "Localhost",
                    "source": "Example Wire",
                    "url": "https://localhost/internal",
                },
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubCompanyNewsTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT", "days": 7, "limit": 2}, _context())

    assert isinstance(outcome, ToolSuccess)
    statement = (
        f"On {NOW.isoformat()}, Example Wire reported for MSFT: "
        "Microsoft announces a bounded update "
        "Source URL: https://example.org/news/msft-update"
    )
    observation = _observation("market.get_company_news", outcome)
    assert observation.expires_at == NOW + timedelta(minutes=15)
    assert observation.data["rejected_item_count"] == 3
    assert observation.source.url == "https://example.org/news/msft-update"
    assert _verify(observation, statement) is VerifierStatus.PASS
    assert (
        _verify(
            observation,
            statement.replace(
                "https://example.org/news/msft-update",
                "https://example.org/news/other",
            ),
        )
        is VerifierStatus.FAIL
    )


@pytest.mark.asyncio
async def test_finnhub_multi_article_observation_omits_ambiguous_source_ref_url() -> None:
    timestamp = int(NOW.timestamp())

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "datetime": timestamp,
                    "headline": "First update",
                    "source": "First Wire",
                    "url": "https://example.org/news/first",
                },
                {
                    "datetime": timestamp - 60,
                    "headline": "Second update",
                    "source": "Second Wire",
                    "url": "https://example.org/news/second",
                },
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubCompanyNewsTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT", "days": 7, "limit": 2}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["item_count"] == 2
    assert outcome.source.url is None


@pytest.mark.asyncio
async def test_finnhub_earnings_surprise_is_exactly_grounded_in_numeric_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/stock/earnings"
        return httpx.Response(
            200,
            json=[
                {
                    "actual": 3.65,
                    "estimate": 3.37,
                    "period": "2026-06-30",
                    "quarter": 2,
                    "surprise": 0.28,
                    "surprisePercent": 8.31,
                    "symbol": "MSFT",
                    "year": 2026,
                }
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubEarningsSurprisesTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT"}, _context())

    assert isinstance(outcome, ToolSuccess)
    statement = "MSFT reported actual EPS 3.65 versus estimate 3.37 for period 2026-06-30."
    summary = (
        "Across 1 normalized Finnhub earnings observations for periods 2026-06-30, "
        "MSFT beat the EPS estimate in 1, missed it in 0, and matched it in 0."
    )
    observation = _observation("market.get_earnings_surprises", outcome)
    assert observation.source.url is None
    assert observation.expires_at == NOW + timedelta(hours=6)
    assert outcome.data["statements"] == [summary, statement]
    assert _verify(observation, summary) is VerifierStatus.PASS
    assert _verify(observation, statement) is VerifierStatus.PASS
    assert (
        _verify(
            observation,
            "MSFT reported actual EPS 3.66 versus estimate 3.37 for period 2026-06-30.",
        )
        is VerifierStatus.FAIL
    )
    assert (
        _verify(
            observation,
            summary.replace("missed it in 0", "missed it in 1"),
        )
        is VerifierStatus.FAIL
    )


@pytest.mark.asyncio
async def test_finnhub_earnings_rejects_future_and_invalid_periods() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[
                {
                    "actual": 3.65,
                    "estimate": 3.37,
                    "period": "2027-06-30",
                    "symbol": "MSFT",
                },
                {
                    "actual": 3.65,
                    "estimate": 3.37,
                    "period": "2026-02-31",
                    "symbol": "MSFT",
                },
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubEarningsSurprisesTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "FINNHUB_NO_EARNINGS"


@pytest.mark.asyncio
async def test_finnhub_basic_financials_whitelists_and_grounds_exact_metrics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/stock/metric"
        assert request.url.params["metric"] == "all"
        return httpx.Response(
            200,
            json={
                "metric": {
                    "beta": 0.91,
                    "52WeekHigh": 555.45,
                    "unsupportedSecretMetric": 999,
                },
                "metricType": "all",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await FinnhubBasicFinancialsTool(
            client=client,
            api_key="test-finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "MSFT"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert "unsupportedSecretMetric" not in outcome.data["metrics"]
    observation = _observation("market.get_basic_financials", outcome)
    assert observation.source.url is None
    assert observation.expires_at == NOW + timedelta(hours=6)
    statement = "MSFT has Finnhub beta 0.91."
    assert _verify(observation, statement) is VerifierStatus.PASS
    assert _verify(observation, "MSFT has Finnhub beta 0.92.") is VerifierStatus.FAIL


@pytest.mark.asyncio
async def test_shared_finnhub_limiter_caps_cross_endpoint_concurrency() -> None:
    active = 0
    maximum_active = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        if request.url.path.endswith("/stock/profile2"):
            return httpx.Response(
                200,
                json={"ticker": "MSFT", "name": "Microsoft", "exchange": "NASDAQ"},
            )
        return httpx.Response(200, json={"metric": {"beta": 0.91}})

    limiter = FinnhubProviderLimiter(max_concurrency=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        profile = FinnhubCompanyProfileTool(
            client=client,
            api_key="key",
            clock=FixedClock(NOW),
            limiter=limiter,
        )
        financials = FinnhubBasicFinancialsTool(
            client=client,
            api_key="key",
            clock=FixedClock(NOW),
            limiter=limiter,
        )
        outcomes = await asyncio.gather(
            profile.execute({"symbol": "MSFT"}, _context()),
            financials.execute({"symbol": "MSFT"}, _context()),
        )

    assert all(isinstance(item, ToolSuccess) for item in outcomes)
    assert maximum_active == 1


def test_finnhub_canonical_statement_cannot_self_attest_after_payload_mutation() -> None:
    statement = "MSFT is Microsoft Corp, listed on NASDAQ."
    success = ToolSuccess(
        data={
            "symbol": "MSFT",
            "name": "A forged provider name",
            "exchange": "NASDAQ",
            "statements": [statement],
        },
        source={
            "provider": "finnhub",
            "reference": "company-profile:MSFT",
            "url": "https://finnhub.io/docs/api/company-profile2",
        },
        observed_at=NOW,
    )

    assert (
        _verify(_observation("market.get_company_profile", success), statement)
        is VerifierStatus.FAIL
    )


def test_optional_provider_settings_stay_secret_and_do_not_gate_conversation() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        leo_model="fixture/model",
        tavily_api_key="tavily-secret",
        slack_user_token="slack-user-secret",
    )

    assert settings.missing_for_conversation_providers() == ()
    assert "tavily-secret" not in repr(settings)
    assert "slack-user-secret" not in repr(settings)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("objective", "maximum_depth"),
    [
        ("Compare these", 0),
        (
            "Compare two unspecified options; ask exactly one concise clarifying question; "
            "do not research or use tools.",
            1,
        ),
    ],
)
async def test_live_short_ambiguity_completes_with_one_clarification_turn(
    objective: str,
    maximum_depth: int,
) -> None:
    calls = 0
    answer = "Which two options would you like me to compare?"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.host == "openrouter.test"
        payload = json.loads(request.content)
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        user_payload = json.loads(payload["messages"][1]["content"])
        assert (
            f"Depth envelope 0-{maximum_depth}; advisory clarify"
            in (user_payload["completion_contract"]["guidance"])
        )
        return httpx.Response(
            200,
            json={
                "id": "clarify-1",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [],
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
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == answer
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
    assert calls == 1
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    source_manifest = context_event.payload["source_manifest"]
    assert isinstance(source_manifest, dict)
    included = source_manifest["included_source_ids"]
    assert isinstance(included, list)
    assert any(
        isinstance(item, str)
        and item.startswith(f"deliberation-v1:recommended=clarify:depth=0-{maximum_depth}:")
        for item in included
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", ["NVDA price?", "What's NVDA trading at right now?"])
async def test_live_natural_quote_request_executes_one_constrained_tool_then_stops(
    objective: str,
) -> None:
    model_calls = 0
    finnhub_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, finnhub_calls
        if request.url.host == "finnhub.io":
            finnhub_calls += 1
            assert request.url.path == "/api/v1/quote"
            return httpx.Response(
                200,
                json={
                    "c": 181.25,
                    "d": 1.5,
                    "dp": 0.83,
                    "h": 183.0,
                    "l": 178.0,
                    "o": 179.0,
                    "pc": 179.75,
                    "t": int(datetime.now(UTC).timestamp()),
                },
            )
        model_calls += 1
        payload = json.loads(request.content)
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "market_get_quote"},
        }
        return httpx.Response(
            200,
            json={
                "id": "quote-1",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-quote",
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
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        finnhub_api_key="finnhub-key",
        finnhub_base_url="https://finnhub.io/api/v1",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == "NVDA is quoted at 181.25."
    # The coordinator accounts for the harness-owned canonical completion as a
    # second model boundary; only one external provider call was made.
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 1
    assert model_calls == finnhub_calls == 1


@pytest.mark.asyncio
async def test_live_unknown_equity_symbol_prefers_provider_neutral_quote_with_all_keys() -> None:
    market_timestamp = int(datetime.now(UTC).timestamp())
    provider_hosts: list[str] = []
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "finnhub.io":
            provider_hosts.append("finnhub")
            assert request.url.path == "/api/v1/quote"
            assert request.url.params["symbol"] == "PLTR"
            return httpx.Response(
                200,
                json={
                    "c": 25.0,
                    "d": 0.1,
                    "dp": 0.4,
                    "h": 25.2,
                    "l": 24.8,
                    "o": 24.9,
                    "pc": 24.9,
                    "t": market_timestamp,
                },
            )
        if request.url.host == "api.massive.com":
            provider_hosts.append("massive")
            assert request.url.path == "/v3/snapshot"
            assert request.url.params["ticker"] == "PLTR"
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": [
                        {
                            "ticker": "PLTR",
                            "type": "stocks",
                            "session": {
                                "price": 25.1,
                                "last_updated": market_timestamp * 1_000_000_000,
                            },
                        }
                    ],
                },
            )
        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        advertised = {item["function"]["name"] for item in payload["tools"]}
        assert "market_get_quote" in advertised
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "market_get_quote"},
        }
        return httpx.Response(
            200,
            json={
                "id": "pltr-quote-tool",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-pltr-quote",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_quote",
                                        "arguments": '{"symbol":"PLTR"}',
                                    },
                                }
                            ],
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
        finnhub_api_key="finnhub-key",
        tavily_api_key="tavily-key",
        exa_api_key="exa-key",
        alpha_vantage_api_key="alpha-key",
        massive_api_key="massive-key",
        ticker_layer_api_key="ticker-key",
        coingecko_api_key="coingecko-key",
        coin_market_cap_api_key="coinmarketcap-test-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="PLTR price?",
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == "PLTR is quoted at 25."
    assert provider_hosts == ["finnhub", "massive"]
    assert model_calls == 1


@pytest.mark.asyncio
async def test_live_unknown_equity_symbol_prefers_provider_neutral_profile_with_all_keys() -> None:
    model_calls = 0
    provider_hosts: list[str] = []
    statement = (
        "Finnhub reports PLTR as Palantir Technologies Inc., listed on NYSE, "
        "in industry Technology."
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "finnhub.io":
            provider_hosts.append("finnhub")
            assert request.url.path == "/api/v1/stock/profile2"
            assert request.url.params["symbol"] == "PLTR"
            return httpx.Response(
                200,
                json={
                    "ticker": "PLTR",
                    "name": "Palantir Technologies Inc.",
                    "exchange": "NYSE",
                    "finnhubIndustry": "Technology",
                },
            )
        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        if not observations:
            advertised = {item["function"]["name"] for item in payload["tools"]}
            assert "market_get_equity_profile" in advertised
            assert payload["tool_choice"] == {
                "type": "function",
                "function": {"name": "market_get_equity_profile"},
            }
            content = None
            tool_calls = [
                {
                    "id": "call-pltr-profile",
                    "type": "function",
                    "function": {
                        "name": "market_get_equity_profile",
                        "arguments": '{"symbol":"PLTR"}',
                    },
                }
            ]
        else:
            observation_id = next(
                item["id"] for item in observations if item["kind"] == "market.get_equity_profile"
            )
            content = json.dumps(
                {
                    "answer": statement,
                    "source_claims": [
                        {"statement": statement, "observation_ids": [observation_id]}
                    ],
                    "inferences": [],
                }
            )
            tool_calls = []
        return httpx.Response(
            200,
            json={
                "id": f"pltr-profile-{model_calls}",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": content,
                            "tool_calls": tool_calls,
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
        finnhub_api_key="finnhub-key",
        tavily_api_key="tavily-key",
        exa_api_key="exa-key",
        alpha_vantage_api_key="alpha-key",
        massive_api_key="massive-key",
        ticker_layer_api_key="ticker-key",
        coingecko_api_key="coingecko-key",
        coin_market_cap_api_key="coinmarketcap-test-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="PLTR company profile",
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == statement
    assert provider_hosts == ["finnhub"]
    assert model_calls == 2


@pytest.mark.asyncio
async def test_live_natural_earnings_question_uses_exact_bounded_summary_in_two_turns() -> None:
    objective = "Has Nvidia been beating or missing earnings expectations lately?"
    model_calls = 0
    finnhub_calls = 0
    items = [
        {"actual": 0.88, "estimate": 0.82, "period": "2026-06-30", "symbol": "NVDA"},
        {"actual": 0.76, "estimate": 0.79, "period": "2026-03-31", "symbol": "NVDA"},
        {"actual": 0.70, "estimate": 0.70, "period": "2025-12-31", "symbol": "NVDA"},
        {"actual": 0.65, "estimate": 0.61, "period": "2025-09-30", "symbol": "NVDA"},
    ]
    canonical = canonical_earnings_statements("NVDA", items)
    assert canonical is not None
    summary = canonical[0]

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, finnhub_calls
        if request.url.host == "finnhub.io":
            finnhub_calls += 1
            assert request.url.path == "/api/v1/stock/earnings"
            assert request.url.params["symbol"] == "NVDA"
            return httpx.Response(200, json=items)

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "market_get_earnings_surprises"},
        }
        return httpx.Response(
            200,
            json={
                "id": "earnings-live-shaped-1",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-earnings",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_earnings_surprises",
                                        "arguments": '{"symbol":"NVDA"}',
                                    },
                                }
                            ],
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
        finnhub_api_key="finnhub-key",
        finnhub_base_url="https://finnhub.io/api/v1",
        leo_max_model_turns=2,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == summary
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 1
    assert model_calls == finnhub_calls == 1
    assert len(result.claims) == 1
    assert result.claims[0].statement == summary
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)


@pytest.mark.asyncio
async def test_live_current_event_prompt_allows_semantic_multi_source_route() -> None:
    model_calls = 0
    finnhub_calls = 0
    published_timestamp = int((datetime.now(UTC) - timedelta(minutes=5)).timestamp())
    published_at = datetime.fromtimestamp(published_timestamp, tz=UTC).isoformat()
    statement = (
        f"On {published_at}, Example Wire reported for NVDA: New product update "
        "Source URL: https://example.org/news/nvda-update"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, finnhub_calls
        if request.url.host == "finnhub.io":
            finnhub_calls += 1
            assert request.url.path == "/api/v1/company-news"
            assert request.url.params["symbol"] == "NVDA"
            return httpx.Response(
                200,
                json=[
                    {
                        "datetime": published_timestamp,
                        "headline": "New product update",
                        "source": "Example Wire",
                        "url": "https://example.org/news/nvda-update",
                    }
                ],
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        if not observations:
            advertised = {item["function"]["name"] for item in payload["tools"]}
            assert payload["tool_choice"] == "auto"
            assert "web_search_tavily" in advertised
            assert "market_get_company_news" in advertised
            assert "advisory multi_source" in user_payload["completion_contract"]["guidance"]
            return httpx.Response(
                200,
                json={
                    "id": "current-event-tool",
                    "model": "fixture/model",
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-news",
                                        "type": "function",
                                        "function": {
                                            "name": "market_get_company_news",
                                            "arguments": '{"symbol":"NVDA"}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                },
            )

        observation_id = next(
            item["id"] for item in observations if item["kind"] == "market.get_company_news"
        )
        return httpx.Response(
            200,
            json={
                "id": "current-event-completion",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation_id],
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
        finnhub_api_key="finnhub-key",
        finnhub_base_url="https://finnhub.io/api/v1",
        tavily_api_key="tavily-key",
        exa_api_key=None,
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="What happened to NVDA today?",
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == statement
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 1
    assert model_calls == 2
    assert finnhub_calls == 1


@pytest.mark.asyncio
async def test_live_natural_web_question_uses_tavily_then_fetches_verified_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "leo.integrations.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    objective = "Find out what current web pages report about changes to NASA's Artemis II mission."
    empty_url = "https://example.net/artemis-ii-navigation"
    result_url = "https://example.org/artemis-ii-update"
    snippet = "A discovery snippet claims an Artemis II schedule update."
    statement = "NASA moved the Artemis II target launch window to April 2026."
    model_calls = 0
    tavily_calls = 0
    fetch_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, tavily_calls, fetch_calls
        if request.url.host == "api.tavily.com":
            tavily_calls += 1
            request_payload = json.loads(request.content)
            assert request_payload["include_answer"] is False
            assert request_payload["include_raw_content"] is False
            return httpx.Response(
                200,
                json={
                    "request_id": "tavily-natural-1",
                    "results": [
                        {
                            "title": "Artemis II navigation page",
                            "url": empty_url,
                            "content": "A high-scoring page with a long navigation shell.",
                            "score": 0.99,
                        },
                        {
                            "title": "Artemis II mission update",
                            "url": result_url,
                            "content": snippet,
                            "score": 0.97,
                        },
                    ],
                },
            )
        if request.url.host == "example.org":
            fetch_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=statement,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )
        if request.url.host == "example.net":
            fetch_calls += 1
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><script>" + "x" * 40_000,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        kinds = [item["kind"] for item in observations]
        advertised = {item["function"]["name"] for item in payload["tools"]}

        assert model_calls == 1
        assert kinds == [
            "web.search_tavily",
            "web.fetch_public_text",
        ]
        assert {"web_search_tavily", "web_fetch_public_text"}.issubset(advertised)
        fetch_id = next(
            item["id"]
            for item in observations
            if item["kind"] == "web.fetch_public_text" and item["data"]["truncated"] is False
        )
        content = json.dumps(
            {
                "answer": statement,
                "source_claims": [{"statement": statement, "observation_ids": [fetch_id]}],
                "inferences": [],
            }
        )
        return httpx.Response(
            200,
            json={
                "id": f"natural-web-{model_calls}",
                "model": "fixture/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        tavily_api_key="tavily-key",
        tavily_endpoint=("https://mcp.tavily.example/mcp?token=native-rest-must-not-use-this"),
        exa_api_key=None,
        finnhub_api_key=None,
        leo_max_model_turns=5,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == statement
    assert tuple(item.kind for item in result.observations) == (
        "web.search_tavily",
        "web.fetch_public_text",
    )
    assert len(result.claims) == 1
    assert result.observations[1].data["truncated"] is False
    assert result.observations[1].data["candidate_attempt_count"] == 2
    assert result.observations[1].data["failed_candidates"] == [
        {"url": empty_url, "code": "fetch_empty_content"}
    ]
    assert result.claims[0].observation_ids == (result.observations[1].id,)
    assert result.run.usage.model_calls == 3
    assert result.run.usage.tool_calls == 2
    assert not result.task.verifier_feedback
    assert model_calls == 1
    assert tavily_calls == 1
    assert fetch_calls == 2


@pytest.mark.asyncio
async def test_live_version_question_repairs_future_work_into_tavily_fetch_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "leo.integrations.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    objective = "What's one noteworthy change in Python 3.14?"
    community_url = "https://dev.to/example/python-314-overview"
    result_url = "https://docs.python.org/3.14/whatsnew/3.14.html"
    statement = "Python 3.14 adds deferred evaluation of annotations as a language feature."
    model_calls = 0
    tavily_calls = 0
    fetch_calls = 0
    fetched_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, tavily_calls, fetch_calls
        if request.url.host == "api.tavily.com":
            tavily_calls += 1
            search_payload = json.loads(request.content)
            assert search_payload["query"].endswith("official documentation primary source")
            assert search_payload["max_results"] == 5
            assert search_payload["search_depth"] == "advanced"
            return httpx.Response(
                200,
                json={
                    "request_id": "tavily-python-314",
                    "results": [
                        {
                            "title": "Python 3.14 release: what's new?",
                            "url": "https://www.youtube.com/watch?v=fixture",
                            "content": "A video result cannot provide article text.",
                            "score": 1.0,
                        },
                        {
                            "title": "A community overview of Python 3.14",
                            "url": community_url,
                            "content": "A community page summarizes Python 3.14.",
                            "score": 0.98,
                        },
                        {
                            "title": "What's New In Python 3.14",
                            "url": result_url,
                            "content": "Python 3.14 includes several language changes.",
                            "score": 0.99,
                        },
                    ],
                },
            )
        if request.url.host == "docs.python.org":
            fetch_calls += 1
            fetched_hosts.append(request.url.host)
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=statement,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )
        if request.url.host == "dev.to":
            fetch_calls += 1
            fetched_hosts.append(request.url.host)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>community navigation shell</body><script>" + "x" * 40_000,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )
        if request.url.host == "www.youtube.com":
            fetch_calls += 1
            fetched_hosts.append(request.url.host)
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><script>video application shell</script>",
                extensions={"leo_peer_ip": "93.184.216.34"},
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        advertised = {item["function"]["name"] for item in payload["tools"]}
        assert {"web_search_tavily", "web_fetch_public_text"}.issubset(advertised)
        kinds = [item["kind"] for item in user_payload["observations"]]
        if kinds == []:
            # This is the exact provider-shaped live failure: a fabricated
            # observation reference.  The trusted route must not invoke the
            # semantic provider until the evidence chain is complete.
            content = json.dumps(
                {
                    "answer": statement,
                    "source_claims": [
                        {
                            "statement": statement,
                            "observation_ids": ["obs-python314-copy-replace"],
                        }
                    ],
                    "inferences": [],
                }
            )
        elif kinds == ["web.search_tavily"]:
            # A second live-shaped bad decision cites discovery and promises a
            # future fetch.  This branch must likewise be unreachable.
            discovery_id = user_payload["observations"][0]["id"]
            content = json.dumps(
                {
                    "answer": "I'll open the selected result before answering.",
                    "source_claims": [
                        {
                            "statement": "Python 3.14 includes several language changes.",
                            "observation_ids": [discovery_id],
                        }
                    ],
                    "inferences": [],
                }
            )
        else:
            assert model_calls == 1
            assert kinds == ["web.search_tavily", "web.fetch_public_text"]
            fetch_id = next(
                item["id"]
                for item in user_payload["observations"]
                if item["kind"] == "web.fetch_public_text"
            )
            content = json.dumps(
                {
                    "answer": statement,
                    "source_claims": [{"statement": statement, "observation_ids": [fetch_id]}],
                    "inferences": [],
                }
            )
        return httpx.Response(
            200,
            json={
                "id": f"python-314-{model_calls}",
                "model": "fixture/model",
                "choices": [{"message": {"content": content, "tool_calls": []}}],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        tavily_api_key="tavily-key",
        exa_api_key=None,
        finnhub_api_key=None,
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == statement
    assert tuple(item.kind for item in result.observations) == (
        "web.search_tavily",
        "web.fetch_public_text",
    )
    assert result.run.usage.model_calls == 3
    assert result.run.usage.tool_calls == 2
    assert model_calls == 1
    assert tavily_calls == fetch_calls == 1
    assert fetched_hosts == ["docs.python.org"]
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
