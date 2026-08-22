from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
from pydantic import JsonValue

from leo.capabilities.equity_descriptors import EQUITY_CAPABILITY_DESCRIPTORS
from leo.config import Settings
from leo.harness.equity_market import (
    canonical_equity_profile_statements,
    canonical_equity_quote_disagreement_statement,
    canonical_equity_quote_statement,
    canonical_equity_quote_time_skew_statement,
    canonical_equity_search_statements,
    equity_quote_agreement_status,
    valid_equity_quote_aggregate,
)
from leo.harness.models import (
    RunPhase,
    ScopeKey,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.ports import Tool
from leo.integrations.alpha_vantage import (
    AlphaVantageCompanyProfileTool,
    AlphaVantageQuoteTool,
    AlphaVantageSymbolSearchTool,
)
from leo.integrations.equity_composition import (
    build_equity_market_tools,
    resolve_alpha_vantage_rest_base_url,
    resolve_massive_rest_base_url,
)
from leo.integrations.equity_market import (
    EquityProfileRoute,
    EquityQuoteRoute,
    EquitySearchRoute,
    RedundantEquityProfileTool,
    RedundantEquityQuoteTool,
    RedundantEquitySymbolSearchTool,
)
from leo.integrations.fake import FixedClock
from leo.integrations.massive import (
    MassiveCompanyProfileTool,
    MassiveStockSnapshotTool,
    MassiveSymbolSearchTool,
)
from leo.integrations.provider_runtime import ProviderCallGate, ProviderGateRegistry
from leo.integrations.tickerlayer import (
    TickerLayerCompanyProfileTool,
    TickerLayerStockSnapshotTool,
    TickerLayerSymbolSearchTool,
)
from leo.live import _conversation_capability_catalog

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
FIXTURE_NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
FIXTURES = Path(__file__).parent / "fixtures" / "providers"


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
            actor_id="actor",
        ),
        run_id="run",
        tool_call_id="call",
    )


def _fixture(name: str) -> object:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_quote_adapters_use_exact_official_requests_and_provenance() -> None:
    alpha_payload = _fixture("alpha_vantage_quote.json")
    massive_payload = _fixture("massive_stock_snapshot.json")
    ticker_payload = _fixture("tickerlayer_stock_snapshot.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.alphavantage.co":
            assert request.url.path == "/query"
            assert request.url.params["function"] == "GLOBAL_QUOTE"
            assert request.url.params["symbol"] == "AAPL"
            assert request.url.params["apikey"] == "alpha-test-key"
            assert "authorization" not in request.headers
            return httpx.Response(200, json=alpha_payload)
        if request.url.host == "api.massive.com":
            assert request.url.path == "/v3/snapshot"
            assert request.url.params == httpx.QueryParams(
                {"ticker": "AAPL", "type": "stocks", "limit": "1"}
            )
            assert request.headers["authorization"] == "Bearer massive-test-key"
            return httpx.Response(200, json=massive_payload)
        assert request.url.host == "api.tickerlayer.com"
        assert request.url.path == "/stocks/snapshot/US:AAPL"
        assert request.headers["x-api-key"] == "ticker-test-key"
        return httpx.Response(200, json=ticker_payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcomes = (
            await AlphaVantageQuoteTool(
                client=client,
                api_key="alpha-test-key",
                clock=FixedClock(FIXTURE_NOW),
            ).execute({"symbol": "AAPL"}, _context()),
            await MassiveStockSnapshotTool(
                client=client,
                api_key="massive-test-key",
                clock=FixedClock(FIXTURE_NOW),
            ).execute({"symbol": "AAPL"}, _context()),
            await TickerLayerStockSnapshotTool(
                client=client,
                api_key="ticker-test-key",
                clock=FixedClock(FIXTURE_NOW),
            ).execute({"symbol": "AAPL"}, _context()),
        )

    assert all(isinstance(outcome, ToolSuccess) for outcome in outcomes)
    alpha, massive, ticker = outcomes
    assert isinstance(alpha, ToolSuccess)
    assert isinstance(massive, ToolSuccess)
    assert isinstance(ticker, ToolSuccess)
    assert alpha.data["price"] == 230.5
    assert alpha.data["data_freshness"] == "end_of_day"
    assert alpha.source.reference == "global-quote:AAPL:2026-08-20"
    assert massive.data["price"] == 230.5
    assert massive.source.reference == "snapshot:AAPL:1787227200000000000"
    assert ticker.data["data_provenance"] == "derived_non_exchange_indicative"
    assert ticker.source.reference == "stock-snapshot:US:AAPL:1787227200000"
    for outcome in outcomes:
        assert isinstance(outcome, ToolSuccess)
        assert outcome.data["statements"] == [canonical_equity_quote_statement(outcome.data)]
    alpha_statement = canonical_equity_quote_statement(alpha.data)
    massive_statement = canonical_equity_quote_statement(massive.data)
    ticker_statement = canonical_equity_quote_statement(ticker.data)
    assert alpha_statement is not None and "end-of-day historical data" in alpha_statement
    assert massive_statement is not None and "provider-plan-dependent" in massive_statement
    assert (
        ticker_statement is not None
        and "derived, non-exchange, indicative data" in ticker_statement
    )


@pytest.mark.asyncio
async def test_credential_bearing_equity_requests_never_follow_client_redirects() -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"location": "https://untrusted.invalid/collect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        tools: tuple[Tool, ...] = (
            AlphaVantageQuoteTool(
                client=client, api_key="alpha-test-key", clock=FixedClock(FIXTURE_NOW)
            ),
            MassiveStockSnapshotTool(
                client=client, api_key="massive-test-key", clock=FixedClock(FIXTURE_NOW)
            ),
            TickerLayerStockSnapshotTool(
                client=client, api_key="ticker-test-key", clock=FixedClock(FIXTURE_NOW)
            ),
        )
        outcomes = [
            await tool.execute(tool.validate({"symbol": "AAPL"}), _context()) for tool in tools
        ]

    assert all(isinstance(outcome, ToolFailure) for outcome in outcomes)
    assert requested_hosts == [
        "www.alphavantage.co",
        "api.massive.com",
        "api.tickerlayer.com",
    ]


@pytest.mark.asyncio
async def test_alpha_vantage_daily_allowance_fails_fast_and_rolls_at_utc_day() -> None:
    clock = FixedClock(FIXTURE_NOW)
    gate = ProviderCallGate(
        provider="alpha_vantage",
        clock=clock,
        max_calls_per_minute=10,
        max_calls_per_day=1,
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_fixture("alpha_vantage_quote.json"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = AlphaVantageQuoteTool(
            client=client,
            api_key="alpha-test-key",
            clock=clock,
            gate=gate,
        )
        first = await tool.execute({"symbol": "AAPL"}, _context())
        rejected = await tool.execute({"symbol": "AAPL"}, _context())
        clock.advance(seconds=86_400)
        after_rollover = await tool.execute({"symbol": "AAPL"}, _context())

    assert isinstance(first, ToolSuccess)
    assert isinstance(rejected, ToolFailure)
    assert rejected.code == "ALPHA_VANTAGE_LOCAL_DAILY_RATE_LIMIT"
    assert isinstance(after_rollover, ToolSuccess)
    assert calls == 2
    health = await gate.snapshot()
    assert health.calls_in_day == 1
    assert health.remaining_local_daily_calls == 0


@pytest.mark.asyncio
async def test_tickerlayer_monthly_allowance_rolls_and_quota_429_is_credit_free() -> None:
    clock = FixedClock(datetime(2026, 8, 31, 23, 59, tzinfo=UTC))
    gate = ProviderCallGate(
        provider="ticker_layer",
        clock=clock,
        max_calls_per_minute=10,
        max_calls_per_month=1,
    )
    calls = 0

    def search_handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={
                "symbols": [
                    {
                        "symbol": "US:AAPL",
                        "base_symbol": "AAPL",
                        "market": "US",
                        "name": "Apple Inc.",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(search_handler)) as client:
        tool = TickerLayerSymbolSearchTool(
            client=client,
            api_key="ticker-test-key",
            clock=clock,
            gate=gate,
        )
        assert isinstance(
            await tool.execute({"query": "Apple", "limit": 5, "market": "US"}, _context()),
            ToolSuccess,
        )
        rejected = await tool.execute({"query": "Apple", "limit": 5, "market": "US"}, _context())
        clock.advance(seconds=120)
        assert isinstance(
            await tool.execute({"query": "Apple", "limit": 5, "market": "US"}, _context()),
            ToolSuccess,
        )
    assert isinstance(rejected, ToolFailure)
    assert rejected.code == "TICKER_LAYER_LOCAL_MONTHLY_RATE_LIMIT"
    assert calls == 2

    quota_gate = ProviderCallGate(provider="ticker_layer", clock=FixedClock(NOW))
    quota_transport = httpx.MockTransport(
        lambda _request: httpx.Response(
            429,
            json={"error": "REST_QUOTA_EXCEEDED", "retryAfterMs": 1_000},
        )
    )
    async with httpx.AsyncClient(transport=quota_transport) as client:
        outcome = await TickerLayerStockSnapshotTool(
            client=client,
            api_key="ticker-test-key",
            clock=FixedClock(NOW),
            gate=quota_gate,
        ).execute({"symbol": "AAPL"}, _context())
    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "TICKER_LAYER_MONTHLY_QUOTA_EXHAUSTED"
    assert (await quota_gate.snapshot()).provider_credits_used == 0


@pytest.mark.asyncio
async def test_search_and_profile_adapters_emit_exact_canonical_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.alphavantage.co":
            if request.url.params["function"] == "SYMBOL_SEARCH":
                return httpx.Response(
                    200,
                    json={
                        "bestMatches": [
                            {
                                "1. symbol": "AAPL",
                                "2. name": "Apple Inc.",
                                "3. type": "Equity",
                                "4. region": "United States",
                                "8. currency": "USD",
                                "9. matchScore": "1.0000",
                            }
                        ]
                    },
                )
            return httpx.Response(
                200,
                json={
                    "Symbol": "AAPL",
                    "Name": "Apple Inc.",
                    "Exchange": "NASDAQ",
                    "Industry": "Consumer Electronics",
                    "Sector": "Technology",
                    "Country": "USA",
                    "Currency": "USD",
                },
            )
        if request.url.host == "api.massive.com":
            if request.url.path == "/v3/reference/tickers":
                return httpx.Response(
                    200,
                    json={
                        "status": "OK",
                        "request_id": "request-search",
                        "results": [
                            {
                                "ticker": "AAPL",
                                "name": "Apple Inc.",
                                "market": "stocks",
                                "locale": "us",
                                "primary_exchange": "XNAS",
                                "currency_name": "usd",
                                "active": True,
                            }
                        ],
                    },
                )
            return httpx.Response(
                200,
                json={
                    "status": "OK",
                    "results": {
                        "ticker": "AAPL",
                        "name": "Apple Inc.",
                        "primary_exchange": "XNAS",
                        "sic_description": "Electronic Computers",
                        "locale": "us",
                        "currency_name": "usd",
                    },
                },
            )
        if request.url.path == "/stocks/symbols":
            return httpx.Response(
                200,
                json={
                    "symbols": [
                        {
                            "symbol": "US:AAPL",
                            "base_symbol": "AAPL",
                            "market": "US",
                            "name": "Apple Inc.",
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "symbol": "US:AAPL",
                "base_symbol": "AAPL",
                "company": "Apple Inc.",
                "exchange": "NASDAQ",
                "industry": "Consumer Electronics",
                "sector": "Technology",
                "country": "US",
                "as_of": "2026-08-22T11:00:00Z",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        searches: tuple[Tool, ...] = (
            AlphaVantageSymbolSearchTool(
                client=client, api_key="alpha-test-key", clock=FixedClock(NOW)
            ),
            MassiveSymbolSearchTool(
                client=client, api_key="massive-test-key", clock=FixedClock(NOW)
            ),
            TickerLayerSymbolSearchTool(
                client=client, api_key="ticker-test-key", clock=FixedClock(NOW)
            ),
        )
        profiles: tuple[Tool, ...] = (
            AlphaVantageCompanyProfileTool(
                client=client, api_key="alpha-test-key", clock=FixedClock(NOW)
            ),
            MassiveCompanyProfileTool(
                client=client, api_key="massive-test-key", clock=FixedClock(NOW)
            ),
            TickerLayerCompanyProfileTool(
                client=client, api_key="ticker-test-key", clock=FixedClock(NOW)
            ),
        )
        search_outcomes = [
            await tool.execute(
                tool.validate({"query": "Apple", "limit": 5, "market": "US"}),
                _context(),
            )
            for tool in searches
        ]
        profile_outcomes = [
            await tool.execute(tool.validate({"symbol": "AAPL", "market": "US"}), _context())
            for tool in profiles
        ]

    for outcome in search_outcomes:
        assert isinstance(outcome, ToolSuccess)
        assert outcome.data["statements"] == list(
            canonical_equity_search_statements(outcome.data) or ()
        )
        assert outcome.data["result_count"] == 1
    for outcome in profile_outcomes:
        assert isinstance(outcome, ToolSuccess)
        assert outcome.data["statements"] == list(
            canonical_equity_profile_statements(outcome.data) or ()
        )


@pytest.mark.parametrize(
    ("factory", "base_url"),
    [
        (
            lambda client: AlphaVantageQuoteTool(
                client=client,
                api_key="test-key",
                clock=FixedClock(NOW),
                base_url="https://untrusted.invalid/query",
            ),
            "https://untrusted.invalid/query",
        ),
        (
            lambda client: MassiveStockSnapshotTool(
                client=client,
                api_key="test-key",
                clock=FixedClock(NOW),
                base_url="https://api.massive.com:444",
            ),
            "https://api.massive.com:444",
        ),
        (
            lambda client: TickerLayerStockSnapshotTool(
                client=client,
                api_key="test-key",
                clock=FixedClock(NOW),
                base_url="https://api.tickerlayer.com/redirect",
            ),
            "https://api.tickerlayer.com/redirect",
        ),
    ],
)
def test_equity_adapters_reject_untrusted_or_noncanonical_rest_bases(
    factory: Callable[[httpx.AsyncClient], Tool],
    base_url: str,
) -> None:
    del base_url
    with pytest.raises(ValueError, match="official credential-free REST URL"):
        factory(httpx.AsyncClient())


def test_mcp_endpoint_settings_never_become_credential_bearing_rest_bases() -> None:
    assert (
        resolve_alpha_vantage_rest_base_url(
            configured_endpoint="https://mcp.alphavantage.co/mcp",
            configured_legacy_endpoint="https://untrusted.invalid/query",
        )
        == "https://www.alphavantage.co/query"
    )
    assert (
        resolve_alpha_vantage_rest_base_url(
            configured_endpoint="https://www.alphavantage.co/query",
            configured_legacy_endpoint=None,
        )
        == "https://www.alphavantage.co/query"
    )
    assert (
        resolve_massive_rest_base_url(configured_endpoint="https://mcp.massive.com/mcp")
        == "https://api.massive.com"
    )
    assert (
        resolve_massive_rest_base_url(configured_endpoint="https://api.massive.com")
        == "https://api.massive.com"
    )


@pytest.mark.asyncio
async def test_equity_composition_has_unique_ids_and_process_owned_provider_gates() -> None:
    settings = Settings(
        _env_file=None,
        alpha_vantage_api_key="alpha-test-key",
        massive_api_key="massive-test-key",
        ticker_layer_api_key="ticker-test-key",
        finnhub_api_key="finnhub-test-key",
    )
    clock = FixedClock(NOW)
    registry = ProviderGateRegistry(clock)
    async with httpx.AsyncClient() as client:
        tools = build_equity_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=registry,
        )

    names = tuple(tool.spec.name for tool in tools)
    assert len(names) == len(set(names))
    assert names.count("market.get_quote") == 1
    assert {
        "market.get_quote_alpha_vantage",
        "market.get_quote_finnhub",
        "market.get_quote_massive",
        "market.get_quote_ticker_layer",
        "market.search_equity_symbols",
        "market.get_equity_profile",
    }.issubset(names)
    aggregate = next(tool for tool in tools if isinstance(tool, RedundantEquityQuoteTool))
    assert aggregate.provider_order == (
        "finnhub",
        "massive",
        "ticker-layer",
        "alpha-vantage",
    )
    assert registry.registered_providers == (
        "alpha_vantage",
        "finnhub",
        "massive",
        "ticker_layer",
    )


@pytest.mark.asyncio
async def test_finnhub_only_quote_uses_truthful_aggregate_catalog_descriptor() -> None:
    settings = Settings(_env_file=None, finnhub_api_key="finnhub-test-key")
    clock = FixedClock(NOW)
    registry = ProviderGateRegistry(clock)
    async with httpx.AsyncClient() as client:
        tools = build_equity_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=registry,
        )

    names = tuple(tool.spec.name for tool in tools)
    assert names.count("market.get_quote") == 1
    assert names.count("market.get_quote_finnhub") == 1
    assert len(names) == len(set(names))
    aggregate_tool = next(tool for tool in tools if isinstance(tool, RedundantEquityQuoteTool))
    assert aggregate_tool.provider_order == ("finnhub",)

    catalog = _conversation_capability_catalog(list(tools))
    aggregate = catalog.get("market.get_quote")
    direct = catalog.get("market.get_quote_finnhub")
    assert aggregate.provider == "equity-corroboration"
    assert "deterministic_failover" in aggregate.verification_expectations
    assert direct.provider == "finnhub"


def test_equity_descriptors_cover_stable_and_direct_quote_capabilities() -> None:
    assert EQUITY_CAPABILITY_DESCRIPTORS["market.get_quote"].freshness_seconds == 900
    assert {
        "market.get_quote",
        "market.get_quote_alpha_vantage",
        "market.get_quote_finnhub",
        "market.get_quote_massive",
        "market.get_quote_ticker_layer",
        "market.search_equity_symbols",
        "market.get_equity_profile",
    }.issubset(EQUITY_CAPABILITY_DESCRIPTORS)
    assert {
        EQUITY_CAPABILITY_DESCRIPTORS[name].rate_limit_per_minute
        for name in (
            "market.get_quote_ticker_layer",
            "market.search_symbols_ticker_layer",
            "market.get_company_profile_ticker_layer",
        )
    } == {60}


class _ScriptedTool:
    def __init__(
        self,
        *outcomes: ToolOutcome | Exception,
    ) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0
        self._spec = ToolSpec(
            name=f"test.equity_provider_{id(self)}",
            description="Return one scripted provider outcome.",
            domain="MARKET",
            input_schema={"type": "object"},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return arguments

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del arguments, context
        self.calls += 1
        if not self._outcomes:
            raise AssertionError("unexpected scripted provider call")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _HealthScriptedTool(_ScriptedTool):
    def __init__(
        self,
        *outcomes: ToolOutcome | Exception,
        gate: ProviderCallGate,
    ) -> None:
        super().__init__(*outcomes)
        self._gate = gate

    async def provider_health(self):  # type: ignore[no-untyped-def]
        return await self._gate.snapshot()


def _quote_success(
    provider: str,
    *,
    price: float,
    observed_at: datetime,
) -> ToolSuccess:
    references = {
        "finnhub": f"quote:AAPL:{int(observed_at.timestamp())}",
        "alpha-vantage": f"global-quote:AAPL:{observed_at.date().isoformat()}",
        "massive": f"snapshot:AAPL:{int(observed_at.timestamp() * 1_000_000_000)}",
        "ticker-layer": f"stock-snapshot:US:AAPL:{int(observed_at.timestamp() * 1_000)}",
    }
    data: dict[str, JsonValue] = {
        "provider": provider,
        "symbol": "AAPL",
        "price": price,
        "as_of": observed_at.isoformat(),
    }
    if provider == "alpha-vantage":
        data["data_freshness"] = "end_of_day"
        data["market_data_entitlement"] = "historical"
    elif provider == "massive":
        data["data_freshness"] = "provider_plan_dependent"
    elif provider == "ticker-layer":
        data["data_provenance"] = "derived_non_exchange_indicative"
    statement = canonical_equity_quote_statement(data)
    assert statement is not None
    data["statements"] = [statement]
    return ToolSuccess(
        data=data,
        source=SourceRef(provider=provider, reference=references[provider]),
        observed_at=observed_at,
        expires_at=NOW + timedelta(minutes=10),
    )


@pytest.mark.asyncio
async def test_quote_router_is_bounded_deterministic_and_preserves_failures() -> None:
    finnhub = _ScriptedTool(
        ToolFailure(code="FINNHUB_RATE_LIMITED", retryable=True, safe_message="limited")
    )
    alpha = _ScriptedTool(
        _quote_success("alpha-vantage", price=230.0, observed_at=NOW - timedelta(seconds=20))
    )
    massive = _ScriptedTool(
        _quote_success("massive", price=240.0, observed_at=NOW - timedelta(seconds=10))
    )
    ticker = _ScriptedTool(_quote_success("ticker-layer", price=239.0, observed_at=NOW))
    tool = RedundantEquityQuoteTool(
        routes=(
            EquityQuoteRoute("finnhub", finnhub),
            EquityQuoteRoute("alpha-vantage", alpha),
            EquityQuoteRoute("massive", massive),
            EquityQuoteRoute("ticker-layer", ticker),
        ),
        clock=FixedClock(NOW),
        agreement_threshold_percent=1.0,
    )

    outcome = await tool.execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["selected_provider"] == "massive"
    assert outcome.data["provider_attempt_count"] == 3
    assert outcome.data["provider_success_count"] == 2
    assert outcome.data["provider_failure_count"] == 1
    assert outcome.data["provider_health_skip_count"] == 0
    assert outcome.data["provider_skipped_count"] == 1
    assert outcome.data["agreement_status"] == "disagree"
    assert outcome.data["temporally_aligned"] is True
    assert outcome.data["corroborated"] is False
    assert equity_quote_agreement_status(outcome.data) == "disagree"
    statements = outcome.data["statements"]
    assert isinstance(statements, list)
    assert canonical_equity_quote_disagreement_statement(outcome.data) in statements
    assert [finnhub.calls, alpha.calls, massive.calls, ticker.calls] == [1, 1, 1, 0]
    quotes = outcome.data["provider_quotes"]
    assert isinstance(quotes, list)
    assert quotes == [
        {
            "provider": "alpha-vantage",
            "reference": "global-quote:AAPL:2026-08-22",
            "price": 230.0,
            "as_of": (NOW - timedelta(seconds=20)).isoformat(),
            "expires_at": (NOW + timedelta(minutes=10)).isoformat(),
        },
        {
            "provider": "massive",
            "reference": (
                f"snapshot:AAPL:{int((NOW - timedelta(seconds=10)).timestamp() * 1_000_000_000)}"
            ),
            "price": 240.0,
            "as_of": (NOW - timedelta(seconds=10)).isoformat(),
            "expires_at": (NOW + timedelta(minutes=10)).isoformat(),
        },
    ]
    assert valid_equity_quote_aggregate(
        outcome.data,
        source_provider=outcome.source.provider,
        source_reference=outcome.source.reference,
        observed_at=outcome.observed_at,
        expires_at=outcome.expires_at,
    )


@pytest.mark.asyncio
async def test_quote_router_uses_earliest_peer_expiry_and_rejects_expiry_tampering() -> None:
    alpha_expiry = NOW + timedelta(minutes=3)
    massive_expiry = NOW + timedelta(minutes=12)
    alpha = _quote_success(
        "alpha-vantage", price=230.0, observed_at=NOW - timedelta(seconds=20)
    ).model_copy(update={"expires_at": alpha_expiry})
    massive = _quote_success(
        "massive", price=230.5, observed_at=NOW - timedelta(seconds=10)
    ).model_copy(update={"expires_at": massive_expiry})
    outcome = await RedundantEquityQuoteTool(
        routes=(
            EquityQuoteRoute("alpha-vantage", _ScriptedTool(alpha)),
            EquityQuoteRoute("massive", _ScriptedTool(massive)),
        ),
        clock=FixedClock(NOW),
    ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.expires_at == alpha_expiry
    quote_rows = outcome.data["provider_quotes"]
    assert isinstance(quote_rows, list)
    assert [row["expires_at"] for row in quote_rows if isinstance(row, dict)] == [
        alpha_expiry.isoformat(),
        massive_expiry.isoformat(),
    ]
    assert valid_equity_quote_aggregate(
        outcome.data,
        source_provider=outcome.source.provider,
        source_reference=outcome.source.reference,
        observed_at=outcome.observed_at,
        expires_at=outcome.expires_at,
    )

    tampered: dict[str, JsonValue] = dict(outcome.data)
    tampered_rows: list[JsonValue] = []
    for index, row in enumerate(quote_rows):
        assert isinstance(row, dict)
        copied: dict[str, JsonValue] = dict(row)
        if index == 0:
            copied["expires_at"] = None
        tampered_rows.append(copied)
    tampered["provider_quotes"] = tampered_rows
    assert not valid_equity_quote_aggregate(
        tampered,
        source_provider=outcome.source.provider,
        source_reference=outcome.source.reference,
        observed_at=outcome.observed_at,
        expires_at=outcome.expires_at,
    )
    assert not valid_equity_quote_aggregate(
        outcome.data,
        source_provider=outcome.source.provider,
        source_reference=outcome.source.reference,
        observed_at=outcome.observed_at,
        expires_at=massive_expiry,
    )


@pytest.mark.asyncio
async def test_quote_router_marks_price_agreement_with_timestamp_skew_uncorroborated() -> None:
    alpha = _ScriptedTool(
        _quote_success("alpha-vantage", price=230.0, observed_at=NOW - timedelta(hours=12))
    )
    massive = _ScriptedTool(
        _quote_success("massive", price=230.5, observed_at=NOW - timedelta(seconds=10))
    )
    outcome = await RedundantEquityQuoteTool(
        routes=(
            EquityQuoteRoute("alpha-vantage", alpha),
            EquityQuoteRoute("massive", massive),
        ),
        clock=FixedClock(NOW),
        agreement_threshold_percent=1.0,
        max_corroboration_skew_seconds=900,
    ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["agreement_status"] == "time_skewed"
    assert outcome.data["temporally_aligned"] is False
    assert outcome.data["corroborated"] is False
    assert equity_quote_agreement_status(outcome.data) == "time_skewed"
    statements = outcome.data["statements"]
    assert isinstance(statements, list)
    assert canonical_equity_quote_time_skew_statement(outcome.data) in statements


@pytest.mark.asyncio
async def test_quote_router_contains_unexpected_errors_and_health_skips() -> None:
    clock = FixedClock(NOW)
    health_gate = ProviderCallGate(provider="finnhub", clock=clock)
    await health_gate.record_failure(
        "FINNHUB_LOCAL_RATE_LIMIT", rate_limited=True, retry_after_seconds=60
    )
    unhealthy = _HealthScriptedTool(
        _quote_success("finnhub", price=228.0, observed_at=NOW), gate=health_gate
    )
    broken = _ScriptedTool(RuntimeError("credential-bearing internal detail"))
    massive = _ScriptedTool(
        _quote_success("massive", price=230.0, observed_at=NOW - timedelta(seconds=5))
    )
    outcome = await RedundantEquityQuoteTool(
        routes=(
            EquityQuoteRoute("finnhub", unhealthy),
            EquityQuoteRoute("alpha-vantage", broken),
            EquityQuoteRoute("massive", massive),
        ),
        clock=clock,
        corroboration_target=1,
    ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert unhealthy.calls == 0
    assert broken.calls == 1
    assert massive.calls == 1
    assert outcome.data["provider_failure_count"] == 1
    assert outcome.data["provider_health_skip_count"] == 1
    assert "credential-bearing" not in str(outcome.data)
    attempts = outcome.data["provider_attempts"]
    assert isinstance(attempts, list)
    assert isinstance(attempts[0], dict)
    assert isinstance(attempts[1], dict)
    assert attempts[0]["status"] == "skipped"
    assert attempts[1]["code"] == "EQUITY_PROVIDER_UNEXPECTED_ERROR"


@pytest.mark.asyncio
async def test_quote_router_all_failed_keeps_content_free_provider_accounting() -> None:
    first = _ScriptedTool(RuntimeError("do-not-leak"))
    second = _ScriptedTool(
        ToolFailure(
            code="MASSIVE_ENTITLEMENT_REQUIRED",
            safe_message="plan does not include snapshot",
        )
    )
    outcome = await RedundantEquityQuoteTool(
        routes=(
            EquityQuoteRoute("alpha-vantage", first),
            EquityQuoteRoute("massive", second),
        ),
        clock=FixedClock(NOW),
    ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "EQUITY_QUOTE_ALL_PROVIDERS_FAILED"
    assert "alpha-vantage:EQUITY_PROVIDER_UNEXPECTED_ERROR" in outcome.safe_message
    assert "massive:MASSIVE_ENTITLEMENT_REQUIRED" in outcome.safe_message
    assert "do-not-leak" not in outcome.safe_message


def _search_success(provider: str) -> ToolSuccess:
    from leo.harness.equity_market import equity_query_hash

    query_hash = equity_query_hash("Apple")
    references = {
        "alpha-vantage": f"symbol-search:{query_hash}",
        "massive": f"ticker-search:{query_hash}",
        "ticker-layer": f"stock-symbol-search:US:{query_hash}",
    }
    provider_symbol = "US:AAPL" if provider == "ticker-layer" else "AAPL"
    data: dict[str, JsonValue] = {
        "provider": provider,
        "query": "Apple",
        "query_hash": query_hash,
        "requested_market": "US",
        "result_count": 1,
        "results": [
            {
                "symbol": "AAPL",
                "provider_symbol": provider_symbol,
                "name": "Apple Inc.",
            }
        ],
    }
    data["statements"] = list(canonical_equity_search_statements(data) or ())
    return ToolSuccess(
        data=data,
        source=SourceRef(provider=provider, reference=references[provider]),
        observed_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


def _profile_success(provider: str) -> ToolSuccess:
    provider_symbol = "US:AAPL" if provider == "ticker-layer" else "AAPL"
    references = {
        "alpha-vantage": "company-overview:AAPL",
        "massive": "ticker-overview:AAPL",
        "ticker-layer": "stock-fundamentals:US:AAPL:2026-08-22T11:00:00Z",
    }
    data: dict[str, JsonValue] = {
        "provider": provider,
        "symbol": "AAPL",
        "provider_symbol": provider_symbol,
        "name": "Apple Inc.",
        "exchange": "NASDAQ",
        "industry": "Consumer Electronics",
        "as_of": NOW.isoformat(),
    }
    data["statements"] = list(canonical_equity_profile_statements(data) or ())
    return ToolSuccess(
        data=data,
        source=SourceRef(provider=provider, reference=references[provider]),
        observed_at=NOW,
        expires_at=NOW + timedelta(hours=1),
    )


@pytest.mark.asyncio
async def test_search_and_profile_wrappers_fail_over_with_explicit_attempt_ledgers() -> None:
    denied = ToolFailure(
        code="MASSIVE_ENTITLEMENT_REQUIRED",
        safe_message="provider-local permission denial",
    )
    search_last = _ScriptedTool(_search_success("alpha-vantage"))
    search = await RedundantEquitySymbolSearchTool(
        routes=(
            EquitySearchRoute("massive", _ScriptedTool(denied)),
            EquitySearchRoute("ticker-layer", _ScriptedTool(RuntimeError("hidden"))),
            EquitySearchRoute("alpha-vantage", search_last),
        ),
        clock=FixedClock(NOW),
    ).execute({"query": "Apple", "limit": 5, "market": "US"}, _context())
    assert isinstance(search, ToolSuccess)
    assert search.data["selected_provider"] == "alpha-vantage"
    assert search.data["provider_failure_count"] == 2
    assert search.data["provider_attempt_count"] == 3

    profile_last = _ScriptedTool(_profile_success("ticker-layer"))
    profile = await RedundantEquityProfileTool(
        routes=(
            EquityProfileRoute("massive", _ScriptedTool(denied)),
            EquityProfileRoute("ticker-layer", profile_last),
            EquityProfileRoute("alpha-vantage", _ScriptedTool(_profile_success("alpha-vantage"))),
        ),
        clock=FixedClock(NOW),
    ).execute({"symbol": "AAPL", "market": "US"}, _context())
    assert isinstance(profile, ToolSuccess)
    assert profile.data["selected_provider"] == "ticker-layer"
    assert profile.data["provider_failure_count"] == 1
    assert profile.data["provider_skipped_count"] == 1
    assert profile.data["provider_health_skip_count"] == 0


def test_aggregate_agreement_contract_fails_closed_on_tampering() -> None:
    data: dict[str, JsonValue] = {
        "symbol": "AAPL",
        "provider_success_count": 2,
        "agreement_threshold_percent": 1.0,
        "price_disagreement_percent": 0.5,
        "corroboration_skew_threshold_seconds": 900,
        "freshness_spread_seconds": 30,
        "temporally_aligned": True,
        "agreement_status": "agree",
        "corroborated": True,
    }
    assert equity_quote_agreement_status(data) == "agree"
    data["corroborated"] = False
    assert equity_quote_agreement_status(data) is None
