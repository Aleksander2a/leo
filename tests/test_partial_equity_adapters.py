from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from leo.harness.models import (
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
)
from leo.integrations.alpha_vantage import (
    AlphaVantageCompanyProfileTool,
    AlphaVantageQuoteTool,
    AlphaVantageSymbolSearchTool,
)
from leo.integrations.fake import FixedClock
from leo.integrations.finnhub import FinnhubCompanyProfileTool, FinnhubQuoteTool
from leo.integrations.massive import (
    MassiveCompanyProfileTool,
    MassiveStockSnapshotTool,
    MassiveSymbolSearchTool,
)
from leo.integrations.tickerlayer import (
    TickerLayerCompanyProfileTool,
    TickerLayerStockSnapshotTool,
    TickerLayerSymbolSearchTool,
)

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
            actor_id="actor",
        ),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_finnhub_partial_quote_keeps_required_facts_and_bounds_missing_fields() -> None:
    timestamp = int(NOW.timestamp()) - 60
    payload = {
        "c": "230.50",
        "t": str(timestamp),
        "d": "not-a-number",
        "h": 232.0,
        "l": 0,
        "o": 229.0,
        "pc": None,
        "new_provider_field": {"ignored": True},
    }
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))

    async with httpx.AsyncClient(transport=transport) as client:
        outcome = await FinnhubQuoteTool(
            client=client,
            api_key="test-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["symbol"] == "AAPL"
    assert outcome.data["price"] == 230.5
    assert outcome.data["high"] == 232.0
    assert outcome.data["open"] == 229.0
    assert outcome.data["missing_fields"] == [
        "change",
        "percent_change",
        "low",
        "previous_close",
    ]
    assert "change" not in outcome.data
    assert "low" not in outcome.data
    assert outcome.source.reference == f"quote:AAPL:{timestamp}"


@pytest.mark.asyncio
async def test_equity_quote_adapters_report_missing_optional_fields_without_rejecting() -> None:
    timestamp_seconds = int(NOW.timestamp()) - 60

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.alphavantage.co":
            return httpx.Response(
                200,
                json={
                    "Global Quote": {
                        "01. symbol": "AAPL",
                        "02. open": "not-a-number",
                        "05. price": "230.50",
                        "06. volume": "3.5",
                        "07. latest trading day": "2026-08-22",
                        "10. change percent": "broken%",
                    }
                },
            )
        if request.url.host == "api.massive.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "ticker": "AAPL",
                            "type": "stocks",
                            "market_status": "\u0000",
                            "session": {
                                "price": 230.5,
                                "last_updated": timestamp_seconds * 1_000_000_000,
                                "change": "not-a-number",
                                "volume": -1,
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "symbol": "US:AAPL",
                "last_price": 230.5,
                "last_timestamp": timestamp_seconds * 1_000,
                "bid": "not-a-number",
                "bid_size": -1,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        alpha = await AlphaVantageQuoteTool(
            client=client,
            api_key="alpha-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())
        massive = await MassiveStockSnapshotTool(
            client=client,
            api_key="massive-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())
        ticker = await TickerLayerStockSnapshotTool(
            client=client,
            api_key="ticker-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(alpha, ToolSuccess)
    assert alpha.data["missing_fields"] == [
        "open",
        "high",
        "low",
        "previous_close",
        "change",
        "percent_change",
        "volume",
    ]
    assert isinstance(massive, ToolSuccess)
    assert massive.data["missing_fields"] == [
        "market_status",
        "change",
        "percent_change",
        "open",
        "high",
        "low",
        "previous_close",
        "volume",
        "provider_timeframe",
    ]
    assert isinstance(ticker, ToolSuccess)
    assert ticker.data["missing_fields"] == [
        "bid",
        "ask",
        "previous_close",
        "change",
        "percent_change",
        "bid_size",
        "ask_size",
        "last_size",
    ]
    assert alpha.data["price"] == massive.data["price"] == ticker.data["price"] == 230.5


@pytest.mark.asyncio
async def test_company_profiles_keep_partial_identity_facts_with_exact_missing_ledger() -> None:
    as_of = NOW.replace(second=0).isoformat().replace("+00:00", "Z")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.alphavantage.co":
            return httpx.Response(200, json={"Symbol": "AAPL", "Name": "Apple Inc."})
        if request.url.host == "api.massive.com":
            return httpx.Response(
                200,
                json={"results": {"ticker": "AAPL", "primary_exchange": "XNAS"}},
            )
        if request.url.host == "api.tickerlayer.com":
            return httpx.Response(
                200,
                json={
                    "symbol": "US:AAPL",
                    "base_symbol": "AAPL",
                    "industry": "Technology Hardware",
                    "as_of": as_of,
                },
            )
        return httpx.Response(
            200,
            json={"ticker": "AAPL", "finnhubIndustry": "Technology"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        alpha = await AlphaVantageCompanyProfileTool(
            client=client,
            api_key="alpha-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())
        massive = await MassiveCompanyProfileTool(
            client=client,
            api_key="massive-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())
        ticker = await TickerLayerCompanyProfileTool(
            client=client,
            api_key="ticker-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())
        finnhub = await FinnhubCompanyProfileTool(
            client=client,
            api_key="finnhub-key",
            clock=FixedClock(NOW),
        ).execute({"symbol": "AAPL"}, _context())

    assert isinstance(alpha, ToolSuccess)
    assert alpha.data["missing_fields"] == ["exchange", "industry"]
    assert alpha.data["statements"] == ["Alpha Vantage reports AAPL as Apple Inc."]
    assert isinstance(massive, ToolSuccess)
    assert massive.data["missing_fields"] == ["industry", "name"]
    assert massive.data["statements"] == ["Massive reports AAPL listed on XNAS."]
    assert isinstance(ticker, ToolSuccess)
    assert ticker.data["missing_fields"] == ["exchange", "name"]
    assert ticker.data["statements"] == [
        "TickerLayer reports AAPL in industry Technology Hardware."
    ]
    assert isinstance(finnhub, ToolSuccess)
    assert finnhub.data["missing_fields"] == ["exchange", "name"]
    assert finnhub.data["statements"] == ["AAPL is in Finnhub industry Technology."]


@pytest.mark.asyncio
async def test_symbol_searches_skip_bad_rows_and_report_bounded_rejection_counts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.alphavantage.co":
            return httpx.Response(
                200,
                json={
                    "bestMatches": [
                        {"1. symbol": "AAPL"},
                        {"1. symbol": "AAPL", "2. name": "Apple Inc."},
                    ]
                },
            )
        if request.url.host == "api.massive.com":
            return httpx.Response(
                200,
                json={
                    "results": [
                        "malformed",
                        {"ticker": "AAPL", "name": "Apple Inc."},
                        {"ticker": "BROKEN"},
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "symbols": [
                    {
                        "symbol": "GB:AAPL",
                        "base_symbol": "AAPL",
                        "market": "US",
                        "name": "Mismatched entity",
                    },
                    {
                        "symbol": "US:AAPL",
                        "base_symbol": "AAPL",
                        "market": "US",
                        "name": "Apple Inc.",
                    },
                ]
            },
        )

    arguments = {"query": "Apple", "limit": 5, "market": "US"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcomes = (
            await AlphaVantageSymbolSearchTool(
                client=client,
                api_key="alpha-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
            await MassiveSymbolSearchTool(
                client=client,
                api_key="massive-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
            await TickerLayerSymbolSearchTool(
                client=client,
                api_key="ticker-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
        )

    for outcome, expected_rejections in zip(outcomes, (1, 2, 1), strict=True):
        assert isinstance(outcome, ToolSuccess)
        assert outcome.data["result_count"] == 1
        assert outcome.data["rejected_result_count"] == expected_rejections
        assert outcome.data["results"] == [
            {
                "symbol": "AAPL",
                "provider_symbol": (
                    "US:AAPL" if outcome.source.provider == "ticker-layer" else "AAPL"
                ),
                "name": "Apple Inc.",
                **(
                    {"market": "US", "match_score": 0.9}
                    if outcome.source.provider == "ticker-layer"
                    else {}
                ),
            }
        ]


@pytest.mark.asyncio
async def test_symbol_searches_return_accounted_empty_success_when_every_row_is_bad() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        key = "bestMatches" if request.url.host == "www.alphavantage.co" else "results"
        if request.url.host == "api.tickerlayer.com":
            key = "symbols"
        return httpx.Response(200, json={key: [None]})

    arguments = {"query": "Apple", "limit": 5, "market": "US"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcomes = (
            await AlphaVantageSymbolSearchTool(
                client=client,
                api_key="alpha-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
            await MassiveSymbolSearchTool(
                client=client,
                api_key="massive-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
            await TickerLayerSymbolSearchTool(
                client=client,
                api_key="ticker-key",
                clock=FixedClock(NOW),
            ).execute(arguments, _context()),
        )

    for outcome in outcomes:
        assert isinstance(outcome, ToolSuccess)
        assert outcome.data["result_count"] == 0
        assert outcome.data["rejected_result_count"] == 1
        assert outcome.data["results"] == []
        assert outcome.data["statements"] == []


@pytest.mark.asyncio
async def test_partial_search_tolerance_does_not_weaken_duplicate_entity_guard() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.massive.com":
            row = {"ticker": "AAPL", "name": "Apple Inc."}
            return httpx.Response(200, json={"results": [None, row, row]})
        row = {
            "symbol": "US:AAPL",
            "base_symbol": "AAPL",
            "market": "US",
            "name": "Apple Inc.",
        }
        return httpx.Response(200, json={"symbols": [None, row, row]})

    arguments = {"query": "Apple", "limit": 5, "market": "US"}
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        massive = await MassiveSymbolSearchTool(
            client=client,
            api_key="massive-key",
            clock=FixedClock(NOW),
        ).execute(arguments, _context())
        ticker = await TickerLayerSymbolSearchTool(
            client=client,
            api_key="ticker-key",
            clock=FixedClock(NOW),
        ).execute(arguments, _context())

    assert isinstance(massive, ToolFailure)
    assert massive.code == "MASSIVE_SCHEMA_DRIFT"
    assert isinstance(ticker, ToolFailure)
    assert ticker.code == "TICKER_LAYER_SCHEMA_DRIFT"
