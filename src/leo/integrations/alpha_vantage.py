"""Typed Alpha Vantage equity adapters over the bounded raw HTTP API.

Canonical contract source: https://www.alphavantage.co/documentation/
No Alpha Vantage Agent or third-party agent framework is used.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import cast

import httpx
from pydantic import JsonValue

from leo.agent.contracts import (
    Clock,
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRetryPolicy,
    ToolSpec,
    ToolSuccess,
)
from leo.integrations.equity_market import (
    EquityProfileArguments,
    EquitySearchArguments,
    bounded_provider_text,
    finite_provider_number,
)
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRejected,
    bounded_retry_after,
)
from leo.providers.equity import (
    canonical_equity_profile_statements,
    canonical_equity_quote_statement,
    canonical_equity_search_statements,
    equity_query_hash,
)
from leo.providers.health import ProviderHealthSnapshot

_DEFAULT_QUERY_URL = "https://www.alphavantage.co/query"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_SEARCH_MATCHES = 1_000


class AlphaVantageQuoteTool:
    """Normalize one GLOBAL_QUOTE response into Leo's quote contract."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_QUERY_URL,
        gate: ProviderCallGate | None = None,
        max_quote_age_seconds: int = 604_800,
        max_future_skew_seconds: int = 86_400,
        evidence_ttl_seconds: int = 900,
    ) -> None:
        self._reader = _AlphaVantageReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        if max_quote_age_seconds < 1 or max_future_skew_seconds < 0 or evidence_ttl_seconds < 1:
            raise ValueError("Alpha Vantage quote freshness bounds are invalid")
        self._clock = clock
        self._max_quote_age_seconds = max_quote_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._ttl_seconds = min(evidence_ttl_seconds, max_quote_age_seconds)
        self._spec = ToolSpec(
            name="market.get_quote_alpha_vantage",
            version="1.0.0",
            description=(
                "Return Alpha Vantage GLOBAL_QUOTE data for one equity symbol. The request "
                "does not claim a realtime entitlement; the default feed is end-of-day."
            ),
            domain="MARKET",
            input_schema=EquityProfileArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=8_192,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = EquityProfileArguments.model_validate(arguments)
        return {"symbol": parsed.symbol}

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._reader.provider_health()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = EquityProfileArguments.model_validate(
            {"symbol": arguments.get("symbol"), "market": "US"}
        )
        payload = await self._reader.get_payload(
            {"function": "GLOBAL_QUOTE", "symbol": parsed.symbol}
        )
        if isinstance(payload, ToolFailure):
            return payload
        raw_quote = payload.get("Global Quote")
        if not isinstance(raw_quote, dict):
            return await self._reader.schema_failure("quote")
        provider_symbol = bounded_provider_text(raw_quote.get("01. symbol"), limit=20)
        price = finite_provider_number(raw_quote.get("05. price"), positive=True)
        trading_day_value = bounded_provider_text(raw_quote.get("07. latest trading day"), limit=10)
        if provider_symbol != parsed.symbol or price is None or trading_day_value is None:
            return await self._reader.empty_failure("quote", parsed.symbol)
        try:
            trading_day = date.fromisoformat(trading_day_value)
            observed_at = datetime.combine(trading_day, time.min, tzinfo=UTC)
        except ValueError:
            return await self._reader.schema_failure("quote timestamp")
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_quote_age_seconds:
            return await self._reader.failure(
                "ALPHA_VANTAGE_STALE_QUOTE",
                "Alpha Vantage returned a quote outside the configured freshness window.",
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._reader.failure(
                "ALPHA_VANTAGE_FUTURE_QUOTE",
                "Alpha Vantage returned a quote date too far in the future.",
            )

        data: dict[str, JsonValue] = {
            "provider": "alpha-vantage",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "price": price,
            "as_of": observed_at.isoformat(),
            "data_freshness": "end_of_day",
            "market_data_entitlement": "historical",
        }
        optional_numbers = {
            "open": ("02. open", True),
            "high": ("03. high", True),
            "low": ("04. low", True),
            "previous_close": ("08. previous close", True),
            "change": ("09. change", False),
        }
        missing_fields: list[JsonValue] = []
        for key, (provider_key, positive) in optional_numbers.items():
            value = finite_provider_number(raw_quote.get(provider_key), positive=positive)
            if value is None:
                missing_fields.append(key)
                continue
            data[key] = value
        percent_value = raw_quote.get("10. change percent")
        if isinstance(percent_value, str):
            percent_value = percent_value.rstrip("%")
        percent_change = finite_provider_number(percent_value)
        if percent_change is None:
            missing_fields.append("percent_change")
        else:
            data["percent_change"] = percent_change
        volume = finite_provider_number(raw_quote.get("06. volume"))
        if volume is not None and volume >= 0 and volume.is_integer():
            data["volume"] = int(volume)
        else:
            missing_fields.append("volume")
        if missing_fields:
            # The adapter owns exactly seven optional quote fields, so this
            # diagnostic cannot grow with provider-controlled input.
            data["missing_fields"] = missing_fields
        statement = canonical_equity_quote_statement(data)
        if statement is None:
            return await self._reader.schema_failure("quote")
        data["statements"] = [statement]
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="alpha-vantage",
                reference=f"global-quote:{parsed.symbol}:{trading_day_value}",
            ),
            observed_at=observed_at,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class AlphaVantageSymbolSearchTool:
    """Return bounded SYMBOL_SEARCH matches with exact provider attribution."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_QUERY_URL,
        gate: ProviderCallGate | None = None,
        evidence_ttl_seconds: int = 3_600,
    ) -> None:
        if evidence_ttl_seconds < 1:
            raise ValueError("Alpha Vantage search TTL must be positive")
        self._reader = _AlphaVantageReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.search_symbols_alpha_vantage",
            version="1.0.0",
            description=(
                "Search Alpha Vantage SYMBOL_SEARCH for bounded global equity symbol and "
                "market metadata matches."
            ),
            domain="MARKET",
            input_schema=EquitySearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=16_384,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return EquitySearchArguments.model_validate(arguments).model_dump(mode="json")

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._reader.provider_health()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = EquitySearchArguments.model_validate(arguments)
        payload = await self._reader.get_payload(
            {"function": "SYMBOL_SEARCH", "keywords": parsed.query}
        )
        if isinstance(payload, ToolFailure):
            return payload
        raw_matches = payload.get("bestMatches")
        if not isinstance(raw_matches, list) or len(raw_matches) > _MAX_SEARCH_MATCHES:
            return await self._reader.schema_failure("symbol search")
        results: list[JsonValue] = []
        rejected_result_count = 0
        for raw_match in raw_matches:
            normalized = _normalize_search_match(raw_match)
            if normalized is None:
                rejected_result_count += 1
                continue
            if len(results) < parsed.limit:
                results.append(normalized)
        query_digest = equity_query_hash(parsed.query)
        data: dict[str, JsonValue] = {
            "provider": "alpha-vantage",
            "query": parsed.query,
            "query_hash": query_digest,
            "requested_market": parsed.market,
            "result_count": len(results),
            "rejected_result_count": rejected_result_count,
            "results": results,
        }
        statements = canonical_equity_search_statements(data)
        if statements is None:
            return await self._reader.schema_failure("symbol search")
        data["statements"] = list(statements)
        now = self._clock.now()
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="alpha-vantage",
                reference=f"symbol-search:{query_digest}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class AlphaVantageCompanyProfileTool:
    """Return a bounded company identity slice from OVERVIEW."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_QUERY_URL,
        gate: ProviderCallGate | None = None,
        evidence_ttl_seconds: int = 86_400,
    ) -> None:
        if evidence_ttl_seconds < 1:
            raise ValueError("Alpha Vantage profile TTL must be positive")
        self._reader = _AlphaVantageReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.get_company_profile_alpha_vantage",
            version="1.0.0",
            description=(
                "Return bounded Alpha Vantage OVERVIEW company identity, exchange, sector, "
                "and industry metadata."
            ),
            domain="MARKET",
            input_schema=EquityProfileArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=8_192,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return EquityProfileArguments.model_validate(arguments).model_dump(mode="json")

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._reader.provider_health()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = EquityProfileArguments.model_validate(arguments)
        payload = await self._reader.get_payload({"function": "OVERVIEW", "symbol": parsed.symbol})
        if isinstance(payload, ToolFailure):
            return payload
        provider_symbol = bounded_provider_text(payload.get("Symbol"), limit=20)
        name = bounded_provider_text(payload.get("Name"), limit=240)
        exchange = bounded_provider_text(payload.get("Exchange"), limit=120)
        industry = bounded_provider_text(payload.get("Industry"), limit=160)
        if provider_symbol != parsed.symbol or not any((name, exchange, industry)):
            return await self._reader.empty_failure("company overview", parsed.symbol)
        now = self._clock.now()
        data: dict[str, JsonValue] = {
            "provider": "alpha-vantage",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "as_of": now.isoformat(),
            "provider_refresh_policy": "same_day_after_reported_financials",
        }
        for key, value in (("name", name), ("exchange", exchange), ("industry", industry)):
            if value is not None:
                data[key] = value
        missing_fields: list[JsonValue] = []
        for key, value in (("exchange", exchange), ("industry", industry), ("name", name)):
            if value is None:
                missing_fields.append(key)
        data["missing_fields"] = missing_fields
        for key, provider_key, limit in (
            ("sector", "Sector", 160),
            ("country", "Country", 120),
            ("currency", "Currency", 12),
            ("asset_type", "AssetType", 80),
        ):
            value = bounded_provider_text(payload.get(provider_key), limit=limit)
            if value is not None:
                data[key] = value.upper() if key == "currency" else value
        statements = canonical_equity_profile_statements(data)
        if statements is None:
            return await self._reader.schema_failure("company overview")
        data["statements"] = list(statements)
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="alpha-vantage",
                reference=f"company-overview:{provider_symbol}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class _AlphaVantageReader:
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str,
        gate: ProviderCallGate | None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Alpha Vantage API key is required")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = _safe_base_url(base_url)
        self._gate = gate or ProviderCallGate(
            provider="alpha_vantage",
            clock=clock,
            max_concurrency=1,
            max_calls_per_minute=5,
            max_calls_per_day=25,
        )
        if self._gate.provider != "alpha_vantage":
            raise ValueError("Alpha Vantage gate authority is mismatched")

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._gate.snapshot()

    async def get_payload(
        self,
        params: dict[str, str],
    ) -> dict[str, object] | ToolFailure:
        request_params = {**params, "apikey": self._api_key}
        try:
            async with self._gate.slot():
                response = await self._client.get(
                    self._base_url,
                    params=request_params,
                    headers={"Accept": "application/json"},
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self.failure(
                "ALPHA_VANTAGE_TIMEOUT",
                "Alpha Vantage did not respond before the adapter timeout.",
                retryable=True,
                provider_credits_used=0,
            )
        except httpx.TransportError:
            return await self.failure(
                "ALPHA_VANTAGE_TRANSPORT_ERROR",
                "Alpha Vantage failed before a response was received.",
                retryable=True,
                provider_credits_used=0,
            )

        http_failure = await self._http_failure(response)
        if http_failure is not None:
            return http_failure
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self.failure(
                "ALPHA_VANTAGE_RESPONSE_TOO_LARGE",
                "Alpha Vantage returned an oversized payload.",
            )
        try:
            raw_payload: object = response.json()
        except ValueError:
            return await self.schema_failure("response")
        if not isinstance(raw_payload, dict):
            return await self.schema_failure("response")
        payload = cast(dict[str, object], raw_payload)
        rate_message = payload.get("Note")
        info_message = payload.get("Information")
        if isinstance(rate_message, str):
            return await self.failure(
                "ALPHA_VANTAGE_RATE_LIMITED",
                "Alpha Vantage reported that its API call allowance was reached.",
                retryable=True,
                rate_limited=True,
            )
        if isinstance(info_message, str):
            normalized = info_message.casefold()
            if any(term in normalized for term in ("rate limit", "frequency", "requests per")):
                return await self.failure(
                    "ALPHA_VANTAGE_RATE_LIMITED",
                    "Alpha Vantage reported that its API call allowance was reached.",
                    retryable=True,
                    rate_limited=True,
                )
            return await self.failure(
                "ALPHA_VANTAGE_ACCESS_REJECTED",
                "Alpha Vantage returned an account or entitlement information response.",
            )
        if isinstance(payload.get("Error Message"), str):
            return await self.failure(
                "ALPHA_VANTAGE_REQUEST_REJECTED",
                "Alpha Vantage rejected the requested function or symbol.",
            )
        return payload

    async def _http_failure(self, response: httpx.Response) -> ToolFailure | None:
        status = response.status_code
        if status == 429:
            return await self.failure(
                "ALPHA_VANTAGE_RATE_LIMITED",
                "Alpha Vantage rate limit was reached.",
                retryable=True,
                rate_limited=True,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        if status >= 500:
            return await self.failure(
                "ALPHA_VANTAGE_UNAVAILABLE",
                f"Alpha Vantage returned HTTP {status}.",
                retryable=True,
            )
        if status in {401, 403}:
            return await self.failure(
                "ALPHA_VANTAGE_AUTH_REJECTED",
                "Alpha Vantage rejected the configured API credential or entitlement.",
            )
        if status >= 400:
            return await self.failure(
                "ALPHA_VANTAGE_REQUEST_REJECTED",
                f"Alpha Vantage returned HTTP {status}.",
            )
        return None

    async def record_success(self) -> None:
        await self._gate.record_success(provider_credits_used=1)

    async def failure(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        rate_limited: bool = False,
        retry_after_seconds: int | None = None,
        provider_credits_used: int = 1,
    ) -> ToolFailure:
        await self._gate.record_failure(
            code,
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
            provider_credits_used=provider_credits_used,
        )
        return ToolFailure(code=code, retryable=retryable, safe_message=message)

    async def schema_failure(self, kind: str) -> ToolFailure:
        return await self.failure(
            "ALPHA_VANTAGE_SCHEMA_DRIFT",
            f"Alpha Vantage returned an unsupported {kind} payload.",
        )

    async def empty_failure(self, kind: str, symbol: str) -> ToolFailure:
        return await self.failure(
            "ALPHA_VANTAGE_EMPTY_RESULT",
            f"Alpha Vantage returned no complete {kind} for {symbol}.",
        )


def _normalize_search_match(value: object) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    provider_symbol = bounded_provider_text(value.get("1. symbol"), limit=20)
    name = bounded_provider_text(value.get("2. name"), limit=240)
    if provider_symbol is None or name is None:
        return None
    result: dict[str, JsonValue] = {
        "symbol": provider_symbol,
        "provider_symbol": provider_symbol,
        "name": name,
    }
    for key, provider_key, limit in (
        ("type", "3. type", 80),
        ("region", "4. region", 120),
        ("market_open", "5. marketOpen", 10),
        ("market_close", "6. marketClose", 10),
        ("timezone", "7. timezone", 80),
        ("currency", "8. currency", 12),
    ):
        item = bounded_provider_text(value.get(provider_key), limit=limit)
        if item is not None:
            result[key] = item.upper() if key == "currency" else item
    score = finite_provider_number(value.get("9. matchScore"))
    if score is not None and 0 <= score <= 1:
        result["match_score"] = score
    return result


def _safe_base_url(value: str) -> str:
    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Alpha Vantage base URL is invalid") from exc
    if (
        url.scheme != "https"
        or url.host != "www.alphavantage.co"
        or url.port not in {None, 443}
        or url.path.rstrip("/") != "/query"
        or bool(url.username)
        or bool(url.password)
        or url.query
        or url.fragment
    ):
        raise ValueError("Alpha Vantage base URL must be the official credential-free REST URL")
    return str(url).rstrip("/")


__all__ = (
    "AlphaVantageCompanyProfileTool",
    "AlphaVantageQuoteTool",
    "AlphaVantageSymbolSearchTool",
)
