"""Typed TickerLayer equity adapters over bounded REST endpoints.

Canonical contract source: https://tickerlayer.com/docs
TickerLayer documents its market data as derived, indicative, and non-exchange;
the normalized evidence preserves that distinction explicitly.
"""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime, timedelta
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

_DEFAULT_BASE_URL = "https://api.tickerlayer.com"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_SYMBOL_ROWS = 25_000
_QUALIFIED_SYMBOL = re.compile(r"(?P<market>[A-Z]{2}):(?P<base>[A-Z0-9][A-Z0-9.-]{0,19})")


class TickerLayerStockSnapshotTool:
    """Normalize a market-qualified TickerLayer stock snapshot as a quote."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_BASE_URL,
        default_market: str = "US",
        gate: ProviderCallGate | None = None,
        max_quote_age_seconds: int = 345_600,
        max_future_skew_seconds: int = 60,
        evidence_ttl_seconds: int = 900,
    ) -> None:
        if (
            re.fullmatch(r"[A-Z]{2}", default_market) is None
            or max_quote_age_seconds < 1
            or max_future_skew_seconds < 0
            or evidence_ttl_seconds < 1
        ):
            raise ValueError("TickerLayer stock snapshot bounds are invalid")
        self._reader = _TickerLayerReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._default_market = default_market
        self._max_quote_age_seconds = max_quote_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._ttl_seconds = min(evidence_ttl_seconds, max_quote_age_seconds)
        self._spec = ToolSpec(
            name="market.get_quote_ticker_layer",
            version="1.0.0",
            description=(
                "Return one TickerLayer /stocks/snapshot quote using a market-qualified symbol. "
                "The data is explicitly derived, indicative, and non-exchange."
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
            {"symbol": arguments.get("symbol"), "market": self._default_market}
        )
        provider_symbol = f"{self._default_market}:{parsed.symbol}"
        payload = await self._reader.get_payload(f"/stocks/snapshot/{provider_symbol}", {})
        if isinstance(payload, ToolFailure):
            return payload
        returned_symbol = bounded_provider_text(payload.get("symbol"), limit=23)
        price = finite_provider_number(payload.get("last_price"), positive=True)
        timestamp = _positive_integer(payload.get("last_timestamp"))
        if returned_symbol != provider_symbol or price is None or timestamp is None:
            return await self._reader.empty_failure("stock snapshot", provider_symbol)
        try:
            observed_at = datetime.fromtimestamp(timestamp / 1_000, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return await self._reader.schema_failure("stock snapshot timestamp")
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_quote_age_seconds:
            return await self._reader.failure(
                "TICKER_LAYER_STALE_SNAPSHOT",
                "TickerLayer returned a stock snapshot outside the freshness window.",
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._reader.failure(
                "TICKER_LAYER_FUTURE_SNAPSHOT",
                "TickerLayer returned a stock snapshot timestamp too far in the future.",
            )

        data: dict[str, JsonValue] = {
            "provider": "ticker-layer",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "price": price,
            "as_of": observed_at.isoformat(),
            "data_freshness": "provider_entitlement_dependent",
            "data_provenance": "derived_non_exchange_indicative",
            "redistribution": "prohibited_unless_authorized",
        }
        missing_fields: list[JsonValue] = []
        for key, provider_key, positive in (
            ("bid", "bid", True),
            ("ask", "ask", True),
            ("previous_close", "prev_close", True),
            ("change", "change", False),
            ("percent_change", "change_percent", False),
        ):
            value = finite_provider_number(payload.get(provider_key), positive=positive)
            if value is not None:
                data[key] = value
            else:
                missing_fields.append(key)
        for key in ("bid_size", "ask_size", "last_size"):
            value = _nonnegative_integer(payload.get(key))
            if value is not None:
                data[key] = value
            else:
                missing_fields.append(key)
        if missing_fields:
            # The eight possible names come from the fixed optional-field map,
            # never from provider-controlled keys.
            data["missing_fields"] = missing_fields
        statement = canonical_equity_quote_statement(data)
        if statement is None:
            return await self._reader.schema_failure("stock snapshot")
        data["statements"] = [statement]
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="ticker-layer",
                reference=f"stock-snapshot:{provider_symbol}:{timestamp}",
            ),
            observed_at=observed_at,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class TickerLayerSymbolSearchTool:
    """Filter the enabled market symbol list into a bounded discovery result."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_BASE_URL,
        gate: ProviderCallGate | None = None,
        evidence_ttl_seconds: int = 3_600,
    ) -> None:
        if evidence_ttl_seconds < 1:
            raise ValueError("TickerLayer symbol search TTL must be positive")
        self._reader = _TickerLayerReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.search_symbols_ticker_layer",
            version="1.0.0",
            description=(
                "List TickerLayer stocks for one enabled market and locally return at most ten "
                "query matches. The adapter rejects an oversized upstream symbol list."
            ),
            domain="MARKET",
            input_schema=EquitySearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=15.0,
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
        payload = await self._reader.get_payload("/stocks/symbols", {"market": parsed.market})
        if isinstance(payload, ToolFailure):
            return payload
        raw_symbols = payload.get("symbols")
        if not isinstance(raw_symbols, list) or len(raw_symbols) > _MAX_SYMBOL_ROWS:
            return await self._reader.schema_failure("stock symbol list")
        candidates: list[tuple[float, str, dict[str, JsonValue]]] = []
        rejected_result_count = 0
        for raw_symbol in raw_symbols:
            normalized = _normalize_symbol(raw_symbol, expected_market=parsed.market)
            if normalized is None:
                rejected_result_count += 1
                continue
            score = _match_score(parsed.query, normalized)
            if score is not None:
                provider_symbol = normalized["provider_symbol"]
                assert isinstance(provider_symbol, str)
                candidates.append((score, provider_symbol, normalized))
        candidates.sort(key=lambda item: (-item[0], item[1]))
        results: list[JsonValue] = []
        for score, _provider_symbol, normalized in candidates[: parsed.limit]:
            results.append({**normalized, "match_score": score})
        query_digest = equity_query_hash(parsed.query)
        data: dict[str, JsonValue] = {
            "provider": "ticker-layer",
            "query": parsed.query,
            "query_hash": query_digest,
            "requested_market": parsed.market,
            "result_count": len(results),
            "rejected_result_count": rejected_result_count,
            "results": results,
            "data_provenance": "derived_non_exchange_indicative",
        }
        statements = canonical_equity_search_statements(data)
        if statements is None:
            return await self._reader.schema_failure("stock symbol list")
        data["statements"] = list(statements)
        now = self._clock.now()
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="ticker-layer",
                reference=f"stock-symbol-search:{parsed.market}:{query_digest}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class TickerLayerCompanyProfileTool:
    """Return the profile slice of the separately entitled Fundamentals product."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_BASE_URL,
        gate: ProviderCallGate | None = None,
        evidence_ttl_seconds: int = 86_400,
        max_profile_age_seconds: int = 7_776_000,
        max_future_skew_seconds: int = 60,
    ) -> None:
        if evidence_ttl_seconds < 1 or max_profile_age_seconds < 1 or max_future_skew_seconds < 0:
            raise ValueError("TickerLayer company profile bounds are invalid")
        self._reader = _TickerLayerReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._max_profile_age_seconds = max_profile_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._spec = ToolSpec(
            name="market.get_company_profile_ticker_layer",
            version="1.0.0",
            description=(
                "Return TickerLayer Fundamentals company, exchange, sector, and industry "
                "metadata. This endpoint requires the separate Fundamentals permission."
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
        provider_symbol = f"{parsed.market}:{parsed.symbol}"
        payload = await self._reader.get_payload(f"/fundamentals/stocks/{provider_symbol}", {})
        if isinstance(payload, ToolFailure):
            return payload
        returned_symbol = bounded_provider_text(payload.get("symbol"), limit=23)
        base_symbol = bounded_provider_text(payload.get("base_symbol"), limit=20)
        name = bounded_provider_text(payload.get("company"), limit=240)
        exchange = bounded_provider_text(payload.get("exchange"), limit=120)
        industry = bounded_provider_text(payload.get("industry"), limit=160)
        as_of_value = bounded_provider_text(payload.get("as_of"), limit=40)
        if (
            returned_symbol != provider_symbol
            or base_symbol != parsed.symbol
            or not any((name, exchange, industry))
            or as_of_value is None
        ):
            return await self._reader.empty_failure("company fundamentals", provider_symbol)
        try:
            observed_at = datetime.fromisoformat(as_of_value.replace("Z", "+00:00"))
        except ValueError:
            return await self._reader.schema_failure("company fundamentals timestamp")
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_profile_age_seconds:
            return await self._reader.failure(
                "TICKER_LAYER_STALE_PROFILE",
                "TickerLayer returned company fundamentals outside the freshness window.",
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._reader.failure(
                "TICKER_LAYER_FUTURE_PROFILE",
                "TickerLayer returned a company fundamentals timestamp too far in the future.",
            )
        data: dict[str, JsonValue] = {
            "provider": "ticker-layer",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "as_of": observed_at.isoformat(),
            "profile_refresh_policy": "monthly",
            "data_provenance": "derived_non_exchange_indicative",
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
            ("sector", "sector", 160),
            ("country", "country", 120),
            ("mic", "mic", 24),
            ("security_type", "security_type", 80),
        ):
            value = bounded_provider_text(payload.get(provider_key), limit=limit)
            if value is not None:
                data[key] = value
        statements = canonical_equity_profile_statements(data)
        if statements is None:
            return await self._reader.schema_failure("company fundamentals")
        data["statements"] = list(statements)
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="ticker-layer",
                reference=f"stock-fundamentals:{provider_symbol}:{as_of_value}",
            ),
            observed_at=observed_at,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class _TickerLayerReader:
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
            raise ValueError("TickerLayer API key is required")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = _safe_base_url(base_url)
        self._gate = gate or ProviderCallGate(
            provider="ticker_layer",
            clock=clock,
            max_concurrency=2,
            max_calls_per_minute=60,
            # The documented free allowance is a fixed 3,000-call UTC-month quota.
            # Provider 429 bodies remain authoritative for plan-specific limits.
            max_calls_per_month=3_000,
        )
        if self._gate.provider != "ticker_layer":
            raise ValueError("TickerLayer gate authority is mismatched")

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._gate.snapshot()

    async def get_payload(
        self,
        endpoint: str,
        params: dict[str, str],
    ) -> dict[str, object] | ToolFailure:
        try:
            async with self._gate.slot():
                response = await self._client.get(
                    f"{self._base_url}/{endpoint.lstrip('/')}",
                    params=params,
                    headers={"x-api-key": self._api_key, "Accept": "application/json"},
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self.failure(
                "TICKER_LAYER_TIMEOUT",
                "TickerLayer did not respond before the adapter timeout.",
                retryable=True,
                provider_credits_used=0,
            )
        except httpx.TransportError:
            return await self.failure(
                "TICKER_LAYER_TRANSPORT_ERROR",
                "TickerLayer failed before a response was received.",
                retryable=True,
                provider_credits_used=0,
            )
        http_failure = await self._http_failure(response)
        if http_failure is not None:
            return http_failure
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self.failure(
                "TICKER_LAYER_RESPONSE_TOO_LARGE",
                "TickerLayer returned an oversized payload.",
            )
        try:
            raw_payload: object = response.json()
        except ValueError:
            return await self.schema_failure("response")
        if not isinstance(raw_payload, dict):
            return await self.schema_failure("response")
        return cast(dict[str, object], raw_payload)

    async def _http_failure(self, response: httpx.Response) -> ToolFailure | None:
        status = response.status_code
        if status == 429:
            error_code, retry_seconds = _tickerlayer_429(response)
            if error_code == "REST_QUOTA_EXCEEDED":
                return await self.failure(
                    "TICKER_LAYER_MONTHLY_QUOTA_EXHAUSTED",
                    "TickerLayer reported that the monthly REST quota is exhausted.",
                    rate_limited=True,
                    retry_after_seconds=300,
                    provider_credits_used=0,
                )
            return await self.failure(
                "TICKER_LAYER_RATE_LIMITED",
                "TickerLayer REST rate limit was reached.",
                retryable=True,
                rate_limited=True,
                retry_after_seconds=retry_seconds,
                provider_credits_used=0,
            )
        if status >= 500:
            return await self.failure(
                "TICKER_LAYER_UNAVAILABLE",
                f"TickerLayer returned HTTP {status}.",
                retryable=True,
                provider_credits_used=0 if status == 503 else 1,
            )
        if status == 401:
            return await self.failure(
                "TICKER_LAYER_AUTH_REJECTED",
                "TickerLayer rejected the configured API credential.",
                provider_credits_used=0,
            )
        if status == 403:
            return await self.failure(
                "TICKER_LAYER_ENTITLEMENT_REQUIRED",
                "TickerLayer denied the required market or Fundamentals permission.",
                provider_credits_used=0,
            )
        if status == 404:
            return await self.failure(
                "TICKER_LAYER_NOT_FOUND",
                "TickerLayer did not find the requested market-qualified resource.",
            )
        if status >= 400:
            return await self.failure(
                "TICKER_LAYER_REQUEST_REJECTED",
                f"TickerLayer returned HTTP {status}.",
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
            "TICKER_LAYER_SCHEMA_DRIFT",
            f"TickerLayer returned an unsupported {kind} payload.",
        )

    async def empty_failure(self, kind: str, symbol: str) -> ToolFailure:
        return await self.failure(
            "TICKER_LAYER_EMPTY_RESULT",
            f"TickerLayer returned no complete {kind} for {symbol}.",
        )


def _normalize_symbol(
    value: object,
    *,
    expected_market: str,
) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    provider_symbol = bounded_provider_text(value.get("symbol"), limit=23)
    base_symbol = bounded_provider_text(value.get("base_symbol"), limit=20)
    market = bounded_provider_text(value.get("market"), limit=2)
    name = bounded_provider_text(value.get("name"), limit=240)
    if provider_symbol is None or base_symbol is None or market != expected_market or name is None:
        return None
    match = _QUALIFIED_SYMBOL.fullmatch(provider_symbol)
    if match is None or match.group("market") != market or match.group("base") != base_symbol:
        return None
    return {
        "symbol": base_symbol,
        "provider_symbol": provider_symbol,
        "name": name,
        "market": market,
    }


def _match_score(query: str, item: dict[str, JsonValue]) -> float | None:
    normalized_query = query.casefold()
    symbol = item.get("symbol")
    name = item.get("name")
    if not isinstance(symbol, str) or not isinstance(name, str):
        return None
    normalized_symbol = symbol.casefold()
    normalized_name = name.casefold()
    if normalized_query in {normalized_symbol, normalized_name}:
        return 1.0
    if normalized_symbol.startswith(normalized_query) or normalized_name.startswith(
        normalized_query
    ):
        return 0.9
    if normalized_query in normalized_symbol or normalized_query in normalized_name:
        return 0.75
    return None


def _tickerlayer_429(response: httpx.Response) -> tuple[str | None, int | None]:
    if len(response.content) > 16_384:
        return None, bounded_retry_after(response.headers.get("retry-after"))
    try:
        payload: object = response.json()
    except ValueError:
        return None, bounded_retry_after(response.headers.get("retry-after"))
    if not isinstance(payload, dict):
        return None, bounded_retry_after(response.headers.get("retry-after"))
    error = payload.get("error")
    error_code = error if isinstance(error, str) else None
    retry_ms = finite_provider_number(payload.get("retryAfterMs"), positive=True)
    retry_seconds = min(max(math.ceil(retry_ms / 1_000), 1), 300) if retry_ms is not None else None
    return error_code, retry_seconds or bounded_retry_after(response.headers.get("retry-after"))


def _positive_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


def _nonnegative_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _safe_base_url(value: str) -> str:
    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("TickerLayer base URL is invalid") from exc
    if (
        url.scheme != "https"
        or url.host != "api.tickerlayer.com"
        or url.port not in {None, 443}
        or url.path.rstrip("/")
        or bool(url.username)
        or bool(url.password)
        or url.query
        or url.fragment
    ):
        raise ValueError("TickerLayer base URL must be the official credential-free REST URL")
    return str(url).rstrip("/")


__all__ = (
    "TickerLayerCompanyProfileTool",
    "TickerLayerStockSnapshotTool",
    "TickerLayerSymbolSearchTool",
)
