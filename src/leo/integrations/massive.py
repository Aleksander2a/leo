"""Typed Massive REST equity adapters with bounded normalized evidence.

Canonical contract sources:
https://www.massive.com/docs/ai-tools/quickstart and
https://massive.com/docs/llms-full.txt
"""

from __future__ import annotations

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
    safe_provider_request_id,
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

_DEFAULT_BASE_URL = "https://api.massive.com"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_PROVIDER_RESULTS = 1_000


class MassiveStockSnapshotTool:
    """Normalize one Massive v3 unified stock snapshot as a quote."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_BASE_URL,
        gate: ProviderCallGate | None = None,
        max_quote_age_seconds: int = 345_600,
        max_future_skew_seconds: int = 60,
        evidence_ttl_seconds: int = 900,
    ) -> None:
        if max_quote_age_seconds < 1 or max_future_skew_seconds < 0 or evidence_ttl_seconds < 1:
            raise ValueError("Massive snapshot freshness bounds are invalid")
        self._reader = _MassiveReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._max_quote_age_seconds = max_quote_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._ttl_seconds = min(evidence_ttl_seconds, max_quote_age_seconds)
        self._spec = ToolSpec(
            name="market.get_quote_massive",
            version="1.0.0",
            description=(
                "Return one Massive /v3/snapshot stock price. Snapshot access depends on the "
                "configured Stocks plan; a permission denial is a typed local provider failure."
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
            "/v3/snapshot",
            {"ticker": parsed.symbol, "type": "stocks", "limit": "1"},
        )
        if isinstance(payload, ToolFailure):
            return payload
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > 1:
            return await self._reader.schema_failure("stock snapshot")
        if not raw_results:
            return await self._reader.empty_failure("stock snapshot", parsed.symbol)
        raw_snapshot = raw_results[0]
        if not isinstance(raw_snapshot, dict):
            return await self._reader.schema_failure("stock snapshot")
        provider_symbol = bounded_provider_text(raw_snapshot.get("ticker"), limit=20)
        asset_type = bounded_provider_text(raw_snapshot.get("type"), limit=24)
        session = raw_snapshot.get("session")
        if (
            provider_symbol != parsed.symbol
            or asset_type != "stocks"
            or not isinstance(session, dict)
        ):
            return await self._reader.empty_failure("stock snapshot", parsed.symbol)
        price = finite_provider_number(session.get("price"), positive=True)
        timestamp_ns = _positive_integer(session.get("last_updated"))
        if price is None or timestamp_ns is None:
            return await self._reader.schema_failure("stock snapshot")
        try:
            observed_at = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return await self._reader.schema_failure("stock snapshot timestamp")
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_quote_age_seconds:
            return await self._reader.failure(
                "MASSIVE_STALE_SNAPSHOT",
                "Massive returned a stock snapshot outside the freshness window.",
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._reader.failure(
                "MASSIVE_FUTURE_SNAPSHOT",
                "Massive returned a stock snapshot timestamp too far in the future.",
            )

        data: dict[str, JsonValue] = {
            "provider": "massive",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "price": price,
            "as_of": observed_at.isoformat(),
            "data_freshness": "provider_plan_dependent",
        }
        missing_fields: list[JsonValue] = []
        market_status = bounded_provider_text(raw_snapshot.get("market_status"), limit=32)
        if market_status is not None:
            data["market_status"] = market_status
        else:
            missing_fields.append("market_status")
        for key, provider_key, positive in (
            ("change", "change", False),
            ("percent_change", "change_percent", False),
            ("open", "open", True),
            ("high", "high", True),
            ("low", "low", True),
            ("previous_close", "previous_close", True),
            ("volume", "volume", False),
        ):
            value = finite_provider_number(session.get(provider_key), positive=positive)
            if value is not None and (key != "volume" or value >= 0):
                data[key] = int(value) if key == "volume" and value.is_integer() else value
            else:
                missing_fields.append(key)
        timeframe = _snapshot_timeframe(raw_snapshot)
        if timeframe is not None:
            data["provider_timeframe"] = timeframe
        else:
            missing_fields.append("provider_timeframe")
        if missing_fields:
            # Static adapter-owned names keep this diagnostic bounded even if
            # Massive adds arbitrary fields to a future response version.
            data["missing_fields"] = missing_fields
        statement = canonical_equity_quote_statement(data)
        if statement is None:
            return await self._reader.schema_failure("stock snapshot")
        data["statements"] = [statement]
        _add_request_id(data, payload)
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="massive",
                reference=f"snapshot:{parsed.symbol}:{timestamp_ns}",
            ),
            observed_at=observed_at,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class MassiveSymbolSearchTool:
    """Search the all-tickers reference endpoint with an API-side result bound."""

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
            raise ValueError("Massive symbol search TTL must be positive")
        self._reader = _MassiveReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.search_symbols_massive",
            version="1.0.0",
            description=(
                "Search Massive /v3/reference/tickers for up to ten active stock symbols. "
                "The reference endpoint is included in all documented Stocks plans."
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
            "/v3/reference/tickers",
            {
                "market": "stocks",
                "active": "true",
                "search": parsed.query,
                "order": "asc",
                "sort": "ticker",
                "limit": str(parsed.limit),
            },
        )
        if isinstance(payload, ToolFailure):
            return payload
        raw_results = payload.get("results")
        if not isinstance(raw_results, list) or len(raw_results) > _MAX_PROVIDER_RESULTS:
            return await self._reader.schema_failure("ticker search")
        results: list[JsonValue] = []
        rejected_result_count = 0
        for item in raw_results:
            normalized = _normalize_ticker(item)
            if normalized is None:
                rejected_result_count += 1
                continue
            if len(results) < parsed.limit:
                results.append(normalized)
        query_digest = equity_query_hash(parsed.query)
        data: dict[str, JsonValue] = {
            "provider": "massive",
            "query": parsed.query,
            "query_hash": query_digest,
            "requested_market": parsed.market,
            "result_count": len(results),
            "rejected_result_count": rejected_result_count,
            "results": results,
            "provider_refresh_policy": "updated_daily",
        }
        statements = canonical_equity_search_statements(data)
        if statements is None:
            return await self._reader.schema_failure("ticker search")
        data["statements"] = list(statements)
        _add_request_id(data, payload)
        now = self._clock.now()
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="massive",
                reference=f"ticker-search:{query_digest}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class MassiveCompanyProfileTool:
    """Return bounded /v3/reference/tickers/{ticker} company metadata."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = _DEFAULT_BASE_URL,
        gate: ProviderCallGate | None = None,
        evidence_ttl_seconds: int = 86_400,
    ) -> None:
        if evidence_ttl_seconds < 1:
            raise ValueError("Massive profile TTL must be positive")
        self._reader = _MassiveReader(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
        )
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.get_company_profile_massive",
            version="1.0.0",
            description=(
                "Return bounded Massive ticker overview identity, primary exchange, currency, "
                "and SIC industry metadata."
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
        payload = await self._reader.get_payload(f"/v3/reference/tickers/{parsed.symbol}", {})
        if isinstance(payload, ToolFailure):
            return payload
        raw_profile = payload.get("results")
        if not isinstance(raw_profile, dict):
            return await self._reader.empty_failure("ticker overview", parsed.symbol)
        provider_symbol = bounded_provider_text(raw_profile.get("ticker"), limit=20)
        name = bounded_provider_text(raw_profile.get("name"), limit=240)
        exchange = bounded_provider_text(raw_profile.get("primary_exchange"), limit=120)
        industry = bounded_provider_text(raw_profile.get("sic_description"), limit=160)
        if provider_symbol != parsed.symbol or not any((name, exchange, industry)):
            return await self._reader.empty_failure("ticker overview", parsed.symbol)
        now = self._clock.now()
        data: dict[str, JsonValue] = {
            "provider": "massive",
            "symbol": parsed.symbol,
            "provider_symbol": provider_symbol,
            "as_of": now.isoformat(),
            "provider_refresh_policy": "updated_daily",
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
            ("currency", "currency_name", 12),
            ("locale", "locale", 24),
            ("security_type", "type", 40),
            ("cik", "cik", 20),
        ):
            value = bounded_provider_text(raw_profile.get(provider_key), limit=limit)
            if value is not None:
                data[key] = value.upper() if key == "currency" else value
        statements = canonical_equity_profile_statements(data)
        if statements is None:
            return await self._reader.schema_failure("ticker overview")
        data["statements"] = list(statements)
        _add_request_id(data, payload)
        await self._reader.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="massive",
                reference=f"ticker-overview:{provider_symbol}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class _MassiveReader:
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
            raise ValueError("Massive API key is required")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = _safe_base_url(base_url, "Massive")
        self._gate = gate or ProviderCallGate(
            provider="massive",
            clock=clock,
            max_concurrency=2,
            max_calls_per_minute=10,
        )
        if self._gate.provider != "massive":
            raise ValueError("Massive gate authority is mismatched")

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
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self.failure(
                "MASSIVE_TIMEOUT",
                "Massive did not respond before the adapter timeout.",
                retryable=True,
                provider_credits_used=0,
            )
        except httpx.TransportError:
            return await self.failure(
                "MASSIVE_TRANSPORT_ERROR",
                "Massive failed before a response was received.",
                retryable=True,
                provider_credits_used=0,
            )
        http_failure = await self._http_failure(response)
        if http_failure is not None:
            return http_failure
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self.failure(
                "MASSIVE_RESPONSE_TOO_LARGE",
                "Massive returned an oversized payload.",
            )
        try:
            raw_payload: object = response.json()
        except ValueError:
            return await self.schema_failure("response")
        if not isinstance(raw_payload, dict):
            return await self.schema_failure("response")
        payload = cast(dict[str, object], raw_payload)
        status = payload.get("status")
        if isinstance(status, str) and status.upper() not in {"OK", "DELAYED"}:
            return await self.failure(
                "MASSIVE_REQUEST_REJECTED",
                "Massive returned a non-success response status.",
            )
        return payload

    async def _http_failure(self, response: httpx.Response) -> ToolFailure | None:
        status = response.status_code
        if status == 429:
            return await self.failure(
                "MASSIVE_RATE_LIMITED",
                "Massive rate limit was reached.",
                retryable=True,
                rate_limited=True,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        if status >= 500:
            return await self.failure(
                "MASSIVE_UNAVAILABLE",
                f"Massive returned HTTP {status}.",
                retryable=True,
            )
        if status == 401:
            return await self.failure(
                "MASSIVE_AUTH_REJECTED",
                "Massive rejected the configured API credential.",
            )
        if status == 403:
            return await self.failure(
                "MASSIVE_ENTITLEMENT_REQUIRED",
                "Massive denied this endpoint for the configured account plan.",
            )
        if status == 404:
            return await self.failure(
                "MASSIVE_NOT_FOUND",
                "Massive did not find the requested market resource.",
            )
        if status >= 400:
            return await self.failure(
                "MASSIVE_REQUEST_REJECTED",
                f"Massive returned HTTP {status}.",
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
            "MASSIVE_SCHEMA_DRIFT", f"Massive returned an unsupported {kind} payload."
        )

    async def empty_failure(self, kind: str, symbol: str) -> ToolFailure:
        return await self.failure(
            "MASSIVE_EMPTY_RESULT", f"Massive returned no complete {kind} for {symbol}."
        )


def _normalize_ticker(value: object) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    symbol = bounded_provider_text(value.get("ticker"), limit=20)
    name = bounded_provider_text(value.get("name"), limit=240)
    if symbol is None or name is None:
        return None
    result: dict[str, JsonValue] = {
        "symbol": symbol,
        "provider_symbol": symbol,
        "name": name,
    }
    for key, provider_key, limit in (
        ("exchange", "primary_exchange", 120),
        ("currency", "currency_name", 12),
        ("region", "locale", 24),
        ("type", "type", 40),
    ):
        item = bounded_provider_text(value.get(provider_key), limit=limit)
        if item is not None:
            result[key] = item.upper() if key == "currency" else item
    active = value.get("active")
    if isinstance(active, bool):
        result["active"] = active
    return result


def _snapshot_timeframe(snapshot: dict[object, object]) -> str | None:
    for key in ("last_trade", "last_quote"):
        item = snapshot.get(key)
        if isinstance(item, dict):
            timeframe = bounded_provider_text(item.get("timeframe"), limit=32)
            if timeframe is not None:
                return timeframe
    return None


def _add_request_id(data: dict[str, JsonValue], payload: dict[str, object]) -> None:
    request_id = safe_provider_request_id(payload.get("request_id"))
    if request_id is not None:
        data["provider_request_id"] = request_id


def _safe_base_url(value: str, provider: str) -> str:
    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{provider} base URL is invalid") from exc
    if (
        url.scheme != "https"
        or url.host != "api.massive.com"
        or url.port not in {None, 443}
        or url.path.rstrip("/")
        or bool(url.username)
        or bool(url.password)
        or url.query
        or url.fragment
    ):
        raise ValueError(f"{provider} base URL must be the official credential-free REST URL")
    return str(url).rstrip("/")


def _positive_integer(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit() and int(value) > 0:
        return int(value)
    return None


__all__ = (
    "MassiveCompanyProfileTool",
    "MassiveStockSnapshotTool",
    "MassiveSymbolSearchTool",
)
