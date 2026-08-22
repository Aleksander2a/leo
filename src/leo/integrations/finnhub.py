"""Async Finnhub tools using Leo's normalized tool contract."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.harness.earnings import canonical_earnings_statements
from leo.harness.models import (
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
from leo.harness.ports import Clock
from leo.harness.provider_canonical import canonical_finnhub_profile_statements
from leo.harness.provider_health import ProviderHealthSnapshot
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRejected,
    bounded_retry_after,
)
from leo.url_policy import is_public_https_url

_MAX_RESPONSE_BYTES = 1_048_576


class FinnhubProviderLimiter:
    """One composition-root-owned concurrency boundary shared by all endpoints."""

    def __init__(self, max_concurrency: int = 4) -> None:
        if max_concurrency < 1:
            raise ValueError("Finnhub concurrency must be positive")
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        async with self._semaphore:
            yield


class _FinnhubProviderBoundary:
    """Share Finnhub call accounting while retaining the legacy limiter seam."""

    def __init__(
        self,
        *,
        clock: Clock,
        max_concurrency: int,
        limiter: FinnhubProviderLimiter | None,
        gate: ProviderCallGate | None,
    ) -> None:
        self._gate = gate or ProviderCallGate(
            provider="finnhub",
            clock=clock,
            max_concurrency=max_concurrency,
            max_calls_per_minute=60,
        )
        if self._gate.provider != "finnhub":
            raise ValueError("Finnhub gate authority is mismatched")
        # An explicitly supplied legacy limiter may still coordinate old callers
        # across endpoints. New live composition can rely on the shared gate alone.
        self._limiter = limiter

    @asynccontextmanager
    async def _provider_slot(self) -> AsyncIterator[None]:
        if self._limiter is None:
            async with self._gate.slot():
                yield
            return
        async with self._limiter.slot():
            async with self._gate.slot():
                yield

    async def provider_health(self) -> ProviderHealthSnapshot:
        return await self._gate.snapshot()

    async def _record_provider_success(self) -> None:
        await self._gate.record_success(provider_credits_used=1)

    async def _record_provider_failure(
        self,
        failure: ToolFailure,
        *,
        rate_limited: bool = False,
        retry_after_seconds: int | None = None,
        provider_credits_used: int = 1,
    ) -> ToolFailure:
        await self._gate.record_failure(
            failure.code,
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
            provider_credits_used=provider_credits_used,
        )
        return failure


class _QuoteArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")


def normalize_quote_symbol(symbol: str) -> str:
    """Normalize and validate a trusted quote symbol once at the composition boundary."""

    return _QuoteArguments.model_validate({"symbol": symbol.strip().upper()}).symbol


def _safe_finnhub_base_url(value: str) -> str:
    """Pin credential-bearing requests to Finnhub's official REST root."""

    try:
        url = httpx.URL(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Finnhub base URL is invalid") from exc
    if (
        url.scheme != "https"
        or url.host != "finnhub.io"
        or url.port not in {None, 443}
        or url.path not in {"/api/v1", "/api/v1/"}
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Finnhub base URL must be the official credential-free HTTPS REST root")
    return str(url).rstrip("/")


class _FinnhubQuote(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    # Finnhub occasionally adds or withholds optional quote metrics. Keep the
    # provider boundary strict about the two authoritative fields, while parsing
    # optional metrics independently so one malformed value cannot discard a
    # usable current price.
    current: object = Field(alias="c")
    timestamp: object = Field(alias="t")
    change: object | None = Field(default=None, alias="d")
    percent_change: object | None = Field(default=None, alias="dp")
    high: object | None = Field(default=None, alias="h")
    low: object | None = Field(default=None, alias="l")
    open: object | None = Field(default=None, alias="o")
    previous_close: object | None = Field(default=None, alias="pc")


class FinnhubQuoteTool(_FinnhubProviderBoundary):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://finnhub.io/api/v1",
        max_quote_age_seconds: int = 345_600,
        max_future_skew_seconds: int = 60,
        evidence_ttl_seconds: int = 900,
        max_concurrency: int = 4,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Finnhub API key is required")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = _safe_finnhub_base_url(base_url)
        if (
            max_quote_age_seconds < 1
            or max_future_skew_seconds < 0
            or evidence_ttl_seconds < 1
            or max_concurrency < 1
        ):
            raise ValueError("Finnhub freshness and concurrency limits are invalid")
        self._max_quote_age_seconds = max_quote_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._evidence_ttl_seconds = evidence_ttl_seconds
        super().__init__(
            clock=clock,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )
        self._spec = ToolSpec(
            name="market.get_quote",
            version="1.1.0",
            description=(
                "Return a current Finnhub quote for one normalized symbol. Currency is not "
                "inferred by this narrow quote tool."
            ),
            domain="MARKET",
            input_schema=_QuoteArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = _QuoteArguments.model_validate(arguments)
        return {"symbol": parsed.symbol}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        symbol = arguments["symbol"]
        if not isinstance(symbol, str):
            raise TypeError("validated symbol must be a string")
        try:
            async with self._provider_slot():
                response = await self._client.get(
                    f"{self._base_url}/quote",
                    params={"symbol": symbol},
                    headers={"X-Finnhub-Token": self._api_key},
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_TIMEOUT",
                    "Finnhub did not respond before the adapter timeout.",
                    retryable=True,
                ),
                provider_credits_used=0,
            )
        except httpx.TransportError:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_TRANSPORT_ERROR",
                    "Finnhub request failed before a response was received.",
                    retryable=True,
                ),
                provider_credits_used=0,
            )
        if response.status_code == 429:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_RATE_LIMITED",
                    "Finnhub rate limit was reached.",
                    retryable=True,
                ),
                rate_limited=True,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        if response.status_code >= 500:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_UNAVAILABLE",
                    f"Finnhub returned HTTP {response.status_code}.",
                    retryable=True,
                )
            )
        if response.status_code >= 400:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_REQUEST_REJECTED",
                    f"Finnhub returned HTTP {response.status_code}.",
                )
            )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_RESPONSE_TOO_LARGE",
                    "Finnhub returned a response larger than the adapter safety limit.",
                )
            )
        try:
            quote = _FinnhubQuote.model_validate(response.json())
        except (TypeError, ValueError):
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_SCHEMA_DRIFT",
                    "Finnhub returned an unsupported quote payload.",
                )
            )
        current = _finite_quote_number(quote.current)
        if current is None:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_NON_FINITE_QUOTE",
                    "Finnhub returned an invalid current quote value.",
                )
            )
        if current <= 0:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_EMPTY_QUOTE",
                    f"Finnhub returned no current quote for {symbol}.",
                )
            )
        timestamp = _positive_quote_integer(quote.timestamp)
        if timestamp is None:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_INVALID_TIMESTAMP",
                    "Finnhub returned an invalid quote timestamp.",
                )
            )
        try:
            observed_at = datetime.fromtimestamp(timestamp, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_INVALID_TIMESTAMP",
                    "Finnhub returned an invalid quote timestamp.",
                )
            )
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_quote_age_seconds:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_STALE_QUOTE",
                    "Finnhub returned a quote outside the freshness window.",
                )
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_FUTURE_QUOTE",
                    "Finnhub returned a quote timestamp too far in the future.",
                )
            )
        effective_evidence_ttl = min(
            self._evidence_ttl_seconds,
            self._max_quote_age_seconds,
        )
        if age_seconds > self._max_quote_age_seconds - effective_evidence_ttl:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_STALE_QUOTE",
                    "Finnhub returned a quote without enough remaining freshness for the "
                    "bounded evidence workflow.",
                )
            )
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        data: dict[str, JsonValue] = {
            "symbol": symbol,
            "price": current,
            "as_of": observed_at.isoformat(),
        }
        missing_fields: list[JsonValue] = []
        for key, raw_value, positive in (
            ("change", quote.change, False),
            ("percent_change", quote.percent_change, False),
            ("high", quote.high, True),
            ("low", quote.low, True),
            ("open", quote.open, True),
            ("previous_close", quote.previous_close, True),
        ):
            value = _finite_quote_number(raw_value)
            if value is None or (positive and value <= 0):
                missing_fields.append(key)
                continue
            data[key] = value
        if missing_fields:
            # This list is bounded by the six adapter-owned optional fields above.
            data["missing_fields"] = missing_fields
        if request_id is not None:
            data["provider_request_id"] = request_id
        await self._record_provider_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="finnhub",
                reference=f"quote:{symbol}:{timestamp}",
            ),
            observed_at=observed_at,
            expires_at=now + timedelta(seconds=effective_evidence_ttl),
        )


def _safe_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None


def _finite_quote_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, str | int | float):
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _positive_quote_integer(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


class _CompanyProfileArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")


class _CompanyNewsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")
    days: int = Field(default=7, ge=1, le=30)
    limit: int = Field(default=5, ge=1, le=10)


class _EarningsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")
    limit: int = Field(default=4, ge=1, le=4)


class _BasicFinancialsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")


class _FinnhubReadTool(_FinnhubProviderBoundary):
    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str,
        max_concurrency: int,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Finnhub API key is required")
        if max_concurrency < 1:
            raise ValueError("Finnhub concurrency must be positive")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = _safe_finnhub_base_url(base_url)
        super().__init__(
            clock=clock,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )

    async def _get(
        self,
        endpoint: str,
        params: dict[str, str],
    ) -> httpx.Response | ToolFailure:
        try:
            async with self._provider_slot():
                response = await self._client.get(
                    f"{self._base_url}/{endpoint.lstrip('/')}",
                    params=params,
                    headers={"X-Finnhub-Token": self._api_key, "Accept": "application/json"},
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_TIMEOUT",
                    "Finnhub did not respond before the adapter timeout.",
                    retryable=True,
                ),
                provider_credits_used=0,
            )
        except httpx.TransportError:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_TRANSPORT_ERROR",
                    "Finnhub request failed before a response was received.",
                    retryable=True,
                ),
                provider_credits_used=0,
            )
        if response.status_code == 429:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_RATE_LIMITED",
                    "Finnhub rate limit was reached.",
                    retryable=True,
                ),
                rate_limited=True,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        if response.status_code >= 500:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_UNAVAILABLE",
                    f"Finnhub returned HTTP {response.status_code}.",
                    retryable=True,
                )
            )
        if response.status_code >= 400:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_REQUEST_REJECTED",
                    f"Finnhub returned HTTP {response.status_code}.",
                )
            )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_RESPONSE_TOO_LARGE",
                    "Finnhub returned a response larger than the adapter safety limit.",
                )
            )
        return response


class FinnhubCompanyProfileTool(_FinnhubReadTool):
    """Return bounded Company Profile 2 metadata for one symbol."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://finnhub.io/api/v1",
        evidence_ttl_seconds: int = 86_400,
        max_concurrency: int = 4,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )
        if evidence_ttl_seconds < 1:
            raise ValueError("Finnhub profile TTL must be positive")
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.get_company_profile",
            version="1.0.0",
            description=(
                "Return Finnhub Company Profile 2 identity, listing, industry, and market-cap "
                "metadata for one normalized symbol."
            ),
            domain="MARKET",
            input_schema=_CompanyProfileArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=8_192,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _CompanyProfileArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _CompanyProfileArguments.model_validate(arguments)
        response = await self._get("stock/profile2", {"symbol": parsed.symbol})
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._record_provider_failure(_finnhub_schema_failure("company profile"))
        if not isinstance(payload, dict):
            return await self._record_provider_failure(_finnhub_schema_failure("company profile"))
        ticker = _bounded_text(payload.get("ticker"), 20)
        name = _bounded_text(payload.get("name"), 240)
        exchange = _bounded_text(payload.get("exchange"), 120)
        industry = _bounded_text(payload.get("finnhubIndustry"), 160)
        if ticker != parsed.symbol or not any((name, exchange, industry)):
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_EMPTY_PROFILE",
                    f"Finnhub returned no complete company profile for {parsed.symbol}.",
                )
            )
        data: dict[str, JsonValue] = {
            "symbol": ticker,
        }
        for key, value in (("name", name), ("exchange", exchange), ("industry", industry)):
            if value is not None:
                data[key] = value
        missing_fields: list[JsonValue] = []
        for key, value in (("exchange", exchange), ("industry", industry), ("name", name)):
            if value is None:
                missing_fields.append(key)
        data["missing_fields"] = missing_fields
        optional_text = {
            "country": (payload.get("country"), 120),
            "currency": (payload.get("currency"), 12),
            "industry": (payload.get("finnhubIndustry"), 160),
            "ipo_date": (payload.get("ipo"), 20),
            "web_url": (payload.get("weburl"), 2_048),
        }
        for key, (raw_value, limit) in optional_text.items():
            value = _bounded_text(raw_value, limit)
            if value and (key != "web_url" or _valid_https_url(value)):
                data[key] = value
        for key, provider_key in (
            ("market_capitalization", "marketCapitalization"),
            ("share_outstanding", "shareOutstanding"),
        ):
            value = payload.get(provider_key)
            if (
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                data[key] = float(value)
        statements = canonical_finnhub_profile_statements(data)
        if statements is None:
            return await self._record_provider_failure(_finnhub_schema_failure("company profile"))
        data["statements"] = list(statements)
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        now = self._clock.now()
        await self._record_provider_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="finnhub",
                reference=f"company-profile:{ticker}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class FinnhubCompanyNewsTool(_FinnhubReadTool):
    """Return recent capped company news with canonical provider statements."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://finnhub.io/api/v1",
        evidence_ttl_seconds: int = 900,
        max_concurrency: int = 4,
        max_future_skew_seconds: int = 60,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )
        if evidence_ttl_seconds < 1 or max_future_skew_seconds < 0:
            raise ValueError("Finnhub news TTL must be positive")
        self._ttl_seconds = evidence_ttl_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._spec = ToolSpec(
            name="market.get_company_news",
            version="1.0.0",
            description=(
                "Return up to ten recent Finnhub company-news items for one symbol and a "
                "bounded 1-30 day lookback."
            ),
            domain="MARKET",
            input_schema=_CompanyNewsArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=24_576,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _CompanyNewsArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _CompanyNewsArguments.model_validate(arguments)
        now = self._clock.now()
        to_date = now.date()
        from_date = to_date - timedelta(days=parsed.days)
        response = await self._get(
            "company-news",
            {
                "symbol": parsed.symbol,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
            },
        )
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._record_provider_failure(_finnhub_schema_failure("company news"))
        if not isinstance(payload, list):
            return await self._record_provider_failure(_finnhub_schema_failure("company news"))
        items: list[JsonValue] = []
        statements: list[JsonValue] = []
        rejected_item_count = 0
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                return await self._record_provider_failure(_finnhub_schema_failure("company news"))
            headline = _bounded_text(raw_item.get("headline"), 500)
            provider = _bounded_text(raw_item.get("source"), 160)
            url = _bounded_text(raw_item.get("url"), 2_048)
            timestamp = raw_item.get("datetime")
            if not (
                headline
                and provider
                and url
                and is_public_https_url(url)
                and isinstance(timestamp, int)
                and not isinstance(timestamp, bool)
                and timestamp > 0
            ):
                rejected_item_count += 1
                continue
            try:
                published_at = datetime.fromtimestamp(timestamp, tz=UTC)
            except (OverflowError, OSError, ValueError):
                rejected_item_count += 1
                continue
            if not (
                from_date <= published_at.date() <= to_date
                and published_at <= now + timedelta(seconds=self._max_future_skew_seconds)
            ):
                rejected_item_count += 1
                continue
            statement = (
                f"On {published_at.isoformat()}, {provider} reported for {parsed.symbol}: "
                f"{headline} Source URL: {url}"
            )
            statements.append(statement)
            item: dict[str, JsonValue] = {
                "published_at": published_at.isoformat(),
                "headline": headline,
                "source": provider,
                "url": url,
            }
            summary = _bounded_text(raw_item.get("summary"), 1_200)
            if summary:
                item["summary"] = summary
            items.append(item)
            if len(items) == parsed.limit:
                break
        if not items:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_NO_COMPANY_NEWS",
                    f"Finnhub returned no valid company news for {parsed.symbol}.",
                )
            )
        data: dict[str, JsonValue] = {
            "symbol": parsed.symbol,
            "from_date": from_date.isoformat(),
            "to_date": to_date.isoformat(),
            "items": items,
            "statements": statements,
            "item_count": len(items),
            "rejected_item_count": rejected_item_count,
        }
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        # SourceRef has observation cardinality while this result may contain
        # several articles.  Preserve a URL only when it is unambiguous; the
        # renderer independently binds each verified claim statement to its exact
        # item URL and otherwise omits the link.
        source_url: str | None = None
        if len(items) == 1:
            only_item = items[0]
            assert isinstance(only_item, dict)
            candidate_url = only_item.get("url")
            assert isinstance(candidate_url, str)
            source_url = candidate_url
        await self._record_provider_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="finnhub",
                reference=(
                    f"company-news:{parsed.symbol}:{from_date.isoformat()}:{to_date.isoformat()}"
                ),
                url=source_url,
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class FinnhubEarningsSurprisesTool(_FinnhubReadTool):
    """Return the last four reported Finnhub earnings surprises."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://finnhub.io/api/v1",
        evidence_ttl_seconds: int = 21_600,
        max_concurrency: int = 4,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )
        if evidence_ttl_seconds < 1:
            raise ValueError("Finnhub earnings TTL must be positive")
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.get_earnings_surprises",
            version="1.0.0",
            description=(
                "Return up to four recent Finnhub reported-versus-estimated EPS observations "
                "for one symbol."
            ),
            domain="MARKET",
            input_schema=_EarningsArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=12_288,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _EarningsArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _EarningsArguments.model_validate(arguments)
        response = await self._get("stock/earnings", {"symbol": parsed.symbol})
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._record_provider_failure(_finnhub_schema_failure("earnings"))
        if not isinstance(payload, list):
            return await self._record_provider_failure(_finnhub_schema_failure("earnings"))
        now = self._clock.now()
        items: list[JsonValue] = []
        for raw_item in payload:
            if not isinstance(raw_item, dict):
                return await self._record_provider_failure(_finnhub_schema_failure("earnings"))
            actual = _finite_number(raw_item.get("actual"))
            estimate = _finite_number(raw_item.get("estimate"))
            period = _bounded_text(raw_item.get("period"), 20)
            symbol = _bounded_text(raw_item.get("symbol"), 20)
            try:
                period_date = date.fromisoformat(period) if period is not None else None
            except ValueError:
                period_date = None
            if (
                actual is None
                or estimate is None
                or period_date is None
                or period_date > now.date()
                or symbol != parsed.symbol
            ):
                continue
            item: dict[str, JsonValue] = {
                "symbol": symbol,
                "period": period,
                "actual": actual,
                "estimate": estimate,
            }
            for key in ("surprise", "surprisePercent"):
                value = _finite_number(raw_item.get(key))
                if value is not None:
                    item["surprise_percent" if key == "surprisePercent" else key] = value
            items.append(item)
            if len(items) == parsed.limit:
                break
        if not items:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_NO_EARNINGS",
                    f"Finnhub returned no valid earnings observations for {parsed.symbol}.",
                )
            )
        canonical = canonical_earnings_statements(parsed.symbol, items)
        if canonical is None:
            return await self._record_provider_failure(_finnhub_schema_failure("earnings"))
        data: dict[str, JsonValue] = {
            "symbol": parsed.symbol,
            "items": items,
            "statements": list(canonical),
            "item_count": len(items),
        }
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        await self._record_provider_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="finnhub",
                reference=f"earnings-surprises:{parsed.symbol}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


class FinnhubBasicFinancialsTool(_FinnhubReadTool):
    """Return a small whitelist from Finnhub's basic-financials metric endpoint."""

    _METRICS: ClassVar[dict[str, str]] = {
        "beta": "beta",
        "52WeekHigh": "52-week high",
        "52WeekLow": "52-week low",
        "10DayAverageTradingVolume": "10-day average trading volume",
        "marketCapitalization": "market capitalization",
        "peBasicExclExtraTTM": "basic P/E excluding extraordinary items (TTM)",
    }

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://finnhub.io/api/v1",
        evidence_ttl_seconds: int = 21_600,
        max_concurrency: int = 4,
        limiter: FinnhubProviderLimiter | None = None,
        gate: ProviderCallGate | None = None,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            max_concurrency=max_concurrency,
            limiter=limiter,
            gate=gate,
        )
        if evidence_ttl_seconds < 1:
            raise ValueError("Finnhub basic-financials TTL must be positive")
        self._ttl_seconds = evidence_ttl_seconds
        self._spec = ToolSpec(
            name="market.get_basic_financials",
            version="1.0.0",
            description=(
                "Return a bounded whitelist of Finnhub basic financial metrics for one symbol."
            ),
            domain="MARKET",
            input_schema=_BasicFinancialsArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=8_192,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _BasicFinancialsArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _BasicFinancialsArguments.model_validate(arguments)
        response = await self._get(
            "stock/metric",
            {"symbol": parsed.symbol, "metric": "all"},
        )
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._record_provider_failure(_finnhub_schema_failure("basic financials"))
        if not isinstance(payload, dict) or not isinstance(payload.get("metric"), dict):
            return await self._record_provider_failure(_finnhub_schema_failure("basic financials"))
        provider_metrics = payload["metric"]
        assert isinstance(provider_metrics, dict)
        metrics: dict[str, JsonValue] = {}
        statements: list[JsonValue] = []
        for provider_key, label in self._METRICS.items():
            value = _finite_number(provider_metrics.get(provider_key))
            if value is None:
                continue
            metrics[provider_key] = value
            statements.append(f"{parsed.symbol} has Finnhub {label} {format(value, 'g')}.")
        if not metrics:
            return await self._record_provider_failure(
                _finnhub_failure(
                    "FINNHUB_NO_BASIC_FINANCIALS",
                    f"Finnhub returned no supported basic financials for {parsed.symbol}.",
                )
            )
        data: dict[str, JsonValue] = {
            "symbol": parsed.symbol,
            "metrics": metrics,
            "statements": statements,
            "metric_count": len(metrics),
        }
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        now = self._clock.now()
        await self._record_provider_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="finnhub",
                reference=f"basic-financials:{parsed.symbol}",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )


def _finnhub_failure(code: str, message: str, *, retryable: bool = False) -> ToolFailure:
    return ToolFailure(code=code, safe_message=message, retryable=retryable)


def _finnhub_schema_failure(kind: str) -> ToolFailure:
    return _finnhub_failure(
        "FINNHUB_SCHEMA_DRIFT",
        f"Finnhub returned an unsupported {kind} payload.",
    )


def _bounded_text(value: object, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", value).split())[:limit]
    return cleaned or None


def _finite_number(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _valid_https_url(value: str) -> bool:
    """Compatibility spelling for profile URLs; performs no DNS or fetch."""

    return is_public_https_url(value)
