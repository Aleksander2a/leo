"""Resilient CoinGecko and CoinMarketCap cryptocurrency market tools.

The adapters use the providers' official REST APIs directly through ``httpx``.  They
share Leo's provider-neutral snapshot schema and can be composed into a bounded
corroboration call that remains useful when either free-tier provider is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, Protocol

import httpx
from pydantic import JsonValue, ValidationError

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
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRejected,
    bounded_retry_after,
)
from leo.providers.crypto import (
    CryptoAggregatePayload,
    CryptoProviderPayload,
    CryptoProviderSnapshot,
    CryptoSnapshotArguments,
    calculate_crypto_agreement,
    canonical_crypto_aggregate_summary,
    canonical_crypto_agreement_statement,
    canonical_crypto_snapshot_statement,
    crypto_provenance_digest,
)

_COINGECKO_DOC_URL = "https://docs.coingecko.com/reference/coins-markets"
_COINMARKETCAP_DOC_URL = (
    "https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency"
)
_PROVIDER_ORDER = {"coingecko": 0, "coinmarketcap": 1}
_COINGECKO_REST_HOSTS = frozenset({"api.coingecko.com", "pro-api.coingecko.com"})
_COINGECKO_REST_PATH = "/api/v3"
_COINMARKETCAP_REST_HOST = "pro-api.coinmarketcap.com"
_MAX_RESPONSE_BYTES = 1_048_576


class CryptoSnapshotProvider(Protocol):
    @property
    def provider_name(self) -> Literal["coingecko", "coinmarketcap"]: ...

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome: ...


class _CryptoProviderTool:
    _provider_name: Literal["coingecko", "coinmarketcap"]

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str,
        gate: ProviderCallGate | None,
        max_snapshot_age_seconds: int,
        max_future_skew_seconds: int,
        evidence_ttl_seconds: int,
        max_concurrency: int,
        max_calls_per_minute: int,
    ) -> None:
        if not api_key.strip():
            raise ValueError(f"{self._provider_name} API key is required")
        if not base_url.strip():
            raise ValueError(f"{self._provider_name} base URL is required")
        parsed_base = httpx.URL(base_url)
        if (
            parsed_base.scheme != "https"
            or not parsed_base.host
            or parsed_base.userinfo
            or parsed_base.query
            or parsed_base.fragment
            or parsed_base.port is not None
        ):
            raise ValueError("crypto provider base URL must be a plain HTTPS origin/path")
        normalized_path = parsed_base.path.rstrip("/")
        if self._provider_name == "coingecko":
            official_base = (
                parsed_base.host in _COINGECKO_REST_HOSTS
                and normalized_path == _COINGECKO_REST_PATH
            )
        else:
            official_base = parsed_base.host == _COINMARKETCAP_REST_HOST and normalized_path == ""
        if not official_base:
            raise ValueError(
                f"{self._provider_name} base URL must use its official REST API host and path"
            )
        if (
            max_snapshot_age_seconds < 1
            or max_future_skew_seconds < 0
            or evidence_ttl_seconds < 1
            or max_concurrency < 1
            or max_calls_per_minute < 1
        ):
            raise ValueError("crypto provider freshness and call limits are invalid")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._base_url = base_url.rstrip("/")
        self._max_snapshot_age_seconds = max_snapshot_age_seconds
        self._max_future_skew_seconds = max_future_skew_seconds
        self._evidence_ttl_seconds = evidence_ttl_seconds
        self._gate = gate or ProviderCallGate(
            provider=self._provider_name,
            clock=clock,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
        )

    @property
    def provider_name(self) -> Literal["coingecko", "coinmarketcap"]:
        return self._provider_name

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return CryptoSnapshotArguments.model_validate(arguments).model_dump(mode="json")

    async def _get(
        self,
        path: str,
        *,
        params: dict[str, str],
        headers: dict[str, str],
    ) -> httpx.Response | ToolFailure:
        try:
            async with self._gate.slot():
                response = await self._client.get(
                    f"{self._base_url}/{path.lstrip('/')}",
                    params=params,
                    headers=headers,
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self._failure(
                "TIMEOUT",
                f"{self._provider_name} did not respond before the adapter timeout.",
                retryable=True,
            )
        except httpx.TransportError:
            return await self._failure(
                "TRANSPORT_ERROR",
                f"{self._provider_name} failed before a response was received.",
                retryable=True,
            )
        status_code = response.status_code
        response_within_size_bound = len(response.content) <= _MAX_RESPONSE_BYTES
        provider_credits_used = (
            self._response_provider_credits(response) if response_within_size_bound else 0
        )
        if status_code == 429:
            code = self._code("RATE_LIMITED")
            await self._gate.record_failure(
                code,
                rate_limited=True,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
                provider_credits_used=provider_credits_used,
            )
            return ToolFailure(
                code=code,
                retryable=True,
                safe_message=f"{self._provider_name} rate limit was reached.",
            )
        if status_code >= 500:
            return await self._failure(
                "UNAVAILABLE",
                f"{self._provider_name} returned HTTP {status_code}.",
                retryable=True,
                provider_credits_used=provider_credits_used,
            )
        if status_code in {401, 403}:
            return await self._failure(
                "AUTH_REJECTED",
                f"{self._provider_name} rejected its configured API credential or plan.",
                provider_credits_used=provider_credits_used,
            )
        if status_code >= 400:
            return await self._failure(
                "REQUEST_REJECTED",
                f"{self._provider_name} returned HTTP {status_code}.",
                provider_credits_used=provider_credits_used,
            )
        if not response_within_size_bound:
            return await self._failure(
                "RESPONSE_TOO_LARGE",
                f"{self._provider_name} returned a response above Leo's bounded size limit.",
            )
        return response

    async def _failure(
        self,
        suffix: str,
        safe_message: str,
        *,
        retryable: bool = False,
        provider_credits_used: int = 0,
    ) -> ToolFailure:
        code = self._code(suffix)
        await self._gate.record_failure(
            code,
            provider_credits_used=provider_credits_used,
        )
        return ToolFailure(code=code, retryable=retryable, safe_message=safe_message)

    async def _fresh_expiry(
        self,
        observed_at: datetime,
        *,
        provider_credits_used: int = 0,
    ) -> datetime | ToolFailure:
        now = self._clock.now()
        age_seconds = (now - observed_at).total_seconds()
        if age_seconds > self._max_snapshot_age_seconds:
            return await self._failure(
                "STALE_SNAPSHOT",
                f"{self._provider_name} returned a cryptocurrency snapshot outside the "
                "freshness window.",
                provider_credits_used=provider_credits_used,
            )
        if age_seconds < -self._max_future_skew_seconds:
            return await self._failure(
                "FUTURE_SNAPSHOT",
                f"{self._provider_name} returned a cryptocurrency timestamp too far in the future.",
                provider_credits_used=provider_credits_used,
            )
        remaining_freshness = self._max_snapshot_age_seconds - max(age_seconds, 0)
        ttl_seconds = min(self._evidence_ttl_seconds, int(remaining_freshness))
        if ttl_seconds < 1:
            return await self._failure(
                "STALE_SNAPSHOT",
                f"{self._provider_name} returned a cryptocurrency snapshot with no bounded "
                "freshness remaining.",
                provider_credits_used=provider_credits_used,
            )
        return now + timedelta(seconds=ttl_seconds)

    def _response_provider_credits(self, response: httpx.Response) -> int:
        del response
        return 0

    def _code(self, suffix: str) -> str:
        prefix = "COINGECKO" if self._provider_name == "coingecko" else "COINMARKETCAP"
        return f"{prefix}_{suffix}"


class CoinGeckoMarketSnapshotTool(_CryptoProviderTool):
    """Return one current `/coins/markets` snapshot from CoinGecko."""

    _provider_name: Literal["coingecko"] = "coingecko"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://api.coingecko.com/api/v3",
        api_tier: Literal["demo", "pro"] | None = None,
        gate: ProviderCallGate | None = None,
        max_snapshot_age_seconds: int = 900,
        max_future_skew_seconds: int = 60,
        evidence_ttl_seconds: int = 180,
        max_concurrency: int = 2,
        max_calls_per_minute: int = 20,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
            max_snapshot_age_seconds=max_snapshot_age_seconds,
            max_future_skew_seconds=max_future_skew_seconds,
            evidence_ttl_seconds=evidence_ttl_seconds,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
        )
        inferred_api_tier = "pro" if httpx.URL(base_url).host == "pro-api.coingecko.com" else "demo"
        if api_tier is not None and api_tier != inferred_api_tier:
            raise ValueError("CoinGecko API tier must match the official REST host")
        self._api_tier = inferred_api_tier
        self._spec = ToolSpec(
            name="market.get_crypto_snapshot_coingecko",
            version="1.0.0",
            description=(
                "Get one fresh CoinGecko cryptocurrency price, market cap, 24-hour volume, "
                "and 24-hour change by canonical asset slug."
            ),
            domain="MARKET",
            input_schema=CryptoSnapshotArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=12_288,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = CryptoSnapshotArguments.model_validate(arguments)
        auth_header = "x-cg-pro-api-key" if self._api_tier == "pro" else "x-cg-demo-api-key"
        response = await self._get(
            "coins/markets",
            params={
                "vs_currency": parsed.quote_currency.casefold(),
                "ids": parsed.asset_id,
                "order": "market_cap_desc",
                "per_page": "1",
                "page": "1",
                "sparkline": "false",
                "price_change_percentage": "24h",
                "locale": "en",
                "precision": "full",
            },
            headers={
                "Accept": "application/json",
                auth_header: self._api_key,
            },
        )
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinGecko returned an unsupported cryptocurrency payload.",
            )
        raw_items = (
            payload
            if isinstance(payload, list)
            else payload.get("data")
            if isinstance(payload, dict) and isinstance(payload.get("data"), list)
            else payload.get("coins")
            if isinstance(payload, dict) and isinstance(payload.get("coins"), list)
            else None
        )
        if (
            not isinstance(raw_items, list)
            or len(raw_items) != 1
            or not isinstance(raw_items[0], dict)
        ):
            return await self._failure(
                "ASSET_NOT_FOUND" if raw_items == [] else "SCHEMA_DRIFT",
                (
                    f"CoinGecko returned no exact market snapshot for {parsed.asset_id}."
                    if raw_items == []
                    else "CoinGecko returned an unsupported cryptocurrency payload."
                ),
            )
        item = raw_items[0]
        provider_asset_id = _bounded_slug(item.get("id"))
        name = _bounded_text(item.get("name"), limit=120)
        symbol = _bounded_symbol(item.get("symbol"))
        price = _finite_number(item.get("current_price"), minimum_exclusive=0)
        observed_at = _parse_timestamp(item.get("last_updated"))
        if provider_asset_id != parsed.asset_id or price is None or observed_at is None:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinGecko returned an unsupported cryptocurrency payload.",
            )
        expires_at = await self._fresh_expiry(observed_at)
        if isinstance(expires_at, ToolFailure):
            return expires_at
        await self._gate.record_success()
        health = await self._gate.snapshot()
        provider_reference = (
            f"coins-markets:{provider_asset_id}:{parsed.quote_currency}:{observed_at.isoformat()}"
        )
        try:
            optional_values = {
                "market_cap": _finite_number(item.get("market_cap"), minimum=0),
                "name": name,
                "percent_change_24h": _finite_number(item.get("price_change_percentage_24h")),
                "symbol": symbol,
                "volume_24h": _finite_number(item.get("total_volume"), minimum=0),
            }
            snapshot = CryptoProviderSnapshot(
                provider="coingecko",
                asset_id=parsed.asset_id,
                provider_asset_id=provider_asset_id,
                name=name,
                symbol=symbol,
                quote_currency=parsed.quote_currency,
                price=price,
                market_cap=optional_values["market_cap"],
                volume_24h=optional_values["volume_24h"],
                percent_change_24h=optional_values["percent_change_24h"],
                as_of=observed_at,
                evidence_expires_at=expires_at,
                provider_reference=provider_reference,
                provider_request_id=_safe_request_id(response.headers.get("x-request-id")),
                provider_payload_sha256=hashlib.sha256(response.content).hexdigest(),
                missing_fields=tuple(
                    sorted(key for key, value in optional_values.items() if value is None)
                ),
                health=health,
            )
            provider_payload = CryptoProviderPayload(
                snapshot=snapshot,
                statements=(canonical_crypto_snapshot_statement(snapshot),),
            )
        except ValidationError:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinGecko returned an unsupported cryptocurrency payload.",
            )
        return ToolSuccess(
            data=provider_payload.model_dump(mode="json"),
            source=SourceRef(provider="coingecko", reference=provider_reference),
            observed_at=observed_at,
            expires_at=expires_at,
        )


class CoinMarketCapMarketSnapshotTool(_CryptoProviderTool):
    """Return one current `/v2/simple/price` snapshot from CoinMarketCap."""

    _provider_name: Literal["coinmarketcap"] = "coinmarketcap"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        base_url: str = "https://pro-api.coinmarketcap.com",
        gate: ProviderCallGate | None = None,
        max_snapshot_age_seconds: int = 900,
        max_future_skew_seconds: int = 60,
        evidence_ttl_seconds: int = 180,
        max_concurrency: int = 2,
        max_calls_per_minute: int = 20,
    ) -> None:
        super().__init__(
            client=client,
            api_key=api_key,
            clock=clock,
            base_url=base_url,
            gate=gate,
            max_snapshot_age_seconds=max_snapshot_age_seconds,
            max_future_skew_seconds=max_future_skew_seconds,
            evidence_ttl_seconds=evidence_ttl_seconds,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
        )
        self._spec = ToolSpec(
            name="market.get_crypto_snapshot_coinmarketcap",
            version="1.0.0",
            description=(
                "Get one fresh CoinMarketCap cryptocurrency price, market cap, 24-hour volume, "
                "and 24-hour change by canonical asset slug."
            ),
            domain="MARKET",
            input_schema=CryptoSnapshotArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=12_288,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def _response_provider_credits(self, response: httpx.Response) -> int:
        try:
            payload = response.json()
        except ValueError:
            return 0
        if not isinstance(payload, dict) or not isinstance(payload.get("status"), dict):
            return 0
        return _nonnegative_int(payload["status"].get("credit_count")) or 0

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = CryptoSnapshotArguments.model_validate(arguments)
        response = await self._get(
            "v2/simple/price",
            params={
                "slug": parsed.asset_id,
                "convert": parsed.quote_currency,
                "include_all": "true",
                "skip_invalid": "true",
            },
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "deflate, gzip",
                "X-CMC_PRO_API_KEY": self._api_key,
            },
        )
        if isinstance(response, ToolFailure):
            return response
        try:
            payload = response.json()
        except ValueError:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
            )
        if not isinstance(payload, dict):
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
            )
        status = payload.get("status")
        credit_count = (
            _nonnegative_int(status.get("credit_count")) if isinstance(status, dict) else None
        )
        if not isinstance(status, dict) or status.get("error_code") not in {0, "0"}:
            return await self._failure(
                "PROVIDER_ERROR",
                "CoinMarketCap reported that the cryptocurrency request could not be completed.",
                provider_credits_used=credit_count or 0,
            )
        data = payload.get("data")
        if not isinstance(data, list | dict):
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        if data == [] or data == {}:
            return await self._failure(
                "ASSET_NOT_FOUND",
                f"CoinMarketCap returned no exact market snapshot for {parsed.asset_id}.",
                provider_credits_used=credit_count or 0,
            )
        rows = (
            data
            if isinstance(data, list)
            else [value for value in data.values() if isinstance(value, dict)]
        )
        if len(rows) != 1 or not isinstance(rows[0], dict):
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        item = rows[0]
        provider_asset_id_value = _positive_int(item.get("id"))
        provider_asset_slug = _bounded_slug(item.get("slug"))
        name = _bounded_text(item.get("name"), limit=120)
        symbol = _bounded_symbol(item.get("symbol"))
        quote = _coinmarketcap_quote(item, parsed.quote_currency)
        if quote is None:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        if provider_asset_slug != parsed.asset_id:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        price = _finite_number(quote.get("price"), minimum_exclusive=0)
        observed_at = _parse_timestamp(quote.get("last_updated"))
        if price is None or observed_at is None:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        expires_at = await self._fresh_expiry(
            observed_at,
            provider_credits_used=credit_count or 0,
        )
        if isinstance(expires_at, ToolFailure):
            return expires_at
        await self._gate.record_success(provider_credits_used=credit_count or 0)
        health = await self._gate.snapshot()
        provider_asset_id = (
            str(provider_asset_id_value)
            if provider_asset_id_value is not None
            else provider_asset_slug
        )
        provider_reference = (
            f"simple-price:{provider_asset_id}:{parsed.quote_currency}:{observed_at.isoformat()}"
        )
        try:
            optional_values = {
                "market_cap": _finite_number(quote.get("market_cap"), minimum=0),
                "name": name,
                "percent_change_24h": _finite_number(quote.get("percent_change_24h")),
                "symbol": symbol,
                "volume_24h": _finite_number(quote.get("volume_24h"), minimum=0),
            }
            missing_fields = {key for key, value in optional_values.items() if value is None}
            if provider_asset_id_value is None:
                missing_fields.add("provider_asset_id")
            if credit_count is None:
                missing_fields.add("provider_credits_used")
            snapshot = CryptoProviderSnapshot(
                provider="coinmarketcap",
                asset_id=parsed.asset_id,
                provider_asset_id=provider_asset_id,
                name=name,
                symbol=symbol,
                quote_currency=parsed.quote_currency,
                price=price,
                market_cap=optional_values["market_cap"],
                volume_24h=optional_values["volume_24h"],
                percent_change_24h=optional_values["percent_change_24h"],
                as_of=observed_at,
                evidence_expires_at=expires_at,
                provider_reference=provider_reference,
                provider_request_id=_safe_request_id(response.headers.get("x-request-id")),
                provider_payload_sha256=hashlib.sha256(response.content).hexdigest(),
                provider_credits_used=credit_count or 0,
                missing_fields=tuple(sorted(missing_fields)),
                health=health,
            )
            provider_payload = CryptoProviderPayload(
                snapshot=snapshot,
                statements=(canonical_crypto_snapshot_statement(snapshot),),
            )
        except ValidationError:
            return await self._failure(
                "SCHEMA_DRIFT",
                "CoinMarketCap returned an unsupported cryptocurrency payload.",
                provider_credits_used=credit_count or 0,
            )
        return ToolSuccess(
            data=provider_payload.model_dump(mode="json"),
            source=SourceRef(provider="coinmarketcap", reference=provider_reference),
            observed_at=observed_at,
            expires_at=expires_at,
        )


class CryptoMarketSnapshotTool:
    """Query up to two providers concurrently and retain bounded disagreement evidence."""

    def __init__(
        self,
        providers: Sequence[CryptoSnapshotProvider],
        *,
        agreement_threshold_bps: float = 250.0,
        max_corroboration_skew_seconds: float = 120.0,
    ) -> None:
        names = tuple(provider.provider_name for provider in providers)
        if not 1 <= len(providers) <= 2 or len(names) != len(set(names)):
            raise ValueError("crypto corroboration requires one or two unique providers")
        if any(name not in _PROVIDER_ORDER for name in names):
            raise ValueError("unsupported crypto corroboration provider")
        if not math.isfinite(agreement_threshold_bps) or not 0 <= agreement_threshold_bps <= 10_000:
            raise ValueError("crypto agreement threshold is invalid")
        if (
            not math.isfinite(max_corroboration_skew_seconds)
            or not 0 <= max_corroboration_skew_seconds <= 86_400
        ):
            raise ValueError("crypto corroboration skew threshold is invalid")
        self._providers = tuple(providers)
        self._agreement_threshold_bps = agreement_threshold_bps
        self._max_corroboration_skew_seconds = max_corroboration_skew_seconds
        self._spec = ToolSpec(
            name="market.get_crypto_snapshot",
            version="1.0.0",
            description=(
                "Get a resilient cryptocurrency market snapshot from CoinGecko and/or "
                "CoinMarketCap, retaining exact provider provenance and measured agreement or "
                "divergence only inside a bounded temporal corroboration window; succeeds when "
                "at least one configured provider succeeds."
            ),
            domain="MARKET",
            input_schema=CryptoSnapshotArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=15.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=28_672,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def max_corroboration_skew_seconds(self) -> float:
        """Expose the immutable aggregate policy for audit and acceptance tests."""

        return self._max_corroboration_skew_seconds

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return CryptoSnapshotArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = CryptoSnapshotArguments.model_validate(arguments)
        outcomes = await asyncio.gather(
            *(_safe_provider_execute(provider, arguments, context) for provider in self._providers)
        )
        successes: list[tuple[CryptoProviderSnapshot, ToolSuccess]] = []
        failures: dict[str, str] = {}
        retryable_failures: list[bool] = []
        for provider, outcome in zip(self._providers, outcomes, strict=True):
            if isinstance(outcome, ToolFailure):
                failures[provider.provider_name] = outcome.code
                retryable_failures.append(outcome.retryable)
                continue
            snapshot = _validated_snapshot_from_outcome(provider.provider_name, outcome)
            if snapshot is None:
                failures[provider.provider_name] = "CRYPTO_PROVIDER_CONTRACT_REJECTED"
                retryable_failures.append(False)
                continue
            successes.append((snapshot, outcome))
        if not successes:
            return ToolFailure(
                code="CRYPTO_PROVIDERS_UNAVAILABLE",
                retryable=bool(retryable_failures) and all(retryable_failures),
                safe_message=(
                    "No configured cryptocurrency market provider returned a fresh, valid "
                    "snapshot; the rest of Leo's tools remain available."
                ),
            )
        successes.sort(key=lambda item: _PROVIDER_ORDER[item[0].provider])
        snapshots = tuple(item[0] for item in successes)
        agreement = calculate_crypto_agreement(
            snapshots,
            agreement_threshold_bps=self._agreement_threshold_bps,
            max_corroboration_skew_seconds=self._max_corroboration_skew_seconds,
        )
        statements = [canonical_crypto_snapshot_statement(item) for item in snapshots]
        agreement_statement = canonical_crypto_agreement_statement(
            asset_id=parsed.asset_id,
            quote_currency=parsed.quote_currency,
            agreement=agreement,
        )
        if agreement_statement is not None:
            statements.append(agreement_statement)
        summary = canonical_crypto_aggregate_summary(
            snapshots=snapshots,
            agreement_statement=agreement_statement,
        )
        provenance_digest = crypto_provenance_digest(
            asset_id=parsed.asset_id,
            quote_currency=parsed.quote_currency,
            snapshots=snapshots,
            provider_failures=failures,
            agreement=agreement,
        )
        try:
            payload = CryptoAggregatePayload(
                asset_id=parsed.asset_id,
                quote_currency=parsed.quote_currency,
                snapshots=snapshots,
                providers_succeeded=tuple(item.provider for item in snapshots),
                provider_failures=failures,
                agreement=agreement,
                provenance_digest=provenance_digest,
                statements=tuple(statements),
                summary=summary,
            )
        except ValidationError:
            return ToolFailure(
                code="CRYPTO_AGGREGATE_CONTRACT_REJECTED",
                safe_message="The cryptocurrency provider results could not be safely combined.",
            )
        observed_at = max(item.as_of for item in snapshots)
        expires_at = min(item.evidence_expires_at for item in snapshots)
        return ToolSuccess(
            data=payload.model_dump(mode="json"),
            source=SourceRef(
                provider="crypto-corroboration",
                reference=(
                    f"snapshot:{parsed.asset_id}:{parsed.quote_currency}:{provenance_digest}"
                ),
            ),
            observed_at=observed_at,
            expires_at=expires_at,
        )


async def _safe_provider_execute(
    provider: CryptoSnapshotProvider,
    arguments: dict[str, JsonValue],
    context: ToolExecutionContext,
) -> ToolOutcome:
    try:
        return await provider.execute(
            arguments,
            context.model_copy(
                update={"tool_call_id": f"{context.tool_call_id}:{provider.provider_name}"}
            ),
        )
    except Exception:
        return ToolFailure(
            code=f"{provider.provider_name.upper()}_UNEXPECTED_FAILURE",
            safe_message=(
                f"{provider.provider_name} failed inside the bounded cryptocurrency adapter."
            ),
        )


def _validated_snapshot_from_outcome(
    expected_provider: Literal["coingecko", "coinmarketcap"],
    outcome: ToolSuccess,
) -> CryptoProviderSnapshot | None:
    try:
        payload = CryptoProviderPayload.model_validate(outcome.data)
    except ValidationError:
        return None
    snapshot = payload.snapshot
    if (
        snapshot.provider != expected_provider
        or outcome.source.provider != expected_provider
        or outcome.source.reference != snapshot.provider_reference
        or outcome.observed_at != snapshot.as_of
        or outcome.expires_at != snapshot.evidence_expires_at
    ):
        return None
    return snapshot


def _coinmarketcap_quote(item: dict[str, object], currency: str) -> dict[str, object] | None:
    """Select one exact quote currency across documented response variants."""

    candidates: list[dict[str, object]] = []
    for container_name in ("quotes", "quote"):
        container = item.get(container_name)
        if isinstance(container, list):
            candidates.extend(
                value
                for value in container
                if isinstance(value, dict) and value.get("symbol") == currency
            )
            continue
        if not isinstance(container, dict):
            continue
        nested = container.get(currency)
        if isinstance(nested, dict):
            candidates.append(nested)
        elif container.get("symbol") == currency:
            candidates.append(container)
    unique = tuple({id(candidate): candidate for candidate in candidates}.values())
    return unique[0] if len(unique) == 1 else None


def _finite_number(
    value: object,
    *,
    minimum: float | None = None,
    minimum_exclusive: float | None = None,
) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    if minimum is not None and numeric < minimum:
        return None
    if minimum_exclusive is not None and numeric <= minimum_exclusive:
        return None
    return numeric


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _parse_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or len(value) > 64:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(UTC)


def _bounded_text(value: object, *, limit: int) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", value).split())[:limit]
    return cleaned or None


def _bounded_slug(value: object) -> str | None:
    cleaned = _bounded_text(value, limit=80)
    if cleaned is None or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", cleaned) is None:
        return None
    return cleaned


def _bounded_symbol(value: object) -> str | None:
    cleaned = _bounded_text(value, limit=20)
    if cleaned is None:
        return None
    normalized = cleaned.upper()
    if re.fullmatch(r"[A-Z0-9$@.-]+", normalized) is None:
        return None
    return normalized


def _safe_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None


__all__ = [
    "_COINGECKO_DOC_URL",
    "_COINMARKETCAP_DOC_URL",
    "CoinGeckoMarketSnapshotTool",
    "CoinMarketCapMarketSnapshotTool",
    "CryptoMarketSnapshotTool",
    "CryptoSnapshotProvider",
    "ProviderCallGate",
]
