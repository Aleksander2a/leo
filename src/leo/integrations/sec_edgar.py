"""Narrow, read-only SEC EDGAR recent-filings adapter."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

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


class _RecentFilingsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    ticker: str = Field(min_length=1, max_length=8, pattern=r"^[A-Z][A-Z0-9.-]*$")
    limit: int = Field(default=5, ge=1, le=20)


class SecEdgarRecentFilingsTool:
    """Resolve a configured ticker mapping and return capped recent SEC filings."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        clock: Clock,
        ticker_to_cik: Mapping[str, str] | None = None,
        user_agent: str,
        base_url: str = "https://data.sec.gov/submissions",
        cache_seconds: int = 900,
        max_requests_per_second: float = 8.0,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        if not user_agent.strip() or "@" not in user_agent:
            raise ValueError("SEC User-Agent must identify an operator contact")
        self._client = client
        self._clock = clock
        if cache_seconds < 1 or not 0 < max_requests_per_second <= 10:
            raise ValueError("SEC cache lifetime and request rate are invalid")
        normalized_mapping: dict[str, str] = {}
        for raw_ticker, raw_cik in (ticker_to_cik or {}).items():
            ticker = raw_ticker.strip().upper()
            cik_value = raw_cik.strip()
            if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,7}", ticker):
                raise ValueError("SEC ticker map contains an invalid ticker")
            if not re.fullmatch(r"\d{1,10}", cik_value):
                raise ValueError("SEC ticker map contains an invalid CIK")
            cik = cik_value.zfill(10)
            if ticker in normalized_mapping and normalized_mapping[ticker] != cik:
                raise ValueError("SEC ticker map contains an ambiguous identity")
            normalized_mapping[ticker] = cik
        self._ticker_to_cik = normalized_mapping
        self._user_agent = user_agent
        self._base_url = base_url.rstrip("/")
        self._cache_seconds = cache_seconds
        self._minimum_request_interval = 1.0 / max_requests_per_second
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._last_request_started: float | None = None
        self._cache: dict[tuple[str, int], ToolSuccess] = {}
        self._lock = asyncio.Lock()
        self._directory: dict[str, str] | None = None
        self._directory_lock = asyncio.Lock()
        self._spec = ToolSpec(
            name="sec.get_recent_filings",
            version="1.1.0",
            description="Return capped recent SEC filings for one explicitly mapped ticker.",
            domain="SEC",
            input_schema=_RecentFilingsArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=15.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=24_576,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = _RecentFilingsArguments.model_validate(arguments)
        return {"ticker": parsed.ticker, "limit": parsed.limit}

    async def _resolve_ticker(self, ticker: str) -> str | None:
        """Resolve any EDGAR-registered ticker from SEC's own published index.

        A hardcoded map covered eight symbols, which made this tool useless for
        every other public company. SEC publishes the full ticker-to-CIK index
        itself; it is fetched once per process and cached.
        """

        async with self._directory_lock:
            if self._directory is None:
                try:
                    response = await self._client.get(
                        "https://www.sec.gov/files/company_tickers.json",
                        headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                        timeout=20.0,
                    )
                    response.raise_for_status()
                    payload = response.json()
                except (httpx.HTTPError, ValueError):
                    return None
                directory: dict[str, str] = {}
                entries = payload.values() if isinstance(payload, dict) else payload
                for entry in entries if isinstance(entries, (list, type({}.values()))) else ():
                    if not isinstance(entry, dict):
                        continue
                    symbol = str(entry.get("ticker") or "").strip().upper()
                    cik_value = entry.get("cik_str")
                    if symbol and isinstance(cik_value, (int, str)):
                        directory.setdefault(symbol, str(cik_value).zfill(10))
                self._directory = directory
        return self._directory.get(ticker.upper())

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _RecentFilingsArguments.model_validate(arguments)
        cik = self._ticker_to_cik.get(parsed.ticker) or await self._resolve_ticker(parsed.ticker)
        if cik is None:
            return ToolFailure(
                code="SEC_IDENTITY_UNMAPPED",
                safe_message=(
                    f"SEC has no EDGAR filer registered under the ticker {parsed.ticker}. "
                    "Check the symbol, or use a web source for this company."
                ),
            )
        async with self._lock:
            cache_key = (parsed.ticker, parsed.limit)
            cached = self._cache.get(cache_key)
            now = self._clock.now()
            if cached is not None and cached.expires_at is not None and cached.expires_at > now:
                return cached
            await self._wait_for_request_slot()
            try:
                response = await self._client.get(
                    f"{self._base_url}/CIK{cik}.json",
                    headers={"User-Agent": self._user_agent, "Accept": "application/json"},
                )
            except httpx.TimeoutException:
                return ToolFailure(
                    code="SEC_TIMEOUT",
                    retryable=True,
                    safe_message="SEC did not respond before the adapter timeout.",
                )
            except httpx.TransportError:
                return ToolFailure(
                    code="SEC_TRANSPORT_ERROR",
                    retryable=True,
                    safe_message="SEC request failed before a response was received.",
                )
            if response.status_code == 429:
                return ToolFailure(
                    code="SEC_RATE_LIMITED",
                    retryable=True,
                    safe_message="SEC rate limit was reached.",
                )
            if response.status_code == 403:
                return ToolFailure(
                    code="SEC_ACCESS_DENIED",
                    safe_message="SEC denied this adapter request.",
                )
            if response.status_code >= 500:
                return ToolFailure(
                    code="SEC_UNAVAILABLE",
                    retryable=True,
                    safe_message=f"SEC returned HTTP {response.status_code}.",
                )
            if response.status_code >= 400:
                return ToolFailure(
                    code="SEC_REQUEST_REJECTED",
                    safe_message=f"SEC returned HTTP {response.status_code}.",
                )
            try:
                payload = response.json()
                recent = payload["filings"]["recent"]
                forms = recent["form"]
                accessions = recent["accessionNumber"]
                filing_dates = recent["filingDate"]
                primary_documents = recent["primaryDocument"]
            except (KeyError, IndexError, TypeError, ValueError):
                return ToolFailure(
                    code="SEC_SCHEMA_DRIFT",
                    safe_message="SEC returned an unsupported filing payload.",
                )
            arrays = (forms, accessions, filing_dates, primary_documents)
            if (
                not all(isinstance(items, list) for items in arrays)
                or len({len(items) for items in arrays if isinstance(items, list)}) != 1
            ):
                return ToolFailure(
                    code="SEC_SCHEMA_DRIFT",
                    safe_message="SEC returned malformed filing arrays.",
                )
            filings: list[JsonValue] = []
            for values in zip(forms, accessions, filing_dates, primary_documents, strict=True):
                form, accession, filing_date, document = values
                if not _valid_filing_entry(form, accession, filing_date, document):
                    continue
                assert all(isinstance(item, str) for item in values)
                accession_path = accession.replace("-", "")
                filings.append(
                    {
                        "form": form,
                        "accession": accession,
                        "filing_date": filing_date,
                        "primary_document": document,
                        "filing_url": (
                            "https://www.sec.gov/Archives/edgar/data/"
                            f"{int(cik)}/{accession_path}/{document}"
                        ),
                    }
                )
                if len(filings) == parsed.limit:
                    break
            if not filings:
                return ToolFailure(
                    code="SEC_NO_FILINGS",
                    safe_message="SEC returned no valid recent filings for the mapped ticker.",
                )
            data: dict[str, JsonValue] = {
                "ticker": parsed.ticker,
                "cik": cik,
                "filings": filings,
            }
            company_name = payload.get("name") if isinstance(payload, dict) else None
            if isinstance(company_name, str) and company_name.strip():
                data["company_name"] = company_name.strip()[:240]
            request_id = _safe_request_id(response.headers.get("x-request-id"))
            if request_id is not None:
                data["provider_request_id"] = request_id
            success = ToolSuccess(
                data=data,
                source=SourceRef(
                    provider="sec-edgar",
                    reference=f"submissions:{cik}",
                    url=f"https://data.sec.gov/submissions/CIK{cik}.json",
                ),
                observed_at=now,
                expires_at=now + timedelta(seconds=self._cache_seconds),
            )
            self._cache[cache_key] = success
            return success

    async def _wait_for_request_slot(self) -> None:
        current = self._monotonic()
        if self._last_request_started is not None:
            remaining = self._minimum_request_interval - (current - self._last_request_started)
            if remaining > 0:
                await self._sleeper(remaining)
                current = self._monotonic()
        self._last_request_started = current


def _valid_filing_entry(
    form: object,
    accession: object,
    filing_date: object,
    document: object,
) -> bool:
    if not (
        isinstance(form, str)
        and re.fullmatch(r"[A-Za-z0-9-]{1,24}", form) is not None
        and isinstance(accession, str)
        and re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession)
        and isinstance(filing_date, str)
        and isinstance(document, str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", document) is not None
        and ".." not in document
    ):
        return False
    try:
        date.fromisoformat(filing_date)
    except ValueError:
        return False
    return True


def _safe_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None
