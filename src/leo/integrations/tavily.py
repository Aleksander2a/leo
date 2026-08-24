"""Bounded Tavily search discovery adapter.

Search snippets are untrusted discovery metadata.  They intentionally normalize as
``DISCOVERY_ONLY`` and must be followed by ``web.fetch_public_text`` before they can
support an external source claim.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, timedelta
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

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
from leo.providers.health import ProviderHealthSnapshot
from leo.url_policy import is_public_https_url

_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_PROVIDER_RESULTS = 100
TAVILY_FREE_TIER_MONTHLY_CREDITS = 1_000


class _TavilySearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=256)
    max_results: int = Field(default=5, ge=1, le=5)
    search_depth: Literal["basic", "advanced"] = "basic"
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None
    include_domains: tuple[str, ...] = Field(default=(), max_length=5)
    exclude_domains: tuple[str, ...] = Field(default=(), max_length=5)
    start_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    end_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("include_domains", "exclude_domains")
    @classmethod
    def normalize_domains(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized: list[str] = []
        for value in values:
            domain = value.strip().casefold().rstrip(".")
            if not _valid_domain_filter(domain):
                raise ValueError("Tavily domain filters must be plain public DNS names")
            if domain not in normalized:
                normalized.append(domain)
        return tuple(normalized)

    @model_validator(mode="after")
    def filters_are_bounded_and_unambiguous(self) -> _TavilySearchArguments:
        if set(self.include_domains).intersection(self.exclude_domains):
            raise ValueError("Tavily include and exclude domain filters cannot overlap")
        explicit_dates = self.start_date is not None or self.end_date is not None
        if explicit_dates and (self.start_date is None or self.end_date is None):
            raise ValueError("Tavily explicit date filtering requires both start and end dates")
        if explicit_dates and self.time_range is not None:
            raise ValueError("Tavily time_range cannot be combined with explicit dates")
        if self.start_date is not None and self.end_date is not None:
            start = date.fromisoformat(self.start_date)
            end = date.fromisoformat(self.end_date)
            if start > end or (end - start).days > 366:
                raise ValueError("Tavily date window must be ordered and at most 366 days")
        return self


class TavilySearchTool:
    """Return capped public discovery results from Tavily's official Search API."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        evidence_ttl_seconds: int = 600,
        max_concurrency: int = 4,
        max_calls_per_minute: int = 10,
        max_calls_per_month: int = 500,
        gate: ProviderCallGate | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Tavily API key is required")
        if (
            evidence_ttl_seconds < 1
            or max_concurrency < 1
            or max_calls_per_minute < 1
            or not 1 <= max_calls_per_month <= 500
        ):
            raise ValueError("Tavily TTL, concurrency, and free-tier limits are invalid")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._gate = gate or ProviderCallGate(
            provider="tavily",
            clock=clock,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
            max_calls_per_month=max_calls_per_month,
            max_provider_credits_per_month=TAVILY_FREE_TIER_MONTHLY_CREDITS,
        )
        if self._gate.provider != "tavily":
            raise ValueError("Tavily gate authority is mismatched")
        self._spec = ToolSpec(
            name="web.search_tavily",
            version="1.0.0",
            description=(
                "Search the public web with Tavily for up to five URLs. Returned snippets are "
                "untrusted discovery metadata; fetch a selected URL before citing its text."
            ),
            domain="WEB",
            input_schema=_TavilySearchArguments.model_json_schema(),
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
        return _TavilySearchArguments.model_validate(arguments).model_dump(
            mode="json", exclude_none=True
        )

    async def provider_health(self) -> ProviderHealthSnapshot:
        """Return the process-local gate snapshot without request or response content."""

        return await self._gate.snapshot()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _TavilySearchArguments.model_validate(arguments)
        request_payload: dict[str, JsonValue] = {
            "query": parsed.query,
            "max_results": parsed.max_results,
            "search_depth": parsed.search_depth,
            "topic": parsed.topic,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
            "auto_parameters": False,
        }
        if parsed.time_range is not None:
            request_payload["time_range"] = parsed.time_range
        if parsed.include_domains:
            request_payload["include_domains"] = list(parsed.include_domains)
        if parsed.exclude_domains:
            request_payload["exclude_domains"] = list(parsed.exclude_domains)
        if parsed.start_date is not None and parsed.end_date is not None:
            request_payload["start_date"] = parsed.start_date
            request_payload["end_date"] = parsed.end_date
        try:
            async with self._gate.slot():
                response = await self._client.post(
                    _TAVILY_SEARCH_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self._record_failure(
                _failure(
                    "TAVILY_TIMEOUT",
                    "Tavily did not respond before the adapter timeout.",
                    retryable=True,
                )
            )
        except httpx.TransportError:
            return await self._record_failure(
                _failure(
                    "TAVILY_TRANSPORT_ERROR",
                    "Tavily failed before a response was received.",
                    retryable=True,
                )
            )
        http_failure = _http_failure(response.status_code)
        if http_failure is not None:
            return await self._record_failure(
                http_failure,
                rate_limited=http_failure.code == "TAVILY_RATE_LIMITED",
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        provider_credit_cost = 2 if parsed.search_depth == "advanced" else 1
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self._record_failure(
                _failure(
                    "TAVILY_RESPONSE_TOO_LARGE",
                    "Tavily returned a response larger than the adapter safety limit.",
                ),
                provider_credits_used=provider_credit_cost,
            )
        try:
            payload = response.json()
        except ValueError:
            return await self._record_failure(
                _schema_failure(),
                provider_credits_used=provider_credit_cost,
            )
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return await self._record_failure(
                _schema_failure(),
                provider_credits_used=provider_credit_cost,
            )
        raw_results = payload["results"]
        if len(raw_results) > _MAX_PROVIDER_RESULTS:
            return await self._record_failure(
                _schema_failure(),
                provider_credits_used=provider_credit_cost,
            )

        results: list[JsonValue] = []
        rejected_result_count = 0
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                rejected_result_count += 1
                continue
            title = raw_result.get("title")
            url = raw_result.get("url")
            content = raw_result.get("content")
            score = raw_result.get("score")
            required_fields_valid = (
                isinstance(title, str)
                and isinstance(url, str)
                and isinstance(content, str)
                and len(url) <= 2_048
                and is_public_https_url(url)
            )
            score_missing = score is None
            score_valid = score_missing or (
                isinstance(score, int | float) and not isinstance(score, bool) and 0 <= score <= 1
            )
            if not (required_fields_valid and score_valid):
                rejected_result_count += 1
                continue
            assert isinstance(title, str)
            assert isinstance(content, str)
            clean_title = _clean_text(title, limit=240)
            clean_content = _clean_text(content, limit=1_200)
            if not clean_title or not clean_content:
                rejected_result_count += 1
                continue
            if len(results) < parsed.max_results:
                normalized_score: JsonValue = None
                if not score_missing:
                    assert isinstance(score, int | float) and not isinstance(score, bool)
                    normalized_score = float(score)
                normalized_result: dict[str, JsonValue] = {
                    "title": clean_title,
                    "url": url,
                    "snippet": clean_content,
                    "score": normalized_score,
                }
                if score_missing:
                    normalized_result["missing_fields"] = ["score"]
                results.append(normalized_result)
        if not results:
            return await self._record_failure(
                _failure(
                    "TAVILY_NO_RESULTS",
                    "Tavily returned no valid public results for the query.",
                ),
                provider_credits_used=provider_credit_cost,
            )

        query_hash = hashlib.sha256(parsed.query.encode("utf-8")).hexdigest()
        request_id = _safe_request_id(payload.get("request_id"))
        data: dict[str, JsonValue] = {
            "query": parsed.query,
            "query_hash": query_hash,
            "topic": parsed.topic,
            "search_depth": parsed.search_depth,
            "results": results,
            "result_count": len(results),
            "rejected_result_count": rejected_result_count,
        }
        if request_id is not None:
            data["provider_request_id"] = request_id
        now = self._clock.now()
        await self._gate.record_success(provider_credits_used=provider_credit_cost)
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="tavily",
                reference=f"search:{query_hash}",
                url="https://docs.tavily.com/documentation/api-reference/endpoint/search",
            ),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )

    async def _record_failure(
        self,
        failure: ToolFailure,
        *,
        rate_limited: bool = False,
        retry_after_seconds: int | None = None,
        provider_credits_used: int = 0,
    ) -> ToolFailure:
        await self._gate.record_failure(
            failure.code,
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
            provider_credits_used=provider_credits_used,
        )
        return failure


def _http_failure(status_code: int) -> ToolFailure | None:
    if status_code == 429:
        return _failure(
            "TAVILY_RATE_LIMITED",
            "Tavily rate limit was reached.",
            retryable=True,
        )
    if status_code >= 500:
        return _failure(
            "TAVILY_UNAVAILABLE",
            f"Tavily returned HTTP {status_code}.",
            retryable=True,
        )
    if status_code >= 400:
        return _failure(
            "TAVILY_REQUEST_REJECTED",
            f"Tavily returned HTTP {status_code}.",
        )
    return None


def _failure(code: str, message: str, *, retryable: bool = False) -> ToolFailure:
    return ToolFailure(code=code, safe_message=message, retryable=retryable)


def _schema_failure() -> ToolFailure:
    return _failure(
        "TAVILY_SCHEMA_DRIFT",
        "Tavily returned an unsupported search payload.",
    )


def _valid_domain_filter(value: str) -> bool:
    if not 1 <= len(value) <= 253 or value == "localhost" or "." not in value:
        return False
    if urlsplit(f"//{value}").hostname != value:
        return False
    labels = value.split(".")
    return all(
        1 <= len(label) <= 63
        and re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label) is not None
        for label in labels
    )


def _clean_text(value: str, *, limit: int) -> str:
    without_controls = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    return " ".join(without_controls.split())[:limit]


def _safe_request_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None
