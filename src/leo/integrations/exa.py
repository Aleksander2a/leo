"""Bounded raw-HTTP Exa search adapter with URL-bound highlight evidence.

Leo uses Exa's Search endpoint directly.  It does not use Exa Agent or an
external agent framework.  The request deliberately selects one content mode:
highlights.  A successful observation exposes only the first result with a public
URL and nonempty bounded highlights, so its canonical statements and ``SourceRef``
remain bound to one exact URL.  Optional title/score omissions are marked; missing
claim-bearing fields fail closed and never become evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.harness.exa_search import (
    canonical_exa_highlight_statement,
    normalize_complete_exa_result,
)
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
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRejected,
    bounded_retry_after,
)

_EXA_SEARCH_URL = "https://api.exa.ai/search"
_MAX_RESPONSE_BYTES = 1_048_576
_MAX_PROVIDER_RESULTS = 100


@dataclass(frozen=True, slots=True)
class ExaCapabilityDescriptor:
    """Provider-owned discovery metadata for composition catalogs."""

    provider: str
    tags: frozenset[str]
    freshness_seconds: int
    max_calls_per_minute: int
    verification_expectations: frozenset[str]


EXA_CAPABILITY_DESCRIPTOR = ExaCapabilityDescriptor(
    provider="exa",
    tags=frozenset(
        {
            "web",
            "internet",
            "search",
            "find",
            "research",
            "source",
            "news",
            "current",
            "comparison",
            "landscape",
            "outlook",
            "forecast",
            "forecasts",
            "prediction",
            "predictions",
            "prognosis",
            "analysis",
            "opinion",
            "sentiment",
            "index",
            "indices",
            "trend",
            "trends",
        }
    ),
    freshness_seconds=600,
    max_calls_per_minute=10,
    verification_expectations=frozenset(
        {"canonical_statement", "exact_url_bound_highlight", "untrusted_content"}
    ),
)


class _ExaSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=512)


class ExaSearchTool:
    """Search Exa and return one complete, exact-URL highlight result."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        evidence_ttl_seconds: int = 600,
        max_concurrency: int = 4,
        max_calls_per_minute: int = 10,
        gate: ProviderCallGate | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("Exa API key is required")
        if evidence_ttl_seconds < 1 or max_concurrency < 1 or max_calls_per_minute < 1:
            raise ValueError("Exa TTL and concurrency limits must be positive")
        self._client = client
        self._api_key = api_key
        self._clock = clock
        self._ttl_seconds = evidence_ttl_seconds
        self._gate = gate or ProviderCallGate(
            provider="exa",
            clock=clock,
            max_concurrency=max_concurrency,
            max_calls_per_minute=max_calls_per_minute,
        )
        self._spec = ToolSpec(
            name="web.search_exa",
            version="1.0.0",
            description=(
                "Search the public web through Exa raw Search with type auto and highlights. "
                "Returns canonical excerpts bound to one exact public result URL; missing "
                "optional title or score metadata is marked explicitly."
            ),
            domain="WEB",
            input_schema=_ExaSearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=15.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=16_384,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _ExaSearchArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _ExaSearchArguments.model_validate(arguments)
        # Keep the provider request intentionally minimal and stable.  In
        # particular, contents owns highlights and no second content mode is sent.
        request_payload: dict[str, JsonValue] = {
            "query": parsed.query,
            "type": "auto",
            "contents": {"highlights": True},
        }
        try:
            async with self._gate.slot():
                response = await self._client.post(
                    _EXA_SEARCH_URL,
                    headers={
                        "x-api-key": self._api_key,
                        "Accept": "application/json",
                        "Content-Type": "application/json",
                    },
                    json=request_payload,
                    # Credential egress stays pinned even when a caller supplies a
                    # shared client configured to follow redirects globally.
                    follow_redirects=False,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except httpx.TimeoutException:
            return await self._record_failure(
                _failure(
                    "EXA_TIMEOUT",
                    "Exa did not respond before the adapter timeout.",
                    retryable=True,
                )
            )
        except httpx.TransportError:
            return await self._record_failure(
                _failure(
                    "EXA_TRANSPORT_ERROR",
                    "Exa failed before a response was received.",
                    retryable=True,
                )
            )

        http_failure = _http_failure(response.status_code)
        if http_failure is not None:
            return await self._record_failure(
                http_failure,
                rate_limited=response.status_code == 429,
                retry_after_seconds=bounded_retry_after(response.headers.get("retry-after")),
            )
        if len(response.content) > _MAX_RESPONSE_BYTES:
            return await self._record_failure(
                _failure(
                    "EXA_RESPONSE_TOO_LARGE",
                    "Exa returned an oversized search payload.",
                )
            )
        try:
            payload = response.json()
        except ValueError:
            return await self._record_failure(_schema_failure())
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            return await self._record_failure(_schema_failure())
        raw_results = payload["results"]
        if not raw_results:
            return await self._record_failure(
                _failure("EXA_NO_RESULTS", "Exa returned no search results for the query.")
            )
        if len(raw_results) > _MAX_PROVIDER_RESULTS:
            return await self._record_failure(_schema_failure())

        selected: dict[str, JsonValue] | None = None
        selected_rank = 0
        for index, raw_result in enumerate(raw_results, start=1):
            normalized = normalize_complete_exa_result(raw_result)
            if normalized is not None:
                selected = normalized
                selected_rank = index
                break
        if selected is None:
            return await self._record_failure(
                _failure(
                    "EXA_NO_COMPLETE_HIGHLIGHTS",
                    "Exa returned no structurally complete URL-bound highlights.",
                )
            )

        title = selected["title"]
        url = selected["url"]
        highlights = selected["highlights"]
        assert title is None or isinstance(title, str)
        assert isinstance(url, str)
        assert isinstance(highlights, list)
        statements: list[JsonValue] = [
            canonical_exa_highlight_statement(title=title, url=url, highlight=highlight)
            for highlight in highlights
            if isinstance(highlight, str)
        ]
        result_hash = hashlib.sha256(
            json.dumps(selected, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        query_hash = hashlib.sha256(parsed.query.encode("utf-8")).hexdigest()
        data: dict[str, JsonValue] = {
            "query": parsed.query,
            "query_hash": query_hash,
            "search_type": "auto",
            "contents_mode": "highlights",
            "provider_result_count": len(raw_results),
            "selected_result_rank": selected_rank,
            "result": selected,
            "result_hash": result_hash,
            "highlight_count": len(statements),
            "statements": statements,
            "untrusted": True,
            "exact_url_bound_claims": True,
        }
        request_id = _safe_request_id(payload.get("requestId"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        now = self._clock.now()
        await self._gate.record_success()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="exa",
                reference=f"search:{query_hash}:{result_hash}",
                url=url,
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
    ) -> ToolFailure:
        await self._gate.record_failure(
            failure.code,
            rate_limited=rate_limited,
            retry_after_seconds=retry_after_seconds,
        )
        return failure


def _http_failure(status_code: int) -> ToolFailure | None:
    if status_code == 402:
        return _failure(
            "EXA_CREDITS_EXHAUSTED",
            "Exa credits or the configured API-key budget are exhausted.",
        )
    if status_code == 429:
        return _failure("EXA_RATE_LIMITED", "Exa rate limit was reached.", retryable=True)
    if status_code >= 500:
        return _failure(
            "EXA_UNAVAILABLE",
            f"Exa returned HTTP {status_code}.",
            retryable=True,
        )
    if status_code == 401:
        return _failure("EXA_AUTH_REJECTED", "Exa rejected the configured API key.")
    if status_code == 403:
        return _failure("EXA_ACCESS_DENIED", "Exa denied access for the configured account.")
    if status_code >= 400:
        return _failure("EXA_REQUEST_REJECTED", f"Exa returned HTTP {status_code}.")
    return None


def _failure(code: str, message: str, *, retryable: bool = False) -> ToolFailure:
    return ToolFailure(code=code, safe_message=message, retryable=retryable)


def _schema_failure() -> ToolFailure:
    return _failure("EXA_SCHEMA_DRIFT", "Exa returned an unsupported search payload.")


def _safe_request_id(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None


__all__ = ["EXA_CAPABILITY_DESCRIPTOR", "ExaCapabilityDescriptor", "ExaSearchTool"]
