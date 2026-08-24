"""Bounded public web-search discovery backed by Wikipedia OpenSearch."""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import timedelta
from urllib.parse import quote_plus, urlsplit

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

# Wikimedia's robot policy requires a descriptive User-Agent with contact info.
DEFAULT_WEB_SEARCH_USER_AGENT = "LeoResearchAgent/1.0 (https://github.com/Aleksander2a/leo)"


class _SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=256)
    limit: int = Field(default=5, ge=1, le=5)


class PublicWebSearchTool:
    """Discover capped public URLs; callers must fetch a result before citing its text."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        clock: Clock,
        base_url: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str | None = None,
        max_concurrency: int = 4,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("web-search concurrency must be positive")
        parsed_base = urlsplit(base_url)
        if parsed_base.scheme != "https" or not parsed_base.hostname:
            raise ValueError("web-search base URL must be public HTTPS")
        self._client = client
        self._clock = clock
        self._base_url = base_url
        # Wikimedia enforces its robot policy by rejecting User-Agent-less API
        # requests with HTTP 403 (phabricator T400119). The adapter sent only an
        # Accept header, so *every* public search failed with a non-retryable
        # WEB_SEARCH_REQUEST_REJECTED -- which in turn failed the whole run.
        self._user_agent = user_agent or DEFAULT_WEB_SEARCH_USER_AGENT
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._spec = ToolSpec(
            name="web.search_public",
            version="1.0.0",
            description=(
                "Search Wikipedia for up to five public pages. Results are untrusted discovery "
                "metadata, not factual evidence; fetch a selected URL before making a claim."
            ),
            domain="WEB",
            input_schema=_SearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=10.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=16_384,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _SearchArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _SearchArguments.model_validate(arguments)
        try:
            async with self._semaphore:
                response = await self._client.get(
                    self._base_url,
                    params={
                        "action": "opensearch",
                        "search": parsed.query,
                        "limit": str(parsed.limit),
                        "namespace": "0",
                        "format": "json",
                        "origin": "*",
                    },
                    headers={
                        "Accept": "application/json",
                        "User-Agent": self._user_agent,
                    },
                )
        except httpx.TimeoutException:
            return ToolFailure(
                code="WEB_SEARCH_TIMEOUT",
                retryable=True,
                safe_message="Public search did not respond before the adapter timeout.",
            )
        except httpx.TransportError:
            return ToolFailure(
                code="WEB_SEARCH_TRANSPORT_ERROR",
                retryable=True,
                safe_message="Public search failed before a response was received.",
            )
        if response.status_code == 429:
            return ToolFailure(
                code="WEB_SEARCH_RATE_LIMITED",
                retryable=True,
                safe_message="Public search rate limit was reached.",
            )
        if response.status_code >= 500:
            return ToolFailure(
                code="WEB_SEARCH_UNAVAILABLE",
                retryable=True,
                safe_message=f"Public search returned HTTP {response.status_code}.",
            )
        if response.status_code >= 400:
            return ToolFailure(
                code="WEB_SEARCH_REQUEST_REJECTED",
                safe_message=f"Public search returned HTTP {response.status_code}.",
            )
        try:
            payload = response.json()
        except ValueError:
            return _schema_failure()
        if not (
            isinstance(payload, list)
            and len(payload) == 4
            and isinstance(payload[0], str)
            and all(isinstance(items, list) for items in payload[1:])
            and len({len(items) for items in payload[1:]}) == 1
        ):
            return _schema_failure()
        titles, descriptions, urls = payload[1:]
        results: list[JsonValue] = []
        for title, description, url in zip(titles, descriptions, urls, strict=True):
            if not (
                isinstance(title, str)
                and isinstance(description, str)
                and isinstance(url, str)
                and _is_allowed_result_url(url)
            ):
                continue
            clean_title = _clean_text(title, limit=240)
            clean_description = _clean_text(description, limit=1_000)
            if not clean_title:
                continue
            results.append(
                {
                    "title": clean_title,
                    "description": clean_description,
                    "url": url[:2_048],
                }
            )
            if len(results) == parsed.limit:
                break
        # An empty result set is a legitimate *answer* from the provider, not an
        # adapter error. Returning a failure here made a normal "nothing matched"
        # outcome indistinguishable from a broken provider, and -- because tool
        # failures terminate the run -- silently ended the whole Slack turn. The
        # model now observes the empty result and can rephrase or switch route.
        query_hash = hashlib.sha256(parsed.query.encode("utf-8")).hexdigest()
        data: dict[str, JsonValue] = {
            "query": parsed.query,
            "query_hash": query_hash,
            "results": results,
            "result_count": len(results),
        }
        request_id = _safe_request_id(response.headers.get("x-request-id"))
        if request_id is not None:
            data["provider_request_id"] = request_id
        now = self._clock.now()
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider="wikipedia-opensearch",
                reference=f"search:{query_hash}",
                url=(
                    "https://en.wikipedia.org/wiki/Special:Search?search="
                    f"{quote_plus(parsed.query)}"
                ),
            ),
            observed_at=now,
            expires_at=now + timedelta(minutes=10),
        )


def _schema_failure() -> ToolFailure:
    return ToolFailure(
        code="WEB_SEARCH_SCHEMA_DRIFT",
        safe_message="Public search returned an unsupported result payload.",
    )


def _is_allowed_result_url(value: str) -> bool:
    if len(value) > 2_048:
        return False
    parsed = urlsplit(value)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "en.wikipedia.org"
        and not parsed.username
        and not parsed.password
        and parsed.path.startswith("/wiki/")
    )


def _clean_text(value: str, *, limit: int) -> str:
    without_controls = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    return " ".join(without_controls.split())[:limit]


def _safe_request_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None
