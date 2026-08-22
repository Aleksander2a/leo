"""Bounded verified-web provider family with fail-closed cross-provider fallback.

The composite keeps provider failover outside model deliberation.  Exa may supply
one structurally complete exact-URL highlight result directly.  Every typed Exa
failure falls through once to Tavily discovery and a public-text fetch; Tavily
snippets are never returned as evidence.  The normalized success retains the
selected provider's evidence shape plus a bounded, secret-free attempt ledger.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.harness.models import (
    RunPhase,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRetryPolicy,
    ToolSpec,
    ToolSuccess,
)
from leo.harness.web_research import rank_tavily_result_urls
from leo.integrations.exa import ExaSearchTool
from leo.integrations.tavily import TavilySearchTool
from leo.integrations.web_fetch import PublicTextFetchTool

_PROVIDER_FAMILY_VERSION = "verified-web-provider-family-v1"


class _VerifiedWebArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=256)


class VerifiedWebResearchTool:
    """Return claim-eligible web evidence after bounded provider-local recovery."""

    def __init__(
        self,
        *,
        exa: ExaSearchTool | None,
        tavily: TavilySearchTool | None,
        fetch: PublicTextFetchTool | None,
    ) -> None:
        if exa is None and tavily is None:
            raise ValueError("At least one verified-web search provider is required")
        if tavily is not None and fetch is None:
            raise ValueError("Tavily verified-web fallback requires a public-text fetcher")
        self._exa = exa
        self._tavily = tavily
        self._fetch = fetch
        self._spec = ToolSpec(
            name="web.research_verified",
            version="1.0.0",
            description=(
                "Get URL-bound public-web evidence with bounded provider failover. Exa "
                "complete highlights may support exact claims; otherwise Tavily is used "
                "only for discovery and an eligible page is fetched before any claim."
            ),
            domain="WEB",
            input_schema=_VerifiedWebArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=105.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=40_960,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _VerifiedWebArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = _VerifiedWebArguments.model_validate(arguments)
        attempts: list[dict[str, JsonValue]] = []

        if self._exa is not None:
            try:
                exa_outcome = await self._exa.execute({"query": parsed.query}, context)
            except Exception:  # Provider boundary must fail over without leaking details.
                attempts.append(_failure_attempt("exa", "search", "EXA_ADAPTER_EXCEPTION"))
            else:
                if isinstance(exa_outcome, ToolSuccess):
                    attempts.append(_success_attempt("exa", "search"))
                    return _family_success(
                        exa_outcome,
                        selected_provider="exa",
                        attempts=attempts,
                    )
                attempts.append(_failure_attempt("exa", "search", exa_outcome.code))

        if self._tavily is not None and self._fetch is not None:
            try:
                tavily_outcome = await self._tavily.execute(
                    {
                        "query": parsed.query,
                        "max_results": 5,
                        "search_depth": "advanced",
                        "topic": "general",
                    },
                    context,
                )
            except Exception:  # Provider boundary emits only a stable safe code.
                attempts.append(_failure_attempt("tavily", "search", "TAVILY_ADAPTER_EXCEPTION"))
                return _providers_exhausted(attempts)
            if isinstance(tavily_outcome, ToolFailure):
                attempts.append(_failure_attempt("tavily", "search", tavily_outcome.code))
                return _providers_exhausted(attempts)

            ranked_urls = rank_tavily_result_urls(
                tavily_outcome.data.get("results"),
                parsed.query,
            )
            if not ranked_urls:
                attempts.append(_failure_attempt("tavily", "search", "TAVILY_NO_ARTICLE_RESULTS"))
                return _providers_exhausted(attempts)
            attempts.append(_success_attempt("tavily", "search"))
            try:
                fetch_outcome = await self._fetch.execute(
                    {
                        "url": ranked_urls[0],
                        "fallback_urls": list(ranked_urls[1:5]),
                    },
                    context,
                )
            except Exception:  # Fetch implementation details are never user-visible.
                attempts.append(
                    _failure_attempt("public-web", "fetch", "PUBLIC_FETCH_ADAPTER_EXCEPTION")
                )
                return _providers_exhausted(attempts)
            if isinstance(fetch_outcome, ToolFailure):
                attempts.append(_failure_attempt("public-web", "fetch", fetch_outcome.code))
                return _providers_exhausted(attempts)
            attempts.append(_success_attempt("public-web", "fetch"))
            return _family_success(
                fetch_outcome,
                selected_provider="tavily_public_fetch",
                attempts=attempts,
            )

        return _providers_exhausted(attempts)


def _family_success(
    outcome: ToolSuccess,
    *,
    selected_provider: str,
    attempts: list[dict[str, JsonValue]],
) -> ToolSuccess:
    data = dict(outcome.data)
    serialized_attempts: list[JsonValue] = [dict(attempt) for attempt in attempts]
    data.update(
        {
            "provider_family_version": _PROVIDER_FAMILY_VERSION,
            "selected_provider": selected_provider,
            "provider_attempt_count": len(attempts),
            "provider_attempts": serialized_attempts,
        }
    )
    return ToolSuccess(
        data=data,
        source=outcome.source,
        observed_at=outcome.observed_at,
        expires_at=outcome.expires_at,
    )


def _success_attempt(provider: str, stage: str) -> dict[str, JsonValue]:
    return {"provider": provider, "stage": stage, "status": "succeeded"}


def _failure_attempt(provider: str, stage: str, code: str) -> dict[str, JsonValue]:
    return {"provider": provider, "stage": stage, "status": "failed", "code": code[:96]}


def _providers_exhausted(attempts: list[dict[str, JsonValue]]) -> ToolFailure:
    codes = tuple(
        str(attempt["code"])
        for attempt in attempts
        if attempt.get("status") == "failed" and isinstance(attempt.get("code"), str)
    )
    summary = "; ".join(codes[:3]) or "NO_PROVIDER_SUCCEEDED"
    return ToolFailure(
        code="VERIFIED_WEB_PROVIDERS_EXHAUSTED",
        retryable=False,
        safe_message=f"Verified web providers were exhausted ({summary}).",
    )


__all__ = ["VerifiedWebResearchTool"]
