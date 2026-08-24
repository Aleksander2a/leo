"""Redundant MCP-sourced research tools, additive alongside Leo's native REST adapters.

Discovery against the real servers this session found: Tavily and Alpha Vantage's
hosted MCP endpoints authorize cleanly with the credentials already in `.env`
(``TAVILY_ENDPOINT``'s embedded query key, ``ALPHA_VANTAGE_ENDPOINT_LEGACY``'s
embedded query key) and expose rich tool catalogs -- Tavily's ``tavily_research``
in particular is a genuinely new capability (bounded deep research), not just a
redundant path to something Leo already has. CoinGecko's *free* MCP endpoint
(``https://mcp.coingecko.com/mcp``) needs no credential at all. CoinGecko's
*Pro* MCP endpoint (what ``COINGECKO_ENDPOINT`` configures) is a full OAuth 2.0
protected resource that requires an interactive browser authorization-code
flow on first connect (confirmed via its ``WWW-Authenticate``/
``.well-known/oauth-protected-resource`` metadata) -- architecturally out of
reach for a headless backend, so this module uses the free endpoint instead.
Massive's hosted ``mcp.massive.com`` expects a JWT that ``MASSIVE_API_KEY``
(a plain REST key) cannot satisfy, and isn't documented publicly; it is not
wired up here.

Every tool built here reports an observation kind under the `market.*`/`web.*`
prefixes the harness already treats as a relaxed-integration source (see
``leo.harness.verifier._RELAXED_INTEGRATION_KINDS``), so no exact-statement
grounding rule is registered or required -- the raw MCP result text is passed
to the model as trusted context, mirroring how ``VerifiedWebResearchTool``
already treats Tavily/Exa payloads as model-synthesized rather than
copy-exact prose.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

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
from leo.integrations.mcp_client import McpToolCallError, call_mcp_tool
from leo.integrations.provider_runtime import ProviderCallGate, ProviderGateRejected

_MAX_RESULT_CHARS = 6_000
_COINGECKO_FREE_MCP_ENDPOINT = "https://mcp.coingecko.com/mcp"


class _McpQueryTool:
    """Shared execute()/gating/observation-shaping for one MCP-backed tool."""

    def __init__(
        self,
        *,
        spec: ToolSpec,
        endpoint: str,
        headers: dict[str, str],
        mcp_tool_name: str,
        arguments_model: type[BaseModel],
        observation_provider: str,
        clock: Clock,
        gate: ProviderCallGate,
        evidence_ttl_seconds: int = 300,
        to_mcp_arguments: Callable[[BaseModel], dict[str, JsonValue]] | None = None,
    ) -> None:
        self._spec = spec
        self._endpoint = endpoint
        self._headers = headers
        self._mcp_tool_name = mcp_tool_name
        self._arguments_model = arguments_model
        self._observation_provider = observation_provider
        self._clock = clock
        self._gate = gate
        self._evidence_ttl_seconds = evidence_ttl_seconds
        # Most MCP tools take the harness-facing argument shape verbatim; a few
        # (CoinGecko's get-coin-markets wants an `ids` array for one asset_id)
        # need a translation step, provided here rather than via a fragile
        # execute()-calling-super() override that would re-validate against the
        # wrong (already-translated) shape.
        self._to_mcp_arguments = to_mcp_arguments or (
            lambda parsed: parsed.model_dump(mode="json", exclude_none=True)
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = self._arguments_model.model_validate(arguments)
        return parsed.model_dump(mode="json", exclude_none=True)

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = self._arguments_model.model_validate(arguments)
        call_arguments = self._to_mcp_arguments(parsed)
        error_code = f"{self._observation_provider.upper().replace('-', '_')}_MCP_ERROR"
        try:
            async with self._gate.slot():
                text, structured = await call_mcp_tool(
                    endpoint=self._endpoint,
                    headers=self._headers,
                    tool_name=self._mcp_tool_name,
                    arguments=call_arguments,
                )
        except ProviderGateRejected as exc:
            return ToolFailure(code=exc.code, retryable=True, safe_message=exc.safe_message)
        except McpToolCallError:
            await self._gate.record_failure(error_code)
            return ToolFailure(
                code=error_code,
                retryable=True,
                safe_message=f"{self._observation_provider} MCP tool call did not complete.",
            )
        await self._gate.record_success()
        now = self._clock.now()
        bounded_text = text[:_MAX_RESULT_CHARS]
        payload: dict[str, JsonValue] = {
            "provider": self._observation_provider,
            "mcp_tool": self._mcp_tool_name,
            "result_text": bounded_text,
        }
        if structured is not None:
            payload["structured"] = structured
        reference = (
            f"{self._mcp_tool_name}:"
            f"{hashlib.sha256((bounded_text or str(structured)).encode('utf-8')).hexdigest()[:16]}"
        )
        return ToolSuccess(
            data=payload,
            source=SourceRef(provider=f"{self._observation_provider}-mcp", reference=reference),
            observed_at=now,
            expires_at=now + timedelta(seconds=self._evidence_ttl_seconds),
        )


# --- Tavily -----------------------------------------------------------------


class TavilyMcpSearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=2, max_length=256)
    max_results: int = Field(default=5, ge=1, le=5)
    topic: Literal["general", "news", "finance"] = "general"
    time_range: Literal["day", "week", "month", "year"] | None = None


class TavilyMcpResearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    input: str = Field(min_length=4, max_length=500)
    model: Literal["mini", "pro", "auto"] = "auto"


def build_tavily_mcp_tools(
    *,
    endpoint: str,
    clock: Clock,
    gate: ProviderCallGate,
) -> tuple[_McpQueryTool, _McpQueryTool]:
    search = _McpQueryTool(
        spec=ToolSpec(
            name="web.search_tavily_mcp",
            version="1.0.0",
            description=(
                "Search the web via Tavily's MCP server for current information -- news, "
                "facts, or data beyond the model's training cutoff. Redundant with "
                "web.search_tavily; use both when cross-checking matters."
            ),
            domain="WEB",
            input_schema=TavilyMcpSearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=16_384,
        ),
        endpoint=endpoint,
        headers={},
        mcp_tool_name="tavily_search",
        arguments_model=TavilyMcpSearchArguments,
        observation_provider="tavily",
        clock=clock,
        gate=gate,
    )
    research = _McpQueryTool(
        spec=ToolSpec(
            name="web.research_tavily_mcp",
            version="1.0.0",
            description=(
                "Run Tavily's bounded deep-research agent on a question or topic and get back "
                "a synthesized, multi-source answer -- broader than a single search call. Good "
                "for open-ended screening/comparison questions a single search can't answer "
                "(e.g. 'which large-cap stocks have grown their dividend for 10+ years'). Rate "
                "limited to 20 calls/minute by Tavily."
            ),
            domain="WEB",
            input_schema=TavilyMcpResearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=45.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=24_576,
        ),
        endpoint=endpoint,
        headers={},
        mcp_tool_name="tavily_research",
        arguments_model=TavilyMcpResearchArguments,
        observation_provider="tavily",
        clock=clock,
        gate=gate,
        evidence_ttl_seconds=180,
    )
    return search, research


# --- Alpha Vantage ------------------------------------------------------------

_SYMBOL_PATTERN = r"^[A-Z][A-Z0-9.-]{0,19}$"


class AlphaVantageMcpSymbolArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=_SYMBOL_PATTERN)

    @field_validator("symbol", mode="before")
    @classmethod
    def normalize_symbol(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


def build_alpha_vantage_mcp_tools(
    *,
    endpoint: str,
    clock: Clock,
    gate: ProviderCallGate,
) -> tuple[_McpQueryTool, _McpQueryTool]:
    quote = _McpQueryTool(
        spec=ToolSpec(
            name="market.get_quote_alpha_vantage_mcp",
            version="1.0.0",
            description=(
                "Get the latest price/volume for one equity ticker via Alpha Vantage's MCP "
                "server. Redundant with market.get_quote_alpha_vantage."
            ),
            domain="MARKET",
            input_schema=AlphaVantageMcpSymbolArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=12_288,
        ),
        endpoint=endpoint,
        headers={},
        mcp_tool_name="GLOBAL_QUOTE",
        arguments_model=AlphaVantageMcpSymbolArguments,
        observation_provider="alpha-vantage",
        clock=clock,
        gate=gate,
    )
    overview = _McpQueryTool(
        spec=ToolSpec(
            name="market.get_company_overview_alpha_vantage_mcp",
            version="1.0.0",
            description=(
                "Get company overview, financial ratios, and key metrics for one equity ticker "
                "via Alpha Vantage's MCP server -- includes dividend yield/per-share, P/E, "
                "market cap, sector, and analyst targets. Not available via Leo's REST "
                "integrations; use this for company-fundamentals questions, including dividend "
                "screening."
            ),
            domain="MARKET",
            input_schema=AlphaVantageMcpSymbolArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=16_384,
        ),
        endpoint=endpoint,
        headers={},
        mcp_tool_name="COMPANY_OVERVIEW",
        arguments_model=AlphaVantageMcpSymbolArguments,
        observation_provider="alpha-vantage",
        clock=clock,
        gate=gate,
        evidence_ttl_seconds=3600,
    )
    return quote, overview


# --- CoinGecko (free MCP tier only -- see module docstring) ------------------


class CoinGeckoMcpMarketsArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description="Canonical CoinGecko coin id, such as bitcoin, ethereum, or solana.",
    )

    @field_validator("asset_id", mode="before")
    @classmethod
    def normalize_asset_id(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value


def _coingecko_markets_mcp_arguments(parsed: BaseModel) -> dict[str, JsonValue]:
    """get-coin-markets wants an `ids` array, not the harness-facing `asset_id`."""

    assert isinstance(parsed, CoinGeckoMcpMarketsArguments)
    return {"ids": [parsed.asset_id]}


def build_coingecko_mcp_tools(
    *,
    clock: Clock,
    gate: ProviderCallGate,
) -> tuple[_McpQueryTool]:
    snapshot = _McpQueryTool(
        spec=ToolSpec(
            name="market.get_crypto_snapshot_coingecko_mcp",
            version="1.0.0",
            description=(
                "Get one current cryptocurrency price/market-cap/volume snapshot via "
                "CoinGecko's free public MCP server, by canonical coin id. Redundant with "
                "market.get_crypto_snapshot_coingecko. This uses CoinGecko's keyless free MCP "
                "endpoint, not the Pro endpoint -- CoinGecko's Pro MCP server requires an "
                "interactive OAuth browser flow that a backend service cannot complete."
            ),
            domain="MARKET",
            input_schema=CoinGeckoMcpMarketsArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            retry=ToolRetryPolicy(max_attempts=2),
            max_result_bytes=12_288,
        ),
        endpoint=_COINGECKO_FREE_MCP_ENDPOINT,
        headers={},
        mcp_tool_name="get-coin-markets",
        arguments_model=CoinGeckoMcpMarketsArguments,
        observation_provider="coingecko",
        clock=clock,
        gate=gate,
        to_mcp_arguments=_coingecko_markets_mcp_arguments,
    )
    return (snapshot,)


__all__ = [
    "AlphaVantageMcpSymbolArguments",
    "CoinGeckoMcpMarketsArguments",
    "TavilyMcpResearchArguments",
    "TavilyMcpSearchArguments",
    "build_alpha_vantage_mcp_tools",
    "build_coingecko_mcp_tools",
    "build_tavily_mcp_tools",
]
