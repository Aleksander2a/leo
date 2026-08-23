"""Classify a tool call's origin for dashboard display.

Leo has no dedicated "source kind" field anywhere in its schema -- MCP-backed
tools, REST-backed tools, and internal (memory/subagent/thread-context)
capabilities are only distinguishable today by their harness-owned naming
conventions:

- a tool name ending in ``_mcp`` is one of this session's MCP-sourced tools
  (see ``leo.integrations.mcp_tools``); the domain prefix before that names
  the capability area and the remainder (minus the suffix) names the
  integration
- a tool/observation domain of ``memory.``/``agent.``/``thread_context.`` is
  harness-internal, not a third-party call at all
- an ``Observation.source.provider`` string (when a call succeeded) is the
  most precise integration label available; free-text but consistent by
  convention (``"tavily"``, ``"finnhub"``, ``"coingecko-mcp"``, ...)
- anything else with a domain prefix (``market.``, ``web.``, ``sec.``) is a
  native REST integration

This module centralizes that inference so every dashboard router applies the
same rule instead of each re-deriving it slightly differently.
"""

from __future__ import annotations

_INTERNAL_DOMAIN_LABELS = {
    "memory": "Internal Memory",
    "agent": "Internal Subagent",
    "thread_context": "Internal Context",
}

_DISPLAY_NAME_OVERRIDES = {
    "finnhub": "Finnhub",
    "alpha-vantage": "Alpha Vantage",
    "alpha_vantage": "Alpha Vantage",
    "massive": "Massive",
    "ticker-layer": "TickerLayer",
    "exa": "Exa",
    "sec-edgar": "SEC EDGAR",
    "tavily": "Tavily",
    "coingecko": "CoinGecko",
    "coinmarketcap": "CoinMarketCap",
    "coin-market-cap": "CoinMarketCap",
    "openrouter": "OpenRouter",
    "slack": "Slack",
    "public-web": "Public web fetch",
    "crypto-corroboration": "Crypto corroboration",
    "leo-subagent": "Subagent",
    "leo-subagent-plan": "Subagent plan",
    "leo_memory": "Memory",
}


def classify_call(*, tool_name: str | None, provider: str | None) -> dict[str, str]:
    """Return {"call_kind", "integration"} for one tool call or observation.

    ``call_kind`` is one of "mcp", "rest_api", "internal_memory",
    "internal_subagent", "internal_context", or "unknown". ``integration`` is
    a short human label for the specific provider/capability, best-effort.
    """

    domain = (tool_name or "").split(".", 1)[0] if tool_name else ""
    if domain in _INTERNAL_DOMAIN_LABELS:
        kind = f"internal_{domain}" if domain != "thread_context" else "internal_context"
        return {"call_kind": kind, "integration": _INTERNAL_DOMAIN_LABELS[domain]}

    is_mcp = bool(tool_name and tool_name.endswith("_mcp")) or bool(
        provider and (provider.endswith("-mcp") or provider.startswith("mcp:"))
    )
    integration = _integration_label(tool_name, provider)
    if is_mcp:
        return {"call_kind": "mcp", "integration": integration}
    if tool_name or provider:
        return {"call_kind": "rest_api", "integration": integration}
    return {"call_kind": "unknown", "integration": "Unknown"}


def _integration_label(tool_name: str | None, provider: str | None) -> str:
    if provider:
        bare = provider[4:] if provider.startswith("mcp:") else provider
        bare = bare[:-4] if bare.endswith("-mcp") else bare
        return _DISPLAY_NAME_OVERRIDES.get(bare, bare.replace("-", " ").replace("_", " ").title())
    if not tool_name:
        return "Unknown"
    # market.get_quote_alpha_vantage_mcp -> alpha_vantage ; web.search_tavily -> tavily
    remainder = tool_name.split(".", 1)[-1]
    if remainder.endswith("_mcp"):
        remainder = remainder[: -len("_mcp")]
    known_suffixes = ("alpha_vantage", "finnhub", "massive", "ticker_layer", "tavily", "coingecko")
    for suffix in known_suffixes:
        if remainder.endswith(suffix):
            bare = suffix.replace("_", "-")
            return _DISPLAY_NAME_OVERRIDES.get(bare, suffix.replace("_", " ").title())
    return remainder.replace("_", " ").title()


__all__ = ["classify_call"]
