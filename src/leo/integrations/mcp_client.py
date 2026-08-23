"""Generic Streamable-HTTP MCP tool-calling adapter.

Leo's primary market/research adapters stay pinned to each provider's native REST
API (see crypto_composition.py / equity_composition.py): REST payloads are
schema-stable enough for the harness to derive exact canonical statements from,
and their base URLs are validated against an official host+path allowlist.  MCP
servers return less structured, LLM-oriented payloads under a protocol whose
authority (session-negotiated, JSON-RPC framed) doesn't fit that same exact-match
model.  MCP-sourced tools built on this adapter are therefore additive
corroboration sources, not replacements -- the model may call both a REST tool
and its MCP counterpart for the same fact and reconcile them itself.  Grounding
for the `market.*_mcp` / `web.*_mcp` observation kinds these tools emit relies on
the harness's `relax_integration_grounding` path (trusted-integration synthesis)
rather than a registered exact-statement grounding rule.
"""

from __future__ import annotations

from typing import Any

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent


class McpToolCallError(RuntimeError):
    """A transport, protocol, or tool-reported failure calling one MCP tool."""


async def call_mcp_tool(
    *,
    endpoint: str,
    headers: dict[str, str],
    tool_name: str,
    arguments: dict[str, Any],
    timeout_seconds: float = 15.0,
) -> tuple[str, dict[str, Any] | None]:
    """Open one short-lived MCP session, call one tool, return (text, structured).

    A fresh session per call keeps this stateless and consistent with the rest of
    Leo's per-request tool adapters -- MCP servers are expected to handle cheap
    session setup over Streamable HTTP. Raises McpToolCallError on any transport,
    protocol, or tool-reported failure; never raises anything else, so callers can
    translate it into a bounded ToolFailure without a bare except.
    """

    try:
        async with httpx.AsyncClient(headers=headers, timeout=timeout_seconds) as http_client:
            # mcp's stubs reference an internally vendored httpx client type here;
            # a real httpx.AsyncClient is the documented and, per this session's live
            # discovery probes against Tavily/Alpha Vantage/CoinGecko, empirically
            # working argument.
            async with streamable_http_client(
                endpoint,
                http_client=http_client,  # type: ignore[arg-type]
            ) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=arguments)
    except* Exception as excgroup:
        raise McpToolCallError(_flatten_exception_detail(excgroup)) from excgroup
    if result.is_error:
        raise McpToolCallError(_result_text(result) or "MCP tool call reported an error.")
    text = _result_text(result)
    structured = result.structured_content if isinstance(result.structured_content, dict) else None
    if not text and structured is None:
        raise McpToolCallError("MCP tool returned no usable content.")
    return text, structured


def _result_text(result: CallToolResult) -> str:
    parts = [item.text for item in result.content if isinstance(item, TextContent) and item.text]
    return "\n".join(parts).strip()


def _flatten_exception_detail(group: BaseException, *, depth: int = 0) -> str:
    """Reduce a (possibly nested) ExceptionGroup to one safe, bounded detail string."""

    if depth >= 6:
        return "MCP call failed (error detail truncated)."
    if isinstance(group, BaseExceptionGroup):
        for sub in group.exceptions:
            detail = _flatten_exception_detail(sub, depth=depth + 1)
            if detail:
                return detail
        return "MCP call failed with no further detail."
    return f"{type(group).__name__}: {group}"[:300]


__all__ = ["McpToolCallError", "call_mcp_tool"]
