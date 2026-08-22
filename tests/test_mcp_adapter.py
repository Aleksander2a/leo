from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from mcp.server import MCPServer
from pydantic import JsonValue

from leo.capabilities.catalog import CapabilityHealth
from leo.harness.models import (
    RunPhase,
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import FixedClock
from leo.integrations.mcp import (
    MCP_PROTOCOL_MODE,
    MCP_SDK_VERSION,
    McpAdapterError,
    McpAdapterState,
    McpClientAdapter,
    McpToolDescriptor,
    OfficialMcpServer,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(organization_id="org", strategy_id="domain"),
            actor_id="actor",
        ),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_official_mcp_sdk_discovery_execution_and_close_parity() -> None:
    server = MCPServer("leo-mcp-fixture", version="1.0.0")

    @server.tool(name="lookup", structured_output=True)
    def lookup(symbol: str) -> dict[str, object]:
        return {"symbol": symbol, "value": 17}

    official = OfficialMcpServer(
        target=server,
        tool_versions={"lookup": "1.2.0"},
    )
    adapter = McpClientAdapter(
        alias="fixture",
        server=official,
        allowlist=frozenset({"lookup"}),
        clock=FixedClock(NOW),
    )

    assert MCP_SDK_VERSION == "2.0.0"
    assert MCP_PROTOCOL_MODE == "auto"
    await adapter.initialize()
    records = await adapter.discover()
    assert tuple(record.id for record in records) == ("mcp:fixture:lookup",)
    assert adapter.discovery_fingerprint is not None

    registry = ToolRegistry(adapter.tools())
    outcome = await registry.execute(
        ToolRequest(id="request-1", name="mcp:fixture:lookup", arguments={"symbol": "DEMO"}),
        _context(),
        RunPhase.RESEARCH,
    )
    assert isinstance(outcome, ToolSuccess)
    assert outcome.data == {"symbol": "DEMO", "value": 17}
    assert outcome.source.provider == "mcp:fixture"
    assert await adapter.refresh_health() is CapabilityHealth.HEALTHY

    await adapter.close()
    assert adapter.state is McpAdapterState.CLOSED
    assert adapter.tools() == ()


class _MutableMcpServer:
    def __init__(self) -> None:
        self.descriptors = (
            McpToolDescriptor(
                name="read_demo",
                version="1.0.0",
                description="Read fixture data.",
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "additionalProperties": False,
                },
            ),
        )
        self.result: dict[str, JsonValue] = {"value": "ok"}
        self.started = asyncio.Event()
        self.release: asyncio.Event | None = None
        self.closed = 0

    async def initialize(self) -> None:
        return None

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        return self.descriptors

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        assert name == "read_demo"
        del arguments
        self.started.set()
        if self.release is not None:
            await self.release.wait()
        return self.result

    async def close(self) -> None:
        self.closed += 1


def _mutable_adapter(
    server: _MutableMcpServer,
    *,
    timeout_seconds: float = 1.0,
    max_result_bytes: int = 1024,
) -> McpClientAdapter:
    return McpClientAdapter(
        alias="fixture",
        server=server,
        allowlist=frozenset({"read_demo"}),
        timeout_seconds=timeout_seconds,
        max_result_bytes=max_result_bytes,
        clock=FixedClock(NOW),
    )


@pytest.mark.asyncio
async def test_mcp_health_drift_invalidates_authority_until_reconnect() -> None:
    server = _MutableMcpServer()
    adapter = _mutable_adapter(server)
    await adapter.initialize()
    initial = await adapter.discover()
    initial_fingerprint = adapter.discovery_fingerprint

    server.descriptors = (server.descriptors[0].model_copy(update={"version": "1.1.0"}),)
    assert await adapter.refresh_health() is CapabilityHealth.UNHEALTHY
    assert adapter.state is McpAdapterState.DEGRADED
    assert adapter.tools() == ()
    denied = await adapter.call(initial[0].id, {}, _context())
    assert isinstance(denied, ToolFailure)
    assert denied.code == "MCP_UNAVAILABLE"

    refreshed = await adapter.reconnect()
    assert refreshed[0].semantic_version == "1.1.0"
    assert adapter.discovery_fingerprint != initial_fingerprint
    assert adapter.generation == 2


@pytest.mark.asyncio
async def test_mcp_discovery_rejects_duplicates_and_reserved_authority_schema() -> None:
    server = _MutableMcpServer()
    adapter = _mutable_adapter(server)
    await adapter.initialize()
    server.descriptors = (server.descriptors[0], server.descriptors[0])
    with pytest.raises(McpAdapterError, match="mcp_duplicate_tool"):
        await adapter.discover()
    assert adapter.state is McpAdapterState.DEGRADED

    server = _MutableMcpServer()
    server.descriptors = (
        server.descriptors[0].model_copy(
            update={
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "nested": {
                            "type": "object",
                            "properties": {"conversation_id": {"type": "string"}},
                        }
                    },
                }
            }
        ),
    )
    adapter = _mutable_adapter(server)
    await adapter.initialize()
    with pytest.raises(McpAdapterError, match="mcp_reserved_authority_schema"):
        await adapter.discover()


@pytest.mark.asyncio
async def test_mcp_result_cap_timeout_and_cancellation_fail_closed() -> None:
    oversized_server = _MutableMcpServer()
    oversized_server.result = {"value": "x" * 512}
    oversized = _mutable_adapter(oversized_server, max_result_bytes=256)
    await oversized.initialize()
    record = (await oversized.discover())[0]
    result = await oversized.call(record.id, {}, _context())
    assert isinstance(result, ToolFailure)
    assert result.code == "MCP_RESULT_TOO_LARGE"

    timeout_server = _MutableMcpServer()
    timeout_server.release = asyncio.Event()
    timed = _mutable_adapter(timeout_server, timeout_seconds=0.001)
    await timed.initialize()
    record = (await timed.discover())[0]
    result = await timed.call(record.id, {}, _context())
    assert isinstance(result, ToolFailure)
    assert result.code == "MCP_CALL_TIMEOUT"

    cancel_server = _MutableMcpServer()
    cancel_server.release = asyncio.Event()
    cancellable = _mutable_adapter(cancel_server)
    await cancellable.initialize()
    record = (await cancellable.discover())[0]
    task = asyncio.create_task(cancellable.call(record.id, {}, _context()))
    await cancel_server.started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_mcp_unknown_or_schema_invalid_calls_never_reach_server() -> None:
    server = _MutableMcpServer()
    adapter = _mutable_adapter(server)
    await adapter.initialize()
    record = (await adapter.discover())[0]

    unknown = await adapter.call("mcp:fixture:missing", {}, _context())
    invalid = await adapter.call(record.id, {"key": 3}, _context())
    assert isinstance(unknown, ToolFailure)
    assert unknown.code == "MCP_CAPABILITY_NOT_ELIGIBLE"
    assert isinstance(invalid, ToolFailure)
    assert invalid.code == "MCP_ARGUMENTS_INVALID"
    assert not server.started.is_set()
