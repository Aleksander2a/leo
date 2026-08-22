"""Official MCP client lifecycle normalized behind Leo's policy boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, cast

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError
from mcp import Client
from pydantic import Field, JsonValue, model_validator

from leo.capabilities.catalog import CapabilityHealth, CatalogTool
from leo.harness.models import (
    ContractModel,
    NonEmptyStr,
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.harness.ports import Clock
from leo.harness.tools import RESERVED_AUTHORITY_KEYS

MCP_SDK_VERSION = "2.0.0"
MCP_PROTOCOL_MODE = "auto"
_ALIAS = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class McpAdapterError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class McpAdapterState(StrEnum):
    NEW = "new"
    READY = "ready"
    DEGRADED = "degraded"
    CLOSED = "closed"


class McpToolDescriptor(ContractModel):
    name: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    description: NonEmptyStr = Field(max_length=240)
    input_schema: dict[str, JsonValue]
    effect: ToolEffect = ToolEffect.READ

    @model_validator(mode="after")
    def object_input_schema(self) -> McpToolDescriptor:
        if self.input_schema.get("type") != "object":
            raise ValueError("MCP tool input schema must describe an object")
        try:
            Draft202012Validator.check_schema(self.input_schema)
        except SchemaError as exc:
            raise ValueError("MCP tool input schema is invalid") from exc
        return self


class McpServer(Protocol):
    """Structural interface implemented by the official SDK wrapper and test fakes."""

    async def initialize(self) -> None: ...

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]: ...

    async def call_tool(
        self, name: str, arguments: dict[str, JsonValue]
    ) -> dict[str, JsonValue]: ...

    async def close(self) -> None: ...


class OfficialMcpServer(McpServer):
    """Translate the official SDK v2 Client into Leo's narrow read-tool port.

    Tool versions are trusted configuration because MCP tool metadata does not provide a
    semantic version. Only configured names are projected; prompts, roots, sampling, elicitation,
    server instructions, unstructured content, and server-initiated authority are ignored.
    """

    def __init__(
        self,
        *,
        target: object,
        tool_versions: Mapping[str, str],
        read_timeout_seconds: float = 15.0,
        max_tools: int = 128,
        max_pages: int = 8,
    ) -> None:
        if not tool_versions or read_timeout_seconds <= 0:
            raise ValueError("official MCP target policy and timeout are required")
        if not 1 <= max_tools <= 512 or not 1 <= max_pages <= 32:
            raise ValueError("official MCP discovery limits are invalid")
        for name, version in tool_versions.items():
            if not name or re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
                raise ValueError("official MCP tool versions must be explicit semantic versions")
        self._target = target
        self._tool_versions = dict(tool_versions)
        self._read_timeout_seconds = read_timeout_seconds
        self._max_tools = max_tools
        self._max_pages = max_pages
        self._client_context: Client | None = None
        self._client: Client | None = None

    async def initialize(self) -> None:
        if self._client is not None:
            return
        client = Client(
            cast(Any, self._target),
            mode=MCP_PROTOCOL_MODE,
            read_timeout_seconds=self._read_timeout_seconds,
            input_required_max_rounds=0,
        )
        self._client_context = client
        self._client = await client.__aenter__()

    async def list_tools(self) -> tuple[McpToolDescriptor, ...]:
        client = self._require_client()
        descriptors: list[McpToolDescriptor] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for _page in range(self._max_pages):
            result = await client.list_tools(cursor=cursor, cache_mode="refresh")
            for tool in result.tools:
                version = self._tool_versions.get(tool.name)
                if version is None:
                    continue
                try:
                    serialized_schema = json.dumps(
                        tool.input_schema,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    schema = json.loads(serialized_schema)
                    if not isinstance(schema, dict):
                        raise ValueError("schema is not an object")
                    description = (
                        tool.description or tool.title or f"Configured MCP tool {tool.name}"
                    )
                    descriptor = McpToolDescriptor(
                        name=tool.name,
                        version=version,
                        description=" ".join(description.split())[:240],
                        input_schema=cast(dict[str, JsonValue], schema),
                        effect=ToolEffect.READ,
                    )
                except (TypeError, ValueError) as exc:
                    raise McpAdapterError("mcp_malformed_tool_metadata") from exc
                descriptors.append(descriptor)
                if len(descriptors) > self._max_tools:
                    raise McpAdapterError("mcp_discovery_limit_exceeded")
            cursor = result.next_cursor
            if cursor is None:
                return tuple(descriptors)
            if cursor in seen_cursors:
                raise McpAdapterError("mcp_discovery_cursor_loop")
            seen_cursors.add(cursor)
        raise McpAdapterError("mcp_discovery_page_limit")

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        client = self._require_client()
        if name not in self._tool_versions:
            raise McpAdapterError("mcp_capability_not_configured")
        result = await client.call_tool(
            name,
            cast(dict[str, Any], arguments),
            read_timeout_seconds=self._read_timeout_seconds,
        )
        if result.is_error:
            raise McpAdapterError("mcp_tool_reported_error")
        if result.result_type != "complete" or not isinstance(result.structured_content, dict):
            raise McpAdapterError("mcp_unstructured_result_denied")
        try:
            encoded = json.dumps(
                result.structured_content,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            decoded = json.loads(encoded)
        except (TypeError, ValueError) as exc:
            raise McpAdapterError("mcp_malformed_result") from exc
        if not isinstance(decoded, dict):
            raise McpAdapterError("mcp_unstructured_result_denied")
        return cast(dict[str, JsonValue], decoded)

    async def close(self) -> None:
        context = self._client_context
        self._client = None
        self._client_context = None
        if context is not None:
            await context.__aexit__(None, None, None)

    def _require_client(self) -> Client:
        if self._client is None:
            raise McpAdapterError("mcp_not_initialized")
        return self._client


class McpDiscoveredTool:
    """Executable ToolRegistry adapter for one currently discovered MCP record."""

    def __init__(self, adapter: McpClientAdapter, record: CatalogTool) -> None:
        self._adapter = adapter
        self._spec = record.spec
        self._validator = Draft202012Validator(self._spec.input_schema)

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        try:
            self._validator.validate(arguments)
        except JsonSchemaValidationError as exc:
            raise ValueError("MCP tool arguments do not match the discovered schema") from exc
        return arguments

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        return await self._adapter.call(self._spec.name, arguments, context)


class McpClientAdapter:
    def __init__(
        self,
        *,
        alias: str,
        server: McpServer,
        allowlist: frozenset[str],
        max_result_bytes: int = 16_384,
        timeout_seconds: float = 20.0,
        max_concurrency: int = 4,
        clock: Clock | None = None,
    ) -> None:
        if _ALIAS.fullmatch(alias) is None or not allowlist:
            raise ValueError("MCP alias and allowlist are required")
        if not 256 <= max_result_bytes <= 1_048_576:
            raise ValueError("MCP result cap must be between 256 and 1048576 bytes")
        if not 0 < timeout_seconds <= 120 or not 1 <= max_concurrency <= 16:
            raise ValueError("MCP timeout or concurrency limit is invalid")
        self._alias = alias
        self._server = server
        self._allowlist = allowlist
        self._max_result_bytes = max_result_bytes
        self._timeout_seconds = timeout_seconds
        self._clock = clock
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._records: dict[str, McpToolDescriptor] = {}
        self._catalog_records: tuple[CatalogTool, ...] = ()
        self._discovery_fingerprint: str | None = None
        self._state = McpAdapterState.NEW
        self._generation = 0

    @property
    def state(self) -> McpAdapterState:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def discovery_fingerprint(self) -> str | None:
        return self._discovery_fingerprint

    async def initialize(self) -> None:
        if self._state is McpAdapterState.READY:
            return
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._server.initialize()
        except TimeoutError as exc:
            self._state = McpAdapterState.DEGRADED
            raise McpAdapterError("mcp_initialize_timeout") from exc
        except Exception as exc:
            self._state = McpAdapterState.DEGRADED
            raise McpAdapterError("mcp_initialize_failed") from exc
        self._state = McpAdapterState.READY

    async def discover(self) -> tuple[CatalogTool, ...]:
        if self._state is not McpAdapterState.READY:
            raise McpAdapterError("mcp_not_initialized")
        try:
            async with asyncio.timeout(self._timeout_seconds):
                descriptors = await self._server.list_tools()
            records, record_map, fingerprint = self._normalize_descriptors(descriptors)
        except TimeoutError as exc:
            self._state = McpAdapterState.DEGRADED
            raise McpAdapterError("mcp_discovery_timeout") from exc
        except McpAdapterError:
            self._state = McpAdapterState.DEGRADED
            raise
        except Exception as exc:
            self._state = McpAdapterState.DEGRADED
            raise McpAdapterError("mcp_discovery_failed") from exc
        self._records = record_map
        self._catalog_records = records
        self._discovery_fingerprint = fingerprint
        self._generation += 1
        return records

    def tools(self) -> tuple[McpDiscoveredTool, ...]:
        if self._state is not McpAdapterState.READY or not self._catalog_records:
            return ()
        return tuple(McpDiscoveredTool(self, record) for record in self._catalog_records)

    async def refresh_health(self) -> CapabilityHealth:
        if self._state is not McpAdapterState.READY or self._discovery_fingerprint is None:
            return CapabilityHealth.UNHEALTHY
        try:
            async with asyncio.timeout(self._timeout_seconds):
                descriptors = await self._server.list_tools()
            _records, _record_map, fingerprint = self._normalize_descriptors(descriptors)
        except Exception:
            self._state = McpAdapterState.DEGRADED
            self._records = {}
            self._catalog_records = ()
            return CapabilityHealth.UNHEALTHY
        if fingerprint != self._discovery_fingerprint:
            self._state = McpAdapterState.DEGRADED
            self._records = {}
            self._catalog_records = ()
            return CapabilityHealth.UNHEALTHY
        return CapabilityHealth.HEALTHY

    async def reconnect(self) -> tuple[CatalogTool, ...]:
        await self.close()
        self._state = McpAdapterState.NEW
        await self.initialize()
        return await self.discover()

    async def call(
        self,
        capability_id: str,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        if self._state is not McpAdapterState.READY:
            return ToolFailure(code="MCP_UNAVAILABLE", safe_message="MCP server is not ready.")
        descriptor = self._records.get(capability_id)
        if descriptor is None:
            return ToolFailure(
                code="MCP_CAPABILITY_NOT_ELIGIBLE", safe_message="MCP tool is not eligible."
            )
        try:
            Draft202012Validator(descriptor.input_schema).validate(arguments)
        except JsonSchemaValidationError:
            return ToolFailure(
                code="MCP_ARGUMENTS_INVALID", safe_message="MCP tool arguments are invalid."
            )
        try:
            async with self._semaphore, asyncio.timeout(self._timeout_seconds):
                payload = await self._server.call_tool(descriptor.name, arguments)
            encoded = json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        except TimeoutError:
            return ToolFailure(
                code="MCP_CALL_TIMEOUT",
                retryable=True,
                safe_message="MCP tool call timed out.",
            )
        except McpAdapterError as exc:
            return ToolFailure(
                code=exc.safe_code.upper(),
                safe_message="MCP tool call failed safely.",
            )
        except (TypeError, ValueError):
            return ToolFailure(
                code="MCP_MALFORMED_RESULT", safe_message="MCP returned malformed data."
            )
        except Exception:
            return ToolFailure(code="MCP_CALL_FAILED", safe_message="MCP tool call failed.")
        if len(encoded) > self._max_result_bytes:
            return ToolFailure(
                code="MCP_RESULT_TOO_LARGE", safe_message="MCP result exceeded the result cap."
            )
        return ToolSuccess(
            data=payload,
            source=SourceRef(provider=f"mcp:{self._alias}", reference=capability_id),
            observed_at=self._clock.now() if self._clock is not None else datetime.now(UTC),
        )

    async def close(self) -> None:
        try:
            async with asyncio.timeout(self._timeout_seconds):
                await self._server.close()
        except Exception as exc:
            self._state = McpAdapterState.DEGRADED
            raise McpAdapterError("mcp_close_failed") from exc
        self._records = {}
        self._catalog_records = ()
        self._discovery_fingerprint = None
        self._state = McpAdapterState.CLOSED

    def _normalize_descriptors(
        self,
        descriptors: tuple[McpToolDescriptor, ...],
    ) -> tuple[tuple[CatalogTool, ...], dict[str, McpToolDescriptor], str]:
        records: list[CatalogTool] = []
        record_map: dict[str, McpToolDescriptor] = {}
        seen: set[str] = set()
        for descriptor in descriptors:
            capability_id = f"mcp:{self._alias}:{descriptor.name}"
            if capability_id in seen:
                raise McpAdapterError("mcp_duplicate_tool")
            seen.add(capability_id)
            if descriptor.name not in self._allowlist or descriptor.effect is not ToolEffect.READ:
                continue
            if RESERVED_AUTHORITY_KEYS.intersection(
                _schema_property_names(descriptor.input_schema)
            ):
                raise McpAdapterError("mcp_reserved_authority_schema")
            record_map[capability_id] = descriptor
            records.append(
                CatalogTool(
                    id=capability_id,
                    semantic_version=descriptor.version,
                    provider=f"mcp:{self._alias}",
                    spec=ToolSpec(
                        name=capability_id,
                        version=descriptor.version,
                        description=(
                            "Configured read-only MCP capability. Server summary is untrusted: "
                            f"{descriptor.description}"
                        ),
                        domain="MCP",
                        input_schema=descriptor.input_schema,
                        effect=ToolEffect.READ,
                        allowed_phases=frozenset({RunPhase.RESEARCH}),
                        timeout_seconds=self._timeout_seconds,
                        max_result_bytes=self._max_result_bytes,
                    ),
                    short_description=descriptor.description,
                    tags=frozenset({"mcp", self._alias}),
                    health=CapabilityHealth.HEALTHY,
                    verification_expectations=frozenset({"source_claim_or_labeled_inference"}),
                )
            )
        ordered = tuple(sorted(records, key=lambda record: record.id))
        encoded = json.dumps(
            [record.model_dump(mode="json") for record in ordered],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return ordered, record_map, hashlib.sha256(encoded).hexdigest()


def _schema_property_names(value: JsonValue) -> set[str]:
    if isinstance(value, dict):
        names: set[str] = set()
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
        for nested in value.values():
            names.update(_schema_property_names(nested))
        return names
    if isinstance(value, list):
        nested_names: set[str] = set()
        for nested in value:
            nested_names.update(_schema_property_names(nested))
        return nested_names
    return set()
