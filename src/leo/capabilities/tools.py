"""Bounded harness-owned tools for progressive capability discovery."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.capabilities.discovery import query_hash
from leo.capabilities.runtime import CapabilityDiscoveryError, CapabilityRuntime
from leo.harness.models import (
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.harness.ports import Clock, Tool


class _SearchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1, max_length=256)
    limit: int = Field(default=5, ge=1, le=5)


class _DescribeArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    capability_ids: tuple[str, ...] = Field(min_length=1, max_length=3)


class ToolSearchTool:
    def __init__(self, runtime: CapabilityRuntime, clock: Clock) -> None:
        self._runtime = runtime
        self._clock = clock
        self._spec = ToolSpec(
            name="tool.search",
            version="1",
            description=(
                "Search only the current run's already-authorized, healthy read-capability "
                "catalog. Returns a bounded shortlist of summaries, never permission or execution."
            ),
            domain="HARNESS",
            input_schema=_SearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=2,
            max_result_bytes=12_288,
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
        parsed = _SearchArguments.model_validate(arguments)
        try:
            summaries = await self._runtime.search(
                run_id=context.run_id,
                trusted_scope=context.trusted_scope,
                query=parsed.query,
                limit=parsed.limit,
            )
        except CapabilityDiscoveryError as exc:
            return ToolFailure(
                code=exc.safe_code.upper(),
                safe_message="Capability search was denied by its bounded run policy.",
            )
        return ToolSuccess(
            data={
                "catalog_fingerprint": self._runtime.catalog_fingerprint,
                "query_hash": query_hash(parsed.query),
                "result_count": len(summaries),
                "capabilities": [item.model_dump(mode="json") for item in summaries],
            },
            source=SourceRef(
                provider="leo.capability_catalog",
                reference=self._runtime.catalog_fingerprint,
            ),
            observed_at=self._clock.now(),
        )


class ToolDescribeTool:
    def __init__(self, runtime: CapabilityRuntime, clock: Clock) -> None:
        self._runtime = runtime
        self._clock = clock
        self._spec = ToolSpec(
            name="tool.describe",
            version="1",
            description=(
                "Load exact executable schemas for up to three capability IDs returned by "
                "tool.search in this run. Cannot grant or execute a capability."
            ),
            domain="HARNESS",
            input_schema=_DescribeArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=2,
            max_result_bytes=24_576,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _DescribeArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = _DescribeArguments.model_validate(arguments)
        try:
            records = self._runtime.describe(
                run_id=context.run_id,
                trusted_scope=context.trusted_scope,
                capability_ids=parsed.capability_ids,
            )
        except CapabilityDiscoveryError as exc:
            return ToolFailure(
                code=exc.safe_code.upper(),
                safe_message="Capability description was denied by its bounded run policy.",
            )
        return ToolSuccess(
            data={
                "catalog_fingerprint": self._runtime.catalog_fingerprint,
                "capabilities": [
                    {
                        "id": record.id,
                        "version": record.semantic_version,
                        "schema_fingerprint": record.schema_fingerprint,
                        "spec": record.spec.model_dump(mode="json"),
                    }
                    for record in records
                ],
            },
            source=SourceRef(
                provider="leo.capability_catalog",
                reference=self._runtime.catalog_fingerprint,
            ),
            observed_at=self._clock.now(),
        )


def build_capability_discovery_tools(
    runtime: CapabilityRuntime,
    clock: Clock,
) -> tuple[Tool, ...]:
    return (ToolSearchTool(runtime, clock), ToolDescribeTool(runtime, clock))
