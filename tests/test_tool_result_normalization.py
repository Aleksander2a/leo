from __future__ import annotations

from collections.abc import Callable

import pytest
from pydantic import JsonValue

from leo.demo import run_quote_smoke
from leo.harness.models import (
    BudgetLimits,
    EventType,
    EvidenceQuality,
    RunPhase,
    RunStatus,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolSpec,
    ToolSuccess,
)
from leo.harness.normalization import normalize_success as harness_normalize_success
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import FakeQuoteTool, FixedClock, TwoToolBatchModel
from leo.integrations.normalization import (
    NORMALIZATION_VERSION,
)
from leo.integrations.normalization import (
    normalize_success as compatibility_normalize_success,
)


class _InvalidResultTool:
    def __init__(self, outcome: Callable[[], ToolSuccess]) -> None:
        self._outcome = outcome
        self._spec = ToolSpec(
            name="market.fail",
            description="Return an injected invalid provider result.",
            domain="TEST",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            max_result_bytes=65_536,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if arguments:
            raise ValueError("tool takes no arguments")
        return {}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del arguments, context
        return self._outcome()


def _success(data: object, *, source: object | None = None) -> ToolSuccess:
    return ToolSuccess.model_construct(
        data=data,
        source=source or SourceRef(provider="fixture", reference="fixture-invalid-provider-result"),
        observed_at=FixedClock().now(),
        expires_at=None,
    )


def _non_finite() -> ToolSuccess:
    return _success({"value": float("nan")})


def _oversized() -> ToolSuccess:
    return _success({"payload": "x" * 40_000})


def _malformed() -> ToolSuccess:
    return _success({"values": {1, 2, 3}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("factory", "expected_code"),
    [
        (_non_finite, "TOOL_RESULT_NON_FINITE"),
        (_oversized, "TOOL_RESULT_TOO_LARGE"),
        (_malformed, "TOOL_RESULT_INVALID"),
    ],
    ids=("nonfinite", "oversized", "malformed"),
)
async def test_invalid_parallel_provider_result_is_an_atomic_typed_failure(
    factory: Callable[[], ToolSuccess],
    expected_code: str,
) -> None:
    result = await run_quote_smoke(
        model=TwoToolBatchModel(),
        tool_registry=ToolRegistry(
            (
                FakeQuoteTool(FixedClock()),
                _InvalidResultTool(factory),
            )
        ),
        limits=BudgetLimits(max_iterations=2, max_model_calls=2, max_tool_calls=2),
    )

    assert result.run.status is RunStatus.FAILED
    assert result.run.terminal_reason == f"tool_failure:{expected_code}"
    assert result.observations == ()
    assert result.task.observation_ids == ()
    assert EventType.OBSERVATION_CREATED not in {event.type for event in result.events}
    assert EventType.TOOL_COMPLETED not in {event.type for event in result.events}
    failed = [event for event in result.events if event.type is EventType.TOOL_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["code"] == expected_code
    assert failed[0].payload["retryable"] is False


def test_integration_api_reexports_the_harness_normalizer_for_adapter_mcp_parity() -> None:
    assert NORMALIZATION_VERSION == "normalization-v1"
    assert compatibility_normalize_success is harness_normalize_success

    data = {"symbol": "NVDA", "price": 181.25}
    adapter = harness_normalize_success(
        _success(data),
        observation_id="obs-adapter",
        scope={"organization_id": "org", "strategy_id": "strategy"},
        run_id="run",
        tool_call_id="adapter-call",
        observation_kind="market.get_quote",
    )
    mcp = compatibility_normalize_success(
        _success(data, source=SourceRef(provider="mcp:demo", reference="mcp:quote")),
        observation_id="obs-mcp",
        scope={"organization_id": "org", "strategy_id": "strategy"},
        run_id="run",
        tool_call_id="mcp-call",
        observation_kind="market.get_quote",
    )

    assert adapter.data == mcp.data == data
    assert adapter.raw_hash == mcp.raw_hash
    assert adapter.status.value == mcp.status.value == "retrieved"
    assert adapter.quality is mcp.quality is EvidenceQuality.PROVIDER_REPORTED
    assert adapter.schema_version == mcp.schema_version == "observation-v2"
    assert adapter.normalization_version == mcp.normalization_version == NORMALIZATION_VERSION
