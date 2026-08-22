from __future__ import annotations

import pytest

from leo.harness.models import (
    RunPhase,
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolRetryPolicy,
    ToolSpec,
    TrustedScope,
)
from leo.harness.tools import ToolRegistry, ToolRegistryError
from leo.integrations.fake import FakeQuoteTool, FakeWriteTool, FixedClock


class ReservedFieldTool(FakeQuoteTool):
    @property
    def spec(self) -> ToolSpec:
        return super().spec.model_copy(
            update={
                "name": "bad.reserved_scope",
                "input_schema": {
                    "type": "object",
                    "properties": {"strategy_id": {"type": "string"}},
                },
                "allowed_phases": frozenset({RunPhase.RESEARCH}),
            }
        )


class RoleProtectedTool(FakeQuoteTool):
    @property
    def spec(self) -> ToolSpec:
        return super().spec.model_copy(update={"required_roles": frozenset({"operator"})})


class OversizedResultTool(FakeQuoteTool):
    @property
    def spec(self) -> ToolSpec:
        return super().spec.model_copy(update={"max_result_bytes": 8})


class RetryingWriteTool(FakeWriteTool):
    @property
    def spec(self) -> ToolSpec:
        return super().spec.model_copy(update={"retry": ToolRetryPolicy(max_attempts=2)})


class UndeclaredRetryTool(FakeQuoteTool):
    async def execute(self, arguments, context):
        del arguments, context
        return ToolFailure(
            code="RETRY_NOT_DECLARED",
            retryable=True,
            safe_message="Retryable fixture failure.",
        )


def test_registry_rejects_duplicate_tool() -> None:
    tool = FakeQuoteTool(FixedClock())
    registry = ToolRegistry((tool,))
    with pytest.raises(ToolRegistryError, match="duplicate tool"):
        registry.register(tool)


def test_registry_rejects_model_controlled_scope_argument() -> None:
    with pytest.raises(ToolRegistryError, match="reserved authority"):
        ToolRegistry((ReservedFieldTool(FixedClock()),))


def test_registry_rejects_write_tool_advertised_in_research() -> None:
    with pytest.raises(ToolRegistryError, match="write tool cannot be registered"):
        ToolRegistry(
            (
                FakeWriteTool(
                    FixedClock(),
                    allowed_phases=frozenset({RunPhase.RESEARCH}),
                ),
            )
        )


def test_tool_spec_declares_retry_cost_result_and_permission_metadata() -> None:
    spec = FakeQuoteTool(FixedClock()).spec

    assert spec.version == "1"
    assert spec.retry.max_attempts == 1
    assert spec.estimated_cost == 0
    assert spec.max_result_bytes == 8192
    assert spec.required_roles == frozenset()


def test_registry_rejects_automatic_write_retries() -> None:
    with pytest.raises(ToolRegistryError, match="automatic retries"):
        ToolRegistry((RetryingWriteTool(FixedClock()),))


@pytest.mark.asyncio
async def test_registry_does_not_discover_or_execute_write_tool_in_research() -> None:
    clock = FixedClock()
    tool = FakeWriteTool(clock)
    registry = ToolRegistry((tool,))

    assert registry.specs_for_phase(RunPhase.RESEARCH) == ()
    assert registry.specs_for_phase(RunPhase.EXECUTION) == (tool.spec,)

    outcome = await registry.execute(
        ToolRequest(id="write-call", name=tool.spec.name, arguments={}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                actor_id="actor",
            ),
            run_id="run",
            tool_call_id="write-call",
        ),
        RunPhase.RESEARCH,
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "TOOL_EFFECT_NOT_ALLOWED_IN_PHASE"
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_registry_denies_missing_required_role_before_execution() -> None:
    clock = FixedClock()
    tool = RoleProtectedTool(clock)
    registry = ToolRegistry((tool,))

    outcome = await registry.execute(
        ToolRequest(id="role-call", name=tool.spec.name, arguments={"symbol": "NVDA"}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                actor_id="actor",
                roles=frozenset({"researcher"}),
            ),
            run_id="run",
            tool_call_id="role-call",
        ),
        RunPhase.RESEARCH,
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "TOOL_PERMISSION_DENIED"
    assert tool.calls == 0


@pytest.mark.asyncio
async def test_registry_rejects_oversized_success_before_observation() -> None:
    clock = FixedClock()
    tool = OversizedResultTool(clock)
    registry = ToolRegistry((tool,))

    outcome = await registry.execute(
        ToolRequest(id="size-call", name=tool.spec.name, arguments={"symbol": "NVDA"}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                actor_id="actor",
            ),
            run_id="run",
            tool_call_id="size-call",
        ),
        RunPhase.RESEARCH,
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "TOOL_RESULT_TOO_LARGE"


@pytest.mark.asyncio
async def test_registry_strips_undeclared_retryability() -> None:
    clock = FixedClock()
    tool = UndeclaredRetryTool(clock)
    registry = ToolRegistry((tool,))

    outcome = await registry.execute(
        ToolRequest(id="retry-call", name=tool.spec.name, arguments={"symbol": "NVDA"}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="strategy"),
                actor_id="actor",
            ),
            run_id="run",
            tool_call_id="retry-call",
        ),
        RunPhase.RESEARCH,
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "RETRY_NOT_DECLARED"
    assert outcome.retryable is False
