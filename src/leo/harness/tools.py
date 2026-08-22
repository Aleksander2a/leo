"""Tool registry, phase permissions, validation, and safe execution."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable

from pydantic import JsonValue, ValidationError

from leo.harness.models import (
    RunPhase,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRequest,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.ports import Tool

RESERVED_AUTHORITY_KEYS = frozenset(
    {
        "organization_id",
        "org_id",
        "strategy_id",
        "scope",
        "trusted_scope",
        "actor_id",
        "roles",
        "approved",
        "approval",
        "visibility",
        "namespace_id",
        "conversation_id",
        "channel_id",
        "team_id",
        "event_id",
        "task_id",
        "message_reference",
    }
)


class ToolRegistryError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        spec = tool.spec
        if spec.name in self._tools:
            raise ToolRegistryError(f"duplicate tool: {spec.name}")
        reserved = RESERVED_AUTHORITY_KEYS.intersection(_schema_property_names(spec.input_schema))
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ToolRegistryError(f"tool schema contains reserved authority fields: {names}")
        disallowed_phases = spec.allowed_phases.difference(_EFFECT_ALLOWED_PHASES[spec.effect])
        if disallowed_phases:
            if spec.effect is ToolEffect.WRITE and RunPhase.RESEARCH in disallowed_phases:
                raise ToolRegistryError("write tool cannot be registered for research phase")
            raise ToolRegistryError("tool effect is not allowed in its advertised phase")
        if spec.effect is not ToolEffect.READ and spec.retry.max_attempts > 1:
            raise ToolRegistryError("effectful tool cannot declare automatic retries")
        self._tools[spec.name] = tool

    def specs_for_phase(self, phase: RunPhase) -> tuple[ToolSpec, ...]:
        return tuple(
            tool.spec
            for tool in self._tools.values()
            if phase in tool.spec.allowed_phases and _effect_allowed(phase, tool.spec.effect)
        )

    def specs_for_context(
        self,
        phase: RunPhase,
        trusted_scope: TrustedScope,
    ) -> tuple[ToolSpec, ...]:
        """Return only schemas executable for trusted phase and role authority.

        This is the policy boundary before any semantic capability recall.  It is
        deliberately separate from ``specs_for_phase`` so legacy inspection callers
        retain their phase-only view while model-visible selection fails closed.
        """

        return tuple(
            tool.spec
            for tool in self._tools.values()
            if phase in tool.spec.allowed_phases
            and _effect_allowed(phase, tool.spec.effect)
            and tool.spec.required_roles.issubset(trusted_scope.roles)
        )

    def requests_are_parallel_safe(
        self,
        requests: tuple[ToolRequest, ...],
        phase: RunPhase,
    ) -> bool:
        """Return true only for an independent batch of eligible read tools.

        A model tool batch has no dependency-reference syntax, so calls in the same
        batch are independent by contract. Unknown, phase-ineligible, or effectful
        calls retain sequential fail-closed behavior.
        """

        if len(requests) < 2:
            return False
        for request in requests:
            tool = self._tools.get(request.name)
            if (
                tool is None
                or tool.spec.effect is not ToolEffect.READ
                or phase not in tool.spec.allowed_phases
                or not _effect_allowed(phase, tool.spec.effect)
            ):
                return False
        return True

    async def execute(
        self,
        request: ToolRequest,
        context: ToolExecutionContext,
        phase: RunPhase,
    ) -> ToolOutcome:
        tool = self._tools.get(request.name)
        if tool is None:
            return ToolFailure(
                code="UNKNOWN_TOOL",
                safe_message=f"Tool {request.name!r} is not registered.",
            )
        if not _effect_allowed(phase, tool.spec.effect):
            return ToolFailure(
                code="TOOL_EFFECT_NOT_ALLOWED_IN_PHASE",
                safe_message=f"Tool {request.name!r} is unavailable in phase {phase.value!r}.",
            )
        if phase not in tool.spec.allowed_phases:
            return ToolFailure(
                code="TOOL_NOT_ALLOWED_IN_PHASE",
                safe_message=f"Tool {request.name!r} is unavailable in phase {phase.value!r}.",
            )
        if not tool.spec.required_roles.issubset(context.trusted_scope.roles):
            return ToolFailure(
                code="TOOL_PERMISSION_DENIED",
                safe_message=f"Tool {request.name!r} is unavailable for the current role.",
            )
        try:
            arguments = tool.validate(request.arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            return ToolFailure(
                code="INVALID_TOOL_ARGUMENTS",
                safe_message=(
                    f"Arguments for {request.name!r} failed validation: {type(exc).__name__}."
                ),
            )
        try:
            async with asyncio.timeout(tool.spec.timeout_seconds):
                outcome = await tool.execute(arguments, context)
        except TimeoutError:
            outcome = ToolFailure(
                code="TOOL_TIMEOUT",
                retryable=True,
                safe_message=f"Tool {request.name!r} timed out.",
            )
        except Exception as exc:
            outcome = ToolFailure(
                code="TOOL_EXECUTION_ERROR",
                safe_message=f"Tool {request.name!r} failed with {type(exc).__name__}.",
            )
        if isinstance(outcome, ToolFailure):
            if outcome.retryable and tool.spec.retry.max_attempts <= 1:
                return outcome.model_copy(update={"retryable": False})
            return outcome
        if isinstance(outcome, ToolSuccess):
            result_size = len(
                json.dumps(
                    outcome.data,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                ).encode("utf-8")
            )
            if result_size > tool.spec.max_result_bytes:
                return ToolFailure(
                    code="TOOL_RESULT_TOO_LARGE",
                    safe_message=f"Tool {request.name!r} returned an oversized result.",
                )
        return outcome


_EFFECT_ALLOWED_PHASES: dict[ToolEffect, frozenset[RunPhase]] = {
    ToolEffect.READ: frozenset(RunPhase),
    ToolEffect.STATE_MUTATION: frozenset({RunPhase.RESEARCH}),
    ToolEffect.WRITE: frozenset({RunPhase.EXECUTION}),
}


def _effect_allowed(phase: RunPhase, effect: ToolEffect) -> bool:
    return phase in _EFFECT_ALLOWED_PHASES[effect]


def _schema_property_names(value: JsonValue) -> set[str]:
    names: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            names.update(str(name) for name in properties)
        for child in value.values():
            names.update(_schema_property_names(child))
    elif isinstance(value, list):
        for child in value:
            names.update(_schema_property_names(child))
    return names
