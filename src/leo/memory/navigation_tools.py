"""Typed progressive-memory tools whose authority is sealed by the Slack runtime."""

from __future__ import annotations

import logging
import re

from pydantic import Field, JsonValue, ValidationError

from leo.harness.models import (
    ContractModel,
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
from leo.harness.ports import Clock, Tool
from leo.memory.navigation import MemoryNavigationAuthority, MemoryNavigationError
from leo.persistence.memory_navigation import PostgresProgressiveMemoryService

logger = logging.getLogger(__name__)


class _SearchArguments(ContractModel):
    query: str = Field(min_length=1, max_length=500)
    limit: int = Field(default=8, ge=1, le=12)


class _OpenArguments(ContractModel):
    handle: str = Field(min_length=16, max_length=256)
    start_ordinal: int = Field(default=0, ge=0, le=127)
    max_chunks: int = Field(default=4, ge=1, le=8)


class _SearchWithinArguments(ContractModel):
    handle: str = Field(min_length=16, max_length=256)
    query: str = Field(min_length=1, max_length=500)
    max_chunks: int = Field(default=4, ge=1, le=8)


class _MemoryNavigationTool:
    def __init__(
        self,
        *,
        service: PostgresProgressiveMemoryService,
        authority: MemoryNavigationAuthority,
        clock: Clock,
        spec: ToolSpec,
    ) -> None:
        self._service = service
        self._authority = authority
        self._clock = clock
        self._spec = spec

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def _authority_failure(self, context: ToolExecutionContext) -> ToolFailure | None:
        if context.trusted_scope.namespace != self._authority.scope:
            return _failure("MEMORY_AUTHORITY_MISMATCH", "Memory scope changed during the run.")
        if context.trusted_scope.actor_id != self._authority.actor_id:
            return _failure("MEMORY_AUTHORITY_MISMATCH", "Memory actor changed during the run.")
        if context.run_id != self._authority.run_id:
            return _failure("MEMORY_AUTHORITY_MISMATCH", "Memory run authority changed.")
        return None


class MemorySearchTool(_MemoryNavigationTool):
    def __init__(
        self,
        *,
        service: PostgresProgressiveMemoryService,
        authority: MemoryNavigationAuthority,
        clock: Clock,
    ) -> None:
        super().__init__(
            service=service,
            authority=authority,
            clock=clock,
            spec=_spec(
                name="memory.search",
                description=(
                    "Search only the current Slack destination or the current authorized 1:1-DM "
                    "source set. Returns short inline memories or bounded cards with opaque "
                    "handles."
                ),
                properties={
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 12, "default": 8},
                },
                required=("query",),
            ),
        )

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _SearchArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        mismatch = self._authority_failure(context)
        if mismatch is not None:
            return mismatch
        try:
            parsed = _SearchArguments.model_validate(arguments)
        except ValidationError:
            return _failure("MEMORY_QUERY_INVALID", "The bounded memory query was invalid.")
        try:
            result = await self._service.search(
                self._authority,
                query=parsed.query,
                limit=parsed.limit,
                now=self._clock.now(),
            )
            outcome = ToolSuccess(
                data=result.model_dump(mode="json"),
                source=SourceRef(provider="leo_memory", reference=result.query_hash),
                observed_at=self._clock.now(),
            )
        except MemoryNavigationError as exc:
            return _failure(
                "MEMORY_SEARCH_DENIED", f"Memory search stopped safely: {exc.safe_code}."
            )
        except Exception as exc:
            return _unexpected_read_failure(self._spec.name, exc)
        return outcome


class MemoryOpenTool(_MemoryNavigationTool):
    def __init__(
        self,
        *,
        service: PostgresProgressiveMemoryService,
        authority: MemoryNavigationAuthority,
        clock: Clock,
    ) -> None:
        super().__init__(
            service=service,
            authority=authority,
            clock=clock,
            spec=_spec(
                name="memory.open",
                description=(
                    "Open a bounded chunk window from an opaque memory handle. The handle is "
                    "reauthorized for run, destination, membership, lifecycle, revision, and "
                    "budget."
                ),
                properties={
                    "handle": {"type": "string", "minLength": 16, "maxLength": 256},
                    "start_ordinal": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 127,
                        "default": 0,
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                },
                required=("handle",),
            ),
        )

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _OpenArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        mismatch = self._authority_failure(context)
        if mismatch is not None:
            return mismatch
        try:
            parsed = _OpenArguments.model_validate(arguments)
        except ValidationError:
            return _failure("MEMORY_OPEN_INVALID", "The bounded memory-open request was invalid.")
        try:
            result = await self._service.open(
                self._authority,
                handle=parsed.handle,
                start_ordinal=parsed.start_ordinal,
                max_chunks=parsed.max_chunks,
                now=self._clock.now(),
            )
            outcome = ToolSuccess(
                data=result.model_dump(mode="json"),
                source=SourceRef(provider="leo_memory", reference=result.reference),
                observed_at=self._clock.now(),
            )
        except MemoryNavigationError as exc:
            return _failure("MEMORY_OPEN_DENIED", f"Memory open stopped safely: {exc.safe_code}.")
        except Exception as exc:
            return _unexpected_read_failure(self._spec.name, exc)
        return outcome


class MemorySearchWithinTool(_MemoryNavigationTool):
    def __init__(
        self,
        *,
        service: PostgresProgressiveMemoryService,
        authority: MemoryNavigationAuthority,
        clock: Clock,
    ) -> None:
        super().__init__(
            service=service,
            authority=authority,
            clock=clock,
            spec=_spec(
                name="memory.search_within",
                description=(
                    "Search within one already-authorized opaque memory handle and return only "
                    "matching bounded chunks after full reauthorization."
                ),
                properties={
                    "handle": {"type": "string", "minLength": 16, "maxLength": 256},
                    "query": {"type": "string", "minLength": 1, "maxLength": 500},
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                },
                required=("handle", "query"),
            ),
        )

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _SearchWithinArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        mismatch = self._authority_failure(context)
        if mismatch is not None:
            return mismatch
        try:
            parsed = _SearchWithinArguments.model_validate(arguments)
        except ValidationError:
            return _failure(
                "MEMORY_SEARCH_WITHIN_INVALID",
                "The bounded memory search-within request was invalid.",
            )
        try:
            result = await self._service.search_within(
                self._authority,
                handle=parsed.handle,
                query=parsed.query,
                max_chunks=parsed.max_chunks,
                now=self._clock.now(),
            )
            outcome = ToolSuccess(
                data=result.model_dump(mode="json"),
                source=SourceRef(provider="leo_memory", reference=result.reference),
                observed_at=self._clock.now(),
            )
        except MemoryNavigationError as exc:
            return _failure(
                "MEMORY_SEARCH_WITHIN_DENIED",
                f"Memory search-within stopped safely: {exc.safe_code}.",
            )
        except Exception as exc:
            return _unexpected_read_failure(self._spec.name, exc)
        return outcome


def build_memory_navigation_tools(
    *,
    service: PostgresProgressiveMemoryService,
    authority: MemoryNavigationAuthority,
    clock: Clock,
) -> tuple[Tool, ...]:
    return (
        MemorySearchTool(service=service, authority=authority, clock=clock),
        MemoryOpenTool(service=service, authority=authority, clock=clock),
        MemorySearchWithinTool(service=service, authority=authority, clock=clock),
    )


def _spec(
    *,
    name: str,
    description: str,
    properties: dict[str, JsonValue],
    required: tuple[str, ...],
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        domain="memory",
        input_schema={
            "type": "object",
            "properties": properties,
            "required": list(required),
            "additionalProperties": False,
        },
        effect=ToolEffect.READ,
        allowed_phases=frozenset({RunPhase.RESEARCH}),
        retry=ToolRetryPolicy(max_attempts=2),
        timeout_seconds=10,
        max_result_bytes=16_384,
        required_roles=frozenset({"researcher"}),
    )


def _failure(code: str, message: str, *, retryable: bool = False) -> ToolFailure:
    return ToolFailure(code=code, safe_message=message, retryable=retryable)


def _unexpected_read_failure(tool_name: str, exc: Exception) -> ToolFailure:
    """Expose only a bounded retry signal while retaining content-free diagnostics."""

    candidate_code = getattr(exc, "safe_code", None)
    safe_code = (
        candidate_code
        if isinstance(candidate_code, str) and re.fullmatch(r"[a-z0-9_]{1,64}", candidate_code)
        else "unclassified"
    )
    logger.warning(
        "Memory navigation read failed safely: tool=%s exception_type=%s safe_code=%s",
        tool_name,
        type(exc).__name__,
        safe_code,
    )
    return _failure(
        "MEMORY_SEARCH_UNAVAILABLE",
        "The authorized memory read was temporarily unavailable; retrying once is safe.",
        retryable=True,
    )
