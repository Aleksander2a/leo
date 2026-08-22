"""Run-bound progressive reopening for compacted Slack thread context."""

from __future__ import annotations

import asyncio
import hashlib
import json

from pydantic import Field, JsonValue, ValidationError, model_validator

from leo.harness.models import (
    ContractModel,
    NonEmptyStr,
    RunPhase,
    ScopeKey,
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
from leo.harness.thread_context import ThreadContextRange, thread_context_source_digest
from leo.memory.navigation import deterministic_memory_chunks


class ThreadContextNavigationError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class ThreadContextAuthority(ContractModel):
    """Sealed authority for one admitted run's exact Slack thread snapshot."""

    scope: ScopeKey
    team_id: NonEmptyStr
    destination_id: NonEmptyStr
    actor_id: NonEmptyStr
    task_id: NonEmptyStr
    run_id: NonEmptyStr
    thread_root_ts: NonEmptyStr
    current_message_ts: NonEmptyStr
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def exact_destination_is_authorized(self) -> ThreadContextAuthority:
        if self.destination_id not in self.allowed_conversation_ids:
            raise ValueError("thread context destination is outside its access projection")
        if self.allowed_conversation_ids != tuple(sorted(set(self.allowed_conversation_ids))):
            raise ValueError("thread context access projection must be sorted and unique")
        return self


class ThreadContextChunk(ContractModel):
    ordinal: int = Field(ge=0)
    source_item_digest: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    text: NonEmptyStr = Field(max_length=1_200)


class ThreadContextOpenResult(ContractModel):
    handle: NonEmptyStr
    range_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    chunks: tuple[ThreadContextChunk, ...] = Field(min_length=1, max_length=8)
    next_ordinal: int | None = Field(default=None, ge=0)
    source_conversation: NonEmptyStr
    thread_root_ts: NonEmptyStr
    policy_version: NonEmptyStr = "thread-context-navigation-v1"


class _OpenArguments(ContractModel):
    handle: str = Field(min_length=16, max_length=256)
    start_ordinal: int = Field(default=0, ge=0, le=4_095)
    max_chunks: int = Field(default=4, ge=1, le=8)


class ThreadContextSnapshotService:
    """In-memory exact snapshot rebuilt on every admitted run/restart."""

    def __init__(
        self,
        *,
        authority: ThreadContextAuthority,
        ranges: tuple[ThreadContextRange, ...],
        max_calls: int = 8,
        max_returned_bytes: int = 64_000,
    ) -> None:
        if max_calls < 1 or max_calls > 64:
            raise ValueError("thread context open-call budget must be between 1 and 64")
        if max_returned_bytes < 1_200 or max_returned_bytes > 256_000:
            raise ValueError("thread context returned-byte budget is invalid")
        if len({item.handle for item in ranges}) != len(ranges):
            raise ValueError("thread context handles must be unique")
        if any(
            thread_context_source_digest(source_range.items) != source_range.digest
            for source_range in ranges
        ):
            raise ValueError("thread context range digest does not match its exact source")
        if any(
            item.conversation_id != authority.destination_id
            for source_range in ranges
            for item in source_range.items
        ):
            raise ValueError("thread context range escaped the exact destination")
        self._authority = authority
        self._ranges = {item.handle: item for item in ranges}
        self._max_calls = max_calls
        self._max_returned_bytes = max_returned_bytes
        self._calls = 0
        self._returned_bytes = 0
        self._lock = asyncio.Lock()

    async def open(
        self,
        authority: ThreadContextAuthority,
        *,
        handle: str,
        start_ordinal: int,
        max_chunks: int,
    ) -> ThreadContextOpenResult:
        if authority != self._authority:
            raise ThreadContextNavigationError("thread_context_authority_mismatch")
        async with self._lock:
            self._calls += 1
            if self._calls > self._max_calls:
                raise ThreadContextNavigationError("thread_context_open_budget_exhausted")
            source_range = self._ranges.get(handle)
            if source_range is None:
                raise ThreadContextNavigationError("thread_context_handle_not_authorized")
            chunks = _range_chunks(source_range)
            if start_ordinal >= len(chunks):
                raise ThreadContextNavigationError("thread_context_chunk_ordinal_out_of_range")
            end = min(len(chunks), start_ordinal + max_chunks)
            selected = chunks[start_ordinal:end]
            returned_bytes = len(
                json.dumps(
                    [item.model_dump(mode="json") for item in selected],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            if self._returned_bytes + returned_bytes > self._max_returned_bytes:
                raise ThreadContextNavigationError("thread_context_byte_budget_exhausted")
            self._returned_bytes += returned_bytes
            return ThreadContextOpenResult(
                handle=source_range.handle,
                range_digest=source_range.digest,
                chunks=selected,
                next_ordinal=end if end < len(chunks) else None,
                source_conversation=authority.destination_id,
                thread_root_ts=authority.thread_root_ts,
            )


class ThreadContextOpenTool:
    def __init__(
        self,
        *,
        service: ThreadContextSnapshotService,
        authority: ThreadContextAuthority,
        clock: Clock,
    ) -> None:
        self._service = service
        self._authority = authority
        self._clock = clock
        self._spec = ToolSpec(
            name="thread_context.open",
            description=(
                "Open bounded exact chunks from an opaque handle emitted by a compacted Slack "
                "thread summary. Scope, conversation, thread, actor, run, and access projection "
                "are sealed by the server and cannot be supplied by the model."
            ),
            domain="memory",
            input_schema={
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "minLength": 16, "maxLength": 256},
                    "start_ordinal": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 4_095,
                        "default": 0,
                    },
                    "max_chunks": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                },
                "required": ["handle"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            retry=ToolRetryPolicy(max_attempts=1),
            timeout_seconds=5,
            max_result_bytes=16_384,
            required_roles=frozenset({"researcher"}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _OpenArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        if (
            context.run_id != self._authority.run_id
            or context.trusted_scope.namespace != self._authority.scope
            or context.trusted_scope.actor_id != self._authority.actor_id
        ):
            return _failure(
                "THREAD_CONTEXT_AUTHORITY_MISMATCH",
                "Thread context authority changed during the run.",
            )
        try:
            parsed = _OpenArguments.model_validate(arguments)
            result = await self._service.open(
                self._authority,
                handle=parsed.handle,
                start_ordinal=parsed.start_ordinal,
                max_chunks=parsed.max_chunks,
            )
        except ValidationError:
            return _failure(
                "THREAD_CONTEXT_OPEN_INVALID",
                "The bounded thread-context request was invalid.",
            )
        except ThreadContextNavigationError as exc:
            return _failure(
                "THREAD_CONTEXT_OPEN_DENIED",
                f"Thread context open stopped safely: {exc.safe_code}.",
            )
        return ToolSuccess(
            data=result.model_dump(mode="json"),
            source=SourceRef(provider="leo_thread_context", reference=result.range_digest),
            observed_at=self._clock.now(),
        )


def build_thread_context_tools(
    *,
    ranges: tuple[ThreadContextRange, ...],
    authority: ThreadContextAuthority,
    clock: Clock,
) -> tuple[Tool, ...]:
    if not ranges:
        return ()
    service = ThreadContextSnapshotService(authority=authority, ranges=ranges)
    return (ThreadContextOpenTool(service=service, authority=authority, clock=clock),)


def _range_chunks(source_range: ThreadContextRange) -> tuple[ThreadContextChunk, ...]:
    chunks: list[ThreadContextChunk] = []
    for item in source_range.items:
        source_digest = hashlib.sha256(item.content.encode("utf-8")).hexdigest()
        for text in deterministic_memory_chunks(item.content):
            chunks.append(
                ThreadContextChunk(
                    ordinal=len(chunks),
                    source_item_digest=source_digest,
                    text=text,
                )
            )
    return tuple(chunks)


def _failure(code: str, message: str) -> ToolFailure:
    return ToolFailure(code=code, safe_message=message)
