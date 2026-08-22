"""Explicit, sanitized, content-addressed provider recordings and strict replay."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from enum import StrEnum
from typing import cast

from pydantic import Field, JsonValue, TypeAdapter, model_validator

from leo.harness.models import (
    ContractModel,
    ModelRequest,
    ModelTurnResult,
    NonEmptyStr,
    ToolExecutionContext,
    ToolOutcome,
    ToolSpec,
)
from leo.harness.ports import ModelGateway, Tool


class RecordingMode(StrEnum):
    CAPTURE = "capture"
    REPLAY = "replay"


class RecordingLane(StrEnum):
    PARENT_MODEL = "parent_model"
    CHILD_MODEL = "child_model"
    TOOL = "tool"


class RecordingSanitizationError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class RecordingMiss(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class RecordingKey(ContractModel):
    provider: NonEmptyStr
    operation: NonEmptyStr
    version: NonEmptyStr
    lane: RecordingLane = RecordingLane.TOOL
    parent_id: NonEmptyStr = "root"
    node_id: str | None = None
    call_id: NonEmptyStr = "call"
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sequence: int = Field(ge=0)

    @model_validator(mode="after")
    def metadata_is_sanitized(self) -> RecordingKey:
        metadata = tuple(
            item
            for item in (
                self.provider,
                self.operation,
                self.version,
                self.parent_id,
                self.node_id,
                self.call_id,
            )
            if item is not None
        )
        if any(_secret_like(item) for item in metadata):
            raise RecordingSanitizationError("recording_secret_detected")
        return self


class RecordedExchange(ContractModel):
    recording_id: NonEmptyStr
    key: RecordingKey
    request: dict[str, JsonValue]
    response: dict[str, JsonValue]
    safe_headers: dict[str, NonEmptyStr] = Field(default_factory=dict)
    usage: dict[str, int | float | str] = Field(default_factory=dict)
    elapsed_ms: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class RecordingStore:
    def __init__(self) -> None:
        self._records: dict[
            tuple[
                str,
                str,
                str,
                str,
                str,
                str,
                str | None,
                str,
                int,
            ],
            RecordedExchange,
        ] = {}

    def put(self, exchange: RecordedExchange) -> None:
        expected = _exchange_digest(exchange)
        if exchange.digest != expected:
            raise RecordingMiss("recording_digest_mismatch")
        key = _key_tuple(exchange.key)
        existing = self._records.get(key)
        if existing is not None and existing.digest != exchange.digest:
            raise RecordingMiss("recording_key_conflict")
        self._records[key] = exchange

    def capture(
        self,
        *,
        provider: str,
        operation: str,
        version: str,
        request: Mapping[str, JsonValue],
        response: Mapping[str, JsonValue],
        sequence: int,
        lane: RecordingLane = RecordingLane.TOOL,
        parent_id: str = "root",
        node_id: str | None = None,
        call_id: str = "call",
        safe_headers: Mapping[str, str] | None = None,
        usage: Mapping[str, int | float | str] | None = None,
        elapsed_ms: int = 0,
    ) -> RecordedExchange:
        sanitized_request = sanitize_payload(request)
        sanitized_response = sanitize_payload(response)
        request_hash = _digest(sanitized_request)
        key = RecordingKey(
            provider=provider,
            operation=operation,
            version=version,
            lane=lane,
            parent_id=parent_id,
            node_id=node_id,
            call_id=call_id,
            request_hash=request_hash,
            sequence=sequence,
        )
        clean_headers = dict(safe_headers or {})
        if any(key.casefold() not in _SAFE_HEADER_NAMES for key in clean_headers):
            raise RecordingSanitizationError("recording_header_not_allowlisted")
        if any(_secret_like(value) for value in clean_headers.values()):
            raise RecordingSanitizationError("recording_secret_detected")
        clean_usage = dict(usage or {})
        if any(
            _secret_like(str(key)) or _secret_like(str(value)) for key, value in clean_usage.items()
        ):
            raise RecordingSanitizationError("recording_secret_detected")
        envelope = {
            "key": key.model_dump(mode="json"),
            "request": sanitized_request,
            "response": sanitized_response,
            "safe_headers": clean_headers,
            "usage": clean_usage,
            "elapsed_ms": elapsed_ms,
        }
        digest = _digest(envelope)
        exchange = RecordedExchange(
            recording_id=f"recording:{digest[:16]}",
            key=key,
            request=sanitized_request,
            response=sanitized_response,
            safe_headers=clean_headers,
            usage=clean_usage,
            elapsed_ms=elapsed_ms,
            digest=digest,
        )
        self.put(exchange)
        return exchange

    def replay(
        self,
        *,
        provider: str,
        operation: str,
        version: str,
        request: Mapping[str, JsonValue],
        sequence: int,
        lane: RecordingLane = RecordingLane.TOOL,
        parent_id: str = "root",
        node_id: str | None = None,
        call_id: str = "call",
    ) -> RecordedExchange:
        key = (
            provider,
            operation,
            version,
            _digest(sanitize_payload(request)),
            lane.value,
            parent_id,
            node_id,
            call_id,
            sequence,
        )
        try:
            return self._records[key]
        except KeyError as exc:
            raise RecordingMiss("recording_miss") from exc

    def replay_call(
        self,
        *,
        provider: str,
        operation: str,
        version: str,
        request: Mapping[str, JsonValue],
        lane: RecordingLane,
        parent_id: str,
        node_id: str | None,
        call_id: str,
    ) -> RecordedExchange:
        """Match a stable parallel call without depending on completion order."""

        request_hash = _digest(sanitize_payload(request))
        matches = tuple(
            exchange
            for exchange in self._records.values()
            if exchange.key.provider == provider
            and exchange.key.operation == operation
            and exchange.key.version == version
            and exchange.key.request_hash == request_hash
            and exchange.key.lane is lane
            and exchange.key.parent_id == parent_id
            and exchange.key.node_id == node_id
            and exchange.key.call_id == call_id
        )
        if not matches:
            raise RecordingMiss("recording_miss")
        if len(matches) != 1:
            raise RecordingMiss("recording_ambiguous")
        return matches[0]

    def snapshot(self) -> tuple[RecordedExchange, ...]:
        return tuple(sorted(self._records.values(), key=_recording_sort_key))


class RecordingReplayCursor:
    """Strict per-parent/node/lane sequence cursor for executable replay."""

    def __init__(self, store: RecordingStore) -> None:
        self._store = store
        self._next: dict[tuple[str, str, str, str, str, str | None], int] = {}
        self._consumed_call_recordings: set[str] = set()

    def replay_next(
        self,
        *,
        provider: str,
        operation: str,
        version: str,
        request: Mapping[str, JsonValue],
        lane: RecordingLane,
        parent_id: str,
        node_id: str | None,
        call_id: str,
    ) -> RecordedExchange:
        lane_key = (provider, operation, version, lane.value, parent_id, node_id)
        sequence = self._next.get(lane_key, 0)
        exchange = self._store.replay(
            provider=provider,
            operation=operation,
            version=version,
            request=request,
            sequence=sequence,
            lane=lane,
            parent_id=parent_id,
            node_id=node_id,
            call_id=call_id,
        )
        self._next[lane_key] = sequence + 1
        return exchange

    def replay_parallel_call(
        self,
        *,
        provider: str,
        operation: str,
        version: str,
        request: Mapping[str, JsonValue],
        lane: RecordingLane,
        parent_id: str,
        node_id: str | None,
        call_id: str,
    ) -> RecordedExchange:
        exchange = self._store.replay_call(
            provider=provider,
            operation=operation,
            version=version,
            request=request,
            lane=lane,
            parent_id=parent_id,
            node_id=node_id,
            call_id=call_id,
        )
        if exchange.recording_id in self._consumed_call_recordings:
            raise RecordingMiss("recording_call_already_consumed")
        self._consumed_call_recordings.add(exchange.recording_id)
        return exchange

    @property
    def positions(self) -> dict[tuple[str, str, str, str, str, str | None], int]:
        return dict(self._next)


class RecordingModelGateway:
    """Capture or replay a normalized model port without a replay network delegate."""

    def __init__(
        self,
        store: RecordingStore,
        *,
        mode: RecordingMode,
        provider: str,
        version: str,
        lane: RecordingLane,
        parent_id: str,
        node_id: str | None = None,
        delegate: ModelGateway | None = None,
    ) -> None:
        if lane not in {RecordingLane.PARENT_MODEL, RecordingLane.CHILD_MODEL}:
            raise ValueError("model recording requires a parent or child model lane")
        if mode is RecordingMode.CAPTURE and delegate is None:
            raise ValueError("model capture requires an explicitly supplied delegate")
        self._store = store
        self._mode = mode
        self._provider = provider
        self._version = version
        self._lane = lane
        self._parent_id = parent_id
        self._node_id = node_id
        self._delegate = delegate
        self._sequence = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        sequence = self._sequence
        self._sequence += 1
        call_id = f"model-turn-{sequence}"
        request_payload: dict[str, JsonValue] = {
            "model_request": cast(JsonValue, request.model_dump(mode="json"))
        }
        if self._mode is RecordingMode.CAPTURE:
            if self._delegate is None:  # Defensive; constructor rejects this state.
                raise RuntimeError("recording_capture_delegate_missing")
            result = await self._delegate.decide(request)
            self._store.capture(
                provider=self._provider,
                operation="model.decide",
                version=self._version,
                request=request_payload,
                response={"model_result": cast(JsonValue, result.model_dump(mode="json"))},
                sequence=sequence,
                lane=self._lane,
                parent_id=self._parent_id,
                node_id=self._node_id,
                call_id=call_id,
                usage=cast(
                    dict[str, int | float | str],
                    result.usage.model_dump(exclude_none=True),
                ),
            )
            return result
        exchange = self._store.replay(
            provider=self._provider,
            operation="model.decide",
            version=self._version,
            request=request_payload,
            sequence=sequence,
            lane=self._lane,
            parent_id=self._parent_id,
            node_id=self._node_id,
            call_id=call_id,
        )
        try:
            return ModelTurnResult.model_validate(exchange.response["model_result"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordingMiss("recording_response_invalid") from exc


class RecordingTool:
    """Capture or replay one normalized tool adapter with stable call-ID matching."""

    def __init__(
        self,
        store: RecordingStore,
        *,
        mode: RecordingMode,
        provider: str,
        version: str,
        spec: ToolSpec,
        node_id: str | None = None,
        delegate: Tool | None = None,
    ) -> None:
        if mode is RecordingMode.CAPTURE and delegate is None:
            raise ValueError("tool capture requires an explicitly supplied delegate")
        if delegate is not None and delegate.spec != spec:
            raise ValueError("recording tool spec must equal its delegate spec")
        self._store = store
        self._mode = mode
        self._provider = provider
        self._version = version
        self._spec = spec
        self._node_id = node_id
        self._delegate = delegate
        self._call_sequences: dict[str, int] = {}

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if self._delegate is None:
            return dict(arguments)
        return self._delegate.validate(arguments)

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        sequence = self._call_sequences.get(context.tool_call_id, 0)
        self._call_sequences[context.tool_call_id] = sequence + 1
        request_payload: dict[str, JsonValue] = {"arguments": arguments}
        if self._mode is RecordingMode.CAPTURE:
            if self._delegate is None:  # Defensive; constructor rejects this state.
                raise RuntimeError("recording_capture_delegate_missing")
            outcome = await self._delegate.execute(arguments, context)
            self._store.capture(
                provider=self._provider,
                operation=self._spec.name,
                version=self._version,
                request=request_payload,
                response={
                    "tool_outcome": cast(
                        JsonValue,
                        _TOOL_OUTCOME_ADAPTER.dump_python(outcome, mode="json"),
                    )
                },
                sequence=sequence,
                lane=RecordingLane.TOOL,
                parent_id=context.run_id,
                node_id=self._node_id,
                call_id=context.tool_call_id,
            )
            return outcome
        exchange = self._store.replay(
            provider=self._provider,
            operation=self._spec.name,
            version=self._version,
            request=request_payload,
            sequence=sequence,
            lane=RecordingLane.TOOL,
            parent_id=context.run_id,
            node_id=self._node_id,
            call_id=context.tool_call_id,
        )
        try:
            return _TOOL_OUTCOME_ADAPTER.validate_python(exchange.response["tool_outcome"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RecordingMiss("recording_response_invalid") from exc


def sanitize_payload(payload: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    output: dict[str, JsonValue] = {}
    for key, value in payload.items():
        normalized_key = key.lower().replace("-", "_")
        if _secret_field_name(normalized_key):
            output[key] = "[REDACTED]"
        elif normalized_key in {"prompt", "content", "body", "raw"}:
            output[key] = {"sha256": _digest(value), "length": len(str(value))}
        else:
            output[key] = _sanitize_value(value)
    if any(_secret_like(str(value)) for value in _walk(output)):
        raise RecordingSanitizationError("recording_secret_detected")
    return output


def _sanitize_value(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return sanitize_payload(value)
    if isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    return value


def _walk(value: JsonValue) -> Iterator[object]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def _secret_like(value: str) -> bool:
    return bool(
        re.search(
            r"(?i)(?:\bbearer\s+\S+|\bxox[baprs]-\S+|\bsk-[A-Za-z0-9_-]{8,}"
            r"|\bgh[pousr]_[A-Za-z0-9_]{8,}|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\."
            r"[A-Za-z]{2,}\b|\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\."
            r"[A-Za-z0-9_-]+\b|(?:[A-Za-z]:\\(?:Users|home)\\|/(?:home|Users)/)\S+)",
            value,
        )
    )


def _secret_field_name(normalized_key: str) -> bool:
    if normalized_key in {
        "authorization",
        "api_key",
        "apikey",
        "token",
        "password",
        "secret",
        "cookie",
        "set_cookie",
        "access_token",
        "refresh_token",
        "auth_token",
        "id_token",
        "bot_token",
        "slack_token",
    }:
        return True
    safe_token_metrics = (
        "tokens",
        "token_count",
        "token_limit",
        "token_budget",
    )
    if normalized_key.endswith(safe_token_metrics):
        return False
    return normalized_key.endswith(("_api_key", "_password", "_secret", "_cookie", "_token"))


def _digest(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _key_tuple(
    key: RecordingKey,
) -> tuple[str, str, str, str, str, str, str | None, str, int]:
    return (
        key.provider,
        key.operation,
        key.version,
        key.request_hash,
        key.lane.value,
        key.parent_id,
        key.node_id,
        key.call_id,
        key.sequence,
    )


def _exchange_digest(exchange: RecordedExchange) -> str:
    return _digest(
        {
            "key": exchange.key.model_dump(mode="json"),
            "request": exchange.request,
            "response": exchange.response,
            "safe_headers": exchange.safe_headers,
            "usage": exchange.usage,
            "elapsed_ms": exchange.elapsed_ms,
        }
    )


def _recording_sort_key(
    exchange: RecordedExchange,
) -> tuple[str, str, str, str, str, str, str, str, int]:
    key = exchange.key
    return (
        key.provider,
        key.operation,
        key.version,
        key.request_hash,
        key.lane.value,
        key.parent_id,
        key.node_id or "",
        key.call_id,
        key.sequence,
    )


_SAFE_HEADER_NAMES = frozenset(
    {"content-type", "etag", "request-id", "x-request-id", "retry-after"}
)
_TOOL_OUTCOME_ADAPTER: TypeAdapter[ToolOutcome] = TypeAdapter(ToolOutcome)
