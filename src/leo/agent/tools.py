"""The tool registry: schemas out, results back.

One rule governs this module, and it is the reason the previous harness died:

    **No tool outcome ever ends the run.**

A missing tool, bad arguments, a provider outage, a timeout, an oversized
payload -- every one of them comes back to the model as an ordinary ``tool``
message describing what went wrong. The model then adapts: it tries another
source, narrows the query, or answers with what it has. That is what a ReAct
loop is for. The old runtime instead treated several of these as terminal
gateway errors, which is how a working answer turned into "the reasoning
service stopped unexpectedly".
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from leo.agent.contracts import (
    Scope,
    Tool,
    ToolExecutionContext,
    ToolFailure,
    ToolSpec,
    ToolSuccess,
)

logger = logging.getLogger(__name__)

TRUNCATION_MARKER = "… [truncated]"


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, always renderable back to the model."""

    call_id: str
    name: str
    arguments: dict[str, Any]
    ok: bool
    payload: dict[str, Any]
    duration_ms: int
    source: str | None = None

    def as_message(self) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "name": self.name,
            "content": json.dumps(self.payload, ensure_ascii=False, default=str),
        }


class ToolRegistry:
    """Holds the executable tools and exposes them in the provider's schema shape."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            name = tool.spec.name
            if name in self._tools:
                logger.warning("duplicate tool name ignored: %s", name)
                continue
            self._tools[name] = tool

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._tools))

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(tool.spec for tool in self._tools.values())

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def add(self, tool: Tool) -> None:
        self._tools[tool.spec.name] = tool

    def schemas(self, names: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        selected = names if names is not None else self.names
        return [function_schema(self._tools[name].spec) for name in selected if name in self._tools]

    async def execute(
        self,
        *,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        scope: Scope,
        run_id: str,
        parse_error: str | None = None,
    ) -> ToolResult:
        started = time.monotonic()

        def finish(ok: bool, payload: dict[str, Any], source: str | None = None) -> ToolResult:
            return ToolResult(
                call_id=call_id,
                name=name,
                arguments=arguments,
                ok=ok,
                payload=payload,
                duration_ms=int((time.monotonic() - started) * 1000),
                source=source,
            )

        if parse_error is not None:
            return finish(False, {"error": "invalid_arguments", "message": parse_error})

        tool = self._tools.get(name)
        if tool is None:
            close = _closest_names(name, self.names)
            return finish(
                False,
                {
                    "error": "unknown_tool",
                    "message": f"No tool named {name!r} is available on this turn.",
                    "available_tools": list(close or self.names[:20]),
                },
            )

        try:
            validated = tool.validate(arguments)
        except (ValidationError, ValueError, TypeError) as exc:
            return finish(
                False,
                {
                    "error": "invalid_arguments",
                    "message": _safe_validation_message(exc),
                    "expected_schema": tool.spec.input_schema,
                },
            )

        context = ToolExecutionContext(
            trusted_scope=scope,
            run_id=run_id,
            tool_call_id=call_id,
        )
        try:
            outcome = await asyncio.wait_for(
                tool.execute(validated, context),
                timeout=tool.spec.timeout_seconds,
            )
        except TimeoutError:
            return finish(
                False,
                {
                    "error": "timeout",
                    "message": (
                        f"{name} did not respond within {tool.spec.timeout_seconds:.0f}s. "
                        "Try a narrower request or a different source."
                    ),
                },
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("tool %s raised", name)
            return finish(
                False,
                {
                    "error": "tool_crashed",
                    "message": f"{name} failed internally ({type(exc).__name__}).",
                },
            )

        if isinstance(outcome, ToolFailure):
            return finish(
                False,
                {
                    "error": outcome.code,
                    "message": outcome.safe_message,
                    "retryable": outcome.retryable,
                },
            )
        if not isinstance(outcome, ToolSuccess):
            return finish(
                False,
                {"error": "malformed_result", "message": f"{name} returned an unusable result."},
            )

        data = shrink(dict(outcome.data), tool.spec.max_result_bytes)
        payload: dict[str, Any] = {
            "source": outcome.source.provider,
            "reference": outcome.source.reference,
            "observed_at": outcome.observed_at.isoformat(),
            "data": data,
        }
        if outcome.source.url:
            payload["url"] = outcome.source.url
        return finish(True, payload, source=outcome.source.provider)


def function_schema(spec: ToolSpec) -> dict[str, Any]:
    """Render a ``ToolSpec`` as an OpenAI-compatible function definition."""

    schema = _sanitize_schema(spec.input_schema)
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description[:1024],
            "parameters": schema,
        },
    }


def _sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Guarantee a JSON-Schema object body, which every provider requires."""

    if not isinstance(schema, dict) or schema.get("type") != "object":
        return {"type": "object", "properties": {}, "additionalProperties": True}
    cleaned = dict(schema)
    cleaned.setdefault("properties", {})
    return cleaned


def shrink(data: dict[str, Any], max_bytes: int) -> dict[str, Any]:
    """Cut a payload down to ``max_bytes``, largest contributor first.

    Strings keep their leading text (where the answer usually is) and lists lose
    their tail. Partial evidence beats an empty result: the old runtime dropped
    oversize payloads entirely, so Leo would fetch a long filing and then report
    that it had found no source.
    """

    working = _jsonable(data)
    if not isinstance(working, dict):
        working = {"value": working}
    # Halve the biggest field each pass rather than cutting it by the exact
    # overshoot. Cutting by overshoot collapses one large field to nothing while
    # the rest of the payload stays intact; halving converges in log steps and
    # leaves every field represented by its opening -- which is where the answer
    # usually is.
    for _ in range(200):
        if _encoded_size(working) <= max_bytes:
            return working
        widest_key, widest_size = None, 0
        for key, value in working.items():
            size = _encoded_size(value)
            if size > widest_size:
                widest_key, widest_size = key, size
        if widest_key is None or widest_size <= len(TRUNCATION_MARKER) + 8:
            break
        value = working[widest_key]
        if isinstance(value, str):
            keep = max(1, len(value) // 2)
            working[widest_key] = value[:keep] + TRUNCATION_MARKER
        elif isinstance(value, list) and value:
            working[widest_key] = value[: len(value) // 2] if len(value) > 1 else []
        elif isinstance(value, dict) and value:
            working[widest_key] = shrink(value, max(32, widest_size // 2))
        else:
            working[widest_key] = TRUNCATION_MARKER
    if _encoded_size(working) > max_bytes:
        # Nothing sensible left to cut field by field; keep the payload valid.
        return {"_truncated": True, "_keys": sorted(working)[:50]}
    return working


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _encoded_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(value).encode("utf-8"))


def _safe_validation_message(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in error['loc']) or 'arguments'}: {error['msg']}"
            for error in exc.errors()[:5]
        ]
        return "; ".join(parts)
    return str(exc)[:300] or "the arguments were not valid for this tool"


def _closest_names(name: str, candidates: tuple[str, ...]) -> tuple[str, ...]:
    lowered = name.lower()
    stem = lowered.split(".")[-1]
    close = tuple(c for c in candidates if stem and (stem in c.lower() or c.lower() in lowered))
    return close[:10]
