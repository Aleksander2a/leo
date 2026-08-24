"""Shared fixtures and test doubles for the agent suite."""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from leo.agent.contracts import (
    Scope,
    SourceRef,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.agent.llm import Completion, ToolCall, Usage

SCOPE = Scope(key="test:scope:alpha", actor_id="tester")
OTHER_SCOPE = Scope(key="test:scope:beta", actor_id="tester")


@pytest.fixture
def scope() -> Scope:
    return SCOPE


class FakeTool:
    """A tool whose behaviour each test dictates."""

    def __init__(
        self,
        name: str = "test.echo",
        *,
        outcome: ToolOutcome | None = None,
        raises: Exception | None = None,
        delay: float = 0.0,
        timeout_seconds: float = 5.0,
        max_result_bytes: int = 8192,
        schema: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> None:
        self._name = name
        self._description = description or f"Test tool {name}"
        self._outcome = outcome
        self._raises = raises
        self._delay = delay
        self._timeout = timeout_seconds
        self._max_bytes = max_result_bytes
        self._schema = schema or {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        self.calls: list[dict[str, Any]] = []

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self._name,
            description=self._description,
            domain="test",
            input_schema=self._schema,
            timeout_seconds=self._timeout,
            max_result_bytes=self._max_bytes,
        )

    def validate(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if "value" in self._schema.get("required", []) and "value" not in arguments:
            raise ValueError("value is required")
        return dict(arguments)

    async def execute(
        self, arguments: dict[str, Any], context: ToolExecutionContext
    ) -> ToolOutcome:
        self.calls.append(dict(arguments))
        if self._delay:
            import asyncio

            await asyncio.sleep(self._delay)
        if self._raises is not None:
            raise self._raises
        if self._outcome is not None:
            return self._outcome
        return ToolSuccess(
            data={"echo": arguments.get("value", "")},
            source=SourceRef(provider="test", reference=f"echo:{arguments.get('value', '')}"),
            observed_at=datetime.now(UTC),
        )


def failure(code: str = "PROVIDER_DOWN", message: str = "provider unavailable") -> ToolFailure:
    return ToolFailure(code=code, safe_message=message, retryable=True)


class FakeLLM:
    """Replays a scripted sequence of completions and records every request."""

    def __init__(self, script: list[Completion], *, embedding: list[float] | None = None) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.model = "fake/model"
        self.embedding_model = "fake/embed"
        self._embedding = embedding

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Completion:
        self.requests.append({"messages": list(messages), "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM ran out of scripted completions")
        return self.script.pop(0)

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        return [self._embedding for _ in texts]


def says(text: str) -> Completion:
    return Completion(content=text, usage=Usage(prompt_tokens=10, completion_tokens=5))


def calls(*specs: tuple[str, dict[str, Any]], content: str = "") -> Completion:
    return Completion(
        content=content,
        tool_calls=tuple(
            ToolCall(id=f"call-{index}", name=name, arguments=arguments)
            for index, (name, arguments) in enumerate(specs)
        ),
        finish_reason="tool_calls",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )


def database_url() -> str | None:
    return os.environ.get("DATABASE_URL") or _dotenv_database_url()


def _dotenv_database_url() -> str | None:
    try:
        from leo.config import Settings

        settings = Settings()
    except Exception:
        return None
    if settings.database_url is None:
        return None
    return settings.database_url.get_secret_value()


requires_database = pytest.mark.skipif(
    database_url() is None, reason="DATABASE_URL is not configured"
)


if sys.platform == "win32":
    # psycopg's async mode refuses Windows' default Proactor loop, and the
    # database-backed tests would otherwise fail on connect rather than on
    # anything they are actually asserting.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


@pytest.fixture
def frozen_now() -> Iterator[datetime]:
    yield datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
