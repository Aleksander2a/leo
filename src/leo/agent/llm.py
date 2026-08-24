"""OpenAI-compatible chat + embedding client (OpenRouter).

Deliberately thin. It speaks the wire format the model was trained on --
``messages`` in, ``content`` and ``tool_calls`` out -- and does nothing else.
There is no response schema constraining what the model may say, because the
previous harness's constrained-JSON completion contract is exactly what made
the model unable to answer: it had to satisfy a claim/citation shape before it
was allowed to speak, and when it could not, the run died with no answer.

The only intelligence here is transport resilience: retry the retryable, and
turn a malformed tool-call argument blob into something the *model* can see and
correct on the next turn, rather than a fatal error.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any

import httpx

from leo.agent.contracts import ProviderError

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524})


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    #: Set when the provider emitted arguments that were not valid JSON. The
    #: loop surfaces this back to the model as a tool error so it can retry with
    #: well-formed arguments; it is never a run-ending condition.
    parse_error: str | None = None


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cost=self.cost + other.cost,
        )


@dataclass(frozen=True)
class Completion:
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    request_id: str = ""

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLM:
    """Chat completions and embeddings against an OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_output_tokens: int = 4000,
        timeout_seconds: float = 180.0,
        max_attempts: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required")
        if not model:
            raise ValueError("a model id is required")
        self._client = client
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._embedding_model = embedding_model
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts

    @property
    def model(self) -> str:
        return self._model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.3,
        max_tokens: int | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or self._max_output_tokens,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        body = await self._post("/chat/completions", payload)
        return _parse_completion(body)

    async def embed(self, texts: list[str]) -> list[list[float] | None]:
        """Embed a batch. Returns a same-length list; ``None`` for anything that failed.

        Embeddings drive tool discovery and memory recall. Both degrade to a
        usable default when a vector is missing, so an embedding outage must
        never raise into the loop.
        """

        if not texts:
            return []
        try:
            body = await self._post(
                "/embeddings",
                {"model": self._embedding_model, "input": texts},
                retries=2,
            )
        except (ProviderError, httpx.HTTPError):
            logger.warning("embedding request failed; continuing without vectors")
            return [None] * len(texts)
        by_index: dict[int, list[float]] = {}
        for item in body.get("data") or []:
            if not isinstance(item, dict):
                continue
            vector = item.get("embedding")
            index = item.get("index")
            if isinstance(vector, list) and isinstance(index, int):
                by_index[index] = [float(v) for v in vector]
        return [by_index.get(i) for i in range(len(texts))]

    async def _post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        retries: int | None = None,
    ) -> dict[str, Any]:
        attempts = retries if retries is not None else self._max_attempts
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    f"{self._base_url}{path}",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self._timeout,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last = exc
                await self._backoff(attempt)
                continue
            if response.status_code in _RETRYABLE_STATUS and attempt < attempts - 1:
                last = ProviderError(f"http_{response.status_code}", response.text[:400])
                await self._backoff(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                raise ProviderError(f"http_{response.status_code}", response.text[:400])
            try:
                body = response.json()
            except ValueError as exc:
                last = ProviderError("invalid_json", str(exc))
                await self._backoff(attempt)
                continue
            if not isinstance(body, dict):
                last = ProviderError("invalid_body", "provider returned a non-object body")
                await self._backoff(attempt)
                continue
            # OpenRouter can return HTTP 200 with an error object in the body.
            error = body.get("error")
            if isinstance(error, dict):
                code = str(error.get("code") or "provider_error")
                message = str(error.get("message") or "")[:400]
                if attempt < attempts - 1 and _looks_retryable(code, message):
                    last = ProviderError(code, message)
                    await self._backoff(attempt)
                    continue
                raise ProviderError(code, message)
            return body
        raise ProviderError(
            getattr(last, "code", "provider_unreachable"),
            str(last) if last else "the model provider could not be reached",
        )

    async def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        if retry_after:
            try:
                await asyncio.sleep(min(30.0, float(retry_after)))
                return
            except ValueError:
                pass
        await asyncio.sleep(min(20.0, (2**attempt) + random.random()))


def _looks_retryable(code: str, message: str) -> bool:
    haystack = f"{code} {message}".lower()
    return any(
        marker in haystack
        for marker in ("rate", "overload", "timeout", "temporar", "unavailable", "capacity", "502")
    )


def _parse_completion(body: dict[str, Any]) -> Completion:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderError("no_choices", "the model returned no choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise ProviderError("no_choices", "the model returned a malformed choice")
    message = choice.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    text = content.strip() if isinstance(content, str) else ""
    calls: list[ToolCall] = []
    raw_calls = message.get("tool_calls")
    if isinstance(raw_calls, list):
        for index, raw in enumerate(raw_calls):
            parsed = _parse_tool_call(raw, index)
            if parsed is not None:
                calls.append(parsed)
    usage_body = body.get("usage")
    usage_body = usage_body if isinstance(usage_body, dict) else {}
    usage = Usage(
        prompt_tokens=_as_int(usage_body.get("prompt_tokens")),
        completion_tokens=_as_int(usage_body.get("completion_tokens")),
        total_tokens=_as_int(usage_body.get("total_tokens")),
        cost=_as_float(usage_body.get("cost")),
    )
    return Completion(
        content=text,
        tool_calls=tuple(calls),
        finish_reason=str(choice.get("finish_reason") or "stop"),
        usage=usage,
        model=str(body.get("model") or ""),
        request_id=str(body.get("id") or ""),
    )


def _parse_tool_call(raw: object, index: int) -> ToolCall | None:
    if not isinstance(raw, dict):
        return None
    function = raw.get("function")
    function = function if isinstance(function, dict) else {}
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    call_id = raw.get("id")
    call_id = call_id if isinstance(call_id, str) and call_id else f"call_{index}"
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments=dict(arguments))
    if not isinstance(arguments, str) or not arguments.strip():
        return ToolCall(id=call_id, name=name, arguments={})
    try:
        decoded = json.loads(arguments)
    except ValueError:
        decoded = _repair_json(arguments)
    if isinstance(decoded, dict):
        return ToolCall(id=call_id, name=name, arguments=decoded)
    return ToolCall(
        id=call_id,
        name=name,
        arguments={},
        parse_error=(
            "Your tool-call arguments were not a valid JSON object. "
            "Call the tool again with a JSON object matching its schema."
        ),
    )


def _repair_json(text: str) -> object:
    """Recover the common provider mangles of a JSON argument blob."""

    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        _, _, candidate = candidate.partition("\n")
    start, end = candidate.find("{"), candidate.rfind("}")
    if start != -1 and end > start:
        candidate = candidate[start : end + 1]
    for attempt in (candidate, candidate.replace("'", '"')):
        try:
            return json.loads(attempt)
        except ValueError:
            continue
    return None


def _as_int(value: object) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _as_float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
