"""The ReAct loop.

Reason, act, observe, repeat -- and then answer. That is the whole contract.

What is deliberately absent is as important as what is here. There is no
verifier that can reject the model's answer, no deliberation envelope that can
veto its chosen depth, no completion contract it must satisfy before it is
allowed to speak, and no keyword router that decides for it which tools the
question "really" needs. Every one of those existed in the previous runtime,
and every one of them could -- and did -- turn a good answer into a failed run.

The loop's only jobs are to keep the model supplied with context, to run the
tools it asks for, to keep both within budget, and to make sure the turn ends
with something the model actually wrote.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from leo.agent.contracts import ProviderError, Scope
from leo.agent.discovery import ToolDiscovery, ToolFinderTool
from leo.agent.llm import LLM, Completion, ToolCall, Usage
from leo.agent.prompts import EMPTY_REPLY_NUDGE, FINAL_TURN_NUDGE, system_prompt
from leo.agent.store import Turn
from leo.agent.tools import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

StepCallback = Callable[[str], Awaitable[None]]


@dataclass
class AgentResult:
    answer: str
    status: str
    turns: int = 0
    tool_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None
    tools_used: tuple[str, ...] = ()
    steps: list[ToolResult] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.status == "answered"


@dataclass(frozen=True)
class LoopLimits:
    max_turns: int = 10
    max_tool_calls: int = 20
    max_seconds: float = 300.0
    max_parallel_tools: int = 6


class Agent:
    """One conversational turn, from question to answer."""

    def __init__(
        self,
        *,
        llm: LLM,
        registry: ToolRegistry,
        discovery: ToolDiscovery,
        finder: ToolFinderTool | None = None,
        limits: LoopLimits | None = None,
        on_step: StepCallback | None = None,
    ) -> None:
        self._llm = llm
        self._registry = registry
        self._discovery = discovery
        self._finder = finder
        self._limits = limits or LoopLimits()
        self._on_step = on_step

    async def run(
        self,
        *,
        question: str,
        scope: Scope,
        history: list[Turn] | None = None,
        memories: str = "",
        run_id: str = "",
        scope_description: str = "a Slack conversation",
        extra_instructions: str = "",
        on_model_step: Callable[[int, Completion, int], Awaitable[None]] | None = None,
        on_tool_step: Callable[[int, ToolResult], Awaitable[None]] | None = None,
    ) -> AgentResult:
        started = time.monotonic()
        deadline = started + self._limits.max_seconds
        # Tools require a real correlation id. A caller that does not persist
        # runs (tests, a one-shot script) still gets one rather than a crash on
        # the first tool call.
        run_id = run_id or f"run-{uuid.uuid4()}"

        await self._discovery.prepare()
        active = list(await self._discovery.select(question))
        if self._finder is not None:
            self._finder.discovered.clear()

        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt(
                    now=datetime.now(UTC),
                    scope_description=scope_description,
                    memories=memories,
                    extra=extra_instructions,
                ),
            }
        ]
        for turn in history or []:
            messages.append({"role": turn.role, "content": turn.content})
        messages.append({"role": "user", "content": question})

        usage = Usage()
        tool_calls_made = 0
        tools_used: list[str] = []
        results: list[ToolResult] = []
        seq = 0
        #: Exact (tool, arguments) pairs already executed this run, so a model that
        #: loops on the same call gets its previous result back instead of burning
        #: the budget on it. Structural, and invisible unless it actually repeats.
        seen: dict[str, ToolResult] = {}

        for turn_index in range(self._limits.max_turns):
            out_of_time = time.monotonic() >= deadline
            out_of_tools = tool_calls_made >= self._limits.max_tool_calls
            last_turn = turn_index == self._limits.max_turns - 1
            final = last_turn or out_of_time or out_of_tools

            if final:
                messages.append({"role": "system", "content": FINAL_TURN_NUDGE})

            schemas = None if final else self._schemas(active)
            call_started = time.monotonic()
            try:
                completion = await self._complete(messages, schemas)
            except ProviderError as exc:
                logger.error("model provider failed: %s %s", exc.code, exc.message)
                return AgentResult(
                    answer="",
                    status="failed",
                    turns=turn_index,
                    tool_calls=tool_calls_made,
                    usage=usage,
                    error=f"{exc.code}: {exc.message}"[:500],
                    tools_used=tuple(dict.fromkeys(tools_used)),
                    steps=results,
                )
            usage = usage + completion.usage
            seq += 1
            if on_model_step is not None:
                await on_model_step(seq, completion, int((time.monotonic() - call_started) * 1000))

            if completion.tool_calls and not final:
                messages.append(_assistant_tool_message(completion))
                batch = list(completion.tool_calls)[: self._limits.max_parallel_tools]
                remaining = self._limits.max_tool_calls - tool_calls_made
                batch = batch[: max(1, remaining)]
                await self._announce(batch)
                executed = await self._execute(batch, scope, run_id, seen)
                for result in executed:
                    tool_calls_made += 1
                    tools_used.append(result.name)
                    results.append(result)
                    seq += 1
                    if on_tool_step is not None:
                        await on_tool_step(seq, result)
                    messages.append(result.as_message())
                if self._finder is not None and self._finder.discovered:
                    for name in sorted(self._finder.discovered):
                        if name not in active and self._registry.get(name) is not None:
                            active.append(name)
                    self._finder.discovered.clear()
                continue

            answer = completion.content.strip()
            if answer:
                return AgentResult(
                    answer=answer,
                    status="answered",
                    turns=turn_index + 1,
                    tool_calls=tool_calls_made,
                    usage=usage,
                    tools_used=tuple(dict.fromkeys(tools_used)),
                    steps=results,
                )

            # Empty content and no usable tool calls. Say so and go round again;
            # on the last turn there is nowhere left to go.
            if final:
                return AgentResult(
                    answer="",
                    status="failed",
                    turns=turn_index + 1,
                    tool_calls=tool_calls_made,
                    usage=usage,
                    error="the model returned an empty final answer",
                    tools_used=tuple(dict.fromkeys(tools_used)),
                    steps=results,
                )
            messages.append({"role": "system", "content": EMPTY_REPLY_NUDGE})

        return AgentResult(
            answer="",
            status="failed",
            turns=self._limits.max_turns,
            tool_calls=tool_calls_made,
            usage=usage,
            error="the loop ended without an answer",
            tools_used=tuple(dict.fromkeys(tools_used)),
            steps=results,
        )

    # -- internals --------------------------------------------------------

    def _schemas(self, active: list[str]) -> list[dict[str, Any]]:
        return self._registry.schemas(tuple(dict.fromkeys(active)))

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        schemas: list[dict[str, Any]] | None,
    ) -> Completion:
        """Call the model, shedding context if the window is the thing that broke."""

        try:
            return await self._llm.complete(messages, tools=schemas)
        except ProviderError as exc:
            if not _is_context_overflow(exc):
                raise
            pruned = _prune_oldest_exchange(messages)
            if pruned is None:
                raise
            logger.warning("context overflow; retrying with the oldest tool exchange dropped")
            messages[:] = pruned
            return await self._llm.complete(messages, tools=schemas)

    async def _announce(self, batch: list[ToolCall]) -> None:
        if self._on_step is None or not batch:
            return
        names = ", ".join(dict.fromkeys(call.name for call in batch))
        try:
            await self._on_step(names)
        except Exception:
            logger.debug("progress callback failed", exc_info=True)

    async def _execute(
        self,
        batch: list[ToolCall],
        scope: Scope,
        run_id: str,
        seen: dict[str, ToolResult],
    ) -> list[ToolResult]:
        async def one(call: ToolCall) -> ToolResult:
            key = _call_key(call)
            previous = seen.get(key)
            if previous is not None:
                repeated = dict(previous.payload)
                repeated["note"] = (
                    "You already made this exact call on an earlier turn. This is the "
                    "same result. Use it, or try a different tool or different arguments."
                )
                return ToolResult(
                    call_id=call.id,
                    name=call.name,
                    arguments=call.arguments,
                    ok=previous.ok,
                    payload=repeated,
                    duration_ms=0,
                    source=previous.source,
                )
            result = await self._registry.execute(
                call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                scope=scope,
                run_id=run_id,
                parse_error=call.parse_error,
            )
            seen[key] = result
            return result

        return list(await asyncio.gather(*(one(call) for call in batch)))


def _assistant_tool_message(completion: Completion) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": completion.content or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False, default=str),
                },
            }
            for call in completion.tool_calls
        ],
    }


def _call_key(call: ToolCall) -> str:
    return f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"


def _is_context_overflow(exc: ProviderError) -> bool:
    haystack = f"{exc.code} {exc.message}".lower()
    return any(
        marker in haystack
        for marker in ("context length", "context_length", "too many tokens", "maximum context")
    )


def _prune_oldest_exchange(messages: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    """Drop the oldest assistant-tool-call block and every result it owns.

    Removing a tool message without its assistant call (or the reverse) produces
    a request the provider rejects outright, so the pair is always dropped
    together.
    """

    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        owned = {
            call.get("id")
            for call in message["tool_calls"]
            if isinstance(call, dict) and call.get("id")
        }
        return [
            item
            for position, item in enumerate(messages)
            if position != index
            and not (item.get("role") == "tool" and item.get("tool_call_id") in owned)
        ]
    return None
