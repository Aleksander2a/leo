"""The loop's guarantees.

The heading each test hangs under is the same one sentence: a turn ends with
something the model wrote, or with a truthful account of why it could not. The
old runtime broke that in half a dozen places, and each of those places has a
test here.
"""

from __future__ import annotations

import pytest

from leo.agent.contracts import ProviderError
from leo.agent.discovery import ToolDiscovery, ToolFinderTool
from leo.agent.llm import Completion, ToolCall, Usage
from leo.agent.loop import Agent, LoopLimits, _prune_oldest_exchange
from leo.agent.prompts import EMPTY_REPLY_NUDGE, FINAL_TURN_NUDGE
from leo.agent.store import Turn
from leo.agent.tools import ToolRegistry
from tests.conftest import SCOPE, FakeLLM, FakeTool, calls, failure, says


def build(llm: FakeLLM, tools: list, *, limits: LoopLimits | None = None) -> Agent:
    registry = ToolRegistry(tools)
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    finder = ToolFinderTool(discovery)
    registry.add(finder)
    return Agent(
        llm=llm,  # type: ignore[arg-type]
        registry=registry,
        discovery=discovery,
        finder=finder,
        limits=limits or LoopLimits(max_turns=4, max_tool_calls=6, max_seconds=30),
    )


@pytest.mark.asyncio
async def test_a_plain_question_is_answered_without_tools() -> None:
    llm = FakeLLM([says("Canberra.")])
    result = await build(llm, [FakeTool()]).run(question="Capital of Australia?", scope=SCOPE)
    assert result.answered
    assert result.answer == "Canberra."
    assert result.tool_calls == 0


@pytest.mark.asyncio
async def test_tool_result_flows_back_into_the_next_turn() -> None:
    tool = FakeTool()
    llm = FakeLLM([calls(("test.echo", {"value": "BTC"})), says("BTC looks fine.")])
    result = await build(llm, [tool]).run(question="How is BTC?", scope=SCOPE)
    assert result.answered
    assert tool.calls == [{"value": "BTC"}]
    # The second request carries the assistant tool-call turn *and* its result.
    second = llm.requests[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-1]["role"] == "tool"
    assert "BTC" in second[-1]["content"]


@pytest.mark.asyncio
async def test_a_failing_tool_does_not_end_the_run() -> None:
    tool = FakeTool(outcome=failure())
    llm = FakeLLM(
        [calls(("test.echo", {"value": "x"})), says("That source was down; here's what I know.")]
    )
    result = await build(llm, [tool]).run(question="anything", scope=SCOPE)
    assert result.answered
    assert "source was down" in result.answer


@pytest.mark.asyncio
async def test_parallel_calls_are_all_executed_and_returned() -> None:
    a, b = FakeTool("tool.a"), FakeTool("tool.b")
    llm = FakeLLM(
        [calls(("tool.a", {"value": "1"}), ("tool.b", {"value": "2"})), says("both done")]
    )
    result = await build(llm, [a, b]).run(question="q", scope=SCOPE)
    assert result.tool_calls == 2
    assert set(result.tools_used) == {"tool.a", "tool.b"}
    tool_messages = [m for m in llm.requests[1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2


@pytest.mark.asyncio
async def test_the_same_call_twice_is_answered_from_the_first_result() -> None:
    """A model that loops on one call must not burn the budget on it."""

    tool = FakeTool()
    llm = FakeLLM(
        [
            calls(("test.echo", {"value": "x"})),
            calls(("test.echo", {"value": "x"})),
            says("done"),
        ]
    )
    result = await build(llm, [tool]).run(question="q", scope=SCOPE)
    assert result.answered
    assert tool.calls == [{"value": "x"}]  # executed exactly once
    repeat = [m for m in llm.requests[2]["messages"] if m["role"] == "tool"][-1]
    assert "already made this exact call" in repeat["content"]


@pytest.mark.asyncio
async def test_exhausting_the_turn_budget_still_produces_a_model_answer() -> None:
    """The old runtime failed the run here. The model gets a last word instead."""

    llm = FakeLLM(
        [
            calls(("test.echo", {"value": "1"})),
            calls(("test.echo", {"value": "2"})),
            says("Here is what I found."),
        ]
    )
    agent = build(llm, [FakeTool()], limits=LoopLimits(max_turns=3, max_tool_calls=9))
    result = await agent.run(question="q", scope=SCOPE)
    assert result.answered
    assert result.answer == "Here is what I found."
    final = llm.requests[-1]
    assert final["tools"] is None  # tools withdrawn on the final turn
    assert final["messages"][-1]["content"] == FINAL_TURN_NUDGE


@pytest.mark.asyncio
async def test_exhausting_the_tool_budget_forces_the_final_turn() -> None:
    llm = FakeLLM([calls(("test.echo", {"value": "1"})), says("done with one source")])
    agent = build(llm, [FakeTool()], limits=LoopLimits(max_turns=8, max_tool_calls=1))
    result = await agent.run(question="q", scope=SCOPE)
    assert result.answered
    assert result.tool_calls == 1
    assert llm.requests[-1]["tools"] is None


@pytest.mark.asyncio
async def test_an_empty_reply_is_nudged_rather_than_dropped() -> None:
    llm = FakeLLM([says(""), says("Sorry — here it is.")])
    result = await build(llm, [FakeTool()]).run(question="q", scope=SCOPE)
    assert result.answered
    assert llm.requests[1]["messages"][-1]["content"] == EMPTY_REPLY_NUDGE


@pytest.mark.asyncio
async def test_a_provider_outage_is_reported_truthfully() -> None:
    """No canned apology, and no invented answer: say what actually broke."""

    class Broken(FakeLLM):
        async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise ProviderError("http_503", "upstream is down")

    result = await build(Broken([]), [FakeTool()]).run(question="q", scope=SCOPE)
    assert not result.answered
    assert result.status == "failed"
    assert "http_503" in (result.error or "")
    assert result.answer == ""


@pytest.mark.asyncio
async def test_history_and_memory_reach_the_prompt() -> None:
    llm = FakeLLM([says("ok")])
    await build(llm, [FakeTool()]).run(
        question="and now?",
        scope=SCOPE,
        history=[
            Turn(role="user", content="earlier question"),
            Turn(role="assistant", content="earlier answer"),
        ],
        memories="- [fact] holdings: 3 BTC",
    )
    messages = llm.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "3 BTC" in messages[0]["content"]
    assert [m["content"] for m in messages[1:]] == [
        "earlier question",
        "earlier answer",
        "and now?",
    ]


@pytest.mark.asyncio
async def test_content_alongside_tool_calls_is_preserved_for_the_provider() -> None:
    llm = FakeLLM([calls(("test.echo", {"value": "x"}), content="Let me check."), says("Done.")])
    await build(llm, [FakeTool()]).run(question="q", scope=SCOPE)
    assistant = next(m for m in llm.requests[1]["messages"] if m["role"] == "assistant")
    assert assistant["content"] == "Let me check."
    assert assistant["tool_calls"][0]["function"]["name"] == "test.echo"


@pytest.mark.asyncio
async def test_unparseable_tool_arguments_come_back_as_a_correction() -> None:
    llm = FakeLLM(
        [
            Completion(
                content="",
                tool_calls=(
                    ToolCall(id="c1", name="test.echo", arguments={}, parse_error="bad JSON"),
                ),
                finish_reason="tool_calls",
                usage=Usage(),
            ),
            says("recovered"),
        ]
    )
    result = await build(llm, [FakeTool()]).run(question="q", scope=SCOPE)
    assert result.answered
    assert (
        "bad JSON" in next(m for m in llm.requests[1]["messages"] if m["role"] == "tool")["content"]
    )


@pytest.mark.asyncio
async def test_tools_find_adds_a_tool_to_the_live_set() -> None:
    """Discovery is not a gate: a tool left off the roster stays reachable."""

    hidden = FakeTool("market.obscure_provider")
    registry = ToolRegistry([hidden])
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    finder = ToolFinderTool(discovery)
    registry.add(finder)
    llm = FakeLLM([calls(("tools.find", {"need": "obscure data"})), says("found it")])
    agent = Agent(
        llm=llm,  # type: ignore[arg-type]
        registry=registry,
        discovery=discovery,
        finder=finder,
        limits=LoopLimits(max_turns=3, max_tool_calls=4),
    )
    result = await agent.run(question="q", scope=SCOPE)
    assert result.answered
    offered = {t["function"]["name"] for t in (llm.requests[1]["tools"] or [])}
    assert "market.obscure_provider" in offered


@pytest.mark.asyncio
async def test_usage_accumulates_across_turns() -> None:
    llm = FakeLLM([calls(("test.echo", {"value": "x"})), says("done")])
    result = await build(llm, [FakeTool()]).run(question="q", scope=SCOPE)
    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_progress_callback_names_the_tools_being_called() -> None:
    seen: list[str] = []
    registry = ToolRegistry([FakeTool()])
    discovery = ToolDiscovery(registry=registry, llm=None, sessions=None)
    llm = FakeLLM([calls(("test.echo", {"value": "x"})), says("done")])
    agent = Agent(
        llm=llm,  # type: ignore[arg-type]
        registry=registry,
        discovery=discovery,
        limits=LoopLimits(max_turns=3),
        on_step=lambda names: _record(seen, names),
    )
    await agent.run(question="q", scope=SCOPE)
    assert seen == ["test.echo"]


async def _record(sink: list[str], names: str) -> None:
    sink.append(names)


def test_pruning_drops_an_assistant_call_with_all_of_its_results() -> None:
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "tool_calls": [{"id": "a"}, {"id": "b"}]},
        {"role": "tool", "tool_call_id": "a", "content": "1"},
        {"role": "tool", "tool_call_id": "b", "content": "2"},
        {"role": "assistant", "tool_calls": [{"id": "c"}]},
        {"role": "tool", "tool_call_id": "c", "content": "3"},
    ]
    pruned = _prune_oldest_exchange(messages)
    assert pruned is not None
    assert [m["role"] for m in pruned] == ["system", "user", "assistant", "tool"]
    assert pruned[-1]["tool_call_id"] == "c"


def test_pruning_reports_when_there_is_nothing_left_to_shed() -> None:
    assert _prune_oldest_exchange([{"role": "user", "content": "q"}]) is None
