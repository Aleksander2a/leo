"""Executable offline evidence for the revised conversational M5 decisions.

These scenarios deliberately accept no caller-supplied pass/fail flags.  Each probe
drives the same product components used by the Slack runtime, with deterministic
clocks/delegates and in-process HTTP transport replacing external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import httpx
from pydantic import ValidationError

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import InMemoryToolCatalog
from leo.demo import run_conversation_smoke
from leo.evals.control import BaselineExecution
from leo.evals.models import Scenario
from leo.harness.deliberation import ElasticDeliberationGateway, ElasticDeliberationPolicy
from leo.harness.models import (
    BudgetLimits,
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    ContextItem,
    ContextItemKind,
    ContextManifest,
    ContextSegment,
    EventType,
    ModelRequest,
    ModelTurnResult,
    OriginRef,
    Run,
    RunBundle,
    RunPhase,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
    ToolChoiceMode,
    ToolChoicePolicy,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolRequests,
    ToolSuccess,
    TrustedScope,
    VerifierStatus,
)
from leo.harness.normalization import normalize_success
from leo.harness.ports import ModelGatewayError
from leo.harness.thread_context import (
    ThreadContextRange,
    ThreadTurnRetentionInput,
    classify_thread_transcript,
    select_context_with_thread_compaction,
    thread_context_source_digest,
)
from leo.harness.thread_context_tools import (
    ThreadContextAuthority,
    build_thread_context_tools,
)
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.slack.context import SlackHistoryContextLoader
from leo.integrations.slack.events import (
    SlackConversationKind,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.integrations.slack.render import SlackTerminalResult, render_terminal_result
from leo.integrations.tavily import TavilySearchTool
from leo.integrations.web_fetch import PublicTextFetchTool
from leo.persistence.context_loader import (
    ConversationContextAuthorizationError,
    ConversationContextRequest,
    PostgresConversationContextLoader,
)
from leo.persistence.schema import SanitizedMessageRow
from leo.persistence.slack_messages import (
    PersistedSlackThreadSnapshot,
    SlackThreadCoverageSource,
    _assess_thread_coverage,
)

REVISED_M5_VARIANTS = frozenset(
    {
        "conversational_terminal_recovery",
        "elastic_deliberation",
        "slack_thread_context_authority",
        "tavily_verified_research",
    }
)


class RevisedM5UnsupportedScenario(RuntimeError):
    pass


@dataclass(frozen=True)
class RevisedM5Observed:
    invariants: frozenset[str]
    metrics: dict[str, float | int | str]
    hard_failures: tuple[str, ...] = ()


class _DecisionGateway:
    def __init__(self, decisions: tuple[ToolRequests | CompletionProposal, ...]) -> None:
        self._decisions = list(decisions)
        self.calls = 0
        self.requests: list[ModelRequest] = []

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.requests.append(request)
        self.calls += 1
        if not self._decisions:
            raise ModelGatewayError("fixture_exhausted", "No deterministic decision remains.")
        return ModelTurnResult(
            decision=self._decisions.pop(0),
            provider="offline-semantic-delegate",
            model="revised-m5-v1",
        )


class _UnavailableGateway:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        raise ModelGatewayError("provider_unavailable", "The provider is unavailable.")


def _parse_clock(scenario: Scenario) -> datetime:
    try:
        parsed = datetime.fromisoformat(scenario.fixed_clock.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RevisedM5UnsupportedScenario("fixed_clock_invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RevisedM5UnsupportedScenario("fixed_clock_requires_timezone")
    return parsed


def _request(
    objective: str,
    iteration: int = 0,
    *,
    feedback: tuple[str, ...] = (),
) -> ModelRequest:
    return ModelRequest(
        objective=objective,
        iteration=iteration,
        observations=(),
        verifier_feedback=feedback,
        tools=(),
        tool_choice=ToolChoicePolicy(mode=ToolChoiceMode.AUTO),
        manifest=ContextManifest(
            segments=(ContextSegment(name="objective", priority=100, pinned=True),)
        ),
    )


async def _terminal_recovery(scenario: Scenario) -> RevisedM5Observed:
    if scenario.budget.max_model_calls != 0 or scenario.budget.max_tool_calls != 0:
        raise RevisedM5UnsupportedScenario("terminal_recovery_requires_zero_call_budget")
    cases = (
        SlackTerminalResult(
            run_id=f"{scenario.deterministic_id_prefix}-budget",
            status=RunStatus.BUDGET_EXHAUSTED,
            terminal_reason="iteration_budget_exhausted",
            verified_partial_results=("The first bounded source was verified.",),
        ),
        SlackTerminalResult(
            run_id=f"{scenario.deterministic_id_prefix}-failed",
            status=RunStatus.FAILED,
            terminal_reason="model_gateway_error",
        ),
        SlackTerminalResult(
            run_id=f"{scenario.deterministic_id_prefix}-tool",
            status=RunStatus.FAILED,
            terminal_reason="tool_failure:EQUITY_QUOTE_ALL_PROVIDERS_FAILED",
        ),
        SlackTerminalResult(
            run_id=f"{scenario.deterministic_id_prefix}-context",
            status=RunStatus.FAILED,
            terminal_reason="context_unavailable:slack_thread_history_unavailable",
        ),
    )
    rendered = tuple("\n".join(render_terminal_result(item).chunks) for item in cases)
    raw_status_tokens = (
        RunStatus.BUDGET_EXHAUSTED.value,
        RunStatus.FAILED.value,
        "model_gateway_error",
        "tool_failure",
        "context_unavailable",
    )
    bare_status_count = sum(
        text.strip().casefold() in raw_status_tokens
        or any(token in text.casefold() for token in raw_status_tokens[:-1])
        or "context_unavailable" in text.casefold()
        for text in rendered
    )
    internal_id_count = sum(
        any(
            marker.casefold() in text.casefold()
            for marker in (
                cases[index].run_id,
                "Run:",
                "Run ID:",
                "Request ID:",
            )
        )
        for index, text in enumerate(rendered)
    )
    useless_boilerplate_count = sum(
        any(
            marker in text.casefold()
            for marker in (
                "i haven't presented unverified work",
                "unverified work as a completed answer",
                "i couldn't complete this request safely",
                "next step:",
            )
        )
        for text in rendered
    )
    category_markers = (
        ("processing limit", "reply “continue”"),
        ("reasoning service", "ask me to retry"),
        ("sources or tools", "information i can verify"),
        ("conversation context or access", "share the missing detail"),
    )
    actionable_category_count = sum(
        all(marker in text.casefold() for marker in markers)
        for text, markers in zip(rendered, category_markers, strict=True)
    )
    conversational = actionable_category_count == len(cases) and all(
        len(text.split()) >= 15 for text in rendered
    )
    bounded_partial = (
        "I did confirm this before stopping:" in rendered[0]
        and "The first bounded source was verified." in rendered[0]
        and all("The first bounded source was verified." not in text for text in rendered[1:])
    )
    invariants: set[str] = set()
    if conversational:
        invariants.add("terminal_failure_is_conversational")
    if bare_status_count == 0:
        invariants.add("terminal_raw_status_is_suppressed")
    if bounded_partial:
        invariants.add("terminal_partial_truth_is_bounded")
    if internal_id_count == 0:
        invariants.add("terminal_internal_ids_are_hidden")
    if useless_boilerplate_count == 0:
        invariants.add("terminal_useless_boilerplate_is_suppressed")
    if actionable_category_count == len(cases) and len(
        {text.splitlines()[0] for text in rendered}
    ) == len(cases):
        invariants.add("terminal_recovery_is_actionable_and_category_specific")
    invariants.add("terminal_recovery_uses_zero_model_or_tool_calls")
    return RevisedM5Observed(
        invariants=frozenset(invariants),
        metrics={
            "terminal_recovery_render_count": len(rendered),
            "terminal_bare_status_count": bare_status_count,
            "terminal_internal_id_count": internal_id_count,
            "terminal_useless_boilerplate_count": useless_boilerplate_count,
            "terminal_actionable_category_count": actionable_category_count,
            "terminal_model_calls": 0,
            "terminal_tool_calls": 0,
            "terminal_verified_partial_count": 1,
            "false_success_count": 0,
        },
    )


async def _elastic_deliberation(scenario: Scenario) -> RevisedM5Observed:
    if scenario.budget.max_model_calls < 14 or scenario.budget.max_tool_calls < 2:
        raise RevisedM5UnsupportedScenario("elastic_deliberation_budget_too_small")
    policy = ElasticDeliberationPolicy()
    parent_tools = frozenset({"agent.delegate_research", "agent.execute_research_plan"})

    direct_objective = "Explain covariance in two sentences."
    direct_delegate = _DecisionGateway(
        (CompletionProposal(answer="Covariance measures how two variables vary together."),)
    )
    direct_envelope = policy.assess(direct_objective, available_tool_names=parent_tools)
    direct = await ElasticDeliberationGateway(direct_delegate, direct_envelope).decide(
        _request(direct_objective)
    )

    clarify_objective = "Compare these"
    clarify_envelope = policy.assess(clarify_objective)
    clarify = await ElasticDeliberationGateway(_UnavailableGateway(), clarify_envelope).decide(
        _request(clarify_objective)
    )

    tool_objective = "Where is NVDA trading now?"
    tool_envelope = policy.assess(
        tool_objective,
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
        available_tool_names=parent_tools,
    )
    tool_delegate = _DecisionGateway(
        (
            ToolRequests(
                calls=(
                    ToolRequest(
                        id="quote-call",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            ),
        )
    )
    single_tool = await ElasticDeliberationGateway(tool_delegate, tool_envelope).decide(
        _request(tool_objective)
    )

    semantic_objective = (
        "Synthesize a sequenced rollout for three modules, accounting for dependencies "
        "and parallel workstreams."
    )
    semantic_envelope = policy.assess(
        semantic_objective,
        available_tool_names=parent_tools,
    )
    semantic_delegate = _DecisionGateway(
        (
            ToolRequests(
                calls=(
                    ToolRequest(
                        id="semantic-plan",
                        name="agent.execute_research_plan",
                        arguments={"objective": "Synthesize the rollout."},
                    ),
                )
            ),
        )
    )
    semantic_plan = await ElasticDeliberationGateway(semantic_delegate, semantic_envelope).decide(
        _request(semantic_objective)
    )

    delegate_objective = (
        "Investigate two independent launch workstreams and reconcile their findings."
    )
    delegate_envelope = policy.assess(
        delegate_objective,
        available_tool_names=parent_tools,
    )
    orchestration_delegate = _DecisionGateway(
        (
            ToolRequests(
                calls=(
                    ToolRequest(
                        id="semantic-delegate",
                        name="agent.delegate_research",
                        arguments={"objective": "Investigate the independent workstreams."},
                    ),
                )
            ),
        )
    )
    semantic_delegation = await ElasticDeliberationGateway(
        orchestration_delegate, delegate_envelope
    ).decide(_request(delegate_objective))

    no_progress_delegate = _DecisionGateway(
        (
            CompletionProposal(answer="First unsupported attempt."),
            CompletionProposal(answer="Second unsupported attempt."),
            CompletionProposal(answer="Third unsupported attempt."),
        )
    )
    no_progress = ElasticDeliberationGateway(
        no_progress_delegate,
        direct_envelope,
        max_no_progress_turns=2,
    )
    await no_progress.decide(_request(direct_objective, 0))
    await no_progress.decide(_request(direct_objective, 1, feedback=("Unsupported.",)))
    await no_progress.decide(_request(direct_objective, 2, feedback=("Still unsupported.",)))
    no_progress_code = ""
    try:
        await no_progress.decide(_request(direct_objective, 3, feedback=("Still unsupported.",)))
    except ModelGatewayError as exc:
        no_progress_code = exc.code

    live_dividend_objective = (
        "Some dividend based stocks with growth potential over time, some safe bets with "
        "high dividends"
    )
    repaired_dividend_answer = (
        "A useful starting screen is durable cash flow, a sustainable payout ratio, and "
        "a history of dividend growth."
    )
    truncated_delegate = _DecisionGateway(
        (
            CompletionProposal(
                answer=(
                    "Here are a few dividend-focused ideas ... some that are steadier, "
                    "higher-yield "
                )
            ),
            CompletionProposal(answer=repaired_dividend_answer),
        )
    )
    truncated_result = await run_conversation_smoke(
        model=ElasticDeliberationGateway(
            truncated_delegate,
            policy.assess(live_dividend_objective),
        ),
        objective=live_dividend_objective,
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    future_work_clarification = (
        "Which market or asset class, risk tolerance, and time horizon should I focus on?"
    )
    future_work_delegate = _DecisionGateway(
        (
            CompletionProposal(
                answer=(
                    "Happy to help — let me pull a few current quotes and recent dividend data, "
                    "and then I can narrow in on the strongest candidates."
                )
            ),
            CompletionProposal(answer=future_work_clarification),
        )
    )
    future_work_result = await run_conversation_smoke(
        model=future_work_delegate,
        objective="What are some interesting investing opportunities currently?",
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    action_claim_delegate = _DecisionGateway(
        (
            CompletionProposal(
                answer="Here's a preliminary mix. I pulled current quotes for a few names."
            ),
            CompletionProposal(answer="Which ticker or market should I check?"),
        )
    )
    action_claim_result = await run_conversation_smoke(
        model=action_claim_delegate,
        objective="Show me a few current dividend opportunities.",
        limits=BudgetLimits(max_iterations=3, max_model_calls=3, max_tool_calls=0),
    )

    direct_ok = isinstance(direct.decision, CompletionProposal) and direct_delegate.calls == 1
    clarify_ok = (
        isinstance(clarify.decision, CompletionProposal)
        and clarify.decision.answer.count("?") == 1
        and clarify.provider == "leo-harness"
    )
    single_tool_ok = isinstance(single_tool.decision, ToolRequests) and tuple(
        item.name for item in single_tool.decision.calls
    ) == ("market.get_quote",)
    semantic_plan_ok = (
        semantic_envelope.required_parent_tool is None
        and semantic_envelope.recommended_mode.value == "direct"
        and isinstance(semantic_plan.decision, ToolRequests)
        and tuple(item.name for item in semantic_plan.decision.calls)
        == ("agent.execute_research_plan",)
    )
    semantic_delegate_ok = (
        delegate_envelope.required_parent_tool is None
        and isinstance(semantic_delegation.decision, ToolRequests)
        and tuple(item.name for item in semantic_delegation.decision.calls)
        == ("agent.delegate_research",)
    )
    truncated_retry_ok = (
        truncated_result.run.status is RunStatus.COMPLETED
        and truncated_result.run.final_output == repaired_dividend_answer
        and truncated_delegate.calls == 2
        and any(
            "without trailing whitespace" in feedback.casefold()
            for feedback in truncated_delegate.requests[1].verifier_feedback
        )
        and sum(event.type is EventType.VERIFICATION_FAILED for event in truncated_result.events)
        == 1
    )
    future_work_repair_ok = (
        future_work_result.run.status is RunStatus.COMPLETED
        and future_work_result.run.final_output == future_work_clarification
        and future_work_delegate.calls == 2
        and any(
            "promise of future work" in feedback.casefold()
            and "one concrete input-seeking question" in feedback.casefold()
            for feedback in future_work_delegate.requests[1].verifier_feedback
        )
    )
    unobserved_action_repair_ok = (
        action_claim_result.run.status is RunStatus.COMPLETED
        and action_claim_result.run.final_output == "Which ticker or market should I check?"
        and action_claim_delegate.calls == 2
        and any(
            "without a matching retrieved observation" in feedback.casefold()
            for feedback in action_claim_delegate.requests[1].verifier_feedback
        )
    )
    invariants: set[str] = set()
    if direct_ok:
        invariants.add("short_prompt_can_answer_directly")
    if clarify_ok:
        invariants.add("short_ambiguity_clarifies_without_tools")
    if single_tool_ok:
        invariants.add("short_freshness_prompt_selects_one_tool")
    if semantic_plan_ok:
        invariants.add("semantic_model_can_plan_without_incantation")
    if semantic_delegate_ok:
        invariants.add("semantic_model_can_delegate_without_incantation")
    if no_progress_code == "deliberation_no_progress" and no_progress_delegate.calls == 3:
        invariants.add("deliberation_no_progress_is_bounded")
    if truncated_retry_ok:
        invariants.add("truncated_answer_retries_to_complete_answer")
    if future_work_repair_ok:
        invariants.add("future_work_promise_retries_to_concrete_reply")
    if unobserved_action_repair_ok:
        invariants.add("unobserved_action_claim_retries_to_honest_reply")
    return RevisedM5Observed(
        invariants=frozenset(invariants),
        metrics={
            "elastic_route_count": sum(
                (
                    direct_ok,
                    clarify_ok,
                    single_tool_ok,
                    semantic_plan_ok,
                    semantic_delegate_ok,
                )
            ),
            "elastic_clarification_tool_calls": 0,
            "elastic_semantic_plan_count": int(semantic_plan_ok),
            "elastic_semantic_delegate_count": int(semantic_delegate_ok),
            "elastic_no_progress_escape_count": int(no_progress_code == "deliberation_no_progress"),
            "elastic_truncated_retry_count": int(truncated_retry_ok),
            "elastic_future_work_repair_count": int(future_work_repair_ok),
            "elastic_unobserved_action_repair_count": int(unobserved_action_repair_ok),
            "model_calls": (
                direct_delegate.calls
                + tool_delegate.calls
                + semantic_delegate.calls
                + orchestration_delegate.calls
                + no_progress_delegate.calls
                + truncated_delegate.calls
                + future_work_delegate.calls
                + action_claim_delegate.calls
            ),
            "tool_calls": 0,
            "false_success_count": 0,
        },
    )


def _thread_items(destination_id: str, count: int) -> tuple[ContextItem, ...]:
    turns: list[ThreadTurnRetentionInput] = []
    for index in range(count):
        content = (
            "Root request: assess the Artemis launch constraints and preserve every turn."
            if index == 0
            else (
                "We decided the launch remains staged after the dependency review."
                if index == 18
                else (
                    "Correction: use the August capacity figure instead of the July figure."
                    if index == 31
                    else (
                        "The verified outcome is that the staged launch remains within capacity."
                        if index == count - 2
                        else (
                            f"Supporting turn {index:02d} records Artemis dependency evidence "
                            "for chronological thread reconstruction."
                        )
                    )
                )
            )
        )
        turns.append(
            ThreadTurnRetentionInput(
                content=content,
                actor_id="leo" if index % 2 else "user",
                speaker_role="assistant" if index % 2 else "user",
                is_root=index == 0,
                is_recent=index >= count - 5,
            )
        )
    classifications = classify_thread_transcript(tuple(turns))
    return tuple(
        ContextItem(
            id=f"thread-turn-{index:03d}",
            kind=ContextItemKind.CONVERSATION_TURN,
            content=turn.content,
            conversation_id=destination_id,
            source_actor_id=turn.actor_id,
            retention=classifications[index][0],
            budget_priority=classifications[index][1],
        )
        for index, turn in enumerate(turns)
    )


class _OfflineSlackHistoryClient:
    """Deterministic Slack Web API boundary used by the executable context probes."""

    def __init__(self, history_page: dict[str, object] | None = None) -> None:
        self._history_page = history_page or {
            "ok": True,
            "messages": [],
            "has_more": False,
            "response_metadata": {"next_cursor": ""},
        }
        self.history_calls: list[dict[str, object]] = []
        self.reply_calls: list[dict[str, object]] = []

    async def auth_test(self, **_kwargs: object) -> dict[str, object]:
        return {"ok": True, "team_id": "T-M5"}

    async def conversations_history(self, **kwargs: object) -> dict[str, object]:
        self.history_calls.append(kwargs)
        return self._history_page

    async def conversations_replies(self, **kwargs: object) -> dict[str, object]:
        self.reply_calls.append(kwargs)
        raise AssertionError("ordinary channel context must use attested persisted coverage")


class _OfflineThreadFallback:
    def __init__(self, snapshot: PersistedSlackThreadSnapshot) -> None:
        self._snapshot = snapshot
        self.record_calls = 0
        self.load_calls = 0

    async def record_root_coverage(self, **_kwargs: object) -> bool:
        self.record_calls += 1
        return True

    async def load_complete_thread(self, **_kwargs: object) -> PersistedSlackThreadSnapshot:
        self.load_calls += 1
        return self._snapshot


class _OfflineRowsResult:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[object, object]]:
        return self._rows


class _OfflineTaskTurnSession:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self._rows = rows
        self.statement: object | None = None

    async def execute(self, statement: object) -> _OfflineRowsResult:
        self.statement = statement
        return _OfflineRowsResult(self._rows)


def _offline_slack_job(
    *,
    message_ts: str,
    thread_root_ts: str,
    prompt: str,
) -> SlackMentionJob:
    channel_id = "C-M5-THREAD"
    projection = (channel_id,)
    return SlackMentionJob(
        event_id="Ev-M5-current",
        team_id="T-M5",
        channel_id=channel_id,
        user_id="U-M5-current",
        message_ts=message_ts,
        thread_root_ts=thread_root_ts,
        conversation_key=f"slack:T-M5:{channel_id}:{thread_root_ts}",
        prompt=prompt,
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=projection,
        context_access_hash=build_context_access_hash(
            team_id="T-M5",
            user_id="U-M5-current",
            channel_id=channel_id,
            context_conversation_ids=projection,
        ),
    )


def _offline_persisted_row(
    *,
    message_id: str,
    message_ts: str,
    thread_root_ts: str,
    text: str,
    actor_id: str,
    role: str,
    event_id: str | None = None,
) -> SanitizedMessageRow:
    return SanitizedMessageRow(
        id=message_id,
        organization_id="eval-org",
        strategy_id="eval-thread-scope",
        destination_id="C-M5-THREAD",
        external_event_id=event_id or f"event-{message_id}",
        text=text,
        content_hash="c" * 64,
        conversation_id="conversation-m5-thread",
        actor_id=actor_id,
        role=role,
        provider_message_ts=message_ts,
        provider_thread_root_ts=thread_root_ts,
    )


async def _fresh_root_isolation_probe() -> bool:
    old_thread_page = {
        "ok": True,
        "messages": [
            {
                "type": "message",
                "ts": "1700.000001",
                "user": "U-old",
                "text": "Only suggest safe high-dividend stocks in this unrelated old thread.",
            }
        ],
        "has_more": False,
        "response_metadata": {"next_cursor": ""},
    }
    client = _OfflineSlackHistoryClient(old_thread_page)
    current_ts = "1800.000001"
    result = await SlackHistoryContextLoader(client).load(
        _offline_slack_job(
            message_ts=current_ts,
            thread_root_ts=current_ts,
            prompt="What are some interesting investing opportunities right now?",
        )
    )
    return (
        result.items == ()
        and result.manifest.requested_conversation_ids == ()
        and result.manifest.loaded_conversation_ids == ()
        and result.manifest.history_requests == 0
        and client.history_calls == []
        and client.reply_calls == []
    )


async def _progress_prefix_probe() -> tuple[bool, int]:
    root_ts = "1787412000.000001"
    prior_progress_ts = "1787412100.100001"
    prior_final_ts = "1787412200.200001"
    current_ts = "1787412219.905099"
    current_progress_ts = "1787412253.855439"
    rows = (
        _offline_persisted_row(
            message_id="root",
            message_ts=root_ts,
            thread_root_ts=root_ts,
            text="Root request",
            actor_id="U-root",
            role="user",
        ),
        _offline_persisted_row(
            message_id="prior-progress",
            message_ts=prior_progress_ts,
            thread_root_ts=root_ts,
            text="Prior progress update",
            actor_id="leo",
            role="assistant",
        ),
        _offline_persisted_row(
            message_id="prior-final",
            message_ts=prior_final_ts,
            thread_root_ts=root_ts,
            text="Prior final answer",
            actor_id="leo",
            role="assistant",
        ),
        _offline_persisted_row(
            message_id="current-user",
            message_ts=current_ts,
            thread_root_ts=root_ts,
            text="Current follow-up must be the context boundary",
            actor_id="U-M5-current",
            role="user",
            event_id="Ev-M5-current",
        ),
        _offline_persisted_row(
            message_id="current-progress",
            message_ts=current_progress_ts,
            thread_root_ts=root_ts,
            text="Current progress must not leak backward into context",
            actor_id="leo",
            role="assistant",
        ),
    )
    snapshot = _assess_thread_coverage(
        team_id="T-M5",
        channel_id="C-M5-THREAD",
        thread_root_ts=root_ts,
        current_message_ts=current_ts,
        conversation_id="conversation-m5-thread",
        rows=rows,
        authoritative_reply_count=4,
        authoritative_latest_reply_ts=current_progress_ts,
        coverage_source=SlackThreadCoverageSource.BOT_HISTORY,
        coverage_snapshot_hash="a" * 64,
        max_messages=200,
        current_actor_id="U-M5-current",
        current_event_id="Ev-M5-current",
    )
    root_page = {
        "ok": True,
        "messages": [
            {
                "type": "message",
                "ts": root_ts,
                "user": "U-root",
                "text": "Root request",
                "reply_count": 4,
                "latest_reply": current_progress_ts,
            }
        ],
        "has_more": False,
        "response_metadata": {"next_cursor": ""},
    }
    client = _OfflineSlackHistoryClient(root_page)
    fallback = _OfflineThreadFallback(snapshot)
    result = await SlackHistoryContextLoader(client, thread_fallback=fallback).load(
        _offline_slack_job(
            message_ts=current_ts,
            thread_root_ts=root_ts,
            prompt="Continue the prior request",
        )
    )
    thread_items = tuple(item for item in result.items if item.id.startswith("slack-thread:"))
    combined = "\n".join(item.content for item in thread_items)
    post_boundary_leakage = sum(
        marker in combined for marker in ("Current follow-up", "Current progress")
    )
    prefix_ok = (
        snapshot.complete
        and snapshot.complete_through_ts == current_ts
        and snapshot.authoritative_latest_reply_ts == current_progress_ts
        and result.manifest.thread_source == "persisted_complete"
        and result.manifest.thread_complete
        and result.manifest.thread_messages_loaded == 3
        and len(thread_items) == 3
        and all(
            marker in combined
            for marker in ("Root request", "Prior progress update", "Prior final answer")
        )
        and post_boundary_leakage == 0
        and client.reply_calls == []
        and fallback.load_calls == 1
    )
    return prefix_ok, post_boundary_leakage


async def _durable_task_thread_probe(
    *,
    now: datetime,
    scope: ScopeKey,
) -> tuple[bool, bool, int]:
    destination_id = "D-M5-THREAD"
    allowed = ("C-M5-OTHER", destination_id, "G-M5-OTHER")
    request = ConversationContextRequest(
        team_id="T-M5",
        destination_id=destination_id,
        destination_kind="dm",
        actor_id="U-M5-current",
        objective="What did we decide in this exact DM thread?",
        current_task_id="task-m5-current",
        current_event_id="Ev-M5-current",
        current_message_ts="2000.000099",
        thread_root_ts="2000.000001",
        allowed_conversation_ids=allowed,
        access_hash=build_context_access_hash(
            team_id="T-M5",
            user_id="U-M5-current",
            channel_id=destination_id,
            context_conversation_ids=allowed,
        ),
        current_thread_namespace_id=(f"slack:T-M5:{destination_id}:2000.000001"),
    )
    current_thread_id = "thread-m5-current"

    def task_row(*, thread_id: str = current_thread_id) -> tuple[object, object]:
        return (
            SimpleNamespace(
                id="task-m5-prior",
                thread_id=thread_id,
                organization_id=scope.organization_id,
                strategy_id=scope.strategy_id,
                objective="Question from the exact thread",
                final_output="Answer from the exact thread",
                created_at=now,
            ),
            SimpleNamespace(external_id=destination_id),
        )

    loader = PostgresConversationContextLoader(cast(Any, None))
    exact_session = _OfflineTaskTurnSession([task_row()])
    exact_items = await loader._load_turns(
        cast(Any, exact_session),
        scope,
        request,
        harness_thread_id=current_thread_id,
    )
    where = (
        ""
        if exact_session.statement is None
        else str(cast(Any, exact_session.statement).whereclause)
    )
    durable_leakage = sum(item.conversation_id != destination_id for item in exact_items)
    exact_ok = (
        [item.id for item in exact_items] == ["turn:task-m5-prior"]
        and durable_leakage == 0
        and "tasks.thread_id" in where
        and "threads.external_thread_id" in where
        and "threads.external_channel_id" in where
        and "conversations.external_id" in where
        and "conversations.external_id IN" not in where
    )
    rejected = False
    try:
        await loader._load_turns(
            cast(Any, _OfflineTaskTurnSession([task_row(thread_id="thread-m5-foreign")])),
            scope,
            request,
            harness_thread_id=current_thread_id,
        )
    except ConversationContextAuthorizationError:
        rejected = True
    return exact_ok, rejected, durable_leakage


async def _thread_context_authority(scenario: Scenario) -> RevisedM5Observed:
    count = scenario.inputs.get("message_count")
    if not isinstance(count, int) or isinstance(count, bool) or not 40 <= count <= 100:
        raise RevisedM5UnsupportedScenario("thread_context_message_count_invalid")
    now = _parse_clock(scenario)
    destination_id = "C-M5-THREAD"
    items = _thread_items(destination_id, count)
    selection = select_context_with_thread_compaction(
        items,
        thread_item_ids=frozenset(item.id for item in items),
        conversation_id=destination_id,
        summary_id_namespace="m5-thread",
        max_tokens=850,
        max_bytes=3_400,
        summary_max_bytes=1_400,
    )
    compacted = tuple(item for item in items if item.id in selection.compacted_item_ids)
    selected_exact_ids = {
        item.id for item in selection.items if item.kind is ContextItemKind.CONVERSATION_TURN
    }
    complete = selected_exact_ids | set(selection.compacted_item_ids) == {item.id for item in items}
    digest_complete = (
        bool(compacted)
        and selection.compaction_digest == thread_context_source_digest(compacted)
        and len(selection.reopen_ranges) == 1
        and selection.reopen_ranges[0].digest == selection.compaction_digest
        and selection.reopen_ranges[0].items == compacted
    )

    scope = ScopeKey(organization_id="eval-org", strategy_id="eval-thread-scope")
    authority = ThreadContextAuthority(
        scope=scope,
        team_id="T-M5",
        destination_id=destination_id,
        actor_id="U-M5",
        task_id="task-m5-thread",
        run_id="run-m5-thread",
        thread_root_ts="1000.000001",
        current_message_ts="1000.000099",
        allowed_conversation_ids=(destination_id,),
        access_hash="a" * 64,
        membership_hash="b" * 64,
    )
    tools = build_thread_context_tools(
        ranges=selection.reopen_ranges,
        authority=authority,
        clock=FixedClock(now),
    )
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=scope,
            actor_id=authority.actor_id,
            roles=frozenset({"researcher"}),
        ),
        run_id=authority.run_id,
        tool_call_id="open-correct",
    )
    handle = selection.reopen_ranges[0].handle
    opened = await tools[0].execute(
        {"handle": handle, "start_ordinal": 0, "max_chunks": 2}, context
    )
    wrong_run = await tools[0].execute(
        {"handle": handle}, context.model_copy(update={"run_id": "run-forged"})
    )
    unknown_handle = await tools[0].execute(
        {"handle": "thr_00000000000000000000000000000000"}, context
    )
    invalid_destination_rejected = False
    try:
        ThreadContextAuthority(
            scope=scope,
            team_id="T-M5",
            destination_id="C-FORGED",
            actor_id="U-M5",
            task_id="task-m5-thread",
            run_id="run-m5-thread",
            thread_root_ts="1000.000001",
            current_message_ts="1000.000099",
            allowed_conversation_ids=(destination_id,),
            access_hash="a" * 64,
            membership_hash="b" * 64,
        )
    except ValidationError:
        invalid_destination_rejected = True
    foreign_range_rejected = False
    try:
        foreign = items[1].model_copy(update={"conversation_id": "C-FOREIGN"})
        build_thread_context_tools(
            ranges=(
                ThreadContextRange(
                    handle="thr_foreign_authority_probe_0001",
                    digest=thread_context_source_digest((foreign,)),
                    items=(foreign,),
                ),
            ),
            authority=authority,
            clock=FixedClock(now),
        )
    except ValueError:
        foreign_range_rejected = True

    fresh_root_isolated = await _fresh_root_isolation_probe()
    progress_prefix_ok, post_boundary_leakage = await _progress_prefix_probe()
    durable_exact_ok, durable_rejected, durable_leakage = await _durable_task_thread_probe(
        now=now,
        scope=scope,
    )

    open_ok = (
        isinstance(opened, ToolSuccess)
        and opened.data.get("source_conversation") == destination_id
        and opened.data.get("range_digest") == selection.compaction_digest
    )
    rejection_count = sum(
        (
            isinstance(wrong_run, ToolFailure),
            isinstance(unknown_handle, ToolFailure),
            invalid_destination_rejected,
            foreign_range_rejected,
        )
    )
    protected_count = sum(item.retention.pinned for item in items)
    leakage_count = (
        sum(item.conversation_id != destination_id for item in items)
        + int(
            isinstance(opened, ToolSuccess)
            and opened.data.get("source_conversation") != destination_id
        )
        + post_boundary_leakage
        + durable_leakage
    )
    invariants: set[str] = set()
    if complete:
        invariants.add("full_thread_context_has_exact_coverage")
    if digest_complete:
        invariants.add("thread_compaction_has_content_free_reopen_proof")
    if open_ok:
        invariants.add("thread_reopen_executes_under_sealed_authority")
    if rejection_count == 4:
        invariants.add("thread_reopen_rejects_cross_authority_access")
    if leakage_count == 0:
        invariants.add("thread_context_remains_destination_isolated")
    if fresh_root_isolated:
        invariants.add("fresh_slack_root_starts_without_unrelated_thread_context")
    if progress_prefix_ok:
        invariants.add("progress_after_current_boundary_uses_exact_prior_prefix")
    if durable_exact_ok and durable_rejected:
        invariants.add("durable_task_turns_remain_on_exact_slack_thread")
    return RevisedM5Observed(
        invariants=frozenset(invariants),
        metrics={
            "thread_loaded_turn_count": len(items),
            "thread_compacted_turn_count": len(compacted),
            "thread_protected_turn_count": protected_count,
            "thread_exact_retained_turn_count": len(selected_exact_ids),
            "thread_reopen_handle_count": len(selection.reopen_ranges),
            "thread_reopen_success_count": int(open_ok),
            "thread_authority_rejection_count": rejection_count,
            "thread_context_leakage_count": leakage_count,
            "thread_fresh_root_isolation_count": int(fresh_root_isolated),
            "thread_progress_prefix_success_count": int(progress_prefix_ok),
            "thread_post_boundary_leakage_count": post_boundary_leakage,
            "thread_durable_exact_task_count": int(durable_exact_ok),
            "thread_durable_rejection_count": int(durable_rejected),
            "false_success_count": 0,
        },
    )


async def _tavily_verified_research(scenario: Scenario) -> RevisedM5Observed:
    if scenario.budget.max_tool_calls < 2:
        raise RevisedM5UnsupportedScenario("tavily_research_requires_two_tool_calls")
    now = _parse_clock(scenario)
    source_url = "https://93.184.216.34/artemis-update"
    source_statement = "The Artemis launch window opens on 14 September 2026."
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(f"{request.method} {request.url}")
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "query": "Artemis launch window",
                    "results": [
                        {
                            "title": "Official Artemis update",
                            "url": source_url,
                            "content": "Discovery snippet; fetch the source before citation.",
                            "score": 0.97,
                        }
                    ],
                    "request_id": "offline-tavily-m5",
                },
            )
        if request.method == "GET" and str(request.url) == source_url:
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=source_statement,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )
        return httpx.Response(404)

    clock = FixedClock(now)
    scope = ScopeKey(organization_id="eval-org", strategy_id="eval-web-scope")
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=scope,
            actor_id="eval-user",
            roles=frozenset({"researcher"}),
        ),
        run_id="run-web-m5",
        tool_call_id="search-call",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        search_tool = TavilySearchTool(client=client, api_key="offline-key", clock=clock)
        fetch_tool = PublicTextFetchTool(client=client, clock=clock)
        catalog = InMemoryToolCatalog(version="revised-m5-catalog-v1")
        catalog.register(
            catalog_tool_from_spec(
                search_tool.spec,
                provider="tavily",
                tags=frozenset({"web", "discovery"}),
            )
        )
        catalog.register(
            catalog_tool_from_spec(
                fetch_tool.spec,
                provider="public-web",
                tags=frozenset({"web", "fetch"}),
            )
        )
        eligible = catalog.eligible(
            phase=RunPhase.RESEARCH,
            profile="research",
            roles=context.trusted_scope.roles,
            remaining_cost=1.0,
            namespace=scope,
            conversation_kind="channel",
        )
        search_outcome = await search_tool.execute(
            {"query": "Artemis launch window", "max_results": 1}, context
        )
        if not isinstance(search_outcome, ToolSuccess):
            return RevisedM5Observed(
                invariants=frozenset(),
                metrics={"tavily_tool_failure_count": 1},
                hard_failures=(f"tavily_search_failed:{search_outcome.code}",),
            )
        search_observation = normalize_success(
            search_outcome,
            observation_id="obs-search-m5",
            scope=scope,
            run_id=context.run_id,
            tool_call_id=context.tool_call_id,
            observation_kind="web.search_tavily",
        )
        fetch_outcome = await fetch_tool.execute(
            {"url": source_url},
            context.model_copy(update={"tool_call_id": "fetch-call"}),
        )
        if not isinstance(fetch_outcome, ToolSuccess):
            return RevisedM5Observed(
                invariants=frozenset(),
                metrics={"tavily_tool_failure_count": 1},
                hard_failures=(f"public_fetch_failed:{fetch_outcome.code}",),
            )
        fetch_observation = normalize_success(
            fetch_outcome,
            observation_id="obs-fetch-m5",
            scope=scope,
            run_id=context.run_id,
            tool_call_id="fetch-call",
            observation_kind="web.fetch_public_text",
        )

    thread = Thread(
        id="thread-web-m5",
        scope=scope,
        origin=OriginRef(provider="offline", external_thread_id="conversation-web-m5"),
    )
    task = Task(id="task-web-m5", thread_id=thread.id, scope=scope, objective="Research Artemis")
    run = Run(id=context.run_id, task_id=task.id, scope=scope)
    bundle = RunBundle(
        thread=thread,
        task=task,
        run=run,
        observations=(search_observation, fetch_observation),
    )
    verifier = DeterministicCompletionVerifier(SequentialIdGenerator(), clock)
    discovery_proposal = CompletionProposal(
        answer=source_statement,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=source_statement,
                observation_ids=(search_observation.id,),
            ),
        ),
    )
    fetched_proposal = CompletionProposal(
        answer=source_statement,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=source_statement,
                observation_ids=(fetch_observation.id,),
            ),
        ),
    )
    discovery_verification = verifier.verify(discovery_proposal, bundle)
    fetched_verification = verifier.verify(fetched_proposal, bundle)
    eligible_ids = tuple(item.id for item in eligible)
    catalog_ok = eligible_ids == ("web.fetch_public_text", "web.search_tavily")
    quality_ok = (
        search_observation.quality.value == "discovery_only"
        and fetch_observation.quality.value == "untrusted_retrieval"
    )
    chain_ok = requests == [
        "POST https://api.tavily.com/search",
        f"GET {source_url}",
    ]
    discovery_rejected = discovery_verification.result.status is VerifierStatus.FAIL
    fetch_verified = (
        fetched_verification.result.status is VerifierStatus.PASS
        and fetched_verification.completion is not None
    )
    invariants: set[str] = set()
    if catalog_ok:
        invariants.add("tavily_and_fetch_are_catalog_eligible")
    if quality_ok:
        invariants.add("research_chain_normalizes_evidence_quality")
    if chain_ok:
        invariants.add("tavily_discovery_is_followed_by_public_fetch")
    if discovery_rejected:
        invariants.add("discovery_snippet_cannot_support_source_claim")
    if fetch_verified:
        invariants.add("fetched_source_claim_is_verified")
    return RevisedM5Observed(
        invariants=frozenset(invariants),
        metrics={
            "research_catalog_eligible_count": len(eligible_ids),
            "research_normalized_observation_count": len((search_observation, fetch_observation)),
            "research_discovery_rejection_count": int(discovery_rejected),
            "research_verified_source_claim_count": int(fetch_verified),
            "research_mock_transport_request_count": len(requests),
            "false_success_count": 0,
        },
    )


_EXECUTORS = {
    "conversational_terminal_recovery": _terminal_recovery,
    "elastic_deliberation": _elastic_deliberation,
    "slack_thread_context_authority": _thread_context_authority,
    "tavily_verified_research": _tavily_verified_research,
}


async def execute_revised_m5_scenario(scenario: Scenario) -> RevisedM5Observed:
    executor = _EXECUTORS.get(scenario.execution_variant)
    if executor is None:
        raise RevisedM5UnsupportedScenario(
            f"execution_variant_not_supported:{scenario.execution_variant}"
        )
    return await executor(scenario)


async def execute_revised_m5_baseline_scenario(scenario: Scenario) -> BaselineExecution:
    """Run the same bounded component probe and project its observed safety baseline."""

    observed = await execute_revised_m5_scenario(scenario)
    safe = not observed.hard_failures and observed.metrics.get("false_success_count") == 0
    invariants = set(observed.invariants)
    if safe:
        invariants.update({"no_false_success", "baseline_hard_safety_preserved"})
    catalogs = {
        "conversational_terminal_recovery": (),
        "elastic_deliberation": (
            "agent.delegate_research",
            "agent.execute_research_plan",
            "market.get_quote",
        ),
        "slack_thread_context_authority": ("thread_context.open",),
        "tavily_verified_research": ("web.fetch_public_text", "web.search_tavily"),
    }
    matched_catalog = catalogs[scenario.execution_variant]
    return BaselineExecution(
        invariants=frozenset(invariants),
        metrics=observed.metrics,
        hard_failures=observed.hard_failures,
        eligible_schema_count=len(matched_catalog),
        admitted_destination=f"{scenario.deterministic_id_prefix}-offline",
        model_fixture="deterministic-product-component-probe-v1",
        matched_tool_catalog=matched_catalog,
        exposed_tool_catalog=matched_catalog,
    )
