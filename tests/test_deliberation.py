from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from leo.demo import run_conversation_smoke
from leo.harness.deliberation import (
    DeliberationMode,
    ElasticDeliberationGateway,
    ElasticDeliberationPolicy,
    memory_recovery_query,
)
from leo.harness.models import (
    BudgetLimits,
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    ContextManifest,
    ContextSegment,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    Observation,
    ScopeKey,
    SourceRef,
    ToolArgumentConstraint,
    ToolChoiceMode,
    ToolChoicePolicy,
    ToolRequest,
    ToolRequests,
    ToolRetryPolicy,
    ToolSpec,
)
from leo.harness.ports import ModelGatewayError

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
PARENT_TOOLS = frozenset({"agent.delegate_research", "agent.execute_research_plan"})


def test_short_natural_requests_select_outcome_appropriate_depth() -> None:
    policy = ElasticDeliberationPolicy()

    direct = policy.assess("Explain covariance in two sentences.")
    clarify = policy.assess(
        "Compare two unspecified options; ask exactly one concise clarifying question; "
        "do not research or use tools.",
        explicit_tool_free=True,
    )
    recall = policy.assess(
        "What did we decide earlier?",
        memory_recall_required=True,
        context_item_count=2,
    )
    quote = policy.assess(
        "Where is NVDA trading now?",
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )

    assert direct.mode is DeliberationMode.DIRECT
    assert clarify.mode is DeliberationMode.CLARIFY
    assert recall.mode is DeliberationMode.CONTEXT_MEMORY
    assert quote.mode is DeliberationMode.SINGLE_TOOL


@pytest.mark.parametrize(
    "objective",
    [
        (
            "How fragile is Microsoft's valuation if demand cools? Ground the answer in its "
            "current trading level and newest regulatory disclosures."
        ),
        (
            "Assess Microsoft's valuation downside under weaker enterprise demand using today's "
            "share level alongside its most recent filing disclosures."
        ),
    ],
)
def test_multi_source_natural_language_builds_broad_envelope_with_plan_advisory(
    objective: str,
) -> None:
    decision = ElasticDeliberationPolicy().assess(
        objective,
        evidence_tool_names=("market.get_quote", "sec.get_recent_filings"),
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )

    assert decision.recommended_mode is DeliberationMode.PLAN
    assert DeliberationMode.PARALLEL_READS in decision.allowed_modes
    assert DeliberationMode.DELEGATE in decision.allowed_modes
    assert decision.required_parent_tool is None
    assert decision.minimum_depth == 2
    assert decision.maximum_depth == 6
    assert decision.signals.independent_evidence_count == 2


def test_underspecified_consequential_action_clarifies_before_acting() -> None:
    decision = ElasticDeliberationPolicy().assess("Sell it now.")

    assert decision.mode is DeliberationMode.CLARIFY
    assert decision.signals.action_risk
    assert decision.signals.ambiguous


def test_short_prompt_depth_depends_on_context_and_freshness_not_length() -> None:
    policy = ElasticDeliberationPolicy()

    why_without_context = policy.assess("Why?")
    why_with_context = policy.assess("Why?", context_item_count=1)
    still_true = policy.assess("Is that still true?", context_item_count=2)
    compare_missing = policy.assess("Compare these")

    assert why_without_context.mode is DeliberationMode.CLARIFY
    assert why_with_context.mode is DeliberationMode.CONTEXT_MEMORY
    assert still_true.mode is DeliberationMode.CONTEXT_MEMORY
    assert still_true.signals.freshness_required
    assert compare_missing.mode is DeliberationMode.CLARIFY


@pytest.mark.parametrize(
    "objective",
    [
        (
            "Rewrite this in a friendly tone under 45 words: The deployment is delayed "
            "because the migration review found two safety issues. We will share a revised "
            "date after the fixes are verified."
        ),
        (
            "Give me exactly three friendly code names for a project that helps teams find "
            "reliable information faster. Names only, one per line."
        ),
        "Draft a tagline.",
        "Rewrite this: Deployment delayed.",
    ],
)
def test_complete_short_or_relative_clause_prompts_stay_direct(objective: str) -> None:
    decision = ElasticDeliberationPolicy().assess(objective)

    assert decision.mode is DeliberationMode.DIRECT
    assert not decision.signals.ambiguous
    assert not decision.hard_require_clarification


@pytest.mark.parametrize(
    "objective",
    [
        "Compare these",
        "Sell it now.",
        "Rewrite this in two sentences.",
        "What does that mean?",
        "Summarize those findings.",
    ],
)
def test_unresolved_direct_objects_still_require_clarification(objective: str) -> None:
    decision = ElasticDeliberationPolicy().assess(objective)

    assert decision.mode is DeliberationMode.CLARIFY
    assert decision.signals.ambiguous
    assert decision.hard_require_clarification


def test_current_event_question_selects_open_ended_research_without_incantation() -> None:
    decision = ElasticDeliberationPolicy().assess(
        "What happened to NVDA today?",
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )

    assert decision.mode is DeliberationMode.MULTI_SOURCE
    assert decision.signals.open_ended_current_event
    assert decision.required_parent_tool is None
    assert decision.minimum_depth == 1


@pytest.mark.parametrize(
    ("objective", "expected_tool", "expected_mode", "minimum_depth"),
    [
        (
            "Build and execute a two-step research plan for NVDA.",
            "agent.execute_research_plan",
            DeliberationMode.PLAN,
            5,
        ),
        (
            "Delegate NVDA quote research to a subagent now.",
            "agent.delegate_research",
            DeliberationMode.DELEGATE,
            4,
        ),
    ],
)
def test_explicit_orchestration_effect_is_hard_while_recommendations_are_not(
    objective: str,
    expected_tool: str,
    expected_mode: DeliberationMode,
    minimum_depth: int,
) -> None:
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        available_tool_names=PARENT_TOOLS,
    )

    assert envelope.recommended_mode is expected_mode
    assert envelope.required_parent_tool == expected_tool
    assert envelope.minimum_depth == minimum_depth
    assert DeliberationMode.SINGLE_TOOL not in envelope.allowed_modes


def test_explicit_no_tools_caps_even_complex_prompt_at_direct_or_clarify() -> None:
    decision = ElasticDeliberationPolicy().assess(
        "Compare three scenarios, reconcile every disagreement, and recommend one. "
        "Do not research or use tools.",
        explicit_tool_free=True,
        evidence_tool_names=("market.get_quote", "sec.get_recent_filings"),
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )

    assert decision.mode in {DeliberationMode.DIRECT, DeliberationMode.CLARIFY}
    assert decision.required_parent_tool is None


def test_orchestration_discussion_is_not_mistaken_for_requested_effect() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "Should we use a parallel plan for the next analysis?",
        available_tool_names=PARENT_TOOLS,
    )

    assert envelope.required_parent_tool is None
    assert DeliberationMode.PLAN in envelope.allowed_modes


class _RecordingGateway:
    def __init__(self, decisions: list[ToolRequests | CompletionProposal]) -> None:
        self.decisions = decisions
        self.requests: list[ModelRequest] = []

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.requests.append(request)
        decision = self.decisions.pop(0)
        return ModelTurnResult(
            decision=decision,
            provider="fixture",
            model="recording",
        )


class _UnavailableGateway:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        raise ModelGatewayError("provider_unavailable", "The provider is unavailable.")


class _MeteredOffEnvelopeGateway:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        self.calls += 1
        return ModelTurnResult(
            decision=_tool_request("off-envelope-tool"),
            provider="openrouter",
            model="live-model",
            request_id="live-attempt-1",
            finish_reason="tool_calls",
            usage=ModelUsage(
                prompt_tokens=12,
                completion_tokens=4,
                total_tokens=16,
                cost=0.001,
            ),
        )


def _tool_request(call_id: str = "provider-generated") -> ToolRequests:
    return ToolRequests(
        calls=(
            ToolRequest(
                id=call_id,
                name="market.get_quote",
                arguments={"symbol": "NVDA"},
            ),
        )
    )


def _request(
    iteration: int,
    *,
    observations: tuple[Observation, ...] = (),
    feedback: tuple[str, ...] = (),
    tools: tuple[ToolSpec, ...] = (),
) -> ModelRequest:
    return ModelRequest(
        objective="Where is NVDA trading now?",
        iteration=iteration,
        observations=observations,
        verifier_feedback=feedback,
        tools=tools,
        tool_choice=ToolChoicePolicy(mode=ToolChoiceMode.AUTO),
        manifest=ContextManifest(
            segments=(ContextSegment(name="objective", priority=100, pinned=True),)
        ),
    )


@pytest.mark.asyncio
async def test_gateway_rejects_same_action_without_new_evidence_ignoring_call_id() -> None:
    delegate = _RecordingGateway([_tool_request("call-1"), _tool_request("call-2")])
    decision = ElasticDeliberationPolicy().assess(
        "Where is NVDA trading now?",
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
    )
    gateway = ElasticDeliberationGateway(delegate, decision)

    await gateway.decide(_request(0))
    with pytest.raises(ModelGatewayError) as captured:
        await gateway.decide(_request(1, feedback=("The required tool was not used.",)))

    assert captured.value.code == "deliberation_repeated_decision"
    assert len(delegate.requests) == 2
    assert "advisory multi_source" in delegate.requests[1].completion_contract.guidance


@pytest.mark.asyncio
async def test_gateway_allows_exactly_one_server_authorized_retryable_read() -> None:
    delegate = _RecordingGateway(
        [_tool_request("call-1"), _tool_request("call-2"), _tool_request("call-3")]
    )
    decision = ElasticDeliberationPolicy().assess(
        "Where is NVDA trading now?",
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
    )
    gateway = ElasticDeliberationGateway(delegate, decision)
    retryable_quote = ToolSpec(
        name="market.get_quote",
        description="Read a current market quote.",
        domain="market",
        input_schema={"type": "object", "properties": {}},
        retry=ToolRetryPolicy(max_attempts=2),
    )

    await gateway.decide(_request(0, tools=(retryable_quote,)))
    await gateway.decide(
        _request(
            1,
            feedback=("The provider read was temporarily unavailable.",),
            tools=(retryable_quote,),
        )
    )
    with pytest.raises(ModelGatewayError) as captured:
        await gateway.decide(
            _request(
                2,
                feedback=(
                    "The provider read was temporarily unavailable.",
                    "The provider read was temporarily unavailable.",
                ),
                tools=(retryable_quote,),
            )
        )

    assert captured.value.code == "deliberation_repeated_decision"
    assert len(delegate.requests) == 3


@pytest.mark.asyncio
async def test_new_observation_resets_no_progress_and_allows_next_decision() -> None:
    delegate = _RecordingGateway([_tool_request("call-1"), _tool_request("call-2")])
    decision = ElasticDeliberationPolicy().assess(
        "Where is NVDA trading now?",
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
    )
    gateway = ElasticDeliberationGateway(delegate, decision)
    observation = Observation(
        id="obs-1",
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        run_id="run-1",
        tool_call_id="call-1",
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="fixture", reference="quote:NVDA"),
        observed_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        raw_hash="a" * 64,
    )

    await gateway.decide(_request(0))
    result = await gateway.decide(_request(1, observations=(observation,)))

    assert isinstance(result.decision, ToolRequests)
    assert len(delegate.requests) == 2


@pytest.mark.asyncio
async def test_gateway_has_bounded_escalation_even_when_provider_varies_output() -> None:
    delegate = _RecordingGateway(
        [
            CompletionProposal(answer="First attempt."),
            CompletionProposal(answer="Second attempt."),
            CompletionProposal(answer="Third attempt."),
        ]
    )
    decision = ElasticDeliberationPolicy().assess("Explain covariance in two sentences.")
    gateway = ElasticDeliberationGateway(delegate, decision, max_no_progress_turns=2)

    await gateway.decide(_request(0))
    await gateway.decide(_request(1, feedback=("Unsupported.",)))
    await gateway.decide(_request(2, feedback=("Still unsupported.",)))
    with pytest.raises(ModelGatewayError) as captured:
        await gateway.decide(_request(3, feedback=("Still unsupported.",)))

    assert captured.value.code == "deliberation_no_progress"
    assert len(delegate.requests) == 3
    assert gateway.decision.depth <= 6


@pytest.mark.asyncio
async def test_context_only_completion_drops_invalid_context_item_citations() -> None:
    delegate = _RecordingGateway(
        [
            CompletionProposal(
                answer="Confirm test received.",
                claims=(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement="The test asked for a receipt confirmation.",
                        observation_ids=("thread-message:context-only",),
                    ),
                ),
            )
        ]
    )
    decision = ElasticDeliberationPolicy().assess(
        "Summarize my test request in exactly three words.",
        context_item_count=2,
    )
    gateway = ElasticDeliberationGateway(delegate, decision)

    result = await gateway.decide(_request(0))

    assert isinstance(result.decision, CompletionProposal)
    assert result.decision.claims == ()


@pytest.mark.asyncio
async def test_a_stuck_loop_fails_with_the_models_answer_not_a_harness_written_one() -> None:
    """The harness must not paste source text over the model's decision.

    This previously returned the Exa statement verbatim as Leo's answer. In
    production that path seized twelve consecutive turns of a live run and
    delivered a scraped page dump -- headings, volume figures, cross-listings --
    to Slack as the final reply. A stuck loop is now surfaced as the failure it
    is, carrying the model's own last answer as the fallback the coordinator
    delivers.
    """

    statement = (
        "Example issuer reported resilient revenue growth while noting execution risk. "
        "Source: https://example.test/research"
    )
    observation = Observation(
        id="obs-exa",
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        run_id="run-exa",
        tool_call_id="call-exa",
        kind="web.research_verified",
        data={
            "selected_provider": "exa",
            "exact_url_bound_claims": True,
            "statements": [statement],
        },
        source=SourceRef(provider="exa", reference="https://example.test/research"),
        observed_at=NOW,
        expires_at=NOW + timedelta(minutes=5),
        raw_hash="b" * 64,
    )
    delegate = _RecordingGateway(
        [
            CompletionProposal(answer="A paraphrase."),
            CompletionProposal(answer="Another paraphrase."),
            CompletionProposal(answer="A third paraphrase."),
        ]
    )
    gateway = ElasticDeliberationGateway(
        delegate,
        ElasticDeliberationPolicy().assess(
            "Summarize the current research.",
            external_evidence_required=True,
        ),
    )

    await gateway.decide(_request(0, observations=(observation,)))
    await gateway.decide(
        _request(1, observations=(observation,), feedback=("Use exact source evidence.",))
    )
    result = await gateway.decide(
        _request(2, observations=(observation,), feedback=("Still use exact source evidence.",))
    )

    # A fourth turn with no new evidence is a genuinely stuck loop. It is reported
    # as a failure carrying the model's own last answer, never repaired by pasting
    # the source text over the model's decision.
    with pytest.raises(ModelGatewayError) as stuck:
        await gateway.decide(
            _request(3, observations=(observation,), feedback=("Exact evidence required.",))
        )
    assert stuck.value.code == "deliberation_no_progress"
    assert stuck.value.fallback_answer == "A third paraphrase."

    assert isinstance(result.decision, CompletionProposal)
    assert result.decision.answer == "A third paraphrase."


@pytest.mark.asyncio
async def test_semantic_model_can_choose_parallel_reads_inside_plan_recommendation() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "How fragile is Microsoft's valuation if demand cools? Ground it in current market "
        "and regulatory evidence.",
        evidence_tool_names=("market.get_quote", "sec.get_recent_filings"),
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )
    parallel = ToolRequests(
        calls=(
            ToolRequest(id="quote", name="market.get_quote", arguments={"symbol": "MSFT"}),
            ToolRequest(
                id="filings",
                name="sec.get_recent_filings",
                arguments={"ticker": "MSFT"},
            ),
        )
    )
    delegate = _RecordingGateway([parallel])

    result = await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert envelope.recommended_mode is DeliberationMode.PLAN
    assert isinstance(result.decision, ToolRequests)
    assert tuple(call.name for call in result.decision.calls) == (
        "market.get_quote",
        "sec.get_recent_filings",
    )


@pytest.mark.asyncio
async def test_semantic_model_can_choose_plan_inside_direct_recommendation() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "Synthesize a sequenced rollout for three modules, accounting for dependencies "
        "and parallel workstreams.",
        available_tool_names=PARENT_TOOLS,
    )
    plan = ToolRequests(
        calls=(
            ToolRequest(
                id="plan",
                name="agent.execute_research_plan",
                arguments={"objective": "Synthesize the rollout."},
            ),
        )
    )

    result = await ElasticDeliberationGateway(_RecordingGateway([plan]), envelope).decide(
        _request(0)
    )

    assert envelope.recommended_mode is DeliberationMode.DIRECT
    assert envelope.maximum_depth == 6
    assert envelope.required_parent_tool is None
    assert isinstance(result.decision, ToolRequests)


@pytest.mark.asyncio
async def test_simple_prompt_can_stay_one_turn_inside_elastic_ceiling() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "Explain covariance in two sentences.",
        available_tool_names=PARENT_TOOLS,
    )
    delegate = _RecordingGateway(
        [CompletionProposal(answer="Covariance measures joint variation.")]
    )

    result = await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert envelope.recommended_mode is DeliberationMode.DIRECT
    assert envelope.maximum_depth == 6
    assert isinstance(result.decision, CompletionProposal)
    assert len(delegate.requests) == 1


@pytest.mark.asyncio
async def test_hard_no_tool_envelope_rejects_model_tool_proposal() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "Explain covariance; do not use tools.",
        explicit_tool_free=True,
    )
    delegate = _RecordingGateway([_tool_request()])

    with pytest.raises(ModelGatewayError) as captured:
        await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert captured.value.code == "deliberation_mode_outside_envelope"


@pytest.mark.asyncio
async def test_gateway_enforces_minimum_depth_for_fresh_evidence_request() -> None:
    envelope = ElasticDeliberationPolicy().assess(
        "What happened to NVDA today?",
        external_evidence_required=True,
        available_tool_names=PARENT_TOOLS,
    )
    delegate = _RecordingGateway([CompletionProposal(answer="Nothing changed.")])

    with pytest.raises(ModelGatewayError) as captured:
        await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert captured.value.code == "deliberation_depth_below_minimum"


@pytest.mark.asyncio
async def test_a_provider_outage_surfaces_instead_of_being_masked_as_a_question() -> None:
    """An outage is an outage, not a clarifying question.

    This used to manufacture "What specific information should I use to complete
    that request?" whenever the provider failed on an ambiguous-looking turn. The
    user was asked to supply something that would not have helped -- the model was
    never reached -- and a real outage was hidden behind what looked like Leo
    being curious. The error now propagates to the coordinator, which owns the
    bounded retry and the best-effort delivery path.
    """

    envelope = ElasticDeliberationPolicy().assess("Compare these")

    with pytest.raises(ModelGatewayError) as outage:
        await ElasticDeliberationGateway(_UnavailableGateway(), envelope).decide(_request(0))

    assert outage.value.code == "provider_unavailable"


@pytest.mark.asyncio
async def test_known_ambiguity_accepts_multiple_genuine_clarifying_questions() -> None:
    envelope = ElasticDeliberationPolicy().assess("Compare these")
    delegate = _RecordingGateway(
        [
            CompletionProposal(
                answer="Which two options should I compare? Which criteria matter most?"
            )
        ]
    )

    result = await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert isinstance(result.decision, CompletionProposal)
    assert result.decision.answer.count("?") == 2


@pytest.mark.asyncio
async def test_an_off_envelope_answer_is_nudged_once_then_the_model_decides() -> None:
    """The envelope argues its case once; it does not overwrite the answer.

    "Compare these" has no resolvable subject, so the envelope asks for
    clarification -- and this answer is not one. The gateway used to replace it
    outright with "Which specific target, options, or missing details should I
    use?", discarding whatever the model had written.

    Now the turn is bounced back once with corrective feedback. If the model
    stands by its route on the retry, the model wins: it can see the thread and
    the request, and the envelope is only a regex over the prompt.
    """

    envelope = ElasticDeliberationPolicy().assess("Compare these")
    answer = "Option A is cheaper. Does that help?"
    delegate = _RecordingGateway([CompletionProposal(answer=answer)] * 2)
    gateway = ElasticDeliberationGateway(delegate, envelope)

    with pytest.raises(ModelGatewayError) as nudge:
        await gateway.decide(_request(0))
    assert nudge.value.code == "deliberation_mode_outside_envelope"

    result = await gateway.decide(_request(1))

    assert isinstance(result.decision, CompletionProposal)
    assert result.decision.answer == answer
    assert result.provider == "fixture"
    assert result.model == "recording"


@pytest.mark.asyncio
async def test_a_future_work_promise_becomes_a_real_research_call() -> None:
    """A promise is converted into the work it promised, not into a question.

    This used to replace the model's answer with a fixed "Which market or asset
    class, risk tolerance, and time horizon should I focus on?" -- a question the
    harness invented, thrown over whatever the model had actually written. The
    deferral recovery still fires, but every branch it has now issues a *tool
    call*: if the model says it is about to research something, the harness makes
    that research happen instead of stalling the conversation.
    """

    objective = "What are some interesting investing opportunities currently?"
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        external_evidence_required=True,
        available_tool_names=frozenset({"web.research_verified"}),
    )
    delegate = _RecordingGateway(
        [
            CompletionProposal(
                answer=(
                    "Happy to help — let me pull a few current quotes and recent dividend "
                    "data, and then I can narrow in on the strongest candidates."
                )
            )
        ]
    )
    request = _request(
        0,
        tools=(
            ToolSpec(
                name="web.research_verified",
                description="Search verified current sources.",
                domain="WEB",
                input_schema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ),
    ).model_copy(update={"objective": objective})

    result = await ElasticDeliberationGateway(delegate, envelope).decide(request)

    assert isinstance(result.decision, ToolRequests)
    assert [call.name for call in result.decision.calls] == ["web.research_verified"]
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_an_off_envelope_tool_is_nudged_and_the_attempt_is_accounted() -> None:
    objective = "What are some interesting investing opportunities right now?"
    delegate = _MeteredOffEnvelopeGateway()
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        external_evidence_required=True,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    result = await run_conversation_smoke(
        model=ElasticDeliberationGateway(delegate, envelope),
        objective=objective,
        limits=BudgetLimits(
            max_iterations=2,
            max_model_calls=2,
            max_tool_calls=0,
            max_cost=0.01,
        ),
    )

    # The model proposes a tool the envelope did not expect. It is nudged once
    # with corrective feedback and charged for the attempt -- it is not replaced
    # by a harness-written question, which is what this used to assert ("Which
    # market or asset class, risk tolerance, and time horizon should I focus
    # on?"). That sentence was never the model's, and answering it would not have
    # helped: the model had a usable route the whole time.
    assert delegate.calls == 2
    assert result.run.usage.model_calls == 2
    assert result.run.usage.tool_calls == 0
    assert result.run.final_output != (
        "Which market or asset class, risk tolerance, and time horizon should I focus on?"
    )
    assert result.run.usage.prompt_tokens == 12
    assert result.run.usage.completion_tokens == 4
    assert result.run.usage.total_tokens == 16
    assert result.run.usage.cost == 0.001


@pytest.mark.asyncio
async def test_clarification_recovery_does_not_override_explicit_effect_envelope() -> None:
    objective = "Build and execute a two-step research plan for NVDA."
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        available_tool_names=PARENT_TOOLS,
    )
    delegate = _RecordingGateway([_tool_request("wrong-single-read")])

    with pytest.raises(ModelGatewayError) as captured:
        await ElasticDeliberationGateway(delegate, envelope).decide(_request(0))

    assert envelope.hard_required_parent_tool == "agent.execute_research_plan"
    assert captured.value.code == "deliberation_mode_outside_envelope"


@pytest.mark.asyncio
async def test_specific_current_evidence_promise_dispatches_sealed_required_read_now() -> None:
    objective = "What is NVDA trading at now?"
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        evidence_tool_names=("market.get_quote",),
        external_evidence_required=True,
        available_tool_names=frozenset({"market.get_quote"}),
    )
    delegate = _RecordingGateway(
        [CompletionProposal(answer="I'll pull the current NVDA quote, then I can answer.")]
    )
    quote = ToolSpec(
        name="market.get_quote",
        description="Read a current market quote.",
        domain="MARKET",
        input_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
            "additionalProperties": False,
        },
    )
    request = _request(0, tools=(quote,)).model_copy(
        update={
            "objective": objective,
            "tool_choice": ToolChoicePolicy(
                mode=ToolChoiceMode.REQUIRED,
                required_tool_name="market.get_quote",
                required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
            ),
        }
    )

    result = await ElasticDeliberationGateway(delegate, envelope).decide(request)

    assert isinstance(result.decision, ToolRequests)
    assert len(result.decision.calls) == 1
    assert result.decision.calls[0].name == "market.get_quote"
    assert result.decision.calls[0].arguments == {"symbol": "NVDA"}
    assert result.provider == "leo-harness"
    assert result.model == "elastic-required-read-v1"


@pytest.mark.asyncio
async def test_live_grab_promise_dispatches_advertised_verified_read_now() -> None:
    objective = "Research current dividend data for JNJ."
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        external_evidence_required=True,
        available_tool_names=frozenset({"web.research_verified"}),
    )
    delegate = _RecordingGateway(
        [
            CompletionProposal(
                answer=(
                    "Here's a preliminary mix. I pulled current quotes for a few names. "
                    "Let me grab live data first."
                )
            )
        ]
    )
    verified_web = ToolSpec(
        name="web.research_verified",
        description="Search verified current sources.",
        domain="WEB",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    )
    request = _request(0, tools=(verified_web,)).model_copy(update={"objective": objective})

    result = await ElasticDeliberationGateway(delegate, envelope).decide(request)

    assert isinstance(result.decision, ToolRequests)
    assert len(result.decision.calls) == 1
    assert result.decision.calls[0].name == "web.research_verified"
    assert result.decision.calls[0].arguments == {"query": objective}
    assert result.provider == "leo-harness"
    assert result.model == "elastic-web-research-verified-v2"


@pytest.mark.asyncio
async def test_contextual_memory_future_work_becomes_required_authorized_search() -> None:
    objective = "What color was it again, and which conversation did that come from?"
    envelope = ElasticDeliberationPolicy().assess(
        objective,
        context_item_count=2,
        memory_recall_required=True,
        available_tool_names=PARENT_TOOLS,
    )
    delegate = _RecordingGateway(
        [
            CompletionProposal(
                answer="I need to pin down the source. Let me check current DM memory."
            )
        ]
    )
    request = ModelRequest(
        objective=objective,
        iteration=0,
        observations=(),
        verifier_feedback=(),
        tools=(
            ToolSpec(
                name="memory.search",
                description="Search authorized memory.",
                domain="MEMORY",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            ),
        ),
        tool_choice=ToolChoicePolicy(
            mode=ToolChoiceMode.REQUIRED,
            required_tool_name="memory.search",
        ),
        manifest=ContextManifest(
            segments=(ContextSegment(name="objective", priority=100, pinned=True),)
        ),
        context_items=(
            ContextItem(
                id="thread-root",
                kind=ContextItemKind.CONVERSATION_TURN,
                content="User: What did I ask Leo to remember about Project Borealis?",
                conversation_id="D1",
            ),
            ContextItem(
                id="thread-answer",
                kind=ContextItemKind.CONVERSATION_TURN,
                content=(
                    "Assistant: Project Borealis uses amber hexagons, from conversation "
                    "C0BRFU0LQF8."
                ),
                conversation_id="D1",
            ),
        ),
    )

    result = await ElasticDeliberationGateway(delegate, envelope).decide(request)

    assert isinstance(result.decision, ToolRequests)
    call = result.decision.calls[0]
    assert call.name == "memory.search"
    assert call.arguments["query"] == "project borealis"
    assert result.provider == "leo-harness"


def test_memory_recovery_query_pins_root_and_latest_material_answer_over_failed_turns() -> None:
    request = _request(0).model_copy(
        update={
            "objective": "What color was it again, and which conversation did that come from?",
            "context_items": (
                ContextItem(
                    id="root",
                    kind=ContextItemKind.CONVERSATION_TURN,
                    content="User: What did I ask Leo to remember about Project Borealis?",
                    conversation_id="D1",
                    retention=ContextItemRetention.THREAD_ROOT,
                ),
                ContextItem(
                    id="answer",
                    kind=ContextItemKind.CONVERSATION_TURN,
                    content=(
                        "Assistant: Project Borealis uses amber hexagons, from conversation "
                        "C0BRFU0LQF8."
                    ),
                    conversation_id="D1",
                    retention=ContextItemRetention.PRIOR_OUTCOME,
                ),
                *(
                    ContextItem(
                        id=f"failed-{index}",
                        kind=ContextItemKind.CONVERSATION_TURN,
                        content=(
                            "Assistant: I need to pin down the source and check current DM "
                            f"memory attempt {index}."
                        ),
                        conversation_id="D1",
                        retention=ContextItemRetention.RECENT,
                    )
                    for index in range(8)
                ),
            ),
        }
    )

    query = memory_recovery_query(request)

    assert query == "project borealis"
    assert "attempt" not in query


@pytest.mark.asyncio
async def test_an_empty_memory_result_is_answered_by_the_model_not_the_harness() -> None:
    """A gateway used to short-circuit this and write the reply itself.

    _RequiredMemorySearchGateway intercepted an empty memory.search and returned
    a fixed sentence without consulting the model at all. That is the harness
    authoring the user's answer: it cannot explain *why* nothing matched, adapt
    the phrasing to the question, or decide that another route is worth trying.
    The gateway is gone, so an empty result is simply an observation the model
    reasons about like any other.
    """

    empty_search = Observation(
        id="obs-empty-memory",
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        run_id="run-1",
        tool_call_id="memory-call-1",
        kind="memory.search",
        data={"selected_count": 0, "items": []},
        source=SourceRef(provider="leo-memory", reference="authorized-search"),
        observed_at=NOW,
        raw_hash="a" * 64,
    )
    delegate = _RecordingGateway(
        [CompletionProposal(answer="I have nothing recorded for this channel yet.")]
    )
    request = _request(1, observations=(empty_search,))

    result = await delegate.decide(request)

    assert isinstance(result.decision, CompletionProposal)
    assert result.decision.answer == "I have nothing recorded for this channel yet."
    # The model was actually asked, rather than bypassed.
    assert delegate.requests == [request]


def test_envelope_audit_identity_is_content_free_stable_and_inspectable() -> None:
    objective = "A private objective phrase that must not enter the trace identity."
    envelope = ElasticDeliberationPolicy().assess(objective)

    source_id = envelope.audit_source_id()

    assert objective not in source_id
    assert f"recommended={envelope.recommended_mode.value}" in source_id
    assert f"depth={envelope.minimum_depth}-{envelope.maximum_depth}" in source_id
    assert f"reason={envelope.reason_code}" in source_id
    assert source_id == ElasticDeliberationPolicy().assess(objective).audit_source_id()
