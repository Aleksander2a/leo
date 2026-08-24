"""Elastic but bounded deliberation policy for conversational work.

The deterministic policy owns a minimum/maximum depth envelope and hard safety
constraints. Its recommended mode is advisory: the model still chooses a direct
answer, clarification, tool read(s), plan, or delegation from the full admitted
context and policy-eligible catalog. The wrapper validates that semantic proposal
and prevents evidence-free decision loops without making hidden provider calls.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum

from pydantic import JsonValue

from leo.harness.models import (
    ClaimKind,
    CompletionContract,
    CompletionProposal,
    ContextItemKind,
    ContextItemRetention,
    ContractModel,
    EvidenceQuality,
    ModelRequest,
    ModelTurnResult,
    NonEmptyStr,
    Observation,
    ObservationStatus,
    ToolChoiceMode,
    ToolEffect,
    ToolRequest,
    ToolRequests,
)
from leo.harness.ports import ModelGateway, ModelGatewayError
from leo.harness.terminal_quality import contains_future_action_promise
from leo.harness.web_research import rank_tavily_result_urls


class DeliberationMode(StrEnum):
    DIRECT = "direct"
    CLARIFY = "clarify"
    CONTEXT_MEMORY = "context_memory"
    SINGLE_TOOL = "single_tool"
    MULTI_SOURCE = "multi_source"
    PARALLEL_READS = "parallel_reads"
    DELEGATE = "delegate"
    PLAN = "plan"
    REPLAN_VERIFY = "replan_verify"


# How many times a run may be bounced back to the model for choosing a route the
# advisory envelope did not expect, before the model's own judgement is accepted.
_MAX_ROUTE_NUDGES = 1

_ALL_MODES = frozenset(DeliberationMode)
_DEPTH_BY_MODE = {
    DeliberationMode.DIRECT: 0,
    DeliberationMode.CLARIFY: 0,
    DeliberationMode.CONTEXT_MEMORY: 1,
    DeliberationMode.SINGLE_TOOL: 2,
    DeliberationMode.MULTI_SOURCE: 3,
    DeliberationMode.PARALLEL_READS: 4,
    DeliberationMode.DELEGATE: 4,
    DeliberationMode.PLAN: 5,
    DeliberationMode.REPLAN_VERIFY: 6,
}


class DeliberationSignals(ContractModel):
    word_count: int
    clause_count: int
    requested_outputs: int
    entity_count: int
    independent_evidence_count: int
    external_evidence_required: bool
    freshness_required: bool
    memory_recall_required: bool
    state_mutation_required: bool
    context_available: bool
    contextual_followup: bool
    # The request names something it cannot resolve; research cannot help.
    unresolved_reference: bool
    open_ended_current_event: bool
    ambiguous: bool
    action_risk: bool
    explicit_tool_free: bool
    complexity_score: int


class DeliberationEnvelope(ContractModel):
    """Hard bounds plus a non-authoritative model-facing recommendation."""

    minimum_depth: int
    maximum_depth: int
    allowed_modes: frozenset[DeliberationMode]
    recommended_mode: DeliberationMode
    recommended_depth: int
    signals: DeliberationSignals
    hard_disable_tools: bool = False
    hard_require_clarification: bool = False
    hard_required_parent_tool: NonEmptyStr | None = None
    reason_code: NonEmptyStr
    reason: NonEmptyStr

    @property
    def mode(self) -> DeliberationMode:
        """Compatibility spelling; this value is advisory, never execution authority."""

        return self.recommended_mode

    @property
    def depth(self) -> int:
        return self.recommended_depth

    @property
    def required_parent_tool(self) -> str | None:
        """Return only an explicit user-requested effect, never an advisory route."""

        return self.hard_required_parent_tool

    def instruction(self) -> str:
        allowed = ",".join(sorted(item.value for item in self.allowed_modes))
        required = (
            f" Required effect tool: {self.hard_required_parent_tool}."
            if self.hard_required_parent_tool is not None
            else ""
        )
        return (
            f"Depth envelope {self.minimum_depth}-{self.maximum_depth}; advisory "
            f"{self.recommended_mode.value}; allowed [{allowed}]. Choose from admitted context "
            f"and eligible tools; act now or ask for missing input, never promise future work; "
            f"do not repeat without new evidence.{required}"
        )

    def audit_source_id(self) -> str:
        """Content-free replay identity persisted through the context source manifest."""

        signal_payload = self.signals.model_dump(mode="json")
        signal_digest = hashlib.sha256(
            json.dumps(signal_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        return (
            "deliberation-v1:"
            f"recommended={self.recommended_mode.value}:"
            f"depth={self.minimum_depth}-{self.maximum_depth}:"
            f"effect={self.hard_required_parent_tool or 'none'}:"
            f"reason={self.reason_code}:signals={signal_digest}"
        )


# Compatibility alias for callers that used the initial slice name.
DeliberationDecision = DeliberationEnvelope


class ElasticDeliberationPolicy:
    """Build a safety envelope from independent request and authority signals."""

    def assess(
        self,
        objective: str,
        *,
        context_item_count: int = 0,
        memory_recall_required: bool = False,
        state_mutation_required: bool = False,
        evidence_tool_names: tuple[str, ...] = (),
        external_evidence_required: bool = False,
        explicit_tool_free: bool = False,
        available_tool_names: frozenset[str] = frozenset(),
    ) -> DeliberationEnvelope:
        signals = _signals(
            objective,
            context_item_count=context_item_count,
            memory_recall_required=memory_recall_required,
            state_mutation_required=state_mutation_required,
            evidence_tool_names=evidence_tool_names,
            external_evidence_required=external_evidence_required,
            explicit_tool_free=explicit_tool_free,
        )
        parent_plan_available = "agent.execute_research_plan" in available_tool_names
        delegation_available = "agent.delegate_research" in available_tool_names

        # These are the only prompt-derived hard route constraints. Everything
        # else remains a recommendation that the semantic model may override.
        if explicit_tool_free:
            clarification = signals.ambiguous or _asks_for_clarification(objective)
            return _envelope(
                recommended=(
                    DeliberationMode.CLARIFY if clarification else DeliberationMode.DIRECT
                ),
                minimum_depth=0,
                maximum_depth=1,
                allowed_modes=(
                    frozenset({DeliberationMode.CLARIFY})
                    if clarification
                    else frozenset({DeliberationMode.DIRECT, DeliberationMode.CLARIFY})
                ),
                signals=signals,
                hard_disable_tools=True,
                hard_require_clarification=clarification,
                reason_code="explicit_tool_free",
                reason="the user explicitly disabled tools and external research",
            )
        if (
            signals.ambiguous
            and not signals.context_available
            and not signals.memory_recall_required
            and not signals.state_mutation_required
        ):
            # Advisory, not binding. This branch used to pin the envelope to
            # CLARIFY only *and* disable every tool, on the strength of a regex
            # over the prompt. "What are some interesting investing opportunities
            # currently?" tripped it, so Leo was forbidden from researching the
            # one thing it was asked about and could only ask a question back.
            #
            # Asking for missing detail is often right, so it stays the
            # recommendation -- but the model can see the request and the thread,
            # and the regex cannot. If it judges that it can research or answer,
            # it may.
            if signals.unresolved_reference:
                # The subject itself is unknown ("Compare these", "what does that
                # mean?"). No tool resolves that, so clarifying stays binding.
                return _envelope(
                    recommended=DeliberationMode.CLARIFY,
                    minimum_depth=0,
                    maximum_depth=0,
                    allowed_modes=frozenset({DeliberationMode.CLARIFY}),
                    signals=signals,
                    hard_disable_tools=True,
                    hard_require_clarification=True,
                    reason_code="unresolved_reference",
                    reason="the request refers to something it does not name",
                )
            return _envelope(
                recommended=DeliberationMode.CLARIFY,
                minimum_depth=0,
                maximum_depth=_DEPTH_BY_MODE[DeliberationMode.MULTI_SOURCE],
                allowed_modes=frozenset(
                    {
                        DeliberationMode.CLARIFY,
                        DeliberationMode.DIRECT,
                        DeliberationMode.CONTEXT_MEMORY,
                        DeliberationMode.SINGLE_TOOL,
                        DeliberationMode.MULTI_SOURCE,
                    }
                ),
                signals=signals,
                reason_code="missing_antecedent_or_arguments",
                reason="the outcome lacks a required antecedent or argument",
            )

        # An unambiguous request to *perform* orchestration is an explicit effect,
        # not a lexical routing hint. The parent tool is therefore mandatory, while
        # ordinary prompts remain free to choose orchestration semantically.
        explicit_orchestration = _explicit_orchestration_intent(objective)
        if explicit_orchestration == "plan" and parent_plan_available:
            return _envelope(
                recommended=DeliberationMode.PLAN,
                minimum_depth=_DEPTH_BY_MODE[DeliberationMode.PLAN],
                maximum_depth=_DEPTH_BY_MODE[DeliberationMode.REPLAN_VERIFY],
                allowed_modes=frozenset(
                    {
                        DeliberationMode.DIRECT,
                        DeliberationMode.PLAN,
                        DeliberationMode.REPLAN_VERIFY,
                    }
                ),
                signals=signals,
                hard_required_parent_tool="agent.execute_research_plan",
                reason_code="explicit_plan_effect",
                reason="the user explicitly requested execution of a parent research plan",
            )
        if explicit_orchestration == "delegate" and delegation_available:
            return _envelope(
                recommended=DeliberationMode.DELEGATE,
                minimum_depth=_DEPTH_BY_MODE[DeliberationMode.DELEGATE],
                maximum_depth=_DEPTH_BY_MODE[DeliberationMode.PLAN],
                allowed_modes=frozenset({DeliberationMode.DIRECT, DeliberationMode.DELEGATE}),
                signals=signals,
                hard_required_parent_tool="agent.delegate_research",
                reason_code="explicit_delegation_effect",
                reason="the user explicitly requested delegated execution",
            )

        recommended, reason_code, reason = _recommendation(
            objective,
            signals=signals,
            parent_plan_available=parent_plan_available,
            delegation_available=delegation_available,
        )
        minimum_depth = 0
        if signals.independent_evidence_count >= 2:
            minimum_depth = 2
        elif (
            signals.independent_evidence_count == 1
            or memory_recall_required
            or state_mutation_required
            or external_evidence_required
        ):
            minimum_depth = 1
        # The maximum is an authority/capability ceiling, not a lexical guess at
        # task complexity. This leaves an admitted model free to discover that a
        # natural prompt needs tools or orchestration while keeping the fallback
        # recommendation as shallow as possible.
        capability_ceiling = 1
        if available_tool_names or evidence_tool_names or external_evidence_required:
            capability_ceiling = _DEPTH_BY_MODE[DeliberationMode.PARALLEL_READS]
        if delegation_available:
            capability_ceiling = max(capability_ceiling, _DEPTH_BY_MODE[DeliberationMode.DELEGATE])
        if parent_plan_available:
            capability_ceiling = _DEPTH_BY_MODE[DeliberationMode.REPLAN_VERIFY]
        maximum_depth = min(6, max(capability_ceiling, minimum_depth + 1))
        return _envelope(
            recommended=recommended,
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
            allowed_modes=_ALL_MODES,
            signals=signals,
            reason_code=reason_code,
            reason=reason,
        )


class ElasticDeliberationGateway:
    """Validate semantic model proposals and fail closed on no-progress loops."""

    def __init__(
        self,
        delegate: ModelGateway,
        envelope: DeliberationEnvelope,
        *,
        max_no_progress_turns: int = 2,
    ) -> None:
        if max_no_progress_turns < 1:
            raise ValueError("max_no_progress_turns must be positive")
        self._delegate = delegate
        self._envelope = envelope
        self._max_no_progress_turns = max_no_progress_turns
        self._last_observation_signature: tuple[str, ...] | None = None
        self._last_decision_fingerprint: str | None = None
        self._last_iteration: int | None = None
        self._last_feedback_count = 0
        self._same_decision_count = 0
        self._no_progress_turns = 0
        self._last_result: ModelTurnResult | None = None
        self._recommended_mode = envelope.recommended_mode
        self._recommended_depth = envelope.recommended_depth
        self._route_nudges = 0

    @property
    def decision(self) -> DeliberationEnvelope:
        return self._envelope.model_copy(
            update={
                "recommended_mode": self._recommended_mode,
                "recommended_depth": self._recommended_depth,
            }
        )

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        observation_signature = tuple(sorted({item.raw_hash for item in request.observations}))
        progressed = (
            self._last_observation_signature is None
            or observation_signature != self._last_observation_signature
        )
        if progressed:
            self._no_progress_turns = 0
        elif request.verifier_feedback:
            self._no_progress_turns += 1
            self._escalate_recommendation()
            if self._no_progress_turns > self._max_no_progress_turns:
                # The harness used to author an answer here -- pasting a raw Exa
                # page dump over the model's decision, or synthesizing names from
                # a fixed list to satisfy a format check. Both produced replies no
                # model wrote: one shipped scraped page markup to Slack as the
                # final answer, the other invented content outright.
                #
                # A stuck loop is a real failure and is now reported as one. The
                # run still delivers the best answer the *model* actually wrote,
                # through the coordinator's best-effort path, so nothing useful
                # is lost -- only the fabricated substitutes are.
                raise ModelGatewayError(
                    "deliberation_no_progress",
                    "The bounded reasoning loop made no evidence progress.",
                    fallback_answer=_last_proposal_answer(self._last_result),
                )

        guided_request = request.model_copy(
            update={
                "completion_contract": request.completion_contract.model_copy(
                    update={"guidance": self._guided_contract(request.completion_contract)}
                )
            }
        )
        # A provider failure is a provider failure. Manufacturing a clarifying
        # question here hid real outages behind a question the user could not
        # usefully answer, and the coordinator already has a bounded retry path.
        result = await self._delegate.decide(guided_request)

        # _recover_required_clarification used to run here, replacing the model's
        # answer with "Which specific target, options, or missing details should
        # I use?" whenever an ambiguity regex fired. It fired on anything under
        # four words, so "CoinGecko BTC price" -- a perfectly clear request --
        # had its real answer thrown away and replaced by a canned question. If a
        # turn genuinely needs clarification, the model asks for it in its own
        # words, and can ask something far more specific than a fixed sentence.
        result = _recover_non_terminal_deferral(result, request, self._envelope)
        result = _drop_unsourced_optional_claims(result, request)

        proposed_mode = _semantic_mode(result, request)
        if proposed_mode not in self._envelope.allowed_modes:
            # The envelope is advisory by construction (see the module docstring):
            # it is derived from regexes over the prompt, not from knowledge of
            # what the turn actually needs. Killing the run over it let a keyword
            # miss veto a correct answer. Nudge once so a genuinely under-reasoned
            # turn gets a second chance, then defer to the model's own judgement.
            # Evidence sufficiency is still enforced downstream by the verifier,
            # which reasons about observations rather than about wording.
            if self._route_nudges < _MAX_ROUTE_NUDGES:
                self._route_nudges += 1
                raise ModelGatewayError(
                    "deliberation_mode_outside_envelope",
                    "The model proposed a reasoning mode outside the trusted depth envelope.",
                )
        proposed_depth = _semantic_depth(result, request, proposed_mode)
        deferred_to_required_effect_gate = (
            self._envelope.hard_required_parent_tool is not None
            and isinstance(result.decision, CompletionProposal)
        )
        if (
            proposed_depth < self._envelope.minimum_depth
            and proposed_mode is not DeliberationMode.CLARIFY
            and not deferred_to_required_effect_gate
            and self._route_nudges < _MAX_ROUTE_NUDGES
        ):
            self._route_nudges += 1
            raise ModelGatewayError(
                "deliberation_depth_below_minimum",
                "The model proposed reasoning shallower than the trusted depth envelope.",
            )
        if proposed_depth > self._envelope.maximum_depth:
            raise ModelGatewayError(
                "deliberation_depth_exceeded",
                "The model proposed reasoning deeper than the trusted depth envelope.",
            )

        self._last_result = result
        fingerprint = _decision_fingerprint(result)
        repeated = (
            self._last_decision_fingerprint == fingerprint
            and self._last_observation_signature == observation_signature
            and self._last_iteration is not None
            and request.iteration > self._last_iteration
        )
        bounded_retry = repeated and self._allows_one_retryable_read(result, request)
        if repeated and not bounded_retry:
            # Same removal as the no-progress branch above: the harness does not
            # get to write the user's answer. A repeated decision is surfaced as
            # the failure it is, and the best model-authored answer still ships
            # through the coordinator's best-effort path.
            raise ModelGatewayError(
                "deliberation_repeated_decision",
                "The model repeated a decision without adding evidence.",
                fallback_answer=_last_proposal_answer(result),
            )
        self._same_decision_count = self._same_decision_count + 1 if repeated else 1
        self._last_observation_signature = observation_signature
        self._last_decision_fingerprint = fingerprint
        self._last_iteration = request.iteration
        self._last_feedback_count = len(request.verifier_feedback)
        return result

    def _guided_contract(self, contract: CompletionContract) -> str:
        return apply_deliberation_guidance(contract.guidance, self.decision)

    def _allows_one_retryable_read(
        self,
        result: ModelTurnResult,
        request: ModelRequest,
    ) -> bool:
        """Allow one exact retry only after the coordinator advanced a retryable read.

        The coordinator advances to another model turn after a tool failure only when
        the typed ``ToolFailure.retryable`` flag is true.  A repeated single-tool
        decision with no new observation and newly appended feedback is therefore the
        retry turn.  The advertised retry policy remains the server-owned upper bound;
        a third identical decision is always rejected as no progress.
        """

        if (
            self._same_decision_count != 1
            or len(request.verifier_feedback) <= self._last_feedback_count
            or not isinstance(result.decision, ToolRequests)
            or len(result.decision.calls) != 1
        ):
            return False
        call = result.decision.calls[0]
        matching = tuple(tool for tool in request.tools if tool.name == call.name)
        return len(matching) == 1 and matching[0].retry.max_attempts > 1

    def _escalate_recommendation(self) -> None:
        candidates = {
            DeliberationMode.DIRECT: DeliberationMode.CLARIFY,
            DeliberationMode.CLARIFY: DeliberationMode.CLARIFY,
            DeliberationMode.CONTEXT_MEMORY: DeliberationMode.SINGLE_TOOL,
            DeliberationMode.SINGLE_TOOL: DeliberationMode.MULTI_SOURCE,
            DeliberationMode.MULTI_SOURCE: DeliberationMode.PARALLEL_READS,
            DeliberationMode.PARALLEL_READS: DeliberationMode.PLAN,
            DeliberationMode.DELEGATE: DeliberationMode.REPLAN_VERIFY,
            DeliberationMode.PLAN: DeliberationMode.REPLAN_VERIFY,
            DeliberationMode.REPLAN_VERIFY: DeliberationMode.REPLAN_VERIFY,
        }
        candidate = candidates[self._recommended_mode]
        if (
            candidate in self._envelope.allowed_modes
            and _DEPTH_BY_MODE[candidate] <= self._envelope.maximum_depth
        ):
            self._recommended_mode = candidate
        self._recommended_depth = min(
            self._envelope.maximum_depth,
            max(self._recommended_depth + 1, _DEPTH_BY_MODE[self._recommended_mode]),
        )


def _last_proposal_answer(result: ModelTurnResult | None) -> str | None:
    """Extract a salvageable answer text from the last decision, if any.

    Used when the bounded repair loop is about to give up: a self-contained answer
    the model already produced is a better outcome for the user than no answer at
    all, even though it never satisfied the verifier or the no-progress bound.

    Only a claim-free answer is eligible. A claim is an unverified evidentiary
    assertion (a source citation or inference); delivering one without ever
    passing it through the verifier would let a fabricated or ungrounded claim
    reach the user, which is a correctness failure, not merely a formatting one.
    A plain prose answer ("I need current provider evidence to answer.") carries
    no such risk and is always safe to salvage.
    """

    if (
        result is None
        or not isinstance(result.decision, CompletionProposal)
        or result.decision.claims
    ):
        return None
    answer = result.decision.answer.strip()
    return answer or None


def _drop_unsourced_optional_claims(
    result: ModelTurnResult,
    request: ModelRequest,
) -> ModelTurnResult:
    """Remove citations that cannot be valid in a context-only turn.

    Context items are selected server-side and are deliberately not Observation
    records. If a provider cites a context-item ID while no observations exist and
    source claims are optional, that is a format mistake rather than new evidence.
    Discard those unsupported citations before they create a verifier retry loop.
    """

    if not isinstance(result.decision, CompletionProposal):
        return result
    contract = request.completion_contract
    if request.observations or contract.source_claim_count.minimum > 0:
        return result
    valid_observation_ids = {observation.id for observation in request.observations}
    claims = tuple(
        claim
        for claim in result.decision.claims
        if claim.kind is ClaimKind.INFERENCE
        and (
            not claim.observation_ids
            or all(
                observation_id in valid_observation_ids for observation_id in claim.observation_ids
            )
        )
    )
    if claims == result.decision.claims:
        return result
    return result.model_copy(
        update={"decision": result.decision.model_copy(update={"claims": claims})}
    )


def apply_deliberation_guidance(base_guidance: str, envelope: DeliberationEnvelope) -> str:
    """Bind the initial envelope to the durable completion-contract manifest."""

    suffix = f" Deliberation: {envelope.instruction()}"
    base = re.sub(r"\s+Deliberation:.*$", "", base_guidance).rstrip()
    available = 500 - len(suffix)
    if available < 1:
        raise ModelGatewayError(
            "deliberation_guidance_invalid",
            "The bounded depth guidance exceeds the completion contract.",
        )
    return f"{base[:available].rstrip()}{suffix}"


def _envelope(
    *,
    recommended: DeliberationMode,
    minimum_depth: int,
    maximum_depth: int,
    allowed_modes: frozenset[DeliberationMode],
    signals: DeliberationSignals,
    reason_code: str,
    reason: str,
    hard_disable_tools: bool = False,
    hard_require_clarification: bool = False,
    hard_required_parent_tool: str | None = None,
) -> DeliberationEnvelope:
    return DeliberationEnvelope(
        minimum_depth=minimum_depth,
        maximum_depth=maximum_depth,
        allowed_modes=allowed_modes,
        recommended_mode=recommended,
        recommended_depth=_DEPTH_BY_MODE[recommended],
        signals=signals,
        hard_disable_tools=hard_disable_tools,
        hard_require_clarification=hard_require_clarification,
        hard_required_parent_tool=hard_required_parent_tool,
        reason_code=reason_code,
        reason=reason,
    )


def _recommendation(
    objective: str,
    *,
    signals: DeliberationSignals,
    parent_plan_available: bool,
    delegation_available: bool,
) -> tuple[DeliberationMode, str, str]:
    if (
        signals.memory_recall_required
        or signals.state_mutation_required
        or signals.contextual_followup
    ):
        return (
            DeliberationMode.CONTEXT_MEMORY,
            "admitted_context_followup",
            "admitted context or memory can resolve the follow-up",
        )
    if signals.open_ended_current_event and signals.external_evidence_required:
        return (
            DeliberationMode.MULTI_SOURCE,
            "open_current_event",
            "current-event discovery and selected evidence are recommended",
        )
    if signals.independent_evidence_count >= 2:
        recommended = (
            DeliberationMode.PLAN
            if signals.complexity_score >= 5 and parent_plan_available
            else DeliberationMode.PARALLEL_READS
        )
        return (
            recommended,
            "multiple_evidence_obligations",
            "independent evidence can be read in parallel and reconciled",
        )
    if signals.independent_evidence_count == 1 or signals.external_evidence_required:
        return (
            DeliberationMode.SINGLE_TOOL,
            "fresh_evidence_candidate",
            "one eligible evidence read is the shallowest likely route",
        )
    if signals.action_risk and delegation_available:
        return (
            DeliberationMode.CLARIFY,
            "consequential_action_review",
            "clarification is recommended before a consequential action",
        )
    return (
        DeliberationMode.DIRECT,
        "context_sufficient",
        "the admitted context appears sufficient for a direct response",
    )


def _open_ended_investment_input_missing(objective: str) -> bool:
    normalized = " ".join(objective.casefold().split())
    investment_domain = re.search(
        r"\b(?:invest(?:ing|ment|ments)?|stocks?|equities|crypto(?:currencies)?|"
        r"bonds?|etfs?|funds?|portfolio|dividend)\b",
        normalized,
    )
    recommendation = re.search(
        r"\b(?:opportunit(?:y|ies)|ideas?|interesting|recommend(?:ation|ations)?|"
        r"suggest(?:ion|ions)?|what should i (?:buy|invest in)|where should i invest)\b",
        normalized,
    )
    preference = re.search(
        r"\b(?:risk|horizon|short[- ]term|long[- ]term|income|dividend|growth|value|"
        r"turnaround|safe|conservative|aggressive|speculative|sector|industry|technology|"
        r"tech|healthcare|energy|utilities|financials|real estate|region|country|u\.?s\.?|"
        r"europe|asia|emerging markets?|small[- ]cap|mid[- ]cap|large[- ]cap|budget|"
        r"capital|amount)\b",
        normalized,
    )
    return investment_domain is not None and recommendation is not None and preference is None


def _has_unresolved_antecedent(objective: str, normalized: str) -> bool:
    """Identify a missing deictic target without treating ordinary grammar as context.

    Words such as ``that`` are often relative pronouns (``a project that helps teams``),
    and ``this`` is commonly followed by source text in an inline transformation request.
    Treating every occurrence as an unresolved placeholder made complete creative and
    rewriting prompts collapse into the generic clarification fallback.  This check is
    deliberately role-aware: it recognizes only a bounded set of direct-object and
    follow-up shapes, and regards an explicitly supplied inline payload as its antecedent.
    """

    if _has_inline_source_payload(objective):
        return False

    deictic = r"(?:it|this|that|them|these|those)"
    direct_object = re.search(
        rf"\b(?:analy[sz]e|assess|approve|buy|cancel|change|choose|compare|contrast|"
        rf"delete|edit|evaluate|explain|fix|publish|rank|remove|review|rewrite|sell|send|"
        rf"summari[sz]e|translate|use)\s+(?:(?:all|both)\s+of\s+)?{deictic}\b",
        normalized,
    )
    if direct_object is not None or re.search(r"\bdo\s+it\b", normalized) is not None:
        return True
    if re.search(
        rf"^(?:please\s+)?(?:make|keep)\s+{deictic}\b|"
        rf"^(?:please\s+)?{deictic}(?:\s+again)?[.!?]*$|"
        rf"\b(?:what\s+about|which\s+of)\s+{deictic}\b|"
        rf"\b(?:does|is|was|are|were|can|could|would|should)\s+{deictic}\b|"
        rf"\b{deictic}\s+again\b",
        normalized,
    ):
        return True
    return "the thing" in normalized


def _has_inline_source_payload(objective: str) -> bool:
    transform = (
        r"(?:analy[sz]e|compare|contrast|edit|explain|proofread|rephrase|review|rewrite|"
        r"summari[sz]e|translate)"
    )
    if re.search(
        rf"\b{transform}\s+(?:this|that|it|these|those|them)?\s*[\"'\u2018\u201c]"
        rf"[^\"'\u2019\u201d\r\n]+[\"'\u2019\u201d]",
        objective,
        flags=re.IGNORECASE,
    ):
        return True
    prefix, separator, payload = objective.partition(":")
    if not separator or not payload.strip():
        return False
    return bool(
        re.search(rf"\b{transform}\b", prefix, flags=re.IGNORECASE)
        or re.search(
            r"\b(?:this|that)\s+(?:copy|message|paragraph|passage|sentence|text)\b",
            prefix,
            flags=re.IGNORECASE,
        )
    )


def _looks_like_self_contained_output_request(
    objective: str,
    normalized: str,
    *,
    unresolved_antecedent: bool,
) -> bool:
    """Keep short but complete creative/transform requests in the direct envelope."""

    if unresolved_antecedent:
        return False
    if _has_inline_source_payload(objective):
        return True
    creative_output = (
        r"(?:code[ -]?names?|codenames?|descriptions?|drafts?|emails?|headlines?|ideas?|"
        r"messages?|names?|outlines?|poems?|stories|slogans?|summaries|taglines?|titles?)"
    )
    if re.match(
        rf"^(?:please\s+)?(?:brainstorm|compose|create|draft|generate|give(?:\s+me)?|"
        rf"suggest|write)\b[^.!?]*\b{creative_output}\b",
        normalized,
    ):
        return True
    return bool(
        re.fullmatch(
            rf"(?:exactly\s+)?(?:one|two|three|four|five|six|seven|eight|nine|ten|"
            rf"\d{{1,2}})\s+(?:[a-z-]+\s+){{0,3}}{creative_output}[.!?]*",
            normalized,
        )
    )


def _signals(
    objective: str,
    *,
    context_item_count: int,
    memory_recall_required: bool,
    state_mutation_required: bool,
    evidence_tool_names: tuple[str, ...],
    external_evidence_required: bool,
    explicit_tool_free: bool,
) -> DeliberationSignals:
    normalized = " ".join(objective.casefold().split())
    words = re.findall(r"[^\W_]+(?:[-'][^\W_]+)?", objective, flags=re.UNICODE)
    clauses = tuple(
        item.strip()
        for item in re.split(r"[;.!?]+|\b(?:and then|then|while|whereas)\b", normalized)
        if item.strip()
    )
    output_markers = re.findall(
        r"\b(?:compare|contrast|assess|analy[sz]e|evaluate|explain|summari[sz]e|"
        r"recommend|rank|verify|reconcile|draft|build|calculate)\b",
        normalized,
    )
    symbol_entities = set(re.findall(r"\b[A-Z]{2,6}\b", objective)) - {
        "AI",
        "API",
        "EPS",
        "HTTP",
        "HTTPS",
        "SEC",
        "URL",
        "USD",
    }
    named_entities = set(
        re.findall(
            r"\b(?:apple|amazon|google|meta|microsoft|nvidia|tesla)\b",
            normalized,
        )
    )
    entity_count = len(symbol_entities | named_entities)
    freshness = bool(
        re.search(
            r"\b(?:current|currently|latest|live|newest|now|recent|recently|still|today|"
            r"this week)\b",
            normalized,
        )
    )
    action_risk = bool(
        re.search(
            r"\b(?:buy|sell|trade|execute|send|publish|delete|remove|transfer|approve|"
            r"deploy|write to|change|cancel)\b",
            normalized,
        )
    )
    conditional_analysis = bool(
        re.search(r"\b(?:assuming|if|scenario|sensitivity|under)\b", normalized)
    )
    unresolved_antecedent = _has_unresolved_antecedent(objective, normalized)
    unspecified = bool(re.search(r"\b(?:unspecified|not specified|unknown|tbd)\b", normalized))
    contextual_followup = _looks_like_contextual_followup(normalized, len(words))
    open_ended_investment_input_missing = (
        entity_count == 0
        and context_item_count == 0
        and _open_ended_investment_input_missing(objective)
    )
    capability_discovery_requested = bool(
        re.search(
            r"\b(?:appropriate|available|eligible)\s+(?:capability|tool)\b"
            r"|\b(?:find|identify|search for)\s+(?:an?\s+)?(?:capability|tool)\b",
            normalized,
        )
    )
    open_ended_current_event = bool(
        freshness
        and re.search(
            r"\b(?:what happened|what changed|what's new|what is new|latest developments?)\b",
            normalized,
        )
    )
    evidence_domains = {name.split(".", 1)[0] for name in evidence_tool_names}
    independent_evidence_count = max(len(evidence_tool_names), len(evidence_domains))
    clause_count = max(1, len(clauses))
    requested_outputs = len(set(output_markers))
    # A reference the request cannot resolve on its own -- "Compare these" with
    # no antecedent, "what does that mean?" with no thread. No amount of research
    # answers these, because the *subject* is unknown, so clarifying is the only
    # correct move and the envelope binds it.
    #
    # This is deliberately narrower than `ambiguous`, which also covers merely
    # open-ended or terse requests ("interesting investing opportunities?"). Those
    # have a knowable subject; treating them the same way is what stopped Leo from
    # researching the very thing it was asked about.
    unresolved_reference = bool(
        (unresolved_antecedent or contextual_followup)
        and context_item_count == 0
        and not evidence_tool_names
        and not _looks_like_self_contained_question(normalized)
        and not capability_discovery_requested
    )
    ambiguous = bool(
        unspecified
        or open_ended_investment_input_missing
        or unresolved_reference
        or (
            (
                (
                    len(words) <= 3
                    and not _looks_like_self_contained_output_request(
                        objective,
                        normalized,
                        unresolved_antecedent=unresolved_antecedent,
                    )
                )
                or unresolved_antecedent
            )
            and context_item_count == 0
            and not evidence_tool_names
            and not _looks_like_self_contained_question(normalized)
            and not capability_discovery_requested
        )
        or (contextual_followup and context_item_count == 0)
    )
    complexity_score = min(
        8,
        (1 if len(words) >= 24 else 0)
        + (1 if clause_count >= 2 else 0)
        + min(2, requested_outputs)
        + min(2, independent_evidence_count)
        + (1 if entity_count >= 2 else 0)
        + (1 if action_risk else 0)
        + (1 if conditional_analysis else 0)
        + (1 if freshness and external_evidence_required else 0),
    )
    return DeliberationSignals(
        word_count=len(words),
        clause_count=clause_count,
        requested_outputs=requested_outputs,
        entity_count=entity_count,
        independent_evidence_count=independent_evidence_count,
        external_evidence_required=external_evidence_required or bool(evidence_tool_names),
        freshness_required=freshness,
        memory_recall_required=memory_recall_required,
        state_mutation_required=state_mutation_required,
        context_available=context_item_count > 0,
        contextual_followup=contextual_followup,
        unresolved_reference=unresolved_reference,
        open_ended_current_event=open_ended_current_event,
        ambiguous=ambiguous,
        action_risk=action_risk,
        explicit_tool_free=explicit_tool_free,
        complexity_score=complexity_score,
    )


def _semantic_mode(result: ModelTurnResult, request: ModelRequest) -> DeliberationMode:
    decision = result.decision
    if isinstance(decision, CompletionProposal):
        return (
            DeliberationMode.CLARIFY
            if _is_genuine_clarification(decision)
            else DeliberationMode.DIRECT
        )
    names = tuple(call.name for call in decision.calls)
    if "agent.execute_research_plan" in names:
        return (
            DeliberationMode.REPLAN_VERIFY if request.verifier_feedback else DeliberationMode.PLAN
        )
    if "agent.delegate_research" in names:
        return DeliberationMode.DELEGATE
    if names and all(name.startswith("memory.") for name in names):
        return DeliberationMode.CONTEXT_MEMORY
    if len(names) > 1:
        return DeliberationMode.PARALLEL_READS
    return DeliberationMode.SINGLE_TOOL


def _semantic_depth(
    result: ModelTurnResult,
    request: ModelRequest,
    mode: DeliberationMode,
) -> int:
    if isinstance(result.decision, CompletionProposal):
        evidence_kinds = {item.kind for item in request.observations}
        if "agent.execute_research_plan" in evidence_kinds:
            return _DEPTH_BY_MODE[
                DeliberationMode.REPLAN_VERIFY
                if request.verifier_feedback
                else DeliberationMode.PLAN
            ]
        if "agent.delegate_research" in evidence_kinds:
            return _DEPTH_BY_MODE[DeliberationMode.DELEGATE]
        if len(evidence_kinds) >= 2 or len(result.decision.claims) >= 2:
            return 3
        if evidence_kinds or result.decision.claims:
            return 2
        if request.context_items:
            return 1
    return _DEPTH_BY_MODE[mode]


def _is_genuine_clarification(decision: CompletionProposal) -> bool:
    """Recognize one or more input-seeking questions, not a tagged-on question.

    A trailing ``Does that help?`` must not turn a substantive answer into a
    zero-depth clarification and thereby evade tool/evidence policy.  We therefore
    accept only a question-only response (optionally with a narrow clarification
    preamble), with no claims or harness-owned conclusion metadata.
    """

    if decision.claims or decision.affected_assumption or decision.uncertainty:
        return False
    segments = tuple(
        item.strip() for item in re.findall(r"[^.!?]+(?:[.!?]+|$)", decision.answer) if item.strip()
    )
    if not segments:
        return False
    question_count = 0
    for segment in segments:
        if not segment.endswith("?"):
            if _is_clarification_preamble(segment.rstrip(".!")):
                continue
            return False
        question = segment[:-1].strip()
        if ":" in question:
            prefix, remainder = question.split(":", 1)
            if _is_clarification_preamble(prefix):
                question = remainder.strip()
        if not _is_input_seeking_question(question):
            return False
        question_count += 1
    return question_count >= 1


def _recover_non_terminal_deferral(
    result: ModelTurnResult,
    request: ModelRequest,
    envelope: DeliberationEnvelope,
) -> ModelTurnResult:
    """Turn a model's research promise into bounded work or a real question.

    A completion that says Leo still needs to check/search cannot be terminal.  When
    a policy-selected web-search capability is advertised, the harness performs the
    next deterministic read.  If it is not available, Leo asks for genuinely missing
    source input instead of claiming that work will happen later.
    """

    decision = result.decision
    if (
        not isinstance(decision, CompletionProposal)
        or _is_genuine_clarification(decision)
        or not _is_non_terminal_deferral(decision.answer)
        or envelope.hard_required_parent_tool is not None
    ):
        return result

    advertised = frozenset(tool.name for tool in request.tools)
    required_name = request.tool_choice.required_tool_name
    if (
        request.tool_choice.mode is ToolChoiceMode.REQUIRED
        and required_name is not None
        and required_name not in advertised
    ):
        return result
    selected_url = selected_tavily_result_url(request)
    sealed_required_read = _sealed_required_read(request)
    # Every branch below turns a deferral into *work*: a tool call the model
    # should have made. None of them writes an answer. A branch that did -- a
    # canned "Which market or asset class, risk tolerance, and time horizon
    # should I focus on?" -- was removed with the rest of the harness-authored
    # replies: it fired on any open-ended investment question and threw away
    # whatever the model had actually said.
    repaired_decision: ToolRequests | CompletionProposal
    if sealed_required_read is not None:
        repaired_decision = sealed_required_read
        finish_reason = "tool_calls"
        model = "elastic-required-read-v1"
    elif selected_url is not None and "web.fetch_public_text" in advertised:
        repaired_decision = ToolRequests(
            calls=(
                ToolRequest(
                    id=f"elastic-fetch-{request.iteration}",
                    name="web.fetch_public_text",
                    arguments={"url": selected_url},
                ),
            )
        )
        finish_reason = "tool_calls"
        model = "elastic-research-fetch-v1"
    elif (next_search := _next_untried_search(request, advertised)) is not None:
        repaired_decision = ToolRequests(calls=(next_search,))
        finish_reason = "tool_calls"
        model = f"elastic-{next_search.name.replace('.', '-').replace('_', '-')}-v2"
    elif (
        "memory.search" in advertised
        and not any(item.kind == "memory.search" for item in request.observations)
        and (
            required_name == "memory.search"
            or re.search(r"\b(?:memory|recall|remember)\b", decision.answer.casefold()) is not None
        )
    ):
        repaired_decision = ToolRequests(
            calls=(
                ToolRequest(
                    id=f"elastic-memory-search-{request.iteration}",
                    name="memory.search",
                    arguments={"query": memory_recovery_query(request), "limit": 8},
                ),
            )
        )
        finish_reason = "tool_calls"
        model = "elastic-memory-search-v1"
    else:
        # Nothing left to try deterministically. Historically this branch emitted
        # "I'm missing a reliable source needed for that answer", which discarded
        # a real model answer and told the user nothing. The model's own proposal
        # is strictly more useful than that sentence, so keep it: the verifier
        # still gets to judge it, and terminal delivery still applies its own
        # caveats. Refusal is never manufactured here.
        return result
    return result.model_copy(
        update={
            "decision": repaired_decision,
            "provider": "leo-harness",
            "model": model,
            "finish_reason": finish_reason,
        }
    )


_SEARCH_LADDER: tuple[str, ...] = (
    "web.research_verified",
    "web.search_exa",
    "web.search_tavily",
    "web.search_public",
)


def _search_arguments(name: str, query: str) -> dict[str, JsonValue]:
    if name == "web.search_tavily":
        return {
            "query": query,
            "max_results": 5,
            "search_depth": "advanced",
            "topic": "general",
        }
    return {"query": query}


def _productive_observation(observation: Observation) -> bool:
    """Return whether a read actually yielded something worth reasoning over.

    A retrieved-but-empty search is not evidence that the web has been
    consulted. Treating it as such was what capped Leo at one search attempt per
    run: the next branch was skipped because an observation of that kind merely
    *existed*.
    """

    if observation.status is not ObservationStatus.RETRIEVED:
        return False
    for field in ("results", "statements", "highlights", "items"):
        value = observation.data.get(field)
        if isinstance(value, list) and value:
            return True
    text = observation.data.get("text")
    return isinstance(text, str) and len(text.strip()) > 40


def _next_untried_search(
    request: ModelRequest,
    advertised: frozenset[str],
) -> ToolRequest | None:
    """Pick the next advertised search provider that has not yet produced evidence.

    This replaces a mutually-exclusive if/elif chain in which Tavily was only
    reachable when Exa was *absent*, and in which any single attempt -- however
    empty -- suppressed every remaining provider. Failover is the point: if one
    provider returns nothing useful, try the next one before giving up.
    """

    productive = {
        observation.kind
        for observation in request.observations
        if _productive_observation(observation)
    }
    if productive.intersection(_SEARCH_LADDER):
        return None
    attempted = {observation.kind for observation in request.observations}
    query = " ".join(request.objective.split())[:400].strip()
    if len(query) < 2:
        return None
    for name in _SEARCH_LADDER:
        if name in advertised and name not in attempted:
            return ToolRequest(
                id=f"elastic-{name.replace('.', '-')}-{request.iteration}",
                name=name,
                arguments=_search_arguments(name, query),
            )
    return None


def _sealed_required_read(request: ModelRequest) -> ToolRequests | None:
    """Materialize only a trusted REQUIRED read whose full required input is sealed."""

    if request.tool_choice.mode is not ToolChoiceMode.REQUIRED:
        return None
    required_name = request.tool_choice.required_tool_name
    if required_name is None:
        return None
    matching = tuple(tool for tool in request.tools if tool.name == required_name)
    if len(matching) != 1 or matching[0].effect is not ToolEffect.READ:
        return None
    schema_required = matching[0].input_schema.get("required", [])
    schema_properties = matching[0].input_schema.get("properties", {})
    if (
        not isinstance(schema_required, list)
        or any(not isinstance(item, str) for item in schema_required)
        or not isinstance(schema_properties, dict)
    ):
        return None
    arguments = {item.name: item.value for item in request.tool_choice.required_arguments}
    if (
        any(value is None for value in arguments.values())
        or not set(schema_required).issubset(arguments)
        or not set(arguments).issubset(schema_properties)
    ):
        return None
    return ToolRequests(
        calls=(
            ToolRequest(
                id=f"elastic-required-read-{request.iteration}",
                name=required_name,
                arguments=arguments,
            ),
        )
    )


_SUBSTANTIVE_CONTENT = re.compile(
    # A number, a money/percent figure, a date, a ticker, or a proper noun --
    # anything indicating the answer carries content rather than only an apology.
    r"\d|\$|%|\b[A-Z]{2,6}\b|\b[A-Z][a-z]{2,}\b"
)


def answer_is_substantive(answer: str) -> bool:
    """Return whether an answer carries content a reader could actually use.

    Length alone is a poor signal (a long apology is still an apology), so this
    also requires at least one concrete token: a figure, a date, a symbol, or a
    proper noun.
    """

    stripped = answer.strip()
    if len(stripped.split()) < 12:
        return False
    return _SUBSTANTIVE_CONTENT.search(stripped) is not None


def _is_non_terminal_deferral(answer: str) -> bool:
    """Detect an answer that defers work instead of delivering it.

    Two deliberate narrowings relative to the original predicate:

    1. A *substantive* answer is never a deferral. Previously any answer
       containing "I don't have current data ..." was rewritten -- including
       "I don't have current market data for GOOG, but consensus year-end
       targets cluster around $210-$230", which is a genuinely useful reply.
       Rewriting those both discarded good work and applied steady pressure
       toward unhedged, overconfident prose.
    2. The missing-evidence phrasing must be the answer's *substance*, not an
       aside inside a real answer.

    A forward promise of work Leo will not actually do stays a deferral at any
    length: that one is a broken promise regardless of what surrounds it.
    """

    if contains_future_action_promise(answer):
        return True
    if answer_is_substantive(answer):
        return False
    normalized = " ".join(answer.casefold().replace("\u2019", "'").replace("\u2018", "'").split())
    missing_evidence = re.search(
        r"\b(?:i|we)\s+(?:do not|don't|cannot|can't)\s+(?:have|know)\b.{0,100}"
        r"\b(?:reliable|current|enough|source|information|evidence|detail)\b"
        r"|\b(?:i|we)\s+(?:still\s+)?(?:need|require)\b.{0,80}"
        r"\b(?:source|information|evidence|provider data|web access)\b"
        # "I'm missing a reliable source ..." -- the harness's own former canned
        # refusal, which its own classifier did not recognize as one.
        r"|\b(?:i'm|i am|we're|we are)\s+missing\b.{0,80}"
        r"\b(?:source|information|evidence|data|material)\b",
        normalized,
    )
    return missing_evidence is not None


def ranked_tavily_result_urls(request: ModelRequest) -> tuple[str, ...]:
    """Rank normalized Tavily results with a bounded authority preference."""

    for observation in reversed(request.observations):
        if (
            observation.kind != "web.search_tavily"
            or observation.status is not ObservationStatus.RETRIEVED
            or observation.quality is not EvidenceQuality.DISCOVERY_ONLY
        ):
            continue
        return rank_tavily_result_urls(observation.data.get("results"), request.objective)
    return ()


def selected_tavily_result_url(request: ModelRequest) -> str | None:
    """Return the highest-ranked normalized public result from Tavily."""

    return next(iter(ranked_tavily_result_urls(request)), None)


def memory_recovery_query(request: ModelRequest) -> str:
    """Build a concise authorized query anchored to material thread context.

    Ambiguous follow-ups often contain only words such as ``it`` or ``again``.  A
    reverse-recency query can then spend its token budget on failed/progress turns
    and omit the entity established by the root.  Prefer the pinned root and latest
    material assistant outcome selected by the trusted transcript classifier, then
    fall back to reverse chronology for callers without retention metadata.
    """

    stop = {
        "again",
        "answer",
        "and",
        "ask",
        "asked",
        "assistant",
        "about",
        "came",
        "conversation",
        "did",
        "from",
        "have",
        "leo",
        "memory",
        "remember",
        "source",
        "that",
        "the",
        "this",
        "was",
        "what",
        "which",
        "user",
    }
    eligible = tuple(
        item
        for item in request.context_items
        if item.kind in {ContextItemKind.CONVERSATION_TURN, ContextItemKind.MEMORY}
    )
    pinned_root = tuple(
        item for item in eligible if item.retention is ContextItemRetention.THREAD_ROOT
    )
    prior_outcomes = tuple(
        item for item in eligible if item.retention is ContextItemRetention.PRIOR_OUTCOME
    )
    root_anchor = pinned_root[:1] or eligible[:1]
    outcome_anchor = prior_outcomes[-1:] or eligible[-1:]

    def lexical_tokens(value: str) -> list[str]:
        return [
            normalized
            for token in re.findall(r"[^\W_]{2,64}(?:[.-][^\W_]{1,64})*", value, re.UNICODE)
            if (normalized := token.casefold()) not in stop
        ]

    # ``memory.search`` deliberately uses PostgreSQL ``plainto_tsquery`` (AND
    # semantics).  Search for the compact entity shared by the root and material
    # outcome instead of conjoining every word in the ambiguous follow-up.
    if root_anchor and outcome_anchor and root_anchor[0].id != outcome_anchor[0].id:
        outcome_tokens = frozenset(lexical_tokens(outcome_anchor[0].content))
        shared_anchor = tuple(
            dict.fromkeys(
                token for token in lexical_tokens(root_anchor[0].content) if token in outcome_tokens
            )
        )
        if shared_anchor:
            return " ".join(shared_anchor[:4])

    prioritized_ids = {item.id for item in (*pinned_root, *prior_outcomes)}
    sources = [request.objective]
    sources.extend(item.content for item in pinned_root)
    sources.extend(item.content for item in reversed(prior_outcomes))
    sources.extend(item.content for item in reversed(eligible) if item.id not in prioritized_ids)
    selected: list[str] = []
    for source in sources:
        for normalized in lexical_tokens(source):
            if normalized in selected:
                continue
            selected.append(normalized)
            if len(selected) == 6:
                return " ".join(selected)
    return " ".join(selected) or "thread context"


def _is_clarification_preamble(value: str) -> bool:
    normalized = " ".join(value.casefold().split()).strip("-—: ")
    return bool(
        re.fullmatch(
            r"(?:i (?:need|would need|am missing|don't have) (?:a little |some |two |three )?"
            r"(?:detail|details|information|context)|"
            r"to (?:answer|compare|complete|continue|help) (?:accurately|that|this)|"
            r"before i (?:answer|compare|continue)|"
            r"a (?:quick|brief) clarification)",
            normalized,
        )
    )


def _is_input_seeking_question(value: str) -> bool:
    normalized = " ".join(value.casefold().split()).strip("-—•* ")
    if not normalized or re.fullmatch(
        r"(?:does (?:this|that) help|does (?:this|that) make sense|"
        r"is that clear|sound good|okay|right)",
        normalized,
    ):
        return False
    if re.match(
        r"^(?:(?:and|also|alternatively)s+)?(?:"
        r"what|which|who|whom|whose|when|where|why|how|"
        r"do|does|did|is|are|was|were|can|could|would|should|will|have|has|may|"
        r"for\s+(?:what|which|how)|between\s+(?:what|which)|"
        r"over\s+(?:what|which)|during\s+(?:what|which))\b",
        normalized,
    ):
        return True
    # Elliptical alternatives such as "Public or private?" are genuine missing
    # input when the whole response is otherwise question-only.
    return len(normalized.split()) <= 12 and re.search(r"\b(?:or|versus)\b", normalized) is not None


def _looks_like_self_contained_question(normalized: str) -> bool:
    return bool(
        re.match(
            r"^(?:what|when|where|which|who|why|how|is|are|can|could|should|would)\b",
            normalized,
        )
        and not re.search(r"\b(?:it|that|them|these|those|this)\b", normalized)
    )


def _looks_like_contextual_followup(normalized: str, word_count: int) -> bool:
    if word_count > 16:
        return False
    return bool(
        re.fullmatch(r"why\??", normalized)
        or re.match(r"^(?:and\s+)?what about\b", normalized)
        or re.match(r"^(?:is|was|are|were)\s+(?:that|it|this)\b", normalized)
        or re.match(r"^(?:can|could|would)\s+you\s+(?:expand|explain|elaborate)\b", normalized)
        or re.search(r"\b(?:again|which conversation did (?:it|that|this) come from)\b", normalized)
    )


def _asks_for_clarification(objective: str) -> bool:
    normalized = objective.casefold()
    return "clarifying question" in normalized or "ask me" in normalized


def _explicit_orchestration_intent(objective: str) -> str | None:
    normalized = " ".join(objective.casefold().split())
    command_prefix = (
        r"(?:^|[.!?]\s+)(?:please\s+)?(?:now\s+)?"
        r"|\b(?:i\s+(?:want|need)\s+you\s+to|you\s+must|go\s+ahead\s+and|"
        r"(?:can|could|would|will)\s+you(?:\s+please)?)\s+"
    )
    plan_command = re.search(
        rf"(?:{command_prefix})(?:build|coordinate|execute|run)\b",
        normalized,
    )
    if plan_command and re.search(r"\b(?:multi-step|parallel|plan|workflow)\b", normalized):
        return "plan"
    delegate_command = re.search(
        rf"(?:{command_prefix})(?:delegate|run)\b",
        normalized,
    )
    if delegate_command and re.search(r"\b(?:delegate|delegated|subagent)\b", normalized):
        return "delegate"
    return None


def _decision_fingerprint(result: ModelTurnResult) -> str:
    decision = result.decision
    if isinstance(decision, ToolRequests):
        payload: object = {
            "kind": decision.kind,
            "calls": [{"name": call.name, "arguments": call.arguments} for call in decision.calls],
        }
    else:
        payload = decision.model_dump(mode="json")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
