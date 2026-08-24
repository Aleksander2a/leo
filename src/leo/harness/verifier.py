"""Deterministic completion verification owned by the harness."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation

from pydantic import JsonValue

from leo.harness.child_evidence import (
    ChildEvidenceEnvelope,
    ChildEvidenceError,
    child_evidence_expires_at,
    parse_child_evidence_envelope,
)
from leo.harness.crypto_market import (
    ground_crypto_aggregate_snapshot,
    ground_crypto_provider_snapshot,
)
from leo.harness.earnings import canonical_earnings_statements
from leo.harness.equity_market import (
    EQUITY_PROFILE_PROVIDERS,
    EQUITY_SEARCH_PROVIDERS,
    canonical_equity_profile_statements,
    canonical_equity_quote_disagreement_statement,
    canonical_equity_quote_statement,
    canonical_equity_quote_time_skew_statement,
    canonical_equity_search_statements,
    valid_equity_observed_at,
    valid_equity_profile_provenance,
    valid_equity_quote_aggregate,
    valid_equity_quote_provenance,
    valid_equity_search_provenance,
)
from leo.harness.exa_search import canonical_exa_highlight_statements, exa_result_hash
from leo.harness.models import (
    Claim,
    ClaimKind,
    CompletionContract,
    CompletionProposal,
    EvidenceQuality,
    EvidenceToolRequirement,
    Observation,
    ObservationStatus,
    RunBundle,
    SourceRef,
    VerificationOutcome,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
    constrained_values_match,
)
from leo.harness.ports import Clock, IdGenerator
from leo.harness.provider_canonical import (
    canonical_finnhub_profile_statements,
)
from leo.harness.research import (
    ResearchClaim,
    ResearchProposal,
    ResearchRequirement,
    verify_research,
)
from leo.harness.terminal_quality import (
    completed_research_action_claim,
    contains_future_action_promise,
)
from leo.harness.web_research import valid_verified_web_attempts
from leo.url_policy import is_public_https_url

GroundingRule = Callable[[str, str, Observation], tuple[bool, str]]
_ClaimAwareGroundingRule = Callable[[ClaimKind, str, str, Observation], tuple[bool, str]]


@dataclass(frozen=True)
class _NestedPlanEvidence:
    """One exact, digest-validated child source projected through its parent row."""

    parent_observation_id: str
    child_run_id: str
    claim_id: str
    statement: str
    observation: Observation


@dataclass(frozen=True)
class _ValidatedResearchPlan:
    node_answers: tuple[str, ...]
    nested_evidence: tuple[_NestedPlanEvidence, ...]


_DEFAULT_GROUNDING_RULES: dict[str, _ClaimAwareGroundingRule] = {
    "market.get_quote": lambda _kind, statement, answer, observation: _ground_quote(
        statement, answer, observation
    ),
    "market.get_quote_alpha_vantage": (
        lambda _kind, statement, answer, observation: _ground_equity_provider_quote(
            statement, answer, observation
        )
    ),
    "market.get_quote_finnhub": (
        lambda _kind, statement, answer, observation: _ground_equity_provider_quote(
            statement, answer, observation
        )
    ),
    "market.get_quote_massive": (
        lambda _kind, statement, answer, observation: _ground_equity_provider_quote(
            statement, answer, observation
        )
    ),
    "market.get_quote_ticker_layer": (
        lambda _kind, statement, answer, observation: _ground_equity_provider_quote(
            statement, answer, observation
        )
    ),
    "market.search_equity_symbols": (
        lambda _kind, statement, answer, observation: _ground_equity_search(
            statement, answer, observation
        )
    ),
    "market.search_symbols_alpha_vantage": (
        lambda _kind, statement, answer, observation: _ground_equity_search(
            statement, answer, observation
        )
    ),
    "market.search_symbols_massive": (
        lambda _kind, statement, answer, observation: _ground_equity_search(
            statement, answer, observation
        )
    ),
    "market.search_symbols_ticker_layer": (
        lambda _kind, statement, answer, observation: _ground_equity_search(
            statement, answer, observation
        )
    ),
    "market.get_equity_profile": (
        lambda _kind, statement, answer, observation: _ground_equity_profile(
            statement, answer, observation
        )
    ),
    "market.get_company_profile_alpha_vantage": (
        lambda _kind, statement, answer, observation: _ground_equity_profile(
            statement, answer, observation
        )
    ),
    "market.get_company_profile_finnhub": (
        lambda _kind, statement, answer, observation: _ground_equity_profile(
            statement, answer, observation
        )
    ),
    "market.get_company_profile_massive": (
        lambda _kind, statement, answer, observation: _ground_equity_profile(
            statement, answer, observation
        )
    ),
    "market.get_company_profile_ticker_layer": (
        lambda _kind, statement, answer, observation: _ground_equity_profile(
            statement, answer, observation
        )
    ),
    "market.get_crypto_snapshot": (
        lambda _kind, statement, answer, observation: ground_crypto_aggregate_snapshot(
            statement, answer, observation
        )
    ),
    "market.get_crypto_snapshot_coingecko": (
        lambda _kind, statement, answer, observation: ground_crypto_provider_snapshot(
            statement, answer, observation
        )
    ),
    "market.get_crypto_snapshot_coinmarketcap": (
        lambda _kind, statement, answer, observation: ground_crypto_provider_snapshot(
            statement, answer, observation
        )
    ),
    "web.fetch_public_text": lambda _kind, statement, answer, observation: _ground_public_text(
        statement, answer, observation
    ),
    "web.search_tavily": lambda kind, statement, answer, observation: _ground_tavily_search(
        kind, statement, answer, observation
    ),
    "web.search_exa": lambda _kind, statement, answer, observation: _ground_exa_search(
        statement, answer, observation
    ),
    "web.research_verified": (
        lambda _kind, statement, answer, observation: _ground_verified_web(
            statement, answer, observation
        )
    ),
    "market.get_company_profile": (
        lambda _kind, statement, answer, observation: _ground_finnhub_profile(
            statement, answer, observation
        )
    ),
    "market.get_company_news": (
        lambda _kind, statement, answer, observation: _ground_finnhub_news(
            statement, answer, observation
        )
    ),
    "market.get_earnings_surprises": (
        lambda _kind, statement, answer, observation: _ground_finnhub_earnings(
            statement, answer, observation
        )
    ),
    "market.get_basic_financials": (
        lambda _kind, statement, answer, observation: _ground_finnhub_basic_financials(
            statement, answer, observation
        )
    ),
    "sec.get_recent_filings": lambda _kind, statement, answer, observation: _ground_sec_filings(
        statement, answer, observation
    ),
    "agent.delegate_research": lambda kind, statement, answer, observation: (
        _ground_delegated_research(kind, statement, answer, observation)
    ),
    "agent.execute_research_plan": lambda kind, statement, answer, observation: (
        _ground_research_plan(kind, statement, answer, observation)
    ),
    "thread_context.open": lambda kind, statement, answer, observation: _ground_thread_context_open(
        kind, statement, answer, observation
    ),
}
_RELAXED_INTEGRATION_KINDS = frozenset(
    kind for kind in _DEFAULT_GROUNDING_RULES if kind.startswith(("market.", "web.", "sec."))
)
# Delegated-research observations (a child task's verified findings) may be paraphrased
# by the parent agent's own INFERENCE claims, same as any other trusted integration
# payload. They are deliberately excluded here for SOURCE_CLAIM: a source claim asserts
# exact provenance, and _ground_delegated_research / _ground_research_plan must keep
# requiring a verbatim match against what the child harness actually verified so the
# model cannot misattribute or subtly alter it.
_RELAXED_DELEGATED_RESEARCH_KINDS = frozenset(
    {"agent.delegate_research", "agent.execute_research_plan"}
)


class DeterministicCompletionVerifier:
    def __init__(
        self,
        ids: IdGenerator,
        clock: Clock,
        *,
        require_source_claim: bool = True,
        required_observation_kinds: frozenset[str] = frozenset(),
        required_any_observation_kinds: frozenset[str] = frozenset(),
        evidence_requirements: tuple[EvidenceToolRequirement, ...] = (),
        grounding_rules: Mapping[str, GroundingRule] | None = None,
        research_requirement: ResearchRequirement | None = None,
        completion_contract: CompletionContract | None = None,
        relax_integration_grounding: bool = False,
    ) -> None:
        self._ids = ids
        self._clock = clock
        self._require_source_claim = require_source_claim
        self._required_observation_kinds = required_observation_kinds
        self._required_any_observation_kinds = required_any_observation_kinds
        self._grounding_rules = dict(_DEFAULT_GROUNDING_RULES)
        if grounding_rules is not None:
            self._grounding_rules.update(
                {kind: _adapt_grounding_rule(rule) for kind, rule in grounding_rules.items()}
            )
        requirement_kinds = tuple(item.observation_kind for item in evidence_requirements)
        if len(requirement_kinds) != len(set(requirement_kinds)):
            raise ValueError("evidence requirements must have unique observation kinds")
        self._evidence_requirements = evidence_requirements
        self._research_requirement = research_requirement
        self._completion_contract = completion_contract
        self._relax_integration_grounding = relax_integration_grounding

    def verify(self, proposal: CompletionProposal, bundle: RunBundle) -> VerificationOutcome:
        """Judge the model's answer. Never write one in its place.

        This used to substitute ``canonical_evidence_completion`` -- provider-
        shaped prose assembled by the harness -- whenever the proposal lacked the
        expected claims. The verifier's job is to decide whether an answer is
        good enough, and a verifier that authors answers cannot do that job: it
        was grading its own output, and the user received machine-assembled
        provider text instead of a reply the model wrote. A deficient proposal is
        now rejected with actionable feedback so the *model* fixes it.
        """

        observations = {item.id: item for item in bundle.observations}
        checks: list[VerifierCheck] = [
            _answer_completeness_check(proposal.answer, tuple(observations.values())),
            _answer_sufficiency_check(proposal, bundle.task.objective),
            _answer_format_check(proposal, bundle.task.objective),
        ]
        cited_source_ids = {
            observation_id
            for claim in proposal.claims
            if claim.kind is ClaimKind.SOURCE_CLAIM
            for observation_id in claim.observation_ids
        }
        cited_ids = {
            observation_id for claim in proposal.claims for observation_id in claim.observation_ids
        }
        nested_plan_evidence = tuple(
            evidence
            for observation in observations.values()
            for evidence in _verified_plan_evidence(observation)
        )

        if self._completion_contract is not None:
            source_claims = tuple(
                claim for claim in proposal.claims if claim.kind is ClaimKind.SOURCE_CLAIM
            )
            inferences = tuple(
                claim for claim in proposal.claims if claim.kind is ClaimKind.INFERENCE
            )
            contract = self._completion_contract
            checks.extend(
                (
                    _cardinality_check(
                        name="completion_source_claim_count",
                        count=len(source_claims),
                        minimum=contract.source_claim_count.minimum,
                        maximum=contract.source_claim_count.maximum,
                    ),
                    _cardinality_check(
                        name="completion_inference_count",
                        count=len(inferences),
                        minimum=contract.inference_count.minimum,
                        maximum=contract.inference_count.maximum,
                    ),
                )
            )
            checks.extend(
                _cardinality_check(
                    name=f"completion_source_claim_{index}_observation_count",
                    count=len(claim.observation_ids),
                    minimum=contract.source_observation_id_count.minimum,
                    maximum=contract.source_observation_id_count.maximum,
                )
                for index, claim in enumerate(source_claims)
            )
            if (
                len(source_claims) == 1
                and contract.source_claim_count.minimum == 1
                and contract.source_claim_count.maximum == 1
                and len(source_claims[0].observation_ids) == 1
            ):
                sole_observation = observations.get(source_claims[0].observation_ids[0])
                if (
                    sole_observation is not None
                    and sole_observation.kind == "sec.get_recent_filings"
                ):
                    supported, detail = _ground_sec_filings(
                        proposal.answer,
                        proposal.answer,
                        sole_observation,
                    )
                    checks.append(
                        VerifierCheck(
                            name="completion_single_sec_answer_supported",
                            passed=supported,
                            detail=detail,
                        )
                    )

        for index, requirement in enumerate(self._evidence_requirements):
            matching_ids = {
                observation.id
                for observation in observations.values()
                if _observation_satisfies_requirement(
                    observation,
                    requirement,
                    self._clock,
                )
            }
            matching_ids.update(
                evidence.parent_observation_id
                for evidence in nested_plan_evidence
                if _observation_satisfies_requirement(
                    evidence.observation,
                    requirement,
                    self._clock,
                )
            )
            present = bool(matching_ids)
            checks.extend(
                (
                    VerifierCheck(
                        name=(f"required_evidence_{index}_{requirement.observation_kind}_present"),
                        passed=present,
                        detail=(
                            "Required constrained evidence is present and fresh."
                            if present
                            else "Required constrained evidence is missing, mismatched, or stale."
                        ),
                    ),
                    VerifierCheck(
                        name=(f"required_evidence_{index}_{requirement.observation_kind}_cited"),
                        passed=bool(matching_ids & cited_source_ids),
                        detail=(
                            "A source-backed claim cites the required constrained evidence."
                            if matching_ids & cited_source_ids
                            else "A source-backed claim must cite required constrained evidence."
                        ),
                    ),
                )
            )

        for kind in sorted(self._required_observation_kinds):
            present = any(item.kind == kind for item in observations.values())
            checks.append(
                VerifierCheck(
                    name=f"required_observation_{kind}",
                    passed=present,
                    detail=(
                        f"Required observation {kind} is present."
                        if present
                        else f"Required observation {kind} is missing."
                    ),
                )
            )

        if self._required_any_observation_kinds:
            matching_ids = {
                item.id
                for item in observations.values()
                if item.kind in self._required_any_observation_kinds
                and (item.expires_at is None or item.expires_at > self._clock.now())
            }
            expected = ", ".join(sorted(self._required_any_observation_kinds))
            checks.extend(
                (
                    VerifierCheck(
                        name="required_any_observation_present",
                        passed=bool(matching_ids),
                        detail=(
                            "A required orchestration result is present and fresh."
                            if matching_ids
                            else (
                                "Execute one required orchestration tool before completing: "
                                f"{expected}."
                            )
                        ),
                    ),
                    VerifierCheck(
                        name="required_any_observation_cited",
                        passed=bool(matching_ids.intersection(cited_ids)),
                        detail=(
                            "The completion uses the required orchestration result."
                            if matching_ids.intersection(cited_ids)
                            else "The completion must cite the executed orchestration result."
                        ),
                    ),
                )
            )

        has_source_claim = any(claim.kind is ClaimKind.SOURCE_CLAIM for claim in proposal.claims)
        checks.append(
            VerifierCheck(
                name="source_claim_required",
                passed=has_source_claim or not self._require_source_claim,
                detail=(
                    "At least one source-backed claim is present."
                    if has_source_claim
                    else (
                        "A source-backed claim is required for this task."
                        if self._require_source_claim
                        else "No source-backed claim is required for this context-only answer."
                    )
                ),
            )
        )

        checks.extend(
            _research_plan_coverage_checks(
                proposal,
                nested_plan_evidence,
                cited_ids=cited_ids,
            )
        )

        for index, candidate in enumerate(proposal.claims):
            if candidate.kind is ClaimKind.SOURCE_CLAIM and not candidate.observation_ids:
                checks.append(
                    VerifierCheck(
                        name=f"claim_{index}_has_observation",
                        passed=False,
                        detail="Source-backed claim has no observation reference.",
                    )
                )
                continue
            for observation_id in candidate.observation_ids:
                observation = observations.get(observation_id)
                exists = observation is not None
                checks.append(
                    VerifierCheck(
                        name=f"claim_{index}_observation_{observation_id}_exists",
                        passed=exists,
                        detail=(
                            "Referenced observation exists."
                            if exists
                            else "Referenced observation does not exist in this run."
                        ),
                    )
                )
                if observation is not None:
                    in_scope = observation.scope == bundle.run.scope
                    checks.append(
                        VerifierCheck(
                            name=f"claim_{index}_observation_{observation_id}_scope",
                            passed=in_scope,
                            detail=(
                                "Referenced observation is in scope."
                                if in_scope
                                else "Referenced observation is outside the run scope."
                            ),
                        )
                    )
                    fresh = observation.status is ObservationStatus.RETRIEVED and (
                        observation.expires_at is None or observation.expires_at > self._clock.now()
                    )
                    checks.append(
                        VerifierCheck(
                            name=f"claim_{index}_observation_{observation_id}_status",
                            passed=observation.status is ObservationStatus.RETRIEVED,
                            detail=(
                                "Referenced observation is eligible evidence."
                                if observation.status is ObservationStatus.RETRIEVED
                                else "Referenced observation is stale or rejected evidence."
                            ),
                        )
                    )
                    checks.append(
                        VerifierCheck(
                            name=f"claim_{index}_observation_{observation_id}_fresh",
                            passed=fresh,
                            detail=(
                                "Referenced observation is fresh."
                                if fresh
                                else "Referenced observation is expired."
                            ),
                        )
                    )
                    if observation.kind == "thread_context.open":
                        checks.append(
                            VerifierCheck(
                                name=(f"claim_{index}_observation_{observation_id}_run_authority"),
                                passed=observation.run_id == bundle.run.id,
                                detail=(
                                    "Opened thread context is bound to this exact run."
                                    if observation.run_id == bundle.run.id
                                    else "Opened thread context belongs to a different run."
                                ),
                            )
                        )
                    if candidate.kind is ClaimKind.SOURCE_CLAIM:
                        checks.append(
                            VerifierCheck(
                                name=f"claim_{index}_observation_{observation_id}_quality",
                                passed=(
                                    (
                                        self._relax_integration_grounding
                                        and _is_relaxed_integration_observation(
                                            observation, candidate.kind
                                        )
                                    )
                                    or observation.quality is not EvidenceQuality.DISCOVERY_ONLY
                                ),
                                detail=(
                                    "Trusted integration evidence is available to the model."
                                    if (
                                        self._relax_integration_grounding
                                        and _is_relaxed_integration_observation(
                                            observation, candidate.kind
                                        )
                                    )
                                    else (
                                        "Referenced evidence quality may support a source claim."
                                        if observation.quality is not EvidenceQuality.DISCOVERY_ONLY
                                        else (
                                            "Discovery-only metadata cannot support a source claim."
                                        )
                                    )
                                ),
                            )
                        )
                    supported, support_detail = _claim_support(
                        candidate.kind,
                        candidate.statement,
                        proposal.answer,
                        observation,
                        self._grounding_rules,
                        relax_integration_grounding=self._relax_integration_grounding,
                    )
                    checks.append(
                        VerifierCheck(
                            name=f"claim_{index}_observation_{observation_id}_supported",
                            passed=supported,
                            detail=support_detail,
                        )
                    )

        if self._research_requirement is not None:
            research_proposal, research_bundle, persisted_ids = _project_nested_research(
                proposal,
                bundle,
                nested_plan_evidence,
            )
            research = verify_research(
                research_proposal,
                research_bundle,
                now=self._clock.now(),
                requirement=self._research_requirement,
                persisted_claim_ids=persisted_ids,
            )
            checks.extend(
                check.model_copy(update={"name": f"research_{check.name}"})
                for check in research.checks
            )

        passed = all(check.passed for check in checks)
        result = VerifierResult(
            status=VerifierStatus.PASS if passed else VerifierStatus.FAIL,
            checks=tuple(checks),
            retryable=not passed,
            allow_unsourced_completion=not self._require_source_claim,
        )
        if not passed:
            return VerificationOutcome(result=result)

        claims = tuple(
            Claim(
                id=self._ids.new("claim"),
                scope=bundle.run.scope,
                run_id=bundle.run.id,
                kind=candidate.kind,
                statement=_canonical_statement(
                    candidate.statement, candidate.observation_ids, observations
                ),
                observation_ids=candidate.observation_ids,
            )
            for candidate in proposal.claims
        )
        epistemic_observation_ids = tuple(
            dict.fromkeys(
                observation_id
                for candidate in proposal.claims
                for observation_id in candidate.observation_ids
            )
        )
        if proposal.affected_assumption is not None:
            claims += (
                Claim(
                    id=self._ids.new("claim"),
                    scope=bundle.run.scope,
                    run_id=bundle.run.id,
                    kind=ClaimKind.AFFECTED_ASSUMPTION,
                    statement=proposal.affected_assumption,
                    observation_ids=epistemic_observation_ids,
                ),
            )
        if proposal.uncertainty is not None:
            claims += (
                Claim(
                    id=self._ids.new("claim"),
                    scope=bundle.run.scope,
                    run_id=bundle.run.id,
                    kind=ClaimKind.UNCERTAINTY,
                    statement=proposal.uncertainty,
                    observation_ids=epistemic_observation_ids,
                ),
            )
        canonical_quote_answer = (
            claims[0].statement
            if (
                not self._relax_integration_grounding
                and len(claims) == 1
                and claims[0].kind is ClaimKind.SOURCE_CLAIM
                and any(
                    observations[item].kind == "market.get_quote"
                    for item in claims[0].observation_ids
                )
            )
            else None
        )
        completion = VerifiedCompletion(
            answer=canonical_quote_answer or proposal.answer,
            claims=claims,
            verifier_result=result,
        )
        return VerificationOutcome(result=result, completion=completion)


def _answer_completeness_check(
    answer: str,
    observations: tuple[Observation, ...],
) -> VerifierCheck:
    """Reject only deterministic terminal shapes that require continuation.

    A provider's ``finish_reason=stop`` says how generation ended, not whether the
    proposed prose is complete.  This deliberately small suffix check catches the
    high-confidence truncation shapes seen in production without trying to judge
    writing quality or requiring sentence punctuation.  Verifier failures are
    retryable, so the ordinary coordinator loop returns the actionable detail to the
    model for a bounded repair turn.
    """

    if answer[-1].isspace():
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "Finish the incomplete final clause and return the complete answer "
                "without trailing whitespace."
            ),
        )

    if contains_future_action_promise(answer):
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "Do not complete with a promise of future work. If a concrete current-evidence "
                "target is available, call an eligible read tool now; otherwise ask one concrete "
                "input-seeking question."
            ),
        )

    if presents_unretrieved_data(answer, observations):
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "This answer presents itself as showing current or live data, but this run "
                "retrieved nothing. Call an eligible read tool now and answer from the result, "
                "or drop the framing and say plainly that the figures are from general "
                "knowledge and may be out of date."
            ),
        )

    completed_action = completed_research_action_claim(answer)
    if completed_action is not None and not _completed_action_is_observed(
        completed_action,
        observations,
    ):
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "Do not claim a completed quote or research action without a matching retrieved "
                "observation. Call the eligible read tool first, or describe only what was "
                "actually observed."
            ),
        )

    if _has_structured_terminal_shape(answer):
        return VerifierCheck(
            name="answer_completeness",
            passed=True,
            detail="The proposed answer has a complete terminal shape.",
        )

    suffix = answer[-128:]
    if _DANGLING_TERMINAL_PUNCTUATION.search(suffix) is not None:
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "Finish the incomplete final clause; the answer ends with punctuation "
                "that requires continuation."
            ),
        )

    connective = _DANGLING_TERMINAL_CONNECTIVE.search(suffix)
    if connective is not None:
        return VerifierCheck(
            name="answer_completeness",
            passed=False,
            detail=(
                "Finish the incomplete final clause after the trailing connective "
                f"'{connective.group(1).lower()}'."
            ),
        )

    return VerifierCheck(
        name="answer_completeness",
        passed=True,
        detail="The proposed answer has a complete terminal shape.",
    )


_PRESENTS_LIVE_DATA = re.compile(
    r"\bhere(?:'s| is| are)\s+what\s+the\s+(?:current|live|latest|recent)\s+"
    r"(?:data|numbers?|figures?|prices?|quotes?|rates?)\b"
    r"|\bbased\s+on\s+(?:the\s+)?(?:current|live|latest)\s+"
    r"(?:data|numbers?|figures?|market data)\b"
    r"|\bbuilt\s+from\s+(?:the\s+)?live\s+data\b"
    r"|\baccording\s+to\s+the\s+(?:live|current|latest)\s+(?:data|feed|numbers?)\b"
)


def presents_unretrieved_data(answer: str, observations: tuple[Observation, ...]) -> bool:
    """Reject "here's what the current data shows" when nothing was retrieved.

    Phrasing guards are a losing game one regex at a time -- each new way of
    promising or implying live data slips through until someone adds it. This is
    the structural version of the same rule: whatever the wording, an answer that
    frames itself as *presenting retrieved data* is false if the run holds no
    retrieved observation at all. General-knowledge answers are unaffected as long
    as they do not claim to be showing live data, which is exactly the honesty the
    framing is meant to convey.
    """

    if _PRESENTS_LIVE_DATA.search(_normalized_answer_prose(answer)) is None:
        return False
    return not any(
        observation.status is ObservationStatus.RETRIEVED for observation in observations
    )


def _normalized_answer_prose(answer: str) -> str:
    normalized = answer.casefold().replace("\u2019", "'").replace("\u2018", "'")
    return " ".join(normalized.split())


def _completed_action_is_observed(
    action: str,
    observations: tuple[Observation, ...],
) -> bool:
    retrieved_kinds = {
        observation.kind
        for observation in observations
        if observation.status is ObservationStatus.RETRIEVED
    }
    if action == "market_quote":
        return any(
            kind == "market.get_quote"
            or kind.startswith("market.get_quote_")
            or kind == "market.get_crypto_snapshot"
            or kind.startswith("market.get_crypto_snapshot_")
            for kind in retrieved_kinds
        )
    if action == "filing":
        return "sec.get_recent_filings" in retrieved_kinds
    return any(
        kind.startswith(("web.", "market.", "sec.", "agent.", "memory.", "thread_context."))
        for kind in retrieved_kinds
    )


def _answer_sufficiency_check(
    proposal: CompletionProposal,
    objective: str,
) -> VerifierCheck:
    requested = _requests_concrete_options(objective)
    clarification = _is_genuine_input_clarification(proposal)
    preamble_only = requested and not clarification and _is_output_preamble_only(proposal.answer)
    return VerifierCheck(
        name="answer_sufficiency",
        passed=not preamble_only,
        detail=(
            "The requested concrete recommendations, examples, options, or comparison are present."
            if not preamble_only
            else (
                "Provide the requested concrete recommendations, examples, options, or "
                "comparison now, or ask one genuine input-seeking clarification. Do not stop "
                "after a list introduction, methodology note, or disclaimer."
            )
        ),
    )


@dataclass(frozen=True)
class _RequestedAnswerFormat:
    bullet_count: int | None = None
    sentence_count: int | None = None
    name_line_count: int | None = None
    names_only_one_per_line: bool = False
    maximum_words: int | None = None

    @property
    def requested(self) -> bool:
        return any(
            (
                self.bullet_count is not None,
                self.sentence_count is not None,
                self.names_only_one_per_line,
                self.maximum_words is not None,
            )
        )


def _answer_format_check(
    proposal: CompletionProposal,
    objective: str,
) -> VerifierCheck:
    """Enforce only explicit, deterministic small output-shape constraints.

    This is intentionally not a style grader.  It covers exact counts and layouts
    whose compliance can be decided without semantic judgment, then feeds a bounded,
    retryable repair instruction back through the ordinary coordinator loop.
    """

    requested = _requested_answer_format(objective)
    if not requested.requested:
        return VerifierCheck(
            name="answer_format",
            passed=True,
            detail="No deterministic answer-format constraint was requested.",
        )

    clarification = _is_genuine_input_clarification(proposal)
    if clarification:
        required = _objective_requires_format_clarification(objective)
        return VerifierCheck(
            name="answer_format",
            passed=required,
            detail=(
                "A genuine missing-input clarification may bypass the requested output format."
                if required
                else (
                    "The request already supplies the target and output constraints. Answer it "
                    "directly in the requested format instead of asking for missing details."
                )
            ),
        )

    answer = proposal.answer
    failures: list[str] = []
    nonempty_lines = tuple(line.strip() for line in answer.splitlines() if line.strip())

    if requested.bullet_count is not None:
        bullet_lines = tuple(
            line for line in nonempty_lines if _UNORDERED_BULLET_ITEM.match(line) is not None
        )
        if (
            len(nonempty_lines) != requested.bullet_count
            or len(bullet_lines) != requested.bullet_count
        ):
            failures.append(
                f"Return exactly {requested.bullet_count} bullet lines, each beginning with "
                "-, *, +, or •; do not use a numbered paragraph."
            )

    if requested.sentence_count is not None:
        actual_sentences = _answer_sentence_count(answer)
        if actual_sentences != requested.sentence_count:
            failures.append(
                f"Return exactly {requested.sentence_count} sentences; the proposed answer has "
                f"{actual_sentences}."
            )

    if requested.names_only_one_per_line:
        if requested.name_line_count is not None and (
            len(nonempty_lines) != requested.name_line_count
        ):
            failures.append(
                f"Return exactly {requested.name_line_count} names, one non-empty name per line."
            )
        if not nonempty_lines or any(not _is_plain_name_line(line) for line in nonempty_lines):
            failures.append(
                "Return names only, one plain name per line, without bullets, numbering, or "
                "explanations."
            )

    if requested.maximum_words is not None:
        actual_words = len(_ANSWER_WORD.findall(answer))
        if actual_words > requested.maximum_words:
            failures.append(
                f"Keep the answer to at most {requested.maximum_words} words; the proposed "
                f"answer has {actual_words}."
            )

    return VerifierCheck(
        name="answer_format",
        passed=not failures,
        detail=(
            "The answer follows the explicit deterministic format constraints."
            if not failures
            else " ".join(failures)
        ),
    )


def _requested_answer_format(objective: str) -> _RequestedAnswerFormat:
    normalized = " ".join(objective.casefold().split())
    bullet_match = _BULLET_COUNT_REQUEST.search(normalized)
    sentence_match = _SENTENCE_COUNT_REQUEST.search(normalized)
    name_layout = bool(
        re.search(r"\bnames?\s+only\b", normalized)
        and re.search(r"\bone(?:\s+name)?\s+per\s+line\b", normalized)
    )
    name_match = _NAME_COUNT_REQUEST.search(normalized) if name_layout else None
    strict_cap = _STRICT_WORD_CAP.search(normalized)
    inclusive_cap = _INCLUSIVE_WORD_CAP.search(normalized)

    maximum_words: int | None = None
    if strict_cap is not None:
        parsed = _bounded_number(strict_cap.group("count"), maximum=500)
        maximum_words = None if parsed is None else max(0, parsed - 1)
    elif inclusive_cap is not None:
        maximum_words = _bounded_number(inclusive_cap.group("count"), maximum=500)

    return _RequestedAnswerFormat(
        bullet_count=(
            None
            if bullet_match is None
            else _bounded_number(bullet_match.group("count"), maximum=12)
        ),
        sentence_count=(
            None
            if sentence_match is None
            else _bounded_number(sentence_match.group("count"), maximum=12)
        ),
        name_line_count=(
            None if name_match is None else _bounded_number(name_match.group("count"), maximum=12)
        ),
        names_only_one_per_line=name_layout,
        maximum_words=maximum_words,
    )


def _bounded_number(value: str, *, maximum: int) -> int | None:
    normalized = value.casefold()
    parsed = int(normalized) if normalized.isdigit() else _SMALL_NUMBERS.get(normalized)
    return parsed if parsed is not None and 1 <= parsed <= maximum else None


def _answer_sentence_count(answer: str) -> int:
    count = 0
    for line in (item.strip() for item in answer.splitlines()):
        if not line:
            continue
        protected = _HTTP_URL.sub("URL", line)
        protected = _DECIMAL_POINT.sub("\u0000", protected)
        protected = _DOTTED_INITIALISM.sub(
            lambda item: item.group(0).replace(".", "\u0000"), protected
        )
        protected = _COMMON_ABBREVIATION.sub(
            lambda item: item.group(0).replace(".", "\u0000"), protected
        )
        count += sum(
            1
            for segment in re.findall(r"[^.!?]+(?:[.!?]+|$)", protected)
            if _ANSWER_WORD.search(segment) is not None
        )
    return count


def _is_plain_name_line(line: str) -> bool:
    words = _ANSWER_WORD.findall(line)
    return (
        bool(words)
        and len(words) <= 8
        and not (
            _MARKDOWN_LIST_ITEM.match(line) is not None
            or _UNORDERED_BULLET_ITEM.match(line) is not None
            or re.match(r"\s*\d+[.)]\s+", line) is not None
            or re.search(r"[:;!?]|[.!]$|\s[-\u2013\u2014]\s|[()[\]{}]", line) is not None
        )
    )


def _objective_requires_format_clarification(objective: str) -> bool:
    normalized = " ".join(objective.casefold().split())
    if re.search(r"\b(?:unspecified|not specified|unknown|tbd)\b", normalized):
        return True
    inline_payload = bool(
        re.search(r":\s*\S", objective)
        or re.search(
            r"\b(?:edit|proofread|rephrase|rewrite|summari[sz]e|translate)\s+"
            r"[\"'\u2018\u201c][^\"'\u2019\u201d\r\n]+[\"'\u2019\u201d]",
            objective,
            flags=re.IGNORECASE,
        )
    )
    if inline_payload:
        return False
    return bool(
        re.search(
            r"\b(?:compare|contrast|edit|explain|review|rewrite|summari[sz]e|translate)\s+"
            r"(?:it|this|that|them|these|those)\b|\b(?:sell|send|delete|remove|approve|"
            r"cancel)\s+(?:it|this|that|them|these|those)\b|\bdo\s+it\b",
            normalized,
        )
    )


def _requests_concrete_options(objective: str) -> bool:
    normalized = " ".join(objective.casefold().split())
    return bool(
        re.search(
            r"\b(?:recommend(?:ation|ations)?|suggest(?:ion|ions)?|examples?|options?|"
            r"alternatives?|compare|comparison|versus|rank|(?:short|watch|wish|check|"
            r"bucket)?lists?|ideas?|opportunit(?:y|ies))\b",
            normalized,
        )
        or re.search(
            r"\b(?:some|a few|few|a couple of|couple of)\b[^.!?]{0,80}\b(?:stocks?|"
            r"companies|investments?|funds?|etfs?|bonds?|candidates?|names?|choices?)\b",
            normalized,
        )
    )


def _is_output_preamble_only(answer: str) -> bool:
    normalized_quotes = answer.replace("\u2019", "'").replace("\u2018", "'")
    without_notes = _TERMINAL_DISCLAIMER_PARENTHETICAL.sub(" ", normalized_quotes).strip()
    introduction = _OUTPUT_INTRODUCTION.match(without_notes)
    if introduction is None:
        return False
    remainder = without_notes[introduction.end() :].strip()
    qualifier = _INTRODUCTORY_QUALIFIER.match(remainder)
    if qualifier is not None:
        remainder = remainder[qualifier.end() :].strip()
    remainder = remainder.lstrip(".:;,-\u2013\u2014 ")
    substantive_sentences = tuple(
        sentence.strip()
        for sentence in re.findall(r"[^.!?]+(?:[.!?]+|$)", remainder)
        if sentence.strip()
        and re.search(r"\w", sentence) is not None
        and not _is_terminal_disclaimer(sentence)
    )
    return not substantive_sentences


def _is_terminal_disclaimer(sentence: str) -> bool:
    normalized = " ".join(sentence.casefold().split()).strip("()[]{} .!?:;-\u2013\u2014")
    return bool(
        re.search(
            r"\b(?:not (?:personalized |financial |investment )*advice|"
            r"treat (?:these|this) as (?:a )?starting points?|"
            r"do your own research|consult (?:a|your) (?:financial )?(?:adviser|advisor)|"
            r"i (?:do not|don't) have your (?:risk tolerance|time horizon)|"
            r"it depends on your (?:goals|preferences|risk tolerance|time horizon)|"
            r"does (?:this|that) help|does (?:this|that) make sense|is that clear)\b",
            normalized,
        )
    )


def _is_genuine_input_clarification(proposal: CompletionProposal) -> bool:
    if proposal.claims or proposal.affected_assumption or proposal.uncertainty:
        return False
    segments = tuple(
        item.strip() for item in re.findall(r"[^.!?]+(?:[.!?]+|$)", proposal.answer) if item.strip()
    )
    if not segments:
        return False
    question_count = 0
    for segment in segments:
        if not segment.endswith("?"):
            if _CLARIFICATION_PREAMBLE.fullmatch(segment.rstrip(".!")) is not None:
                continue
            return False
        question = segment[:-1].strip()
        if ":" in question:
            prefix, remainder = question.split(":", 1)
            if _CLARIFICATION_PREAMBLE.fullmatch(prefix.strip()) is not None:
                question = remainder.strip()
        normalized = " ".join(question.casefold().split()).strip("-\u2013\u2014* ")
        input_seeking = re.match(
            r"^(?:what|which|who|whom|whose|when|where|why|how|do|does|did|is|are|"
            r"was|were|can|could|would|should|will|have|has|may|for (?:what|which|how)|"
            r"between (?:what|which)|over (?:what|which)|during (?:what|which))\b",
            normalized,
        )
        tag_on = re.fullmatch(
            r"(?:does (?:this|that) help|does (?:this|that) make sense|is that clear|"
            r"sound good|okay|right)",
            normalized,
        )
        if input_seeking is None or tag_on is not None:
            return False
        question_count += 1
    return question_count > 0


def _has_structured_terminal_shape(answer: str) -> bool:
    """Recognize complete non-prose answers before applying prose suffix rules."""

    if is_public_https_url(answer) or _HTTP_URL.fullmatch(answer) is not None:
        return True
    if _STANDALONE_NUMBER.fullmatch(answer) is not None:
        return True
    if _COMPLETE_FENCED_CODE.fullmatch(answer) is not None:
        return True
    if _COMPLETE_INLINE_CODE.fullmatch(answer) is not None:
        return True
    if _COMPLETE_SEMICOLON_CODE.fullmatch(answer) is not None:
        return True
    lines = answer.splitlines()
    nonempty_lines = tuple(line for line in lines if line.strip())
    return bool(nonempty_lines) and all(
        _MARKDOWN_LIST_ITEM.match(line) is not None for line in nonempty_lines
    )


def _observation_satisfies_requirement(
    observation: Observation,
    requirement: EvidenceToolRequirement,
    clock: Clock,
) -> bool:
    return (
        observation.kind == requirement.observation_kind
        and constrained_values_match(
            requirement.required_arguments,
            observation.data,
            exact=False,
        )
        and (observation.expires_at is None or observation.expires_at > clock.now())
    )


def _verified_plan_evidence(observation: Observation) -> tuple[_NestedPlanEvidence, ...]:
    if observation.kind != "agent.execute_research_plan":
        return ()
    plan, _detail = _validate_research_plan(observation, require_verified_authority=True)
    return () if plan is None else plan.nested_evidence


def _research_plan_coverage_checks(
    proposal: CompletionProposal,
    nested_evidence: tuple[_NestedPlanEvidence, ...],
    *,
    cited_ids: set[str],
) -> tuple[VerifierCheck, ...]:
    """Require every verified child statement when a parent plan result is used."""

    checks: list[VerifierCheck] = []
    parent_ids = {
        evidence.parent_observation_id
        for evidence in nested_evidence
        if evidence.parent_observation_id in cited_ids
    }
    for parent_id in sorted(parent_ids):
        statements = tuple(
            dict.fromkeys(
                evidence.statement
                for evidence in nested_evidence
                if evidence.parent_observation_id == parent_id
            )
        )
        for index, statement in enumerate(statements):
            matching_claim = any(
                candidate.kind is ClaimKind.SOURCE_CLAIM
                and parent_id in candidate.observation_ids
                and _same_statement(candidate.statement, statement)
                for candidate in proposal.claims
            )
            carried = _contains_statement(proposal.answer, statement)
            checks.extend(
                (
                    VerifierCheck(
                        name=f"plan_{parent_id}_child_statement_{index}_claimed",
                        passed=matching_claim,
                        detail=(
                            "A source claim cites the parent plan and exactly copies this verified "
                            f"child statement: {statement}"
                        ),
                    ),
                    VerifierCheck(
                        name=f"plan_{parent_id}_child_statement_{index}_carried",
                        passed=carried,
                        detail=(
                            "The final answer exactly carries this verified child statement: "
                            f"{statement}"
                        ),
                    ),
                )
            )
    return tuple(checks)


def _project_nested_research(
    proposal: CompletionProposal,
    bundle: RunBundle,
    nested_evidence: tuple[_NestedPlanEvidence, ...],
) -> tuple[ResearchProposal, RunBundle, frozenset[str]]:
    """Expose only digest-validated child sources to deterministic research checks.

    The synthetic IDs are verifier-local views of data already persisted inside the
    parent plan observation. They cannot be cited by the model and never become claims.
    """

    synthetic_by_id = {item.observation.id: item.observation for item in nested_evidence}
    research_claims: list[ResearchClaim] = []
    for candidate in proposal.claims:
        if not candidate.observation_ids:
            continue
        projected_ids: list[str] = []
        for observation_id in candidate.observation_ids:
            matching = tuple(
                evidence.observation.id
                for evidence in nested_evidence
                if evidence.parent_observation_id == observation_id
                and _same_statement(evidence.statement, candidate.statement)
            )
            projected_ids.extend(matching or (observation_id,))
        research_claims.append(
            ResearchClaim(
                kind=candidate.kind,
                statement=candidate.statement,
                observation_ids=tuple(dict.fromkeys(projected_ids)),
            )
        )
    research_bundle = bundle.model_copy(
        update={"observations": (*bundle.observations, *synthetic_by_id.values())}
    )
    return (
        ResearchProposal(
            answer=proposal.answer,
            uncertainty=proposal.uncertainty,
            affected_assumption=proposal.affected_assumption,
            claims=tuple(research_claims),
        ),
        research_bundle,
        frozenset({item.id for item in research_bundle.observations}),
    )


def _claim_support(
    claim_kind: ClaimKind,
    statement: str,
    answer: str,
    observation: Observation,
    grounding_rules: Mapping[str, _ClaimAwareGroundingRule],
    *,
    relax_integration_grounding: bool = False,
) -> tuple[bool, str]:
    rule = grounding_rules.get(observation.kind)
    if rule is None:
        if relax_integration_grounding and _is_relaxed_integration_observation(
            observation, claim_kind
        ):
            return True, "Trusted integration payload is available to the model."
        return (
            False,
            f"No registered grounding rule exists for observation kind {observation.kind}.",
        )
    supported, detail = rule(claim_kind, statement, answer, observation)
    if supported or not (
        relax_integration_grounding and _is_relaxed_integration_observation(observation, claim_kind)
    ):
        return supported, detail
    # Relaxation trusts a claim's WORDING against a connected-integration payload --
    # the model may paraphrase/synthesize instead of copying exact prose. It must not
    # trust the payload ITSELF once the tool's own grounding rule has flagged it as
    # unsound (truncated, digest-mismatched/tampered, malformed, stale, expired,
    # discovery-only). Those are integrity failures the generic structural checks
    # above (exists/in-scope/fresh/quality) cannot detect -- only the grounding rule
    # inspects payload content -- so they must still fail closed even when relaxed.
    if _looks_like_integration_payload_failure(detail):
        return False, detail
    return True, "Trusted integration payload is available; the model may synthesize the answer."


def _is_relaxed_integration_observation(observation: Observation, claim_kind: ClaimKind) -> bool:
    """Treat trusted adapter payloads as model context, not copy-exact prose.

    Delegated-research observations only qualify for INFERENCE claims: a SOURCE_CLAIM
    against them must still pass the exact-match grounding that _ground_delegated_research
    / _ground_research_plan enforce, so misattribution of a child's verified findings
    cannot be waved through by this relaxation.
    """

    if observation.kind in _RELAXED_INTEGRATION_KINDS or observation.kind.startswith(
        ("api.", "integration.", "mcp.")
    ):
        return True
    if claim_kind is ClaimKind.INFERENCE and observation.kind in _RELAXED_DELEGATED_RESEARCH_KINDS:
        return True
    return False


def _looks_like_integration_payload_failure(detail: str) -> bool:
    """True when a grounding rule rejected the payload itself, not just the wording.

    Relaxation (see _claim_support) is meant to excuse a claim that paraphrases
    real data instead of copying it verbatim -- never one built on data the tool
    itself flagged as truncated, tampered, stale, or otherwise unsound.
    """

    normalized = detail.casefold()
    return any(
        marker in normalized
        for marker in (
            "malformed",
            "unsupported provider",
            "provenance",
            "integrity",
            "was changed",
            "was altered",
            "stale",
            "expired",
            "discovery-only",
            "cannot support",
            "cannot establish",
            "truncated",
            "authority",
            "does not match the",
        )
    )


def _adapt_grounding_rule(rule: GroundingRule) -> _ClaimAwareGroundingRule:
    def adapted(
        _claim_kind: ClaimKind,
        statement: str,
        answer: str,
        observation: Observation,
    ) -> tuple[bool, str]:
        return rule(statement, answer, observation)

    return adapted


def _ground_quote(statement: str, answer: str, observation: Observation) -> tuple[bool, str]:
    if "selected_provider" in observation.data:
        return _ground_equity_aggregate_quote(statement, answer, observation)
    if observation.data.get("provider") in {
        "alpha-vantage",
        "finnhub",
        "massive",
        "ticker-layer",
    }:
        return _ground_equity_provider_quote(statement, answer, observation)
    if any(
        key in observation.data
        for key in (
            "agreement_status",
            "provider",
            "provider_attempts",
            "provider_quotes",
            "selected_reference",
            "statements",
        )
    ):
        return False, "Normalized equity quote evidence is malformed or was downgraded."
    symbol = observation.data.get("symbol")
    price = observation.data.get("price")
    if not isinstance(symbol, str) or not isinstance(price, int | float) or isinstance(price, bool):
        return False, "Quote observation lacks a valid symbol or current price."
    # Schema-v1 fixture rows remain readable for deterministic regression replay.
    # Every production adapter now emits ``data.provider`` and takes the strict path.
    if observation.source.provider not in {"finnhub", "fixture"}:
        return False, "Quote observation uses an unsupported provider or reference."
    missing_locations: list[str] = []
    if not _contains_symbol(statement, symbol):
        missing_locations.append("source claim symbol")
    if not _contains_numeric_value(statement, price):
        missing_locations.append("source claim price")
    if not _contains_symbol(answer, symbol):
        missing_locations.append("answer symbol")
    if not _contains_numeric_value(answer, price):
        missing_locations.append("answer price")
    if not missing_locations:
        return True, "Claim and answer copy the quote symbol and exact current price."
    return (
        False,
        f"Missing required exact value in: {', '.join(missing_locations)}. Both the source claim "
        f"and answer must copy symbol {symbol} and exact current price {format(price, 'g')} "
        "without rounding.",
    )


def _ground_equity_aggregate_quote(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    if not valid_equity_quote_aggregate(
        observation.data,
        source_provider=observation.source.provider,
        source_reference=observation.source.reference,
        observed_at=observation.observed_at,
        expires_at=observation.expires_at,
    ):
        return False, "Provider-neutral quote routing or provenance is malformed."
    canonical = canonical_equity_quote_statement(observation.data)
    disagreement = canonical_equity_quote_disagreement_statement(observation.data)
    time_skew = canonical_equity_quote_time_skew_statement(observation.data)
    expected = tuple(item for item in (canonical, disagreement, time_skew) if item is not None)
    canonical_claim = " ".join(expected)
    if not canonical_claim or not _same_statement(statement, canonical_claim):
        return False, "The quote claim must exactly copy its canonical price and caveat text."
    if not all(_contains_statement(answer, item) for item in expected):
        return False, "The final answer must include every canonical quote diagnostic caveat."
    return True, "Quote claim preserves exact selected provenance and provider diagnostics."


def _ground_equity_provider_quote(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    symbol = observation.data.get("symbol")
    canonical = canonical_equity_quote_statement(observation.data)
    statements = observation.data.get("statements")
    if not (
        isinstance(symbol, str)
        and canonical is not None
        and observation.source.provider
        in {
            "alpha-vantage",
            "finnhub",
            "massive",
            "ticker-layer",
        }
        and valid_equity_quote_provenance(
            provider=observation.source.provider,
            reference=observation.source.reference,
            symbol=symbol,
            observed_at=observation.observed_at,
        )
        and valid_equity_observed_at(observation.data, observation.observed_at)
        and observation.expires_at is not None
        and observation.expires_at > observation.observed_at
        and statements == [canonical]
    ):
        return False, "Equity quote evidence is malformed or lacks exact provenance."
    if not _same_statement(statement, canonical) or not _contains_statement(answer, canonical):
        return False, "The claim and answer must copy the canonical equity quote exactly."
    return True, "Claim copies one exact provider-attributed equity quote."


def _ground_equity_search(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    provider = observation.data.get("provider")
    query_hash = observation.data.get("query_hash")
    canonical = canonical_equity_search_statements(observation.data)
    if not (
        isinstance(provider, str)
        and provider in EQUITY_SEARCH_PROVIDERS
        and provider == observation.source.provider
        and isinstance(query_hash, str)
        and valid_equity_search_provenance(
            provider=provider,
            reference=observation.source.reference,
            query_hash=query_hash,
        )
        and canonical is not None
        and observation.data.get("statements") == list(canonical)
        and (
            "selected_provider" not in observation.data
            or (
                observation.data.get("selected_provider") == provider
                and observation.data.get("selected_reference") == observation.source.reference
            )
        )
    ):
        return False, "Equity symbol-search evidence is malformed or lacks exact provenance."
    return _ground_exact_statement_list(
        statement,
        answer,
        observation.data.get("statements"),
        canonical,
        kind="equity symbol search",
    )


def _ground_equity_profile(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    provider = observation.data.get("provider")
    provider_symbol = observation.data.get("provider_symbol")
    canonical = canonical_equity_profile_statements(observation.data)
    if not (
        isinstance(provider, str)
        and provider in EQUITY_PROFILE_PROVIDERS
        and provider == observation.source.provider
        and isinstance(provider_symbol, str)
        and valid_equity_profile_provenance(
            provider=provider,
            reference=observation.source.reference,
            provider_symbol=provider_symbol,
        )
        and valid_equity_observed_at(observation.data, observation.observed_at)
        and canonical is not None
        and observation.data.get("statements") == list(canonical)
        and (
            "selected_provider" not in observation.data
            or (
                observation.data.get("selected_provider") == provider
                and observation.data.get("selected_reference") == observation.source.reference
            )
        )
    ):
        return False, "Equity company-profile evidence is malformed or lacks exact provenance."
    return _ground_exact_statement_list(
        statement,
        answer,
        observation.data.get("statements"),
        canonical,
        kind="equity company profile",
    )


def _ground_tavily_search(
    claim_kind: ClaimKind,
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    query = data.get("query")
    query_hash = data.get("query_hash")
    results = data.get("results")
    result_count = data.get("result_count")
    rejected_result_count = data.get("rejected_result_count")
    if claim_kind is ClaimKind.SOURCE_CLAIM:
        return False, "Tavily search metadata is discovery-only; fetch a result before citing it."
    if not (
        observation.source.provider == "tavily"
        and isinstance(query, str)
        and bool(query.strip())
        and isinstance(query_hash, str)
        and _SHA256.fullmatch(query_hash) is not None
        and observation.source.reference == f"search:{query_hash}"
        and isinstance(results, list)
        and 1 <= len(results) <= 5
        and isinstance(result_count, int)
        and not isinstance(result_count, bool)
        and result_count == len(results)
        and isinstance(rejected_result_count, int)
        and not isinstance(rejected_result_count, bool)
        and 0 <= rejected_result_count <= 100
        and data.get("untrusted") is True
        and data.get("requires_fetch_for_source_claim") is True
        and all(_valid_tavily_result(item) for item in results)
    ):
        return False, "Tavily discovery observation is malformed or lacks provenance metadata."
    canonical = f"Tavily returned {result_count} discovery results for query: {query}"
    if _same_statement(statement, canonical) and _contains_statement(answer, canonical):
        return True, "Inference reports only the exact bounded Tavily discovery result count."
    return False, "A Tavily inference may report only its exact query and discovery result count."


def _valid_tavily_result(value: JsonValue) -> bool:
    if not isinstance(value, dict):
        return False
    title = value.get("title")
    url = value.get("url")
    snippet = value.get("snippet")
    score = value.get("score")
    missing_fields = value.get("missing_fields")
    score_shape_valid = (score is None and missing_fields == ["score"]) or (
        isinstance(score, int | float)
        and not isinstance(score, bool)
        and 0 <= score <= 1
        and missing_fields is None
    )
    return bool(
        isinstance(title, str)
        and title.strip()
        and len(title) <= 240
        and isinstance(url, str)
        and len(url) <= 2_048
        and is_public_https_url(url)
        and isinstance(snippet, str)
        and snippet.strip()
        and len(snippet) <= 1_200
        and score_shape_valid
    )


def _ground_finnhub_profile(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    symbol = data.get("symbol")
    canonical = canonical_finnhub_profile_statements(data)
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"company-profile:{symbol}"
        and canonical is not None
    ):
        return False, "Finnhub company-profile observation is malformed."
    return _ground_exact_statement_list(
        statement,
        answer,
        data.get("statements"),
        canonical,
        kind="Finnhub company profile",
    )


def _ground_finnhub_news(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    symbol = data.get("symbol")
    items = data.get("items")
    item_count = data.get("item_count")
    from_date_value = data.get("from_date")
    to_date_value = data.get("to_date")
    try:
        from_date = (
            date.fromisoformat(from_date_value) if isinstance(from_date_value, str) else None
        )
        to_date = date.fromisoformat(to_date_value) if isinstance(to_date_value, str) else None
    except ValueError:
        from_date = None
        to_date = None
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and from_date is not None
        and to_date is not None
        and from_date <= to_date
        and observation.source.reference
        == f"company-news:{symbol}:{from_date_value}:{to_date_value}"
        and isinstance(items, list)
        and 1 <= len(items) <= 10
        and isinstance(item_count, int)
        and not isinstance(item_count, bool)
        and item_count == len(items)
    ):
        return False, "Finnhub company-news observation is malformed."
    canonical: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            return False, "Finnhub company-news observation contains a malformed item."
        published_at = item.get("published_at")
        provider = item.get("source")
        headline = item.get("headline")
        url = item.get("url")
        try:
            published = (
                datetime.fromisoformat(published_at) if isinstance(published_at, str) else None
            )
        except ValueError:
            published = None
        if not (
            published is not None
            and published.tzinfo is not None
            and from_date <= published.date() <= to_date
            and published <= observation.observed_at + timedelta(seconds=60)
            and isinstance(provider, str)
            and bool(provider.strip())
            and isinstance(headline, str)
            and bool(headline.strip())
            and isinstance(url, str)
            and is_public_https_url(url)
        ):
            return False, "Finnhub company-news observation contains a malformed item."
        canonical.append(
            f"On {published_at}, {provider} reported for {symbol}: {headline} Source URL: {url}"
        )
    return _ground_exact_statement_list(
        statement,
        answer,
        data.get("statements"),
        tuple(canonical),
        kind="Finnhub company news",
    )


def _ground_finnhub_earnings(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    symbol = data.get("symbol")
    items = data.get("items")
    item_count = data.get("item_count")
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"earnings-surprises:{symbol}"
        and isinstance(items, list)
        and 1 <= len(items) <= 4
        and isinstance(item_count, int)
        and not isinstance(item_count, bool)
        and item_count == len(items)
    ):
        return False, "Finnhub earnings observation is malformed."
    for item in items:
        if not isinstance(item, dict):
            return False, "Finnhub earnings observation contains a malformed item."
        item_symbol = item.get("symbol")
        period = item.get("period")
        actual = item.get("actual")
        estimate = item.get("estimate")
        try:
            period_date = date.fromisoformat(period) if isinstance(period, str) else None
        except ValueError:
            period_date = None
        if not (
            item_symbol == symbol
            and period_date is not None
            and period_date <= observation.observed_at.date()
            and isinstance(actual, int | float)
            and not isinstance(actual, bool)
            and isinstance(estimate, int | float)
            and not isinstance(estimate, bool)
        ):
            return False, "Finnhub earnings observation contains a malformed item."
    canonical = canonical_earnings_statements(symbol, items)
    if canonical is None:
        return False, "Finnhub earnings observation contains malformed canonical data."
    return _ground_exact_statement_list(
        statement,
        answer,
        data.get("statements"),
        canonical,
        kind="Finnhub earnings",
    )


def _ground_finnhub_basic_financials(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    symbol = data.get("symbol")
    metrics = data.get("metrics")
    metric_count = data.get("metric_count")
    labels = {
        "beta": "beta",
        "52WeekHigh": "52-week high",
        "52WeekLow": "52-week low",
        "10DayAverageTradingVolume": "10-day average trading volume",
        "marketCapitalization": "market capitalization",
        "peBasicExclExtraTTM": "basic P/E excluding extraordinary items (TTM)",
    }
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"basic-financials:{symbol}"
        and isinstance(metrics, dict)
        and 1 <= len(metrics) <= len(labels)
        and isinstance(metric_count, int)
        and not isinstance(metric_count, bool)
        and metric_count == len(metrics)
        and set(metrics).issubset(labels)
    ):
        return False, "Finnhub basic-financials observation is malformed."
    canonical: list[str] = []
    for key, value in metrics.items():
        if not (
            isinstance(key, str) and isinstance(value, int | float) and not isinstance(value, bool)
        ):
            return False, "Finnhub basic-financials observation contains a malformed metric."
        canonical.append(f"{symbol} has Finnhub {labels[key]} {format(value, 'g')}.")
    return _ground_exact_statement_list(
        statement,
        answer,
        data.get("statements"),
        tuple(canonical),
        kind="Finnhub basic financials",
    )


def _ground_exact_statement_list(
    statement: str,
    answer: str,
    raw_statements: JsonValue | None,
    expected: tuple[str, ...],
    *,
    kind: str,
) -> tuple[bool, str]:
    normalized_raw = (
        tuple(item for item in raw_statements if isinstance(item, str))
        if isinstance(raw_statements, list)
        else ()
    )
    if not (
        isinstance(raw_statements, list)
        and len(raw_statements) == len(expected)
        and len(normalized_raw) == len(raw_statements)
        and len(set(normalized_raw)) == len(normalized_raw)
        and sorted(normalized_raw) == sorted(expected)
    ):
        return False, f"{kind} canonical statements do not match the underlying payload."
    matches = tuple(item for item in expected if _same_statement(statement, item))
    if len(matches) != 1:
        return False, f"The claim must exactly copy one canonical {kind} statement."
    if not _contains_statement(answer, matches[0]):
        return False, "The final answer must carry the grounded source statement."
    return True, f"Claim exactly copies one payload-derived {kind} statement."


def _ground_public_text(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    url = data.get("url")
    content_type = data.get("content_type")
    text = data.get("text")
    digest = data.get("content_sha256")
    byte_count = data.get("byte_count")
    truncated = data.get("truncated")
    if not (
        isinstance(url, str)
        and url.startswith(("http://", "https://"))
        and isinstance(content_type, str)
        and bool(content_type)
        and isinstance(text, str)
        and bool(text.strip())
        and isinstance(digest, str)
        and _SHA256.fullmatch(digest) is not None
        and isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count > 0
        and isinstance(truncated, bool)
        and data.get("untrusted") is True
        and observation.source.reference == digest
        and observation.source.url == url
    ):
        return False, "Public-text observation payload is malformed or lacks integrity metadata."
    if truncated:
        return False, "Truncated public text cannot establish a completed source claim."
    return _ground_text_statement(
        statement,
        answer,
        text,
        supported_detail=(
            "Claim is copied from the complete retained public text and carried by the answer."
        ),
        unsupported_detail="Claim must be copied from complete retained public text.",
    )


def _ground_thread_context_open(
    claim_kind: ClaimKind,
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    """Ground only an internal inference in one exact opened thread chunk."""

    if claim_kind is not ClaimKind.INFERENCE:
        return False, "Opened thread context is internal context and cannot support SOURCE_CLAIM."
    data = observation.data
    expected_keys = {
        "handle",
        "range_digest",
        "chunks",
        "next_ordinal",
        "source_conversation",
        "thread_root_ts",
        "policy_version",
    }
    handle = data.get("handle")
    range_digest = data.get("range_digest")
    chunks = data.get("chunks")
    next_ordinal = data.get("next_ordinal")
    source_conversation = data.get("source_conversation")
    thread_root_ts = data.get("thread_root_ts")
    try:
        canonical = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        canonical = b""
    valid_shape = bool(
        set(data) == expected_keys
        and observation.quality is EvidenceQuality.INTERNAL_CONTEXT
        and observation.source.provider == "leo_thread_context"
        and observation.source.url is None
        and isinstance(handle, str)
        and _THREAD_HANDLE.fullmatch(handle) is not None
        and isinstance(range_digest, str)
        and _SHA256.fullmatch(range_digest) is not None
        and observation.source.reference == range_digest
        and isinstance(source_conversation, str)
        and bool(source_conversation.strip())
        and isinstance(thread_root_ts, str)
        and _SLACK_TIMESTAMP.fullmatch(thread_root_ts) is not None
        and data.get("policy_version") == "thread-context-navigation-v1"
        and isinstance(chunks, list)
        and 1 <= len(chunks) <= 8
        and _SHA256.fullmatch(observation.raw_hash) is not None
        and hashlib.sha256(canonical).hexdigest() == observation.raw_hash
    )
    if not valid_shape or not isinstance(chunks, list):
        return False, "Opened thread-context observation is malformed or lacks exact provenance."
    chunk_texts: list[str] = []
    ordinals: list[int] = []
    for chunk in chunks:
        if not isinstance(chunk, dict) or set(chunk) != {
            "ordinal",
            "source_item_digest",
            "text",
        }:
            return False, "Opened thread-context chunk shape is invalid."
        ordinal = chunk.get("ordinal")
        source_item_digest = chunk.get("source_item_digest")
        text = chunk.get("text")
        if not (
            isinstance(ordinal, int)
            and not isinstance(ordinal, bool)
            and ordinal >= 0
            and isinstance(source_item_digest, str)
            and _SHA256.fullmatch(source_item_digest) is not None
            and isinstance(text, str)
            and bool(text.strip())
            and len(text) <= 1_200
        ):
            return False, "Opened thread-context chunk fields are invalid."
        ordinals.append(ordinal)
        chunk_texts.append(text)
    expected_ordinals = list(range(ordinals[0], ordinals[0] + len(ordinals)))
    if ordinals != expected_ordinals or not (
        next_ordinal is None
        or (
            isinstance(next_ordinal, int)
            and not isinstance(next_ordinal, bool)
            and next_ordinal == ordinals[-1] + 1
        )
    ):
        return False, "Opened thread-context chunk ordinals are not contiguous."
    if not _is_substantive_statement(statement) or not any(
        _contains_statement(chunk_text, statement) for chunk_text in chunk_texts
    ):
        return False, "Thread-context inference must be copied from one returned exact chunk."
    if not _contains_statement(answer, statement):
        return False, "The final answer must carry the grounded thread-context inference."
    return True, "Inference is copied from one exact authorized thread-context chunk."


def _ground_sec_filings(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    ticker = data.get("ticker")
    cik = data.get("cik")
    filings = data.get("filings")
    if not (
        isinstance(ticker, str)
        and _TICKER.fullmatch(ticker) is not None
        and isinstance(cik, str)
        and _CIK.fullmatch(cik) is not None
        and isinstance(filings, list)
        and 0 < len(filings) <= 20
        and observation.source.reference == f"submissions:{cik}"
    ):
        return False, "SEC observation payload is malformed or empty."

    normalized_filings: list[tuple[str, str, str, str, str | None]] = []
    for filing in filings:
        if not isinstance(filing, dict):
            return False, "SEC observation contains a malformed filing entry."
        form = filing.get("form")
        accession = filing.get("accession")
        filing_date = filing.get("filing_date")
        primary_document = filing.get("primary_document")
        filing_url = filing.get("filing_url")
        if not all(
            isinstance(value, str) and bool(value.strip())
            for value in (form, accession, filing_date, primary_document)
        ):
            return False, "SEC observation contains a malformed filing entry."
        if not isinstance(filing_date, str) or _ISO_DATE.fullmatch(filing_date) is None:
            return False, "SEC observation contains an invalid filing date."
        if not (
            isinstance(form, str)
            and isinstance(accession, str)
            and isinstance(primary_document, str)
        ):
            return False, "SEC observation contains a malformed filing entry."
        if filing_url is not None:
            expected_url = (
                "https://www.sec.gov/Archives/edgar/data/"
                f"{int(cik)}/{accession.replace('-', '')}/{primary_document}"
            )
            if filing_url != expected_url:
                return False, "SEC observation contains an invalid filing document URL."
        normalized_filings.append((form, accession, filing_date, primary_document, filing_url))

    company_name = data.get("company_name")
    if company_name is not None and not isinstance(company_name, str):
        return False, "SEC observation contains an invalid company name."
    matching_filings = tuple(
        (form, accession, filing_date, filing_url)
        for form, accession, filing_date, _primary_document, filing_url in normalized_filings
        if _sec_text_matches_tuple(
            statement,
            ticker=ticker,
            form=form,
            filing_date=filing_date,
            accession=accession,
            company_name=company_name,
            filing_url=filing_url,
        )
    )
    canonical = (
        f"{ticker} filed form {normalized_filings[0][0]} on "
        f"{normalized_filings[0][2]} under accession {normalized_filings[0][1]}."
    )
    if normalized_filings[0][4] is not None:
        canonical = f"{canonical} Document URL: {normalized_filings[0][4]}"
    if len(matching_filings) != 1:
        return (
            False,
            "SEC claim must contain exactly one observed ticker/form/date/accession tuple and no "
            f"unsupported assertion. Use: {canonical}",
        )
    form, accession, filing_date, filing_url = matching_filings[0]
    if not _sec_text_matches_tuple(
        answer,
        ticker=ticker,
        form=form,
        filing_date=filing_date,
        accession=accession,
        company_name=company_name,
        filing_url=filing_url,
        strict=False,
    ):
        return (
            False,
            "The final answer must conversationally carry the same exact observed "
            f"ticker={ticker}, form={form}, filing_date={filing_date}, accession={accession} "
            "tuple and no unsupported assertion.",
        )
    return (
        True,
        "Claim and answer contain one exact observed SEC ticker/form/date/accession tuple.",
    )


def _ground_exa_search(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    """Admit only an exact canonical highlight bound to the observation URL."""

    data = observation.data
    query = data.get("query")
    query_hash = data.get("query_hash")
    result_hash = data.get("result_hash")
    result = data.get("result")
    provider_result_count = data.get("provider_result_count")
    selected_result_rank = data.get("selected_result_rank")
    highlight_count = data.get("highlight_count")
    canonical = canonical_exa_highlight_statements(data)
    calculated_result_hash = exa_result_hash(data)
    result_url = result.get("url") if isinstance(result, dict) else None
    if not (
        observation.source.provider == "exa"
        and isinstance(query, str)
        and bool(query.strip())
        and isinstance(query_hash, str)
        and _SHA256.fullmatch(query_hash) is not None
        and hashlib.sha256(query.encode("utf-8")).hexdigest() == query_hash
        and isinstance(result_hash, str)
        and _SHA256.fullmatch(result_hash) is not None
        and calculated_result_hash == result_hash
        and observation.source.reference == f"search:{query_hash}:{result_hash}"
        and isinstance(result_url, str)
        and is_public_https_url(result_url)
        and observation.source.url == result_url
        and isinstance(provider_result_count, int)
        and not isinstance(provider_result_count, bool)
        and 1 <= provider_result_count <= 100
        and isinstance(selected_result_rank, int)
        and not isinstance(selected_result_rank, bool)
        and 1 <= selected_result_rank <= provider_result_count
        and canonical is not None
        and isinstance(highlight_count, int)
        and not isinstance(highlight_count, bool)
        and highlight_count == len(canonical)
        and data.get("search_type") == "auto"
        and data.get("contents_mode") == "highlights"
        and data.get("untrusted") is True
        and data.get("exact_url_bound_claims") is True
    ):
        return False, "Exa highlight evidence is malformed or lacks exact URL provenance."
    return _ground_exact_statement_list(
        statement,
        answer,
        data.get("statements"),
        canonical,
        kind="Exa URL-bound highlight",
    )


def _ground_verified_web(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    """Validate the family attempt ledger before delegating to selected evidence."""

    if not valid_verified_web_attempts(observation.data):
        return False, "Verified-web provider attempt accounting is malformed."
    selected_provider = observation.data.get("selected_provider")
    if selected_provider == "exa":
        return _ground_exa_search(statement, answer, observation)
    if selected_provider == "tavily_public_fetch":
        return _ground_public_text(statement, answer, observation)
    return False, "Verified-web evidence does not identify an admitted provider route."


def _ground_delegated_research(
    claim_kind: ClaimKind,
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    data = observation.data
    child_answer = data.get("answer")
    child_run_id = data.get("child_run_id")
    trace_event_count = data.get("trace_event_count")
    observation_count = data.get("observation_count")
    legacy_shape_valid = (
        isinstance(child_answer, str)
        and bool(child_answer.strip())
        and isinstance(child_run_id, str)
        and bool(child_run_id.strip())
        and observation.source.reference == child_run_id
        and _is_nonnegative_int(trace_event_count, minimum=1)
        and _is_nonnegative_int(observation_count)
        and data.get("truncated") is not True
    )
    if not legacy_shape_valid:
        return False, "Delegated-research observation payload is malformed or truncated."
    assert isinstance(child_answer, str)
    if data.get("schema_version") is None:
        if claim_kind is ClaimKind.SOURCE_CLAIM:
            return (
                False,
                "Legacy child prose has no verified evidence envelope and cannot support a "
                "source claim.",
            )
        return _ground_text_statement(
            statement,
            answer,
            child_answer,
            supported_detail=(
                "Inference is copied from the legacy child result and carried by the answer."
            ),
            unsupported_detail="Inference must be copied from the completed child result.",
        )
    try:
        evidence = parse_child_evidence_envelope(data)
    except ChildEvidenceError:
        return False, "Delegated-research evidence envelope is malformed or was changed."
    if not _direct_envelope_matches_observation(evidence, observation):
        return False, "Delegated-research evidence authority or expiry does not match the child."
    if claim_kind is ClaimKind.SOURCE_CLAIM:
        if not any(
            _same_statement(statement, claim.statement) for claim in evidence.verified_source_claims
        ):
            return (
                False,
                "Source claim must exactly match a source claim verified by the child harness.",
            )
        if not _contains_statement(answer, statement):
            return False, "The final answer must carry the verified child source statement."
        return (
            True,
            "Source claim exactly matches verified child evidence and is carried by the answer.",
        )
    return _ground_text_statement(
        statement,
        answer,
        evidence.answer,
        supported_detail="Inference is copied from the child result and carried by the answer.",
        unsupported_detail="Inference must be copied from the completed child result.",
    )


def _ground_research_plan(
    claim_kind: ClaimKind,
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    plan, detail = _validate_research_plan(
        observation,
        require_verified_authority=claim_kind is ClaimKind.SOURCE_CLAIM,
    )
    if plan is None:
        return False, detail

    if claim_kind is ClaimKind.SOURCE_CLAIM:
        supported = any(
            _same_statement(statement, evidence.statement) for evidence in plan.nested_evidence
        )
        if not supported:
            return (
                False,
                "Source claim must exactly match source evidence verified by a plan child.",
            )
    else:
        supported = any(
            _contains_statement(node_answer, statement) for node_answer in plan.node_answers
        )
    if not supported:
        return False, "Inference must be copied from one completed research-plan node result."
    if not _contains_statement(answer, statement):
        return False, "The final answer must carry the grounded research-plan statement."
    return (
        True,
        (
            "Source claim exactly matches verified plan-child evidence and is carried by the "
            "answer."
            if claim_kind is ClaimKind.SOURCE_CLAIM
            else "Inference is copied from a completed plan node and carried by the answer."
        ),
    )


def _validate_research_plan(
    observation: Observation,
    *,
    require_verified_authority: bool,
) -> tuple[_ValidatedResearchPlan | None, str]:
    data = observation.data
    nodes = data.get("nodes")
    completed_count = data.get("completed_count")
    failed_count = data.get("failed_count")
    blocked_count = data.get("blocked_count")
    plan_id = data.get("plan_id")
    reference_is_bound = (
        observation.source.reference == plan_id
        if isinstance(plan_id, str) and bool(plan_id.strip())
        else observation.source.reference == f"{observation.run_id}:{observation.tool_call_id}"
    )
    basic_shape_valid = (
        isinstance(data.get("goal"), str)
        and bool(str(data["goal"]).strip())
        and observation.source.provider == "leo-subagent-plan"
        and data.get("status") == "completed"
        and isinstance(nodes, list)
        and bool(nodes)
        and _is_nonnegative_int(completed_count, minimum=1)
        and _is_nonnegative_int(failed_count)
        and _is_nonnegative_int(blocked_count)
        and completed_count == len(nodes)
        and failed_count == 0
        and blocked_count == 0
        and data.get("truncated") is not True
    )
    if not basic_shape_valid:
        return None, "Research-plan observation payload is malformed, partial, or truncated."
    if require_verified_authority and not (
        observation.status is ObservationStatus.RETRIEVED
        and observation.quality is EvidenceQuality.VERIFIED_CHILD
        and observation.schema_version == "observation-v2"
        and reference_is_bound
    ):
        return None, "Research-plan source authority, schema, or plan reference is invalid."

    assert isinstance(nodes, list)
    node_answers: list[str] = []
    nested_evidence: list[_NestedPlanEvidence] = []
    child_evidence_expiries = []
    node_ids: set[str] = set()
    child_run_ids: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict):
            return None, "Research-plan observation contains a malformed node result."
        node_id = node.get("id")
        node_answer = node.get("answer")
        child_run_id = node.get("child_run_id")
        trace_event_count = node.get("trace_event_count")
        if not (
            isinstance(node_id, str)
            and bool(node_id.strip())
            and node_id not in node_ids
            and node.get("status") == "completed"
            and isinstance(node_answer, str)
            and bool(node_answer.strip())
            and isinstance(child_run_id, str)
            and bool(child_run_id.strip())
            and child_run_id not in child_run_ids
        ):
            return None, "Research-plan observation contains a malformed node result."
        node_ids.add(node_id)
        child_run_ids.add(child_run_id)
        node_answers.append(node_answer)

        raw_evidence = node.get("child_evidence")
        if raw_evidence is None:
            # Compatibility-only durable rows can still inform inference. They never
            # acquire source authority merely by being stored inside a completed plan.
            if trace_event_count is not None and not _is_nonnegative_int(
                trace_event_count, minimum=1
            ):
                return None, "Research-plan observation contains malformed legacy trace data."
            continue
        try:
            envelope = parse_child_evidence_envelope(raw_evidence)
        except ChildEvidenceError:
            return None, "Research-plan node evidence is malformed or was changed."
        if not _plan_node_matches_envelope(node, envelope):
            return None, "Research-plan node metadata does not match its child evidence."
        for claim in envelope.verified_source_claims:
            for source in claim.sources:
                arguments = _nested_source_arguments(
                    kind=source.kind,
                    provider=source.provider,
                    reference=source.reference,
                    statement=claim.statement,
                )
                if (
                    _SHA256.fullmatch(source.raw_hash) is None
                    or source.observed_at > observation.observed_at
                    or (source.expires_at is not None and source.expires_at <= source.observed_at)
                    or arguments is None
                ):
                    return None, "Research-plan child source integrity or provenance is invalid."
                projected_id = (
                    f"nested:{observation.id}:{envelope.child_run_id}:"
                    f"{claim.claim_id}:{source.observation_id}"
                )
                nested_evidence.append(
                    _NestedPlanEvidence(
                        parent_observation_id=observation.id,
                        child_run_id=envelope.child_run_id,
                        claim_id=claim.claim_id,
                        statement=claim.statement,
                        observation=Observation(
                            id=projected_id,
                            scope=observation.scope,
                            run_id=observation.run_id,
                            tool_call_id=observation.tool_call_id,
                            kind=source.kind,
                            data=arguments,
                            source=SourceRef(
                                provider=source.provider,
                                reference=source.reference,
                                url=source.url,
                            ),
                            observed_at=source.observed_at,
                            expires_at=source.expires_at,
                            raw_hash=source.raw_hash,
                            status=ObservationStatus.RETRIEVED,
                            quality=EvidenceQuality.VERIFIED_CHILD,
                            schema_version="observation-v2",
                            normalization_version="child-evidence-projection-v1",
                        ),
                    )
                )
        expiry = child_evidence_expires_at(envelope)
        if expiry is not None:
            child_evidence_expiries.append(expiry)

    expected_expiry = min(child_evidence_expiries) if child_evidence_expiries else None
    if observation.expires_at != expected_expiry:
        return None, "Research-plan evidence expiry does not match its verified child evidence."
    return (
        _ValidatedResearchPlan(
            node_answers=tuple(node_answers),
            nested_evidence=tuple(nested_evidence),
        ),
        "Research-plan evidence is valid.",
    )


def _nested_source_arguments(
    *,
    kind: str,
    provider: str,
    reference: str,
    statement: str,
) -> dict[str, JsonValue] | None:
    """Recover only deterministic routing arguments from canonical child statements."""

    if kind == "market.get_quote":
        match = _CHILD_QUOTE_STATEMENT.match(" ".join(statement.split()))
        if match is None:
            return None
        symbol = match.group("symbol")
        production_reference = valid_equity_quote_provenance(
            provider=provider,
            reference=reference,
            symbol=symbol,
        )
        fixture_reference = provider == "fixture" and reference == f"fixture-quote-{symbol}"
        if not (production_reference or fixture_reference):
            return None
        return {"symbol": symbol}
    if kind == "market.get_crypto_snapshot":
        match = _CHILD_CRYPTO_REFERENCE.fullmatch(reference)
        if match is None or provider != "crypto-corroboration":
            return None
        return {
            "asset_id": match.group("asset"),
            "quote_currency": match.group("currency"),
        }
    if kind == "sec.get_recent_filings":
        match = _CHILD_SEC_STATEMENT.fullmatch(" ".join(statement.split()))
        if (
            match is None
            or provider != "sec-edgar"
            or _CHILD_SEC_REFERENCE.fullmatch(reference) is None
        ):
            return None
        return {"ticker": match.group("ticker")}
    return {}


def _direct_envelope_matches_observation(
    evidence: ChildEvidenceEnvelope,
    observation: Observation,
) -> bool:
    return (
        observation.source.provider == "leo-subagent"
        and observation.source.reference == evidence.child_run_id
        and observation.expires_at == child_evidence_expires_at(evidence)
    )


def _plan_node_matches_envelope(
    node: Mapping[str, object],
    evidence: ChildEvidenceEnvelope,
) -> bool:
    return (
        node.get("answer") == evidence.answer
        and node.get("child_run_id") == evidence.child_run_id
        and node.get("trace_event_count") == evidence.trace_event_count
        and node.get("observation_count") == evidence.observation_count
    )


def _ground_text_statement(
    statement: str,
    answer: str,
    source_text: str,
    *,
    supported_detail: str,
    unsupported_detail: str,
) -> tuple[bool, str]:
    if not _is_substantive_statement(statement) or not _contains_statement(source_text, statement):
        return False, unsupported_detail
    if not _contains_statement(answer, statement):
        return False, "The final answer must carry the grounded source statement."
    return True, supported_detail


def _contains_statement(text: str, statement: str) -> bool:
    normalized_text = " ".join(text.split()).casefold()
    normalized_statement = " ".join(statement.split()).casefold()
    return bool(normalized_statement) and normalized_statement in normalized_text


def _same_statement(actual: str, expected: str) -> bool:
    return " ".join(actual.split()).casefold() == " ".join(expected.split()).casefold()


def _is_substantive_statement(statement: str) -> bool:
    return len(re.findall(r"\w+", statement, flags=re.UNICODE)) >= 2


def _is_nonnegative_int(value: object, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _cardinality_check(
    *,
    name: str,
    count: int,
    minimum: int,
    maximum: int,
) -> VerifierCheck:
    passed = minimum <= count <= maximum
    return VerifierCheck(
        name=name,
        passed=passed,
        detail=(
            f"Observed count {count} is within trusted bounds [{minimum}, {maximum}]."
            if passed
            else f"Observed count {count} violates trusted bounds [{minimum}, {maximum}]."
        ),
    )


def _sec_text_matches_tuple(
    text: str,
    *,
    ticker: str,
    form: str,
    filing_date: str,
    accession: str,
    company_name: str | None,
    filing_url: str | None,
    strict: bool = True,
) -> bool:
    """Accept bounded conversational wording without accepting a new factual clause."""

    if not (
        _contains_symbol(text, ticker)
        and _contains_literal(text, form)
        and _contains_literal(text, filing_date)
        and _contains_literal(text, accession)
    ):
        return False
    if set(_SEC_ISO_DATE_TOKEN.findall(text)) != {filing_date}:
        return False
    if set(_SEC_ACCESSION.findall(text)) != {accession}:
        return False
    urls = tuple(item.rstrip(".,;:!?") for item in _HTTP_URL.findall(text))
    if len(urls) > 1 or (urls and (filing_url is None or urls[0] != filing_url)):
        return False
    if re.search(r"\b(?:document\s+)?url\b", text, re.IGNORECASE) is not None and not urls:
        return False
    form_tokens = {item.upper() for item in _SEC_FORM_TOKEN.findall(text)}
    if form_tokens and form_tokens != {form.upper()}:
        return False
    allowed_uppercase = {
        ticker.upper(),
        "EDGAR",
        "SEC",
        "URL",
        *(word.upper() for word in _SEC_CONVERSATIONAL_WORDS),
    }
    if company_name is not None:
        allowed_uppercase.update(company_name.upper().split())
    if set(_UPPERCASE_WORD.findall(text)) - allowed_uppercase:
        return False
    if not strict:
        return True

    # Remove the already-validated URL as one opaque token before removing the
    # ticker/date fields.  Those values can legitimately occur inside the SEC
    # document path (for example ``nvda-20260817.htm``); removing them first
    # would fragment the URL and make its harmless path words look like claims.
    remainder = _HTTP_URL.sub(" ", text)
    for value in (ticker, form, filing_date, accession, company_name):
        if value:
            remainder = re.sub(re.escape(value), " ", remainder, flags=re.IGNORECASE)
    words = set(re.findall(r"[a-z]+", remainder.casefold()))
    return words.issubset(_SEC_CONVERSATIONAL_WORDS)


def _contains_literal(text: str, value: str) -> bool:
    return (
        re.search(
            rf"(?<![A-Za-z0-9]){re.escape(value)}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        )
        is not None
    )


_NUMBER_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])[-+]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?"
    r"(?![A-Za-z0-9_%]|\.\d)"
)
_STANDALONE_NUMBER = re.compile(
    r"[-+]?(?:[$\u00a3\u20ac]\u00a0?)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?"
)
_COMPLETE_FENCED_CODE = re.compile(r"```[^\n]*\n[\s\S]*\n```")
_COMPLETE_INLINE_CODE = re.compile(r"`[^`\r\n]+`")
_COMPLETE_SEMICOLON_CODE = re.compile(
    r"(?is)(?:(?:SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|DROP|GRANT|REVOKE)\b.+|"
    r"(?:const|let|var|return|raise|import|from)\b.+|"
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\([^;\r\n]*\)|"
    r"[A-Za-z_]\w*\s*=\s*[^;\r\n]+);"
)
_MARKDOWN_LIST_ITEM = re.compile(r"\s*(?:[-*+] |\d+[.)] )\S")
_UNORDERED_BULLET_ITEM = re.compile(r"\s*(?:[-*+]\s+|\u2022\s*)\S")
_ANSWER_WORD = re.compile(r"[^\W_]+(?:[-'\u2019][^\W_]+)*", re.UNICODE)
_DECIMAL_POINT = re.compile(r"(?<=\d)\.(?=\d)")
_DOTTED_INITIALISM = re.compile(r"(?:\b[A-Za-z]\.){2,}")
_COMMON_ABBREVIATION = re.compile(
    r"\b(?:e\.g|i\.e|mr|mrs|ms|dr|prof|sr|jr|vs)\.",
    re.IGNORECASE,
)
_SMALL_NUMBER_TOKEN = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,3})"
_SMALL_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
_BULLET_COUNT_REQUEST = re.compile(
    rf"\b(?:(?:in|as|with)\s+)?(?:exactly\s+)?(?P<count>{_SMALL_NUMBER_TOKEN})\s+"
    r"(?:(?:brief|clear|concise|short)\s+){0,3}bullet(?:s|\s+points?)\b",
    re.IGNORECASE,
)
_SENTENCE_COUNT_REQUEST = re.compile(
    rf"\b(?:(?:in|as|with)\s+)?(?:exactly\s+)?(?P<count>{_SMALL_NUMBER_TOKEN})\s+"
    r"(?:(?:brief|clear|concise|short)\s+){0,3}sentences?\b",
    re.IGNORECASE,
)
_NAME_COUNT_REQUEST = re.compile(
    rf"\b(?:exactly\s+)?(?P<count>{_SMALL_NUMBER_TOKEN})\s+"
    r"(?:[^\W\d_]+[ -]+){0,4}(?:code[ -]+names?|codenames?|names?)\b",
    re.IGNORECASE,
)
_STRICT_WORD_CAP = re.compile(
    rf"\b(?:under|fewer\s+than)\s+(?P<count>{_SMALL_NUMBER_TOKEN})\s+words?\b",
    re.IGNORECASE,
)
_INCLUSIVE_WORD_CAP = re.compile(
    rf"\b(?:at[- ]most|no\s+more\s+than|up\s+to|maximum\s+of)\s+"
    rf"(?P<count>{_SMALL_NUMBER_TOKEN})\s+words?\b",
    re.IGNORECASE,
)
_DANGLING_TERMINAL_PUNCTUATION = re.compile(r"(?:[,;:]|--?|[\u2013\u2014]|[([{]|=>|[&|/\\])$")
_DANGLING_TERMINAL_CONNECTIVE = re.compile(
    r"(?i)(?:^|\s)(and|or|but|because|although|whereas|including|namely|"
    r"either|neither|such\s+as|as\s+well\s+as|rather\s+than|for\s+example)$"
)
_TERMINAL_DISCLAIMER_PARENTHETICAL = re.compile(
    r"\(\s*(?:note|disclaimer|caveat)\s*:[^()]*\)",
    re.IGNORECASE,
)
_OUTPUT_INTRODUCTION = re.compile(
    r"^\s*(?:"
    r"here(?:\s+(?:are|is)|'s)\s+"
    r"(?:(?:a|an|one|some|several|two|three|a couple of|a few)\s+)?"
    r"(?:(?:brief|concise|quick|short|simple|small)\s+)?"
    r"(?:buckets?|options?|ideas?|examples?|recommendations?|alternatives?|candidates?|"
    r"names?|stocks?|investments?|choices?|approaches?|mix|"
    r"(?:short|watch|wish|check|bucket)?lists?)"
    r"(?:\s+of\s+[^:.!?,]+)?"
    r"(?:\s*,?\s*(?:each\s+with|with\s+one|for\s+each)\b[^:.!?]*)?"
    r"(?:\s+(?:to consider|"
    r"worth considering|for comparison))?"
    r"|(?:(?:some|several|two|three|a couple of|a few)|potential|possible)\s+"
    r"(?:options?|ideas?|examples?|"
    r"recommendations?|alternatives?|candidates?|names?|stocks?|choices?)\s+"
    r"(?:(?:to consider|worth considering)\b|(?:to consider\s+)?(?:include|are)"
    r"(?:\s+the following)?)"
    r"|there are\s+(?:(?:some|several|two|three|a few)\s+)?(?:options?|ideas?|examples?|"
    r"alternatives?|candidates?|choices?)\s+(?:worth considering|to consider|available)"
    r"|i (?:would|might|could)\s+consider(?:\s+the following)?(?:\s+(?:options?|ideas?|"
    r"examples?|alternatives?|candidates?|names?|stocks?|choices?))?"
    r")\b",
    re.IGNORECASE,
)
_INTRODUCTORY_QUALIFIER = re.compile(
    r"^\s*,?\s*(?:based on|using|given|drawing on|depending on)\b[^.!?:]*(?:[.!?]+|$)",
    re.IGNORECASE,
)
_CLARIFICATION_PREAMBLE = re.compile(
    r"\s*(?:i (?:need|would need|am missing|don't have) (?:a little |some |two |three )?"
    r"(?:detail|details|information|context)|to (?:answer|compare|complete|continue|help) "
    r"(?:accurately|that|this)|before i (?:answer|compare|continue)|"
    r"a (?:quick|brief) clarification)\s*",
    re.IGNORECASE,
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_THREAD_HANDLE = re.compile(r"thr_[0-9a-f]{32}")
_SLACK_TIMESTAMP = re.compile(r"[0-9]+\.[0-9]+")
_TICKER = re.compile(r"[A-Z][A-Z0-9.-]{0,7}")
_CIK = re.compile(r"\d{10}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2}")
_CHILD_QUOTE_STATEMENT = re.compile(
    r"(?P<symbol>[A-Z][A-Z0-9.-]{0,7}) is quoted at "
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?: USD)?\."
)
_CHILD_SEC_STATEMENT = re.compile(
    r"(?P<ticker>[A-Z][A-Z0-9.-]{0,7}) filed form [A-Za-z0-9-]+ on "
    r"\d{4}-\d{2}-\d{2} under accession \d{10}-\d{2}-\d{6}\."
)
_CHILD_SEC_REFERENCE = re.compile(r"submissions:\d{10}")
_CHILD_CRYPTO_REFERENCE = re.compile(
    r"snapshot:(?P<asset>[a-z0-9]+(?:-[a-z0-9]+)*):"
    r"(?P<currency>USD|EUR|GBP|JPY):[0-9a-f]{64}"
)
_SEC_ISO_DATE_TOKEN = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")
_SEC_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_HTTP_URL = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)
_SEC_FORM_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:[0-9]{1,2}-(?:K|Q|F)|[A-Z]-[0-9]{1,3})(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)
_UPPERCASE_WORD = re.compile(r"\b[A-Z]{2,5}\b")
_SEC_CONVERSATIONAL_WORDS = frozenset(
    {
        "a",
        "about",
        "accession",
        "according",
        "an",
        "and",
        "as",
        "at",
        "based",
        "company",
        "date",
        "dated",
        "document",
        "edgar",
        "filed",
        "filing",
        "for",
        "found",
        "form",
        "from",
        "has",
        "here",
        "i",
        "in",
        "is",
        "its",
        "latest",
        "listed",
        "lists",
        "metadata",
        "most",
        "newest",
        "number",
        "on",
        "recent",
        "record",
        "records",
        "reported",
        "reports",
        "research",
        "result",
        "s",
        "sec",
        "shows",
        "submission",
        "the",
        "under",
        "url",
        "was",
        "with",
    }
)


def _contains_symbol(text: str, symbol: str) -> bool:
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_.-]){re.escape(symbol)}(?![A-Za-z0-9_.-])",
        re.IGNORECASE,
    )
    return pattern.search(text) is not None


def _contains_numeric_value(text: str, expected: int | float) -> bool:
    try:
        expected_decimal = Decimal(str(expected))
    except InvalidOperation:
        return False
    if not expected_decimal.is_finite():
        return False
    for match in _NUMBER_TOKEN.finditer(text):
        try:
            candidate = Decimal(match.group(0).replace(",", ""))
        except InvalidOperation:
            continue
        if candidate == expected_decimal:
            return True
    return False


def _canonical_statement(
    fallback: str,
    observation_ids: tuple[str, ...],
    observations: dict[str, Observation],
) -> str:
    quote_observations = tuple(
        observations[item]
        for item in observation_ids
        if item in observations and observations[item].kind == "market.get_quote"
    )
    if not quote_observations:
        return fallback
    quote = quote_observations[0]
    symbol = quote.data.get("symbol")
    price = quote.data.get("price")
    if not isinstance(symbol, str) or not isinstance(price, int | float):
        return fallback
    currency = quote.data.get("currency")
    del currency
    canonical = canonical_equity_quote_statement(quote.data)
    if canonical is None:
        return fallback
    diagnostics = (
        canonical_equity_quote_disagreement_statement(quote.data),
        canonical_equity_quote_time_skew_statement(quote.data),
    )
    return " ".join((canonical, *(item for item in diagnostics if item is not None)))
