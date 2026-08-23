from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import JsonValue, ValidationError

from leo.harness.child_evidence import (
    build_child_evidence_envelope,
    child_evidence_data,
)
from leo.harness.models import (
    CandidateClaim,
    CardinalityBounds,
    Claim,
    ClaimKind,
    CompletionContract,
    CompletionProposal,
    EvidenceQuality,
    EvidenceToolRequirement,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolArgumentConstraint,
    VerificationOutcome,
    VerifierStatus,
)
from leo.harness.research import ResearchRequirement
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator

NOW = datetime(2026, 8, 20, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")

WEB_STATEMENT = "Revenue increased by 20 percent."
WEB_DIGEST = "a" * 64
WEB_DATA: dict[str, JsonValue] = {
    "url": "https://example.test/report",
    "content_type": "text/plain",
    "text": f"Company update. {WEB_STATEMENT} Demand remained stable.",
    "content_sha256": WEB_DIGEST,
    "byte_count": 84,
    "truncated": False,
    "untrusted": True,
}
WEB_SOURCE = SourceRef(
    provider="public-web",
    reference=WEB_DIGEST,
    url="https://example.test/report",
)

SEC_STATEMENT = "NVDA filed form 10-Q on 2026-05-28 under accession 0001045810-26-000123."
SEC_FILING_URL = (
    "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000123/nvda-20260426.htm"
)
SEC_DATA: dict[str, JsonValue] = {
    "ticker": "NVDA",
    "cik": "0001045810",
    "filings": [
        {
            "form": "10-Q",
            "accession": "0001045810-26-000123",
            "filing_date": "2026-05-28",
            "primary_document": "nvda-20260426.htm",
            "filing_url": SEC_FILING_URL,
        }
    ],
}
SEC_SOURCE = SourceRef(
    provider="sec-edgar",
    reference="submissions:0001045810",
    url="https://data.sec.gov/submissions/CIK0001045810.json",
)

DELEGATE_STATEMENT = "Demand remains strong in data centers."


def _verified_child_data(
    *,
    child_run_id: str,
    statement: str,
    answer: str,
    source_claim: bool = True,
) -> dict[str, JsonValue]:
    child_observation = Observation(
        id=f"obs:{child_run_id}",
        scope=SCOPE,
        run_id=child_run_id,
        tool_call_id=f"call:{child_run_id}",
        kind="web.fetch_public_text",
        data={"text": statement},
        source=SourceRef(provider="fixture-child-source", reference=f"source:{child_run_id}"),
        observed_at=NOW,
        raw_hash="b" * 64,
    )
    claims = (
        (
            Claim(
                id=f"claim:{child_run_id}",
                scope=SCOPE,
                run_id=child_run_id,
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(child_observation.id,),
            ),
        )
        if source_claim
        else ()
    )
    return child_evidence_data(
        build_child_evidence_envelope(
            child_run_id=child_run_id,
            answer=answer,
            trace_event_count=9,
            observations=(child_observation,) if source_claim else (),
            claims=claims,
        )
    )


DELEGATE_DATA = _verified_child_data(
    child_run_id="subrun-001",
    statement=DELEGATE_STATEMENT,
    answer=f"The delegated review concluded: {DELEGATE_STATEMENT}",
)
DELEGATE_SOURCE = SourceRef(provider="leo-subagent", reference="subrun-001")

PLAN_STATEMENT = "Supply constraints are easing this quarter."
PLAN_CHILD_DATA = _verified_child_data(
    child_run_id="subrun-supply",
    statement=PLAN_STATEMENT,
    answer=f"The supply review found: {PLAN_STATEMENT}",
)
PLAN_DATA: dict[str, JsonValue] = {
    "plan_id": "plan-fixture",
    "goal": "Review demand and supply.",
    "status": "completed",
    "nodes": [
        {
            "id": "supply",
            "status": "completed",
            "answer": f"The supply review found: {PLAN_STATEMENT}",
            "child_run_id": "subrun-supply",
            "trace_event_count": 9,
            "observation_count": 1,
            "child_evidence": PLAN_CHILD_DATA,
        }
    ],
    "completed_count": 1,
    "failed_count": 0,
    "blocked_count": 0,
}
PLAN_SOURCE = SourceRef(provider="leo-subagent-plan", reference="plan-fixture")

VALID_CASES = (
    ("web.fetch_public_text", WEB_DATA, WEB_SOURCE, WEB_STATEMENT),
    ("sec.get_recent_filings", SEC_DATA, SEC_SOURCE, SEC_STATEMENT),
    ("agent.delegate_research", DELEGATE_DATA, DELEGATE_SOURCE, DELEGATE_STATEMENT),
    ("agent.execute_research_plan", PLAN_DATA, PLAN_SOURCE, PLAN_STATEMENT),
)


def _bundle(observation: Observation | None = None) -> RunBundle:
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="fixture-thread"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective="Research the claim")
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return RunBundle(
        thread=thread,
        task=task,
        run=run,
        observations=() if observation is None else (observation,),
    )


def _verify(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    statement: str,
    *,
    answer: str | None = None,
    claim_kind: ClaimKind = ClaimKind.SOURCE_CLAIM,
    expires_at: datetime | None = None,
    relax_integration_grounding: bool = False,
) -> VerificationOutcome:
    observation = Observation(
        id="obs-1",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-1",
        kind=kind,
        data=data,
        source=source,
        observed_at=NOW,
        expires_at=expires_at,
        raw_hash="fixture-hash",
        quality=(
            EvidenceQuality.VERIFIED_CHILD
            if kind == "agent.execute_research_plan"
            else EvidenceQuality.PROVIDER_REPORTED
        ),
    )
    proposal = CompletionProposal(
        answer=answer or f"Research result: {statement}",
        claims=(
            CandidateClaim(
                kind=claim_kind,
                statement=statement,
                observation_ids=(observation.id,),
            ),
        ),
    )
    return DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        require_source_claim=claim_kind is ClaimKind.SOURCE_CLAIM,
        relax_integration_grounding=relax_integration_grounding,
    ).verify(proposal, _bundle(observation))


@pytest.mark.parametrize(("kind", "data", "source", "statement"), VALID_CASES)
def test_generic_observation_grounding_accepts_payload_supported_statements(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    statement: str,
) -> None:
    outcome = _verify(kind, data, source, statement)
    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert statement in outcome.completion.answer


@pytest.mark.parametrize(("kind", "data", "source", "statement"), VALID_CASES)
def test_generic_observation_grounding_requires_statement_in_final_answer(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    statement: str,
) -> None:
    outcome = _verify(kind, data, source, statement, answer="A different unsupported summary.")
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None
    support = next(check for check in outcome.result.checks if check.name.endswith("_supported"))
    assert support.passed is False
    assert "answer" in support.detail.lower()


def test_relaxed_integration_grounding_accepts_model_synthesis_without_copy_exact_text() -> None:
    outcome = _verify(
        "web.fetch_public_text",
        WEB_DATA,
        WEB_SOURCE,
        WEB_STATEMENT,
        answer="The model's concise consolidated summary.",
        relax_integration_grounding=True,
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert outcome.completion.answer == "The model's concise consolidated summary."


# A plan node with no verified nested source claims (context-only child), used for the
# INFERENCE-paraphrase-acceptance case below. A plan node carrying verified nested
# source-claim evidence (PLAN_DATA itself) triggers a separate, unrelated coverage
# requirement (_research_plan_coverage_checks) that always demands an exact-copy
# SOURCE_CLAIM citing every verified nested statement whenever the plan observation is
# cited at all -- by design, regardless of relaxation. That mechanism is untouched by
# this fix, so it is exercised deliberately (not accidentally) by the rejection test
# below rather than the acceptance test.
PLAN_DATA_WITHOUT_NESTED_EVIDENCE: dict[str, JsonValue] = deepcopy(PLAN_DATA)
_plan_node_without_nested_evidence = PLAN_DATA_WITHOUT_NESTED_EVIDENCE["nodes"][0]
assert isinstance(_plan_node_without_nested_evidence, dict)
_plan_node_without_nested_evidence.pop("child_evidence")

INFERENCE_PARAPHRASE_ACCEPT_CASES = (
    (
        "agent.delegate_research",
        DELEGATE_DATA,
        DELEGATE_SOURCE,
        "Data center demand is holding up well.",
    ),
    (
        "agent.execute_research_plan",
        PLAN_DATA_WITHOUT_NESTED_EVIDENCE,
        PLAN_SOURCE,
        "Supply-side constraints appear to be loosening this quarter.",
    ),
)

SOURCE_CLAIM_PARAPHRASE_REJECT_CASES = (
    (
        "agent.delegate_research",
        DELEGATE_DATA,
        DELEGATE_SOURCE,
        "Data center demand is holding up well.",
    ),
    (
        "agent.execute_research_plan",
        PLAN_DATA,
        PLAN_SOURCE,
        "Supply-side constraints appear to be loosening this quarter.",
    ),
)


@pytest.mark.parametrize(
    ("kind", "data", "source", "paraphrase"),
    INFERENCE_PARAPHRASE_ACCEPT_CASES,
)
def test_relaxed_integration_grounding_accepts_delegated_research_inference_paraphrase(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    paraphrase: str,
) -> None:
    """Paraphrased INFERENCE claims over delegated research get the same relaxation.

    A parent agent that delegates research and then synthesizes the child's verified
    findings in its own words should get the same paraphrase relaxation as
    directly-fetched market/web/sec evidence. Regression test for a gap where
    'agent.'-kind observations were never recognized as relaxed integration
    observations, so a correct paraphrase of verified child research failed
    verification even with relax_integration_grounding=True.
    """
    outcome = _verify(
        kind,
        data,
        source,
        paraphrase,
        claim_kind=ClaimKind.INFERENCE,
        answer=f"Summary: {paraphrase}",
        relax_integration_grounding=True,
    )
    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None

    # Without relaxation, the exact same paraphrase must still fail closed -- proving
    # the PASS above comes from the relaxation path, not from a lenient match rule.
    strict_outcome = _verify(
        kind,
        data,
        source,
        paraphrase,
        claim_kind=ClaimKind.INFERENCE,
        answer=f"Summary: {paraphrase}",
        relax_integration_grounding=False,
    )
    assert strict_outcome.result.status is VerifierStatus.FAIL


@pytest.mark.parametrize(
    ("kind", "data", "source", "paraphrase"),
    SOURCE_CLAIM_PARAPHRASE_REJECT_CASES,
)
def test_relaxed_integration_grounding_still_rejects_delegated_research_source_claim_paraphrase(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    paraphrase: str,
) -> None:
    """Safety-preserving regression test.

    The INFERENCE-only relaxation above must never extend to SOURCE_CLAIM. A source
    claim asserts exact provenance from a child research task's verified evidence, so
    a paraphrase must still fail even with relax_integration_grounding=True -- otherwise
    the model could misattribute or subtly alter what the child actually found.
    """
    outcome = _verify(
        kind,
        data,
        source,
        paraphrase,
        claim_kind=ClaimKind.SOURCE_CLAIM,
        answer=f"Summary: {paraphrase}",
        relax_integration_grounding=True,
    )
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None


MALFORMED_CASES = (
    (
        "web.fetch_public_text",
        {**WEB_DATA, "truncated": True},
        WEB_SOURCE,
        WEB_STATEMENT,
    ),
    (
        "sec.get_recent_filings",
        {**SEC_DATA, "filings": [{"form": "10-Q"}]},
        SEC_SOURCE,
        SEC_STATEMENT,
    ),
    (
        "agent.delegate_research",
        {**DELEGATE_DATA, "trace_event_count": True},
        DELEGATE_SOURCE,
        DELEGATE_STATEMENT,
    ),
    (
        "agent.execute_research_plan",
        {**PLAN_DATA, "status": "partial"},
        PLAN_SOURCE,
        PLAN_STATEMENT,
    ),
)


@pytest.mark.parametrize(("kind", "data", "source", "statement"), MALFORMED_CASES)
def test_malformed_or_truncated_generic_payloads_fail_closed(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    statement: str,
) -> None:
    outcome = _verify(kind, deepcopy(data), source, statement)
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None


_RELAXED_MALFORMED_CASES = tuple(
    case for case in MALFORMED_CASES if case[0].startswith(("market.", "web.", "sec."))
)


@pytest.mark.parametrize(("kind", "data", "source", "statement"), _RELAXED_MALFORMED_CASES)
def test_malformed_or_truncated_relaxed_integration_payloads_still_fail_closed(
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    statement: str,
) -> None:
    """relax_integration_grounding excuses non-exact wording, never a corrupt payload.

    Regression test for a real bug: relaxation must not convert a payload-integrity
    rejection (truncated/malformed/tampered) into an accepted claim just because the
    observation's kind happens to be a relaxed-integration one. Both of Leo's live
    conversational entrypoints set relax_integration_grounding=True unconditionally,
    so this combination is not a hypothetical -- it's the production configuration.
    """

    outcome = _verify(kind, deepcopy(data), source, statement, relax_integration_grounding=True)
    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None


def test_generic_citation_without_payload_support_is_rejected() -> None:
    outcome = _verify(
        "agent.delegate_research",
        DELEGATE_DATA,
        DELEGATE_SOURCE,
        "Margins doubled overnight.",
    )
    assert outcome.result.status is VerifierStatus.FAIL


def test_context_only_child_prose_can_support_inference_but_never_source_claim() -> None:
    statement = "This child result is contextual analysis only."
    context_only = _verified_child_data(
        child_run_id="subrun-context",
        statement=statement,
        answer=statement,
        source_claim=False,
    )
    source = SourceRef(provider="leo-subagent", reference="subrun-context")

    assert (
        _verify(
            "agent.delegate_research",
            context_only,
            source,
            statement,
        ).result.status
        is VerifierStatus.FAIL
    )
    assert (
        _verify(
            "agent.delegate_research",
            context_only,
            source,
            statement,
            claim_kind=ClaimKind.INFERENCE,
        ).result.status
        is VerifierStatus.PASS
    )


def test_extra_child_prose_cannot_self_attest_as_a_source_claim() -> None:
    extra = "An uncited forecast says margins will double."
    data = _verified_child_data(
        child_run_id="subrun-mixed",
        statement=DELEGATE_STATEMENT,
        answer=f"{DELEGATE_STATEMENT} {extra}",
    )

    assert (
        _verify(
            "agent.delegate_research",
            data,
            SourceRef(provider="leo-subagent", reference="subrun-mixed"),
            extra,
        ).result.status
        is VerifierStatus.FAIL
    )
    assert (
        _verify(
            "agent.delegate_research",
            data,
            SourceRef(provider="leo-subagent", reference="subrun-mixed"),
            extra,
            claim_kind=ClaimKind.INFERENCE,
        ).result.status
        is VerifierStatus.PASS
    )


def test_changed_child_evidence_digest_fails_closed() -> None:
    changed = deepcopy(DELEGATE_DATA)
    changed["answer"] = f"{DELEGATE_STATEMENT} Forged addition."
    outcome = _verify(
        "agent.delegate_research",
        changed,
        DELEGATE_SOURCE,
        DELEGATE_STATEMENT,
    )
    assert outcome.result.status is VerifierStatus.FAIL


def test_plan_node_cannot_diverge_from_its_verified_child_envelope() -> None:
    changed = deepcopy(PLAN_DATA)
    node = changed["nodes"][0]
    assert isinstance(node, dict)
    node["answer"] = f"{PLAN_STATEMENT} Forged plan synthesis."

    outcome = _verify(
        "agent.execute_research_plan",
        changed,
        PLAN_SOURCE,
        PLAN_STATEMENT,
    )
    assert outcome.result.status is VerifierStatus.FAIL


def test_legacy_child_and_plan_results_remain_inference_only() -> None:
    legacy_child: dict[str, JsonValue] = {
        "answer": DELEGATE_STATEMENT,
        "child_run_id": "legacy-child",
        "trace_event_count": 2,
        "observation_count": 0,
    }
    child_source = SourceRef(provider="leo-subagent", reference="legacy-child")
    legacy_plan = deepcopy(PLAN_DATA)
    node = legacy_plan["nodes"][0]
    assert isinstance(node, dict)
    node.pop("child_evidence")
    plan_source = SourceRef(provider="leo-subagent-plan", reference="legacy-plan")

    assert (
        _verify(
            "agent.delegate_research",
            legacy_child,
            child_source,
            DELEGATE_STATEMENT,
        ).result.status
        is VerifierStatus.FAIL
    )
    assert (
        _verify(
            "agent.delegate_research",
            legacy_child,
            child_source,
            DELEGATE_STATEMENT,
            claim_kind=ClaimKind.INFERENCE,
        ).result.status
        is VerifierStatus.PASS
    )
    assert (
        _verify(
            "agent.execute_research_plan",
            legacy_plan,
            plan_source,
            PLAN_STATEMENT,
        ).result.status
        is VerifierStatus.FAIL
    )
    assert (
        _verify(
            "agent.execute_research_plan",
            legacy_plan,
            plan_source,
            PLAN_STATEMENT,
            claim_kind=ClaimKind.INFERENCE,
        ).result.status
        is VerifierStatus.PASS
    )


def test_sec_fields_do_not_attest_an_extra_unsupported_assertion() -> None:
    outcome = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        f"{SEC_STATEMENT} This guarantees future revenue growth.",
    )
    assert outcome.result.status is VerifierStatus.FAIL


def test_sec_grounding_accepts_conversational_exact_tuple_in_claim_and_answer() -> None:
    statement = (
        "The latest SEC filing for NVDA is form 10-Q, dated 2026-05-28, "
        "with accession 0001045810-26-000123."
    )
    answer = (
        "Here is the latest SEC filing metadata for NVDA: form 10-Q, dated 2026-05-28, "
        "accession 0001045810-26-000123."
    )

    outcome = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        statement,
        answer=answer,
    )

    assert outcome.result.status is VerifierStatus.PASS


def test_sec_grounding_accepts_only_the_derived_exact_document_url() -> None:
    statement = f"{SEC_STATEMENT} Document URL: {SEC_FILING_URL}"

    accepted = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        statement,
        answer=statement,
    )
    forged = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        f"{SEC_STATEMENT} Document URL: https://attacker.test/forged-filing.htm",
    )
    malformed_data = deepcopy(SEC_DATA)
    filings = malformed_data["filings"]
    assert isinstance(filings, list)
    filing = filings[0]
    assert isinstance(filing, dict)
    filing["filing_url"] = "https://attacker.test/forged-filing.htm"
    malformed = _verify(
        "sec.get_recent_filings",
        malformed_data,
        SEC_SOURCE,
        statement,
        answer=statement,
    )

    assert accepted.result.status is VerifierStatus.PASS
    assert forged.result.status is VerifierStatus.FAIL
    assert malformed.result.status is VerifierStatus.FAIL


@pytest.mark.parametrize(
    "statement",
    (
        "NVDA filed form 8-K on 2026-05-28 under accession 0001045810-26-000123.",
        "NVDA filed form 10-Q on 2026-05-29 under accession 0001045810-26-000123.",
        "NVDA filed form 10-Q on 2026-05-28 under accession 0001045810-26-999999.",
        "TSLA filed form 10-Q on 2026-05-28 under accession 0001045810-26-000123.",
        f"{SEC_STATEMENT} This guarantees future revenue growth.",
    ),
)
def test_sec_grounding_rejects_mismatched_or_added_facts(statement: str) -> None:
    outcome = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        statement,
        answer=statement,
    )

    assert outcome.result.status is VerifierStatus.FAIL


def test_sec_grounding_requires_exact_tuple_in_answer() -> None:
    outcome = _verify(
        "sec.get_recent_filings",
        SEC_DATA,
        SEC_SOURCE,
        SEC_STATEMENT,
        answer="NVDA's latest filing was form 10-Q under accession 0001045810-26-000123.",
    )

    assert outcome.result.status is VerifierStatus.FAIL


def test_single_sec_completion_rejects_unsupported_answer_addition() -> None:
    observation = Observation(
        id="obs-1",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-1",
        kind="sec.get_recent_filings",
        data=SEC_DATA,
        source=SEC_SOURCE,
        observed_at=NOW,
        raw_hash="fixture-hash",
    )
    verifier = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        completion_contract=CompletionContract(
            source_claim_count=CardinalityBounds(minimum=1, maximum=1),
            source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
            inference_count=CardinalityBounds(minimum=0, maximum=0),
            guidance="Return one exact SEC source claim.",
        ),
    )

    outcome = verifier.verify(
        CompletionProposal(
            answer=f"{SEC_STATEMENT} This guarantees future revenue growth.",
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=SEC_STATEMENT,
                    observation_ids=(observation.id,),
                ),
            ),
        ),
        _bundle(observation),
    )

    failed = {check.name for check in outcome.result.checks if not check.passed}
    assert outcome.result.status is VerifierStatus.FAIL
    assert "completion_single_sec_answer_supported" in failed


def test_verifier_enforces_completion_contract_source_claim_cardinality() -> None:
    observation = Observation(
        id="obs-1",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-1",
        kind="sec.get_recent_filings",
        data=SEC_DATA,
        source=SEC_SOURCE,
        observed_at=NOW,
        raw_hash="fixture-hash",
    )
    duplicate = CandidateClaim(
        kind=ClaimKind.SOURCE_CLAIM,
        statement=SEC_STATEMENT,
        observation_ids=(observation.id,),
    )
    verifier = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        completion_contract=CompletionContract(
            source_claim_count=CardinalityBounds(minimum=1, maximum=1),
            source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
            inference_count=CardinalityBounds(minimum=0, maximum=0),
            guidance="Return one exact SEC source claim.",
        ),
    )

    outcome = verifier.verify(
        CompletionProposal(
            answer=SEC_STATEMENT,
            claims=(duplicate, duplicate),
        ),
        _bundle(observation),
    )

    failed = {check.name for check in outcome.result.checks if not check.passed}
    assert outcome.result.status is VerifierStatus.FAIL
    assert "completion_source_claim_count" in failed


def test_quote_grounding_remains_exact_and_answer_bound() -> None:
    data: dict[str, JsonValue] = {"symbol": "NVDA", "price": 181.25, "currency": "USD"}
    source = SourceRef(provider="fixture", reference="quote")
    statement = "NVDA is quoted at 181.25 USD."
    assert _verify("market.get_quote", data, source, statement).result.status is VerifierStatus.PASS
    assert (
        _verify(
            "market.get_quote",
            data,
            source,
            statement,
            answer="NVDA has a current quote.",
        ).result.status
        is VerifierStatus.FAIL
    )


def test_normalized_finnhub_quote_fails_closed_on_provenance_or_expiry_tampering() -> None:
    statement = "NVDA is quoted at 181.25 USD."
    data: dict[str, JsonValue] = {
        "provider": "finnhub",
        "symbol": "NVDA",
        "price": 181.25,
        "currency": "USD",
        "as_of": NOW.isoformat(),
        "statements": [statement],
    }
    source = SourceRef(provider="finnhub", reference=f"quote:NVDA:{int(NOW.timestamp())}")
    expiry = NOW + timedelta(minutes=15)

    assert (
        _verify("market.get_quote", data, source, statement, expires_at=expiry).result.status
        is VerifierStatus.PASS
    )
    assert (
        _verify(
            "market.get_quote",
            data,
            SourceRef(provider="finnhub", reference=f"quote:AAPL:{int(NOW.timestamp())}"),
            statement,
            expires_at=expiry,
        ).result.status
        is VerifierStatus.FAIL
    )
    assert _verify("market.get_quote", data, source, statement).result.status is VerifierStatus.FAIL
    downgraded = dict(data)
    downgraded.pop("provider")
    assert (
        _verify("market.get_quote", downgraded, source, statement, expires_at=expiry).result.status
        is VerifierStatus.FAIL
    )


def test_unknown_observation_kind_still_fails_closed() -> None:
    outcome = _verify(
        "fixture.unknown",
        {"text": "Unknown evidence says demand increased."},
        SourceRef(provider="fixture", reference="unknown"),
        "Demand increased.",
    )
    assert outcome.result.status is VerifierStatus.FAIL
    support = next(check for check in outcome.result.checks if check.name.endswith("_supported"))
    assert "No registered grounding rule" in support.detail


def test_unsourced_context_only_completion_requires_explicit_verifier_flag() -> None:
    proposal = CompletionProposal(answer="A context-only response.")
    default = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
    ).verify(proposal, _bundle())
    trusted_context_only = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        require_source_claim=False,
    ).verify(proposal, _bundle())
    assert default.result.status is VerifierStatus.FAIL
    assert trusted_context_only.result.status is VerifierStatus.PASS
    assert trusted_context_only.completion is not None


def test_required_parent_orchestration_rejects_promissory_completion() -> None:
    outcome = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        require_source_claim=True,
        required_any_observation_kinds=frozenset(
            {"agent.execute_research_plan", "agent.delegate_research"}
        ),
    ).verify(
        CompletionProposal(answer="I'll start a research plan next."),
        _bundle(),
    )

    failed = {check.name for check in outcome.result.checks if not check.passed}
    assert outcome.result.status is VerifierStatus.FAIL
    assert "required_any_observation_present" in failed
    assert "required_any_observation_cited" in failed
    assert "source_claim_required" in failed


def test_integrated_research_requirement_corrects_missing_second_source() -> None:
    quote = Observation(
        id="obs-market",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-market",
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 181.25},
        source=SourceRef(provider="finnhub", reference="quote:NVDA:1"),
        observed_at=NOW,
        raw_hash="1" * 64,
    )
    sec = Observation(
        id="obs-sec",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-sec",
        kind="sec.get_recent_filings",
        data=SEC_DATA,
        source=SEC_SOURCE,
        observed_at=NOW,
        raw_hash="2" * 64,
    )
    bundle = _bundle().model_copy(update={"observations": (quote, sec)})
    quote_statement = "NVDA is quoted at 181.25."
    verifier = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        research_requirement=ResearchRequirement(
            required_kinds=frozenset({"market.get_quote", "sec.get_recent_filings"}),
            minimum_source_claims=2,
            minimum_distinct_sources=2,
            counter_evidence_kinds=frozenset({"market.get_quote"}),
            require_uncertainty_on_conflict=False,
            require_affected_assumption_on_conflict=False,
        ),
    )

    incomplete = verifier.verify(
        CompletionProposal(
            answer=quote_statement,
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=quote_statement,
                    observation_ids=(quote.id,),
                ),
            ),
        ),
        bundle,
    )
    failed = {check.name for check in incomplete.result.checks if not check.passed}
    assert incomplete.result.status is VerifierStatus.FAIL
    assert "research_minimum_source_claims" in failed
    assert "research_required_kind_sec.get_recent_filings" in failed

    complete = verifier.verify(
        CompletionProposal(
            answer=f"{quote_statement} {SEC_STATEMENT}",
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=quote_statement,
                    observation_ids=(quote.id,),
                ),
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=SEC_STATEMENT,
                    observation_ids=(sec.id,),
                ),
            ),
        ),
        bundle,
    )
    assert complete.result.status is VerifierStatus.PASS
    assert complete.completion is not None
    assert complete.completion.answer == f"{quote_statement} {SEC_STATEMENT}"

    conflict_verifier = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        research_requirement=ResearchRequirement(
            required_kinds=frozenset({"market.get_quote", "sec.get_recent_filings"}),
            minimum_source_claims=2,
            minimum_distinct_sources=2,
            counter_evidence_kinds=frozenset({"market.get_quote"}),
        ),
    )
    base = CompletionProposal(
        answer=f"{quote_statement} {SEC_STATEMENT}",
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=quote_statement,
                observation_ids=(quote.id,),
            ),
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=SEC_STATEMENT,
                observation_ids=(sec.id,),
            ),
        ),
    )
    unresolved = conflict_verifier.verify(base, bundle)
    failed = {check.name for check in unresolved.result.checks if not check.passed}
    assert failed == {
        "research_affected_assumption_on_conflict",
        "research_uncertainty_on_conflict",
    }

    resolved = conflict_verifier.verify(
        base.model_copy(
            update={
                "affected_assumption": "Demand remains durable.",
                "uncertainty": "Market and filing evidence cover different windows.",
            }
        ),
        bundle,
    )
    assert resolved.result.status is VerifierStatus.PASS
    assert resolved.completion is not None
    assert [claim.kind for claim in resolved.completion.claims[-2:]] == [
        ClaimKind.AFFECTED_ASSUMPTION,
        ClaimKind.UNCERTAINTY,
    ]


LIVE_QUOTE_STATEMENT = "NVDA is quoted at 214.72."
LIVE_SEC_STATEMENT = "NVDA filed form 8-K on 2026-08-17 under accession 0001045810-26-000069."


def _verified_plan_node(
    *,
    node_id: str,
    child_run_id: str,
    statement: str,
    kind: str,
    data: dict[str, JsonValue],
    source: SourceRef,
    raw_hash: str,
    observed_at: datetime = NOW,
    expires_at: datetime | None = None,
) -> dict[str, JsonValue]:
    child_observation = Observation(
        id=f"obs:{child_run_id}",
        scope=SCOPE,
        run_id=child_run_id,
        tool_call_id=f"call:{child_run_id}",
        kind=kind,
        data=data,
        source=source,
        observed_at=observed_at,
        expires_at=expires_at,
        raw_hash=raw_hash,
    )
    child_claim = Claim(
        id=f"claim:{child_run_id}",
        scope=SCOPE,
        run_id=child_run_id,
        kind=ClaimKind.SOURCE_CLAIM,
        statement=statement,
        observation_ids=(child_observation.id,),
    )
    envelope = child_evidence_data(
        build_child_evidence_envelope(
            child_run_id=child_run_id,
            answer=statement,
            trace_event_count=12,
            observations=(child_observation,),
            claims=(child_claim,),
        )
    )
    return {
        "id": node_id,
        "status": "completed",
        "answer": statement,
        "child_run_id": child_run_id,
        "trace_event_count": 12,
        "observation_count": 1,
        "child_evidence": envelope,
    }


def _live_plan_observation(
    *,
    quote_hash: str = "4" * 64,
    quote_provider: str = "finnhub",
    quote_reference: str = "quote:NVDA:1787342400",
    source_observed_at: datetime = NOW,
    source_expires_at: datetime | None = None,
) -> Observation:
    quote_node = _verified_plan_node(
        node_id="nvda_quote",
        child_run_id="subrun-quote",
        statement=LIVE_QUOTE_STATEMENT,
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 214.72},
        source=SourceRef(provider=quote_provider, reference=quote_reference),
        raw_hash=quote_hash,
        observed_at=source_observed_at,
        expires_at=source_expires_at,
    )
    sec_node = _verified_plan_node(
        node_id="nvda_filings",
        child_run_id="subrun-sec",
        statement=LIVE_SEC_STATEMENT,
        kind="sec.get_recent_filings",
        data={
            "ticker": "NVDA",
            "cik": "0001045810",
            "filings": [
                {
                    "form": "8-K",
                    "accession": "0001045810-26-000069",
                    "filing_date": "2026-08-17",
                    "primary_document": "nvda-20260817.htm",
                }
            ],
        },
        source=SourceRef(provider="sec-edgar", reference="submissions:0001045810"),
        raw_hash="5" * 64,
        observed_at=source_observed_at,
        expires_at=source_expires_at,
    )
    return Observation(
        id="obs-plan-live",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call-plan-live",
        kind="agent.execute_research_plan",
        data={
            "plan_id": "plan-live",
            "revision": 1,
            "goal": "Challenge the NVDA thesis with market and SEC evidence.",
            "status": "completed",
            "nodes": [quote_node, sec_node],
            "completed_count": 2,
            "failed_count": 0,
            "blocked_count": 0,
        },
        source=SourceRef(provider="leo-subagent-plan", reference="plan-live"),
        observed_at=NOW,
        expires_at=source_expires_at,
        raw_hash="6" * 64,
        quality=EvidenceQuality.VERIFIED_CHILD,
    )


def _live_plan_verifier() -> DeterministicCompletionVerifier:
    return DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        required_any_observation_kinds=frozenset({"agent.execute_research_plan"}),
        evidence_requirements=(
            EvidenceToolRequirement(
                observation_kind="market.get_quote",
                tool_name="market.get_quote",
                required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
            ),
            EvidenceToolRequirement(
                observation_kind="sec.get_recent_filings",
                tool_name="sec.get_recent_filings",
                required_arguments=(ToolArgumentConstraint(name="ticker", value="NVDA"),),
            ),
        ),
        research_requirement=ResearchRequirement(
            required_kinds=frozenset({"market.get_quote", "sec.get_recent_filings"}),
            minimum_source_claims=2,
            minimum_distinct_sources=2,
            counter_evidence_kinds=frozenset({"market.get_quote"}),
        ),
    )


def _live_plan_proposal(*, include_sec: bool = True) -> CompletionProposal:
    statements = (
        (LIVE_QUOTE_STATEMENT, LIVE_SEC_STATEMENT) if include_sec else (LIVE_QUOTE_STATEMENT,)
    )
    return CompletionProposal(
        answer=" ".join(statements),
        claims=tuple(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=("obs-plan-live",),
            )
            for statement in statements
        ),
        affected_assumption="The thesis assumes filings and price evidence remain supportive.",
        uncertainty="The current quote and filing metadata cover different evidence windows.",
    )


def test_live_like_parent_plan_projects_verified_nested_sources() -> None:
    observation = _live_plan_observation()
    outcome = _live_plan_verifier().verify(
        _live_plan_proposal(),
        _bundle(observation),
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    passed = {check.name for check in outcome.result.checks if check.passed}
    assert "required_evidence_0_market.get_quote_present" in passed
    assert "required_evidence_1_sec.get_recent_filings_cited" in passed
    assert "research_minimum_distinct_sources" in passed
    assert "research_counter_evidence_present" in passed


def test_parent_plan_requires_every_exact_verified_child_statement() -> None:
    outcome = _live_plan_verifier().verify(
        _live_plan_proposal(include_sec=False),
        _bundle(_live_plan_observation()),
    )

    failed = {check.name for check in outcome.result.checks if not check.passed}
    assert outcome.result.status is VerifierStatus.FAIL
    assert "plan_obs-plan-live_child_statement_1_claimed" in failed
    assert "plan_obs-plan-live_child_statement_1_carried" in failed
    assert "research_minimum_source_claims" in failed


@pytest.mark.parametrize(
    ("variant", "expected_detail"),
    (
        ("forged_digest", "malformed or was changed"),
        ("malformed_hash", "integrity or provenance"),
        ("mismatched_provider", "integrity or provenance"),
        ("mismatched_child_run", "does not match"),
        ("mismatched_plan_reference", "source authority"),
    ),
)
def test_parent_plan_rejects_forged_or_mismatched_child_evidence(
    variant: str,
    expected_detail: str,
) -> None:
    if variant == "malformed_hash":
        observation = _live_plan_observation(quote_hash="not-a-sha256")
    elif variant == "mismatched_provider":
        observation = _live_plan_observation(quote_provider="sec-edgar")
    else:
        observation = _live_plan_observation()
    data = deepcopy(observation.data)
    if variant == "forged_digest":
        nodes = data["nodes"]
        assert isinstance(nodes, list)
        quote_node = nodes[0]
        assert isinstance(quote_node, dict)
        child_evidence = quote_node["child_evidence"]
        assert isinstance(child_evidence, dict)
        child_evidence["answer"] = "Forged answer with an unchanged digest."
    elif variant == "mismatched_child_run":
        nodes = data["nodes"]
        assert isinstance(nodes, list)
        quote_node = nodes[0]
        assert isinstance(quote_node, dict)
        quote_node["child_run_id"] = "subrun-forged"
    source = observation.source
    if variant == "mismatched_plan_reference":
        source = SourceRef(provider="leo-subagent-plan", reference="plan-other")
    changed = observation.model_copy(update={"data": data, "source": source})

    outcome = _live_plan_verifier().verify(
        _live_plan_proposal(),
        _bundle(changed),
    )

    assert outcome.result.status is VerifierStatus.FAIL
    support = next(check for check in outcome.result.checks if check.name.endswith("_supported"))
    assert expected_detail in support.detail


def test_parent_plan_rejects_expired_nested_child_sources() -> None:
    observed_at = NOW - timedelta(hours=2)
    expires_at = NOW - timedelta(hours=1)
    observation = _live_plan_observation(
        source_observed_at=observed_at,
        source_expires_at=expires_at,
    )

    outcome = _live_plan_verifier().verify(
        _live_plan_proposal(),
        _bundle(observation),
    )

    failed = {check.name for check in outcome.result.checks if not check.passed}
    assert outcome.result.status is VerifierStatus.FAIL
    assert "required_evidence_0_market.get_quote_present" in failed
    assert (
        "research_claim_0_nested:obs-plan-live:subrun-quote:claim:subrun-quote:obs:subrun-quote_fresh"
        in failed
    )


THREAD_STATEMENT = "Project Borealis uses amber hexagons."
THREAD_RANGE_DIGEST = "7" * 64
THREAD_DATA: dict[str, JsonValue] = {
    "handle": "thr_" + ("8" * 32),
    "range_digest": THREAD_RANGE_DIGEST,
    "chunks": [
        {
            "ordinal": 3,
            "source_item_digest": "9" * 64,
            "text": f"Earlier authorized detail: {THREAD_STATEMENT}",
        }
    ],
    "next_ordinal": 4,
    "source_conversation": "C1",
    "thread_root_ts": "1787342409.043219",
    "policy_version": "thread-context-navigation-v1",
}


def _thread_observation(
    *,
    data: dict[str, JsonValue] | None = None,
    provider: str = "leo_thread_context",
    reference: str = THREAD_RANGE_DIGEST,
    run_id: str = "run",
    raw_hash: str | None = None,
) -> Observation:
    selected = deepcopy(data or THREAD_DATA)
    encoded = json.dumps(
        selected,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return Observation(
        id="obs-thread-open",
        scope=SCOPE,
        run_id=run_id,
        tool_call_id="call-thread-open",
        kind="thread_context.open",
        data=selected,
        source=SourceRef(provider=provider, reference=reference),
        observed_at=NOW,
        raw_hash=raw_hash or hashlib.sha256(encoded).hexdigest(),
        quality=EvidenceQuality.INTERNAL_CONTEXT,
    )


def _verify_thread_open(
    observation: Observation,
    *,
    statement: str = THREAD_STATEMENT,
    answer: str | None = None,
    kind: ClaimKind = ClaimKind.INFERENCE,
) -> VerificationOutcome:
    return DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        require_source_claim=kind is ClaimKind.SOURCE_CLAIM,
    ).verify(
        CompletionProposal(
            answer=answer or f"From the earlier thread: {statement}",
            claims=(
                CandidateClaim(
                    kind=kind,
                    statement=statement,
                    observation_ids=(observation.id,),
                ),
            ),
        ),
        _bundle(observation),
    )


def test_thread_context_open_supports_only_exact_carried_internal_inference() -> None:
    outcome = _verify_thread_open(_thread_observation())

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert outcome.completion.claims[0].kind is ClaimKind.INFERENCE


@pytest.mark.parametrize(
    ("kind", "statement", "answer"),
    (
        (ClaimKind.SOURCE_CLAIM, THREAD_STATEMENT, None),
        (ClaimKind.INFERENCE, "Project Borealis uses blue circles.", None),
        (ClaimKind.INFERENCE, THREAD_STATEMENT, "A summary without the exact statement."),
    ),
)
def test_thread_context_open_rejects_external_wrong_or_uncarried_claims(
    kind: ClaimKind,
    statement: str,
    answer: str | None,
) -> None:
    outcome = _verify_thread_open(
        _thread_observation(),
        kind=kind,
        statement=statement,
        answer=answer,
    )

    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None


@pytest.mark.parametrize(
    "variant",
    (
        "extra_field",
        "noncontiguous_chunks",
        "bad_source_digest",
        "wrong_provider",
        "wrong_reference",
        "tampered_raw_hash",
    ),
)
def test_thread_context_open_rejects_forged_shape_provenance_and_authority(
    variant: str,
) -> None:
    data = deepcopy(THREAD_DATA)
    provider = "leo_thread_context"
    reference = THREAD_RANGE_DIGEST
    run_id = "run"
    raw_hash: str | None = None
    if variant == "extra_field":
        data["conversation_id"] = "C-forged"
    elif variant == "noncontiguous_chunks":
        chunks = data["chunks"]
        assert isinstance(chunks, list)
        chunks.append(
            {
                "ordinal": 9,
                "source_item_digest": "a" * 64,
                "text": "Unrelated forged continuation.",
            }
        )
    elif variant == "bad_source_digest":
        data["range_digest"] = "not-a-digest"
    elif variant == "wrong_provider":
        provider = "public-web"
    elif variant == "wrong_reference":
        reference = "6" * 64
    elif variant == "tampered_raw_hash":
        raw_hash = "5" * 64
    observation = _thread_observation(
        data=data,
        provider=provider,
        reference=reference,
        run_id=run_id,
        raw_hash=raw_hash,
    )

    outcome = _verify_thread_open(observation)

    assert outcome.result.status is VerifierStatus.FAIL
    assert outcome.completion is None


def test_thread_context_open_wrong_run_is_rejected_by_bundle_authority() -> None:
    with pytest.raises(ValidationError, match="outside the run scope"):
        _bundle(_thread_observation(run_id="run-other"))


def test_thread_context_open_requires_one_chunk_to_contain_the_full_inference() -> None:
    data = deepcopy(THREAD_DATA)
    data["chunks"] = [
        {
            "ordinal": 0,
            "source_item_digest": "1" * 64,
            "text": "Project Borealis uses",
        },
        {
            "ordinal": 1,
            "source_item_digest": "1" * 64,
            "text": "amber hexagons.",
        },
    ]
    data["next_ordinal"] = None

    outcome = _verify_thread_open(_thread_observation(data=data))

    assert outcome.result.status is VerifierStatus.FAIL
