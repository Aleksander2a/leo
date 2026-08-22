from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

from leo.harness.earnings import canonical_earnings_statements
from leo.harness.equity_market import (
    canonical_equity_profile_statements,
    canonical_equity_quote_statement,
)
from leo.harness.models import (
    CandidateClaim,
    CardinalityBounds,
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
    VerifierStatus,
)
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.fake import FixedClock, SequentialIdGenerator

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _bundle(
    *observations: Observation,
    objective: str = "Give me the provider result",
) -> RunBundle:
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="thread-root"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective=objective)
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=observations)


def _observation(
    *,
    observation_id: str,
    kind: str,
    data: dict[str, object],
    source: SourceRef,
    observed_at: datetime = NOW,
    expires_at: datetime | None = None,
    quality: EvidenceQuality = EvidenceQuality.PROVIDER_REPORTED,
) -> Observation:
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str).encode()
    return Observation.model_validate(
        {
            "id": observation_id,
            "scope": SCOPE,
            "run_id": "run",
            "tool_call_id": f"call:{observation_id}",
            "kind": kind,
            "data": data,
            "source": source,
            "observed_at": observed_at,
            "expires_at": expires_at or NOW + timedelta(minutes=15),
            "raw_hash": hashlib.sha256(encoded).hexdigest(),
            "quality": quality,
        }
    )


def _verifier(
    *requirements: EvidenceToolRequirement,
    maximum_claims: int | None = None,
) -> DeterministicCompletionVerifier:
    maximum = maximum_claims if maximum_claims is not None else len(requirements)
    return DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
        evidence_requirements=requirements,
        completion_contract=CompletionContract(
            source_claim_count=CardinalityBounds(
                minimum=len(requirements),
                maximum=maximum,
            ),
            source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
            inference_count=CardinalityBounds(minimum=0, maximum=8),
        ),
    )


def _requirement(kind: str, tool: str, name: str, value: str) -> EvidenceToolRequirement:
    return EvidenceToolRequirement(
        observation_kind=kind,
        tool_name=tool,
        required_arguments=(ToolArgumentConstraint(name=name, value=value),),
    )


def test_short_provider_prompt_recovers_from_model_refusal_after_valid_profile_read() -> None:
    data: dict[str, object] = {
        "provider": "alpha-vantage",
        "symbol": "MSFT",
        "provider_symbol": "MSFT",
        "name": "Microsoft Corporation",
        "exchange": "NASDAQ",
        "industry": "Software",
        "as_of": NOW.isoformat(),
    }
    statements = canonical_equity_profile_statements(data)  # type: ignore[arg-type]
    assert statements is not None
    data["statements"] = list(statements)
    observation = _observation(
        observation_id="obs-profile",
        kind="market.get_company_profile_alpha_vantage",
        data=data,
        source=SourceRef(provider="alpha-vantage", reference="company-overview:MSFT"),
    )
    requirement = _requirement(
        "market.get_company_profile_alpha_vantage",
        "market.get_company_profile_alpha_vantage",
        "symbol",
        "MSFT",
    )

    outcome = _verifier(requirement).verify(
        CompletionProposal(answer="I couldn't complete that request."),
        _bundle(observation, objective="MSFT profile"),
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert outcome.completion.answer == statements[0]
    assert outcome.completion.claims[0].observation_ids == (observation.id,)


def test_multi_step_provider_read_recovers_quote_and_earnings_without_another_model_turn() -> None:
    timestamp = int(NOW.timestamp())
    quote_data: dict[str, object] = {
        "provider": "finnhub",
        "symbol": "NVDA",
        "provider_symbol": "NVDA",
        "price": 183.25,
        "as_of": NOW.isoformat(),
    }
    quote_statement = canonical_equity_quote_statement(quote_data)  # type: ignore[arg-type]
    assert quote_statement is not None
    quote_data["statements"] = [quote_statement]
    quote = _observation(
        observation_id="obs-quote",
        kind="market.get_quote_finnhub",
        data=quote_data,
        source=SourceRef(provider="finnhub", reference=f"quote:NVDA:{timestamp}"),
    )
    earnings_items: list[dict[str, object]] = [
        {"symbol": "NVDA", "period": "2026-07-31", "actual": 1.04, "estimate": 1.01}
    ]
    earnings_statements = canonical_earnings_statements("NVDA", earnings_items)  # type: ignore[arg-type]
    assert earnings_statements is not None
    earnings_data: dict[str, object] = {
        "symbol": "NVDA",
        "items": earnings_items,
        "item_count": 1,
        "statements": list(earnings_statements),
    }
    earnings = _observation(
        observation_id="obs-earnings",
        kind="market.get_earnings_surprises",
        data=earnings_data,
        source=SourceRef(provider="finnhub", reference="earnings-surprises:NVDA"),
    )
    requirements = (
        _requirement(
            "market.get_quote_finnhub",
            "market.get_quote_finnhub",
            "symbol",
            "NVDA",
        ),
        _requirement(
            "market.get_earnings_surprises",
            "market.get_earnings_surprises",
            "symbol",
            "NVDA",
        ),
    )

    outcome = _verifier(*requirements).verify(
        CompletionProposal(answer="Let me pull those results next."),
        _bundle(quote, earnings, objective="Give me NVDA's quote and latest earnings surprise"),
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert outcome.completion.answer == f"{quote_statement} {earnings_statements[0]}"
    assert tuple(claim.observation_ids for claim in outcome.completion.claims) == (
        (quote.id,),
        (earnings.id,),
    )


def test_valid_conversational_synthesis_is_preserved_when_canonical_claim_is_present() -> None:
    timestamp = int(NOW.timestamp())
    data: dict[str, object] = {
        "provider": "finnhub",
        "symbol": "NVDA",
        "provider_symbol": "NVDA",
        "price": 183.25,
        "as_of": NOW.isoformat(),
    }
    statement = canonical_equity_quote_statement(data)  # type: ignore[arg-type]
    assert statement is not None
    data["statements"] = [statement]
    observation = _observation(
        observation_id="obs-quote",
        kind="market.get_quote_finnhub",
        data=data,
        source=SourceRef(provider="finnhub", reference=f"quote:NVDA:{timestamp}"),
    )
    requirement = _requirement(
        "market.get_quote_finnhub",
        "market.get_quote_finnhub",
        "symbol",
        "NVDA",
    )
    answer = f"Quick answer: {statement} That gives you the current provider snapshot."
    proposal = CompletionProposal(
        answer=answer,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(observation.id,),
            ),
            CandidateClaim(
                kind=ClaimKind.INFERENCE,
                statement="That gives you the current provider snapshot.",
            ),
        ),
    )

    outcome = _verifier(requirement, maximum_claims=1).verify(
        proposal,
        _bundle(observation, objective="NVDA quote"),
    )

    assert outcome.result.status is VerifierStatus.PASS
    assert outcome.completion is not None
    assert outcome.completion.answer == answer
    assert len(outcome.completion.claims) == 2


def test_stale_or_wrong_provenance_never_triggers_provider_recovery() -> None:
    data: dict[str, object] = {
        "provider": "alpha-vantage",
        "symbol": "MSFT",
        "provider_symbol": "MSFT",
        "name": "Microsoft Corporation",
        "exchange": "NASDAQ",
        "as_of": NOW.isoformat(),
    }
    statements = canonical_equity_profile_statements(data)  # type: ignore[arg-type]
    assert statements is not None
    data["statements"] = list(statements)
    base = _observation(
        observation_id="obs-profile",
        kind="market.get_company_profile_alpha_vantage",
        data=data,
        source=SourceRef(provider="alpha-vantage", reference="company-overview:MSFT"),
    )
    requirement = _requirement(
        "market.get_company_profile_alpha_vantage",
        "market.get_company_profile_alpha_vantage",
        "symbol",
        "MSFT",
    )
    verifier = _verifier(requirement)

    stale = base.model_copy(update={"expires_at": NOW})
    stale_outcome = verifier.verify(
        CompletionProposal(answer="I couldn't complete that request."),
        _bundle(stale, objective="MSFT profile"),
    )
    forged = base.model_copy(
        update={"source": SourceRef(provider="alpha-vantage", reference="company-overview:AMD")}
    )
    forged_outcome = verifier.verify(
        CompletionProposal(answer="I couldn't complete that request."),
        _bundle(forged, objective="MSFT profile"),
    )

    assert stale_outcome.result.status is VerifierStatus.FAIL
    assert stale_outcome.completion is None
    assert forged_outcome.result.status is VerifierStatus.FAIL
    assert forged_outcome.completion is None


def test_discovery_snippets_and_raw_fetch_text_are_not_direct_completion_authority() -> None:
    query = "current Python release"
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    tavily = _observation(
        observation_id="obs-tavily",
        kind="web.search_tavily",
        data={
            "query": query,
            "query_hash": query_hash,
            "results": [
                {
                    "title": "Python",
                    "url": "https://www.python.org/downloads/",
                    "snippet": "Python 9.9 is current.",
                    "score": 1.0,
                }
            ],
            "result_count": 1,
            "untrusted": True,
            "requires_fetch_for_source_claim": True,
        },
        source=SourceRef(provider="tavily", reference=f"search:{query_hash}"),
        quality=EvidenceQuality.DISCOVERY_ONLY,
    )
    fetch_text = "Python 9.9 is current."
    fetch_digest = hashlib.sha256(fetch_text.encode()).hexdigest()
    fetched = _observation(
        observation_id="obs-fetch",
        kind="web.fetch_public_text",
        data={
            "url": "https://www.python.org/downloads/",
            "text": fetch_text,
            "content_sha256": fetch_digest,
            "truncated": False,
            "untrusted": True,
        },
        source=SourceRef(
            provider="public-web",
            reference=fetch_digest,
            url="https://www.python.org/downloads/",
        ),
        quality=EvidenceQuality.UNTRUSTED_RETRIEVAL,
    )
    tavily_requirement = _requirement(
        "web.search_tavily",
        "web.search_tavily",
        "query",
        query,
    )
    fetch_requirement = _requirement(
        "web.fetch_public_text",
        "web.fetch_public_text",
        "url",
        "https://www.python.org/downloads/",
    )

    tavily_outcome = _verifier(tavily_requirement).verify(
        CompletionProposal(answer="Python 9.9 is current."),
        _bundle(tavily, objective=query),
    )
    fetch_outcome = _verifier(fetch_requirement).verify(
        CompletionProposal(answer="Python 9.9 is current."),
        _bundle(fetched, objective=query),
    )

    assert tavily_outcome.result.status is VerifierStatus.FAIL
    assert tavily_outcome.completion is None
    assert fetch_outcome.result.status is VerifierStatus.FAIL
    assert fetch_outcome.completion is None
