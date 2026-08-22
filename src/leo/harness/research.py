"""Deterministic multi-source research verification; semantic judges remain advisory."""

from __future__ import annotations

from datetime import datetime

from pydantic import Field

from leo.harness.models import (
    ClaimKind,
    ContractModel,
    EvidenceQuality,
    NonEmptyStr,
    Observation,
    ObservationStatus,
    RunBundle,
    VerifierCheck,
    VerifierStatus,
)


class ResearchProposal(ContractModel):
    answer: NonEmptyStr
    claims: tuple[ResearchClaim, ...] = ()
    uncertainty: NonEmptyStr | None = None
    affected_assumption: NonEmptyStr | None = None


class ResearchClaim(ContractModel):
    kind: ClaimKind
    statement: NonEmptyStr
    observation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class ResearchRequirement(ContractModel):
    required_kinds: frozenset[NonEmptyStr] = frozenset()
    minimum_source_claims: int = Field(default=1, ge=0, le=32)
    minimum_distinct_sources: int = Field(default=1, ge=0, le=32)
    counter_evidence_kinds: frozenset[NonEmptyStr] = frozenset()
    freshness_seconds: int | None = Field(default=None, ge=1)
    require_uncertainty_on_conflict: bool = True
    require_affected_assumption_on_conflict: bool = True


class ResearchVerification(ContractModel):
    status: VerifierStatus
    checks: tuple[VerifierCheck, ...] = Field(min_length=1)
    retryable: bool
    correction: NonEmptyStr | None = None


def verify_research(
    proposal: ResearchProposal,
    bundle: RunBundle,
    *,
    now: datetime,
    requirement: ResearchRequirement,
    persisted_claim_ids: frozenset[str] = frozenset(),
) -> ResearchVerification:
    observations = {item.id: item for item in bundle.observations}
    checks: list[VerifierCheck] = []
    source_claims = tuple(
        claim for claim in proposal.claims if claim.kind is ClaimKind.SOURCE_CLAIM
    )
    checks.append(
        VerifierCheck(
            name="answer_present",
            passed=bool(proposal.answer.strip()),
            detail=(
                "Research answer is present."
                if proposal.answer.strip()
                else "Research answer is empty."
            ),
        )
    )
    checks.append(
        VerifierCheck(
            name="minimum_source_claims",
            passed=len(source_claims) >= requirement.minimum_source_claims,
            detail=f"Found {len(source_claims)} source-backed claims.",
        )
    )
    cited: list[Observation] = []
    for index, claim in enumerate(source_claims):
        for observation_id in claim.observation_ids:
            observation = observations.get(observation_id)
            valid = observation is not None and observation.scope == bundle.run.scope
            usable = (
                valid
                and observation is not None
                and observation.status is ObservationStatus.RETRIEVED
                and observation.quality is not EvidenceQuality.DISCOVERY_ONLY
            )
            fresh = usable and _is_fresh(observation, now, requirement.freshness_seconds)
            persisted = observation_id in persisted_claim_ids or not persisted_claim_ids
            checks.extend(
                (
                    VerifierCheck(
                        name=f"claim_{index}_{observation_id}_status_quality",
                        passed=usable,
                        detail=(
                            "Citation status and evidence quality are eligible."
                            if usable
                            else "Citation is stale, rejected, or discovery-only metadata."
                        ),
                    ),
                    VerifierCheck(
                        name=f"claim_{index}_{observation_id}_scope",
                        passed=valid,
                        detail=(
                            "Citation is present in the run scope."
                            if valid
                            else "Citation is missing or cross-scope."
                        ),
                    ),
                    VerifierCheck(
                        name=f"claim_{index}_{observation_id}_fresh",
                        passed=fresh,
                        detail=(
                            "Citation is fresh." if fresh else "Citation is stale or unavailable."
                        ),
                    ),
                    VerifierCheck(
                        name=f"claim_{index}_{observation_id}_persisted",
                        passed=persisted,
                        detail=(
                            "Citation persistence is confirmed."
                            if persisted
                            else "Citation persistence is missing."
                        ),
                    ),
                )
            )
            if observation is not None and valid and usable and fresh:
                cited.append(observation)
    distinct_sources = {f"{item.source.provider}:{item.source.reference}" for item in cited}
    checks.append(
        VerifierCheck(
            name="minimum_distinct_sources",
            passed=len(distinct_sources) >= requirement.minimum_distinct_sources,
            detail=f"Found {len(distinct_sources)} distinct source references.",
        )
    )
    for required_kind in sorted(requirement.required_kinds):
        present = any(item.kind == required_kind for item in cited)
        checks.append(
            VerifierCheck(
                name=f"required_kind_{required_kind}",
                passed=present,
                detail=(
                    "Required evidence kind is present."
                    if present
                    else "Required evidence kind is missing."
                ),
            )
        )
    counter_present = any(item.kind in requirement.counter_evidence_kinds for item in cited)
    if requirement.counter_evidence_kinds:
        checks.append(
            VerifierCheck(
                name="counter_evidence_present",
                passed=counter_present,
                detail=(
                    "Counter-evidence is present."
                    if counter_present
                    else "Counter-evidence is missing."
                ),
            )
        )
    if counter_present and requirement.require_uncertainty_on_conflict:
        checks.append(
            VerifierCheck(
                name="uncertainty_on_conflict",
                passed=bool(proposal.uncertainty),
                detail=(
                    "Conflicting evidence is labeled uncertain."
                    if proposal.uncertainty
                    else "Conflicting evidence requires uncertainty."
                ),
            )
        )
    if counter_present and requirement.require_affected_assumption_on_conflict:
        checks.append(
            VerifierCheck(
                name="affected_assumption_on_conflict",
                passed=bool(proposal.affected_assumption),
                detail=(
                    "The affected assumption is explicit."
                    if proposal.affected_assumption
                    else "Conflicting evidence requires an affected assumption."
                ),
            )
        )
    passed = all(check.passed for check in checks)
    return ResearchVerification(
        status=VerifierStatus.PASS if passed else VerifierStatus.FAIL,
        checks=tuple(checks),
        retryable=not passed,
        correction=None if passed else _correction(checks),
    )


def _is_fresh(observation: Observation | None, now: datetime, threshold: int | None) -> bool:
    if observation is None or (
        observation.expires_at is not None and observation.expires_at <= now
    ):
        return False
    if threshold is None:
        return True
    return (now - observation.observed_at).total_seconds() <= threshold


def _correction(checks: list[VerifierCheck]) -> str:
    failed = tuple(check.name for check in checks if not check.passed)
    return "Corrective evidence is required: " + ", ".join(failed[:4])
