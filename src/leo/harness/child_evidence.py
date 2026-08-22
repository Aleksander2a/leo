"""Versioned evidence exported from a verified child run to its parent."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Annotated, Literal, cast

from pydantic import Field, JsonValue, model_validator

from leo.harness.models import Claim, ClaimKind, ContractModel, NonEmptyStr, Observation

CHILD_EVIDENCE_SCHEMA_VERSION: Literal["child-evidence-v1"] = "child-evidence-v1"
_MAX_CHILD_ANSWER_CHARS = 32_768


class ChildEvidenceSource(ContractModel):
    """One exact child Observation cited by a persisted verified source claim."""

    observation_id: NonEmptyStr
    kind: NonEmptyStr
    provider: NonEmptyStr
    reference: NonEmptyStr
    url: str | None = None
    observed_at: datetime
    expires_at: datetime | None = None
    raw_hash: NonEmptyStr


class VerifiedChildSourceClaim(ContractModel):
    """A harness-created child source claim and its exact evidence projection."""

    claim_id: NonEmptyStr
    statement: Annotated[str, Field(min_length=1, max_length=8_192)]
    sources: tuple[ChildEvidenceSource, ...] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def source_ids_are_unique(self) -> VerifiedChildSourceClaim:
        source_ids = tuple(source.observation_id for source in self.sources)
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("verified child claim source IDs must be unique")
        return self


class ChildEvidenceEnvelope(ContractModel):
    """Bounded child result whose source claims came from durable verifier output."""

    schema_version: Literal["child-evidence-v1"] = CHILD_EVIDENCE_SCHEMA_VERSION
    child_run_id: NonEmptyStr
    answer: Annotated[str, Field(min_length=1, max_length=_MAX_CHILD_ANSWER_CHARS)]
    trace_event_count: int = Field(ge=1, le=100_000)
    observation_count: int = Field(ge=0, le=1_024)
    verified_source_claims: tuple[VerifiedChildSourceClaim, ...] = Field(default=(), max_length=8)
    digest: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def validate_evidence_and_digest(self) -> ChildEvidenceEnvelope:
        claim_ids = tuple(claim.claim_id for claim in self.verified_source_claims)
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("verified child claim IDs must be unique")
        cited_source_ids = {
            source.observation_id
            for claim in self.verified_source_claims
            for source in claim.sources
        }
        if len(cited_source_ids) > self.observation_count:
            raise ValueError("verified child sources exceed the child observation count")
        if any(
            not _contains_statement(self.answer, claim.statement)
            for claim in self.verified_source_claims
        ):
            raise ValueError("verified child source claim must be carried by the child answer")
        expected = _child_evidence_digest(self.model_dump(mode="json", exclude={"digest"}))
        if self.digest != expected:
            raise ValueError("child evidence digest does not match its verified payload")
        return self


# Stable milestone name for the existing verified, bounded child return contract.
ChildResult = ChildEvidenceEnvelope


class ChildEvidenceError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def build_child_evidence_envelope(
    *,
    child_run_id: str,
    answer: str,
    trace_event_count: int,
    observations: tuple[Observation, ...],
    claims: tuple[Claim, ...],
) -> ChildEvidenceEnvelope:
    """Project only persisted verified SOURCE_CLAIM records into a parent envelope."""

    observations_by_id = {observation.id: observation for observation in observations}
    verified_claims: list[VerifiedChildSourceClaim] = []
    for claim in claims:
        if claim.kind is not ClaimKind.SOURCE_CLAIM:
            continue
        if claim.run_id != child_run_id or not claim.observation_ids:
            raise ChildEvidenceError("child_claim_authority_invalid")
        sources: list[ChildEvidenceSource] = []
        for observation_id in claim.observation_ids:
            observation = observations_by_id.get(observation_id)
            if (
                observation is None
                or observation.run_id != child_run_id
                or observation.scope != claim.scope
            ):
                raise ChildEvidenceError("child_claim_observation_invalid")
            sources.append(
                ChildEvidenceSource(
                    observation_id=observation.id,
                    kind=observation.kind,
                    provider=observation.source.provider,
                    reference=observation.source.reference,
                    url=observation.source.url,
                    observed_at=observation.observed_at,
                    expires_at=observation.expires_at,
                    raw_hash=observation.raw_hash,
                )
            )
        verified_claims.append(
            VerifiedChildSourceClaim(
                claim_id=claim.id,
                statement=claim.statement,
                sources=tuple(sources),
            )
        )

    payload: dict[str, JsonValue] = {
        "schema_version": CHILD_EVIDENCE_SCHEMA_VERSION,
        "child_run_id": child_run_id,
        "answer": answer,
        "trace_event_count": trace_event_count,
        "observation_count": len(observations),
        "verified_source_claims": cast(
            list[JsonValue],
            [claim.model_dump(mode="json") for claim in verified_claims],
        ),
    }
    return ChildEvidenceEnvelope.model_validate(
        {**payload, "digest": _child_evidence_digest(payload)}
    )


def parse_child_evidence_envelope(value: object) -> ChildEvidenceEnvelope:
    """Parse one current envelope. Legacy values are deliberately not upgraded to evidence."""

    try:
        payload = json.loads(value) if isinstance(value, str) else value
        return ChildEvidenceEnvelope.model_validate(payload)
    except (TypeError, ValueError) as exc:
        raise ChildEvidenceError("child_evidence_malformed") from exc


def serialize_child_evidence_envelope(envelope: ChildEvidenceEnvelope) -> str:
    """Canonical representation suitable for the existing durable plan-node Text column."""

    return json.dumps(
        envelope.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def child_evidence_data(envelope: ChildEvidenceEnvelope) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], envelope.model_dump(mode="json"))


def child_evidence_expires_at(envelope: ChildEvidenceEnvelope) -> datetime | None:
    expiries = tuple(
        source.expires_at
        for claim in envelope.verified_source_claims
        for source in claim.sources
        if source.expires_at is not None
    )
    return min(expiries) if expiries else None


def _child_evidence_digest(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _contains_statement(text: str, statement: str) -> bool:
    normalized_text = " ".join(text.split()).casefold()
    normalized_statement = " ".join(statement.split()).casefold()
    return bool(normalized_statement) and normalized_statement in normalized_text
