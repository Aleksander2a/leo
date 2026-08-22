"""Content-addressed evidence contract for rollback-safe durable recovery probes."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr


class DurableRecoveryOutcome(StrEnum):
    REJECTED_SAFE = "rejected_safe"
    RELOAD_EXACT = "reload_exact"
    RECLAIMED = "reclaimed"
    FENCED = "fenced"
    EXPORTED = "exported"


class DurableRecoveryCase(ContractModel):
    id: NonEmptyStr
    boundary: NonEmptyStr
    outcome: DurableRecoveryOutcome
    observed_before_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    observed_after_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    mutation_applied: bool
    duplicate_committed: bool = False
    terminal_success: bool = False
    detail_code: NonEmptyStr

    @model_validator(mode="after")
    def recovery_is_safe_and_observed(self) -> DurableRecoveryCase:
        if self.duplicate_committed or self.terminal_success:
            raise ValueError("durable recovery artifact cannot attest unsafe success")
        if (
            self.outcome
            in {
                DurableRecoveryOutcome.REJECTED_SAFE,
                DurableRecoveryOutcome.FENCED,
            }
            and self.mutation_applied
        ):
            raise ValueError("rejected/fenced recovery cannot apply the rejected mutation")
        if (
            self.outcome
            in {
                DurableRecoveryOutcome.RECLAIMED,
                DurableRecoveryOutcome.EXPORTED,
            }
            and not self.mutation_applied
        ):
            raise ValueError("reclaim/export recovery requires an observed mutation")
        return self


class DurableRecoveryArtifact(ContractModel):
    version: NonEmptyStr = "durable-recovery-v1"
    database_label: NonEmptyStr = "supabase-postgres-current-head"
    rollback_safe: bool = True
    cases: tuple[DurableRecoveryCase, ...] = Field(min_length=1)
    case_count: int = Field(ge=1)
    duplicate_commit_count: int = Field(ge=0)
    false_success_count: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def counts_and_digest_reconcile(self) -> DurableRecoveryArtifact:
        if not self.rollback_safe:
            raise ValueError("durable recovery artifact must use outer rollback")
        if self.case_count != len(self.cases):
            raise ValueError("durable recovery case count mismatch")
        if self.duplicate_commit_count != sum(item.duplicate_committed for item in self.cases):
            raise ValueError("durable recovery duplicate count mismatch")
        if self.false_success_count != sum(item.terminal_success for item in self.cases):
            raise ValueError("durable recovery false-success count mismatch")
        if len({item.id for item in self.cases}) != len(self.cases):
            raise ValueError("durable recovery case IDs must be unique")
        if self.digest != _digest(self.model_dump(mode="json", exclude={"digest"})):
            raise ValueError("durable recovery artifact digest mismatch")
        return self


def make_durable_recovery_case(
    *,
    case_id: str,
    boundary: str,
    outcome: DurableRecoveryOutcome,
    before: object,
    after: object,
    mutation_applied: bool,
    detail_code: str,
) -> DurableRecoveryCase:
    return DurableRecoveryCase(
        id=case_id,
        boundary=boundary,
        outcome=outcome,
        observed_before_digest=_digest(before),
        observed_after_digest=_digest(after),
        mutation_applied=mutation_applied,
        detail_code=detail_code,
    )


def make_durable_recovery_artifact(
    cases: tuple[DurableRecoveryCase, ...],
) -> DurableRecoveryArtifact:
    payload = {
        "version": "durable-recovery-v1",
        "database_label": "supabase-postgres-current-head",
        "rollback_safe": True,
        "cases": [item.model_dump(mode="json") for item in cases],
        "case_count": len(cases),
        "duplicate_commit_count": sum(item.duplicate_committed for item in cases),
        "false_success_count": sum(item.terminal_success for item in cases),
    }
    return DurableRecoveryArtifact.model_validate({**payload, "digest": _digest(payload)})


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()
