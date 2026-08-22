"""Strict, versioned scenario and machine-result contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, JsonValue

from leo.harness.models import ContractModel, NonEmptyStr


class ProviderMode(StrEnum):
    OFFLINE = "offline"
    RECORDED = "recorded"
    LIVE = "live"


class ScenarioStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class EvalBudget(ContractModel):
    max_model_calls: int = Field(ge=0, le=64)
    max_tool_calls: int = Field(ge=0, le=128)
    max_elapsed_seconds: float = Field(gt=0, le=3600)


class Scenario(ContractModel):
    id: NonEmptyStr
    version: str = Field(pattern=r"^v[0-9]+$")
    purpose: NonEmptyStr
    provider_mode: ProviderMode
    fixture_digest: str = Field(min_length=64, max_length=64)
    fixed_clock: NonEmptyStr
    deterministic_id_prefix: NonEmptyStr
    execution_variant: NonEmptyStr
    budget: EvalBudget
    inputs: dict[str, JsonValue]
    expected_hard_invariants: frozenset[NonEmptyStr] = Field(min_length=1)
    expected_quality_metrics: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    allowed_nondeterminism: frozenset[NonEmptyStr] = Field(default_factory=frozenset)


class ScenarioResult(ContractModel):
    scenario_id: NonEmptyStr
    scenario_version: NonEmptyStr
    status: ScenarioStatus
    provider_mode: ProviderMode
    fixture_digest: str = Field(min_length=64, max_length=64)
    invariant_failures: tuple[NonEmptyStr, ...] = ()
    metrics: dict[str, float | int | str] = Field(default_factory=dict)
    raw_counts: dict[str, float | int] = Field(default_factory=dict)
    replay_pointer: NonEmptyStr
    reason: NonEmptyStr
