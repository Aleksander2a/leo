"""Frozen, executable, deliberately simple matched-scenario baseline."""

from __future__ import annotations

import asyncio
import hashlib
import json

from pydantic import Field, model_validator

from leo.evals.closure_scenarios import (
    CLOSURE_VARIANTS,
    ClosureScenarioUnsupported,
    execute_closure_baseline_scenario,
)
from leo.evals.milestone5 import (
    MILESTONE5_VARIANTS,
    Milestone5UnsupportedScenario,
    execute_milestone5_baseline_scenario,
)
from leo.evals.models import (
    EvalBudget,
    ProviderMode,
    Scenario,
    ScenarioResult,
    ScenarioStatus,
)
from leo.evals.revised_m5_scenarios import (
    REVISED_M5_VARIANTS,
    RevisedM5UnsupportedScenario,
    execute_revised_m5_baseline_scenario,
)
from leo.evals.runner import ControlUnsupportedScenario, execute_control_baseline_scenario
from leo.harness.models import ContractModel, NonEmptyStr


class BaselinePolicy(ContractModel):
    version: NonEmptyStr = "baseline-v2"
    recent_turn_limit: int = Field(default=4, ge=1, le=16)
    exact_destination_only: bool = True
    use_dm_union: bool = False
    use_long_term_memory: bool = False
    use_durable_plans: bool = False
    use_subagents: bool = False
    use_tool_recall: bool = False
    correction_retries: int = Field(default=0, ge=0, le=0)
    eligible_schema_rule: NonEmptyStr = "all_baseline_eligible_that_fit"

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


class BaselineResult(ContractModel):
    scenario_id: NonEmptyStr
    scenario_version: NonEmptyStr
    fixture_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    match_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: NonEmptyStr
    policy_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: ScenarioStatus
    reason: NonEmptyStr
    provider_mode: ProviderMode
    admitted_destination: NonEmptyStr
    model_fixture: NonEmptyStr
    budget: EvalBudget
    matched_tool_catalog: tuple[NonEmptyStr, ...] = ()
    exposed_tool_catalog: tuple[NonEmptyStr, ...] = ()
    tool_schema_count: int = Field(ge=0)
    feature_flags: frozenset[NonEmptyStr]
    observed_invariants: frozenset[NonEmptyStr] = frozenset()
    hard_failures: tuple[NonEmptyStr, ...] = ()
    metrics: dict[str, float | int | str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def executable_catalog_is_consistent(self) -> BaselineResult:
        if self.tool_schema_count != len(self.exposed_tool_catalog):
            raise ValueError("baseline tool schema count does not match exposed catalog")
        if not set(self.exposed_tool_catalog).issubset(self.matched_tool_catalog):
            raise ValueError("baseline exposed tools are not in the matched catalog")
        return self


_BASELINE_FEATURE_FLAGS = frozenset(
    {
        "exact_destination_recent_turns",
        "no_dm_union",
        "no_long_term_memory",
        "no_durable_plans",
        "no_subagents",
        "no_tool_recall",
        "no_correction_retry",
    }
)


async def run_baseline_async(
    scenario: Scenario,
    *,
    eligible_schema_count: int | None = None,
) -> BaselineResult:
    """Execute one real coordinator path under the frozen baseline policy."""

    if eligible_schema_count is not None and eligible_schema_count < 0:
        raise ValueError("eligible schema count cannot be negative")
    policy = BaselinePolicy()
    if scenario.provider_mode is not ProviderMode.OFFLINE:
        return _unsupported_baseline(
            scenario,
            policy,
            eligible_schema_count=eligible_schema_count or 0,
            reason=f"provider_mode_not_executable:{scenario.provider_mode.value}",
        )
    try:
        if scenario.execution_variant in CLOSURE_VARIANTS:
            execution = await execute_closure_baseline_scenario(scenario)
        elif scenario.execution_variant in REVISED_M5_VARIANTS:
            execution = await execute_revised_m5_baseline_scenario(scenario)
        elif scenario.execution_variant in MILESTONE5_VARIANTS:
            execution = await execute_milestone5_baseline_scenario(scenario)
        else:
            execution = await execute_control_baseline_scenario(scenario)
    except (
        ClosureScenarioUnsupported,
        Milestone5UnsupportedScenario,
        RevisedM5UnsupportedScenario,
        ControlUnsupportedScenario,
    ) as exc:
        return _unsupported_baseline(
            scenario,
            policy,
            eligible_schema_count=eligible_schema_count or 0,
            reason=str(exc),
        )
    if (
        eligible_schema_count is not None
        and eligible_schema_count != execution.eligible_schema_count
    ):
        raise ValueError("baseline eligible schema count does not match the executable catalog")

    match_digest = _match_digest(
        scenario,
        admitted_destination=execution.admitted_destination,
        model_fixture=execution.model_fixture,
        matched_tool_catalog=execution.matched_tool_catalog,
    )
    safe = not execution.hard_failures and "baseline_hard_safety_preserved" in execution.invariants
    return BaselineResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        fixture_digest=scenario.fixture_digest,
        match_digest=match_digest,
        policy_version=policy.version,
        policy_digest=policy.digest,
        status=ScenarioStatus.PASSED if safe else ScenarioStatus.FAILED,
        reason="baseline_safe_execution" if safe else "baseline_hard_safety_failure",
        provider_mode=scenario.provider_mode,
        admitted_destination=execution.admitted_destination,
        model_fixture=execution.model_fixture,
        budget=scenario.budget,
        matched_tool_catalog=execution.matched_tool_catalog,
        exposed_tool_catalog=execution.exposed_tool_catalog,
        tool_schema_count=execution.eligible_schema_count,
        feature_flags=_BASELINE_FEATURE_FLAGS,
        observed_invariants=execution.invariants,
        hard_failures=execution.hard_failures,
        metrics=execution.metrics,
    )


def run_baseline(
    scenario: Scenario,
    *,
    eligible_schema_count: int | None = None,
) -> BaselineResult:
    """Synchronous compatibility boundary used by report/CLI callers."""

    return asyncio.run(run_baseline_async(scenario, eligible_schema_count=eligible_schema_count))


def paired_baseline_scenario(
    scenario: Scenario,
    baseline: BaselineResult,
) -> ScenarioResult:
    if (
        baseline.scenario_id != scenario.id
        or baseline.scenario_version != scenario.version
        or baseline.fixture_digest != scenario.fixture_digest
        or baseline.match_digest
        != _match_digest(
            scenario,
            admitted_destination=baseline.admitted_destination,
            model_fixture=baseline.model_fixture,
            matched_tool_catalog=baseline.matched_tool_catalog,
        )
    ):
        raise ValueError("baseline result is not matched to the scenario fixture")
    return ScenarioResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        status=baseline.status,
        provider_mode=baseline.provider_mode,
        fixture_digest=scenario.fixture_digest,
        metrics={f"baseline_{name}": value for name, value in baseline.metrics.items()}
        | {"baseline_tool_schema_count": baseline.tool_schema_count},
        raw_counts={
            f"baseline_{name}": value
            for name, value in baseline.metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        },
        replay_pointer=(
            f"baseline:{baseline.policy_version}:{baseline.policy_digest}:{scenario.id}"
        ),
        reason=baseline.reason,
    )


def _unsupported_baseline(
    scenario: Scenario,
    policy: BaselinePolicy,
    *,
    eligible_schema_count: int,
    reason: str,
) -> BaselineResult:
    placeholder = f"unsupported:{scenario.deterministic_id_prefix}"
    return BaselineResult(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        fixture_digest=scenario.fixture_digest,
        match_digest=_match_digest(
            scenario,
            admitted_destination=placeholder,
            model_fixture="unsupported",
            matched_tool_catalog=(),
        ),
        policy_version=policy.version,
        policy_digest=policy.digest,
        status=ScenarioStatus.UNSUPPORTED,
        reason=reason,
        provider_mode=scenario.provider_mode,
        admitted_destination=placeholder,
        model_fixture="unsupported",
        budget=scenario.budget,
        tool_schema_count=eligible_schema_count,
        feature_flags=_BASELINE_FEATURE_FLAGS,
    )


def _match_digest(
    scenario: Scenario,
    *,
    admitted_destination: str,
    model_fixture: str,
    matched_tool_catalog: tuple[str, ...],
) -> str:
    return _digest(
        {
            "scenario_id": scenario.id,
            "scenario_version": scenario.version,
            "fixture_digest": scenario.fixture_digest,
            "provider_mode": scenario.provider_mode.value,
            "fixed_clock": scenario.fixed_clock,
            "inputs": scenario.inputs,
            "budget": scenario.budget.model_dump(mode="json"),
            "admitted_destination": admitted_destination,
            "model_fixture": model_fixture,
            "matched_tool_catalog": list(matched_tool_catalog),
        }
    )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
