"""Versioned deterministic operator fixtures backed by the real coordinator."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from leo.demo import run_conversation_smoke
from leo.evals.loader import default_scenario_root, load_scenarios
from leo.evals.runner import execute_scenario_trace
from leo.harness.models import (
    CompletionProposal,
    ContractModel,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    NonEmptyStr,
    RunBundle,
    RunStatus,
)
from leo.replay import NormalizedReplay, normalize_replay

FIXTURE_CATALOG_VERSION = "fixture-catalog-v1"


class FixtureKind(StrEnum):
    DIRECT = "direct"
    CLARIFICATION = "clarification"
    ONE_TOOL = "one_tool"
    DELEGATED = "delegated"
    REPLANNING = "replanning"
    FAILURE = "failure"


class FixtureSpec(ContractModel):
    id: NonEmptyStr
    version: str = Field(pattern=r"^v[0-9]+$")
    kind: FixtureKind
    purpose: NonEmptyStr
    scenario_id: str | None = None
    expected_status: RunStatus
    fixture_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def digest_matches_descriptor(self) -> FixtureSpec:
        expected = _digest(self.model_dump(mode="json", exclude={"fixture_digest"}))
        if self.fixture_digest != expected:
            raise ValueError("operator fixture digest mismatch")
        return self


class FixtureRun(ContractModel):
    catalog_version: NonEmptyStr = FIXTURE_CATALOG_VERSION
    fixture: FixtureSpec
    replay: NormalizedReplay

    @model_validator(mode="after")
    def result_matches_expected_status(self) -> FixtureRun:
        if self.replay.status is not self.fixture.expected_status:
            raise ValueError("operator fixture produced an unexpected terminal status")
        if self.replay.status is not RunStatus.COMPLETED and self.replay.final_output is not None:
            raise ValueError("failed operator fixture cannot expose a final output")
        return self


class FixtureNotFoundError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("fixture_not_found")
        self.safe_code = "fixture_not_found"


class _ClarificationModel:
    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        del request
        return ModelTurnResult(
            decision=CompletionProposal(
                answer=(
                    "Could you clarify which portfolio, time horizon, and comparison "
                    "criterion you want me to use?"
                )
            ),
            provider="fixture-model",
            model="clarification-v1",
            request_id="clarification-request-1",
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=8, completion_tokens=18, total_tokens=26),
        )


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _spec(
    fixture_id: str,
    kind: FixtureKind,
    purpose: str,
    *,
    scenario_id: str | None,
    expected_status: RunStatus,
) -> FixtureSpec:
    payload = {
        "id": fixture_id,
        "version": "v1",
        "kind": kind.value,
        "purpose": purpose,
        "scenario_id": scenario_id,
        "expected_status": expected_status.value,
    }
    return FixtureSpec.model_validate({**payload, "fixture_digest": _digest(payload)})


FIXTURE_CATALOG: tuple[FixtureSpec, ...] = (
    _spec(
        "arbitrary-direct",
        FixtureKind.DIRECT,
        "Answer an arbitrary conversational request using exact local context.",
        scenario_id="contextual_conversation",
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "clarification",
        FixtureKind.CLARIFICATION,
        "Ask a bounded clarification when a research request is materially underspecified.",
        scenario_id=None,
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "one-tool",
        FixtureKind.ONE_TOOL,
        "Execute one deterministic read tool and ground the answer.",
        scenario_id="quote_control",
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "nvda-source-rich",
        FixtureKind.ONE_TOOL,
        "Run the source-rich NVDA quote golden.",
        scenario_id="quote_control",
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "delegated-plan",
        FixtureKind.DELEGATED,
        "Execute a dependency-aware bounded child-research plan.",
        scenario_id="delegated_dependency_plan",
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "verifier-replanning",
        FixtureKind.REPLANNING,
        "Correct a rejected proposal within the bounded parent retry loop.",
        scenario_id="verifier_correction",
        expected_status=RunStatus.COMPLETED,
    ),
    _spec(
        "verifier-safe-failure",
        FixtureKind.FAILURE,
        "Reject fabricated evidence without rendering false success.",
        scenario_id="safe_failure",
        expected_status=RunStatus.BUDGET_EXHAUSTED,
    ),
)
_FIXTURES_BY_ID = {fixture.id: fixture for fixture in FIXTURE_CATALOG}


def fixture_ids() -> tuple[str, ...]:
    return tuple(fixture.id for fixture in FIXTURE_CATALOG)


async def run_fixture(fixture_id: str) -> FixtureRun:
    try:
        fixture = _FIXTURES_BY_ID[fixture_id]
    except KeyError as exc:
        raise FixtureNotFoundError from exc
    if fixture.kind is FixtureKind.CLARIFICATION:
        result = await run_conversation_smoke(
            model=_ClarificationModel(),
            objective="Compare it and tell me what to do.",
        )
    else:
        assert fixture.scenario_id is not None
        scenario = load_scenarios(
            default_scenario_root(),
            scenario_ids=frozenset({fixture.scenario_id}),
        )[0]
        result = await execute_scenario_trace(scenario)
    bundle = RunBundle(
        thread=result.thread,
        task=result.task,
        run=result.run,
        observations=result.observations,
        claims=result.claims,
        events=result.events,
    )
    return FixtureRun(fixture=fixture, replay=normalize_replay(bundle))
