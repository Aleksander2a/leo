"""Reproducible milestone-five reliability proof manifest."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field, JsonValue, model_validator

from leo.evals.baseline import BaselineResult
from leo.evals.metrics import build_comparison_report
from leo.evals.models import ProviderMode, Scenario, ScenarioResult, ScenarioStatus
from leo.evals.recordings import sanitize_payload
from leo.harness.models import ContractModel, NonEmptyStr

REQUIRED_PROOF_SCENARIOS: dict[str, frozenset[str]] = {
    "arbitrary_conversational_golden": frozenset({"contextual_conversation"}),
    "nvda_source_golden": frozenset({"quote_control"}),
    "delegated_replanning_golden": frozenset({"delegated_dependency_plan", "verifier_correction"}),
    "restart_recovery": frozenset({"restart_replay_idempotency"}),
    "duplicate_suppression": frozenset({"restart_replay_idempotency"}),
    "dm_membership_exclusion": frozenset({"dm_context_union"}),
    "verifier_rejection": frozenset({"safe_failure"}),
    "memory_lifecycle": frozenset({"memory_lifecycle"}),
    "long_thread_compaction": frozenset({"long_thread_compaction"}),
    "tool_recall_no_progress": frozenset({"tool_recall_progressive"}),
    "shared_group_external_authority": frozenset({"shared_group_external_scope"}),
    "budget_boundary": frozenset({"budget_boundary"}),
    "fault_recovery_matrix": frozenset({"fault_recovery_matrix"}),
    "conversational_terminal_recovery": frozenset({"conversational_terminal_recovery"}),
    "elastic_deliberation": frozenset({"elastic_deliberation"}),
    "slack_thread_context_authority": frozenset({"slack_thread_context_authority"}),
    "tavily_verified_research": frozenset({"tavily_verified_research"}),
    "paired_baseline_report": frozenset(
        {
            "budget_boundary",
            "channel_isolation",
            "contextual_conversation",
            "conversational_terminal_recovery",
            "delegated_dependency_plan",
            "dm_context_union",
            "elastic_deliberation",
            "fault_recovery_matrix",
            "long_thread_compaction",
            "memory_lifecycle",
            "parallel_read_batch",
            "quote_control",
            "restart_replay_idempotency",
            "safe_failure",
            "shared_group_external_scope",
            "slack_thread_context_authority",
            "tavily_verified_research",
            "tool_recall_progressive",
            "verifier_correction",
        }
    ),
}
REQUIRED_PROOF_ARTIFACT_IDS = frozenset(REQUIRED_PROOF_SCENARIOS)


class ProofArtifact(ContractModel):
    id: NonEmptyStr
    kind: NonEmptyStr
    command: NonEmptyStr
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    scenario_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    fixture_digests: tuple[str, ...] = Field(min_length=1)
    provider_label: NonEmptyStr = "offline"
    sanitized_run_ids: tuple[NonEmptyStr, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def scenario_fixture_pairs_are_exact(self) -> ProofArtifact:
        if len(self.scenario_ids) != len(self.fixture_digests):
            raise ValueError("proof scenario IDs and fixture digests must pair exactly")
        if len(self.scenario_ids) != len(set(self.scenario_ids)):
            raise ValueError("proof scenario IDs must be unique")
        if any(
            len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest)
            for digest in self.fixture_digests
        ):
            raise ValueError("proof fixture digests must be lowercase SHA-256 values")
        if any(token in self.command for token in ("\n", "\r", ";", "&&", "||")):
            raise ValueError("proof command must be a single non-chained command")
        sanitize_payload({"proof_command": self.command})
        sanitize_payload(self.metadata)
        if self.digest != _artifact_digest(self):
            raise ValueError("proof artifact digest mismatch")
        return self


class ProofManifest(ContractModel):
    version: NonEmptyStr = "proof-v2"
    code_version: NonEmptyStr
    fixture_versions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    model_catalog_version: NonEmptyStr
    tool_catalog_version: NonEmptyStr
    policy_versions: tuple[NonEmptyStr, ...] = Field(min_length=1)
    artifacts: tuple[ProofArtifact, ...] = Field(min_length=1)
    final_golden_bound: bool = True

    @property
    def reproducible(self) -> bool:
        ids = {artifact.id for artifact in self.artifacts}
        return self.final_golden_bound and REQUIRED_PROOF_ARTIFACT_IDS <= ids

    @property
    def digest(self) -> str:
        return _digest(self.model_dump(mode="json"))


def make_proof_artifact(
    *,
    artifact_id: str,
    kind: str,
    command: str,
    scenario_ids: tuple[str, ...],
    fixture_digests: tuple[str, ...],
    provider_label: str = "offline",
    sanitized_run_ids: tuple[str, ...] = (),
    metadata: dict[str, JsonValue] | None = None,
) -> ProofArtifact:
    payload = {
        "id": artifact_id,
        "kind": kind,
        "command": command,
        "scenario_ids": scenario_ids,
        "fixture_digests": fixture_digests,
        "provider_label": provider_label,
        "sanitized_run_ids": sanitized_run_ids,
        "metadata": metadata or {},
    }
    return ProofArtifact.model_validate({**payload, "digest": _digest(payload)})


def build_offline_proof_manifest(
    scenarios: tuple[Scenario, ...],
    results: tuple[ScenarioResult, ...],
    baselines: tuple[BaselineResult, ...],
    *,
    code_version: str,
    model_catalog_version: str,
    tool_catalog_version: str,
    policy_versions: tuple[str, ...],
) -> ProofManifest:
    """Bind runnable offline artifacts to observed passing fixtures and baseline deltas."""

    scenario_by_id = _unique_scenarios(scenarios)
    result_by_id = {item.scenario_id: item for item in results}
    baseline_by_id = {item.scenario_id: item for item in baselines}
    if (
        len(result_by_id) != len(results)
        or len(baseline_by_id) != len(baselines)
        or set(result_by_id) != set(scenario_by_id)
        or set(baseline_by_id) != set(scenario_by_id)
    ):
        raise ValueError("proof inputs must exactly cover unique scenario fixtures")
    for scenario_id, scenario in scenario_by_id.items():
        result = result_by_id[scenario_id]
        baseline = baseline_by_id[scenario_id]
        if (
            result.status is not ScenarioStatus.PASSED
            or baseline.status is not ScenarioStatus.PASSED
            or result.provider_mode is not ProviderMode.OFFLINE
            or baseline.provider_mode is not ProviderMode.OFFLINE
            or result.scenario_version != scenario.version
            or baseline.scenario_version != scenario.version
            or result.fixture_digest != scenario.fixture_digest
            or baseline.fixture_digest != scenario.fixture_digest
        ):
            raise ValueError("proof inputs contain a failed or unmatched fixture result")
    comparison = build_comparison_report(
        results,
        baselines,
        config_digest=_proof_config_digest(scenarios),
    )
    artifacts: list[ProofArtifact] = []
    for artifact_id, required_ids in REQUIRED_PROOF_SCENARIOS.items():
        ids = tuple(sorted(required_ids))
        command = (
            "python -m leo.evals --baseline"
            if artifact_id == "paired_baseline_report"
            else "python -m leo.evals " + " ".join(f"--id {item}" for item in ids)
        )
        metadata: dict[str, JsonValue] = {
            "result_replay_pointers": {item: result_by_id[item].replay_pointer for item in ids}
        }
        if artifact_id == "paired_baseline_report":
            metadata["comparison_digest"] = comparison.digest
            metadata["paired_delta_count"] = len(comparison.paired_deltas)
        artifacts.append(
            make_proof_artifact(
                artifact_id=artifact_id,
                kind=(
                    "paired_baseline_report"
                    if artifact_id == "paired_baseline_report"
                    else "deterministic_golden"
                ),
                command=command,
                scenario_ids=ids,
                fixture_digests=tuple(scenario_by_id[item].fixture_digest for item in ids),
                metadata=metadata,
            )
        )
    manifest = ProofManifest(
        code_version=code_version,
        fixture_versions=tuple(
            f"{item.id}:{item.version}" for item in sorted(scenarios, key=lambda row: row.id)
        ),
        model_catalog_version=model_catalog_version,
        tool_catalog_version=tool_catalog_version,
        policy_versions=tuple(sorted(set(policy_versions))),
        artifacts=tuple(artifacts),
    )
    validate_proof_manifest(manifest)
    return manifest


def validate_proof_manifest(manifest: ProofManifest) -> None:
    ids = [artifact.id for artifact in manifest.artifacts]
    if len(ids) != len(set(ids)):
        raise ValueError("proof artifact IDs must be unique")
    if not manifest.final_golden_bound:
        raise ValueError("P004-D03 is resolved; all final goldens must now be bound")
    missing = REQUIRED_PROOF_ARTIFACT_IDS - set(ids)
    if missing:
        raise ValueError(f"proof manifest is missing required artifacts: {sorted(missing)}")
    by_id = {artifact.id: artifact for artifact in manifest.artifacts}
    for artifact_id, required_scenarios in REQUIRED_PROOF_SCENARIOS.items():
        artifact = by_id[artifact_id]
        if artifact.digest != _artifact_digest(artifact):
            raise ValueError(f"proof artifact {artifact_id} digest mismatch")
        actual = frozenset(artifact.scenario_ids)
        if actual != required_scenarios:
            raise ValueError(f"proof artifact {artifact_id} has the wrong scenario set")
        if artifact_id == "paired_baseline_report":
            if artifact.command != "python -m leo.evals --baseline":
                raise ValueError("paired baseline proof command is not reproducible")
        elif any(f"--id {scenario_id}" not in artifact.command for scenario_id in actual):
            raise ValueError(f"proof artifact {artifact_id} command omits a scenario")
    declared_fixture_ids = {
        version.split(":", maxsplit=1)[0] for version in manifest.fixture_versions
    }
    required_fixture_ids = set().union(*REQUIRED_PROOF_SCENARIOS.values())
    if not required_fixture_ids <= declared_fixture_ids:
        raise ValueError("proof manifest fixture versions omit required scenarios")
    if any(
        by_id[artifact_id].provider_label != "offline"
        for artifact_id in REQUIRED_PROOF_ARTIFACT_IDS
    ):
        raise ValueError("required reliability proof artifacts need an offline fallback")


def _artifact_digest(artifact: ProofArtifact) -> str:
    return _digest(artifact.model_dump(mode="json", exclude={"digest"}))


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _unique_scenarios(scenarios: tuple[Scenario, ...]) -> dict[str, Scenario]:
    output = {item.id: item for item in scenarios}
    if len(output) != len(scenarios):
        raise ValueError("proof scenarios must be unique")
    return output


def _proof_config_digest(scenarios: tuple[Scenario, ...]) -> str:
    return _digest(
        [
            scenario.model_dump(
                mode="json",
                include={
                    "id",
                    "version",
                    "provider_mode",
                    "fixed_clock",
                    "budget",
                    "inputs",
                },
            )
            for scenario in sorted(scenarios, key=lambda item: item.id)
        ]
    )
