from __future__ import annotations

import hashlib
import inspect
import json
import socket
import subprocess
import sys
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from leo.evals.frozen_report import EXTERNAL_EVIDENCE_CONTRACTS
from leo.evals.live_proof import (
    LIVE_PROOF_ARTIFACT_ID,
    M5_LIVE_EVIDENCE_IDS,
    AsyncLiveProofSource,
    LiveEvidenceId,
    LiveProofAuthority,
    LiveProofBinding,
    LiveProofCase,
    LiveProofIntegrityError,
    LiveProofNotFound,
    LiveProofRequest,
    PostgresLiveProofSource,
    _memory_recall_summary,
    attach_live_collection,
    collect_live_proof,
    require_complete_live_proof,
)
from leo.evals.live_proof_operator import _run_async
from leo.evals.proof import (
    REQUIRED_PROOF_SCENARIOS,
    ProofManifest,
    make_proof_artifact,
    validate_proof_manifest,
)
from leo.persistence.schema import ClaimRow, ObservationRow, RunRow


class _FakeSource(AsyncLiveProofSource):
    def __init__(
        self,
        cases: dict[LiveEvidenceId, LiveProofCase],
        *,
        missing: LiveEvidenceId | None = None,
    ) -> None:
        self.cases = cases
        self.missing = missing
        self.calls: list[LiveEvidenceId] = []

    async def load(
        self,
        *,
        authority: LiveProofAuthority,
        binding: LiveProofBinding,
    ) -> LiveProofCase:
        assert authority.permits(binding)
        self.calls.append(binding.evidence_id)
        if binding.evidence_id is self.missing:
            raise LiveProofNotFound
        return self.cases[binding.evidence_id]


def _binding(evidence_id: LiveEvidenceId, index: int) -> LiveProofBinding:
    if evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION:
        destination_id = "D-DEMO"
        conversation_kind = "dm"
        context_ids = ("C-SHARED", destination_id)
        recall_source = "C-SHARED"
    elif evidence_id is LiveEvidenceId.GROUP_DM:
        destination_id = "G-DEMO"
        conversation_kind = "mpim"
        context_ids = (destination_id,)
        recall_source = None
    else:
        destination_id = "C-DEMO"
        conversation_kind = "ordinary_internal"
        context_ids = (destination_id,)
        recall_source = destination_id if evidence_id is LiveEvidenceId.MEMORY_RECALL else None
    return LiveProofBinding(
        evidence_id=evidence_id,
        message_ts=f"17880000{index:02d}.000001",
        run_id=f"run-00000000-0000-4000-8000-{index:012d}",
        expected_destination_id=destination_id,
        expected_conversation_kind=conversation_kind,
        expected_context_conversation_ids=context_ids,
        expected_context_access_hash=hashlib.sha256(str(evidence_id).encode()).hexdigest(),
        expected_recall_source_conversation_id=recall_source,
        plan_expectation=(
            "required" if evidence_id is LiveEvidenceId.DELEGATED_REPLANNING else "forbidden"
        ),
    )


def _request(*, complete: bool) -> LiveProofRequest:
    admitted_ids = (
        M5_LIVE_EVIDENCE_IDS
        if complete
        else tuple(
            item for item in M5_LIVE_EVIDENCE_IDS if item is not LiveEvidenceId.DELEGATED_REPLANNING
        )
    )
    bindings = tuple(
        sorted(
            (_binding(evidence_id, index + 1) for index, evidence_id in enumerate(admitted_ids)),
            key=lambda item: str(item.evidence_id),
        )
    )
    pending = () if complete else (LiveEvidenceId.DELEGATED_REPLANNING,)
    return LiveProofRequest(bindings=bindings, pending_evidence_ids=pending)


def _authority(request: LiveProofRequest) -> LiveProofAuthority:
    return LiveProofAuthority(
        organization_id="demo-org",
        team_id="T-DEMO",
        actor_id="trusted-operator",
        not_before_received_at=datetime(2026, 8, 22, tzinfo=UTC),
        not_before_message_ts="1787999999.999999",
        allowed_bindings=request.bindings,
    )


def _case(binding: LiveProofBinding) -> LiveProofCase:
    plan_present = binding.evidence_id is LiveEvidenceId.DELEGATED_REPLANNING
    marker = hashlib.sha256(str(binding.evidence_id).encode()).hexdigest()
    payload: dict[str, object] = {
        "evidence_id": binding.evidence_id,
        "binding_digest": binding.digest,
        "message_ts": binding.message_ts,
        "run_id": binding.run_id,
        "task_id": f"task-{binding.evidence_id}",
        "channel_id": binding.expected_destination_id,
        "conversation_kind": binding.expected_conversation_kind,
        "slack_response_ts": binding.message_ts[:-1] + "2",
        "objective_digest": marker,
        "final_output_digest": marker,
        "task_terminal_state": "completed",
        "run_terminal_state": "completed",
        "event_terminal_state": "run_completed",
        "event_count": 8,
        "last_event_sequence": 8,
        "event_timeline_digest": marker,
        "context_access_hash": binding.expected_context_access_hash,
        "context_projection_source": (
            "dm_membership_intersection"
            if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION
            else "exact_destination"
        ),
        "context_conversation_count": len(binding.expected_context_conversation_ids),
        "conversation_source_set_digest": _digest(binding.expected_context_conversation_ids),
        "context_snapshot_digest": marker,
        "current_membership_count": (
            len(binding.expected_context_conversation_ids)
            if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION
            else 0
        ),
        "current_membership_digest": (
            marker if binding.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION else _digest([])
        ),
        "context_manifest_digest": marker,
        "message_plane_digest": marker,
        "memory_recall_verified": binding.expected_recall_source_conversation_id is not None,
        "memory_recall_source_digest": (
            _digest(binding.expected_recall_source_conversation_id)
            if binding.expected_recall_source_conversation_id is not None
            else _digest([])
        ),
        "memory_recall_observation_digest": (
            marker if binding.expected_recall_source_conversation_id is not None else _digest([])
        ),
        "grounded_memory_claim_digest": (
            marker if binding.expected_recall_source_conversation_id is not None else _digest([])
        ),
        "case_invariant_digest": marker,
        "observed_evidence_count": (
            1
            if binding.evidence_id not in {LiveEvidenceId.PRIVATE_CHANNEL, LiveEvidenceId.GROUP_DM}
            else 0
        ),
        "grounded_claim_count": (
            1
            if binding.evidence_id
            not in {
                LiveEvidenceId.MEMORY_WRITE,
                LiveEvidenceId.PRIVATE_CHANNEL,
                LiveEvidenceId.GROUP_DM,
            }
            else 0
        ),
        "memory_mutation_record_count": (
            1 if binding.evidence_id is LiveEvidenceId.MEMORY_WRITE else 0
        ),
        "memory_mutation_revision_count": (
            1 if binding.evidence_id is LiveEvidenceId.MEMORY_WRITE else 0
        ),
        "memory_mutation_source_count": (
            3 if binding.evidence_id is LiveEvidenceId.MEMORY_WRITE else 0
        ),
        "memory_mutation_digest": (
            marker if binding.evidence_id is LiveEvidenceId.MEMORY_WRITE else _digest([])
        ),
        "delegated_child_count": 2 if plan_present else 0,
        "delegated_overlap_verified": plan_present,
        "delegated_evidence_digest": marker if plan_present else _digest([]),
        "plan_present": plan_present,
        "plan_terminal_state": "completed" if plan_present else "absent",
        "plan_revision_digest": marker if plan_present else _digest([]),
        "plan_snapshot_digest": marker if plan_present else _digest([]),
        "plan_node_count": 2 if plan_present else 0,
        "delegation_count": 2 if plan_present else 0,
        "delivery_state": "delivered",
        "outbox_count": 1,
        "outbox_digest": marker,
        "ingress_digest": marker,
        "task_run_digest": marker,
    }
    payload["row_snapshot_digest"] = _digest(payload)
    return LiveProofCase.model_validate(payload)


def _cases(request: LiveProofRequest) -> dict[LiveEvidenceId, LiveProofCase]:
    return {item.evidence_id: _case(item) for item in request.bindings}


def _manifest() -> ProofManifest:
    fixture_ids = sorted(set().union(*REQUIRED_PROOF_SCENARIOS.values()))
    artifacts = tuple(
        make_proof_artifact(
            artifact_id=artifact_id,
            kind="deterministic_golden",
            command=(
                "python -m leo.evals --baseline"
                if artifact_id == "paired_baseline_report"
                else "python -m leo.evals "
                + " ".join(f"--id {item}" for item in sorted(scenario_ids))
            ),
            scenario_ids=tuple(sorted(scenario_ids)),
            fixture_digests=tuple("a" * 64 for _ in scenario_ids),
        )
        for artifact_id, scenario_ids in REQUIRED_PROOF_SCENARIOS.items()
    )
    return ProofManifest(
        code_version="test",
        fixture_versions=tuple(f"{item}:v1" for item in fixture_ids),
        model_catalog_version="models-v1",
        tool_catalog_version="tools-v1",
        policy_versions=("policy-v1",),
        artifacts=artifacts,
    )


@pytest.mark.asyncio
async def test_read_only_live_collection_is_deterministic_and_network_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(complete=False)
    authority = _authority(request)
    source = _FakeSource(_cases(request))

    def reject_network(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("live proof collection attempted network I/O")

    monkeypatch.setattr(socket.socket, "connect", reject_network)
    first = await collect_live_proof(source, authority=authority, request=request)
    second = await collect_live_proof(
        _FakeSource(_cases(request)),
        authority=authority,
        request=request,
    )
    assert first == second
    assert first.status == "partial"
    assert first.pending_evidence_ids == (LiveEvidenceId.DELEGATED_REPLANNING,)
    assert source.calls == [item.evidence_id for item in request.bindings]


@pytest.mark.asyncio
async def test_missing_post_restore_trace_fails_without_an_artifact() -> None:
    request = _request(complete=False)
    authority = _authority(request)
    missing = request.bindings[2].evidence_id
    source = _FakeSource(_cases(request), missing=missing)
    with pytest.raises(LiveProofNotFound):
        await collect_live_proof(source, authority=authority, request=request)
    assert len(source.calls) == 3


def test_old_or_prefix_only_run_inputs_are_rejected_before_collection() -> None:
    old = _binding(LiveEvidenceId.MEMORY_WRITE, 1).model_copy(
        update={"message_ts": "1787000000.000001"}
    )
    with pytest.raises(ValueError, match="predates the post-restore"):
        LiveProofAuthority(
            organization_id="demo-org",
            team_id="T-DEMO",
            actor_id="operator",
            not_before_received_at=datetime(2026, 8, 22, tzinfo=UTC),
            not_before_message_ts="1787999999.999999",
            allowed_bindings=(old,),
        )
    with pytest.raises(ValidationError):
        LiveProofBinding(
            evidence_id=LiveEvidenceId.MEMORY_WRITE,
            message_ts="1788000001.000001",
            run_id="run-2a3fe32a",
            expected_destination_id="C-DEMO",
            expected_conversation_kind="ordinary_internal",
            expected_context_conversation_ids=("C-DEMO",),
            expected_context_access_hash="a" * 64,
        )


@pytest.mark.asyncio
async def test_fake_trace_cannot_self_attest_or_mutate_an_expectation() -> None:
    request = _request(complete=False)
    authority = _authority(request)
    valid = _cases(request)
    binding = request.bindings[0]
    # model_copy simulates a custom source deliberately bypassing validation.
    valid[binding.evidence_id] = valid[binding.evidence_id].model_copy(update={"event_count": 999})
    with pytest.raises(ValidationError, match="row snapshot digest mismatch"):
        await collect_live_proof(
            _FakeSource(valid),
            authority=authority,
            request=request,
        )

    mismatched = _cases(request)
    wrong = mismatched[binding.evidence_id].model_dump(mode="json")
    wrong["conversation_kind"] = "mpim"
    wrong["row_snapshot_digest"] = _digest(
        {key: value for key, value in wrong.items() if key != "row_snapshot_digest"}
    )
    mismatched[binding.evidence_id] = LiveProofCase.model_validate(wrong)
    with pytest.raises(LiveProofIntegrityError, match="unmatched trace"):
        await collect_live_proof(
            _FakeSource(mismatched),
            authority=authority,
            request=request,
        )


@pytest.mark.asyncio
async def test_partial_live_artifact_does_not_claim_completion_and_is_sanitized() -> None:
    request = _request(complete=False)
    collection = await collect_live_proof(
        _FakeSource(_cases(request)),
        authority=_authority(request),
        request=request,
    )
    manifest = attach_live_collection(_manifest(), collection)
    validate_proof_manifest(manifest)
    artifact = next(item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID)
    assert artifact.provider_label == "slack-supabase-live"
    assert artifact.metadata["status"] == "partial"
    with pytest.raises(ValueError, match="lacks complete live"):
        require_complete_live_proof(manifest)
    encoded = manifest.model_dump_json()
    assert "objective text" not in encoded
    assert "final answer text" not in encoded
    assert "DATABASE_URL" not in encoded


@pytest.mark.asyncio
async def test_complete_exact_cohort_promotes_existing_proof_v2() -> None:
    request = _request(complete=True)
    collection = await collect_live_proof(
        _FakeSource(_cases(request)),
        authority=_authority(request),
        request=request,
    )
    assert collection.status == "complete"
    manifest = attach_live_collection(_manifest(), collection)
    require_complete_live_proof(manifest)
    assert attach_live_collection(manifest, collection) == manifest
    artifact = next(item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID)
    assert set(artifact.scenario_ids) == {str(item) for item in M5_LIVE_EVIDENCE_IDS}


def test_untrusted_module_entry_point_cannot_select_database_or_scope() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "leo.evals.live_proof"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert json.loads(completed.stdout) == {
        "code": "trusted_live_proof_composition_required",
        "status": "unavailable",
    }


def test_live_proof_operator_uses_a_database_compatible_event_loop() -> None:
    async def answer() -> int:
        return 42

    assert _run_async(answer()) == 42


def test_postgres_collector_surface_is_select_only_and_has_no_slack_client() -> None:
    source = inspect.getsource(PostgresLiveProofSource)
    helper = inspect.getsource(sys.modules[PostgresLiveProofSource.__module__])
    for forbidden in (
        "session.add(",
        "session.delete(",
        "session.execute(",
        "session.flush(",
        "session.commit(",
        "insert(",
        "update(",
        "delete(",
        "slack_sdk",
        "slack_bolt",
    ):
        assert forbidden not in source
        assert forbidden not in helper
    assert source.count("select(") >= 10


def test_request_cannot_omit_an_m5_live_requirement() -> None:
    request = _request(complete=False)
    with pytest.raises(ValueError, match="exactly partition"):
        LiveProofRequest(
            bindings=request.bindings[:-1],
            pending_evidence_ids=request.pending_evidence_ids,
        )


def test_dm_union_is_distinct_and_group_dm_cannot_aggregate() -> None:
    request = _request(complete=True)
    by_id = {item.evidence_id: item for item in request.bindings}
    same_channel = by_id[LiveEvidenceId.MEMORY_RECALL]
    dm_union = by_id[LiveEvidenceId.DM_MEMBERSHIP_UNION]
    group_dm = by_id[LiveEvidenceId.GROUP_DM]
    assert same_channel.expected_conversation_kind == "ordinary_internal"
    assert same_channel.expected_context_conversation_ids == (same_channel.expected_destination_id,)
    assert dm_union.expected_conversation_kind == "dm"
    assert len(dm_union.expected_context_conversation_ids) == 2
    assert dm_union.expected_recall_source_conversation_id != dm_union.expected_destination_id
    assert group_dm.expected_context_conversation_ids == (group_dm.expected_destination_id,)

    with pytest.raises(ValueError, match="non-DM live proof context"):
        LiveProofBinding.model_validate(
            {
                **group_dm.model_dump(mode="json"),
                "expected_context_conversation_ids": ["C-FORGED", "G-DEMO"],
            }
        )
    with pytest.raises(ValueError, match="non-DM positive recall source"):
        LiveProofBinding.model_validate(
            {
                **dm_union.model_dump(mode="json"),
                "expected_recall_source_conversation_id": dm_union.expected_destination_id,
            }
        )


def test_external_collection_contract_names_dm_union_grounding_fields() -> None:
    by_id = {item.id: item for item in EXTERNAL_EVIDENCE_CONTRACTS}
    slack_fields = set(by_id["slack-primary-acceptance"].expected_artifact_fields)
    durable_fields = set(by_id["supabase-durable-reconciliation"].expected_artifact_fields)
    assert {
        "case_invariant_digest",
        "context_access_hash",
        "current_membership_digest",
        "delegated_evidence_digest",
        "evidence_id",
        "grounded_memory_claim_digest",
        "memory_recall_observation_digest",
    } <= slack_fields
    assert {
        "case_invariant_digest",
        "context_access_hash",
        "conversation_source_set_digest",
        "current_membership_count",
        "current_membership_digest",
        "delegated_child_count",
        "delegated_evidence_digest",
        "delegated_overlap_verified",
        "grounded_claim_count",
        "grounded_memory_claim_digest",
        "memory_mutation_digest",
        "memory_mutation_record_count",
        "memory_mutation_revision_count",
        "memory_mutation_source_count",
        "memory_recall_observation_digest",
        "memory_recall_source_digest",
        "observed_evidence_count",
    } <= durable_fields


def test_dm_union_requires_positive_channel_memory_and_grounded_claim() -> None:
    binding = next(
        item
        for item in _request(complete=True).bindings
        if item.evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION
    )
    run = RunRow(
        id=binding.run_id,
        task_id="task-dm-union",
        organization_id="demo-org",
        strategy_id="demo-strategy",
        status="completed",
        phase="verification",
        iteration=2,
        limits={},
        usage={},
        final_output="I remember the shared channel fact.",
        event_sequence=8,
        version=8,
    )
    observation = ObservationRow(
        id="obs-dm-union-memory",
        run_id=binding.run_id,
        organization_id="demo-org",
        strategy_id="demo-strategy",
        tool_call_id="call-memory-search",
        kind="memory.search",
        data={
            "items": [
                {
                    "kind": "inline",
                    "reference": "mem_reference",
                    "content": "The shared channel fact.",
                    "excerpt": None,
                    "handle": None,
                    "chunk_count": 0,
                    "source_conversation": "C-SHARED",
                    "lifecycle_status": "active",
                    "contested": False,
                }
            ],
            "query_hash": "b" * 64,
            "selected_count": 1,
            "cache_status": "miss",
            "policy_version": "memory-navigation-v1",
        },
        source={"provider": "leo_memory", "reference": "b" * 64},
        raw_hash="c" * 64,
        status="retrieved",
        quality="internal_context",
        schema_version="observation-v2",
        normalization_version="normalization-v1",
        rejection_code=None,
    )
    claim = ClaimRow(
        id="claim-dm-union-memory",
        run_id=binding.run_id,
        organization_id="demo-org",
        strategy_id="demo-strategy",
        kind="inference",
        statement="The recalled fact came from the shared channel.",
        observation_ids=[observation.id],
    )
    summary = _memory_recall_summary(binding, run, (observation,), (claim,))
    assert summary["memory_recall_verified"] is True
    assert summary["memory_recall_source_digest"] == _digest("C-SHARED")

    with pytest.raises(LiveProofIntegrityError, match="positive grounded channel-memory"):
        _memory_recall_summary(binding, run, (observation,), ())
    wrong_source = ObservationRow(
        **{
            key: value
            for key, value in observation.__dict__.items()
            if not key.startswith("_sa_") and key != "data"
        },
        data={
            **observation.data,
            "items": [
                {
                    **observation.data["items"][0],
                    "source_conversation": binding.expected_destination_id,
                }
            ],
        },
    )
    with pytest.raises(LiveProofIntegrityError, match="positive grounded channel-memory"):
        _memory_recall_summary(binding, run, (wrong_source,), (claim,))


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
