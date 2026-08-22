from __future__ import annotations

import hashlib
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from leo.evals.final_evidence import (
    LiveRestartEvidence,
    PostgresReliabilityEvidence,
    build_final_evidence_aggregate,
)
from leo.evals.live_proof import LIVE_PROOF_ARTIFACT_ID, LiveProofCollection
from leo.evals.proof import ProofManifest
from leo.evals.revised_live_acceptance import (
    AsyncRevisedLiveSource,
    DurableRevisedObservation,
    DurableRevisedRun,
    OutboxRecoveryCaseId,
    PostgresRevisedLiveSource,
    RevisedLiveAcceptanceRequest,
    RevisedLiveCaseId,
    RevisedLiveIntegrityError,
    RevisedLiveRunBinding,
    RuntimeHealthReadback,
    SlackRevisedReadback,
    collect_revised_live_acceptance,
    export_contract,
    make_outbox_recovery_evidence,
    make_outbox_recovery_probe,
    make_runtime_health_readback,
    make_slack_revised_readback_case,
)
from leo.evals.revised_live_acceptance import _digest as revised_digest
from leo.evals.revised_live_acceptance_operator import (
    collect_outbox_recovery_postgres_evidence,
)


class _FakeSource(AsyncRevisedLiveSource):
    def __init__(self, observation: DurableRevisedObservation) -> None:
        self.observation = observation

    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        bindings: tuple[RevisedLiveRunBinding, ...],
    ) -> DurableRevisedObservation:
        assert organization_id == "demo-org"
        assert team_id == "TDEMO"
        assert tuple(item.case_id for item in bindings) == tuple(sorted(RevisedLiveCaseId, key=str))
        return self.observation


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT_ROOT = _REPOSITORY_ROOT / "artifacts"


@pytest.mark.asyncio
async def test_revised_live_acceptance_requires_all_real_semantic_progressions(
    tmp_path: Path,
) -> None:
    request, readback, health, durable = _fixture()
    outbox = _outbox_evidence()

    artifact = await collect_revised_live_acceptance(
        _FakeSource(durable),
        request=request,
        slack_readback=readback,
        runtime_health=health,
        postgres_reliability_digest="a" * 64,
        outbox_recovery=outbox,
        live_restart_digest="b" * 64,
    )

    assert artifact.case_count == 6
    assert artifact.max_ingress_latency_ms == 500
    assert artifact.post_restart_case_id is RevisedLiveCaseId.FINNHUB_EARNINGS
    assert {marker for case in artifact.cases for marker in case.semantic_markers}.issuperset(
        {
            "conversational_safe_terminal",
            "elastic_short_clarification",
            "complete_thread_context",
            "dm_membership_projection",
            "memory_grounded",
            "tavily_search_discovery",
            "selected_public_fetch",
            "expanded_finnhub_earnings",
        }
    )
    destination = tmp_path / "revised-live.json"
    export_contract(artifact, destination)
    assert destination.read_bytes().endswith(b"\n")


@pytest.mark.asyncio
async def test_final_milestone_requires_and_accepts_exact_bound_revised_companion(
    tmp_path: Path,
) -> None:
    postgres_path = _ARTIFACT_ROOT / "m5-postgres-reliability-v1.json"
    restart_path = _ARTIFACT_ROOT / "m5-live-restart-v1.json"
    postgres = PostgresReliabilityEvidence.model_validate_json(postgres_path.read_bytes())
    restart = LiveRestartEvidence.model_validate_json(restart_path.read_bytes())
    request, readback, health, durable = _fixture(
        listener_started_at=restart.listener_started_at,
        listener_epoch_digest=restart.listener_epoch_digest,
    )
    artifact = await collect_revised_live_acceptance(
        _FakeSource(durable),
        request=request,
        slack_readback=readback,
        runtime_health=health,
        postgres_reliability_digest=postgres.digest,
        outbox_recovery=_outbox_evidence(alembic_head=postgres.alembic_head),
        live_restart_digest=restart.digest,
    )
    live_manifest = ProofManifest.model_validate_json(
        (_ARTIFACT_ROOT / "m5-live-proof-v2.json").read_bytes()
    )
    live_artifact = next(
        item for item in live_manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID
    )
    metadata = live_artifact.metadata
    collection = LiveProofCollection.model_validate(
        {
            "version": metadata["version"],
            "profile": metadata["profile"],
            "authority_digest": metadata["authority_digest"],
            "cases": metadata["case_summaries"],
            "pending_evidence_ids": metadata["pending_evidence_ids"],
            "status": metadata["status"],
            "digest": metadata["collection_digest"],
        }
    )
    dm_reference = next(
        item for item in collection.cases if str(item.evidence_id) == "dm_membership_union"
    )
    artifact_payload = artifact.model_dump(mode="json")
    artifact_payload.update(
        {
            "dm_root_reference_run_id": dm_reference.run_id,
            "dm_root_reference_request_ts": dm_reference.message_ts,
            "dm_root_reference_response_ts": dm_reference.slack_response_ts,
        }
    )
    artifact_payload["digest"] = revised_digest(
        {key: value for key, value in artifact_payload.items() if key != "digest"}
    )
    artifact = type(artifact).model_validate(artifact_payload)
    revised_path = tmp_path / "revised-live.json"
    export_contract(artifact, revised_path)

    aggregate = await build_final_evidence_aggregate(
        offline_report_path=_ARTIFACT_ROOT / "m5-frozen-offline-report.json",
        live_proof_path=_ARTIFACT_ROOT / "m5-live-proof-v2.json",
        topology_path=_ARTIFACT_ROOT / "m5-slack-topology-v1.json",
        live_restart_path=restart_path,
        postgres_path=postgres_path,
        revised_live_acceptance_path=revised_path,
    )

    assert aggregate.final_milestone_ready
    assert aggregate.pending_requirements == ()
    assert aggregate.revised_live_acceptance == artifact


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case_id", "mutation", "message"),
    (
        (
            RevisedLiveCaseId.TERMINAL_RECOVERY,
            {"final_output": "budget_exhausted"},
            "bare, unsafe, or not durable",
        ),
        (
            RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP,
            {"context_markers": ("slack-thread-complete:false",)},
            "not complete and uncompacted",
        ),
        (
            RevisedLiveCaseId.TAVILY_RESEARCH,
            {"tool_names": ("web.search_tavily",)},
            "search-fetch-verified",
        ),
    ),
)
async def test_revised_live_acceptance_rejects_bare_incomplete_or_shallow_cases(
    case_id: RevisedLiveCaseId,
    mutation: dict[str, object],
    message: str,
) -> None:
    request, readback, health, durable = _fixture()
    runs = tuple(
        item.model_copy(update=mutation) if item.case_id is case_id else item
        for item in durable.runs
    )
    broken = durable.model_copy(update={"runs": runs})

    with pytest.raises(RevisedLiveIntegrityError, match=message):
        await collect_revised_live_acceptance(
            _FakeSource(broken),
            request=request,
            slack_readback=readback,
            runtime_health=health,
            postgres_reliability_digest="a" * 64,
            outbox_recovery=_outbox_evidence(),
            live_restart_digest="b" * 64,
        )


def test_revised_request_and_pg_probes_fail_closed_on_missing_or_forged_inputs() -> None:
    request, _readback, _health, _durable = _fixture()
    with pytest.raises(ValidationError, match="at least 7 items"):
        RevisedLiveAcceptanceRequest(
            organization_id=request.organization_id,
            team_id=request.team_id,
            bindings=request.bindings[:-1],
            post_restart_case_id=request.post_restart_case_id,
        )

    pending = make_outbox_recovery_probe(
        case_id=OutboxRecoveryCaseId.PENDING_FINAL,
        initial_final_outbox_count=1,
        repair_created_count=0,
        before={"state": "pending"},
        after={"state": "delivered"},
    )
    payload = pending.model_dump(mode="json")
    payload["duplicate_delivery_count"] = 1
    with pytest.raises(ValidationError):
        type(pending).model_validate(payload)


def test_outbox_pg_operator_requires_exact_two_typed_probe_files(tmp_path: Path) -> None:
    expected = _outbox_evidence()
    for index, probe in enumerate(expected.probes):
        export_contract(
            probe,
            tmp_path / f"probe-{index}" / "outbox-recovery-probe.json",
        )
    observed = collect_outbox_recovery_postgres_evidence(
        pytest_artifact_root=tmp_path,
        alembic_head=expected.alembic_head,
    )
    assert observed == expected

    export_contract(
        expected.probes[0],
        tmp_path / "duplicate" / "outbox-recovery-probe.json",
    )
    with pytest.raises(ValueError, match="exactly two"):
        collect_outbox_recovery_postgres_evidence(
            pytest_artifact_root=tmp_path,
            alembic_head=expected.alembic_head,
        )


def test_revised_postgres_source_is_structurally_select_only() -> None:
    source = inspect.getsource(PostgresRevisedLiveSource)
    assert "select(" in source
    for forbidden in ("insert(", "update(", "delete(", ".commit(", ".add("):
        assert forbidden not in source


@pytest.mark.asyncio
async def test_slack_text_digest_must_match_durable_final_payload() -> None:
    request, readback, health, durable = _fixture()
    first = readback.cases[0].model_copy(update={"response_text_digest": "f" * 64})
    tampered = readback.model_copy(update={"cases": (first, *readback.cases[1:])})
    with pytest.raises(ValidationError, match="content-addressed"):
        await collect_revised_live_acceptance(
            _FakeSource(durable),
            request=request,
            slack_readback=tampered,
            runtime_health=health,
            postgres_reliability_digest="a" * 64,
            outbox_recovery=_outbox_evidence(),
            live_restart_digest="b" * 64,
        )


def _fixture(
    *,
    listener_started_at: datetime | None = None,
    listener_epoch_digest: str = "c" * 64,
) -> tuple[
    RevisedLiveAcceptanceRequest,
    SlackRevisedReadback,
    RuntimeHealthReadback,
    DurableRevisedObservation,
]:
    epoch = (
        1_787_395_000
        if listener_started_at is None
        else round(listener_started_at.timestamp()) + 60
    )
    ids = tuple(sorted(RevisedLiveCaseId, key=str))
    offsets = {
        RevisedLiveCaseId.TERMINAL_RECOVERY: 0,
        RevisedLiveCaseId.DM_MEMORY_ROOT: 100,
        RevisedLiveCaseId.DM_THREAD_FOLLOWUP: 200,
        RevisedLiveCaseId.FINNHUB_EARNINGS: 300,
        RevisedLiveCaseId.MPIM_CLARIFICATION: 400,
        RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP: 500,
        RevisedLiveCaseId.TAVILY_RESEARCH: 600,
    }
    channels = {
        RevisedLiveCaseId.TERMINAL_RECOVERY: "CTERM",
        RevisedLiveCaseId.MPIM_CLARIFICATION: "CMPIM",
        RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP: "CMPIM",
        RevisedLiveCaseId.DM_MEMORY_ROOT: "DDM",
        RevisedLiveCaseId.DM_THREAD_FOLLOWUP: "DDM",
        RevisedLiveCaseId.TAVILY_RESEARCH: "CTAVILY",
        RevisedLiveCaseId.FINNHUB_EARNINGS: "CFINNHUB",
    }
    roots = {case_id: _ts(epoch + offsets[case_id]) for case_id in ids}
    roots[RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP] = roots[RevisedLiveCaseId.MPIM_CLARIFICATION]
    roots[RevisedLiveCaseId.DM_THREAD_FOLLOWUP] = roots[RevisedLiveCaseId.DM_MEMORY_ROOT]
    requests = dict(roots)
    requests[RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP] = _ts(
        epoch + offsets[RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP]
    )
    requests[RevisedLiveCaseId.DM_THREAD_FOLLOWUP] = _ts(
        epoch + offsets[RevisedLiveCaseId.DM_THREAD_FOLLOWUP]
    )
    responses = {case_id: _ts(float(requests[case_id]) + 10) for case_id in ids}
    contexts = {
        case_id: (
            ("CONE", "DDM")
            if case_id in {RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP}
            else (channels[case_id],)
        )
        for case_id in ids
    }
    bindings = tuple(
        RevisedLiveRunBinding(
            case_id=case_id,
            run_id=f"run-{case_id}",
            channel_id=channels[case_id],
            request_message_ts=requests[case_id],
            thread_root_ts=roots[case_id],
            slack_response_ts=responses[case_id],
            expected_context_conversation_ids=tuple(sorted(contexts[case_id])),
        )
        for case_id in ids
    )
    request = RevisedLiveAcceptanceRequest(
        organization_id="demo-org",
        team_id="TDEMO",
        bindings=bindings,
        post_restart_case_id=RevisedLiveCaseId.FINNHUB_EARNINGS,
    )
    runs = tuple(
        _run(
            binding=next(item for item in bindings if item.case_id is case_id),
            parent_task_id=(
                f"task-{RevisedLiveCaseId.MPIM_CLARIFICATION}"
                if case_id is RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP
                else (
                    f"task-{RevisedLiveCaseId.DM_MEMORY_ROOT}"
                    if case_id is RevisedLiveCaseId.DM_THREAD_FOLLOWUP
                    else None
                )
            ),
        )
        for case_id in ids
    )
    readback_cases = []
    for binding, run in zip(bindings, runs, strict=True):
        timestamps = {binding.thread_root_ts, binding.request_message_ts, binding.slack_response_ts}
        if binding.case_id is RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP:
            root = next(
                item for item in bindings if item.case_id is RevisedLiveCaseId.MPIM_CLARIFICATION
            )
            timestamps.update((root.request_message_ts, root.slack_response_ts))
        if binding.case_id is RevisedLiveCaseId.DM_THREAD_FOLLOWUP:
            root = next(
                item for item in bindings if item.case_id is RevisedLiveCaseId.DM_MEMORY_ROOT
            )
            timestamps.update((root.request_message_ts, root.slack_response_ts))
        if binding.case_id is RevisedLiveCaseId.DM_MEMORY_ROOT:
            continue
        readback_cases.append(
            make_slack_revised_readback_case(
                case_id=binding.case_id,
                run_id=binding.run_id,
                channel_id=binding.channel_id,
                thread_root_ts=binding.thread_root_ts,
                slack_response_ts=binding.slack_response_ts,
                response_text_digest=_sha(run.final_payload),
                thread_message_timestamps=tuple(timestamps),
            )
        )
    delivered_at = max(item.final_delivered_at for item in runs)
    readback = SlackRevisedReadback(
        observed_at=delivered_at + timedelta(seconds=1),
        cases=tuple(readback_cases),
    )
    started = listener_started_at or datetime.fromtimestamp(epoch - 60, tz=UTC)
    health = make_runtime_health_readback(
        listener_epoch_digest=listener_epoch_digest,
        listener_started_at=started,
        listener_connected_at=started + timedelta(seconds=1),
        observed_at=delivered_at + timedelta(seconds=2),
        last_success_at=delivered_at,
        component_states={
            "database": "ok",
            "metadata": "ok",
            "membership": "ok",
            "model": "ok",
            "orchestration": "ok",
            "queue": "ok",
            "outbox": "ok",
            "last_success": "ok",
            "socket": "unknown_cross_process",
        },
    )
    durable = DurableRevisedObservation(
        observed_at=readback.observed_at + timedelta(seconds=1),
        runs=runs,
    )
    return request, readback, health, durable


def _run(*, binding: RevisedLiveRunBinding, parent_task_id: str | None) -> DurableRevisedRun:
    case_id = binding.case_id
    terminal = case_id is RevisedLiveCaseId.TERMINAL_RECOVERY
    completed_events = ("task_started", "verification_passed", "run_completed")
    payload = (
        "I hit this request's work limit before I could verify an answer. "
        "Please narrow the request, and I can continue safely."
    )
    output = "Verified conversational result."
    tool_names: tuple[str, ...] = ()
    context_markers: tuple[str, ...] = ()
    observations: tuple[dict[str, object], ...] = ()
    claims: tuple[dict[str, object], ...] = (
        {"id": "claim", "kind": "inference", "observation_ids": ()},
    )
    if terminal:
        output = ""
        events = ("task_started", "budget_exhausted")
        claims = ()
    else:
        events = completed_events
    if case_id is RevisedLiveCaseId.MPIM_CLARIFICATION:
        output = "Could you clarify which companies and what period you want compared?"
    if case_id in {
        RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP,
        RevisedLiveCaseId.DM_THREAD_FOLLOWUP,
    }:
        context_markers = (
            "slack-thread-compacted-count:0",
            "slack-thread-complete:true",
            "slack-thread-protected-count:3",
            "slack-thread-source:slack_replies_bot",
        )
    if case_id in {RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP}:
        observations = (
            {
                "id": "obs-memory",
                "kind": "memory.search",
                "status": "retrieved",
                "quality": "internal_context",
                "provider": "leo_memory",
            },
        )
        claims = ({"id": "claim", "kind": "source", "observation_ids": ("obs-memory",)},)
    if case_id is RevisedLiveCaseId.TAVILY_RESEARCH:
        tool_names = ("web.search_tavily", "web.fetch_public_text")
        observations = (
            {
                "id": "obs-search",
                "kind": "web.search_tavily",
                "status": "rejected",
                "quality": "discovery_only",
                "provider": "tavily",
            },
            {
                "id": "obs-fetch",
                "kind": "web.fetch_public_text",
                "status": "retrieved",
                "quality": "primary_source",
                "provider": "web",
            },
        )
        claims = ({"id": "claim", "kind": "source", "observation_ids": ("obs-fetch",)},)
    if case_id is RevisedLiveCaseId.FINNHUB_EARNINGS:
        tool_names = ("market.get_earnings_surprises",)
        observations = (
            {
                "id": "obs-earnings",
                "kind": "market.get_earnings_surprises",
                "status": "retrieved",
                "quality": "provider_reported",
                "provider": "finnhub",
            },
        )
        claims = ({"id": "claim", "kind": "source", "observation_ids": ("obs-earnings",)},)
    received_at = datetime.fromtimestamp(float(binding.request_message_ts) + 0.5, tz=UTC)
    persisted_thread_timestamps = {binding.thread_root_ts, binding.request_message_ts}
    if parent_task_id is not None:
        persisted_thread_timestamps.add(_ts(float(binding.thread_root_ts) + 10))
    return DurableRevisedRun(
        case_id=case_id,
        run_id=binding.run_id,
        task_id=f"task-{case_id}",
        parent_task_id=parent_task_id,
        continuation_kind="follow_up" if parent_task_id else "root",
        channel_id=binding.channel_id,
        request_message_ts=binding.request_message_ts,
        thread_root_ts=binding.thread_root_ts,
        received_at=received_at,
        conversation_kind=(
            "dm"
            if case_id in {RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP}
            else (
                "mpim"
                if case_id
                in {
                    RevisedLiveCaseId.MPIM_CLARIFICATION,
                    RevisedLiveCaseId.MPIM_THREAD_FOLLOWUP,
                }
                else "ordinary_internal"
            )
        ),
        context_projection_source=(
            "dm_membership_intersection"
            if case_id in {RevisedLiveCaseId.DM_MEMORY_ROOT, RevisedLiveCaseId.DM_THREAD_FOLLOWUP}
            else "exact_destination"
        ),
        context_conversation_ids=binding.expected_context_conversation_ids,
        context_access_hash="d" * 64,
        prompt="Can you help me answer this naturally?",
        task_status="failed" if terminal else "completed",
        run_status="budget_exhausted" if terminal else "completed",
        terminal_reason="iteration_budget_exhausted" if terminal else "verified_completion",
        final_output=output,
        final_payload=payload,
        final_payload_hash=_sha(payload),
        final_state="delivered",
        final_attempt_count=1,
        final_receipt_message_ts=binding.slack_response_ts,
        final_delivered_at=received_at + timedelta(seconds=9),
        event_types=events,
        tool_names=tool_names,
        context_markers=context_markers,
        context_projection_commitments=(),
        context_projection_source_counts=(),
        persisted_thread_message_timestamps=tuple(sorted(persisted_thread_timestamps, key=float)),
        observations=observations,
        claims=claims,
    )


def _outbox_evidence(*, alembic_head: str = "20260822_0026"):
    pending = make_outbox_recovery_probe(
        case_id=OutboxRecoveryCaseId.PENDING_FINAL,
        initial_final_outbox_count=1,
        repair_created_count=0,
        before={"state": "pending"},
        after={"state": "delivered"},
    )
    missing = make_outbox_recovery_probe(
        case_id=OutboxRecoveryCaseId.MISSING_FINAL,
        initial_final_outbox_count=0,
        repair_created_count=1,
        before={"state": "absent"},
        after={"state": "delivered"},
    )
    return make_outbox_recovery_evidence(
        alembic_head=alembic_head,
        probes=(pending, missing),
    )


def _ts(value: float | int) -> str:
    return f"{float(value):.6f}"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
