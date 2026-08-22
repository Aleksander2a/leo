from __future__ import annotations

import hashlib
import inspect
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

import leo.evals.live_restart_operator as restart_operator
from leo.evals.final_evidence import LiveRestartEvidence
from leo.evals.live_proof import LiveProofCase, LiveProofCollection
from leo.evals.live_restart_operator import (
    AsyncLiveRestartSource,
    DurableRestartCaseObservation,
    DurableRestartObservation,
    LiveRestartIntegrityError,
    PostgresLiveRestartSource,
    SlackRestartReadback,
    SlackRestartReadbackCase,
    collect_live_restart_evidence,
    export_live_restart_evidence,
    load_complete_live_collection,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LIVE_PROOF = REPOSITORY_ROOT / "artifacts" / "m5-live-proof-v2.json"


class _FakeSource(AsyncLiveRestartSource):
    def __init__(self, observation: DurableRestartObservation) -> None:
        self.observation = observation
        self.calls: list[tuple[str, str, tuple[LiveProofCase, ...]]] = []

    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        cases: tuple[LiveProofCase, ...],
    ) -> DurableRestartObservation:
        self.calls.append((organization_id, team_id, cases))
        return self.observation


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _epoch(collection: LiveProofCollection) -> datetime:
    return datetime.fromtimestamp(
        max(float(item.slack_response_ts) for item in collection.cases) + 60,
        tz=UTC,
    )


def _durable_case(
    case: LiveProofCase,
    *,
    listener_started_at: datetime,
) -> DurableRestartCaseObservation:
    marker = str(case.evidence_id)
    return DurableRestartCaseObservation(
        evidence_id=case.evidence_id,
        run_id=case.run_id,
        task_id=case.task_id,
        channel_id=case.channel_id,
        task_status="completed",
        task_version=7,
        run_status="completed",
        run_phase="terminal",
        run_iteration=2,
        run_event_sequence=case.last_event_sequence,
        run_version=11,
        terminal_reason="verified_completion",
        final_output_digest=case.final_output_digest,
        task_final_output_digest=case.final_output_digest,
        task_run_digest=case.task_run_digest,
        outbox_count=case.outbox_count,
        outbox_digest=case.outbox_digest,
        final_outbox_count=1,
        final_outbox_row_digest=_digest(f"final-row:{marker}"),
        final_state="delivered",
        final_delivery_attempt_count=1,
        final_receipt_count=1,
        final_receipt_message_ts=case.slack_response_ts,
        final_delivered_at=listener_started_at - timedelta(minutes=1),
    )


def _durable_observation(
    collection: LiveProofCollection,
    *,
    listener_started_at: datetime,
) -> DurableRestartObservation:
    return DurableRestartObservation(
        observed_at=listener_started_at + timedelta(minutes=2),
        cases=tuple(
            _durable_case(item, listener_started_at=listener_started_at)
            for item in collection.cases
        ),
    )


def _readback(
    collection: LiveProofCollection,
    *,
    listener_started_at: datetime,
    listener_epoch_digest: str,
) -> SlackRestartReadback:
    return SlackRestartReadback(
        listener_epoch_digest=listener_epoch_digest,
        live_collection_digest=collection.digest,
        observed_at=listener_started_at + timedelta(minutes=1),
        cases=tuple(
            SlackRestartReadbackCase(
                evidence_id=item.evidence_id,
                run_id=item.run_id,
                slack_response_ts=item.slack_response_ts,
                matching_message_count=1,
                readback_digest=_digest(f"slack-readback:{item.evidence_id}"),
            )
            for item in collection.cases
        ),
    )


@pytest.mark.asyncio
async def test_exact_nine_restart_observations_export_existing_contract(tmp_path: Path) -> None:
    collection = load_complete_live_collection(LIVE_PROOF)
    listener_started_at = _epoch(collection)
    epoch_digest = _digest("trusted-listener-epoch")
    source = _FakeSource(_durable_observation(collection, listener_started_at=listener_started_at))

    evidence = await collect_live_restart_evidence(
        source,
        organization_id="demo-org",
        team_id="T-DEMO",
        collection=collection,
        listener_epoch_digest=epoch_digest,
        listener_started_at=listener_started_at,
        readback=_readback(
            collection,
            listener_started_at=listener_started_at,
            listener_epoch_digest=epoch_digest,
        ),
    )
    destination = tmp_path / "nested" / "m5-live-restart-v1.json"
    export_live_restart_evidence(evidence, destination)

    assert evidence.case_count == 9
    assert {item.evidence_id for item in evidence.cases} == {
        item.evidence_id for item in collection.cases
    }
    assert source.calls == [("demo-org", "T-DEMO", collection.cases)]
    raw = destination.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\\n")
    assert LiveRestartEvidence.model_validate_json(raw) == evidence


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("final_delivery_attempt_count", 2, "delivery snapshot diverged"),
        ("final_outbox_count", 2, "delivery snapshot diverged"),
        ("outbox_digest", "f" * 64, "delivery snapshot diverged"),
        ("task_run_digest", "e" * 64, "terminal state diverged"),
    ),
)
async def test_mutated_durable_snapshot_is_rejected(
    field: str,
    value: object,
    message: str,
) -> None:
    collection = load_complete_live_collection(LIVE_PROOF)
    listener_started_at = _epoch(collection)
    epoch_digest = _digest("trusted-listener-epoch")
    observed = _durable_observation(collection, listener_started_at=listener_started_at)
    first = observed.cases[0].model_copy(update={field: value})
    source = _FakeSource(observed.model_copy(update={"cases": (first, *observed.cases[1:])}))

    with pytest.raises((LiveRestartIntegrityError, ValidationError), match=message):
        await collect_live_restart_evidence(
            source,
            organization_id="demo-org",
            team_id="T-DEMO",
            collection=collection,
            listener_epoch_digest=epoch_digest,
            listener_started_at=listener_started_at,
            readback=_readback(
                collection,
                listener_started_at=listener_started_at,
                listener_epoch_digest=epoch_digest,
            ),
        )


@pytest.mark.asyncio
async def test_readback_must_cover_one_exact_message_for_every_case() -> None:
    collection = load_complete_live_collection(LIVE_PROOF)
    listener_started_at = _epoch(collection)
    epoch_digest = _digest("trusted-listener-epoch")
    readback = _readback(
        collection,
        listener_started_at=listener_started_at,
        listener_epoch_digest=epoch_digest,
    )
    wrong = readback.cases[0].model_copy(update={"matching_message_count": 2})
    readback = readback.model_copy(update={"cases": (wrong, *readback.cases[1:])})

    with pytest.raises(LiveRestartIntegrityError, match="one exact receipt"):
        await collect_live_restart_evidence(
            _FakeSource(_durable_observation(collection, listener_started_at=listener_started_at)),
            organization_id="demo-org",
            team_id="T-DEMO",
            collection=collection,
            listener_epoch_digest=epoch_digest,
            listener_started_at=listener_started_at,
            readback=readback,
        )


@pytest.mark.asyncio
async def test_readback_and_database_observations_must_be_strictly_post_epoch() -> None:
    collection = load_complete_live_collection(LIVE_PROOF)
    listener_started_at = _epoch(collection)
    epoch_digest = _digest("trusted-listener-epoch")
    readback = _readback(
        collection,
        listener_started_at=listener_started_at,
        listener_epoch_digest=epoch_digest,
    ).model_copy(update={"observed_at": listener_started_at})

    with pytest.raises(LiveRestartIntegrityError, match="not bound after"):
        await collect_live_restart_evidence(
            _FakeSource(_durable_observation(collection, listener_started_at=listener_started_at)),
            organization_id="demo-org",
            team_id="T-DEMO",
            collection=collection,
            listener_epoch_digest=epoch_digest,
            listener_started_at=listener_started_at,
            readback=readback,
        )


def test_postgres_source_contains_no_mutating_statement_path() -> None:
    source = inspect.getsource(PostgresLiveRestartSource.observe).casefold()

    assert "select(" in source
    assert "insert(" not in source
    assert "update(" not in source
    assert "delete(" not in source
    assert ".commit(" not in source
    assert ".begin(" not in source


def test_cli_hides_database_error_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "postgresql://user:super-secret@example.invalid/database"

    async def fail(_arguments: object) -> int:
        raise RuntimeError(secret)

    monkeypatch.setattr(restart_operator, "_run", fail)
    result = restart_operator.main(
        [
            "--live-proof",
            "unused.json",
            "--listener-started-at",
            "2026-08-22T12:00:00+00:00",
            "--listener-epoch-digest",
            "a" * 64,
            "--slack-readback",
            "unused-readback.json",
            "--output",
            "unused-output.json",
        ]
    )

    output = capsys.readouterr().out
    assert result == 2
    assert "live_restart_collection_failed" in output
    assert "super-secret" not in output
