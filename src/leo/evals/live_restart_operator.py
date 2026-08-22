"""Trusted SELECT-only collector for live listener restart/no-redelivery evidence.

The operator deliberately has no Slack client.  A trusted operator supplies a
content-free readback produced from connector reads after the listener epoch;
this module independently re-observes the exact live-proof cohort in Postgres.
Only a cohort whose durable terminal and outbox snapshots are unchanged is
exported through the existing :mod:`leo.evals.final_evidence` contracts.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import selectors
import sys
import tempfile
from collections.abc import Coroutine, Sequence
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

from pydantic import Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.evals.final_evidence import (
    LiveRestartEvidence,
    make_live_restart_case,
    make_live_restart_evidence,
)
from leo.evals.live_proof import (
    LIVE_PROOF_ARTIFACT_ID,
    M5_LIVE_EVIDENCE_IDS,
    LiveEvidenceId,
    LiveProofCase,
    LiveProofCollection,
    SlackMessageTs,
    require_complete_live_proof,
)
from leo.evals.proof import ProofManifest, validate_proof_manifest
from leo.harness.models import ContractModel, NonEmptyStr
from leo.persistence.database import create_database_engine, create_session_factory
from leo.persistence.schema import DeliveryOutboxRow, RunRow, SlackIngressEventRow, TaskRow

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SLACK_RESTART_READBACK_VERSION: Literal["slack-live-restart-readback-v1"] = (
    "slack-live-restart-readback-v1"
)


class LiveRestartNotFound(LookupError):
    """An exact pre-proof durable row was not available in the trusted scope."""


class LiveRestartIntegrityError(ValueError):
    """Post-restart observations diverged from the completed pre-restart proof."""


class SlackRestartReadbackCase(ContractModel):
    """Content-free result of reading one Slack thread after the listener restart."""

    evidence_id: LiveEvidenceId
    run_id: NonEmptyStr
    slack_response_ts: SlackMessageTs
    matching_message_count: int = Field(ge=0)
    readback_digest: Sha256


class SlackRestartReadback(ContractModel):
    """Trusted connector readback, explicitly bound to the epoch and live cohort."""

    version: Literal["slack-live-restart-readback-v1"] = SLACK_RESTART_READBACK_VERSION
    listener_epoch_digest: Sha256
    live_collection_digest: Sha256
    observed_at: datetime
    cases: tuple[SlackRestartReadbackCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_order_and_time(self) -> SlackRestartReadback:
        ids = tuple(item.evidence_id for item in self.cases)
        if self.observed_at.tzinfo is None:
            raise ValueError("Slack restart readback timestamp must be timezone-aware")
        if ids != tuple(sorted(ids, key=str)) or len(ids) != len(set(ids)):
            raise ValueError("Slack restart readback cases must be sorted and unique")
        if len({item.readback_digest for item in self.cases}) != len(self.cases):
            raise ValueError("Slack restart readback digests must be unique")
        return self


class DurableRestartCaseObservation(ContractModel):
    """Content-free terminal/outbox projection read from Postgres after restart."""

    evidence_id: LiveEvidenceId
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    channel_id: NonEmptyStr
    task_status: NonEmptyStr
    task_version: int = Field(ge=0)
    run_status: NonEmptyStr
    run_phase: NonEmptyStr
    run_iteration: int = Field(ge=0)
    run_event_sequence: int = Field(ge=0)
    run_version: int = Field(ge=0)
    terminal_reason: str | None
    final_output_digest: Sha256
    task_final_output_digest: Sha256
    task_run_digest: Sha256
    outbox_count: int = Field(ge=0)
    outbox_digest: Sha256
    final_outbox_count: int = Field(ge=0)
    final_outbox_row_digest: Sha256
    final_state: NonEmptyStr
    final_delivery_attempt_count: int = Field(ge=0)
    final_receipt_count: int = Field(ge=0)
    final_receipt_message_ts: SlackMessageTs | None
    final_delivered_at: datetime

    @model_validator(mode="after")
    def exact_terminal_projection(self) -> DurableRestartCaseObservation:
        if self.final_delivered_at.tzinfo is None:
            raise ValueError("durable restart delivery timestamp must be timezone-aware")
        if self.task_final_output_digest != self.final_output_digest:
            raise ValueError("durable restart task/run outputs diverged")
        return self


class DurableRestartObservation(ContractModel):
    """One SELECT-only database observation of the entire exact live cohort."""

    observed_at: datetime
    cases: tuple[DurableRestartCaseObservation, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_order_and_time(self) -> DurableRestartObservation:
        ids = tuple(item.evidence_id for item in self.cases)
        if self.observed_at.tzinfo is None:
            raise ValueError("durable restart observation timestamp must be timezone-aware")
        if ids != tuple(sorted(ids, key=str)) or len(ids) != len(set(ids)):
            raise ValueError("durable restart observations must be sorted and unique")
        return self


class AsyncLiveRestartSource(Protocol):
    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        cases: tuple[LiveProofCase, ...],
    ) -> DurableRestartObservation: ...


class PostgresLiveRestartSource:
    """SELECT-only durable source; session close rolls back its implicit transaction."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def observe(
        self,
        *,
        organization_id: str,
        team_id: str,
        cases: tuple[LiveProofCase, ...],
    ) -> DurableRestartObservation:
        observations: list[DurableRestartCaseObservation] = []
        async with self._sessions() as session:
            for case in cases:
                ingress = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(SlackIngressEventRow).where(
                                    SlackIngressEventRow.organization_id == organization_id,
                                    SlackIngressEventRow.team_id == team_id,
                                    SlackIngressEventRow.message_ts == case.message_ts,
                                    SlackIngressEventRow.task_id == case.task_id,
                                    SlackIngressEventRow.channel_id == case.channel_id,
                                )
                            )
                        ).all()
                    )
                )
                task = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(TaskRow).where(
                                    TaskRow.id == case.task_id,
                                    TaskRow.organization_id == organization_id,
                                )
                            )
                        ).all()
                    )
                )
                run = _require_one(
                    tuple(
                        (
                            await session.scalars(
                                select(RunRow).where(
                                    RunRow.id == case.run_id,
                                    RunRow.task_id == case.task_id,
                                    RunRow.organization_id == organization_id,
                                )
                            )
                        ).all()
                    )
                )
                outbox_rows = tuple(
                    (
                        await session.scalars(
                            select(DeliveryOutboxRow)
                            .where(
                                DeliveryOutboxRow.ingress_event_id == ingress.event_id,
                                DeliveryOutboxRow.task_id == case.task_id,
                                DeliveryOutboxRow.run_id == case.run_id,
                                DeliveryOutboxRow.organization_id == organization_id,
                            )
                            .order_by(
                                DeliveryOutboxRow.kind,
                                DeliveryOutboxRow.payload_version,
                            )
                        )
                    ).all()
                )
                final_rows = tuple(item for item in outbox_rows if item.kind == "final")
                final = _require_one(final_rows)
                observations.append(
                    _durable_case(
                        case=case,
                        task=task,
                        run=run,
                        outbox_rows=outbox_rows,
                        final=final,
                    )
                )
            observed_at = await session.scalar(select(func.now()))
        if observed_at is None:
            raise LiveRestartNotFound
        return DurableRestartObservation(
            observed_at=observed_at,
            cases=tuple(observations),
        )


async def collect_live_restart_evidence(
    source: AsyncLiveRestartSource,
    *,
    organization_id: str,
    team_id: str,
    collection: LiveProofCollection,
    listener_epoch_digest: str,
    listener_started_at: datetime,
    readback: SlackRestartReadback,
) -> LiveRestartEvidence:
    """Reconcile an exact before/after epoch or fail without exporting evidence."""

    _require_exact_complete_cohort(collection)
    readback = SlackRestartReadback.model_validate(readback.model_dump(mode="json"))
    if listener_started_at.tzinfo is None:
        raise LiveRestartIntegrityError("listener epoch timestamp must be timezone-aware")
    if (
        readback.listener_epoch_digest != listener_epoch_digest
        or readback.live_collection_digest != collection.digest
        or readback.observed_at <= listener_started_at
    ):
        raise LiveRestartIntegrityError("Slack readback is not bound after the trusted epoch")

    # Revalidate serialized output so a fake/custom source cannot bypass model
    # constraints through ``model_construct``.
    durable_raw = await source.observe(
        organization_id=organization_id,
        team_id=team_id,
        cases=collection.cases,
    )
    durable = DurableRestartObservation.model_validate(durable_raw.model_dump(mode="json"))
    if durable.observed_at <= listener_started_at or readback.observed_at > durable.observed_at:
        raise LiveRestartIntegrityError("restart observations lack a strict epoch ordering")

    expected = {item.evidence_id: item for item in collection.cases}
    database_cases = {item.evidence_id: item for item in durable.cases}
    slack_cases = {item.evidence_id: item for item in readback.cases}
    if set(expected) != set(database_cases) or set(expected) != set(slack_cases):
        raise LiveRestartIntegrityError("restart evidence does not cover the exact live cohort")

    restart_cases = []
    for evidence_id in sorted(expected, key=str):
        before = expected[evidence_id]
        after = database_cases[evidence_id]
        slack = slack_cases[evidence_id]
        if (
            after.run_id != before.run_id
            or after.task_id != before.task_id
            or after.channel_id != before.channel_id
            or after.task_status != before.task_terminal_state
            or after.run_status != before.run_terminal_state
            or after.terminal_reason != "verified_completion"
            or after.final_output_digest != before.final_output_digest
            or after.task_final_output_digest != before.final_output_digest
            or after.task_run_digest != before.task_run_digest
        ):
            raise LiveRestartIntegrityError("post-restart terminal state diverged")
        if (
            after.outbox_count != before.outbox_count
            or after.outbox_digest != before.outbox_digest
            or after.final_outbox_count != 1
            or after.final_state != "delivered"
            or after.final_delivery_attempt_count != 1
            or after.final_receipt_count != 1
            or after.final_receipt_message_ts != before.slack_response_ts
            or after.final_delivered_at >= listener_started_at
        ):
            raise LiveRestartIntegrityError("post-restart delivery snapshot diverged")
        if (
            slack.run_id != before.run_id
            or slack.slack_response_ts != before.slack_response_ts
            or slack.matching_message_count != 1
        ):
            raise LiveRestartIntegrityError("Slack readback does not prove one exact receipt")
        restart_cases.append(
            make_live_restart_case(
                evidence_id=evidence_id,
                run_id=before.run_id,
                slack_response_ts=before.slack_response_ts,
                final_outbox_row_digest=after.final_outbox_row_digest,
                final_delivered_at=after.final_delivered_at,
                post_restart_slack_readback_digest=slack.readback_digest,
            )
        )

    return make_live_restart_evidence(
        listener_epoch_digest=listener_epoch_digest,
        listener_started_at=listener_started_at,
        collection_observed_at=durable.observed_at,
        cases=tuple(restart_cases),
    )


def load_complete_live_collection(path: Path) -> LiveProofCollection:
    """Load and independently validate one completed exact-nine live artifact."""

    manifest = ProofManifest.model_validate_json(path.read_bytes())
    validate_proof_manifest(manifest)
    require_complete_live_proof(manifest)
    artifacts = tuple(item for item in manifest.artifacts if item.id == LIVE_PROOF_ARTIFACT_ID)
    if len(artifacts) != 1:
        raise LiveRestartIntegrityError("completed proof lacks one strict live artifact")
    metadata = artifacts[0].metadata
    collection = LiveProofCollection.model_validate(
        {
            "version": metadata.get("version"),
            "profile": metadata.get("profile"),
            "authority_digest": metadata.get("authority_digest"),
            "cases": metadata.get("case_summaries"),
            "pending_evidence_ids": metadata.get("pending_evidence_ids"),
            "status": metadata.get("status"),
            "digest": metadata.get("collection_digest"),
        }
    )
    _require_exact_complete_cohort(collection)
    return collection


def export_live_restart_evidence(evidence: LiveRestartEvidence, destination: Path) -> None:
    """Atomically write only a fully validated live-restart artifact."""

    validated = LiveRestartEvidence.model_validate(evidence.model_dump(mode="json"))
    _atomic_write(destination, validated.model_dump_json(indent=2) + "\n")


def _durable_case(
    *,
    case: LiveProofCase,
    task: TaskRow,
    run: RunRow,
    outbox_rows: tuple[DeliveryOutboxRow, ...],
    final: DeliveryOutboxRow,
) -> DurableRestartCaseObservation:
    task_output_digest = _digest_text(task.final_output or "")
    run_output_digest = _digest_text(run.final_output or "")
    task_run_payload = {
        "task_id": task.id,
        "task_status": task.status,
        "task_version": task.version,
        "run_id": run.id,
        "run_status": run.status,
        "run_phase": run.phase,
        "run_iteration": run.iteration,
        "run_event_sequence": run.event_sequence,
        "run_version": run.version,
        "terminal_reason": run.terminal_reason,
        "final_output_digest": run_output_digest,
    }
    outbox_projection = [
        {
            "id": item.id,
            "kind": item.kind,
            "payload_version": item.payload_version,
            "payload_hash": item.payload_hash,
            "state": item.state,
            "attempt_count": item.attempt_count,
            "receipt_message_ts": item.receipt_message_ts,
            "destination_channel_id": item.destination_channel_id,
            "destination_thread_ts": item.destination_thread_ts,
        }
        for item in outbox_rows
    ]
    final_projection = {
        "id": final.id,
        "kind": final.kind,
        "payload_version": final.payload_version,
        "payload_hash": final.payload_hash,
        "state": final.state,
        "attempt_count": final.attempt_count,
        "receipt_message_ts": final.receipt_message_ts,
        "destination_channel_id": final.destination_channel_id,
        "destination_thread_ts": final.destination_thread_ts,
        "updated_at": final.updated_at,
    }
    return DurableRestartCaseObservation(
        evidence_id=case.evidence_id,
        run_id=run.id,
        task_id=task.id,
        channel_id=case.channel_id,
        task_status=task.status,
        task_version=task.version,
        run_status=run.status,
        run_phase=run.phase,
        run_iteration=run.iteration,
        run_event_sequence=run.event_sequence,
        run_version=run.version,
        terminal_reason=run.terminal_reason,
        final_output_digest=run_output_digest,
        task_final_output_digest=task_output_digest,
        task_run_digest=_digest(task_run_payload),
        outbox_count=len(outbox_rows),
        outbox_digest=_digest(outbox_projection),
        final_outbox_count=sum(item.kind == "final" for item in outbox_rows),
        final_outbox_row_digest=_digest(final_projection),
        final_state=final.state,
        final_delivery_attempt_count=final.attempt_count,
        final_receipt_count=sum(
            item.kind == "final" and item.receipt_message_ts is not None for item in outbox_rows
        ),
        final_receipt_message_ts=final.receipt_message_ts,
        final_delivered_at=final.updated_at,
    )


def _require_exact_complete_cohort(collection: LiveProofCollection) -> None:
    ids = tuple(item.evidence_id for item in collection.cases)
    if (
        collection.status != "complete"
        or collection.pending_evidence_ids
        or len(collection.cases) != len(M5_LIVE_EVIDENCE_IDS)
        or set(ids) != set(M5_LIVE_EVIDENCE_IDS)
    ):
        raise LiveRestartIntegrityError("restart operator requires the complete exact-nine cohort")


def _require_one[T](items: Sequence[T]) -> T:
    if len(items) != 1:
        raise LiveRestartNotFound
    return items[0]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"unsupported restart proof value: {type(value).__name__}")


async def _run(arguments: argparse.Namespace) -> int:
    settings = Settings()
    if settings.database_url is None or settings.leo_slack_team_id is None:
        raise RuntimeError("live_restart_configuration_missing")
    collection = load_complete_live_collection(arguments.live_proof)
    readback = SlackRestartReadback.model_validate_json(arguments.slack_readback.read_bytes())
    listener_started_at = _parse_datetime(arguments.listener_started_at)
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        evidence = await collect_live_restart_evidence(
            PostgresLiveRestartSource(create_session_factory(engine)),
            organization_id=settings.leo_organization_id,
            team_id=settings.leo_slack_team_id,
            collection=collection,
            listener_epoch_digest=arguments.listener_epoch_digest,
            listener_started_at=listener_started_at,
            readback=readback,
        )
    finally:
        await engine.dispose()
    export_live_restart_evidence(evidence, arguments.output)
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "case_count": evidence.case_count,
                "digest": evidence.digest,
                "status": "complete",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-live-restart-proof",
        description=(
            "Reconcile the exact completed M5 cohort after a trusted listener epoch using "
            "SELECT-only durable reads and content-free Slack connector readbacks."
        ),
    )
    parser.add_argument("--live-proof", required=True, type=Path)
    parser.add_argument("--listener-started-at", required=True)
    parser.add_argument("--listener-epoch-digest", required=True)
    parser.add_argument("--slack-readback", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    if not _is_sha256(arguments.listener_epoch_digest):
        parser.error("--listener-epoch-digest must be a lowercase SHA-256 value")
    return arguments


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("listener-started-at must be timezone-aware")
    return parsed


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        return _run_async(_run(arguments))
    except LiveRestartNotFound:
        code = "live_restart_trace_not_found"
    except Exception:
        # DB/client exceptions can contain connection details; never echo them.
        code = "live_restart_collection_failed"
    print(json.dumps({"code": code, "status": "failed"}, sort_keys=True, separators=(",", ":")))
    return 2


def _run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    """Use a Psycopg-compatible selector loop for trusted Windows DB reads."""

    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        ) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
