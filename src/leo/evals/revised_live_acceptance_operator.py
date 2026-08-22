"""Trusted SELECT-only operator for revised D-063--D-066 live acceptance."""

from __future__ import annotations

import argparse
import asyncio
import json
import selectors
import sys
from collections.abc import Coroutine, Sequence
from pathlib import Path
from typing import Any

from leo.config import Settings
from leo.evals.final_evidence import (
    LiveRestartEvidence,
    PostgresReliabilityEvidence,
    repository_alembic_head,
)
from leo.evals.revised_live_acceptance import (
    AsyncRevisedLiveSource,
    OutboxRecoveryPostgresEvidence,
    OutboxRecoveryProbe,
    PostgresRevisedLiveSource,
    RevisedLiveAcceptanceArtifact,
    RevisedLiveAcceptanceRequest,
    RevisedLiveNotFound,
    RuntimeHealthReadback,
    SlackRevisedReadback,
    collect_revised_live_acceptance,
    export_contract,
    make_outbox_recovery_evidence,
)
from leo.persistence.database import create_database_engine, create_session_factory

_OUTBOX_PROBE_NAME = "outbox-recovery-probe.json"


def collect_outbox_recovery_postgres_evidence(
    *,
    pytest_artifact_root: Path,
    alembic_head: str,
) -> OutboxRecoveryPostgresEvidence:
    """Load exactly two typed rollback-preserved probes from one pytest run."""

    if not pytest_artifact_root.is_dir():
        raise ValueError("Postgres pytest artifact root is absent")
    paths = tuple(
        sorted(item for item in pytest_artifact_root.rglob(_OUTBOX_PROBE_NAME) if item.is_file())
    )
    if len(paths) != 2:
        raise ValueError("expected exactly two observed outbox recovery probes")
    probes = tuple(OutboxRecoveryProbe.model_validate_json(path.read_bytes()) for path in paths)
    return make_outbox_recovery_evidence(alembic_head=alembic_head, probes=probes)


async def collect_operator_artifact(
    *,
    source: AsyncRevisedLiveSource,
    request: RevisedLiveAcceptanceRequest,
    slack_readback: SlackRevisedReadback,
    runtime_health: RuntimeHealthReadback,
    postgres: PostgresReliabilityEvidence,
    outbox_recovery: OutboxRecoveryPostgresEvidence,
    live_restart: LiveRestartEvidence,
) -> RevisedLiveAcceptanceArtifact:
    """Bind independently validated component artifacts to exact live SELECTs."""

    head = repository_alembic_head()
    if (
        postgres.alembic_head != head
        or outbox_recovery.alembic_head != head
        or live_restart.case_count != 9
        or runtime_health.listener_epoch_digest != live_restart.listener_epoch_digest
        or runtime_health.listener_started_at != live_restart.listener_started_at
    ):
        raise ValueError("revised acceptance components do not share the current trusted epoch")
    return await collect_revised_live_acceptance(
        source,
        request=request,
        slack_readback=slack_readback,
        runtime_health=runtime_health,
        postgres_reliability_digest=postgres.digest,
        outbox_recovery=outbox_recovery,
        live_restart_digest=live_restart.digest,
    )


async def _run(arguments: argparse.Namespace) -> RevisedLiveAcceptanceArtifact:
    settings = Settings()
    if settings.database_url is None:
        raise RuntimeError("revised_live_database_configuration_missing")
    request = RevisedLiveAcceptanceRequest.model_validate_json(arguments.request.read_bytes())
    slack_readback = SlackRevisedReadback.model_validate_json(arguments.slack_readback.read_bytes())
    runtime_health = RuntimeHealthReadback.model_validate_json(
        arguments.runtime_health_readback.read_bytes()
    )
    postgres = PostgresReliabilityEvidence.model_validate_json(
        arguments.postgres_artifact.read_bytes()
    )
    live_restart = LiveRestartEvidence.model_validate_json(
        arguments.live_restart_artifact.read_bytes()
    )
    outbox_recovery = (
        OutboxRecoveryPostgresEvidence.model_validate_json(
            arguments.outbox_recovery_artifact.read_bytes()
        )
        if arguments.outbox_recovery_artifact is not None
        else collect_outbox_recovery_postgres_evidence(
            pytest_artifact_root=arguments.pytest_artifact_root,
            alembic_head=postgres.alembic_head,
        )
    )
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        artifact = await collect_operator_artifact(
            source=PostgresRevisedLiveSource(create_session_factory(engine)),
            request=request,
            slack_readback=slack_readback,
            runtime_health=runtime_health,
            postgres=postgres,
            outbox_recovery=outbox_recovery,
            live_restart=live_restart,
        )
    finally:
        await engine.dispose()
    export_contract(artifact, arguments.output)
    return artifact


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-revised-live-acceptance",
        description=(
            "Collect exact revised live acceptance using SELECT-only Supabase reads, "
            "content-free Slack/health readbacks, and rollback-preserved PG probes."
        ),
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--slack-readback", required=True, type=Path)
    parser.add_argument("--runtime-health-readback", required=True, type=Path)
    parser.add_argument("--postgres-artifact", required=True, type=Path)
    outbox = parser.add_mutually_exclusive_group(required=True)
    outbox.add_argument("--pytest-artifact-root", type=Path)
    outbox.add_argument("--outbox-recovery-artifact", type=Path)
    parser.add_argument("--live-restart-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        artifact = _run_async(_run(arguments))
    except RevisedLiveNotFound:
        code = "revised_live_trace_not_found"
    except Exception:
        # Client errors can contain credentials or Slack text; never echo them.
        code = "revised_live_acceptance_failed"
    else:
        print(
            json.dumps(
                {
                    "artifact": str(arguments.output),
                    "case_count": artifact.case_count,
                    "digest": artifact.digest,
                    "max_ingress_latency_ms": artifact.max_ingress_latency_ms,
                    "status": "complete",
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    print(json.dumps({"code": code, "status": "failed"}, sort_keys=True, separators=(",", ":")))
    return 2


def _run_async[T](coroutine: Coroutine[Any, Any, T]) -> T:
    if sys.platform == "win32":
        with asyncio.Runner(
            loop_factory=lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        ) as runner:
            return runner.run(coroutine)
    return asyncio.run(coroutine)


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
