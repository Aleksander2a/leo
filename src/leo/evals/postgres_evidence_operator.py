"""Offline operator that binds observed rollback-safe pytest artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from leo.evals.durable_recovery import DurableRecoveryArtifact
from leo.evals.failure import import_failure_bundle
from leo.evals.final_evidence import (
    PostgresReliabilityEvidence,
    export_postgres_reliability_evidence,
    repository_alembic_head,
)

_EVENT_ARTIFACT_NAME = "event-recovery-artifact.json"
_PLAN_ARTIFACT_NAME = "plan-recovery-artifact.json"
_FAILURE_EXPORT_NAME = "durable-failure.json"


def collect_postgres_reliability_evidence(
    *,
    pytest_artifact_root: Path,
    destination: Path,
) -> PostgresReliabilityEvidence:
    """Discover exactly one of each observed artifact and reject ambiguity."""

    if not pytest_artifact_root.is_dir():
        raise ValueError("Postgres pytest artifact root is absent")
    event_path = _exact_artifact(pytest_artifact_root, _EVENT_ARTIFACT_NAME)
    plan_path = _exact_artifact(pytest_artifact_root, _PLAN_ARTIFACT_NAME)
    failure_path = _exact_artifact(pytest_artifact_root, _FAILURE_EXPORT_NAME)
    event_recovery = DurableRecoveryArtifact.model_validate_json(event_path.read_bytes())
    plan_recovery = DurableRecoveryArtifact.model_validate_json(plan_path.read_bytes())
    failure_bundle = import_failure_bundle(failure_path)
    return export_postgres_reliability_evidence(
        alembic_head=repository_alembic_head(),
        event_recovery=event_recovery,
        plan_recovery=plan_recovery,
        failure_bundle=failure_bundle,
        destination=destination,
    )


def _exact_artifact(root: Path, name: str) -> Path:
    matches = tuple(sorted(item for item in root.rglob(name) if item.is_file()))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one observed {name} artifact")
    return matches[0]


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-postgres-evidence",
        description=(
            "Validate and bind rollback-safe pytest artifacts without querying a database."
        ),
    )
    parser.add_argument("--pytest-artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        evidence = collect_postgres_reliability_evidence(
            pytest_artifact_root=arguments.pytest_artifact_root,
            destination=arguments.output,
        )
    except Exception:
        print(
            json.dumps(
                {"code": "postgres_evidence_collection_failed", "status": "failed"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {
                "alembic_head": evidence.alembic_head,
                "artifact": str(arguments.output),
                "digest": evidence.digest,
                "status": "ok",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
