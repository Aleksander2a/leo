"""Offline collector for rollback-preserved pending/missing outbox probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from leo.evals.final_evidence import repository_alembic_head
from leo.evals.revised_live_acceptance import export_contract
from leo.evals.revised_live_acceptance_operator import (
    collect_outbox_recovery_postgres_evidence,
)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-outbox-recovery-evidence",
        description="Bind exact rollback-preserved pending/missing outbox recovery probes.",
    )
    parser.add_argument("--pytest-artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        evidence = collect_outbox_recovery_postgres_evidence(
            pytest_artifact_root=arguments.pytest_artifact_root,
            alembic_head=repository_alembic_head(),
        )
        export_contract(evidence, arguments.output)
    except Exception:
        print(
            json.dumps(
                {"code": "outbox_recovery_evidence_failed", "status": "failed"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
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


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
