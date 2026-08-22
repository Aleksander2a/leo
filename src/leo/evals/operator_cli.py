"""Trusted-operator failure bundle CLI with constructor-bound authority."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from leo.evals.failure import (
    AsyncFailureBundleSource,
    FailureExportAuthority,
    FailureExportNotFound,
    ScopedFailureBundleStore,
    export_failure_bundle,
    export_failure_bundle_async,
    import_failure_bundle,
)


def run_operator_cli(
    argv: Sequence[str],
    *,
    store: ScopedFailureBundleStore,
    authority: FailureExportAuthority,
    stdout: TextIO | None = None,
) -> int:
    """Run with trusted dependencies supplied by the operator composition root."""

    arguments = _parse_arguments(argv)
    stream = stdout or sys.stdout

    try:
        if arguments.command == "export":
            receipt = export_failure_bundle(
                store,
                authority=authority,
                run_id=arguments.run_id,
                destination=arguments.output,
            )
            payload = {
                "action": "export",
                "authority_digest": authority.access_digest,
                "receipt": receipt.model_dump(mode="json"),
            }
        else:
            bundle = import_failure_bundle(arguments.input)
            if bundle.failure.run_id not in authority.allowed_run_ids:
                raise FailureExportNotFound
            payload = {
                "action": "import",
                "authority_digest": authority.access_digest,
                "run_id": bundle.failure.run_id,
                "bundle_digest": bundle.digest,
                "fixture_id": bundle.fixture_id,
            }
    except FailureExportNotFound:
        print(
            json.dumps(
                {"status": "not_found", "code": "failure_bundle_not_found"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=stream,
        )
        return 1
    print(
        json.dumps(
            {"status": "ok", **payload},
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )
    return 0


async def run_operator_cli_async(
    argv: Sequence[str],
    *,
    source: AsyncFailureBundleSource,
    authority: FailureExportAuthority,
    stdout: TextIO | None = None,
) -> int:
    """Run against a durable async event source with the same bound authority."""

    arguments = _parse_arguments(argv)
    stream = stdout or sys.stdout
    try:
        if arguments.command == "export":
            receipt = await export_failure_bundle_async(
                source,
                authority=authority,
                run_id=arguments.run_id,
                destination=arguments.output,
            )
            payload = {
                "action": "export",
                "authority_digest": authority.access_digest,
                "receipt": receipt.model_dump(mode="json"),
            }
        else:
            bundle = import_failure_bundle(arguments.input)
            if bundle.failure.run_id not in authority.allowed_run_ids:
                raise FailureExportNotFound
            payload = {
                "action": "import",
                "authority_digest": authority.access_digest,
                "run_id": bundle.failure.run_id,
                "bundle_digest": bundle.digest,
                "fixture_id": bundle.fixture_id,
            }
    except FailureExportNotFound:
        print(
            json.dumps(
                {"status": "not_found", "code": "failure_bundle_not_found"},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=stream,
        )
        return 1
    print(
        json.dumps(
            {"status": "ok", **payload},
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stream,
    )
    return 0


def _parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="leo-eval-failure")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--run-id", required=True)
    export.add_argument("--output", required=True, type=Path)
    import_command = commands.add_parser("import")
    import_command.add_argument("--input", required=True, type=Path)
    return parser.parse_args(tuple(argv))


def main(argv: Sequence[str] | None = None) -> int:
    """Fail closed: authority cannot be selected from a production command line."""

    del argv
    print(
        json.dumps(
            {
                "status": "unavailable",
                "code": "trusted_operator_composition_required",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 2


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
