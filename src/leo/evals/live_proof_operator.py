"""Trusted operator composition for read-only M5 live-proof collection."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import selectors
import sys
import tempfile
from collections.abc import Coroutine, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from leo.config import Settings
from leo.evals.live_proof import (
    LiveProofAuthority,
    LiveProofNotFound,
    LiveProofRequest,
    PostgresLiveProofSource,
    attach_live_collection,
    collect_live_proof,
)
from leo.evals.proof import ProofManifest
from leo.persistence.database import create_database_engine, create_session_factory


async def _run(arguments: argparse.Namespace) -> int:
    settings = Settings()
    if settings.database_url is None or settings.leo_slack_team_id is None:
        raise RuntimeError("live_proof_configuration_missing")
    request = LiveProofRequest.model_validate_json(arguments.request.read_text(encoding="utf-8"))
    manifest = _load_manifest(arguments.proof)
    authority = LiveProofAuthority(
        organization_id=settings.leo_organization_id,
        team_id=settings.leo_slack_team_id,
        actor_id=arguments.actor_id,
        not_before_received_at=_parse_datetime(arguments.not_before_received_at),
        not_before_message_ts=arguments.not_before_message_ts,
        allowed_bindings=request.bindings,
    )
    engine = create_database_engine(settings.database_url.get_secret_value())
    try:
        collection = await collect_live_proof(
            PostgresLiveProofSource(create_session_factory(engine)),
            authority=authority,
            request=request,
        )
    finally:
        await engine.dispose()
    output = attach_live_collection(manifest, collection)
    _atomic_write(arguments.output, output.model_dump_json(indent=2) + "\n")
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "collection_digest": collection.digest,
                "manifest_digest": output.digest,
                "pending_evidence_ids": [str(item) for item in collection.pending_evidence_ids],
                "status": collection.status,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if collection.status == "complete" else 1


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-live-proof",
        description=(
            "Reconcile exact fresh Slack/run bindings using SELECT-only durable reads. "
            "Organization, team, and DATABASE_URL come only from trusted runtime settings."
        ),
    )
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--proof", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--actor-id", required=True)
    parser.add_argument("--not-before-received-at", required=True)
    parser.add_argument("--not-before-message-ts", required=True)
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> ProofManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("proof input must be a JSON object")
    nested = payload.get("proof_manifest")
    return ProofManifest.model_validate(nested if isinstance(nested, dict) else payload)


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("not-before-received-at must be timezone-aware")
    return parsed


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
    except LiveProofNotFound:
        code = "fresh_live_proof_trace_not_found"
    except Exception:
        # DB/client exceptions can contain connection details; never echo them.
        code = "live_proof_collection_failed"
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
