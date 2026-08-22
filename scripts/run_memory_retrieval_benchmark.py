"""Run and verify the frozen M3 memory-retrieval benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from leo.memory.benchmark import (
    load_frozen_retrieval_fixture,
    run_retrieval_benchmark,
    validate_committed_retrieval_report,
)

DEFAULT_FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/memory-retrieval-v1"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--check-committed",
        action="store_true",
        help="Fail when report.json differs from a deterministic rerun.",
    )
    args = parser.parse_args()
    if args.check_committed:
        report = validate_committed_retrieval_report(args.fixture)
    else:
        report = run_retrieval_benchmark(load_frozen_retrieval_fixture(args.fixture))
    print(report.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
