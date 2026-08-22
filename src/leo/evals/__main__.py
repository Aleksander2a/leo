"""Repeatable offline eval/proof entry point (`python -m leo.evals`)."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence

from leo.evals.baseline import run_baseline
from leo.evals.frozen_report import build_frozen_offline_report
from leo.evals.loader import default_scenario_root, load_scenarios
from leo.evals.metrics import build_comparison_report
from leo.evals.models import Scenario
from leo.evals.report import machine_report
from leo.evals.runner import run_scenarios


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run deterministic offline Leo eval fixtures.")
    parser.add_argument("--id", action="append", dest="scenario_ids", default=[])
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Also emit an exact paired frozen-baseline comparison.",
    )
    parser.add_argument(
        "--frozen-report",
        action="store_true",
        help="Emit the all-scenario frozen M5 offline aggregate and proof manifest.",
    )
    parser.add_argument(
        "--code-version",
        default="working-tree",
        help="Revision label bound into --frozen-report metadata.",
    )
    arguments = parser.parse_args(argv)
    selected = frozenset(arguments.scenario_ids) or None
    if arguments.frozen_report and selected is not None:
        parser.error("--frozen-report requires the complete unfiltered scenario cohort")
    scenarios = load_scenarios(default_scenario_root(), scenario_ids=selected)
    if arguments.frozen_report:
        report = build_frozen_offline_report(
            scenarios,
            code_version=arguments.code_version,
        )
        print(report.model_dump_json(indent=2))
        return int(not report.offline_passed)
    results = run_scenarios(scenarios)
    print(machine_report(results), end="")
    failed = any(item.status.value != "passed" for item in results)
    if arguments.baseline:
        baselines = tuple(run_baseline(scenario) for scenario in scenarios)
        comparison = build_comparison_report(
            results,
            baselines,
            config_digest=_comparison_config_digest(scenarios),
        )
        print(
            json.dumps(
                {"comparison": comparison.model_dump(mode="json")},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        failed = (
            failed
            or not comparison.passed
            or any(item.status.value != "passed" for item in baselines)
        )
    return int(failed)


def _comparison_config_digest(scenarios: tuple[Scenario, ...]) -> str:
    payload = [
        scenario.model_dump(
            mode="json",
            include={
                "id",
                "version",
                "provider_mode",
                "fixed_clock",
                "budget",
                "inputs",
            },
        )
        for scenario in scenarios
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":  # pragma: no cover - exercised through the installed entry point.
    raise SystemExit(main())
