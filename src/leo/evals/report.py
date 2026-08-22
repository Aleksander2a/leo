"""Deterministic machine and human reports for offline scenarios."""

from __future__ import annotations

import json

from leo.evals.models import ScenarioResult, ScenarioStatus


def machine_report(results: tuple[ScenarioResult, ...]) -> str:
    payload = {
        "report_version": "eval-report-v1",
        "counts": {
            "total": len(results),
            "passed": sum(result.status is ScenarioStatus.PASSED for result in results),
            "failed": sum(result.status is ScenarioStatus.FAILED for result in results),
            "unsupported": sum(result.status is ScenarioStatus.UNSUPPORTED for result in results),
        },
        "results": [result.model_dump(mode="json") for result in results],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def markdown_report(results: tuple[ScenarioResult, ...]) -> str:
    lines = ["# Leo offline evaluation", "", "| Scenario | Status | Reason |", "|---|---|---|"]
    lines.extend(
        f"| `{result.scenario_id}` | `{result.status.value}` | `{result.reason}` |"
        for result in results
    )
    return "\n".join(lines) + "\n"
