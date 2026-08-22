from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_m4_machine_report_is_derived_from_executed_guards() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "scripts/m4_adversarial_report.py"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    report = json.loads(completed.stdout)
    assert report["schema_version"] == "m4-adversarial-report-v1"
    assert report["status"] == "pass"
    assert report["scenario_count"] >= 16
    assert all(value == 0 for value in report["absolute_safety_counters"].values())
    assert all(case["actual_result"] == case["expected_result"] for case in report["scenarios"])
    assert all(case["executed_test_count"] > 0 for case in report["scenarios"])
