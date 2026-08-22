from __future__ import annotations

from pathlib import Path

import pytest

from leo.memory.benchmark import load_frozen_retrieval_fixture
from leo.memory.eval_report import run_m3_memory_eval, validate_committed_m3_report

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/memory-retrieval-v1"


@pytest.mark.asyncio
async def test_committed_m3_memory_report_is_replayable_and_has_zero_safety_failures() -> None:
    fixture = load_frozen_retrieval_fixture(FIXTURE)
    report = await validate_committed_m3_report(fixture, FIXTURE / "m3-report.json")

    assert report.passed_count == report.scenario_count == 7
    assert report.leakage_count == 0
    assert report.unauthorized_commit_count == 0
    assert report.forbidden_open_count == 0
    assert {item.category for item in report.scenarios} == {
        "retrieval",
        "memory_write",
        "progressive_navigation",
        "compaction",
        "cache",
        "projection",
        "maintenance",
    }


@pytest.mark.asyncio
async def test_m3_memory_report_digest_is_deterministic() -> None:
    fixture = load_frozen_retrieval_fixture(FIXTURE)

    assert await run_m3_memory_eval(fixture) == await run_m3_memory_eval(fixture)
