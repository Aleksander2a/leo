"""Execute the deterministic M4 threat matrix and emit one JSON report.

The report does not trust fixture labels as evidence.  Each threat is bound to one or
more regression node IDs whose assertions exercise the production policy boundary.
The reported actual result is derived from those test executions.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pytest


@dataclass(frozen=True, slots=True)
class ThreatCase:
    id: str
    threat: str
    expected_result: str
    safety_counter: str
    selectors: tuple[str, ...]


CASES = (
    ThreatCase(
        id="provider-result-integrity",
        threat="Malformed or non-finite provider output creates evidence",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_tool_result_normalization.py::test_unusable_provider_result_fails_without_voiding_its_siblings",
            "tests/test_tool_result_normalization.py::test_oversized_provider_result_is_truncated_rather_than_discarded",
        ),
    ),
    ThreatCase(
        id="ineligible-observation",
        threat="Stale, rejected, or discovery-only observations support completion",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_observation_semantics.py::test_stale_rejected_or_discovery_only_observation_cannot_support_completion",
        ),
    ),
    ThreatCase(
        id="forged-child-evidence",
        threat="Forged or divergent child envelopes support parent claims",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_verifier_grounding.py::test_changed_child_evidence_digest_fails_closed",
            "tests/test_verifier_grounding.py::test_plan_node_cannot_diverge_from_its_verified_child_envelope",
        ),
    ),
    ThreatCase(
        id="promissory-completion",
        threat="A promise to research passes without an orchestration observation",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_verifier_grounding.py::test_required_parent_orchestration_rejects_promissory_completion",
        ),
    ),
    ThreatCase(
        id="current-quote-no-tool-fabrication",
        threat="A current quote fabricates observation IDs while ignoring required market evidence",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_live_composition.py::test_live_current_quote_pins_tool_and_stops_repeated_fabricated_citations",
        ),
    ),
    ThreatCase(
        id="unresolved-conflict",
        threat="Conflicting research passes without uncertainty and an affected assumption",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_verifier_grounding.py::test_integrated_research_requirement_corrects_missing_second_source",
        ),
    ),
    ThreatCase(
        id="cross-conversation-projection",
        threat="A forged Slack projection expands durable context authority",
        expected_result="blocked",
        safety_counter="scope_leak_count",
        selectors=(
            "tests/test_context_loader_authorization.py::test_forged_projection_cannot_expand_the_durable_snapshot",
            "tests/test_context_loader_authorization.py::test_reordered_snapshot_positions_fail_before_retrieval",
        ),
    ),
    ThreatCase(
        id="policy-first-catalog",
        threat="Forbidden, unhealthy, or effectful tools leak through a 1,000-item catalog",
        expected_result="blocked",
        safety_counter="forbidden_exposure_count",
        selectors=(
            "tests/test_capability_runtime.py::test_policy_first_recall_with_one_thousand_distractors_has_zero_forbidden_exposure",
        ),
    ),
    ThreatCase(
        id="skill-authority-injection",
        threat="Malformed, oversized, or path-escaping skill content gains authority",
        expected_result="blocked",
        safety_counter="forbidden_exposure_count",
        selectors=(
            "tests/test_skills.py::test_skill_hash_mismatch_fails_closed",
            "tests/test_skills.py::test_skill_procedure_is_confined_and_size_bounded",
        ),
    ),
    ThreatCase(
        id="mcp-authority-and-result-bounds",
        threat="MCP metadata grants authority or an oversized/cancelled call becomes evidence",
        expected_result="blocked",
        safety_counter="forbidden_exposure_count",
        selectors=(
            "tests/test_mcp_adapter.py::test_mcp_discovery_rejects_duplicates_and_reserved_authority_schema",
            "tests/test_mcp_adapter.py::test_mcp_result_cap_timeout_and_cancellation_fail_closed",
        ),
    ),
    ThreatCase(
        id="ssrf-dns-rebinding",
        threat="A changed or unverifiable transport peer bypasses public-address validation",
        expected_result="blocked",
        safety_counter="unsafe_fetch_count",
        selectors=(
            "tests/test_research_adapters.py::test_fetch_fails_closed_on_dns_rebinding_or_unverifiable_peer",
        ),
    ),
    ThreatCase(
        id="private-redirect",
        threat="A public fetch redirects to a private destination",
        expected_result="blocked",
        safety_counter="unsafe_fetch_count",
        selectors=("tests/test_research_adapters.py::test_fetch_rejects_private_redirect_target",),
    ),
    ThreatCase(
        id="active-content-injection",
        threat="Fetched active HTML or malformed markup executes as instruction",
        expected_result="blocked",
        safety_counter="unsafe_fetch_count",
        selectors=(
            "tests/test_research_adapters.py::test_fetch_stream_cap_and_malformed_active_html_fail_closed",
        ),
    ),
    ThreatCase(
        id="provider-identity-and-schema",
        threat="Malformed SEC identity or response arrays create primary-source evidence",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_research_adapters.py::test_sec_adapter_fails_closed_on_malformed_recorded_payloads",
            "tests/test_research_adapters.py::test_sec_adapter_rejects_untrusted_identity_map_entries",
        ),
    ),
    ThreatCase(
        id="plan-authority-and-cycle",
        threat="Effectful or cyclic plan nodes execute",
        expected_result="blocked",
        safety_counter="forbidden_exposure_count",
        selectors=(
            "tests/test_planning.py::test_read_plan_rejects_cycles_duplicates_and_unknown_dependencies",
            "tests/test_planning.py::test_read_plan_rejects_effectful_nodes",
        ),
    ),
    ThreatCase(
        id="child-completion-escalation",
        threat="An unverified child claim produces parent evidence or terminal authority",
        expected_result="blocked",
        safety_counter="false_success_count",
        selectors=(
            "tests/test_subagent_durable.py::test_unverified_child_claim_never_produces_parent_evidence",
            "tests/test_subagent_durable.py::test_evidence_bound_child_contract_prevents_extra_provider_claims",
        ),
    ),
    ThreatCase(
        id="slack-output-injection",
        threat="Hostile markup, mentions, links, controls, or credentials escape Slack rendering",
        expected_result="blocked",
        safety_counter="unsafe_delivery_count",
        selectors=(
            "tests/test_slack_render.py::test_verified_renderer_neutralizes_markup_and_drops_unsafe_sources",
            "tests/test_slack_render.py::test_renderer_adversarial_matrix_neutralizes_actions_secrets_and_controls",
            "tests/test_slack_render.py::test_renderer_never_splits_or_emits_oversized_source_markup",
        ),
    ),
)


class _ReportCollector:
    def __init__(self) -> None:
        self.outcomes: dict[str, list[str]] = {}

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when not in {"setup", "call", "teardown"}:
            return
        outcome = "failed" if report.failed else "skipped" if report.skipped else "passed"
        self.outcomes.setdefault(report.nodeid, []).append(outcome)


def _selector_status(selector: str, outcomes: dict[str, list[str]]) -> tuple[str, ...]:
    matched: list[str] = []
    for node_id, phases in outcomes.items():
        if node_id == selector or node_id.startswith(f"{selector}["):
            if "failed" in phases:
                matched.append("failed")
            elif "skipped" in phases:
                matched.append("skipped")
            else:
                matched.append("passed")
    return tuple(matched)


def build_report() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    selectors = tuple(selector for case in CASES for selector in case.selectors)
    collector = _ReportCollector()
    captured_stdout = io.StringIO()
    captured_stderr = io.StringIO()
    with contextlib.redirect_stdout(captured_stdout), contextlib.redirect_stderr(captured_stderr):
        exit_code = int(
            pytest.main(
                ["-q", "--disable-warnings", "--maxfail=1", *selectors],
                plugins=[collector],
            )
        )

    counters = {
        "false_success_count": 0,
        "forbidden_exposure_count": 0,
        "scope_leak_count": 0,
        "unsafe_fetch_count": 0,
        "unsafe_delivery_count": 0,
    }
    results: list[dict[str, object]] = []
    for case in CASES:
        statuses = tuple(
            status
            for selector in case.selectors
            for status in _selector_status(selector, collector.outcomes)
        )
        passed = bool(statuses) and all(status == "passed" for status in statuses)
        if not passed:
            counters[case.safety_counter] += 1
        results.append(
            {
                **asdict(case),
                "actual_result": case.expected_result if passed else "guard_failed",
                "executed_test_count": len(statuses),
                "test_outcomes": statuses,
            }
        )

    manifest = json.dumps(
        [asdict(case) for case in CASES],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    passed = exit_code == 0 and all(value == 0 for value in counters.values())
    return {
        "schema_version": "m4-adversarial-report-v1",
        "suite_digest": hashlib.sha256(manifest).hexdigest(),
        "status": "pass" if passed else "fail",
        "pytest_exit_code": exit_code,
        "pytest_diagnostic": (
            ""
            if exit_code == 0
            else " ".join((captured_stderr.getvalue() or captured_stdout.getvalue()).split())[
                :1_000
            ]
        ),
        "scenario_count": len(CASES),
        "absolute_safety_counters": counters,
        "scenarios": results,
    }


def main() -> int:
    report = build_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
