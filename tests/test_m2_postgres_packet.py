from __future__ import annotations

from pathlib import Path


def test_m2_postgres_packet_collects_only_its_current_pytest_run() -> None:
    script = Path("scripts/run_m2_postgres_acceptance.ps1").read_text(encoding="utf-8")

    assert '[Guid]::NewGuid().ToString("N").Substring(0, 12)' in script
    assert '"artifacts\\.pytest-tmp-m2-m5-postgres-$packetRunId"' in script
    assert "--pytest-artifact-root $pytestArtifactRoot" in script
    assert '"artifacts\\m5-postgres-reliability-$packetRunId.json"' in script
    assert '"artifacts\\m5-final-evidence-$packetRunId.json"' in script


def test_m2_postgres_packet_publishes_canonical_evidence_only_after_success() -> None:
    script = Path("scripts/run_m2_postgres_acceptance.ps1").read_text(encoding="utf-8")

    test_gate = script.index('throw "M2 current-head Postgres acceptance packet failed')
    collection = script.index("-m leo.evals.postgres_evidence_operator")
    final_gate = script.index('throw "Final evidence aggregation failed')
    publish_postgres = script.index(
        "Move-Item -LiteralPath $postgresEvidenceCandidate -Destination $postgresEvidence -Force"
    )
    publish_final = script.index(
        "Move-Item -LiteralPath $finalEvidenceCandidate -Destination $finalEvidence -Force"
    )
    passed = script.index('$packetStatus = "passed"', publish_final)

    assert test_gate < collection < final_gate < publish_postgres < publish_final < passed
    assert '$packetStatus = "failed"' in script
    assert '"artifacts\\m2-postgres-acceptance-latest.json"' in script
