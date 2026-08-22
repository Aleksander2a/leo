[CmdletBinding()]
param(
    [switch]$ListenerStopped
)

$ErrorActionPreference = "Stop"

if (-not $ListenerStopped) {
    throw "Stop every Leo Slack listener and standalone worker, then rerun with -ListenerStopped."
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project interpreter not found at $python"
}
$packetRunId = [Guid]::NewGuid().ToString("N").Substring(0, 12)
$pytestArtifactRoot = Join-Path `
    $repositoryRoot `
    "artifacts\.pytest-tmp-m2-m5-postgres-$packetRunId"
$postgresEvidence = Join-Path $repositoryRoot "artifacts\m5-postgres-reliability-v1.json"
$outboxRecoveryEvidence = Join-Path $repositoryRoot "artifacts\m5-outbox-recovery-v1.json"
$finalEvidence = Join-Path $repositoryRoot "artifacts\m5-final-evidence-v1.json"
$postgresEvidenceCandidate = Join-Path `
    $repositoryRoot `
    "artifacts\m5-postgres-reliability-$packetRunId.json"
$outboxRecoveryEvidenceCandidate = Join-Path `
    $repositoryRoot `
    "artifacts\m5-outbox-recovery-$packetRunId.json"
$finalEvidenceCandidate = Join-Path `
    $repositoryRoot `
    "artifacts\m5-final-evidence-$packetRunId.json"
$packetStatusPath = Join-Path $repositoryRoot "artifacts\m2-postgres-acceptance-latest.json"
$packetStatus = "failed"

$testTargets = @(
    "tests/postgres/test_m2_two_connection_contract.py"
    "tests/postgres/test_slack_scope_contract.py"
    "tests/postgres/test_context_loader_authorization.py"
    "tests/postgres/test_conversation_scope.py"
    "tests/postgres/test_launch_intent_contract.py"
    "tests/postgres/test_task_leases.py"
    "tests/postgres/test_outbox.py"
    "tests/postgres/test_health.py"
    "tests/postgres/test_m5_reliability.py"
    "tests/postgres/test_slack_cancellation_recovery.py::test_admission_crash_matrix_rolls_back_or_leaves_recoverable_intent"
    "tests/postgres/test_slack_cancellation_recovery.py::test_launch_crash_matrix_converges_on_one_canonical_task_run"
    "tests/postgres/test_slack_cancellation_recovery.py::test_crash_after_launch_commit_before_notify_is_resignalled_on_startup"
    "tests/postgres/test_slack_cancellation_recovery.py::test_authorized_slack_cancel_terminalizes_parent_and_control_idempotently"
)

$priorAcknowledgement = [Environment]::GetEnvironmentVariable(
    "LEO_SHARED_DEMO_RACE_ACK",
    [EnvironmentVariableTarget]::Process
)

Push-Location $repositoryRoot
try {
    [Environment]::SetEnvironmentVariable(
        "LEO_SHARED_DEMO_RACE_ACK",
        "listener-stopped",
        [EnvironmentVariableTarget]::Process
    )
    & $python -m pytest -q "--basetemp=$pytestArtifactRoot" @testTargets
    if ($LASTEXITCODE -ne 0) {
        throw "M2 current-head Postgres acceptance packet failed with exit code $LASTEXITCODE."
    }
    & $python -m leo.evals.postgres_evidence_operator `
        --pytest-artifact-root $pytestArtifactRoot `
        --output $postgresEvidenceCandidate
    if ($LASTEXITCODE -ne 0) {
        throw "Postgres evidence collection failed with exit code $LASTEXITCODE."
    }
    & $python -m leo.evals.outbox_recovery_operator `
        --pytest-artifact-root $pytestArtifactRoot `
        --output $outboxRecoveryEvidenceCandidate
    if ($LASTEXITCODE -ne 0) {
        throw "Outbox recovery evidence collection failed with exit code $LASTEXITCODE."
    }
    & $python -m leo.evals.final_evidence `
        --offline-report (Join-Path $repositoryRoot "artifacts\m5-frozen-offline-report.json") `
        --live-proof (Join-Path $repositoryRoot "artifacts\m5-live-proof-v2.json") `
        --topology (Join-Path $repositoryRoot "artifacts\m5-slack-topology-v1.json") `
        --postgres-artifact $postgresEvidenceCandidate `
        --output $finalEvidenceCandidate
    if ($LASTEXITCODE -ne 0) {
        throw "Final evidence aggregation failed with exit code $LASTEXITCODE."
    }
    Move-Item -LiteralPath $postgresEvidenceCandidate -Destination $postgresEvidence -Force
    Move-Item `
        -LiteralPath $outboxRecoveryEvidenceCandidate `
        -Destination $outboxRecoveryEvidence `
        -Force
    Move-Item -LiteralPath $finalEvidenceCandidate -Destination $finalEvidence -Force
    $packetStatus = "passed"
}
finally {
    [ordered]@{
        schema_version = 1
        run_id = $packetRunId
        status = $packetStatus
        pytest_artifact_root = $pytestArtifactRoot
        postgres_evidence = if ($packetStatus -eq "passed") { $postgresEvidence } else { $null }
        outbox_recovery_evidence = if ($packetStatus -eq "passed") {
            $outboxRecoveryEvidence
        } else {
            $null
        }
        final_evidence = if ($packetStatus -eq "passed") { $finalEvidence } else { $null }
    } | ConvertTo-Json | Set-Content -LiteralPath $packetStatusPath -Encoding utf8
    [Environment]::SetEnvironmentVariable(
        "LEO_SHARED_DEMO_RACE_ACK",
        $priorAcknowledgement,
        [EnvironmentVariableTarget]::Process
    )
    Pop-Location
}
