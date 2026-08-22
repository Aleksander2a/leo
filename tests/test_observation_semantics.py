from __future__ import annotations

from datetime import UTC, datetime

import pytest

from leo.harness.models import (
    ClaimKind,
    EvidenceQuality,
    Observation,
    ObservationStatus,
    OriginRef,
    Run,
    RunBundle,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
)
from leo.harness.research import (
    ResearchClaim,
    ResearchProposal,
    ResearchRequirement,
    verify_research,
)
from leo.persistence.run_store import _observation_model, _observation_row
from leo.persistence.schema import ObservationRow

NOW = datetime(2026, 1, 1, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="domain")


def _observation(
    *,
    status: ObservationStatus = ObservationStatus.RETRIEVED,
    quality: EvidenceQuality = EvidenceQuality.PRIMARY_SOURCE,
    rejection_code: str | None = None,
) -> Observation:
    return Observation(
        id="obs-1",
        scope=SCOPE,
        run_id="run-1",
        tool_call_id="call-1",
        kind="sec.get_recent_filings",
        data={"ticker": "NVDA"},
        source=SourceRef(provider="sec-edgar", reference="filing:NVDA"),
        observed_at=NOW,
        raw_hash="a" * 64,
        status=status,
        quality=quality,
        rejection_code=rejection_code,
    )


def _bundle(observation: Observation) -> RunBundle:
    thread = Thread(
        id="thread-1",
        scope=SCOPE,
        origin=OriginRef(provider="test", external_thread_id="conversation-1"),
    )
    task = Task(id="task-1", thread_id=thread.id, scope=SCOPE, objective="Research NVDA")
    run = Run(id="run-1", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=(observation,))


@pytest.mark.parametrize(
    ("status", "quality", "rejection_code"),
    [
        (ObservationStatus.STALE, EvidenceQuality.PRIMARY_SOURCE, None),
        (ObservationStatus.REJECTED, EvidenceQuality.PRIMARY_SOURCE, "schema_rejected"),
        (ObservationStatus.RETRIEVED, EvidenceQuality.DISCOVERY_ONLY, None),
    ],
)
def test_stale_rejected_or_discovery_only_observation_cannot_support_completion(
    status: ObservationStatus,
    quality: EvidenceQuality,
    rejection_code: str | None,
) -> None:
    observation = _observation(
        status=status,
        quality=quality,
        rejection_code=rejection_code,
    )
    verification = verify_research(
        ResearchProposal(
            answer="NVDA has a filing.",
            claims=(
                ResearchClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement="NVDA has a filing.",
                    observation_ids=(observation.id,),
                ),
            ),
        ),
        _bundle(observation),
        now=NOW,
        requirement=ResearchRequirement(),
    )

    assert verification.status.value == "fail"
    assert not next(
        check for check in verification.checks if check.name.endswith("status_quality")
    ).passed


def test_observation_row_round_trip_preserves_current_and_legacy_provenance() -> None:
    current = _observation()
    current_row = _observation_row(current)
    assert current_row.status == "retrieved"
    assert current_row.quality == "primary_source"
    assert current_row.schema_version == "observation-v2"
    assert current_row.normalization_version == "normalization-v1"

    legacy_row = ObservationRow(
        id="obs-legacy",
        run_id="run-1",
        organization_id="org",
        strategy_id="domain",
        tool_call_id="call-legacy",
        kind="market.get_quote",
        data={"symbol": "NVDA", "price": 1},
        source={"provider": "fixture", "reference": "legacy"},
        observed_at=NOW,
        expires_at=None,
        raw_hash="b" * 64,
        status="retrieved",
        quality="provider_reported",
        schema_version="observation-v1",
        normalization_version="legacy-v1",
        rejection_code=None,
    )
    legacy = _observation_model(legacy_row)
    assert legacy.schema_version == "observation-v1"
    assert legacy.normalization_version == "legacy-v1"
    assert legacy.status is ObservationStatus.RETRIEVED
    assert legacy.quality is EvidenceQuality.PROVIDER_REPORTED


def test_rejected_observation_requires_explicit_rejection_code() -> None:
    with pytest.raises(ValueError, match="rejection code"):
        _observation(status=ObservationStatus.REJECTED)
