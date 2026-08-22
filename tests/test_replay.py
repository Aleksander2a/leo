from __future__ import annotations

import json
import socket
from datetime import UTC, datetime, timedelta

import pytest
from typer.testing import CliRunner

from leo.cli import app
from leo.config import Settings
from leo.fixtures import FIXTURE_CATALOG, FixtureNotFoundError, run_fixture
from leo.harness.models import (
    BudgetUsage,
    Claim,
    ClaimKind,
    ContextManifest,
    ContextSegment,
    EventType,
    Observation,
    OriginRef,
    Run,
    RunBundle,
    RunEvent,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    TaskStatus,
    Thread,
)
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    PlanStatus,
    revision_digest,
)
from leo.replay import (
    ReplayFormat,
    ReplayLane,
    export_replay,
    normalize_replay,
    render_replay_json,
    render_replay_text,
)


@pytest.fixture(autouse=True)
def _isolate_cli_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    isolated = Settings(_env_file=None)
    monkeypatch.setattr("leo.cli.Settings", lambda: isolated)


NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="org", strategy_id="conversation:C-A")


def _completed_bundle(
    *,
    task_id: str = "parent-task",
    run_id: str = "parent-run",
    parent_task_id: str | None = None,
    secret_payload: bool = False,
) -> RunBundle:
    thread = Thread(
        id=f"thread-{task_id}",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="C-A"),
    )
    task = Task(
        id=task_id,
        thread_id=thread.id,
        scope=SCOPE,
        objective="Research the exact question",
        parent_task_id=parent_task_id,
        continuation_kind="child" if parent_task_id else "root",
        status=TaskStatus.COMPLETED,
        observation_ids=("observation-1",),
        final_output="Verified answer",
        version=2,
    )
    run = Run(
        id=run_id,
        task_id=task.id,
        scope=SCOPE,
        status=RunStatus.COMPLETED,
        iteration=1,
        usage=BudgetUsage(model_calls=1, tool_calls=1),
        started_at=NOW,
        final_output="Verified answer",
        terminal_reason="verified_completion",
        version=2,
    )
    observation = Observation(
        id="observation-1",
        scope=SCOPE,
        run_id=run.id,
        tool_call_id="call-1",
        kind="fixture.read",
        data=(
            {"authorization": "Bearer synthetic-secret", "content": "safe"}
            if secret_payload
            else {"content": "safe"}
        ),
        source=SourceRef(provider="fixture", reference="public-source"),
        observed_at=NOW + timedelta(seconds=1),
        raw_hash="hash-1",
    )
    claim = Claim(
        id="claim-1",
        scope=SCOPE,
        run_id=run.id,
        kind=ClaimKind.SOURCE_CLAIM,
        statement="Verified answer",
        observation_ids=(observation.id,),
    )
    events = (
        RunEvent(
            id=f"{run.id}-event-1",
            run_id=run.id,
            task_id=task.id,
            sequence=1,
            type=EventType.TASK_STARTED,
            occurred_at=NOW,
            iteration=0,
            payload={"phase": "research"},
        ),
        RunEvent(
            id=f"{run.id}-event-2",
            run_id=run.id,
            task_id=task.id,
            sequence=2,
            type=EventType.RUN_COMPLETED,
            occurred_at=NOW + timedelta(seconds=2),
            iteration=1,
            payload={"status": "completed"},
        ),
    )
    return RunBundle(
        thread=thread,
        task=task,
        run=run,
        observations=(observation,),
        claims=(claim,),
        events=events,
    )


def _plan(parent: RunBundle, child: RunBundle) -> PlanSnapshot:
    definition = PlanNodeDefinition(key="research", objective="Research the source")
    digest = revision_digest("Answer", (definition,))
    revision = PlanRevision(
        id="revision-1",
        plan_id="plan-1",
        number=1,
        goal="Answer",
        nodes=(definition,),
        digest=digest,
        reason="initial_plan",
        created_at=NOW,
    )
    plan = Plan(
        id="plan-1",
        scope=SCOPE,
        parent_task_id=parent.task.id,
        parent_run_id=parent.run.id,
        idempotency_key="fixture:plan",
        initial_digest=digest,
        status=PlanStatus.COMPLETED,
        output="Verified answer",
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=2),
    )
    node = PlanNode(
        id="node-1",
        plan_id=plan.id,
        revision_id=revision.id,
        revision_number=1,
        definition=definition,
        status=PlanNodeStatus.COMPLETED,
        attempt=1,
        child_task_id=child.task.id,
        child_run_id=child.run.id,
        output="Verified answer",
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=2),
    )
    delegation = Delegation(
        id="delegation-1",
        plan_id=plan.id,
        revision_id=revision.id,
        node_id=node.id,
        parent_task_id=parent.task.id,
        parent_run_id=parent.run.id,
        attempt=1,
        owner="fixture-worker",
        claim_token="opaque-claim",
        status=DelegationStatus.COMPLETED,
        child_task_id=child.task.id,
        child_run_id=child.run.id,
        output="Verified answer",
        created_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
    )
    return PlanSnapshot(
        plan=plan,
        revisions=(revision,),
        nodes=(node,),
        delegations=(delegation,),
    )


def _manifest() -> ContextManifest:
    return ContextManifest(
        segments=(
            ContextSegment(
                name="source",
                source_type="context_item",
                source_ids=("C-A:message-1",),
                priority=100,
                pinned=True,
            ),
        ),
    )


def test_normalized_replay_contains_parent_plan_child_evidence_and_manifest() -> None:
    parent = _completed_bundle()
    child = _completed_bundle(
        task_id="child-task",
        run_id="child-run",
        parent_task_id=parent.task.id,
    )
    replay = normalize_replay(
        parent,
        plan=_plan(parent, child),
        children=(child,),
        source_manifest=_manifest(),
    )

    lanes = {entry.lane for entry in replay.entries}
    assert {
        ReplayLane.PARENT_EVENT,
        ReplayLane.PLAN_REVISION,
        ReplayLane.PLAN_NODE,
        ReplayLane.DELEGATION,
        ReplayLane.CHILD_EVENT,
        ReplayLane.OBSERVATION,
        ReplayLane.CLAIM,
    } <= lanes
    assert replay.source_manifest is not None
    assert replay.source_manifest.included_source_ids == ("C-A:message-1",)
    assert replay.omissions == ()
    assert render_replay_json(replay) == render_replay_json(
        normalize_replay(
            parent,
            plan=_plan(parent, child),
            children=(child,),
            source_manifest=_manifest(),
        )
    )
    assert f"digest={replay.digest}" in render_replay_text(replay)


def test_normalized_replay_rejects_wrong_plan_and_child_authority() -> None:
    parent = _completed_bundle()
    child = _completed_bundle(
        task_id="child-task",
        run_id="child-run",
        parent_task_id=parent.task.id,
    )
    plan = _plan(parent, child)
    with pytest.raises(ValueError, match="plan is outside"):
        normalize_replay(
            parent,
            plan=plan.model_copy(
                update={
                    "plan": plan.plan.model_copy(
                        update={
                            "scope": ScopeKey(
                                organization_id="other",
                                strategy_id="conversation:C-A",
                            )
                        }
                    )
                }
            ),
        )
    forged_child = child.model_copy(
        update={"task": child.task.model_copy(update={"parent_task_id": "forged"})}
    )
    with pytest.raises(ValueError, match="child is outside"):
        normalize_replay(parent, children=(forged_child,))
    unlinked_child = _completed_bundle(
        task_id="unlinked-child-task",
        run_id="unlinked-child-run",
        parent_task_id=parent.task.id,
    )
    with pytest.raises(ValueError, match="not linked"):
        normalize_replay(parent, plan=plan, children=(unlinked_child,))


def test_normalized_replay_redacts_secret_fields_and_records_truncation() -> None:
    parent = _completed_bundle(secret_payload=True)

    replay = normalize_replay(parent, max_entries=1)
    encoded = render_replay_json(replay)

    assert "synthetic-secret" not in encoded
    assert replay.omitted_entry_count > 0
    assert replay.omissions == (
        "child_runs_not_bound",
        "plan_snapshot_not_bound",
        "source_manifest_not_persisted",
    )


def test_replay_recovers_latest_persisted_source_manifest_and_rejects_malformed() -> None:
    bundle = _completed_bundle()
    manifest = _manifest().model_copy(
        update={
            "manifest_digest": "a" * 64,
            "schema_version": 2,
            "budget_profile": "parent",
            "estimator_version": "fixture-v1",
        }
    )
    source_payload = {
        "schema_version": 2,
        "manifest_digest": manifest.manifest_digest,
        "budget_profile": "parent",
        "estimator_version": "fixture-v1",
        "included_source_ids": ["C-A:message-1"],
        "excluded_source_ids": [],
        "omitted_source_id_count": 0,
        "included_estimated_tokens": 10,
        "excluded_estimated_tokens": 0,
        "included_estimated_bytes": 40,
        "excluded_estimated_bytes": 0,
    }
    context_event = RunEvent(
        id=f"{bundle.run.id}-event-context",
        run_id=bundle.run.id,
        task_id=bundle.task.id,
        sequence=2,
        type=EventType.CONTEXT_BUILT,
        occurred_at=NOW + timedelta(seconds=1),
        iteration=0,
        payload={"source_manifest": source_payload},
    )
    completed = bundle.events[-1].model_copy(update={"sequence": 3})
    durable = bundle.model_copy(update={"events": (bundle.events[0], context_event, completed)})

    replay = normalize_replay(durable)

    assert replay.source_manifest is not None
    assert replay.source_manifest.included_source_ids == ("C-A:message-1",)
    assert replay.source_manifest.included_estimated_tokens == 10
    assert "source_manifest_not_persisted" not in replay.omissions
    malformed = context_event.model_copy(update={"payload": {"source_manifest": []}})
    forged = bundle.model_copy(update={"events": (bundle.events[0], malformed, completed)})
    with pytest.raises(ValueError, match="source manifest is malformed"):
        normalize_replay(forged)


def test_sanitized_replay_export_is_atomic_and_deterministic(tmp_path) -> None:
    replay = normalize_replay(_completed_bundle(secret_payload=True), max_entries=8)
    destination = tmp_path / "replay.json"

    assert export_replay(replay, destination) == destination.resolve()
    first = destination.read_text(encoding="utf-8")
    assert "synthetic-secret" not in first
    assert json.loads(first)["digest"] == replay.digest
    assert (
        export_replay(replay, destination, output_format=ReplayFormat.JSON) == destination.resolve()
    )
    assert destination.read_text(encoding="utf-8") == first
    assert tuple(tmp_path.glob("*.tmp")) == ()


@pytest.mark.asyncio
async def test_named_fixture_catalog_runs_offline_and_is_repeatable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    first = await run_fixture("clarification")
    second = await run_fixture("clarification")
    assert first == second
    assert "Could you clarify" in (first.replay.final_output or "")

    for fixture in FIXTURE_CATALOG:
        result = await run_fixture(fixture.id)
        assert result.fixture == fixture
        assert result.replay.status is fixture.expected_status
    with pytest.raises(FixtureNotFoundError, match="fixture_not_found"):
        await run_fixture("unknown")


def test_run_fixture_cli_has_stable_text_json_and_unknown_errors() -> None:
    runner = CliRunner()
    text_result = runner.invoke(app, ["run-fixture", "clarification", "--format", "text"])
    assert text_result.exit_code == 0
    assert "Leo replay replay-v1" in text_result.output
    json_result = runner.invoke(app, ["run-fixture", "clarification"])
    assert json_result.exit_code == 0
    assert json.loads(json_result.output)["schema_version"] == "replay-v1"
    missing = runner.invoke(app, ["run-fixture", "unknown"])
    assert missing.exit_code != 0
    assert "unknown fixture ID" in missing.output
