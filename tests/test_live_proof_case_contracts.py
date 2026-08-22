from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from leo.evals.live_proof import (
    LiveEvidenceId,
    LiveProofAuthority,
    LiveProofBinding,
    LiveProofIntegrityError,
    _case_contract_summary,
    _context_summary,
    _cross_channel_negative_contract,
    _delegated_contract,
    _direct_conversation_contract,
    _event_summary,
    _memory_write_contract,
    _message_plane_digest,
    _positive_memory_case_contract,
    _provider_contract,
)
from leo.harness.child_evidence import (
    build_child_evidence_envelope,
    serialize_child_evidence_envelope,
)
from leo.harness.models import (
    Claim,
    ClaimKind,
    EventType,
    EvidenceQuality,
    Observation,
    ScopeKey,
    SourceRef,
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
from leo.persistence.schema import (
    ClaimRow,
    ConversationAccessSnapshotRow,
    ConversationActorMembershipRow,
    DeliveryOutboxRow,
    MemoryRecordRow,
    MemoryRevisionRow,
    MemorySourceRow,
    ObservationRow,
    RunEventRow,
    RunRow,
    SanitizedMessageRow,
    SlackIngressEventRow,
    TaskRow,
    ThreadRow,
)

NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
STARTED = NOW
FINISHED = NOW + timedelta(seconds=10)
EXPIRES = NOW + timedelta(minutes=5)
SCOPE = ScopeKey(organization_id="demo-org", strategy_id="demo-strategy")
EMPTY_SCOPE_STATEMENT = "No matching authorized memory was found in this conversation scope."


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _run(
    *,
    run_id: str = "run-parent",
    task_id: str = "task-parent",
    answer: str = "A verified answer.",
    tool_calls: int | None = 1,
    started_at: datetime = STARTED,
    updated_at: datetime = FINISHED,
) -> RunRow:
    usage = {} if tool_calls is None else {"tool_calls": tool_calls}
    return RunRow(
        id=run_id,
        task_id=task_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        status="completed",
        phase="verification",
        iteration=2,
        limits={},
        usage=usage,
        started_at=started_at,
        final_output=answer,
        terminal_reason="verified_completion",
        event_sequence=8,
        version=8,
        created_at=started_at,
        updated_at=updated_at,
    )


def _task(
    *,
    task_id: str = "task-parent",
    parent_task_id: str | None = None,
    continuation_kind: str = "root",
    answer: str = "A verified answer.",
) -> TaskRow:
    return TaskRow(
        id=task_id,
        thread_id="thread-1",
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        objective="Bound objective",
        parent_task_id=parent_task_id,
        continuation_kind=continuation_kind,
        status="completed",
        observation_ids=[],
        verifier_feedback=[],
        final_output=answer,
        version=3,
        created_at=STARTED,
        updated_at=FINISHED,
    )


def _observation(
    *,
    observation_id: str,
    run_id: str = "run-parent",
    tool_call_id: str,
    kind: str,
    data: dict[str, object],
    provider: str,
    reference: str,
    quality: str,
    observed_at: datetime = NOW + timedelta(seconds=2),
    expires_at: datetime | None = EXPIRES,
    url: str | None = None,
) -> ObservationRow:
    source: dict[str, object] = {"provider": provider, "reference": reference}
    if url is not None:
        source["url"] = url
    return ObservationRow(
        id=observation_id,
        run_id=run_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        tool_call_id=tool_call_id,
        kind=kind,
        data=data,
        source=source,
        observed_at=observed_at,
        expires_at=expires_at,
        raw_hash=_hash(f"raw:{observation_id}"),
        status="retrieved",
        quality=quality,
        schema_version="observation-v2",
        normalization_version="normalization-v1",
        rejection_code=None,
    )


def _claim(
    *,
    claim_id: str,
    run_id: str = "run-parent",
    kind: str,
    statement: str,
    observation_id: str,
) -> ClaimRow:
    return ClaimRow(
        id=claim_id,
        run_id=run_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        kind=kind,
        statement=statement,
        observation_ids=[observation_id],
    )


def _event(
    *,
    event_id: str,
    run_id: str,
    task_id: str,
    sequence: int,
    event_type: EventType,
    payload: dict[str, object],
    occurred_at: datetime | None = None,
) -> RunEventRow:
    return RunEventRow(
        id=event_id,
        run_id=run_id,
        task_id=task_id,
        sequence=sequence,
        type=event_type.value,
        occurred_at=occurred_at or NOW + timedelta(seconds=sequence),
        iteration=1,
        schema_version=1,
        payload=payload,
    )


def _tool_path(
    observation: ObservationRow,
    tool: str,
    *,
    task_id: str = "task-parent",
) -> tuple[RunEventRow, ...]:
    common = {
        "run_id": observation.run_id,
        "task_id": task_id,
    }
    return (
        _event(
            event_id=f"event-{observation.id}-start",
            sequence=1,
            event_type=EventType.TOOL_STARTED,
            payload={"tool_call_id": observation.tool_call_id, "tool": tool},
            **common,
        ),
        _event(
            event_id=f"event-{observation.id}-complete",
            sequence=2,
            event_type=EventType.TOOL_COMPLETED,
            payload={"tool_call_id": observation.tool_call_id, "tool": tool},
            **common,
        ),
        _event(
            event_id=f"event-{observation.id}-observed",
            sequence=3,
            event_type=EventType.OBSERVATION_CREATED,
            payload={
                "tool_call_id": observation.tool_call_id,
                "observation_id": observation.id,
            },
            **common,
        ),
    )


def _binding(evidence_id: LiveEvidenceId) -> LiveProofBinding:
    if evidence_id is LiveEvidenceId.DM_MEMBERSHIP_UNION:
        destination = "D-DEMO"
        kind = "dm"
        conversations = ("C-SHARED", destination)
        recall_source = "C-SHARED"
    elif evidence_id is LiveEvidenceId.GROUP_DM:
        destination = "G-DEMO"
        kind = "mpim"
        conversations = (destination,)
        recall_source = None
    else:
        destination = "C-DEMO"
        kind = "ordinary_internal"
        conversations = (destination,)
        recall_source = destination if evidence_id is LiveEvidenceId.MEMORY_RECALL else None
    return LiveProofBinding(
        evidence_id=evidence_id,
        message_ts="1788000001.000001",
        run_id="run-00000000-0000-4000-8000-000000000001",
        expected_destination_id=destination,
        expected_conversation_kind=kind,
        expected_context_conversation_ids=conversations,
        expected_context_access_hash="a" * 64,
        expected_recall_source_conversation_id=recall_source,
        plan_expectation=(
            "required" if evidence_id is LiveEvidenceId.DELEGATED_REPLANNING else "forbidden"
        ),
    )


def _authority() -> LiveProofAuthority:
    return LiveProofAuthority(
        organization_id=SCOPE.organization_id,
        team_id="T-DEMO",
        actor_id="operator",
        not_before_received_at=NOW - timedelta(minutes=1),
        not_before_message_ts="1787999999.000001",
        allowed_bindings=(),
    )


def _ingress() -> SlackIngressEventRow:
    return SlackIngressEventRow(
        event_id="slack-event-1",
        team_id="T-DEMO",
        channel_id="C-DEMO",
        user_id="U-DEMO",
        message_ts="1788000001.000001",
        conversation_kind="ordinary_internal",
    )


def _memory_rows(
    ingress: SlackIngressEventRow,
    task: TaskRow,
    observation: ObservationRow,
) -> tuple[MemoryRecordRow, MemoryRevisionRow, tuple[MemorySourceRow, ...]]:
    record_id = str(observation.data["record_id"])
    source_specs = (
        ("memory-source-event", "slack_event", ingress.event_id),
        ("memory-source-task", "leo_task", task.id),
        ("memory-source-message", "slack_message", ingress.message_ts),
    )
    sources = tuple(
        MemorySourceRow(
            id=source_id,
            organization_id=SCOPE.organization_id,
            strategy_id=SCOPE.strategy_id,
            source_kind=kind,
            reference=reference,
            visibility="conversation_local",
            namespace_id=ingress.channel_id,
        )
        for source_id, kind, reference in source_specs
    )
    content = "The demo project is called Atlas."
    recorded_at = NOW + timedelta(seconds=1)
    record = MemoryRecordRow(
        id=record_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        kind="note",
        visibility="conversation_local",
        namespace_id=ingress.channel_id,
        current_revision=1,
        generation=1,
        status="active",
        created_at=recorded_at,
    )
    revision = MemoryRevisionRow(
        id="memory-revision-1",
        record_id=record_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        number=1,
        content=content,
        content_hash=_hash(content),
        source_ids=[item.id for item in sources],
        visibility="conversation_local",
        namespace_id=ingress.channel_id,
        sensitivity=0.2,
        valid_from=recorded_at,
        valid_until=None,
        recorded_at=recorded_at,
        expires_at=None,
        status="active",
        actor_id=ingress.user_id,
        reason="explicit Slack remember",
        supersedes_revision=None,
    )
    return record, revision, sources


def _memory_search_observation(
    *,
    source_conversation: str,
    selected: bool,
) -> ObservationRow:
    items: list[dict[str, object]] = []
    if selected:
        items.append(
            {
                "kind": "inline",
                "reference": "memory-record-1",
                "content": "The remembered fact.",
                "excerpt": None,
                "handle": None,
                "chunk_count": 0,
                "source_conversation": source_conversation,
                "lifecycle_status": "active",
                "contested": False,
            }
        )
    return _observation(
        observation_id="obs-memory-search",
        tool_call_id="call-memory-search",
        kind="memory.search",
        data={
            "items": items,
            "query_hash": "b" * 64,
            "selected_count": len(items),
            "cache_status": "miss",
            "policy_version": "memory-navigation-v1",
        },
        provider="leo_memory",
        reference="b" * 64,
        quality="internal_context",
        expires_at=None,
    )


def _provider_observation(kind: str, *, run_id: str = "run-parent") -> ObservationRow:
    if kind == "market.get_quote":
        return _observation(
            observation_id=f"obs-quote-{run_id}",
            run_id=run_id,
            tool_call_id=f"call-quote-{run_id}",
            kind=kind,
            data={"symbol": "NVDA", "price": 181.25, "as_of": NOW.isoformat()},
            provider="finnhub",
            reference="quote:NVDA:1788000000",
            quality="provider_reported",
            observed_at=NOW - timedelta(seconds=30),
            url="https://finnhub.io/docs/api/quote",
        )
    return _observation(
        observation_id=f"obs-sec-{run_id}",
        run_id=run_id,
        tool_call_id=f"call-sec-{run_id}",
        kind=kind,
        data={
            "ticker": "NVDA",
            "cik": "0001045810",
            "filings": [
                {
                    "form": "10-Q",
                    "filing_date": "2026-08-20",
                    "accession": "0001045810-26-000001",
                    "primary_document": "nvda-20260731.htm",
                    "filing_url": "https://www.sec.gov/Archives/edgar/data/1045810/doc.htm",
                }
            ],
        },
        provider="sec-edgar",
        reference="submissions:0001045810",
        quality="primary_source",
        url="https://data.sec.gov/submissions/CIK0001045810.json",
    )


def test_memory_write_requires_exact_current_mutation_and_bound_tool_path() -> None:
    ingress = _ingress()
    task = _task()
    run = _run(answer="I remembered that for this conversation.")
    observation = _observation(
        observation_id="obs-memory-remember",
        tool_call_id="call-memory-remember",
        kind="memory.remember",
        data={
            "operation": "remember",
            "record_id": "memory-record-1",
            "revision": 1,
            "status": "active",
        },
        provider="leo_memory",
        reference="memory-record-1",
        quality="internal_context",
        expires_at=None,
    )
    record, revision, sources = _memory_rows(ingress, task, observation)

    invariant, mutation = _memory_write_contract(
        authority=_authority(),
        ingress=ingress,
        task=task,
        run=run,
        events=_tool_path(observation, "memory.remember"),
        observations=(observation,),
        claims=(),
        records=(record,),
        revisions=(revision,),
        sources=sources,
    )
    assert invariant != _hash("[]")
    assert mutation != _hash("[]")

    with pytest.raises(LiveProofIntegrityError, match="foreign or widened"):
        _memory_write_contract(
            authority=_authority(),
            ingress=ingress,
            task=task,
            run=run,
            events=_tool_path(observation, "memory.remember"),
            observations=(observation,),
            claims=(),
            records=(record,),
            revisions=(revision,),
            sources=(
                sources[0].__class__(**{**_row_values(sources[0]), "namespace_id": "C-FOREIGN"}),
                *sources[1:],
            ),
        )

    foreign_reference = sources[0].__class__(
        **{**_row_values(sources[0]), "reference": "other-slack-event"}
    )
    with pytest.raises(LiveProofIntegrityError, match="foreign or widened"):
        _memory_write_contract(
            authority=_authority(),
            ingress=ingress,
            task=task,
            run=run,
            events=_tool_path(observation, "memory.remember"),
            observations=(observation,),
            claims=(),
            records=(record,),
            revisions=(revision,),
            sources=(foreign_reference, *sources[1:]),
        )

    forgotten_record = record.__class__(**{**_row_values(record), "status": "retracted"})
    forgotten_revision = revision.__class__(**{**_row_values(revision), "status": "retracted"})
    with pytest.raises(LiveProofIntegrityError, match="forgotten"):
        _memory_write_contract(
            authority=_authority(),
            ingress=ingress,
            task=task,
            run=run,
            events=_tool_path(observation, "memory.remember"),
            observations=(observation,),
            claims=(),
            records=(forgotten_record,),
            revisions=(forgotten_revision,),
            sources=sources,
        )


def test_memory_write_generic_or_foreign_revision_cannot_self_attest() -> None:
    ingress = _ingress()
    task = _task()
    run = _run(answer="Generic completed answer.")
    with pytest.raises(LiveProofIntegrityError, match="remember observation"):
        _memory_write_contract(
            authority=_authority(),
            ingress=ingress,
            task=task,
            run=run,
            events=(),
            observations=(),
            claims=(),
            records=(),
            revisions=(),
            sources=(),
        )

    observation = _observation(
        observation_id="obs-memory-remember",
        tool_call_id="call-memory-remember",
        kind="memory.remember",
        data={
            "operation": "remember",
            "record_id": "memory-record-1",
            "revision": 1,
            "status": "active",
        },
        provider="leo_memory",
        reference="memory-record-1",
        quality="internal_context",
        expires_at=None,
    )
    record, revision, sources = _memory_rows(ingress, task, observation)
    foreign_revision = revision.__class__(**{**_row_values(revision), "namespace_id": "C-FOREIGN"})
    with pytest.raises(LiveProofIntegrityError, match="foreign, widened, or forgotten"):
        _memory_write_contract(
            authority=_authority(),
            ingress=ingress,
            task=task,
            run=run,
            events=_tool_path(observation, "memory.remember"),
            observations=(observation,),
            claims=(),
            records=(record,),
            revisions=(foreign_revision,),
            sources=sources,
        )


def test_scoped_negative_requires_exact_empty_search_and_canonical_inference() -> None:
    run = _run(answer=EMPTY_SCOPE_STATEMENT)
    observation = _memory_search_observation(source_conversation="C-DEMO", selected=False)
    claim = _claim(
        claim_id="claim-empty-memory",
        kind="inference",
        statement=EMPTY_SCOPE_STATEMENT,
        observation_id=observation.id,
    )
    digest = _cross_channel_negative_contract(
        run,
        _tool_path(observation, "memory.search"),
        (observation,),
        (claim,),
    )
    assert len(digest) == 64

    mutated = _claim(
        claim_id="claim-mutated",
        kind="inference",
        statement="I did not find anything.",
        observation_id=observation.id,
    )
    with pytest.raises(LiveProofIntegrityError, match="exact scoped-empty"):
        _cross_channel_negative_contract(
            run,
            _tool_path(observation, "memory.search"),
            (observation,),
            (mutated,),
        )


@pytest.mark.parametrize(
    ("kind", "provider", "quality", "subject_key", "statement"),
    [
        ("market.get_quote", "finnhub", "provider_reported", "symbol", "NVDA is quoted at 181.25."),
        (
            "sec.get_recent_filings",
            "sec-edgar",
            "primary_source",
            "ticker",
            "NVDA filed form 10-Q on 2026-08-20 under accession 0001045810-26-000001.",
        ),
    ],
)
def test_provider_cases_require_exact_fresh_tool_backed_grounding(
    kind: str,
    provider: str,
    quality: str,
    subject_key: str,
    statement: str,
) -> None:
    observation = _provider_observation(kind)
    run = _run(answer=statement)
    claim = _claim(
        claim_id=f"claim-{kind}",
        kind="source_claim",
        statement=statement,
        observation_id=observation.id,
    )
    digest = _provider_contract(
        run,
        _tool_path(observation, kind),
        (observation,),
        (claim,),
        kind=kind,
        provider=provider,
        quality=quality,
        subject_key=subject_key,
    )
    assert len(digest) == 64

    with pytest.raises(LiveProofIntegrityError, match="tool path"):
        _provider_contract(
            run,
            (),
            (observation,),
            (claim,),
            kind=kind,
            provider=provider,
            quality=quality,
            subject_key=subject_key,
        )
    with pytest.raises(LiveProofIntegrityError, match="one exact observation"):
        _provider_contract(
            run,
            _tool_path(observation, kind),
            (observation, observation),
            (claim,),
            kind=kind,
            provider=provider,
            quality=quality,
            subject_key=subject_key,
        )


def test_positive_memory_recall_rejects_any_foreign_selected_source() -> None:
    binding = _binding(LiveEvidenceId.DM_MEMBERSHIP_UNION)
    observation = _memory_search_observation(source_conversation="C-SHARED", selected=True)
    statement = "The remembered fact came from the shared channel."
    run = _run(answer=statement)
    claim = _claim(
        claim_id="claim-positive-memory",
        kind="inference",
        statement=statement,
        observation_id=observation.id,
    )
    digest = _positive_memory_case_contract(
        binding,
        run,
        _tool_path(observation, "memory.search"),
        (observation,),
        (claim,),
    )
    assert len(digest) == 64

    foreign = observation.__class__(
        **{
            **_row_values(observation),
            "data": {
                **observation.data,
                "selected_count": 2,
                "items": [
                    *observation.data["items"],
                    {
                        **observation.data["items"][0],
                        "source_conversation": "C-FOREIGN",
                    },
                ],
            },
        }
    )
    with pytest.raises(LiveProofIntegrityError, match="wrong source"):
        _positive_memory_case_contract(
            binding,
            run,
            _tool_path(foreign, "memory.search"),
            (foreign,),
            (claim,),
        )


def test_dm_union_context_requires_exact_current_active_membership_rows() -> None:
    binding = _binding(LiveEvidenceId.DM_MEMBERSHIP_UNION)
    ingress = SlackIngressEventRow(
        event_id="slack-event-dm",
        team_id="T-DEMO",
        channel_id=binding.expected_destination_id,
        user_id="U-DEMO",
        message_ts=binding.message_ts,
        conversation_kind="dm",
        context_conversation_ids=list(binding.expected_context_conversation_ids),
        context_access_hash=binding.expected_context_access_hash,
        context_projection_source="dm_membership_intersection",
    )
    snapshots = tuple(
        ConversationAccessSnapshotRow(
            id=f"snapshot-{index}",
            ingress_event_id=ingress.event_id,
            organization_id=SCOPE.organization_id,
            team_id=ingress.team_id,
            actor_id=ingress.user_id,
            destination_external_id=ingress.channel_id,
            conversation_external_id=conversation_id,
            position=index,
            source_kind="dm_membership_intersection",
            context_access_hash=binding.expected_context_access_hash,
            observed_at=NOW,
            created_at=NOW,
        )
        for index, conversation_id in enumerate(binding.expected_context_conversation_ids)
    )
    memberships = tuple(
        ConversationActorMembershipRow(
            id=f"membership-{index}",
            organization_id=SCOPE.organization_id,
            team_id=ingress.team_id,
            actor_id=ingress.user_id,
            conversation_external_id=conversation_id,
            status="active",
            source_kind="dm_membership_intersection",
            context_access_hash=binding.expected_context_access_hash,
            version=2,
            observed_at=NOW,
            created_at=NOW,
            updated_at=NOW,
        )
        for index, conversation_id in enumerate(binding.expected_context_conversation_ids)
    )
    summary = _context_summary(binding, ingress, snapshots, memberships)
    assert summary["current_membership_count"] == 2

    # A later exact-destination turn in one member conversation refreshes that
    # row's observation metadata without changing the actor-and-Leo membership
    # set.  Historical DM authority stays bound to the immutable snapshots;
    # current membership reconciliation therefore compares active identities,
    # not the latest observation's projection hash/source kind.
    refreshed = memberships[0].__class__(
        **{
            **_row_values(memberships[0]),
            "source_kind": "exact_destination",
            "context_access_hash": "b" * 64,
            "version": memberships[0].version + 1,
        }
    )
    refreshed_summary = _context_summary(binding, ingress, snapshots, (refreshed, memberships[1]))
    assert refreshed_summary["current_membership_count"] == 2
    assert refreshed_summary["current_membership_digest"] != summary["current_membership_digest"]

    revoked = memberships[0].__class__(**{**_row_values(memberships[0]), "status": "revoked"})
    with pytest.raises(LiveProofIntegrityError, match="exact active source set"):
        _context_summary(binding, ingress, snapshots, (revoked, memberships[1]))

    extra_active = memberships[0].__class__(
        **{
            **_row_values(memberships[0]),
            "id": "membership-extra",
            "conversation_external_id": "C-FOREIGN",
        }
    )
    with pytest.raises(LiveProofIntegrityError, match="exact active source set"):
        _context_summary(binding, ingress, snapshots, (*memberships, extra_active))

    forged_snapshot = snapshots[0].__class__(
        **{**_row_values(snapshots[0]), "source_kind": "exact_destination"}
    )
    with pytest.raises(LiveProofIntegrityError, match="incomplete or widened"):
        _context_summary(binding, ingress, (forged_snapshot, snapshots[1]), memberships)


def test_private_and_group_direct_cases_forbid_plans_tools_and_claims() -> None:
    run = _run(answer="Hello from Leo.", tool_calls=0)
    assert len(_direct_conversation_contract(run, (), (), (), None)) == 64
    generic = _run(answer="Generic completion.", tool_calls=None)
    with pytest.raises(LiveProofIntegrityError, match="tool, claim, or plan"):
        _direct_conversation_contract(generic, (), (), (), None)
    tool_event = _event(
        event_id="event-tool",
        run_id=run.id,
        task_id=run.task_id,
        sequence=1,
        event_type=EventType.TOOL_STARTED,
        payload={"tool_call_id": "call-forbidden", "tool": "memory.search"},
    )
    with pytest.raises(LiveProofIntegrityError, match="tool, claim, or plan"):
        _direct_conversation_contract(run, (tool_event,), (), (), None)


def _delegated_fixture() -> tuple[
    RunRow,
    tuple[ObservationRow, ...],
    tuple[ClaimRow, ...],
    PlanSnapshot,
    tuple[TaskRow, ...],
    tuple[RunRow, ...],
    tuple[RunEventRow, ...],
    tuple[ObservationRow, ...],
    tuple[ClaimRow, ...],
]:
    child_specs = (
        (
            "quote",
            "child-task-quote",
            "child-run-quote",
            "NVDA is quoted at 181.25.",
            "market.get_quote",
        ),
        (
            "sec",
            "child-task-sec",
            "child-run-sec",
            "NVDA filed form 10-Q on 2026-08-20 under accession 0001045810-26-000001.",
            "sec.get_recent_filings",
        ),
    )
    definitions = tuple(
        PlanNodeDefinition(key=key, objective=f"Research {key}.")
        for key, _task_id, _run_id, _statement, _kind in child_specs
    )
    digest = revision_digest("Research NVDA.", definitions)
    revision = PlanRevision(
        id="plan-revision-1",
        plan_id="plan-1",
        number=1,
        goal="Research NVDA.",
        nodes=definitions,
        digest=digest,
        reason="initial",
        created_at=STARTED,
    )
    child_tasks: list[TaskRow] = []
    child_runs: list[RunRow] = []
    child_events: list[RunEventRow] = []
    child_observations: list[ObservationRow] = []
    child_claims: list[ClaimRow] = []
    nodes: list[PlanNode] = []
    delegations: list[Delegation] = []
    envelopes: list[dict[str, object]] = []
    for index, (key, task_id, run_id, statement, kind) in enumerate(child_specs):
        observed = _provider_observation(kind, run_id=run_id)
        child_run = _run(
            run_id=run_id,
            task_id=task_id,
            answer=statement,
            started_at=STARTED + timedelta(seconds=index),
            updated_at=NOW + timedelta(seconds=8 + index),
        )
        child_task = _task(
            task_id=task_id,
            parent_task_id="task-parent",
            continuation_kind="subagent",
            answer=statement,
        )
        claim_row = _claim(
            claim_id=f"claim-{key}",
            run_id=run_id,
            kind="source_claim",
            statement=statement,
            observation_id=observed.id,
        )
        domain_observation = Observation(
            id=observed.id,
            scope=SCOPE,
            run_id=run_id,
            tool_call_id=observed.tool_call_id,
            kind=observed.kind,
            data=observed.data,
            source=SourceRef.model_validate(observed.source),
            observed_at=observed.observed_at,
            expires_at=observed.expires_at,
            raw_hash=observed.raw_hash,
            quality=EvidenceQuality(observed.quality),
        )
        domain_claim = Claim(
            id=claim_row.id,
            scope=SCOPE,
            run_id=run_id,
            kind=ClaimKind.SOURCE_CLAIM,
            statement=statement,
            observation_ids=(observed.id,),
        )
        envelope = build_child_evidence_envelope(
            child_run_id=run_id,
            answer=statement,
            trace_event_count=2,
            observations=(domain_observation,),
            claims=(domain_claim,),
        )
        serialized = serialize_child_evidence_envelope(envelope)
        node_id = f"plan-node-{key}"
        nodes.append(
            PlanNode(
                id=node_id,
                plan_id="plan-1",
                revision_id=revision.id,
                revision_number=1,
                definition=definitions[index],
                status=PlanNodeStatus.COMPLETED,
                attempt=1,
                child_task_id=task_id,
                child_run_id=run_id,
                output=serialized,
                created_at=STARTED,
                updated_at=child_run.updated_at,
            )
        )
        delegations.append(
            Delegation(
                id=f"delegation-{key}",
                plan_id="plan-1",
                revision_id=revision.id,
                node_id=node_id,
                parent_task_id="task-parent",
                parent_run_id="run-parent",
                attempt=1,
                owner="worker-1",
                claim_token=f"claim-token-{key}",
                status=DelegationStatus.COMPLETED,
                child_task_id=task_id,
                child_run_id=run_id,
                output=serialized,
                created_at=STARTED,
                finished_at=child_run.updated_at,
            )
        )
        child_tasks.append(child_task)
        child_runs.append(child_run)
        child_observations.append(observed)
        child_claims.append(claim_row)
        child_events.extend(
            (
                *_tool_path(observed, kind, task_id=task_id),
                _event(
                    event_id=f"event-{key}-verified",
                    run_id=run_id,
                    task_id=task_id,
                    sequence=4,
                    event_type=EventType.VERIFICATION_PASSED,
                    payload={},
                    occurred_at=NOW + timedelta(seconds=6 + index),
                ),
                _event(
                    event_id=f"event-{key}-complete",
                    run_id=run_id,
                    task_id=task_id,
                    sequence=5,
                    event_type=EventType.RUN_COMPLETED,
                    payload={"reason": "verified_completion"},
                    occurred_at=NOW + timedelta(seconds=8 + index),
                ),
            )
        )
        envelopes.append(envelope.model_dump(mode="json"))
    plan = Plan(
        id="plan-1",
        scope=SCOPE,
        parent_task_id="task-parent",
        parent_run_id="run-parent",
        idempotency_key="plan-key",
        initial_digest=revision.digest,
        status=PlanStatus.COMPLETED,
        current_revision=1,
        output="Plan completed.",
        version=5,
        created_at=STARTED,
        updated_at=FINISHED,
    )
    snapshot = PlanSnapshot(
        plan=plan,
        revisions=(revision,),
        nodes=tuple(nodes),
        delegations=tuple(delegations),
    )
    parent_answer = " ".join(spec[3] for spec in child_specs)
    parent_run = _run(answer=parent_answer)
    parent_observation = _observation(
        observation_id="obs-parent-plan",
        tool_call_id="call-parent-plan",
        kind="agent.execute_research_plan",
        data={
            "status": "completed",
            "completed_count": 2,
            "failed_count": 0,
            "blocked_count": 0,
            "nodes": [{"child_evidence": envelope} for envelope in envelopes],
        },
        provider="leo-subagent-plan",
        reference=plan.id,
        quality="verified_child",
        expires_at=EXPIRES,
    )
    parent_claims = tuple(
        _claim(
            claim_id=f"claim-parent-{index}",
            kind="source_claim",
            statement=spec[3],
            observation_id=parent_observation.id,
        )
        for index, spec in enumerate(child_specs)
    )
    return (
        parent_run,
        (parent_observation,),
        parent_claims,
        snapshot,
        tuple(child_tasks),
        tuple(child_runs),
        tuple(child_events),
        tuple(child_observations),
        tuple(child_claims),
    )


def test_delegated_case_requires_parallel_distinct_durable_child_evidence() -> None:
    fixture = _delegated_fixture()
    invariant, summary = _delegated_contract(
        run=fixture[0],
        parent_events=_tool_path(fixture[1][0], "agent.execute_research_plan"),
        observations=fixture[1],
        claims=fixture[2],
        snapshot=fixture[3],
        child_tasks=fixture[4],
        child_runs=fixture[5],
        child_events=fixture[6],
        child_observations=fixture[7],
        child_claims=fixture[8],
        child_outbox=(),
    )
    assert len(invariant) == 64
    assert summary["delegated_child_count"] == 2
    assert summary["delegated_overlap_verified"] is True

    epistemic_claims = fixture[2] + (
        _claim(
            claim_id="claim-parent-assumption",
            kind="affected_assumption",
            statement="This affects the assumption that both sources update together.",
            observation_id=fixture[1][0].id,
        ),
        _claim(
            claim_id="claim-parent-uncertainty",
            kind="uncertainty",
            statement="Market and filing timestamps have different update cadences.",
            observation_id=fixture[1][0].id,
        ),
    )
    epistemic_invariant, _ = _delegated_contract(
        run=fixture[0],
        parent_events=_tool_path(fixture[1][0], "agent.execute_research_plan"),
        observations=fixture[1],
        claims=epistemic_claims,
        snapshot=fixture[3],
        child_tasks=fixture[4],
        child_runs=fixture[5],
        child_events=fixture[6],
        child_observations=fixture[7],
        child_claims=fixture[8],
        child_outbox=(),
    )
    assert len(epistemic_invariant) == 64

    foreign_epistemic = epistemic_claims[-1].__class__(
        **{**_row_values(epistemic_claims[-1]), "observation_ids": ["foreign-observation"]}
    )
    with pytest.raises(LiveProofIntegrityError, match="not grounded"):
        _delegated_contract(
            run=fixture[0],
            parent_events=_tool_path(fixture[1][0], "agent.execute_research_plan"),
            observations=fixture[1],
            claims=(*epistemic_claims[:-1], foreign_epistemic),
            snapshot=fixture[3],
            child_tasks=fixture[4],
            child_runs=fixture[5],
            child_events=fixture[6],
            child_observations=fixture[7],
            child_claims=fixture[8],
            child_outbox=(),
        )

    sequential_runs = (
        fixture[5][0],
        fixture[5][1].__class__(
            **{
                **_row_values(fixture[5][1]),
                "started_at": NOW + timedelta(seconds=9),
            }
        ),
    )
    with pytest.raises(LiveProofIntegrityError, match="did not overlap"):
        _delegated_contract(
            run=fixture[0],
            parent_events=_tool_path(fixture[1][0], "agent.execute_research_plan"),
            observations=fixture[1],
            claims=fixture[2],
            snapshot=fixture[3],
            child_tasks=fixture[4],
            child_runs=sequential_runs,
            child_events=fixture[6],
            child_observations=fixture[7],
            child_claims=fixture[8],
            child_outbox=(),
        )


def test_event_summary_reads_full_manifest_from_durable_context_event() -> None:
    manifest_digest = _hash("source-manifest")
    run = _run(tool_calls=None)
    run.event_sequence = 3
    rows = (
        _event(
            event_id="event-started",
            run_id=run.id,
            task_id=run.task_id,
            sequence=1,
            event_type=EventType.TASK_STARTED,
            payload={},
        ),
        _event(
            event_id="event-context",
            run_id=run.id,
            task_id=run.task_id,
            sequence=2,
            event_type=EventType.CONTEXT_BUILT,
            payload={
                "source_manifest": {
                    "schema_version": "context-source-manifest-v1",
                    "manifest_digest": manifest_digest,
                }
            },
        ),
        _event(
            event_id="event-completed",
            run_id=run.id,
            task_id=run.task_id,
            sequence=3,
            event_type=EventType.RUN_COMPLETED,
            payload={},
        ),
    )

    summary = _event_summary(_task(), run, rows)

    assert summary["context_manifest_digest"] == manifest_digest


def test_message_plane_binds_execution_recording_to_delivered_slack_receipt() -> None:
    ingress = _ingress()
    ingress.conversation_id = "conversation-1"
    ingress.prompt = "Please answer."
    ingress.context_access_hash = _hash("access")
    thread = ThreadRow(id="thread-1")
    run = _run(answer="A verified answer.", tool_calls=None)
    final = DeliveryOutboxRow(receipt_message_ts="1788000002.000002")

    def message(
        *,
        role: str,
        actor_id: str,
        text: str,
        provider_message_ts: str,
        external_event_id: str | None = None,
    ) -> SanitizedMessageRow:
        return SanitizedMessageRow(
            id=f"message-{role}-{provider_message_ts}",
            organization_id=SCOPE.organization_id,
            strategy_id=SCOPE.strategy_id,
            destination_id=ingress.channel_id,
            external_event_id=external_event_id or ingress.event_id,
            text=text,
            content_hash=_hash(text),
            recorded_at=NOW,
            conversation_id=ingress.conversation_id,
            harness_thread_id=thread.id,
            actor_id=actor_id,
            role=role,
            provider_message_ts=provider_message_ts,
            context_access_hash=ingress.context_access_hash,
        )

    user = message(
        role="user",
        actor_id=ingress.user_id,
        text=ingress.prompt,
        provider_message_ts=ingress.message_ts,
    )
    assistant = message(
        role="assistant",
        actor_id="leo",
        text=run.final_output or "",
        # The execution-time record is durably correlated by external_event_id;
        # the separate outbox receipt proves the later Slack response timestamp.
        provider_message_ts=ingress.message_ts,
    )

    assert len(_message_plane_digest(ingress, thread, run, final, (user, assistant))) == 64

    foreign = assistant.__class__(
        **{**_row_values(assistant), "external_event_id": "foreign-event"}
    )
    with pytest.raises(LiveProofIntegrityError, match="message plane is incomplete"):
        _message_plane_digest(ingress, thread, run, final, (user, foreign))

    progress_payload = "Leo is working on this."
    delivered_progress = DeliveryOutboxRow(
        id="delivery-progress",
        payload=progress_payload,
        payload_hash=_hash(progress_payload),
        receipt_message_ts="1788000001.000001",
    )
    rendered_final_payload = f"{run.final_output}\n\nRun: {run.id}"
    delivered_final = DeliveryOutboxRow(
        id="delivery-final",
        payload=rendered_final_payload,
        payload_hash=_hash(rendered_final_payload),
        receipt_message_ts="1788000002.000002",
    )
    progress_message = message(
        role="assistant",
        actor_id="leo",
        text=progress_payload,
        provider_message_ts=delivered_progress.receipt_message_ts or "",
        external_event_id=f"slack-delivery:{delivered_progress.id}",
    )
    final_message = message(
        role="assistant",
        actor_id="leo",
        text=rendered_final_payload,
        provider_message_ts=delivered_final.receipt_message_ts or "",
        external_event_id=f"slack-delivery:{delivered_final.id}",
    )
    assert (
        len(
            _message_plane_digest(
                ingress,
                thread,
                run,
                delivered_final,
                (user, progress_message, final_message),
                outbox_rows=(delivered_progress, delivered_final),
            )
        )
        == 64
    )
    with pytest.raises(LiveProofIntegrityError, match="message plane is incomplete"):
        _message_plane_digest(
            ingress,
            thread,
            run,
            delivered_final,
            (user, final_message),
            outbox_rows=(delivered_progress, delivered_final),
        )


@pytest.mark.parametrize("evidence_id", list(LiveEvidenceId))
def test_generic_completed_run_cannot_self_attest_any_live_case(
    evidence_id: LiveEvidenceId,
) -> None:
    binding = _binding(evidence_id)
    with pytest.raises(LiveProofIntegrityError):
        _case_contract_summary(
            authority=_authority(),
            binding=binding,
            ingress=_ingress(),
            task=_task(answer="Generic completion."),
            run=_run(answer="Generic completion.", tool_calls=None),
            event_rows=(),
            observations=(),
            claims=(),
            memory_records=(),
            memory_revisions=(),
            memory_sources=(),
            plan_snapshot=None,
            child_tasks=(),
            child_runs=(),
            child_events=(),
            child_observations=(),
            child_claims=(),
            child_outbox=(),
        )


def _row_values(row: object) -> dict[str, object]:
    return {key: value for key, value in vars(row).items() if not key.startswith("_sa_")}
