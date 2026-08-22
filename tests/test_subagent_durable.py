from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import JsonValue, ValidationError

from leo.harness.child_evidence import parse_child_evidence_envelope
from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EvidenceToolRequirement,
    ModelRequest,
    ModelTurnResult,
    RunPhase,
    RunStatus,
    ScopeKey,
    SourceRef,
    ToolArgumentConstraint,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolRequest,
    ToolRequests,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.plan_models import (
    Delegation,
    DelegationStatus,
    Plan,
    PlanNode,
    PlanNodeClaim,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanRevision,
    PlanSnapshot,
    PlanStatus,
    revision_digest,
)
from leo.harness.ports import ModelGatewayError
from leo.harness.storage import InMemoryRunStore
from leo.harness.store_errors import StoreError
from leo.harness.subagents import (
    SubagentPlanTool,
    SubagentResearchTool,
    _child_completion_contract,
    _stable_child_ids,
)
from leo.harness.tools import ToolRegistry
from leo.integrations.fake import (
    FabricatingModel,
    FakeQuoteTool,
    FixedClock,
    ScriptedQuoteModel,
    SequentialIdGenerator,
)
from leo.live import _child_evidence_requirements

SCOPE = ScopeKey(organization_id="org-durable", strategy_id="strategy-provenance")
NOW = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def test_evidence_bound_child_contract_prevents_extra_provider_claims() -> None:
    contract = _child_completion_contract(
        evidence_requirements=(
            EvidenceToolRequirement(
                observation_kind="market.get_quote",
                tool_name="market.get_quote",
                required_arguments=(ToolArgumentConstraint(name="symbol", value="NVDA"),),
            ),
            EvidenceToolRequirement(
                observation_kind="sec.get_recent_filings",
                tool_name="sec.get_recent_filings",
                required_arguments=(ToolArgumentConstraint(name="ticker", value="NVDA"),),
            ),
        ),
        expected_output="A bounded comparison.",
    )

    assert contract.source_claim_count.minimum == contract.source_claim_count.maximum == 2
    assert (
        contract.source_observation_id_count.minimum
        == contract.source_observation_id_count.maximum
        == 1
    )
    assert "no extras" in contract.guidance
    assert "filings[0]" in contract.guidance
    assert "previous close" in contract.guidance
    assert len(contract.guidance) <= 500


class _CountingCompletionModel:
    def __init__(self, *, fail_first: bool = False) -> None:
        self.calls = 0
        self.fail_first = fail_first
        self.active = 0
        self.max_active = 0

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if self.fail_first and self.calls == 1:
                raise ModelGatewayError("synthetic child crash")
            return ModelTurnResult(
                decision=CompletionProposal(answer=f"Completed: {request.objective}", claims=()),
                provider="fixture",
                model="fixture-model",
            )
        finally:
            self.active -= 1


class _CountingQuoteModel:
    def __init__(self) -> None:
        self.calls = 0
        self._delegate = ScriptedQuoteModel()

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        return await self._delegate.decide(request)


class _SecRouteModel:
    def __init__(self, *, complete_if_called_again: bool = False) -> None:
        self.calls = 0
        self.complete_if_called_again = complete_if_called_again

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self.calls += 1
        if not request.observations:
            decision = ToolRequests(
                calls=(
                    ToolRequest(
                        id="child-sec-route",
                        name="sec.get_recent_filings",
                        arguments={"ticker": "NVDA"},
                    ),
                )
            )
        elif self.complete_if_called_again:
            observation = request.observations[-1]
            statement = "NVDA filed form 8-K on 2026-08-17 under accession 0001045810-26-000069."
            decision = CompletionProposal(
                answer=statement,
                claims=(
                    CandidateClaim(
                        kind=ClaimKind.SOURCE_CLAIM,
                        statement=statement,
                        observation_ids=(observation.id,),
                    ),
                ),
            )
        else:
            raise AssertionError("fresh constrained evidence must use the canonical child path")
        return ModelTurnResult(
            decision=decision,
            provider="fixture",
            model="sec-route-model",
        )


class _FixedSecTool:
    def __init__(self, clock: FixedClock, *, expired: bool = False) -> None:
        self.clock = clock
        self.expired = expired
        self.calls = 0
        self._spec = ToolSpec(
            name="sec.get_recent_filings",
            description="Return one fixed SEC filing.",
            domain="SEC",
            input_schema={
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if arguments.get("ticker") != "NVDA":
            raise ValueError("fixture only permits NVDA")
        return {"ticker": "NVDA", "limit": 1}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del arguments, context
        self.calls += 1
        observed_at = self.clock.now() - (timedelta(minutes=2) if self.expired else timedelta())
        expires_at = (
            self.clock.now() - timedelta(minutes=1)
            if self.expired
            else self.clock.now() + timedelta(minutes=15)
        )
        return ToolSuccess(
            data={
                "ticker": "NVDA",
                "cik": "0001045810",
                "filings": [
                    {
                        "form": "8-K",
                        "accession": "0001045810-26-000069",
                        "filing_date": "2026-08-17",
                        "primary_document": "nvda-20260817.htm",
                        "filing_url": (
                            "https://www.sec.gov/Archives/edgar/data/1045810/"
                            "000104581026000069/nvda-20260817.htm"
                        ),
                    }
                ],
            },
            source=SourceRef(
                provider="sec-edgar",
                reference="submissions:0001045810",
                url="https://data.sec.gov/submissions/CIK0001045810.json",
            ),
            observed_at=observed_at,
            expires_at=expires_at,
        )


def _sec_requirement_selector(objective: str) -> tuple[EvidenceToolRequirement, ...]:
    assert "document URL" in objective
    assert "exact tool statement" in objective
    return _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"sec.get_recent_filings"}),
    )


@pytest.mark.asyncio
async def test_ephemeral_child_exports_only_harness_verified_source_claims() -> None:
    clock = FixedClock(NOW)
    quote = FakeQuoteTool(clock)
    tool = SubagentResearchTool(
        model=ScriptedQuoteModel(),
        tools=ToolRegistry((quote,)),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
    )

    outcome = await tool.execute(
        tool.validate({"objective": "Get one grounded quote.", "max_turns": 3}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="grounded-child",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    evidence = parse_child_evidence_envelope(outcome.data)
    assert evidence.answer == "NVDA is quoted at 181.25 USD."
    assert len(evidence.verified_source_claims) == 1
    assert evidence.verified_source_claims[0].sources[0].kind == "market.get_quote"
    assert outcome.expires_at == clock.now() + timedelta(minutes=5)


@pytest.mark.asyncio
async def test_sec_child_canonicalizes_fresh_constrained_evidence_without_second_model_call() -> (
    None
):
    clock = FixedClock(NOW)
    model = _SecRouteModel()
    sec = _FixedSecTool(clock)
    tool = SubagentResearchTool(
        model=model,
        tools=ToolRegistry((sec,)),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
        requirement_selector=_sec_requirement_selector,
    )

    outcome = await tool.execute(
        tool.validate(
            {
                "objective": (
                    "Inspect the NVDA SEC document URL and return the exact tool statement."
                ),
                "expected_output": (
                    "One canonical source statement including the exact SEC document URL."
                ),
                "max_turns": 3,
            }
        ),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="sec-canonical-child",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert model.calls == 1
    assert sec.calls == 1
    evidence = parse_child_evidence_envelope(outcome.data)
    expected = (
        "NVDA filed form 8-K on 2026-08-17 under accession 0001045810-26-000069. "
        "Document URL: https://www.sec.gov/Archives/edgar/data/1045810/"
        "000104581026000069/nvda-20260817.htm"
    )
    assert evidence.answer == expected
    assert tuple(claim.statement for claim in evidence.verified_source_claims) == (expected,)
    assert evidence.verified_source_claims[0].sources[0].kind == "sec.get_recent_filings"
    assert outcome.expires_at == NOW + timedelta(minutes=15)


@pytest.mark.asyncio
async def test_canonical_child_never_accepts_expired_provider_evidence() -> None:
    clock = FixedClock(NOW)
    model = _SecRouteModel(complete_if_called_again=True)
    sec = _FixedSecTool(clock, expired=True)
    tool = SubagentResearchTool(
        model=model,
        tools=ToolRegistry((sec,)),
        context_items=(),
        clock=clock,
        ids=SequentialIdGenerator(),
        requirement_selector=_sec_requirement_selector,
    )

    outcome = await tool.execute(
        tool.validate(
            {
                "objective": (
                    "Inspect the NVDA SEC document URL and return the exact tool statement."
                ),
                "max_turns": 2,
            }
        ),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="expired-sec-child",
        ),
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "SUBAGENT_INCOMPLETE"
    assert model.calls == 2
    assert sec.calls == 1


@pytest.mark.asyncio
async def test_unverified_child_claim_never_produces_parent_evidence() -> None:
    tool = SubagentResearchTool(
        model=FabricatingModel(),
        tools=ToolRegistry(()),
        context_items=(),
        clock=FixedClock(NOW),
        ids=SequentialIdGenerator(),
    )

    outcome = await tool.execute(
        tool.validate({"objective": "Attempt unsupported attestation.", "max_turns": 2}),
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="fabricated-child",
        ),
    )

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "SUBAGENT_INCOMPLETE"


@pytest.mark.asyncio
async def test_durable_research_child_replays_without_another_model_call() -> None:
    clock = FixedClock()
    ids = SequentialIdGenerator()
    model = _CountingCompletionModel()
    tool = SubagentResearchTool(
        model=model,
        tools=ToolRegistry(()),
        context_items=(),
        clock=clock,
        ids=ids,
        run_store=InMemoryRunStore(clock, ids),
        parent_task_id="parent-task",
    )
    arguments = tool.validate({"objective": "Durably answer one subquestion."})
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="parent-run",
        tool_call_id="delegate-durable",
    )

    first = await tool.execute(arguments, context)
    second = await tool.execute(arguments, context)

    assert isinstance(first, ToolSuccess)
    assert second == first
    assert model.calls == 1
    assert str(first.data["child_run_id"]).startswith("subrun-")
    assert first.data["schema_version"] == "child-evidence-v1"
    assert first.data["verified_source_claims"] == []
    with pytest.raises(ValidationError):
        tool.validate(
            {
                "objective": "Model arguments cannot choose persistence.",
                "persistence": "memory",
            }
        )


class _MemoryPlanJournal:
    """Small executable test double for the durable-plan port used by the harness."""

    def __init__(self, clock: FixedClock) -> None:
        self.clock = clock
        self.snapshot: PlanSnapshot | None = None
        self.attach_count = 0
        self.replan_count = 0

    async def create_or_load(
        self,
        *,
        scope: ScopeKey,
        parent_task_id: str,
        parent_run_id: str,
        idempotency_key: str,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        max_revisions: int = 4,
    ) -> PlanSnapshot:
        if self.snapshot is not None:
            return self.snapshot
        revision = self._revision(
            plan_id="durable-plan",
            number=1,
            goal=goal,
            nodes=nodes,
            parent=None,
        )
        plan = Plan(
            id="durable-plan",
            scope=scope,
            parent_task_id=parent_task_id,
            parent_run_id=parent_run_id,
            idempotency_key=idempotency_key,
            initial_digest=revision.digest,
            max_revisions=max_revisions,
            created_at=self.clock.now(),
            updated_at=self.clock.now(),
        )
        self.snapshot = PlanSnapshot(
            plan=plan,
            revisions=(revision,),
            nodes=self._pending_nodes(revision),
            delegations=(),
        )
        return self.snapshot

    async def append_revision(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        reason: str,
    ) -> PlanSnapshot:
        del scope, plan_id, parent_task_id, parent_run_id, reason
        current = self._required()
        parent = current.revisions[-1]
        revision = self._revision(
            plan_id=current.plan.id,
            number=parent.number + 1,
            goal=goal,
            nodes=nodes,
            parent=parent,
        )
        self.replan_count += 1
        self.snapshot = PlanSnapshot(
            plan=current.plan.model_copy(
                update={
                    "current_revision": revision.number,
                    "version": current.plan.version + 1,
                    "updated_at": self.clock.now(),
                }
            ),
            revisions=(*current.revisions, revision),
            nodes=(*current.nodes, *self._pending_nodes(revision)),
            delegations=current.delegations,
        )
        return self.snapshot

    async def claim_ready_node(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        owner: str,
        lease_seconds: float = 60.0,
    ) -> PlanNodeClaim | None:
        del scope, plan_id
        current = self._required()
        completed = {
            node.definition.key
            for node in current.current_nodes
            if node.status is PlanNodeStatus.COMPLETED
        }
        candidate = next(
            (
                node
                for node in current.current_nodes
                if node.status is PlanNodeStatus.PENDING
                and set(node.definition.depends_on) <= completed
            ),
            None,
        )
        if candidate is None:
            if any(node.status is PlanNodeStatus.RUNNING for node in current.current_nodes):
                return None
            if all(node.status is PlanNodeStatus.COMPLETED for node in current.current_nodes):
                return None
            raise StoreError("no progress")
        token = f"claim:{candidate.id}:{candidate.attempt + 1}"
        expiry = self.clock.now() + timedelta(seconds=lease_seconds)
        claimed = candidate.model_copy(
            update={
                "status": PlanNodeStatus.RUNNING,
                "attempt": candidate.attempt + 1,
                "claim_owner": owner,
                "claim_token": token,
                "lease_expires_at": expiry,
                "updated_at": self.clock.now(),
            }
        )
        delegation = Delegation(
            id=f"delegation:{claimed.id}:{claimed.attempt}",
            plan_id=current.plan.id,
            revision_id=claimed.revision_id,
            node_id=claimed.id,
            parent_task_id=current.plan.parent_task_id,
            parent_run_id=current.plan.parent_run_id,
            attempt=claimed.attempt,
            owner=owner,
            claim_token=token,
            status=DelegationStatus.RUNNING,
            created_at=self.clock.now(),
        )
        self._replace(node=claimed, delegation=delegation)
        return PlanNodeClaim(
            scope=current.plan.scope,
            plan_id=current.plan.id,
            revision_id=claimed.revision_id,
            node_id=claimed.id,
            node_key=claimed.definition.key,
            parent_task_id=current.plan.parent_task_id,
            parent_run_id=current.plan.parent_run_id,
            objective=claimed.definition.objective,
            depends_on=claimed.definition.depends_on,
            owner=owner,
            token=token,
            attempt=claimed.attempt,
            expires_at=expiry,
        )

    async def attach_child(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        child_task_id: str,
        child_run_id: str,
    ) -> PlanSnapshot:
        del scope
        node, delegation = self._claimed(claim)
        self.attach_count += 1
        self._replace(
            node=node.model_copy(
                update={"child_task_id": child_task_id, "child_run_id": child_run_id}
            ),
            delegation=delegation.model_copy(
                update={"child_task_id": child_task_id, "child_run_id": child_run_id}
            ),
        )
        return self._required()

    async def complete_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        output: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
    ) -> PlanSnapshot:
        del scope, child_task_id, child_run_id
        node, delegation = self._claimed(claim)
        self._replace(
            node=node.model_copy(
                update={
                    "status": PlanNodeStatus.COMPLETED,
                    "claim_owner": None,
                    "claim_token": None,
                    "lease_expires_at": None,
                    "output": output,
                    "updated_at": self.clock.now(),
                }
            ),
            delegation=delegation.model_copy(
                update={
                    "status": DelegationStatus.COMPLETED,
                    "output": output,
                    "finished_at": self.clock.now(),
                }
            ),
        )
        return self._required()

    async def fail_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        error: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
    ) -> PlanSnapshot:
        del scope, child_task_id, child_run_id
        node, delegation = self._claimed(claim)
        self._replace(
            node=node.model_copy(
                update={
                    "status": PlanNodeStatus.FAILED,
                    "claim_owner": None,
                    "claim_token": None,
                    "lease_expires_at": None,
                    "error": error,
                    "updated_at": self.clock.now(),
                }
            ),
            delegation=delegation.model_copy(
                update={
                    "status": DelegationStatus.FAILED,
                    "error": error,
                    "finished_at": self.clock.now(),
                }
            ),
        )
        return self._required()

    async def finalize(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        status: PlanStatus,
        result: str,
    ) -> PlanSnapshot:
        del scope, plan_id
        current = self._required()
        if (
            parent_task_id != current.plan.parent_task_id
            or parent_run_id != current.plan.parent_run_id
        ):
            raise StoreError("parent mismatch")
        self.snapshot = current.model_copy(
            update={
                "plan": current.plan.model_copy(
                    update={
                        "status": status,
                        "output": result if status is PlanStatus.COMPLETED else None,
                        "error": result if status is PlanStatus.FAILED else None,
                        "version": current.plan.version + 1,
                        "updated_at": self.clock.now(),
                    }
                )
            }
        )
        return self.snapshot

    async def reload(self, *, scope: ScopeKey, plan_id: str) -> PlanSnapshot:
        del scope, plan_id
        return self._required()

    def _revision(
        self,
        *,
        plan_id: str,
        number: int,
        goal: str,
        nodes: tuple[PlanNodeDefinition, ...],
        parent: PlanRevision | None,
    ) -> PlanRevision:
        return PlanRevision(
            id=f"revision-{number}",
            plan_id=plan_id,
            number=number,
            goal=goal,
            nodes=nodes,
            digest=revision_digest(goal, nodes),
            parent_revision_id=parent.id if parent else None,
            parent_digest=parent.digest if parent else None,
            reason="initial" if parent is None else "retry",
            created_at=self.clock.now(),
        )

    def _pending_nodes(self, revision: PlanRevision) -> tuple[PlanNode, ...]:
        return tuple(
            PlanNode(
                id=f"node:{revision.number}:{definition.key}",
                plan_id=revision.plan_id,
                revision_id=revision.id,
                revision_number=revision.number,
                definition=definition,
                created_at=self.clock.now(),
                updated_at=self.clock.now(),
            )
            for definition in revision.nodes
        )

    def _claimed(self, claim: PlanNodeClaim) -> tuple[PlanNode, Delegation]:
        current = self._required()
        node = next(item for item in current.nodes if item.id == claim.node_id)
        delegation = next(
            item
            for item in current.delegations
            if item.node_id == claim.node_id and item.claim_token == claim.token
        )
        if node.claim_token != claim.token or node.status is not PlanNodeStatus.RUNNING:
            raise StoreError("stale claim")
        return node, delegation

    def _replace(self, *, node: PlanNode, delegation: Delegation) -> None:
        current = self._required()
        nodes = tuple(node if item.id == node.id else item for item in current.nodes)
        existing = {item.id for item in current.delegations}
        delegations = tuple(
            delegation if item.id == delegation.id else item for item in current.delegations
        )
        if delegation.id not in existing:
            delegations = (*delegations, delegation)
        self.snapshot = PlanSnapshot(
            plan=current.plan,
            revisions=current.revisions,
            nodes=nodes,
            delegations=delegations,
        )

    def _required(self) -> PlanSnapshot:
        if self.snapshot is None:
            raise StoreError("plan not created")
        return self.snapshot


class _RejectingAttachPlanJournal(_MemoryPlanJournal):
    async def attach_child(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        child_task_id: str,
        child_run_id: str,
    ) -> PlanSnapshot:
        del scope, claim, child_task_id, child_run_id
        raise StoreError("parent authority became terminal before child attach")


@pytest.mark.asyncio
async def test_rejected_plan_attach_terminalizes_the_seeded_child() -> None:
    clock = FixedClock(NOW)
    ids = SequentialIdGenerator()
    children = InMemoryRunStore(clock, ids)
    journal = _RejectingAttachPlanJournal(clock)
    tool = SubagentPlanTool(
        model=_CountingCompletionModel(),
        tools=ToolRegistry(()),
        context_items=(),
        clock=clock,
        ids=ids,
        run_store=children,
        plan_store=journal,
        parent_task_id="parent-task",
        parent_run_id="parent-run",
        max_plan_revisions=1,
    )
    arguments = tool.validate(
        {
            "goal": "Reject stale child attachment.",
            "nodes": [{"id": "one", "objective": "Must not continue."}],
        }
    )

    outcome = await tool.execute(
        arguments,
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="cancelled-parent-plan",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["status"] == "partial"
    node = journal._required().current_nodes[0]
    _thread_id, child_task_id, child_run_id = _stable_child_ids(
        node.plan_id,
        node.id,
        str(node.attempt),
    )
    child = await children.load(child_task_id, child_run_id, SCOPE)
    assert child.run.status is RunStatus.CANCELLED
    assert child.run.terminal_reason == "parent_plan_attach_rejected"


@pytest.mark.asyncio
async def test_durable_plan_claims_attaches_resumes_and_parent_finalizes() -> None:
    clock = FixedClock(NOW)
    ids = SequentialIdGenerator()
    model = _CountingCompletionModel()
    journal = _MemoryPlanJournal(clock)
    tool = SubagentPlanTool(
        model=model,
        tools=ToolRegistry(()),
        context_items=(),
        clock=clock,
        ids=ids,
        run_store=InMemoryRunStore(clock, ids),
        plan_store=journal,
        parent_task_id="parent-task",
        parent_run_id="parent-run",
    )
    arguments: dict[str, JsonValue] = tool.validate(
        {
            "goal": "Research then synthesize.",
            "max_concurrency": 2,
            "nodes": [
                {"id": "a", "objective": "Research A."},
                {"id": "b", "objective": "Research B."},
                {
                    "id": "synthesis",
                    "objective": "Synthesize.",
                    "depends_on": ["a", "b"],
                },
            ],
        }
    )
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="parent-run",
        tool_call_id="durable-plan-call",
    )

    first = await tool.execute(arguments, context)
    calls_after_first = model.calls
    second = await tool.execute(arguments, context)

    assert isinstance(first, ToolSuccess)
    assert first.data["status"] == "completed"
    assert first.data["completed_count"] == 3
    assert second == first
    assert model.calls == calls_after_first == 3
    assert model.max_active == 2
    assert journal.attach_count == 3
    assert journal._required().plan.status is PlanStatus.COMPLETED
    assert all(node.child_run_id for node in journal._required().current_nodes)
    persisted_evidence = tuple(
        parse_child_evidence_envelope(node.output) for node in journal._required().current_nodes
    )
    assert all(not envelope.verified_source_claims for envelope in persisted_evidence)
    assert all(
        node["child_evidence"]["digest"] == envelope.digest
        for node, envelope in zip(first.data["nodes"], persisted_evidence, strict=True)
    )


@pytest.mark.asyncio
async def test_durable_plan_replay_preserves_verified_child_evidence() -> None:
    clock = FixedClock(NOW)
    ids = SequentialIdGenerator()
    model = _CountingQuoteModel()
    quote = FakeQuoteTool(clock)
    journal = _MemoryPlanJournal(clock)
    tool = SubagentPlanTool(
        model=model,
        tools=ToolRegistry((quote,)),
        context_items=(),
        clock=clock,
        ids=ids,
        run_store=InMemoryRunStore(clock, ids),
        plan_store=journal,
        parent_task_id="parent-task",
        parent_run_id="parent-run",
    )
    arguments = tool.validate(
        {
            "goal": "Persist one grounded child result.",
            "nodes": [{"id": "quote", "objective": "Get the current quote."}],
        }
    )
    context = ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="parent-run",
        tool_call_id="durable-grounded-plan",
    )

    first = await tool.execute(arguments, context)
    second = await tool.execute(arguments, context)

    assert isinstance(first, ToolSuccess)
    assert second == first
    assert model.calls == 2
    assert quote.calls == 1
    persisted = parse_child_evidence_envelope(journal._required().current_nodes[0].output)
    assert len(persisted.verified_source_claims) == 1
    exported = first.data["nodes"][0]["child_evidence"]
    assert parse_child_evidence_envelope(exported) == persisted


@pytest.mark.asyncio
async def test_durable_plan_replans_once_after_child_failure() -> None:
    clock = FixedClock(NOW)
    ids = SequentialIdGenerator()
    model = _CountingCompletionModel(fail_first=True)
    journal = _MemoryPlanJournal(clock)
    tool = SubagentPlanTool(
        model=model,
        tools=ToolRegistry(()),
        context_items=(),
        clock=clock,
        ids=ids,
        run_store=InMemoryRunStore(clock, ids),
        plan_store=journal,
        parent_task_id="parent-task",
        parent_run_id="parent-run",
        max_plan_revisions=2,
    )
    arguments = tool.validate(
        {"goal": "Retry bounded work.", "nodes": [{"id": "only", "objective": "Work."}]}
    )

    outcome = await tool.execute(
        arguments,
        ToolExecutionContext(
            trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
            run_id="parent-run",
            tool_call_id="replan-call",
        ),
    )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["status"] == "completed"
    assert journal.replan_count == 1
    assert journal._required().plan.current_revision == 2
