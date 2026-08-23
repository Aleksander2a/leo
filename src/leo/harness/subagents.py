"""Bounded child-agent delegation that inherits, but cannot broaden, parent authority."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from leo.harness.child_evidence import (
    ChildEvidenceEnvelope,
    ChildEvidenceError,
    build_child_evidence_envelope,
    child_evidence_data,
    child_evidence_expires_at,
    parse_child_evidence_envelope,
    serialize_child_evidence_envelope,
)
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.models import (
    BudgetLimits,
    CardinalityBounds,
    CompletionContract,
    ContextItem,
    ContextItemKind,
    EventDraft,
    EventType,
    EvidenceToolRequirement,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    OriginRef,
    Run,
    RunBundle,
    RunPhase,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.plan_models import (
    PlanNodeClaim,
    PlanNodeDefinition,
    PlanNodeStatus,
    PlanSnapshot,
    PlanStatus,
)
from leo.harness.ports import Clock, IdGenerator, ModelGateway, RunStore
from leo.harness.provider_canonical import (
    canonical_evidence_completion as canonical_evidence_completion,
)
from leo.harness.storage import InMemoryRunStore
from leo.harness.store_errors import NotFoundError, StoreError
from leo.harness.tools import ToolRegistry
from leo.harness.transitions import cancel_task_and_run
from leo.harness.verifier import DeterministicCompletionVerifier

# One child turn is a model call plus at most a couple of bounded reads. The
# child's wall clock is derived from its own turn count so the two budgets can
# never drift apart again, and is capped just under the delegating tool's own
# timeout so the parent observes a real child result rather than a tool timeout.
_CHILD_SECONDS_PER_TURN = 20.0
_CHILD_MAX_ELAPSED_SECONDS = 85.0


def _child_elapsed_budget_seconds(max_turns: int) -> float:
    return min(_CHILD_MAX_ELAPSED_SECONDS, max(30.0, max_turns * _CHILD_SECONDS_PER_TURN))


class _DelegateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective: str = Field(min_length=1, max_length=4_000)
    expected_output: str = Field(default="A concise evidence-aware answer.", max_length=500)
    max_turns: int = Field(default=4, ge=1, le=8)


class _PlanNode(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    objective: str = Field(min_length=1, max_length=4_000)
    expected_output: str = Field(default="A concise evidence-aware finding.", max_length=500)
    depends_on: tuple[str, ...] = Field(default=(), max_length=6)
    max_turns: int = Field(default=4, ge=1, le=8)


class _PlanArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(min_length=1, max_length=4_000)
    nodes: tuple[_PlanNode, ...] = Field(min_length=1, max_length=6)
    max_concurrency: int = Field(default=3, ge=1, le=4)

    @model_validator(mode="after")
    def valid_dag(self) -> _PlanArguments:
        ids = tuple(node.id for node in self.nodes)
        if len(ids) != len(set(ids)):
            raise ValueError("plan node IDs must be unique")
        known = set(ids)
        if any(dependency not in known for node in self.nodes for dependency in node.depends_on):
            raise ValueError("plan dependency is unknown")
        if any(node.id in node.depends_on for node in self.nodes):
            raise ValueError("plan node cannot depend on itself")
        remaining = {node.id: set(node.depends_on) for node in self.nodes}
        completed: set[str] = set()
        while remaining:
            ready = {
                node_id for node_id, dependencies in remaining.items() if dependencies <= completed
            }
            if not ready:
                raise ValueError("plan dependencies must be acyclic")
            completed.update(ready)
            for node_id in ready:
                del remaining[node_id]
        return self


class DurablePlanStore(Protocol):
    """Architecture-safe port implemented by ``PostgresPlanStore`` in live composition."""

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
    ) -> PlanSnapshot: ...

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
    ) -> PlanSnapshot: ...

    async def claim_ready_node(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        owner: str,
        lease_seconds: float = 60.0,
    ) -> PlanNodeClaim | None: ...

    async def attach_child(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        child_task_id: str,
        child_run_id: str,
    ) -> PlanSnapshot: ...

    async def complete_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        output: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
    ) -> PlanSnapshot: ...

    async def fail_node(
        self,
        *,
        scope: ScopeKey,
        claim: PlanNodeClaim,
        error: str,
        child_task_id: str | None = None,
        child_run_id: str | None = None,
    ) -> PlanSnapshot: ...

    async def finalize(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        status: PlanStatus,
        result: str,
    ) -> PlanSnapshot: ...

    async def cancel(
        self,
        *,
        scope: ScopeKey,
        plan_id: str,
        parent_task_id: str,
        parent_run_id: str,
        reason: str,
    ) -> PlanSnapshot: ...

    async def reload(self, *, scope: ScopeKey, plan_id: str) -> PlanSnapshot: ...


ChildReadyHook = Callable[[Task, Run], Awaitable[None]]
ChildRequirementSelector = Callable[[str], tuple[EvidenceToolRequirement, ...]]


class _EvidenceBoundChildGateway:
    """Finish constrained provider reads canonically inside the child harness.

    The delegated model still chooses and reasons through the tool route. Once every
    trusted evidence requirement has a fresh, eligible observation, a second provider
    call cannot add authority: the harness emits the one canonical proposal that the
    ordinary deterministic verifier and durable completion path must still accept.
    """

    def __init__(
        self,
        delegate: ModelGateway,
        requirements: tuple[EvidenceToolRequirement, ...],
        clock: Clock,
    ) -> None:
        self._delegate = delegate
        self._requirements = requirements
        self._clock = clock

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        canonical = canonical_evidence_completion(
            request.observations,
            self._requirements,
            now=self._clock.now(),
        )
        if canonical is None:
            return await self._delegate.decide(request)
        return ModelTurnResult(
            decision=canonical,
            provider="leo-child-harness",
            model="canonical-evidence-bound-v1",
            request_id=f"canonical-child-{request.iteration}",
            finish_reason="stop",
            usage=ModelUsage(),
        )


class SubagentResearchTool:
    """Expose one read-only child harness as a parent tool.

    The child receives the parent's immutable ``TrustedScope`` from the execution
    context, a preselected context projection, an independent budget, and only the
    explicitly supplied read-tool registry. It cannot deliver to Slack or commit an
    external effect. The parent remains responsible for synthesis and completion.
    """

    def __init__(
        self,
        *,
        model: ModelGateway,
        tools: ToolRegistry,
        context_items: tuple[ContextItem, ...],
        clock: Clock,
        ids: IdGenerator,
        run_store: RunStore | None = None,
        parent_task_id: str | None = None,
        requirement_selector: ChildRequirementSelector | None = None,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context_items = context_items
        self._clock = clock
        self._ids = ids
        self._run_store = run_store
        self._parent_task_id = parent_task_id
        self._requirement_selector = requirement_selector
        if run_store is not None and not parent_task_id:
            raise ValueError("durable subagent run store requires parent_task_id")
        self._spec = ToolSpec(
            name="agent.delegate_research",
            description=(
                "Delegate one bounded, read-only research subproblem to a child Leo agent. "
                "Use this for an independent subquestion in a complex request. The child inherits "
                "the current trusted scope, sees only preselected context and read tools, and "
                "returns evidence for parent synthesis; it cannot message users or perform writes."
            ),
            domain="HARNESS",
            input_schema=_DelegateArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=90.0,
            max_result_bytes=262_144,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = _DelegateArguments.model_validate(arguments)
        return parsed.model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = _DelegateArguments.model_validate(arguments)
        return await self._execute_parsed(parsed, context)

    async def execute_prepared(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
        *,
        child_ids: tuple[str, str, str],
        before_run: ChildReadyHook,
    ) -> ToolOutcome:
        """Run one preidentified durable child after its parent journal is attached."""

        parsed = _DelegateArguments.model_validate(arguments)
        return await self._execute_parsed(
            parsed,
            context,
            child_ids=child_ids,
            before_run=before_run,
        )

    async def _execute_parsed(
        self,
        parsed: _DelegateArguments,
        context: ToolExecutionContext,
        *,
        child_ids: tuple[str, str, str] | None = None,
        before_run: ChildReadyHook | None = None,
    ) -> ToolOutcome:
        scope = context.trusted_scope.namespace
        if any(
            item.source_scope is not None
            and item.source_scope.organization_id != scope.organization_id
            for item in self._context_items
        ):
            return ToolFailure(
                code="SUBAGENT_CONTEXT_SCOPE_MISMATCH",
                retryable=False,
                safe_message="Delegated context was outside the parent organization scope.",
            )
        if child_ids is None:
            child_ids = (
                _stable_child_ids(context.run_id, context.tool_call_id)
                if self._run_store is not None
                else (
                    self._ids.new("subthread"),
                    self._ids.new("subtask"),
                    self._ids.new("subrun"),
                )
            )
        thread_id, task_id, run_id = child_ids
        thread = Thread(
            id=thread_id,
            scope=scope,
            origin=OriginRef(
                provider="leo-subagent",
                external_thread_id=_bounded_external_ref(
                    context.run_id,
                    context.tool_call_id,
                ),
            ),
        )
        task = Task(
            id=task_id,
            thread_id=thread.id,
            scope=scope,
            objective=parsed.objective,
            parent_task_id=self._parent_task_id,
            continuation_kind="subagent" if self._parent_task_id is not None else "root",
        )
        run = Run(
            id=run_id,
            task_id=task.id,
            scope=scope,
            limits=BudgetLimits(
                max_iterations=parsed.max_turns,
                max_model_calls=parsed.max_turns,
                max_tool_calls=max(1, parsed.max_turns * 2),
                # Without this the child inherited the 60s BudgetLimits default
                # while the parent ran with 600s and this tool allowed 90s, so a
                # 4-turn child reliably self-timed-out mid-research and the
                # failure cascaded into the parent run. Derive it from the same
                # turn count that sizes every other child budget.
                max_elapsed_seconds=_child_elapsed_budget_seconds(parsed.max_turns),
            ),
        )
        store = self._run_store or InMemoryRunStore(self._clock, self._ids)
        bundle: RunBundle | None = None
        try:
            bundle = await _seed_or_load_child(store, thread, task, run)
            if before_run is not None:
                await before_run(bundle.task, bundle.run)
        except StoreError:
            if bundle is not None and before_run is not None:
                await _cancel_unattached_child(store, bundle)
            return ToolFailure(
                code="SUBAGENT_DURABLE_CONFLICT",
                retryable=False,
                safe_message="The durable child identity could not be replayed safely.",
            )
        evidence_requirements = (
            self._requirement_selector(parsed.objective)
            if self._requirement_selector is not None
            else ()
        )
        completion_contract = _child_completion_contract(
            evidence_requirements=evidence_requirements,
            expected_output=parsed.expected_output,
        )
        coordinator = RunCoordinator(
            store=store,
            model=_EvidenceBoundChildGateway(
                self._model,
                evidence_requirements,
                self._clock,
            ),
            tools=self._tools,
            context=DefaultContextAssembler(
                evidence_requirements=evidence_requirements,
                clock=self._clock,
                completion_contract=completion_contract,
                context_items=self._context_items,
            ),
            verifier=DeterministicCompletionVerifier(
                self._ids,
                self._clock,
                evidence_requirements=evidence_requirements,
                require_source_claim=bool(evidence_requirements),
            ),
            clock=self._clock,
            ids=self._ids,
        )
        result = await coordinator.run(
            task_id=task.id,
            run_id=run.id,
            trusted_scope=context.trusted_scope,
        )
        if result.run.status is not RunStatus.COMPLETED or result.run.final_output is None:
            return ToolFailure(
                code="SUBAGENT_INCOMPLETE",
                retryable=False,
                safe_message=(
                    "The delegated child stopped safely without a verified completion "
                    f"({result.run.status.value})."
                ),
            )
        try:
            evidence = build_child_evidence_envelope(
                child_run_id=result.run.id,
                answer=result.run.final_output,
                trace_event_count=len(result.events),
                observations=result.observations,
                claims=result.claims,
            )
        except (ChildEvidenceError, TypeError, ValueError):
            return ToolFailure(
                code="SUBAGENT_EVIDENCE_EXPORT_INVALID",
                retryable=False,
                safe_message="The child result could not be exported with verified provenance.",
            )
        return ToolSuccess(
            data=child_evidence_data(evidence),
            source=SourceRef(provider="leo-subagent", reference=result.run.id),
            observed_at=self._clock.now(),
            expires_at=child_evidence_expires_at(evidence),
        )


async def _cancel_unattached_child(store: RunStore, bundle: RunBundle) -> None:
    """Best-effort fence for a child seeded just before its plan attach lost authority."""

    if bundle.run.status not in {
        RunStatus.QUEUED,
        RunStatus.RUNNING,
        RunStatus.REQUIRES_ACTION,
    }:
        return
    task, run = cancel_task_and_run(
        bundle.task,
        bundle.run,
        "parent_plan_attach_rejected",
        usage=bundle.run.usage,
    )
    try:
        await store.commit(
            expected_task_version=bundle.task.version,
            expected_run_version=bundle.run.version,
            task=task,
            run=run,
            events=(
                EventDraft(
                    type=EventType.RUN_CANCELLED,
                    iteration=run.iteration,
                    payload={"reason": "parent_plan_attach_rejected"},
                ),
            ),
        )
    except StoreError:
        # A concurrent terminal winner is authoritative; this helper must never
        # overwrite it merely to improve orphan cleanup.
        return


def _child_completion_contract(
    *,
    evidence_requirements: tuple[EvidenceToolRequirement, ...],
    expected_output: str,
) -> CompletionContract:
    """Keep evidence-bound child proposals canonical before verification.

    The verifier remains the source of truth. Exact provider-schema cardinality prevents a
    child model from turning every field in one provider payload into a separate source claim,
    which otherwise creates deterministic verifier retry storms without adding evidence.
    """

    if not evidence_requirements:
        guidance = (
            f"Solve only this delegated subproblem. Expected output: {expected_output} "
            "Use read tools when needed and clearly distinguish observations from inference."
        )
        return CompletionContract(
            source_claim_count=CardinalityBounds(minimum=0, maximum=8),
            source_observation_id_count=CardinalityBounds(minimum=1, maximum=8),
            inference_count=CardinalityBounds(minimum=0, maximum=8),
            guidance=guidance[:500],
        )

    claim_count = len(evidence_requirements)
    kinds = {requirement.observation_kind for requirement in evidence_requirements}
    guidance_parts = [
        f"Return exactly {claim_count} SOURCE_CLAIMs: one per required evidence kind, no extras. ",
        "Each claim cites only its matching observation ID and appears verbatim in the answer. ",
    ]
    if "market.get_quote" in kinds:
        guidance_parts.append(
            "Quote claim: use only observation symbol and exact current price; do not claim "
            "change, high, low, open, or previous close. "
        )
    if "sec.get_recent_filings" in kinds:
        guidance_parts.append(
            "SEC claim: use filings[0] exactly as '<ticker> filed form <form> on "
            "<filing_date> under accession <accession>.' "
        )
    guidance_parts.append("Solve only the delegated subproblem; distinguish inference clearly.")
    guidance = "".join(guidance_parts)
    return CompletionContract(
        source_claim_count=CardinalityBounds(minimum=claim_count, maximum=claim_count),
        source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
        inference_count=CardinalityBounds(minimum=0, maximum=8),
        guidance=guidance[:500],
    )


class _DurablePlanExecutorMixin:
    _model: ModelGateway
    _tools: ToolRegistry
    _context_items: tuple[ContextItem, ...]
    _clock: Clock
    _ids: IdGenerator
    _run_store: RunStore | None
    _plan_store: DurablePlanStore | None
    _parent_task_id: str | None
    _parent_run_id: str | None
    _plan_owner: str
    _requirement_selector: ChildRequirementSelector | None
    _max_plan_revisions: int
    _claim_lease_seconds: float

    async def _execute_durable(
        self,
        plan: _PlanArguments,
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        plan_store = self._plan_store
        run_store = self._run_store
        parent_task_id = self._parent_task_id
        parent_run_id = self._parent_run_id
        if (
            plan_store is None
            or run_store is None
            or parent_task_id is None
            or parent_run_id is None
        ):
            raise RuntimeError("durable subagent plan was not fully composed")
        if context.run_id != parent_run_id:
            return ToolFailure(
                code="SUBAGENT_PLAN_PARENT_MISMATCH",
                retryable=False,
                safe_message="The durable plan was bound to a different parent run.",
            )
        scope = context.trusted_scope.namespace
        definitions = _initial_plan_definitions(plan)
        durable_goal = _durable_plan_goal(plan)
        try:
            snapshot = await plan_store.create_or_load(
                scope=scope,
                parent_task_id=parent_task_id,
                parent_run_id=parent_run_id,
                idempotency_key=_plan_idempotency_key(
                    parent_task_id,
                    parent_run_id,
                    context.tool_call_id,
                ),
                goal=durable_goal,
                nodes=definitions,
                max_revisions=self._max_plan_revisions,
            )
        except StoreError:
            return _durable_failure(
                "SUBAGENT_PLAN_PERSISTENCE_CONFLICT",
                "The durable plan could not be created or replayed safely.",
            )

        owner = _plan_execution_owner(self._plan_owner, snapshot.plan.id)
        max_rounds = self._max_plan_revisions * (len(plan.nodes) + 2)
        for _ in range(max_rounds):
            if snapshot.plan.status is not PlanStatus.ACTIVE:
                return _durable_plan_outcome(plan, snapshot, self._clock.now())
            if all(node.status is PlanNodeStatus.COMPLETED for node in snapshot.current_nodes):
                try:
                    snapshot = await plan_store.finalize(
                        scope=scope,
                        plan_id=snapshot.plan.id,
                        parent_task_id=parent_task_id,
                        parent_run_id=parent_run_id,
                        status=PlanStatus.COMPLETED,
                        result=(
                            f"Completed {len(plan.nodes)} delegated research nodes for: {plan.goal}"
                        ),
                    )
                except StoreError:
                    return _durable_failure(
                        "SUBAGENT_PLAN_FINALIZE_FAILED",
                        "The parent could not safely finalize the durable plan.",
                    )
                return _durable_plan_outcome(plan, snapshot, self._clock.now())

            claims = list(_recover_owned_claims(snapshot, owner, self._clock.now()))
            claim_failed = False
            while len(claims) < plan.max_concurrency:
                try:
                    claim = await plan_store.claim_ready_node(
                        scope=scope,
                        plan_id=snapshot.plan.id,
                        owner=owner,
                        lease_seconds=self._claim_lease_seconds,
                    )
                except StoreError:
                    claim_failed = True
                    break
                if claim is None:
                    break
                if all(existing.node_id != claim.node_id for existing in claims):
                    claims.append(claim)

            if claims:
                await asyncio.gather(
                    *(
                        self._execute_durable_claim(
                            plan=plan,
                            context=context,
                            snapshot=snapshot,
                            claim=claim,
                            plan_store=plan_store,
                            run_store=run_store,
                            parent_task_id=parent_task_id,
                        )
                        for claim in claims
                    )
                )
                try:
                    snapshot = await plan_store.reload(scope=scope, plan_id=snapshot.plan.id)
                except StoreError:
                    return _durable_failure(
                        "SUBAGENT_PLAN_REPLAY_FAILED",
                        "The durable plan could not be replayed after child execution.",
                    )
                continue

            try:
                snapshot = await plan_store.reload(scope=scope, plan_id=snapshot.plan.id)
            except StoreError:
                return _durable_failure(
                    "SUBAGENT_PLAN_REPLAY_FAILED",
                    "The durable plan could not be replayed after a no-progress check.",
                )
            if any(node.status is PlanNodeStatus.RUNNING for node in snapshot.current_nodes):
                return ToolFailure(
                    code="SUBAGENT_PLAN_BUSY",
                    retryable=True,
                    safe_message="Another durable child claim is still running.",
                )
            if claim_failed and _has_dependency_ready_node(snapshot):
                return _durable_failure(
                    "SUBAGENT_PLAN_CLAIM_FAILED",
                    "A dependency-ready durable child could not be claimed safely.",
                )
            if any(node.status is PlanNodeStatus.FAILED for node in snapshot.current_nodes):
                if snapshot.plan.current_revision < snapshot.plan.max_revisions:
                    try:
                        snapshot = await plan_store.append_revision(
                            scope=scope,
                            plan_id=snapshot.plan.id,
                            parent_task_id=parent_task_id,
                            parent_run_id=parent_run_id,
                            goal=_replan_goal(
                                durable_goal,
                                snapshot.plan.current_revision + 1,
                            ),
                            nodes=_replan_definitions(plan, snapshot),
                            reason="durable_child_no_progress",
                        )
                        continue
                    except StoreError:
                        pass
                try:
                    snapshot = await plan_store.finalize(
                        scope=scope,
                        plan_id=snapshot.plan.id,
                        parent_task_id=parent_task_id,
                        parent_run_id=parent_run_id,
                        status=PlanStatus.FAILED,
                        result="The bounded delegated research plan could not make progress.",
                    )
                except StoreError:
                    return _durable_failure(
                        "SUBAGENT_PLAN_FINALIZE_FAILED",
                        "The parent could not safely finalize the stalled durable plan.",
                    )
                return _durable_plan_outcome(plan, snapshot, self._clock.now())
            if claim_failed:
                return _durable_failure(
                    "SUBAGENT_PLAN_CLAIM_FAILED",
                    "The next durable child could not be claimed safely.",
                )
            return _durable_failure(
                "SUBAGENT_PLAN_STALLED",
                "The durable plan has unfinished work but no safe next action.",
            )

        return _durable_failure(
            "SUBAGENT_PLAN_ROUND_LIMIT",
            "The durable plan reached its bounded coordinator round limit.",
        )

    async def _execute_durable_claim(
        self,
        *,
        plan: _PlanArguments,
        context: ToolExecutionContext,
        snapshot: PlanSnapshot,
        claim: PlanNodeClaim,
        plan_store: DurablePlanStore,
        run_store: RunStore,
        parent_task_id: str,
    ) -> None:
        original_id = _original_node_id(claim.node_key)
        original = next((node for node in plan.nodes if node.id == original_id), None)
        if original is None:
            await _fail_claim_safely(
                plan_store,
                claim,
                "durable_plan_node_definition_missing",
            )
            return
        results = _aggregate_plan_results(plan, snapshot)
        dependency_items = tuple(
            ContextItem(
                id=f"subagent-result:{dependency}",
                kind=ContextItemKind.SUBAGENT_RESULT,
                content=str(results[dependency].get("answer") or "No answer."),
                conversation_id=(
                    self._context_items[0].conversation_id
                    if self._context_items
                    else context.run_id
                ),
                source_scope=claim.scope,
            )
            for dependency in original.depends_on
            if results.get(dependency, {}).get("status") == "completed"
        )
        delegate = SubagentResearchTool(
            model=self._model,
            tools=self._tools,
            context_items=(*self._context_items, *dependency_items),
            clock=self._clock,
            ids=self._ids,
            run_store=run_store,
            parent_task_id=parent_task_id,
            requirement_selector=self._requirement_selector,
        )
        node_state = next(node for node in snapshot.current_nodes if node.id == claim.node_id)
        stable_ids = _stable_child_ids(
            claim.plan_id,
            claim.node_id,
            str(claim.attempt),
        )
        if node_state.child_task_id is not None and node_state.child_run_id is not None:
            attached_thread_id = _stable_child_ids(
                claim.plan_id,
                claim.node_id,
                str(node_state.attempt),
            )[0]
            child_ids = (
                attached_thread_id,
                node_state.child_task_id,
                node_state.child_run_id,
            )
        else:
            child_ids = stable_ids
        child_context = context.model_copy(
            update={
                "trusted_scope": TrustedScope(
                    namespace=claim.scope,
                    actor_id=context.trusted_scope.actor_id,
                    roles=context.trusted_scope.roles,
                ),
                "tool_call_id": (
                    f"{context.tool_call_id}:{claim.revision_id}:{claim.node_key}:{claim.attempt}"
                ),
            }
        )

        async def attach(task: Task, run: Run) -> None:
            await plan_store.attach_child(
                scope=claim.scope,
                claim=claim,
                child_task_id=task.id,
                child_run_id=run.id,
            )

        try:
            outcome = await delegate.execute_prepared(
                delegate.validate(
                    {
                        "objective": original.objective,
                        "expected_output": original.expected_output,
                        "max_turns": original.max_turns,
                    }
                ),
                child_context,
                child_ids=child_ids,
                before_run=attach,
            )
            if isinstance(outcome, ToolSuccess):
                try:
                    evidence = parse_child_evidence_envelope(outcome.data)
                except ChildEvidenceError:
                    await plan_store.fail_node(
                        scope=claim.scope,
                        claim=claim,
                        error="child_evidence_malformed",
                    )
                    return
                if evidence.child_run_id != child_ids[2]:
                    await plan_store.fail_node(
                        scope=claim.scope,
                        claim=claim,
                        error="child_evidence_run_mismatch",
                    )
                    return
                await plan_store.complete_node(
                    scope=claim.scope,
                    claim=claim,
                    output=serialize_child_evidence_envelope(evidence),
                )
                return
            await plan_store.fail_node(
                scope=claim.scope,
                claim=claim,
                error=outcome.code,
            )
        except StoreError:
            await _fail_claim_safely(plan_store, claim, "durable_child_persistence_error")
        except Exception:
            await _fail_claim_safely(plan_store, claim, "durable_child_execution_error")


class SubagentPlanTool(_DurablePlanExecutorMixin):
    """Execute a bounded dependency plan using isolated read-only child agents."""

    def __init__(
        self,
        *,
        model: ModelGateway,
        tools: ToolRegistry,
        context_items: tuple[ContextItem, ...],
        clock: Clock,
        ids: IdGenerator,
        run_store: RunStore | None = None,
        plan_store: DurablePlanStore | None = None,
        parent_task_id: str | None = None,
        parent_run_id: str | None = None,
        plan_owner: str | None = None,
        requirement_selector: ChildRequirementSelector | None = None,
        max_plan_revisions: int = 3,
        claim_lease_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._tools = tools
        self._context_items = context_items
        self._clock = clock
        self._ids = ids
        self._run_store = run_store
        self._plan_store = plan_store
        self._parent_task_id = parent_task_id
        self._parent_run_id = parent_run_id
        self._plan_owner = plan_owner or (
            ids.new("plan-owner") if plan_store is not None else "leo-subagent-plan"
        )
        self._requirement_selector = requirement_selector
        self._max_plan_revisions = max_plan_revisions
        self._claim_lease_seconds = claim_lease_seconds
        self._durable_lock = asyncio.Lock()
        if plan_store is not None and (
            run_store is None or not parent_task_id or not parent_run_id
        ):
            raise ValueError(
                "durable subagent plan requires run_store, parent_task_id, and parent_run_id"
            )
        if run_store is not None and not parent_task_id:
            raise ValueError("durable child run store requires parent_task_id")
        if (
            not self._plan_owner
            or self._plan_owner != self._plan_owner.strip()
            or len(self._plan_owner) > 64
        ):
            raise ValueError("plan_owner must be a non-empty value of at most 64 characters")
        if not 1 <= max_plan_revisions <= 8:
            raise ValueError("max_plan_revisions must be between 1 and 8")
        if not 1 <= claim_lease_seconds <= 86_400:
            raise ValueError("claim_lease_seconds must be between 1 and 86400")
        self._spec = ToolSpec(
            name="agent.execute_research_plan",
            description=(
                "Execute a bounded read-only research plan with 1-6 child-agent nodes. "
                "Declare dependencies explicitly; independent nodes may run concurrently, "
                "dependent nodes receive prior findings, and the parent Leo agent retains final "
                "synthesis."
            ),
            domain="HARNESS",
            input_schema=_PlanArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=120.0,
            max_result_bytes=1_048_576,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _PlanArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        plan = _PlanArguments.model_validate(arguments)
        if self._plan_store is not None:
            async with self._durable_lock:
                return await self._execute_durable(plan, context)
        return await self._execute_ephemeral(plan, context)

    async def _execute_ephemeral(
        self,
        plan: _PlanArguments,
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        remaining = {node.id: node for node in plan.nodes}
        results: dict[str, dict[str, JsonValue]] = {}
        semaphore = asyncio.Semaphore(plan.max_concurrency)

        async def run_node(node: _PlanNode) -> tuple[str, dict[str, JsonValue]]:
            dependency_items = tuple(
                ContextItem(
                    id=f"subagent-result:{dependency}",
                    kind=ContextItemKind.SUBAGENT_RESULT,
                    content=str(results[dependency].get("answer") or "No answer."),
                    conversation_id=(
                        self._context_items[0].conversation_id
                        if self._context_items
                        else context.run_id
                    ),
                    source_scope=context.trusted_scope.namespace,
                )
                for dependency in node.depends_on
            )
            delegate = SubagentResearchTool(
                model=self._model,
                tools=self._tools,
                context_items=(*self._context_items, *dependency_items),
                clock=self._clock,
                ids=self._ids,
                run_store=self._run_store,
                parent_task_id=self._parent_task_id,
                requirement_selector=self._requirement_selector,
            )
            async with semaphore:
                outcome = await delegate.execute(
                    delegate.validate(
                        {
                            "objective": node.objective,
                            "expected_output": node.expected_output,
                            "max_turns": node.max_turns,
                        }
                    ),
                    context.model_copy(
                        update={"tool_call_id": f"{context.tool_call_id}:{node.id}"}
                    ),
                )
            if isinstance(outcome, ToolSuccess):
                try:
                    evidence = parse_child_evidence_envelope(outcome.data)
                except ChildEvidenceError:
                    return node.id, {
                        "id": node.id,
                        "status": "failed",
                        "error_code": "child_evidence_malformed",
                        "safe_message": "The child result lacked verified provenance metadata.",
                    }
                return node.id, _completed_plan_node_result(node.id, evidence)
            return node.id, {
                "id": node.id,
                "status": "failed",
                "error_code": outcome.code,
                "safe_message": outcome.safe_message,
            }

        while remaining:
            blocked = tuple(
                node
                for node in remaining.values()
                if any(
                    results.get(dependency, {}).get("status") != "completed"
                    for dependency in node.depends_on
                    if dependency in results
                )
            )
            for node in blocked:
                results[node.id] = {
                    "id": node.id,
                    "status": "blocked",
                    "safe_message": "A dependency did not complete.",
                }
                del remaining[node.id]
            ready = tuple(
                node
                for node in remaining.values()
                if all(
                    results.get(dependency, {}).get("status") == "completed"
                    for dependency in node.depends_on
                )
            )
            if not ready:
                if remaining:
                    return ToolFailure(
                        code="SUBAGENT_PLAN_STALLED",
                        retryable=False,
                        safe_message="The validated research plan made no progress.",
                    )
                break
            completed = await asyncio.gather(*(run_node(node) for node in ready))
            for node_id, result in completed:
                results[node_id] = result
                del remaining[node_id]

        ordered = [results[node.id] for node in plan.nodes]
        complete = all(item.get("status") == "completed" for item in ordered)
        return ToolSuccess(
            data={
                "goal": plan.goal,
                "status": "completed" if complete else "partial",
                "nodes": ordered,
                "completed_count": sum(item.get("status") == "completed" for item in ordered),
                "failed_count": sum(item.get("status") == "failed" for item in ordered),
                "blocked_count": sum(item.get("status") == "blocked" for item in ordered),
            },
            source=SourceRef(
                provider="leo-subagent-plan",
                reference=f"{context.run_id}:{context.tool_call_id}",
            ),
            observed_at=self._clock.now(),
            expires_at=_plan_results_expires_at(ordered),
        )


async def _seed_or_load_child(
    store: RunStore,
    thread: Thread,
    task: Task,
    run: Run,
) -> RunBundle:
    try:
        bundle = await store.load(task.id, run.id, task.scope)
    except NotFoundError:
        try:
            bundle = await store.seed(thread, task, run)
        except StoreError:
            bundle = await store.load(task.id, run.id, task.scope)
    if (
        bundle.thread.id != thread.id
        or bundle.task.id != task.id
        or bundle.run.id != run.id
        or bundle.task.objective != task.objective
        or bundle.task.parent_task_id != task.parent_task_id
        or bundle.task.continuation_kind != task.continuation_kind
        or bundle.run.limits != run.limits
    ):
        raise StoreError("durable child identity was reused with a different contract")
    return bundle


def _stable_child_ids(*parts: str) -> tuple[str, str, str]:
    basis = "\x1f".join(parts)

    def stable(prefix: str) -> str:
        digest = hashlib.sha256(f"{prefix}\x1f{basis}".encode()).hexdigest()
        return f"{prefix}-{digest[:48]}"

    return stable("subthread"), stable("subtask"), stable("subrun")


def _bounded_external_ref(*parts: str) -> str:
    value = ":".join(parts)
    if len(value) <= 255:
        return value
    return f"leo-subagent:{hashlib.sha256(value.encode()).hexdigest()}"


def _plan_idempotency_key(
    parent_task_id: str,
    parent_run_id: str,
    tool_call_id: str,
) -> str:
    value = "\x1f".join((parent_task_id, parent_run_id, tool_call_id))
    return f"subagent-plan:{hashlib.sha256(value.encode()).hexdigest()}"


def _plan_execution_owner(configured_owner: str, plan_id: str) -> str:
    suffix = hashlib.sha256(plan_id.encode()).hexdigest()[:24]
    return f"{configured_owner}:{suffix}"


def _durable_plan_goal(plan: _PlanArguments) -> str:
    serialized = plan.model_dump_json()
    contract_hash = hashlib.sha256(serialized.encode()).hexdigest()
    marker = f"[execution-contract:{contract_hash}]"
    available = 4_000 - len(marker) - 1
    return f"{marker}\n{plan.goal[:available]}"


def _initial_plan_definitions(
    plan: _PlanArguments,
) -> tuple[PlanNodeDefinition, ...]:
    return tuple(
        PlanNodeDefinition(
            key=node.id,
            objective=node.objective,
            depends_on=tuple(sorted(node.depends_on)),
            max_attempts=2,
        )
        for node in plan.nodes
    )


def _original_node_id(node_key: str) -> str:
    return node_key


def _recover_owned_claims(
    snapshot: PlanSnapshot,
    owner: str,
    now: datetime,
) -> tuple[PlanNodeClaim, ...]:
    claims: list[PlanNodeClaim] = []
    for node in snapshot.current_nodes:
        if (
            node.status is not PlanNodeStatus.RUNNING
            or node.claim_owner != owner
            or node.claim_token is None
            or node.lease_expires_at is None
            or node.lease_expires_at <= now
        ):
            continue
        claims.append(
            PlanNodeClaim(
                scope=snapshot.plan.scope,
                plan_id=snapshot.plan.id,
                revision_id=node.revision_id,
                node_id=node.id,
                node_key=node.definition.key,
                parent_task_id=snapshot.plan.parent_task_id,
                parent_run_id=snapshot.plan.parent_run_id,
                objective=node.definition.objective,
                depends_on=node.definition.depends_on,
                owner=owner,
                token=node.claim_token,
                attempt=node.attempt,
                expires_at=node.lease_expires_at,
            )
        )
    return tuple(claims)


def _has_dependency_ready_node(snapshot: PlanSnapshot) -> bool:
    completed = {
        node.definition.key
        for node in snapshot.current_nodes
        if node.status is PlanNodeStatus.COMPLETED
    }
    return any(
        node.status is PlanNodeStatus.PENDING
        and node.attempt < node.definition.max_attempts
        and set(node.definition.depends_on) <= completed
        for node in snapshot.current_nodes
    )


def _aggregate_plan_results(
    plan: _PlanArguments,
    snapshot: PlanSnapshot,
) -> dict[str, dict[str, JsonValue]]:
    results: dict[str, dict[str, JsonValue]] = {
        node.id: {"id": node.id, "status": "pending"} for node in plan.nodes
    }
    for persisted_node in sorted(snapshot.nodes, key=lambda item: (item.revision_number, item.id)):
        original_id = _original_node_id(persisted_node.definition.key)
        if original_id not in results:
            continue
        current = results[original_id]
        if (
            current.get("status") == "completed"
            and persisted_node.status is not PlanNodeStatus.COMPLETED
        ):
            continue
        item: dict[str, JsonValue] = {
            "id": original_id,
            "status": persisted_node.status.value,
        }
        if persisted_node.output is not None:
            try:
                evidence = parse_child_evidence_envelope(persisted_node.output)
            except ChildEvidenceError:
                # Legacy durable rows remain useful as inference context, but deliberately
                # receive no evidence envelope and therefore cannot ground SOURCE_CLAIM.
                item["answer"] = persisted_node.output
            else:
                if persisted_node.status is PlanNodeStatus.COMPLETED:
                    item.update(_completed_plan_node_result(original_id, evidence))
                else:
                    item["answer"] = evidence.answer
        if persisted_node.error is not None:
            item["error_code"] = persisted_node.error
        if persisted_node.child_run_id is not None:
            item["child_run_id"] = persisted_node.child_run_id
        results[original_id] = item
    for planned_node in plan.nodes:
        if results[planned_node.id].get("status") != "pending":
            continue
        if any(
            results[dependency].get("status") == "failed" for dependency in planned_node.depends_on
        ):
            results[planned_node.id] = {
                "id": planned_node.id,
                "status": "blocked",
                "safe_message": "A durable dependency did not complete.",
            }
    return results


def _replan_definitions(
    plan: _PlanArguments,
    snapshot: PlanSnapshot,
) -> tuple[PlanNodeDefinition, ...]:
    results = _aggregate_plan_results(plan, snapshot)
    incomplete = {node.id for node in plan.nodes if results[node.id].get("status") != "completed"}
    definitions = tuple(
        PlanNodeDefinition(
            key=node.id,
            objective=node.objective,
            depends_on=tuple(
                sorted(dependency for dependency in node.depends_on if dependency in incomplete)
            ),
            max_attempts=2,
        )
        for node in plan.nodes
        if node.id in incomplete
    )
    if not definitions:
        raise StoreError("replan requested without incomplete nodes")
    return definitions


def _replan_goal(durable_goal: str, revision: int) -> str:
    marker = f"[replan-revision:{revision}]"
    available = 4_000 - len(marker) - 1
    return f"{marker}\n{durable_goal[:available]}"


async def _fail_claim_safely(
    plan_store: DurablePlanStore,
    claim: PlanNodeClaim,
    error: str,
) -> None:
    try:
        await plan_store.fail_node(scope=claim.scope, claim=claim, error=error)
    except StoreError:
        return


def _durable_failure(code: str, message: str) -> ToolFailure:
    return ToolFailure(code=code, retryable=False, safe_message=message)


def _durable_plan_outcome(
    plan: _PlanArguments,
    snapshot: PlanSnapshot,
    observed_at: datetime,
) -> ToolOutcome:
    results = _aggregate_plan_results(plan, snapshot)
    ordered = [results[node.id] for node in plan.nodes]
    return ToolSuccess(
        data={
            "goal": plan.goal,
            "status": ("completed" if snapshot.plan.status is PlanStatus.COMPLETED else "partial"),
            "plan_id": snapshot.plan.id,
            "revision": snapshot.plan.current_revision,
            "nodes": ordered,
            "completed_count": sum(item.get("status") == "completed" for item in ordered),
            "failed_count": sum(item.get("status") == "failed" for item in ordered),
            "blocked_count": sum(item.get("status") == "blocked" for item in ordered),
        },
        source=SourceRef(provider="leo-subagent-plan", reference=snapshot.plan.id),
        observed_at=observed_at,
        expires_at=_plan_results_expires_at(ordered),
    )


def _completed_plan_node_result(
    node_id: str,
    evidence: ChildEvidenceEnvelope,
) -> dict[str, JsonValue]:
    return {
        "id": node_id,
        "status": "completed",
        "answer": evidence.answer,
        "child_run_id": evidence.child_run_id,
        "trace_event_count": evidence.trace_event_count,
        "observation_count": evidence.observation_count,
        "child_evidence": child_evidence_data(evidence),
    }


def _plan_results_expires_at(
    results: list[dict[str, JsonValue]],
) -> datetime | None:
    expiries: list[datetime] = []
    for item in results:
        raw_evidence = item.get("child_evidence")
        if raw_evidence is None:
            continue
        try:
            evidence = parse_child_evidence_envelope(raw_evidence)
        except ChildEvidenceError:
            continue
        expiry = child_evidence_expires_at(evidence)
        if expiry is not None:
            expiries.append(expiry)
    return min(expiries) if expiries else None
