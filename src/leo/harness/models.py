"""Provider-neutral contracts controlled by Leo's harness."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

NonEmptyStr = Annotated[str, Field(min_length=1, pattern=r"\S")]


class ContractModel(BaseModel):
    """Strict immutable base for state crossing a harness boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ScopeKey(ContractModel):
    organization_id: NonEmptyStr
    strategy_id: NonEmptyStr


class TrustedScope(ContractModel):
    """Server-derived request authority; never supplied by a model action."""

    namespace: ScopeKey
    actor_id: NonEmptyStr
    roles: frozenset[str] = Field(default_factory=frozenset)


class OriginRef(ContractModel):
    provider: NonEmptyStr
    external_thread_id: NonEmptyStr
    external_event_id: str | None = None
    external_channel_id: str | None = None


class TaskStatus(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    REQUIRES_ACTION = "requires_action"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    REQUIRES_ACTION = "requires_action"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"


LEGAL_TASK_RUN_PAIRS: frozenset[tuple[TaskStatus, RunStatus]] = frozenset(
    {
        (TaskStatus.QUEUED, RunStatus.QUEUED),
        (TaskStatus.ACTIVE, RunStatus.RUNNING),
        (TaskStatus.REQUIRES_ACTION, RunStatus.REQUIRES_ACTION),
        (TaskStatus.COMPLETED, RunStatus.COMPLETED),
        (TaskStatus.FAILED, RunStatus.FAILED),
        (TaskStatus.FAILED, RunStatus.TIMED_OUT),
        (TaskStatus.FAILED, RunStatus.BUDGET_EXHAUSTED),
        (TaskStatus.CANCELLED, RunStatus.CANCELLED),
    }
)


class RunPhase(StrEnum):
    RESEARCH = "research"
    PROPOSAL = "proposal"
    POLICY = "policy"
    APPROVAL = "approval"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class Thread(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    origin: OriginRef
    mapping_version: int | None = Field(default=None, ge=1)
    version: int = Field(default=0, ge=0)


class ReasoningStep(ContractModel):
    """One iteration's plan and action, carried forward as working memory.

    This is Leo's ReAct trace. It is model-authored narration, never authority:
    nothing here can grant a capability, cite evidence, or satisfy a verifier
    check. Its only job is to let the next turn know what the previous turns were
    trying to do, so the model can build on its own work instead of restarting.
    """

    iteration: int = Field(ge=0)
    # What the model intends to establish, in its own words.
    plan: str = Field(min_length=1, max_length=600)
    # What it actually did: a tool call summary, or "answered".
    action: str = Field(min_length=1, max_length=300)
    # What came back, summarized by the harness from real outcomes -- not by the
    # model, so a hallucinated success cannot enter the trace.
    outcome: str = Field(min_length=1, max_length=300)

    def render(self) -> str:
        return (
            f"[{self.iteration}] plan: {self.plan} | action: {self.action} | result: {self.outcome}"
        )


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    SATISFIED = "satisfied"
    ABANDONED = "abandoned"


class PlannedStep(ContractModel):
    """One step the model committed to before this run is allowed to finish.

    The scratchpad records what already happened; this records what the model
    said it was going to do. Without it the harness had no way to tell a finished
    answer from an abandoned one: a turn that narrated "I'm pulling the earnings
    data now" and then stopped looked exactly like a completed turn, because
    intent left no trace the coordinator could check.

    A step naming a tool is discharged only by a real retrieved observation from
    that tool -- never by the model asserting it is done. A step may be abandoned,
    but only explicitly and with a reason, which is surfaced to the user rather
    than silently dropped.
    """

    key: NonEmptyStr = Field(max_length=64)
    intent: NonEmptyStr = Field(max_length=300)
    # Empty means a reasoning/synthesis step with no external read.
    tool: str = Field(default="", max_length=128)
    status: PlanStepStatus = PlanStepStatus.PENDING
    note: str = Field(default="", max_length=300)

    @property
    def needs_evidence(self) -> bool:
        return bool(self.tool.strip())


class Task(ContractModel):
    id: NonEmptyStr
    thread_id: NonEmptyStr
    scope: ScopeKey
    objective: NonEmptyStr
    parent_task_id: str | None = None
    continuation_kind: str = "root"
    mapping_version: int | None = Field(default=None, ge=1)
    status: TaskStatus = TaskStatus.QUEUED
    observation_ids: tuple[str, ...] = ()
    verifier_feedback: tuple[str, ...] = ()
    # The model's own running account of what it has tried and what it intends
    # next. Without it every iteration rebuilt a stateless prompt, so on turn
    # four the model could not tell which tools it had already called, with what
    # arguments, or what it was trying to establish -- it saw only a bag of
    # observations and a growing list of complaints. Multi-step reasoning
    # ("I have the quote, now I need earnings, then I compare") is impossible
    # when the intermediate reasoning is discarded every turn.
    scratchpad: tuple[ReasoningStep, ...] = Field(default=(), max_length=32)
    # The model's committed step plan for this run. Completion is gated on it:
    # a run cannot finish while a step is still pending, so a turn that plans to
    # read three sources actually reads them instead of narrating the intent and
    # stopping. The model owns the contents and may revise them mid-run.
    step_plan: tuple[PlannedStep, ...] = Field(default=(), max_length=12)
    final_output: str | None = None
    version: int = Field(default=0, ge=0)

    @property
    def pending_steps(self) -> tuple[PlannedStep, ...]:
        return tuple(item for item in self.step_plan if item.status is PlanStepStatus.PENDING)

    @model_validator(mode="after")
    def completion_has_output(self) -> Task:
        if self.parent_task_id == self.id:
            raise ValueError("task cannot parent itself")
        if not self.continuation_kind.strip():
            raise ValueError("continuation kind must be non-empty")
        if self.status is TaskStatus.COMPLETED and not self.final_output:
            raise ValueError("completed task requires final output")
        if self.status is not TaskStatus.COMPLETED and self.final_output is not None:
            raise ValueError("only a completed task may have final output")
        return self


class BudgetLimits(ContractModel):
    max_iterations: int = Field(default=4, ge=1, le=32)
    max_model_calls: int = Field(default=4, ge=1, le=32)
    max_tool_calls: int = Field(default=6, ge=0, le=64)
    max_elapsed_seconds: float = Field(default=60.0, gt=0, le=3600)
    estimated_model_cost: float = Field(default=0.0, ge=0)
    max_cost: float | None = Field(default=None, ge=0)


class BudgetUsage(ContractModel):
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)
    reserved_cost: float = Field(default=0.0, ge=0)
    reservation_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def token_totals_and_reservation_are_consistent(self) -> BudgetUsage:
        if (
            self.prompt_tokens is not None
            and self.completion_tokens is not None
            and self.total_tokens is not None
            and self.total_tokens != self.prompt_tokens + self.completion_tokens
        ):
            raise ValueError("total tokens must equal prompt plus completion tokens")
        if self.reserved_cost > 0 and self.reservation_id is None:
            raise ValueError("positive reserved cost requires a reservation ID")
        return self


class Run(ContractModel):
    id: NonEmptyStr
    task_id: NonEmptyStr
    scope: ScopeKey
    status: RunStatus = RunStatus.QUEUED
    phase: RunPhase = RunPhase.RESEARCH
    iteration: int = Field(default=0, ge=0)
    limits: BudgetLimits = Field(default_factory=BudgetLimits)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    started_at: datetime | None = None
    deadline_at: datetime | None = None
    final_output: str | None = None
    terminal_reason: str | None = None
    version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def lifecycle_fields_are_consistent(self) -> Run:
        if self.status is RunStatus.QUEUED and self.started_at is not None:
            raise ValueError("queued run cannot have started_at")
        if self.status not in {RunStatus.QUEUED, RunStatus.CANCELLED} and self.started_at is None:
            raise ValueError("started run requires started_at")
        if self.status is RunStatus.COMPLETED and not self.final_output:
            raise ValueError("completed run requires final output")
        if self.status is not RunStatus.COMPLETED and self.final_output is not None:
            raise ValueError("only a completed run may have final output")
        if self.status is RunStatus.COMPLETED and self.terminal_reason != "verified_completion":
            raise ValueError("completed run requires the verified completion reason")
        if (
            self.status
            in {
                RunStatus.REQUIRES_ACTION,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
                RunStatus.BUDGET_EXHAUSTED,
            }
            and not self.terminal_reason
        ):
            raise ValueError("paused or terminal run requires a reason")
        if (
            self.status in {RunStatus.QUEUED, RunStatus.RUNNING}
            and self.terminal_reason is not None
        ):
            raise ValueError("queued or running run cannot carry a terminal reason")
        if (
            self.started_at is not None
            and self.deadline_at is not None
            and self.deadline_at <= self.started_at
        ):
            raise ValueError("run deadline must be after its start time")
        return self


class ToolEffect(StrEnum):
    READ = "read"
    STATE_MUTATION = "state_mutation"
    WRITE = "write"


class ToolRetryPolicy(ContractModel):
    """Harness-visible retry declaration for one tool invocation."""

    max_attempts: int = Field(default=1, ge=1, le=5)


class ToolSpec(ContractModel):
    name: NonEmptyStr
    description: NonEmptyStr
    domain: NonEmptyStr
    input_schema: dict[str, JsonValue]
    version: NonEmptyStr = "1"
    effect: ToolEffect = ToolEffect.READ
    allowed_phases: frozenset[RunPhase] = Field(
        default_factory=lambda: frozenset({RunPhase.RESEARCH})
    )
    timeout_seconds: float = Field(default=10.0, gt=0, le=120)
    retry: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)
    estimated_cost: float = Field(default=0.0, ge=0)
    max_result_bytes: int = Field(default=8192, ge=1, le=1_048_576)
    required_roles: frozenset[str] = Field(default_factory=frozenset)


class CapabilitySelection(ContractModel):
    """Policy-owned, inspectable tool selection for one model turn.

    The catalog and selection fingerprints let a durable run explain exactly which
    catalog snapshot and shortlist produced the advertised schemas without exposing
    rejected capability metadata to the model or event log.
    """

    tools: tuple[ToolSpec, ...]
    catalog_version: NonEmptyStr
    catalog_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    eligible_count: int = Field(ge=0)
    candidate_ids: tuple[NonEmptyStr, ...] = Field(default=(), max_length=32)
    selected_ids: tuple[NonEmptyStr, ...] = Field(default=(), max_length=32)
    selected_skill_ids: tuple[NonEmptyStr, ...] = Field(default=(), max_length=8)
    mode: NonEmptyStr
    reason: NonEmptyStr = Field(max_length=240)

    @model_validator(mode="after")
    def selected_tools_match_payload(self) -> CapabilitySelection:
        names = tuple(tool.name for tool in self.tools)
        if len(names) != len(set(names)):
            raise ValueError("capability selection contains duplicate tool schemas")
        if names != self.selected_ids:
            raise ValueError("selected capability IDs must exactly match tool schemas")
        return self


class ToolRequest(ContractModel):
    id: NonEmptyStr
    name: NonEmptyStr
    arguments: dict[str, JsonValue]


class ToolRequests(ContractModel):
    kind: Literal["tool_requests"] = "tool_requests"
    calls: tuple[ToolRequest, ...] = Field(min_length=1)
    # What the model is trying to establish with these calls. Recorded into the
    # scratchpad so the next iteration inherits the intent, not just the result.
    plan: str = Field(default="", max_length=600)
    # Steps proposed or revised alongside these calls, so a plan can be laid out
    # on the same turn that starts executing it.
    steps: tuple[PlanStepDraft, ...] = Field(default=(), max_length=12)


class ClaimKind(StrEnum):
    SOURCE_CLAIM = "source_claim"
    INFERENCE = "inference"
    AFFECTED_ASSUMPTION = "affected_assumption"
    UNCERTAINTY = "uncertainty"


class CandidateClaim(ContractModel):
    kind: ClaimKind
    statement: NonEmptyStr
    observation_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def model_may_only_propose_evidence_or_inference(self) -> CandidateClaim:
        if self.kind not in {ClaimKind.SOURCE_CLAIM, ClaimKind.INFERENCE}:
            raise ValueError("model claim kind is harness-owned")
        return self


class PlanStepDraft(ContractModel):
    """A step the model proposes, or a revision to one it already committed to."""

    key: NonEmptyStr = Field(max_length=64)
    intent: NonEmptyStr = Field(max_length=300)
    tool: str = Field(default="", max_length=128)
    # Set to abandon a step the model no longer intends to do. The reason is
    # required so an abandoned step is an explicit, explainable decision rather
    # than silent attrition, and so the answer can tell the user what is missing.
    abandon_reason: str = Field(default="", max_length=300)


class CompletionProposal(ContractModel):
    kind: Literal["completion"] = "completion"
    answer: NonEmptyStr
    claims: tuple[CandidateClaim, ...] = ()
    affected_assumption: NonEmptyStr | None = None
    uncertainty: NonEmptyStr | None = None
    plan: str = Field(default="", max_length=600)
    steps: tuple[PlanStepDraft, ...] = Field(default=(), max_length=12)


ModelDecision = Annotated[ToolRequests | CompletionProposal, Field(discriminator="kind")]

# Stable names used by the implementation plan.  These are aliases to the existing canonical
# contracts, not a second set of lifecycle or budget abstractions.
TaskState = TaskStatus
ToolCall = ToolRequest
Budget = BudgetLimits


class ModelUsage(ContractModel):
    """Provider-reported usage; missing metrics remain unknown rather than zero."""

    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0)


class ModelTurnResult(ContractModel):
    """One provider response normalized without granting it runtime authority."""

    decision: ModelDecision
    provider: NonEmptyStr
    model: NonEmptyStr
    request_id: NonEmptyStr | None = None
    finish_reason: str | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    # Populated only by gateways that support it (OpenRouterGateway); legacy/fixture
    # models leave these None. raw_request never carries the Authorization header --
    # that lives in the HTTP client call, not the request body -- so it's safe for a
    # best-effort dashboard transcript sink to persist verbatim.
    raw_request: dict[str, JsonValue] | None = None
    raw_response: dict[str, JsonValue] | None = None


class ToolExecutionContext(ContractModel):
    trusted_scope: TrustedScope
    run_id: NonEmptyStr
    tool_call_id: NonEmptyStr


class SourceRef(ContractModel):
    provider: NonEmptyStr
    reference: NonEmptyStr
    url: str | None = None


class ObservationStatus(StrEnum):
    RETRIEVED = "retrieved"
    STALE = "stale"
    REJECTED = "rejected"


class EvidenceQuality(StrEnum):
    PRIMARY_SOURCE = "primary_source"
    PROVIDER_REPORTED = "provider_reported"
    VERIFIED_CHILD = "verified_child"
    INTERNAL_CONTEXT = "internal_context"
    UNTRUSTED_RETRIEVAL = "untrusted_retrieval"
    DISCOVERY_ONLY = "discovery_only"


class ToolSuccess(ContractModel):
    kind: Literal["success"] = "success"
    data: dict[str, JsonValue]
    source: SourceRef
    observed_at: datetime
    expires_at: datetime | None = None


class ToolFailure(ContractModel):
    kind: Literal["failure"] = "failure"
    code: NonEmptyStr
    retryable: bool = False
    safe_message: NonEmptyStr


ToolOutcome = Annotated[ToolSuccess | ToolFailure, Field(discriminator="kind")]


class Observation(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    run_id: NonEmptyStr
    tool_call_id: NonEmptyStr
    kind: NonEmptyStr
    data: dict[str, JsonValue]
    source: SourceRef
    observed_at: datetime
    expires_at: datetime | None = None
    raw_hash: NonEmptyStr
    status: ObservationStatus = ObservationStatus.RETRIEVED
    quality: EvidenceQuality = EvidenceQuality.PROVIDER_REPORTED
    schema_version: Literal["observation-v1", "observation-v2"] = "observation-v2"
    normalization_version: NonEmptyStr = "normalization-v1"
    rejection_code: NonEmptyStr | None = None

    @model_validator(mode="after")
    def rejection_state_is_explicit(self) -> Observation:
        if self.status is ObservationStatus.REJECTED and self.rejection_code is None:
            raise ValueError("rejected observation requires a rejection code")
        if self.status is not ObservationStatus.REJECTED and self.rejection_code is not None:
            raise ValueError("only rejected observations may carry a rejection code")
        return self


class Claim(ContractModel):
    """Harness-created claim after successful verification."""

    id: NonEmptyStr
    scope: ScopeKey
    run_id: NonEmptyStr
    kind: ClaimKind
    statement: NonEmptyStr
    observation_ids: tuple[str, ...]


class VerifierStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"


class VerifierCheck(ContractModel):
    name: NonEmptyStr
    passed: bool
    detail: NonEmptyStr


class VerifierResult(ContractModel):
    status: VerifierStatus
    checks: tuple[VerifierCheck, ...] = Field(min_length=1)
    retryable: bool
    allow_unsourced_completion: bool = False

    @model_validator(mode="after")
    def status_matches_checks(self) -> VerifierResult:
        all_passed = all(check.passed for check in self.checks)
        if self.status is VerifierStatus.PASS and not all_passed:
            raise ValueError("passing verifier result cannot contain failed checks")
        if self.status is VerifierStatus.FAIL and all_passed:
            raise ValueError("failed verifier result requires at least one failed check")
        return self


class VerifiedCompletion(ContractModel):
    answer: NonEmptyStr
    claims: tuple[Claim, ...]
    verifier_result: VerifierResult

    @model_validator(mode="after")
    def require_pass(self) -> VerifiedCompletion:
        if self.verifier_result.status is not VerifierStatus.PASS:
            raise ValueError("verified completion requires a passing verifier result")
        return self


class VerificationOutcome(ContractModel):
    result: VerifierResult
    completion: VerifiedCompletion | None = None

    @model_validator(mode="after")
    def completion_matches_result(self) -> VerificationOutcome:
        if self.result.status is VerifierStatus.PASS and self.completion is None:
            raise ValueError("passing verification requires a completion")
        if self.result.status is VerifierStatus.FAIL and self.completion is not None:
            raise ValueError("failed verification cannot include a completion")
        return self


class EventType(StrEnum):
    TASK_STARTED = "task_started"
    CONTEXT_BUILT = "context_built"
    MODEL_CALLED = "model_called"
    MODEL_BUDGET_RESERVED = "model_budget_reserved"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    OBSERVATION_CREATED = "observation_created"
    VERIFICATION_STARTED = "verification_started"
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_PASSED = "verification_passed"
    # The model answered while steps it committed to still had no evidence, so
    # the run continued instead of shipping a half-finished answer.
    PLAN_STEP_OUTSTANDING = "plan_step_outstanding"
    RUN_COMPLETED = "run_completed"
    RUN_REQUIRES_ACTION = "run_requires_action"
    RUN_RESUMED = "run_resumed"
    RUN_REQUEUED = "run_requeued"
    RUN_CANCELLED = "run_cancelled"
    RUN_FAILED = "run_failed"
    RUN_TIMED_OUT = "run_timed_out"
    BUDGET_EXHAUSTED = "budget_exhausted"


class EventDraft(ContractModel):
    type: EventType
    iteration: int = Field(ge=0)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class RunEvent(ContractModel):
    id: NonEmptyStr
    run_id: NonEmptyStr
    task_id: NonEmptyStr
    sequence: int = Field(ge=1)
    type: EventType
    occurred_at: datetime
    iteration: int = Field(ge=0)
    schema_version: int = Field(default=1, ge=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class ContextSegment(ContractModel):
    name: NonEmptyStr
    source_type: NonEmptyStr = "legacy"
    content_hash: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    content_version: NonEmptyStr = "legacy-v1"
    estimator_version: NonEmptyStr = "legacy"
    priority: int = Field(ge=0, le=100)
    pinned: bool
    source_ids: tuple[str, ...] = ()
    estimated_tokens: int = Field(default=0, ge=0)
    estimated_bytes: int = Field(default=0, ge=0)
    included: bool = True
    reason: NonEmptyStr = "legacy_unbudgeted"


class ContextManifest(ContractModel):
    segments: tuple[ContextSegment, ...]
    schema_version: int = Field(default=1, ge=1)
    budget_profile: NonEmptyStr = "legacy"
    estimator_version: NonEmptyStr = "legacy"
    max_tokens: int = Field(default=0, ge=0)
    max_bytes: int = Field(default=0, ge=0)
    candidate_estimated_tokens: int = Field(default=0, ge=0)
    candidate_estimated_bytes: int = Field(default=0, ge=0)
    included_estimated_tokens: int = Field(default=0, ge=0)
    included_estimated_bytes: int = Field(default=0, ge=0)
    excluded_estimated_tokens: int = Field(default=0, ge=0)
    excluded_estimated_bytes: int = Field(default=0, ge=0)
    included_segment_count: int = Field(default=0, ge=0)
    excluded_segment_count: int = Field(default=0, ge=0)
    manifest_digest: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def budget_totals_are_reconciled(self) -> ContextManifest:
        names = tuple(segment.name for segment in self.segments)
        if len(names) != len(set(names)):
            raise ValueError("context manifest segment names must be unique")
        if self.schema_version < 2:
            return self
        included = tuple(segment for segment in self.segments if segment.included)
        excluded = tuple(segment for segment in self.segments if not segment.included)
        if any(segment.estimator_version != self.estimator_version for segment in self.segments):
            raise ValueError("context manifest estimator versions differ")
        expected = {
            "candidate_estimated_tokens": sum(
                segment.estimated_tokens for segment in self.segments
            ),
            "candidate_estimated_bytes": sum(segment.estimated_bytes for segment in self.segments),
            "included_estimated_tokens": sum(segment.estimated_tokens for segment in included),
            "included_estimated_bytes": sum(segment.estimated_bytes for segment in included),
            "excluded_estimated_tokens": sum(segment.estimated_tokens for segment in excluded),
            "excluded_estimated_bytes": sum(segment.estimated_bytes for segment in excluded),
            "included_segment_count": len(included),
            "excluded_segment_count": len(excluded),
        }
        if any(getattr(self, name) != value for name, value in expected.items()):
            raise ValueError("context manifest totals do not reconcile")
        if (
            self.included_estimated_tokens > self.max_tokens
            or self.included_estimated_bytes > self.max_bytes
        ):
            raise ValueError("included context exceeds its declared budget")
        return self


# Stable milestone name for the existing exact context/source projection. This is an alias rather
# than a competing manifest abstraction.
SourceManifest = ContextManifest


class ContextItemKind(StrEnum):
    """Typed, selected context supplied by the harness rather than the model."""

    CONVERSATION_TURN = "conversation_turn"
    MEMORY = "memory"
    THREAD_SUMMARY = "thread_summary"
    SUBAGENT_RESULT = "subagent_result"
    SKILL_PROCEDURE = "skill_procedure"


class ContextItemRetention(StrEnum):
    """Server-owned retention class for deterministic context budgeting.

    Retention metadata is deliberately excluded from provider payloads.  It controls
    selection inside the harness; it is never an instruction that untrusted context can
    set for itself.
    """

    SUPPORTING = "supporting"
    THREAD_ROOT = "thread_root"
    RECENT = "recent"
    DECISION = "decision"
    CORRECTION = "correction"
    UNRESOLVED_QUESTION = "unresolved_question"
    PRIOR_OUTCOME = "prior_outcome"
    COMPACTION_SUMMARY = "compaction_summary"

    @property
    def pinned(self) -> bool:
        return self is not ContextItemRetention.SUPPORTING


class ContextItem(ContractModel):
    """One provenance-labelled context item selected inside a trusted visibility set.

    ``content`` is still untrusted data: selecting it makes it relevant, not authoritative.
    The optional source scope is provenance for DM aggregation and can never change the
    run's ``TrustedScope``.
    """

    id: NonEmptyStr
    kind: ContextItemKind
    content: NonEmptyStr = Field(max_length=16_384)
    conversation_id: NonEmptyStr
    source_scope: ScopeKey | None = None
    source_actor_id: str | None = None
    retention: ContextItemRetention = Field(
        default=ContextItemRetention.SUPPORTING,
        exclude=True,
    )
    budget_priority: int | None = Field(default=None, ge=0, le=100, exclude=True)


class ToolChoiceMode(StrEnum):
    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"


class ToolArgumentConstraint(ContractModel):
    """One immutable trusted scalar constraint on a model-proposed tool argument."""

    name: NonEmptyStr
    value: str | int | float | bool | None


class ToolChoicePolicy(ContractModel):
    """Harness-owned constraint translated, but never inferred, by model providers."""

    mode: ToolChoiceMode
    required_tool_name: NonEmptyStr | None = None
    required_arguments: tuple[ToolArgumentConstraint, ...] = ()

    @model_validator(mode="after")
    def required_tool_matches_mode(self) -> ToolChoicePolicy:
        if self.mode is ToolChoiceMode.REQUIRED:
            if self.required_tool_name is None:
                raise ValueError("required tool choice requires a tool name")
        elif self.required_tool_name is not None or self.required_arguments:
            raise ValueError("only required tool choice may constrain a tool")
        names = tuple(item.name for item in self.required_arguments)
        if len(names) != len(set(names)):
            raise ValueError("required tool arguments must have unique names")
        return self


class EvidenceToolRequirement(ContractModel):
    """Map one deterministic completion prerequisite to an allowed read tool."""

    observation_kind: NonEmptyStr
    tool_name: NonEmptyStr
    required_arguments: tuple[ToolArgumentConstraint, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def matches_current_observation_contract(self) -> EvidenceToolRequirement:
        if self.observation_kind != self.tool_name:
            raise ValueError("observation kind must match its producing tool name")
        names = tuple(item.name for item in self.required_arguments)
        if len(names) != len(set(names)):
            raise ValueError("evidence arguments must have unique names")
        return self


def constrained_values_match(
    constraints: tuple[ToolArgumentConstraint, ...],
    values: dict[str, JsonValue],
    *,
    exact: bool,
) -> bool:
    """Compare trusted scalar constraints without bool/number coercion."""

    if exact and set(values) != {item.name for item in constraints}:
        return False
    missing = object()
    for constraint in constraints:
        actual = values.get(constraint.name, missing)
        if actual is missing:
            return False
        if type(actual) is not type(constraint.value) or actual != constraint.value:
            return False
    return True


class CardinalityBounds(ContractModel):
    minimum: int = Field(ge=0, le=32)
    maximum: int = Field(ge=0, le=32)

    @model_validator(mode="after")
    def minimum_does_not_exceed_maximum(self) -> CardinalityBounds:
        if self.minimum > self.maximum:
            raise ValueError("minimum count cannot exceed maximum count")
        return self


class CompletionContract(ContractModel):
    """Trusted request-level bounds and guidance for a model completion proposal."""

    source_claim_count: CardinalityBounds = Field(
        default_factory=lambda: CardinalityBounds(minimum=0, maximum=8)
    )
    source_observation_id_count: CardinalityBounds = Field(
        default_factory=lambda: CardinalityBounds(minimum=1, maximum=8)
    )
    inference_count: CardinalityBounds = Field(
        default_factory=lambda: CardinalityBounds(minimum=0, maximum=8)
    )
    require_affected_assumption: bool = False
    require_uncertainty: bool = False
    guidance: Annotated[str, Field(min_length=1, max_length=500)] = (
        "Return concise claims under the declared source and inference bounds."
    )

    @model_validator(mode="after")
    def source_claims_require_observation_ids(self) -> CompletionContract:
        if self.source_observation_id_count.minimum < 1:
            raise ValueError("source claims must require at least one observation ID")
        return self


class ModelRequest(ContractModel):
    objective: NonEmptyStr
    iteration: int = Field(ge=0)
    observations: tuple[Observation, ...]
    verifier_feedback: tuple[str, ...]
    tools: tuple[ToolSpec, ...]
    tool_choice: ToolChoicePolicy
    completion_contract: CompletionContract = Field(default_factory=CompletionContract)
    manifest: ContextManifest
    context_items: tuple[ContextItem, ...] = Field(default=(), max_length=128)
    # Prior iterations' plan/action/result trace. Advisory working memory only.
    scratchpad: tuple[ReasoningStep, ...] = Field(default=(), max_length=32)
    # The committed step plan and each step's current state. Unlike the
    # scratchpad this is not advisory: the run cannot complete while a step is
    # pending, so the model sees exactly what it still owes.
    step_plan: tuple[PlannedStep, ...] = Field(default=(), max_length=12)

    @model_validator(mode="after")
    def required_tool_is_advertised_safe_tool(self) -> ModelRequest:
        if self.tool_choice.mode is not ToolChoiceMode.REQUIRED:
            return self
        required_name = self.tool_choice.required_tool_name
        matching = tuple(tool for tool in self.tools if tool.name == required_name)
        if len(matching) != 1:
            raise ValueError("required tool must appear exactly once in the advertised tools")
        if matching[0].effect not in {ToolEffect.READ, ToolEffect.STATE_MUTATION}:
            raise ValueError("required tool must be read-only or an internal state mutation")
        properties = matching[0].input_schema.get("properties")
        constrained_names = {item.name for item in self.tool_choice.required_arguments}
        if not isinstance(properties, dict) or not constrained_names.issubset(properties):
            raise ValueError("required arguments must exist in the advertised tool schema")
        return self

    @model_validator(mode="after")
    def payload_selection_matches_budget_manifest(self) -> ModelRequest:
        if self.manifest.schema_version < 2:
            return self
        selected_by_type = {
            source_type: tuple(
                segment.source_ids[0]
                for segment in self.manifest.segments
                if segment.source_type == source_type and segment.included and segment.source_ids
            )
            for source_type in ("observation", "tool_schema", "context_item")
        }
        expected = {
            "observation": tuple(item.id for item in self.observations),
            "tool_schema": tuple(item.name for item in self.tools),
            "context_item": tuple(item.id for item in self.context_items),
        }
        if selected_by_type != expected:
            raise ValueError("model payload selection differs from context manifest")
        return self


class RunBundle(ContractModel):
    thread: Thread
    task: Task
    run: Run
    observations: tuple[Observation, ...] = ()
    claims: tuple[Claim, ...] = ()
    events: tuple[RunEvent, ...] = ()

    @model_validator(mode="after")
    def consistent_identity(self) -> RunBundle:
        if self.task.thread_id != self.thread.id:
            raise ValueError("task does not belong to thread")
        if self.thread.scope != self.task.scope:
            raise ValueError("thread and task scopes differ")
        if self.run.task_id != self.task.id:
            raise ValueError("run does not belong to task")
        if self.run.scope != self.task.scope:
            raise ValueError("task and run scopes differ")
        if (self.task.status, self.run.status) not in LEGAL_TASK_RUN_PAIRS:
            raise ValueError("invalid task/run lifecycle pair")
        observation_ids = {item.id for item in self.observations}
        if len(observation_ids) != len(self.observations):
            raise ValueError("run bundle contains duplicate observations")
        if any(
            item.run_id != self.run.id or item.scope != self.run.scope for item in self.observations
        ):
            raise ValueError("observation is outside the run scope")
        if len(self.task.observation_ids) != len(set(self.task.observation_ids)) or any(
            item not in observation_ids for item in self.task.observation_ids
        ):
            raise ValueError("task references an unavailable or duplicate observation")
        claim_ids = {item.id for item in self.claims}
        if len(claim_ids) != len(self.claims):
            raise ValueError("run bundle contains duplicate claims")
        for claim in self.claims:
            if claim.run_id != self.run.id or claim.scope != self.run.scope:
                raise ValueError("claim is outside the run scope")
            if any(item not in observation_ids for item in claim.observation_ids):
                raise ValueError("claim references an unavailable observation")
        if self.events:
            expected_sequences = tuple(range(1, len(self.events) + 1))
            if tuple(item.sequence for item in self.events) != expected_sequences:
                raise ValueError("run events must be contiguous and ordered")
            if any(
                item.run_id != self.run.id or item.task_id != self.task.id for item in self.events
            ):
                raise ValueError("run event is outside the task/run identity")
        return self


class CoordinatorResult(ContractModel):
    thread: Thread
    task: Task
    run: Run
    observations: tuple[Observation, ...]
    claims: tuple[Claim, ...]
    events: tuple[RunEvent, ...]
