"""Typed, versioned eval variants and an explicit executor-support matrix."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from leo.evals.models import Scenario
from leo.harness.models import ContractModel, NonEmptyStr


class ScenarioVariant(StrEnum):
    CONVERSATION = "conversation"
    DM_SOURCE = "dm_source"
    CONTEXT = "context"
    MEMORY = "memory"
    LONG_THREAD = "long_thread"
    ROUTING = "routing"
    TOOL = "tool"
    TOOL_RECALL = "tool_recall"
    AUTONOMY = "autonomy"
    PLAN_CHILD = "plan_child"
    REPLANNING = "replanning"
    PROVIDER = "provider"
    DELIVERY = "delivery"
    LEASE_OUTBOX = "lease_outbox"
    FAULT_RECOVERY = "fault_recovery"
    VERIFIER = "verifier"
    TRIGGER_SILENCE = "trigger_silence"
    APPROVAL_UNKNOWN_EFFECT = "approval_unknown_effect"


class VariantSupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED_BY_RUNNER = "unsupported_by_runner"
    # Deprecated compatibility spelling used by the early foundation contract.
    UNSUPPORTED = "unsupported"


class ConversationAuthorityFixture(ContractModel):
    """Server-derived conversation projection used by an eval, never model authority."""

    team_id: NonEmptyStr
    destination_id: NonEmptyStr
    destination_kind: Literal["channel", "dm", "group_dm", "shared", "external"]
    actor_id: str | None = None
    external_provenance: Literal["internal", "shared", "external", "not_applicable", "unknown"] = (
        "internal"
    )
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    forbidden_conversation_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def exact_authorized_projection(self) -> ConversationAuthorityFixture:
        allowed = tuple(sorted(set(self.allowed_conversation_ids)))
        forbidden = tuple(sorted(set(self.forbidden_conversation_ids)))
        if allowed != self.allowed_conversation_ids:
            raise ValueError("allowed conversation IDs must be sorted and unique")
        if forbidden != self.forbidden_conversation_ids:
            raise ValueError("forbidden conversation IDs must be sorted and unique")
        if set(allowed) & set(forbidden):
            raise ValueError("allowed and forbidden conversation IDs must be disjoint")
        if self.destination_id not in allowed:
            raise ValueError("conversation projection must include its exact destination")
        if self.destination_kind == "dm":
            if not self.actor_id:
                raise ValueError("one-to-one DM projection requires a server-derived actor")
        else:
            if self.actor_id is not None:
                raise ValueError("only one-to-one DM projection may carry an actor")
            if allowed != (self.destination_id,):
                raise ValueError("non-DM projection must use only the exact destination")
        return self


class ChildEffect(StrEnum):
    READ = "read"
    PREPARE = "prepare"


class MemoryFixture(ContractModel):
    operation: Literal["remember", "correct", "forget", "retrieve", "lifecycle"]
    memory_ids: tuple[NonEmptyStr, ...] = ()
    expected_revision: int | None = Field(default=None, ge=1)
    expected_visible: bool | None = None

    @model_validator(mode="after")
    def exact_memory_references(self) -> MemoryFixture:
        if tuple(sorted(set(self.memory_ids))) != self.memory_ids:
            raise ValueError("memory fixture IDs must be sorted and unique")
        if self.operation in {"correct", "forget"} and not self.memory_ids:
            raise ValueError("memory mutation fixture requires an existing memory ID")
        return self


class ContextFixture(ContractModel):
    mode: Literal["recent", "compacted", "long_thread"]
    turn_count: int = Field(ge=0, le=100_000)
    token_budget: int = Field(ge=1, le=1_000_000)
    required_anchor_ids: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def anchors_are_exact(self) -> ContextFixture:
        if tuple(sorted(set(self.required_anchor_ids))) != self.required_anchor_ids:
            raise ValueError("context anchor IDs must be sorted and unique")
        return self


class RouteFixture(ContractModel):
    expected_route: Literal["direct", "clarify", "tool", "plan", "delegate"]
    required_tool_names: tuple[NonEmptyStr, ...] = ()
    forbidden_tool_names: tuple[NonEmptyStr, ...] = ()

    @model_validator(mode="after")
    def tool_sets_are_exact(self) -> RouteFixture:
        required = tuple(sorted(set(self.required_tool_names)))
        forbidden = tuple(sorted(set(self.forbidden_tool_names)))
        if required != self.required_tool_names or forbidden != self.forbidden_tool_names:
            raise ValueError("route tool names must be sorted and unique")
        if set(required) & set(forbidden):
            raise ValueError("required and forbidden route tools must be disjoint")
        if self.expected_route == "tool" and not required:
            raise ValueError("tool route requires at least one exact tool name")
        return self


class ToolFixture(ContractModel):
    catalog_names: tuple[NonEmptyStr, ...] = ()
    expected_call_ids: tuple[NonEmptyStr, ...] = ()
    maximum_effect: ChildEffect = ChildEffect.READ
    evidence_required: bool = True

    @model_validator(mode="after")
    def catalog_and_calls_are_exact(self) -> ToolFixture:
        if tuple(sorted(set(self.catalog_names))) != self.catalog_names:
            raise ValueError("tool catalog names must be sorted and unique")
        if tuple(sorted(set(self.expected_call_ids))) != self.expected_call_ids:
            raise ValueError("tool call IDs must be sorted and unique")
        return self


class DeliveryFixture(ContractModel):
    boundary: Literal["ingress", "lease", "outbox", "slack"]
    duplicate_attempts: int = Field(default=0, ge=0, le=64)
    expected_committed_outcomes: int = Field(default=1, ge=0, le=1)
    expected_physical_deliveries: int = Field(default=1, ge=0, le=1)


class FutureFixture(ContractModel):
    milestone: Literal["M7", "M8"]
    expected_runner_code: Literal["unsupported_by_runner"] = "unsupported_by_runner"


class FaultMatrixFixture(ContractModel):
    point_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    crash_sides: tuple[Literal["before", "after"], ...] = ("before", "after")
    expected_false_success_count: int = Field(default=0, ge=0, le=0)

    @model_validator(mode="after")
    def exact_fault_vocabulary(self) -> FaultMatrixFixture:
        if tuple(sorted(set(self.point_ids))) != self.point_ids:
            raise ValueError("fault point IDs must be sorted and unique")
        if tuple(sorted(set(self.crash_sides))) != tuple(sorted(self.crash_sides)):
            raise ValueError("fault crash sides must be unique")
        return self


class PlanNodeFixture(ContractModel):
    key: NonEmptyStr
    depends_on: tuple[NonEmptyStr, ...] = ()
    child_task_id: str | None = None
    child_run_id: str | None = None
    effect: ChildEffect = ChildEffect.READ

    @model_validator(mode="after")
    def child_identity_and_dependencies_are_exact(self) -> PlanNodeFixture:
        if tuple(sorted(set(self.depends_on))) != self.depends_on:
            raise ValueError("plan fixture dependencies must be sorted and unique")
        if self.key in self.depends_on:
            raise ValueError("plan fixture node cannot depend on itself")
        if (self.child_task_id is None) != (self.child_run_id is None):
            raise ValueError("child task and run IDs must be declared together")
        return self


class OrchestrationFixture(ContractModel):
    parent_task_id: NonEmptyStr
    parent_run_id: NonEmptyStr
    plan_id: NonEmptyStr
    revision: int = Field(default=1, ge=1, le=8)
    nodes: tuple[PlanNodeFixture, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def node_graph_is_bounded_and_acyclic(self) -> OrchestrationFixture:
        nodes = {node.key: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("plan fixture node keys must be unique")
        if any(dependency not in nodes for node in self.nodes for dependency in node.depends_on):
            raise ValueError("plan fixture dependency is unknown")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("plan fixture dependencies must be acyclic")
            if key in visited:
                return
            visiting.add(key)
            for dependency in nodes[key].depends_on:
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in sorted(nodes):
            visit(key)
        return self


class VariantScenario(ContractModel):
    """Compatibility record binding a typed variant to one executable scenario."""

    id: NonEmptyStr
    version: str = Field(pattern=r"^v[0-9]+$")
    variant: ScenarioVariant
    fixture_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    support_status: VariantSupport
    expected_outcome: NonEmptyStr
    executor_variant: str | None = None
    dependencies: tuple[NonEmptyStr, ...] = ()
    conversation: ConversationAuthorityFixture | None = None
    memory: MemoryFixture | None = None
    context: ContextFixture | None = None
    route: RouteFixture | None = None
    tool: ToolFixture | None = None
    orchestration: OrchestrationFixture | None = None
    delivery: DeliveryFixture | None = None
    faults: FaultMatrixFixture | None = None
    future: FutureFixture | None = None
    # Compatibility-only metadata. Authority-bearing keys are rejected below.
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def support_and_authority_are_consistent(self) -> VariantScenario:
        if tuple(sorted(set(self.dependencies))) != self.dependencies:
            raise ValueError("variant dependencies must be sorted and unique")
        if self.supported and not self.executor_variant:
            raise ValueError("supported variant requires an executable variant ID")
        if not self.supported and self.executor_variant is not None:
            raise ValueError("unsupported variant cannot claim an executor")
        if self.variant is ScenarioVariant.DM_SOURCE:
            if self.conversation is None or self.conversation.destination_kind != "dm":
                raise ValueError("DM-source variant requires an exact one-to-one DM projection")
        if self.variant is ScenarioVariant.MEMORY and self.memory is None:
            raise ValueError("memory variant requires a typed memory fixture")
        if self.variant in {ScenarioVariant.CONTEXT, ScenarioVariant.LONG_THREAD}:
            if self.context is None:
                raise ValueError("context variant requires a typed context fixture")
        if self.variant in {ScenarioVariant.ROUTING, ScenarioVariant.AUTONOMY}:
            if self.route is None:
                raise ValueError("routing variant requires a typed route fixture")
        if (
            self.variant
            in {
                ScenarioVariant.TOOL,
                ScenarioVariant.TOOL_RECALL,
                ScenarioVariant.PROVIDER,
            }
            and self.tool is None
        ):
            raise ValueError("tool/provider variant requires a typed tool fixture")
        if self.variant in {ScenarioVariant.PLAN_CHILD, ScenarioVariant.REPLANNING}:
            if self.supported and self.orchestration is None:
                raise ValueError("supported plan variant requires parent/child references")
        if self.variant in {ScenarioVariant.DELIVERY, ScenarioVariant.LEASE_OUTBOX}:
            if self.delivery is None:
                raise ValueError("delivery variant requires a typed delivery fixture")
        if self.variant is ScenarioVariant.FAULT_RECOVERY and self.faults is None:
            raise ValueError("fault recovery variant requires a typed boundary matrix")
        if self.variant in {
            ScenarioVariant.TRIGGER_SILENCE,
            ScenarioVariant.APPROVAL_UNKNOWN_EFFECT,
        }:
            if self.supported:
                raise ValueError("future trigger/action variants are unsupported by this runner")
            if self.future is None:
                raise ValueError("future variant requires an explicit unsupported fixture")
        forbidden_payload_keys = {
            "allowed_conversation_ids",
            "actor_id",
            "child_effect",
            "child_run_id",
            "child_task_id",
            "support_status",
            "trusted_scope",
        }
        if forbidden_payload_keys & set(self.payload):
            raise ValueError("generic variant payload cannot carry runtime authority")
        return self

    @property
    def supported(self) -> bool:
        return self.support_status is VariantSupport.SUPPORTED


_EXECUTABLE_VARIANTS: dict[str, ScenarioVariant] = {
    "channel_isolation": ScenarioVariant.CONVERSATION,
    "contextual_conversation": ScenarioVariant.ROUTING,
    "delegated_dependency_plan": ScenarioVariant.PLAN_CHILD,
    "dm_context_union": ScenarioVariant.DM_SOURCE,
    "parallel_read_batch": ScenarioVariant.TOOL,
    "quote_control": ScenarioVariant.PROVIDER,
    "restart_replay_idempotency": ScenarioVariant.DELIVERY,
    "safe_failure": ScenarioVariant.VERIFIER,
    "verifier_correction": ScenarioVariant.REPLANNING,
    "memory_lifecycle": ScenarioVariant.MEMORY,
    "long_thread_compaction": ScenarioVariant.LONG_THREAD,
    "tool_recall_progressive": ScenarioVariant.TOOL_RECALL,
    "shared_group_external_scope": ScenarioVariant.CONVERSATION,
    "budget_boundary": ScenarioVariant.AUTONOMY,
    "fault_recovery_matrix": ScenarioVariant.FAULT_RECOVERY,
    "conversational_terminal_recovery": ScenarioVariant.VERIFIER,
    "elastic_deliberation": ScenarioVariant.AUTONOMY,
    "slack_thread_context_authority": ScenarioVariant.LONG_THREAD,
    "tavily_verified_research": ScenarioVariant.PROVIDER,
}

_RESERVED_VARIANTS: tuple[tuple[str, ScenarioVariant, tuple[str, ...]], ...] = (
    ("reserved_context_compaction", ScenarioVariant.CONTEXT, ("M1-T05", "M5-T05")),
    ("reserved_lease_outbox", ScenarioVariant.LEASE_OUTBOX, ("M2-T05", "M5-T05")),
    ("future_trigger_silence", ScenarioVariant.TRIGGER_SILENCE, ("M7-T01",)),
    (
        "future_approval_unknown_effect",
        ScenarioVariant.APPROVAL_UNKNOWN_EFFECT,
        ("M8-T06",),
    ),
)


def build_variant_matrix(scenarios: tuple[Scenario, ...]) -> tuple[VariantScenario, ...]:
    """Build the support matrix without allowing a fixture to invent an executor."""

    supported: list[VariantScenario] = []
    for scenario in sorted(scenarios, key=lambda item: item.id):
        variant = _EXECUTABLE_VARIANTS.get(scenario.execution_variant)
        if variant is None:
            raise ValueError(f"scenario executor is absent from variant registry: {scenario.id}")
        supported.append(
            VariantScenario(
                id=scenario.id,
                version=scenario.version,
                variant=variant,
                fixture_digest=scenario.fixture_digest,
                support_status=VariantSupport.SUPPORTED,
                expected_outcome="executable_offline",
                executor_variant=scenario.execution_variant,
                dependencies=_dependencies_for(variant),
                conversation=_conversation_for(scenario),
                memory=_memory_for(scenario),
                context=_context_for(scenario),
                route=_route_for(scenario),
                tool=_tool_for(scenario),
                orchestration=_orchestration_for(scenario),
                delivery=_delivery_for(scenario),
                faults=_faults_for(scenario),
            )
        )
    reserved = tuple(
        VariantScenario(
            id=variant_id,
            version="v1",
            variant=variant,
            fixture_digest=hashlib.sha256(variant_id.encode("utf-8")).hexdigest(),
            support_status=VariantSupport.UNSUPPORTED_BY_RUNNER,
            expected_outcome="unsupported_by_runner",
            dependencies=tuple(sorted(dependencies)),
            memory=_reserved_memory_for(variant),
            context=_reserved_context_for(variant),
            route=_reserved_route_for(variant),
            tool=_reserved_tool_for(variant),
            delivery=_reserved_delivery_for(variant),
            future=_future_for(variant),
        )
        for variant_id, variant, dependencies in _RESERVED_VARIANTS
    )
    matrix = tuple(supported) + reserved
    validate_variant_compatibility(matrix, scenarios)
    return matrix


def validate_variant_compatibility(
    variants: tuple[VariantScenario, ...],
    scenarios: tuple[Scenario, ...],
) -> None:
    ids = tuple(item.id for item in variants)
    if len(ids) != len(set(ids)):
        raise ValueError("variant matrix IDs must be unique")
    scenario_by_id = {scenario.id: scenario for scenario in scenarios}
    supported = {item.id: item for item in variants if item.supported}
    if set(supported) != set(scenario_by_id):
        raise ValueError("supported variant matrix must exactly cover executable scenarios")
    for scenario_id, scenario in scenario_by_id.items():
        item = supported[scenario_id]
        if (
            item.executor_variant != scenario.execution_variant
            or item.version != scenario.version
            or item.fixture_digest != scenario.fixture_digest
        ):
            raise ValueError("variant matrix does not match its executable fixture")
    if any(
        item.support_status is VariantSupport.UNSUPPORTED_BY_RUNNER
        and item.expected_outcome != "unsupported_by_runner"
        for item in variants
    ):
        raise ValueError("unsupported variants require an explicit runner outcome")
    declared = {item.variant for item in variants}
    if declared != set(ScenarioVariant):
        missing = sorted(item.value for item in set(ScenarioVariant) - declared)
        raise ValueError(f"variant matrix omits typed variants: {missing}")


def _dependencies_for(variant: ScenarioVariant) -> tuple[str, ...]:
    dependencies = {
        ScenarioVariant.CONVERSATION: ("M2-T10",),
        ScenarioVariant.DM_SOURCE: ("M2-T10", "M3-T10A"),
        ScenarioVariant.ROUTING: ("M4-T02C",),
        ScenarioVariant.TOOL: ("M4-T04",),
        ScenarioVariant.PROVIDER: ("M4-T04",),
        ScenarioVariant.PLAN_CHILD: ("M4-T06",),
        ScenarioVariant.REPLANNING: ("M4-T06", "M4-T09"),
        ScenarioVariant.DELIVERY: ("M2-T09",),
        ScenarioVariant.VERIFIER: ("M4-T08",),
        ScenarioVariant.MEMORY: ("M3-T03",),
        ScenarioVariant.LONG_THREAD: ("M1-T05", "M3-T07"),
        ScenarioVariant.TOOL_RECALL: ("M4-T02C",),
        ScenarioVariant.AUTONOMY: ("M4-T02C",),
        ScenarioVariant.FAULT_RECOVERY: ("M5-T06",),
    }
    return tuple(sorted(dependencies.get(variant, ())))


def _conversation_for(scenario: Scenario) -> ConversationAuthorityFixture | None:
    if scenario.execution_variant == "channel_isolation":
        return ConversationAuthorityFixture(
            team_id="T-EVAL",
            destination_id="C-BETA",
            destination_kind="channel",
            allowed_conversation_ids=("C-BETA",),
            forbidden_conversation_ids=("C-ALPHA",),
        )
    if scenario.execution_variant == "dm_context_union":
        return ConversationAuthorityFixture(
            team_id="T-EVAL",
            destination_id="D-EVAL",
            destination_kind="dm",
            actor_id="U-EVAL",
            external_provenance="not_applicable",
            allowed_conversation_ids=("C-ALPHA", "C-BETA", "D-EVAL"),
            forbidden_conversation_ids=("C-FORBIDDEN",),
        )
    return None


def _route_for(scenario: Scenario) -> RouteFixture | None:
    if scenario.execution_variant == "contextual_conversation":
        return RouteFixture(expected_route="direct")
    if scenario.execution_variant == "budget_boundary":
        return RouteFixture(
            expected_route="tool",
            required_tool_names=("market.get_quote",),
        )
    if scenario.execution_variant == "elastic_deliberation":
        return RouteFixture(
            expected_route="plan",
            required_tool_names=("agent.execute_research_plan",),
        )
    return None


def _memory_for(scenario: Scenario) -> MemoryFixture | None:
    if scenario.execution_variant != "memory_lifecycle":
        return None
    return MemoryFixture(operation="lifecycle", memory_ids=("memory-1",))


def _context_for(scenario: Scenario) -> ContextFixture | None:
    if scenario.execution_variant == "long_thread_compaction":
        return ContextFixture(
            mode="long_thread",
            turn_count=60,
            token_budget=4_096,
            required_anchor_ids=("green-launch-decision",),
        )
    if scenario.execution_variant == "slack_thread_context_authority":
        return ContextFixture(
            mode="compacted",
            turn_count=60,
            token_budget=850,
            required_anchor_ids=("thread-root",),
        )
    return None


def _tool_for(scenario: Scenario) -> ToolFixture | None:
    if scenario.execution_variant == "parallel_read_batch":
        return ToolFixture(
            catalog_names=("market.get_quote",),
            expected_call_ids=("parallel-call-1", "parallel-call-2"),
        )
    if scenario.execution_variant == "quote_control":
        return ToolFixture(
            catalog_names=("market.get_quote",),
            expected_call_ids=("quote-call-1",),
        )
    if scenario.execution_variant == "tool_recall_progressive":
        return ToolFixture(
            catalog_names=(
                "market.get_quote",
                "sec.get_recent_filings",
                "web.fetch_public_text",
            ),
            expected_call_ids=("tool-recall-search-1",),
        )
    if scenario.execution_variant == "tavily_verified_research":
        return ToolFixture(
            catalog_names=("web.fetch_public_text", "web.search_tavily"),
            expected_call_ids=("fetch-call", "search-call"),
        )
    return None


def _delivery_for(scenario: Scenario) -> DeliveryFixture | None:
    if scenario.execution_variant != "restart_replay_idempotency":
        return None
    return DeliveryFixture(
        boundary="outbox",
        duplicate_attempts=1,
        expected_committed_outcomes=1,
        expected_physical_deliveries=1,
    )


def _faults_for(scenario: Scenario) -> FaultMatrixFixture | None:
    if scenario.execution_variant != "fault_recovery_matrix":
        return None
    return FaultMatrixFixture(
        point_ids=(
            "child_model",
            "database",
            "lease",
            "membership",
            "parent_model",
            "plan",
            "slack",
            "synthesis",
            "tool",
            "verifier",
        )
    )


def _orchestration_for(scenario: Scenario) -> OrchestrationFixture | None:
    if scenario.execution_variant not in {
        "delegated_dependency_plan",
        "verifier_correction",
    }:
        return None
    prefix = scenario.deterministic_id_prefix
    nodes = (
        PlanNodeFixture(
            key="baseline",
            child_task_id=f"{prefix}-child-task-1",
            child_run_id=f"{prefix}-child-run-1",
        ),
        PlanNodeFixture(
            key="dependent",
            depends_on=("baseline",),
            child_task_id=f"{prefix}-child-task-2",
            child_run_id=f"{prefix}-child-run-2",
            effect=ChildEffect.PREPARE,
        ),
    )
    return OrchestrationFixture(
        parent_task_id=f"{prefix}-task",
        parent_run_id=f"{prefix}-run",
        plan_id=f"{prefix}-plan",
        nodes=nodes,
    )


def _reserved_memory_for(variant: ScenarioVariant) -> MemoryFixture | None:
    if variant is not ScenarioVariant.MEMORY:
        return None
    return MemoryFixture(operation="lifecycle", memory_ids=("memory-1",))


def _reserved_context_for(variant: ScenarioVariant) -> ContextFixture | None:
    if variant is ScenarioVariant.CONTEXT:
        return ContextFixture(
            mode="compacted",
            turn_count=64,
            token_budget=4_096,
            required_anchor_ids=("anchor-1",),
        )
    if variant is ScenarioVariant.LONG_THREAD:
        return ContextFixture(
            mode="long_thread",
            turn_count=10_000,
            token_budget=4_096,
            required_anchor_ids=("anchor-1",),
        )
    return None


def _reserved_route_for(variant: ScenarioVariant) -> RouteFixture | None:
    if variant is not ScenarioVariant.AUTONOMY:
        return None
    return RouteFixture(expected_route="clarify")


def _reserved_tool_for(variant: ScenarioVariant) -> ToolFixture | None:
    if variant is not ScenarioVariant.TOOL_RECALL:
        return None
    return ToolFixture(
        catalog_names=("market.get_quote", "sec.get_recent_filings"),
        expected_call_ids=("tool-recall-call-1",),
    )


def _reserved_delivery_for(variant: ScenarioVariant) -> DeliveryFixture | None:
    if variant is not ScenarioVariant.LEASE_OUTBOX:
        return None
    return DeliveryFixture(
        boundary="lease",
        duplicate_attempts=1,
        expected_committed_outcomes=1,
        expected_physical_deliveries=1,
    )


def _future_for(variant: ScenarioVariant) -> FutureFixture | None:
    if variant is ScenarioVariant.TRIGGER_SILENCE:
        return FutureFixture(milestone="M7")
    if variant is ScenarioVariant.APPROVAL_UNKNOWN_EFFECT:
        return FutureFixture(milestone="M8")
    return None
