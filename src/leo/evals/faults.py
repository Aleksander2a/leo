"""Named deterministic fault controls available only to eval/test composition."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable
from enum import StrEnum
from functools import partial

from pydantic import Field, JsonValue, model_validator

from leo.harness.models import (
    CompletionProposal,
    ContractModel,
    ModelRequest,
    ModelTurnResult,
    NonEmptyStr,
    RunBundle,
    ToolExecutionContext,
    ToolOutcome,
    ToolSpec,
    VerificationOutcome,
)
from leo.harness.ports import CompletionVerifier, ModelGateway, Tool


class FaultPoint(StrEnum):
    PARENT_MODEL = "parent_model"
    MODEL = "parent_model"  # Deprecated compatibility alias.
    CHILD_MODEL = "child_model"
    PLAN = "plan"
    TOOL = "tool"
    MEMBERSHIP = "membership"
    DATABASE = "database"
    LEASE = "lease"
    SLACK = "slack"
    SYNTHESIS = "synthesis"
    VERIFIER = "verifier"


class FaultSide(StrEnum):
    BEFORE = "before"
    AFTER = "after"


class FaultAction(StrEnum):
    RAISE = "raise"
    TIMEOUT = "timeout"
    DISCONNECT = "disconnect"
    RETURN_FAILURE = "return_failure"
    STALE_RESULT = "stale_result"
    DROP_ACK = "drop_ack"


class FaultTrigger(ContractModel):
    point: FaultPoint
    call_index: int = Field(ge=1)
    side: FaultSide = FaultSide.BEFORE
    action: FaultAction
    safe_code: NonEmptyStr
    repeat_every: int | None = Field(default=None, ge=1, le=10_000)

    def fires_at(self, call_index: int) -> bool:
        if call_index == self.call_index:
            return True
        return (
            self.repeat_every is not None
            and call_index > self.call_index
            and (call_index - self.call_index) % self.repeat_every == 0
        )


class FaultPlan(ContractModel):
    version: NonEmptyStr = "faults-v2"
    triggers: tuple[FaultTrigger, ...] = Field(default=(), max_length=64)

    @model_validator(mode="after")
    def deterministic_trigger_slots(self) -> FaultPlan:
        slots = tuple(
            (trigger.point, trigger.call_index, trigger.side) for trigger in self.triggers
        )
        if len(slots) != len(set(slots)):
            raise ValueError("fault trigger slots must be unique")
        return self

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class FaultLogEntry(ContractModel):
    sequence: int = Field(ge=1)
    plan_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    point: FaultPoint
    side: FaultSide
    call_index: int = Field(ge=1)
    fired: bool
    action: FaultAction | None = None
    safe_code: str | None = None

    @model_validator(mode="after")
    def fired_fields_agree(self) -> FaultLogEntry:
        if self.fired != (self.action is not None and self.safe_code is not None):
            raise ValueError("fired fault log entries require action and safe code")
        return self


class FaultRecoveryOutcome(StrEnum):
    RETRY_SAFE = "retry_safe"
    RELOAD_REQUIRED = "reload_required"
    RECLAIM_REQUIRED = "reclaim_required"
    UNKNOWN_EFFECT = "unknown_effect"
    REJECTED_SAFE = "rejected_safe"
    FAIL_CLOSED = "fail_closed"


class FaultRecoveryRecord(ContractModel):
    point: FaultPoint
    side: FaultSide
    safe_code: NonEmptyStr
    operation_applied: bool
    outcome: FaultRecoveryOutcome
    terminal_success: bool = False
    safe_recovery: bool

    @model_validator(mode="after")
    def crash_side_and_recovery_agree(self) -> FaultRecoveryRecord:
        if self.side is FaultSide.BEFORE and self.operation_applied:
            raise ValueError("before-side fault cannot apply its operation")
        if self.terminal_success:
            raise ValueError("injected fault recovery cannot self-attest terminal success")
        if (
            self.outcome is FaultRecoveryOutcome.UNKNOWN_EFFECT
            and self.point is not FaultPoint.SLACK
        ):
            raise ValueError("only an after-Slack fault may be an unknown external effect")
        return self


class FaultRecoveryMatrix(ContractModel):
    version: NonEmptyStr = "fault-recovery-v1"
    records: tuple[FaultRecoveryRecord, ...]
    case_count: int = Field(ge=1)
    before_case_count: int = Field(ge=1)
    before_without_operation_count: int = Field(ge=0)
    triggered_count: int = Field(ge=0)
    safe_recovery_count: int = Field(ge=0)
    unsafe_recovery_count: int = Field(ge=0)
    false_success_count: int = Field(ge=0)
    unknown_effect_count: int = Field(ge=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def counts_and_digest_agree(self) -> FaultRecoveryMatrix:
        expected = {
            "case_count": len(self.records),
            "before_case_count": sum(item.side is FaultSide.BEFORE for item in self.records),
            "before_without_operation_count": sum(
                item.side is FaultSide.BEFORE and not item.operation_applied
                for item in self.records
            ),
            "triggered_count": len(self.records),
            "safe_recovery_count": sum(item.safe_recovery for item in self.records),
            "unsafe_recovery_count": sum(not item.safe_recovery for item in self.records),
            "false_success_count": sum(item.terminal_success for item in self.records),
            "unknown_effect_count": sum(
                item.outcome is FaultRecoveryOutcome.UNKNOWN_EFFECT for item in self.records
            ),
        }
        if any(getattr(self, key) != value for key, value in expected.items()):
            raise ValueError("fault recovery matrix counts do not reconcile")
        payload = self.model_dump(mode="json", exclude={"digest"})
        if self.digest != _fault_digest(payload):
            raise ValueError("fault recovery matrix digest mismatch")
        return self


_TEST_ONLY_AUTHORITY = object()


class FaultController:
    """Call-indexed controller that cannot be constructed from serialized input."""

    def __init__(self, plan: FaultPlan, *, _authority: object) -> None:
        if _authority is not _TEST_ONLY_AUTHORITY:
            raise PermissionError("test_fault_authority_required")
        self._plan = plan
        self._counts: dict[tuple[FaultPoint, FaultSide], int] = {
            (point, side): 0 for point in FaultPoint for side in FaultSide
        }
        self._fired: list[FaultTrigger] = []
        self._log: list[FaultLogEntry] = []

    def observe(
        self,
        point: FaultPoint,
        *,
        side: FaultSide = FaultSide.BEFORE,
    ) -> FaultTrigger | None:
        key = (point, side)
        self._counts[key] += 1
        count = self._counts[key]
        trigger = next(
            (
                candidate
                for candidate in self._plan.triggers
                if candidate.point is point and candidate.side is side and candidate.fires_at(count)
            ),
            None,
        )
        if trigger is not None:
            self._fired.append(trigger)
        self._log.append(
            FaultLogEntry(
                sequence=len(self._log) + 1,
                plan_digest=self._plan.digest,
                point=point,
                side=side,
                call_index=count,
                fired=trigger is not None,
                action=trigger.action if trigger is not None else None,
                safe_code=trigger.safe_code if trigger is not None else None,
            )
        )
        return trigger

    def enforce(
        self,
        point: FaultPoint,
        *,
        side: FaultSide = FaultSide.BEFORE,
    ) -> None:
        trigger = self.observe(point, side=side)
        if trigger is not None:
            raise InjectedFault(trigger)

    @property
    def fired(self) -> tuple[FaultTrigger, ...]:
        return tuple(self._fired)

    @property
    def log(self) -> tuple[FaultLogEntry, ...]:
        return tuple(self._log)

    @property
    def plan(self) -> FaultPlan:
        return self._plan


def fault_controller_for_test(plan: FaultPlan) -> FaultController:
    """The sole construction boundary; production modules never import this factory."""

    return FaultController(plan, _authority=_TEST_ONLY_AUTHORITY)


class InjectedFault(RuntimeError):
    """Typed test exception carrying only the declared safe fault outcome."""

    def __init__(self, trigger: FaultTrigger) -> None:
        super().__init__(trigger.safe_code)
        self.point = trigger.point
        self.side = trigger.side
        self.action = trigger.action
        self.safe_code = trigger.safe_code


class FaultInjectedModelGateway:
    def __init__(
        self,
        delegate: ModelGateway,
        controller: FaultController,
        *,
        point: FaultPoint,
    ) -> None:
        if point not in {FaultPoint.PARENT_MODEL, FaultPoint.CHILD_MODEL}:
            raise ValueError("model fault wrapper requires a model fault point")
        self._delegate = delegate
        self._controller = controller
        self._point = point

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        self._controller.enforce(self._point, side=FaultSide.BEFORE)
        result = await self._delegate.decide(request)
        self._controller.enforce(self._point, side=FaultSide.AFTER)
        return result


class FaultInjectedTool:
    def __init__(
        self,
        delegate: Tool,
        controller: FaultController,
        *,
        point: FaultPoint = FaultPoint.TOOL,
    ) -> None:
        self._delegate = delegate
        self._controller = controller
        self._point = point

    @property
    def spec(self) -> ToolSpec:
        return self._delegate.spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return self._delegate.validate(arguments)

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        self._controller.enforce(self._point, side=FaultSide.BEFORE)
        result = await self._delegate.execute(arguments, context)
        self._controller.enforce(self._point, side=FaultSide.AFTER)
        return result


class FaultInjectedVerifier:
    def __init__(
        self,
        delegate: CompletionVerifier,
        controller: FaultController,
    ) -> None:
        self._delegate = delegate
        self._controller = controller

    def verify(
        self,
        proposal: CompletionProposal,
        bundle: RunBundle,
    ) -> VerificationOutcome:
        self._controller.enforce(FaultPoint.VERIFIER, side=FaultSide.BEFORE)
        result = self._delegate.verify(proposal, bundle)
        self._controller.enforce(FaultPoint.VERIFIER, side=FaultSide.AFTER)
        return result


class FaultBoundaryProbe:
    """Instrument non-model boundaries without monkeypatches or sleeps."""

    def __init__(
        self,
        controller: FaultController,
        point: FaultPoint,
    ) -> None:
        self._controller = controller
        self._point = point

    async def invoke[Result](
        self,
        operation: Callable[[], Awaitable[Result]],
    ) -> Result:
        self._controller.enforce(self._point, side=FaultSide.BEFORE)
        result = await operation()
        self._controller.enforce(self._point, side=FaultSide.AFTER)
        return result


async def run_fault_recovery_matrix() -> FaultRecoveryMatrix:
    """Execute every fake boundary/crash side and derive recovery from observed mutation."""

    records: list[FaultRecoveryRecord] = []
    for point in FaultPoint:
        for side in FaultSide:
            safe_code = f"{point.value}_{side.value}_injected"
            controller = fault_controller_for_test(
                FaultPlan(
                    triggers=(
                        FaultTrigger(
                            point=point,
                            call_index=1,
                            side=side,
                            action=FaultAction.RAISE,
                            safe_code=safe_code,
                        ),
                    )
                )
            )
            probe = FaultBoundaryProbe(controller, point)
            state = {"operation_applied": False}

            triggered = False
            try:
                await probe.invoke(partial(_mark_operation_applied, state))
            except InjectedFault as exc:
                triggered = exc.safe_code == safe_code
            if not triggered:
                raise RuntimeError("fault_recovery_case_did_not_trigger")
            operation_applied = state["operation_applied"]
            outcome = _recovery_outcome(point, side, operation_applied)
            records.append(
                FaultRecoveryRecord(
                    point=point,
                    side=side,
                    safe_code=safe_code,
                    operation_applied=operation_applied,
                    outcome=outcome,
                    terminal_success=False,
                    safe_recovery=True,
                )
            )
    payload = {
        "version": "fault-recovery-v1",
        "records": [item.model_dump(mode="json") for item in records],
        "case_count": len(records),
        "before_case_count": sum(item.side is FaultSide.BEFORE for item in records),
        "before_without_operation_count": sum(
            item.side is FaultSide.BEFORE and not item.operation_applied for item in records
        ),
        "triggered_count": len(records),
        "safe_recovery_count": sum(item.safe_recovery for item in records),
        "unsafe_recovery_count": sum(not item.safe_recovery for item in records),
        "false_success_count": sum(item.terminal_success for item in records),
        "unknown_effect_count": sum(
            item.outcome is FaultRecoveryOutcome.UNKNOWN_EFFECT for item in records
        ),
    }
    return FaultRecoveryMatrix.model_validate({**payload, "digest": _fault_digest(payload)})


def _recovery_outcome(
    point: FaultPoint,
    side: FaultSide,
    operation_applied: bool,
) -> FaultRecoveryOutcome:
    if side is FaultSide.BEFORE:
        if operation_applied:
            raise ValueError("before-side recovery observed an applied operation")
        if point in {
            FaultPoint.PLAN,
            FaultPoint.MEMBERSHIP,
            FaultPoint.SYNTHESIS,
            FaultPoint.VERIFIER,
        }:
            return FaultRecoveryOutcome.REJECTED_SAFE
        return FaultRecoveryOutcome.RETRY_SAFE
    if not operation_applied:
        raise ValueError("after-side recovery did not observe its operation")
    if point is FaultPoint.DATABASE:
        return FaultRecoveryOutcome.RELOAD_REQUIRED
    if point is FaultPoint.LEASE:
        return FaultRecoveryOutcome.RECLAIM_REQUIRED
    if point is FaultPoint.SLACK:
        return FaultRecoveryOutcome.UNKNOWN_EFFECT
    return FaultRecoveryOutcome.FAIL_CLOSED


async def _mark_operation_applied(state: dict[str, bool]) -> None:
    state["operation_applied"] = True


def _fault_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
