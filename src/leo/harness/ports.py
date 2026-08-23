"""Ports used by the custom runtime; adapters implement these protocols."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from pydantic import JsonValue

from leo.harness.models import (
    BudgetUsage,
    CapabilitySelection,
    CompletionProposal,
    EventDraft,
    ModelRequest,
    ModelTurnResult,
    Observation,
    Run,
    RunBundle,
    ScopeKey,
    Task,
    Thread,
    ToolExecutionContext,
    ToolOutcome,
    ToolSpec,
    TrustedScope,
    VerificationOutcome,
    VerifiedCompletion,
)


class ModelGatewayError(RuntimeError):
    """Provider-neutral model failure with a redacted operator-safe description."""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class ContextAssemblyError(RuntimeError):
    """Provider-neutral context-policy failure safe to record in the run trace."""

    def __init__(self, code: str, safe_message: str) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message


class Clock(Protocol):
    def now(self) -> datetime: ...


class IdGenerator(Protocol):
    def new(self, prefix: str) -> str: ...


class ModelGateway(Protocol):
    async def decide(self, request: ModelRequest) -> ModelTurnResult: ...


class ModelCallTranscriptSink(Protocol):
    """Best-effort dashboard-only durable store for the exact model request/response.

    Deliberately separate from the run-event log: events are capped at 8KB and
    field-allowlisted (leo.harness.persistence_rules) to keep replay deterministic and
    bounded, which a full request/response transcript cannot respect. A sink failure
    must never affect the run it's recording -- callers wrap invocations accordingly.
    """

    async def record(
        self,
        *,
        run_id: str,
        task_id: str,
        scope: ScopeKey,
        request_id: str,
        iteration: int,
        raw_request: dict[str, JsonValue],
        raw_response: dict[str, JsonValue],
        occurred_at: datetime,
    ) -> None: ...


class Tool(Protocol):
    @property
    def spec(self) -> ToolSpec: ...

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]: ...

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome: ...


class RunStore(Protocol):
    async def seed(self, thread: Thread, task: Task, run: Run) -> RunBundle: ...

    async def load(self, task_id: str, run_id: str, scope: ScopeKey) -> RunBundle: ...

    async def commit(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task: Task,
        run: Run,
        observations: tuple[Observation, ...] = (),
        events: tuple[EventDraft, ...] = (),
    ) -> RunBundle: ...

    async def complete_verified(
        self,
        *,
        expected_task_version: int,
        expected_run_version: int,
        task_id: str,
        run_id: str,
        scope: ScopeKey,
        usage: BudgetUsage,
        completion: VerifiedCompletion,
        preceding_events: tuple[EventDraft, ...] = (),
    ) -> RunBundle:
        """Atomically apply a verifier-issued completion via an intention-specific boundary."""
        ...


class ContextAssembler(Protocol):
    def assemble(self, bundle: RunBundle, tools: tuple[ToolSpec, ...]) -> ModelRequest: ...


class CapabilitySelector(Protocol):
    """Select model-visible schemas from an already phase/role-eligible registry view."""

    def select(
        self,
        *,
        bundle: RunBundle,
        trusted_scope: TrustedScope,
        available_tools: tuple[ToolSpec, ...],
    ) -> CapabilitySelection: ...


class CompletionVerifier(Protocol):
    def verify(self, proposal: CompletionProposal, bundle: RunBundle) -> VerificationOutcome: ...
