"""Value types shared by the agent loop and the provider tool adapters.

This module is deliberately small. It holds only what crosses the boundary
between the loop and a tool: what a tool *is* (``ToolSpec``), what it returns
(``ToolSuccess`` / ``ToolFailure``), who is asking (``Scope``), and where a
piece of data came from (``SourceRef``).

There is no contract here describing what an *answer* must look like. The
model writes the answer; nothing in this package grades it.
"""

from __future__ import annotations

import typing
from datetime import datetime
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue

NonEmptyStr = typing.Annotated[str, Field(min_length=1, pattern=r"\S")]


class ContractModel(BaseModel):
    """Strict immutable base for state crossing a component boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class Scope(ContractModel):
    """The isolation key for one conversation.

    Everything durable -- messages, runs, memories -- is stored under a scope
    and read back under the same scope. That single key, applied in SQL, is the
    whole of Leo's isolation between channels and DMs: a DM's memories are
    simply not in the result set when a channel asks.

    ``key`` is stable and opaque (e.g. ``slack:T123:D456``). ``actor_id`` is the
    human who spoke; it is recorded, never used to widen a read.
    """

    key: NonEmptyStr
    actor_id: NonEmptyStr = "unknown"
    display: str = ""

    @property
    def is_direct_message(self) -> bool:
        return self.key.rsplit(":", 1)[-1].startswith("D")


class SourceRef(ContractModel):
    provider: NonEmptyStr
    reference: NonEmptyStr
    url: str | None = None


class ToolEffect(StrEnum):
    READ = "read"
    STATE_MUTATION = "state_mutation"
    WRITE = "write"


class RunPhase(StrEnum):
    """Retained because provider adapters declare it; the loop does not gate on it."""

    RESEARCH = "research"
    PROPOSAL = "proposal"
    POLICY = "policy"
    APPROVAL = "approval"
    EXECUTION = "execution"
    VERIFICATION = "verification"


class ToolRetryPolicy(ContractModel):
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
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    retry: ToolRetryPolicy = Field(default_factory=ToolRetryPolicy)
    estimated_cost: float = Field(default=0.0, ge=0)
    max_result_bytes: int = Field(default=8192, ge=1, le=1_048_576)
    required_roles: frozenset[str] = Field(default_factory=frozenset)


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


ToolOutcome = typing.Annotated[ToolSuccess | ToolFailure, Field(discriminator="kind")]


class ToolExecutionContext(ContractModel):
    """Everything a tool is allowed to know about the caller."""

    trusted_scope: Scope
    run_id: NonEmptyStr
    tool_call_id: NonEmptyStr


class Clock(Protocol):
    def now(self) -> datetime: ...


class Tool(Protocol):
    """The one interface every capability implements.

    Provider adapters in ``leo.integrations`` already satisfy this, unchanged.
    """

    @property
    def spec(self) -> ToolSpec: ...

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]: ...

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome: ...


class ProviderError(RuntimeError):
    """A model-provider call failed in a way worth reporting verbatim to operators."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
