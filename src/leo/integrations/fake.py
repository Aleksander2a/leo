"""Deterministic adapters for offline smoke tests and eval fixtures."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    ModelDecision,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRequest,
    ToolRequests,
    ToolRetryPolicy,
    ToolSpec,
    ToolSuccess,
)


class FixedClock:
    def __init__(self, value: datetime | None = None) -> None:
        self._value = value or datetime(2026, 8, 20, 12, 0, tzinfo=UTC)

    def now(self) -> datetime:
        return self._value

    def advance(self, *, seconds: float) -> None:
        self._value += timedelta(seconds=seconds)


class SequentialIdGenerator:
    def __init__(self) -> None:
        self._counts: defaultdict[str, int] = defaultdict(int)

    def new(self, prefix: str) -> str:
        self._counts[prefix] += 1
        return f"{prefix}-{self._counts[prefix]:03d}"


class QuoteArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")


class FakeQuoteTool:
    def __init__(
        self,
        clock: FixedClock,
        *,
        retry: ToolRetryPolicy | None = None,
    ) -> None:
        self._clock = clock
        self._calls = 0
        self._spec = ToolSpec(
            name="market.get_quote",
            description="Return the current market quote for one normalized symbol.",
            domain="MARKET",
            input_schema=QuoteArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=1.0,
            retry=retry or ToolRetryPolicy(),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def calls(self) -> int:
        return self._calls

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = QuoteArguments.model_validate(arguments)
        return {"symbol": parsed.symbol}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del context
        self._calls += 1
        symbol = arguments["symbol"]
        if not isinstance(symbol, str):
            raise TypeError("validated symbol must be a string")
        return ToolSuccess(
            data={"symbol": symbol, "price": 181.25, "currency": "USD"},
            source=SourceRef(
                provider="fixture",
                reference=f"fixture-quote-{symbol}",
                url="https://example.invalid/leo-fixtures/quotes",
            ),
            observed_at=self._clock.now(),
            expires_at=self._clock.now() + timedelta(minutes=5),
        )


class FixtureModel:
    """Wrap deterministic decisions in the same envelope as provider adapters."""

    async def decide(self, request: ModelRequest) -> ModelTurnResult:
        decision = await self._decide(request)
        return ModelTurnResult(
            decision=decision,
            provider="fixture",
            model=type(self).__name__,
            request_id=f"fixture-{request.iteration + 1:03d}",
            finish_reason="tool_calls" if isinstance(decision, ToolRequests) else "stop",
            usage=ModelUsage(),
        )

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        raise NotImplementedError


class ScriptedQuoteModel(FixtureModel):
    """Two-turn fixture: request a quote, then propose a cited completion."""

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        if not request.observations:
            return ToolRequests(
                calls=(
                    ToolRequest(
                        id=f"call-{request.iteration + 1:03d}",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            )
        observation = request.observations[-1]
        symbol = observation.data.get("symbol")
        price = observation.data.get("price")
        # Keep the exact value at sentence end to exercise numeric token boundaries.
        statement = f"The current quote for {symbol} is {price}."
        return CompletionProposal(
            answer=statement,
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=statement,
                    observation_ids=(observation.id,),
                ),
            ),
        )


class FabricatingModel(FixtureModel):
    """Adversarial fixture that cites an observation the harness never created."""

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        del request
        return CompletionProposal(
            answer="NVDA is definitely 999 USD.",
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement="NVDA is definitely 999 USD.",
                    observation_ids=("obs-model-invented",),
                ),
            ),
        )


class MisstatingQuoteModel(FixtureModel):
    """Adversarial fixture that cites a real quote but changes its value."""

    def __init__(self, stated_price: str = "999") -> None:
        self._stated_price = stated_price

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        if not request.observations:
            return ToolRequests(
                calls=(
                    ToolRequest(
                        id="call-real-quote",
                        name="market.get_quote",
                        arguments={"symbol": "NVDA"},
                    ),
                )
            )
        observation = request.observations[-1]
        statement = f"NVDA is quoted at {self._stated_price} USD."
        return CompletionProposal(
            answer=statement,
            claims=(
                CandidateClaim(
                    kind=ClaimKind.SOURCE_CLAIM,
                    statement=statement,
                    observation_ids=(observation.id,),
                ),
            ),
        )


class EndlessQuoteModel(FixtureModel):
    """Adversarial fixture used to prove that model/tool budgets stop the loop."""

    async def _decide(self, request: ModelRequest) -> ModelDecision:
        return ToolRequests(
            calls=(
                ToolRequest(
                    id=f"call-{request.iteration + 1:03d}",
                    name="market.get_quote",
                    arguments={"symbol": "NVDA"},
                ),
            )
        )


class FakeWriteTool:
    """A harmless effect-labelled fixture used to test phase denial."""

    def __init__(
        self,
        clock: FixedClock,
        *,
        allowed_phases: frozenset[RunPhase] | None = None,
    ) -> None:
        self._clock = clock
        self._calls = 0
        self._spec = ToolSpec(
            name="fixture.write",
            description="Record a fixture write for effect-policy tests.",
            domain="FIXTURE",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            effect=ToolEffect.WRITE,
            allowed_phases=allowed_phases or frozenset({RunPhase.EXECUTION}),
            timeout_seconds=1.0,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def calls(self) -> int:
        return self._calls

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if arguments:
            raise ValueError("fixture.write takes no arguments")
        return {}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolSuccess:
        del arguments, context
        self._calls += 1
        return ToolSuccess(
            data={"status": "written"},
            source=SourceRef(provider="fixture", reference="fixture-write"),
            observed_at=self._clock.now(),
        )


class SlowModel(FixtureModel):
    async def _decide(self, request: ModelRequest) -> ModelDecision:
        del request
        await asyncio.sleep(60)
        raise AssertionError("run deadline should cancel the model call")


class FlakyQuoteTool:
    """Return one retryable failure, then delegate to the fixed quote tool."""

    def __init__(self, clock: FixedClock) -> None:
        self._delegate = FakeQuoteTool(clock, retry=ToolRetryPolicy(max_attempts=2))
        self._calls = 0

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
        self._calls += 1
        if self._calls == 1:
            return ToolFailure(
                code="FIXTURE_RETRYABLE",
                retryable=True,
                safe_message="Fixture provider is temporarily unavailable.",
            )
        return await self._delegate.execute(arguments, context)


class AlwaysFailTool:
    def __init__(self) -> None:
        self._spec = ToolSpec(
            name="market.fail",
            description="Return a deterministic non-retryable failure.",
            domain="MARKET",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=1.0,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        if arguments:
            raise ValueError("market.fail takes no arguments")
        return {}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolFailure:
        del arguments, context
        return ToolFailure(
            code="FIXTURE_FAILURE",
            retryable=False,
            safe_message="Fixture failure.",
        )


class TwoToolBatchModel(FixtureModel):
    async def _decide(self, request: ModelRequest) -> ModelDecision:
        return ToolRequests(
            calls=(
                ToolRequest(
                    id=f"call-quote-{request.iteration}",
                    name="market.get_quote",
                    arguments={"symbol": "NVDA"},
                ),
                ToolRequest(
                    id=f"call-fail-{request.iteration}",
                    name="market.fail",
                    arguments={},
                ),
            )
        )
