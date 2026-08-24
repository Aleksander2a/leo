# Adding a tool

A tool is one class. There is no catalogue to register with, no keyword tags to maintain, and
no routing table to update — Leo finds a tool by embedding its description, so writing a good
description *is* the integration work.

## The interface

Three members, from `leo.agent.contracts`:

```python
class MyTool:
    @property
    def spec(self) -> ToolSpec: ...

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]: ...

    async def execute(
        self, arguments: dict[str, JsonValue], context: ToolExecutionContext
    ) -> ToolOutcome: ...
```

`execute` returns a `ToolSuccess` (data plus a `SourceRef` and an `observed_at`) or a
`ToolFailure` (a code and a message the model will read). Both are fine outcomes. So is
raising — the registry catches it. Nothing a tool does can end a run.

## A worked example

```python
from datetime import UTC, datetime

import httpx
from pydantic import JsonValue

from leo.agent.contracts import (
    Clock, RunPhase, SourceRef, ToolEffect, ToolExecutionContext,
    ToolFailure, ToolOutcome, ToolSpec, ToolSuccess,
)


class TreasuryYieldTool:
    def __init__(self, *, client: httpx.AsyncClient, clock: Clock) -> None:
        self._client = client
        self._clock = clock

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="market.get_treasury_yield",
            # Written for the model, and for the embedding index that ranks it.
            # Say what question this answers, in the words someone would ask it.
            description=(
                "Current US Treasury yield curve: the yield on 1-month through 30-year "
                "Treasury bills, notes, and bonds. Use for risk-free rates, the shape of "
                "the curve, or comparing a yield against government debt."
            ),
            domain="market",
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=20.0,
            max_result_bytes=8192,
            input_schema={
                "type": "object",
                "properties": {
                    "maturity": {
                        "type": "string",
                        "enum": ["1m", "3m", "2y", "10y", "30y"],
                        "description": "Which point on the curve.",
                    }
                },
                "required": ["maturity"],
            },
        )

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        maturity = str(arguments.get("maturity", "")).lower()
        if maturity not in {"1m", "3m", "2y", "10y", "30y"}:
            raise ValueError("maturity must be one of 1m, 3m, 2y, 10y, 30y")
        return {"maturity": maturity}

    async def execute(
        self, arguments: dict[str, JsonValue], context: ToolExecutionContext
    ) -> ToolOutcome:
        maturity = str(arguments["maturity"])
        try:
            response = await self._client.get(f"https://example.gov/yield/{maturity}")
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError as exc:
            # The model reads this and decides what to do next, so make it useful.
            return ToolFailure(
                code="TREASURY_UNAVAILABLE",
                retryable=True,
                safe_message=f"The Treasury endpoint did not respond ({type(exc).__name__}).",
            )
        return ToolSuccess(
            data={
                "maturity": maturity,
                "yield_percent": payload["yield"],
                # A plain-language line the model can quote directly is worth more
                # than another nested object it has to interpret.
                "statement": f"The {maturity} Treasury yield is {payload['yield']}%.",
            },
            source=SourceRef(
                provider="us-treasury",
                reference=f"yield:{maturity}",
                url="https://example.gov/yield",
            ),
            observed_at=self._clock.now(),
        )
```

Then add it in `leo/agent/runtime.py::build_tools`, gated on whatever credential it needs:

```python
if is_configured_secret(settings.treasury_api_key):
    tools.append(TreasuryYieldTool(client=client, clock=clock))
```

That is the whole integration. The next run embeds the new description, caches the vector in
`agent_tool_index`, and the tool becomes reachable — both from the opening roster when it
ranks highly, and via `tools.find` when it does not.

## Guidance that actually matters

**Write the description for a reader, not a schema.** It is the only thing standing between a
question and this tool. "Current US Treasury yield curve… use for risk-free rates" gets found
by "what's the risk-free rate?"; "Fetches treasury data" does not.

**Fail with a message, not an exception.** `ToolFailure.safe_message` goes straight to the
model. "Symbol NVDAA not found; check the ticker" produces a correction on the next turn.
"Error 400" produces a guess.

**Include a `statement` field when there is a sentence worth quoting.** Several adapters do
this already (`leo/providers/`), and it measurably improves how faithfully numbers survive
into the final answer.

**Set `max_result_bytes` honestly.** Oversize payloads are truncated, largest field first,
leading text kept. A tool that returns a 200KB document with a 4KB cap will have most of it
cut; give it room, or return less.

**Keep provider-domain logic pure.** Canonicalization, agreement calculations, and provenance
checks belong in `leo/providers/` as plain functions over data. The adapter in
`leo/integrations/` handles HTTP and nothing else. The quality gate enforces that
`leo/providers/` never imports the agent loop.

## MCP tools

`leo/integrations/mcp_tools.py` wraps an MCP endpoint into the same interface. An MCP tool is
not special: it satisfies `spec`/`validate`/`execute` like everything else, is ranked by the
same embedding index, and its failures come back to the model the same way.
