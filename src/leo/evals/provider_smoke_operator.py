"""Content-free, one-read-per-provider credentialed smoke operator.

This operator is intentionally separate from live harness composition.  It reuses the
production provider adapters and one shared :class:`ProviderGateRegistry`, but every
provider receives its own transport that can dispatch at most one HTTP request.  Raw
responses, request URLs, credentials, provider messages, and exception text never
cross the artifact or console boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

import httpx
from pydantic import Field, JsonValue, SecretStr, model_validator

from leo.config import Settings
from leo.harness.models import (
    ContractModel,
    ScopeKey,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSuccess,
    TrustedScope,
)
from leo.harness.ports import Clock, Tool
from leo.integrations.alpha_vantage import AlphaVantageQuoteTool
from leo.integrations.crypto_composition import resolve_coingecko_rest_base_url
from leo.integrations.crypto_market import (
    CoinGeckoMarketSnapshotTool,
    CoinMarketCapMarketSnapshotTool,
)
from leo.integrations.exa import ExaSearchTool
from leo.integrations.finnhub import FinnhubQuoteTool
from leo.integrations.massive import MassiveSymbolSearchTool
from leo.integrations.provider_runtime import ProviderCallGate, ProviderGateRegistry
from leo.integrations.system import SystemClock
from leo.integrations.tavily import TavilySearchTool
from leo.integrations.tickerlayer import TickerLayerSymbolSearchTool

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ProviderName = Literal[
    "finnhub",
    "tavily",
    "exa",
    "coingecko",
    "coinmarketcap",
    "alpha_vantage",
    "massive",
    "ticker_layer",
]
PROVIDER_SMOKE_VERSION: Literal["provider-smoke-v1"] = "provider-smoke-v1"
PROVIDER_ORDER: tuple[ProviderName, ...] = (
    "finnhub",
    "tavily",
    "exa",
    "coingecko",
    "coinmarketcap",
    "alpha_vantage",
    "massive",
    "ticker_layer",
)
_PROBE_TIMEOUT_SECONDS = 20.0
_SAFE_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,95}")


class ProviderSmokeStatus(StrEnum):
    SUCCESS = "success"
    NONFATAL_FAILURE = "nonfatal_failure"
    SKIPPED = "skipped"


class ProviderSmokeCase(ContractModel):
    """One content-free provider result; no provider payload field exists."""

    provider: ProviderName
    tool: str = Field(min_length=1, max_length=120)
    status: ProviderSmokeStatus
    safe_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,95}$")
    retryable: bool = False
    started_at: datetime
    completed_at: datetime
    network_attempt_count: int = Field(ge=0, le=1)
    request_bound_violation: bool = False
    outcome_digest: Sha256
    health_digest: Sha256

    @model_validator(mode="after")
    def exact_state_contract(self) -> ProviderSmokeCase:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("provider smoke timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("provider smoke completion cannot predate its start")
        if self.status is ProviderSmokeStatus.SUCCESS:
            if self.safe_code != "OK" or self.network_attempt_count != 1:
                raise ValueError("successful provider smoke requires one request and OK")
        if self.status is ProviderSmokeStatus.SKIPPED:
            if (
                self.safe_code != "CREDENTIAL_MISSING"
                or self.network_attempt_count != 0
                or self.retryable
            ):
                raise ValueError("skipped provider smoke must be a missing credential")
        if self.request_bound_violation and (
            self.status is not ProviderSmokeStatus.NONFATAL_FAILURE
            or self.safe_code != "PROVIDER_SMOKE_ATTEMPT_BOUND_EXCEEDED"
        ):
            raise ValueError("request-bound violations must be explicit nonfatal failures")
        return self


class ProviderSmokeReport(ContractModel):
    """Atomic content-free report covering the exact provider inventory."""

    version: Literal["provider-smoke-v1"] = PROVIDER_SMOKE_VERSION
    status: Literal["completed"] = "completed"
    started_at: datetime
    completed_at: datetime
    provider_count: int = Field(ge=0)
    configured_provider_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    nonfatal_failure_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)
    cases: tuple[ProviderSmokeCase, ...]
    cohort_digest: Sha256

    @model_validator(mode="after")
    def exact_cohort_contract(self) -> ProviderSmokeReport:
        if self.started_at.tzinfo is None or self.completed_at.tzinfo is None:
            raise ValueError("provider smoke report timestamps must be timezone-aware")
        if self.completed_at < self.started_at:
            raise ValueError("provider smoke report completion cannot predate its start")
        if tuple(item.provider for item in self.cases) != PROVIDER_ORDER:
            raise ValueError("provider smoke report must cover the exact ordered inventory")
        successes = sum(item.status is ProviderSmokeStatus.SUCCESS for item in self.cases)
        failures = sum(item.status is ProviderSmokeStatus.NONFATAL_FAILURE for item in self.cases)
        skipped = sum(item.status is ProviderSmokeStatus.SKIPPED for item in self.cases)
        if (
            self.provider_count != len(self.cases)
            or self.configured_provider_count != successes + failures
            or self.success_count != successes
            or self.nonfatal_failure_count != failures
            or self.skipped_count != skipped
            or successes + failures + skipped != len(self.cases)
        ):
            raise ValueError("provider smoke report counters do not match its cases")
        return self


class _ProbeToolBuilder(Protocol):
    def __call__(
        self,
        client: httpx.AsyncClient,
        api_key: str,
        clock: Clock,
        gate: ProviderCallGate,
        settings: Settings,
    ) -> Tool: ...


@dataclass(frozen=True, slots=True)
class _GatePolicy:
    max_concurrency: int
    max_calls_per_minute: int
    max_calls_per_day: int | None = None
    max_calls_per_month: int | None = None


@dataclass(frozen=True, slots=True)
class _ProbeDefinition:
    provider: ProviderName
    tool: str
    credential: SecretStr | None
    arguments: dict[str, JsonValue]
    policy: _GatePolicy
    builder: _ProbeToolBuilder


class _AttemptBoundExceeded(httpx.TransportError):
    """Internal fixed-message signal; it carries no request or provider details."""


class _SingleAttemptTransport(httpx.AsyncBaseTransport):
    """Dispatch at most one request to an underlying async transport."""

    def __init__(self, transport: httpx.AsyncBaseTransport) -> None:
        self._transport = transport
        self._lock = asyncio.Lock()
        self.network_attempt_count = 0
        self.request_bound_violation = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._lock:
            if self.network_attempt_count >= 1:
                self.request_bound_violation = True
                raise _AttemptBoundExceeded("provider smoke request bound exceeded")
            self.network_attempt_count += 1
        return await self._transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._transport.aclose()


TransportFactory = Callable[[ProviderName], httpx.AsyncBaseTransport]


async def collect_provider_smoke(
    settings: Settings,
    *,
    clock: Clock | None = None,
    transport_factory: TransportFactory | None = None,
) -> ProviderSmokeReport:
    """Run the exact provider inventory with isolated, non-propagating failures."""

    runtime_clock = clock or SystemClock()
    registry = ProviderGateRegistry(runtime_clock)
    definitions = _probe_definitions(settings)
    started_at = runtime_clock.now()
    cases = await asyncio.gather(
        *(
            _run_probe(
                definition,
                settings=settings,
                clock=runtime_clock,
                registry=registry,
                transport_factory=transport_factory,
            )
            for definition in definitions
        )
    )
    completed_at = runtime_clock.now()
    case_tuple = tuple(cases)
    successes = sum(item.status is ProviderSmokeStatus.SUCCESS for item in case_tuple)
    failures = sum(item.status is ProviderSmokeStatus.NONFATAL_FAILURE for item in case_tuple)
    skipped = sum(item.status is ProviderSmokeStatus.SKIPPED for item in case_tuple)
    cohort_digest = _digest(
        {
            "version": PROVIDER_SMOKE_VERSION,
            "started_at": started_at.isoformat(),
            "completed_at": completed_at.isoformat(),
            "cases": [item.model_dump(mode="json") for item in case_tuple],
        }
    )
    return ProviderSmokeReport(
        started_at=started_at,
        completed_at=completed_at,
        provider_count=len(case_tuple),
        configured_provider_count=successes + failures,
        success_count=successes,
        nonfatal_failure_count=failures,
        skipped_count=skipped,
        cases=case_tuple,
        cohort_digest=cohort_digest,
    )


async def _run_probe(
    definition: _ProbeDefinition,
    *,
    settings: Settings,
    clock: Clock,
    registry: ProviderGateRegistry,
    transport_factory: TransportFactory | None,
) -> ProviderSmokeCase:
    started_at = clock.now()
    api_key = _credential_value(definition.credential)
    if api_key is None:
        return _synthetic_case(
            definition,
            status=ProviderSmokeStatus.SKIPPED,
            safe_code="CREDENTIAL_MISSING",
            started_at=started_at,
            completed_at=clock.now(),
        )

    gate: ProviderCallGate | None = None
    bounded_transport: _SingleAttemptTransport | None = None
    outcome: ToolOutcome | None = None
    safe_code = "PROVIDER_SMOKE_UNEXPECTED_FAILURE"
    retryable = False
    try:
        gate = registry.get(
            provider=definition.provider,
            max_concurrency=definition.policy.max_concurrency,
            max_calls_per_minute=definition.policy.max_calls_per_minute,
            max_calls_per_day=definition.policy.max_calls_per_day,
            max_calls_per_month=definition.policy.max_calls_per_month,
        )
        underlying = (
            transport_factory(definition.provider)
            if transport_factory is not None
            else httpx.AsyncHTTPTransport(retries=0)
        )
        bounded_transport = _SingleAttemptTransport(underlying)
        timeout = httpx.Timeout(_PROBE_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(
            transport=bounded_transport,
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        ) as client:
            tool = definition.builder(client, api_key, clock, gate, settings)
            if tool.spec.name != definition.tool:
                raise ValueError("provider smoke tool identity mismatch")
            validated = tool.validate(dict(definition.arguments))
            async with asyncio.timeout(_PROBE_TIMEOUT_SECONDS):
                outcome = await tool.execute(
                    validated,
                    _execution_context(definition.provider),
                )
        if bounded_transport.request_bound_violation:
            safe_code = "PROVIDER_SMOKE_ATTEMPT_BOUND_EXCEEDED"
        elif isinstance(outcome, ToolSuccess):
            safe_code = "OK"
        elif isinstance(outcome, ToolFailure):
            safe_code = _safe_failure_code(outcome.code)
            retryable = outcome.retryable
        else:
            safe_code = "PROVIDER_SMOKE_CONTRACT_REJECTED"
    except TimeoutError:
        safe_code = "PROVIDER_SMOKE_TIMEOUT"
        retryable = True
    except (ValueError, httpx.InvalidURL):
        safe_code = "PROVIDER_CONFIGURATION_REJECTED"
    except Exception:
        # Provider exception strings are deliberately discarded.  Cancellation is
        # a BaseException and therefore continues to propagate to the operator.
        safe_code = "PROVIDER_SMOKE_UNEXPECTED_FAILURE"

    bound_violation = bool(
        bounded_transport is not None and bounded_transport.request_bound_violation
    )
    if bound_violation:
        safe_code = "PROVIDER_SMOKE_ATTEMPT_BOUND_EXCEEDED"
        retryable = False
    status = (
        ProviderSmokeStatus.SUCCESS if safe_code == "OK" else ProviderSmokeStatus.NONFATAL_FAILURE
    )
    attempts = bounded_transport.network_attempt_count if bounded_transport is not None else 0
    outcome_digest = _outcome_digest(
        provider=definition.provider,
        tool=definition.tool,
        outcome=outcome,
        status=status,
        safe_code=safe_code,
    )
    health_digest = await _health_digest(
        gate,
        provider=definition.provider,
        safe_code=safe_code,
    )
    return ProviderSmokeCase(
        provider=definition.provider,
        tool=definition.tool,
        status=status,
        safe_code=safe_code,
        retryable=retryable,
        started_at=started_at,
        completed_at=clock.now(),
        network_attempt_count=attempts,
        request_bound_violation=bound_violation,
        outcome_digest=outcome_digest,
        health_digest=health_digest,
    )


def export_provider_smoke(report: ProviderSmokeReport, destination: Path) -> None:
    """Validate and atomically export a content-free provider report."""

    validated = ProviderSmokeReport.model_validate(report.model_dump(mode="json"))
    _atomic_write(destination, validated.model_dump_json(indent=2) + "\n")


def _probe_definitions(settings: Settings) -> tuple[_ProbeDefinition, ...]:
    return (
        _ProbeDefinition(
            provider="finnhub",
            tool="market.get_quote",
            credential=settings.finnhub_api_key,
            arguments={"symbol": "NVDA"},
            policy=_GatePolicy(max_concurrency=4, max_calls_per_minute=60),
            builder=_build_finnhub,
        ),
        _ProbeDefinition(
            provider="tavily",
            tool="web.search_tavily",
            credential=settings.tavily_api_key,
            arguments={
                "query": "OpenAI official website",
                "max_results": 1,
                "search_depth": "basic",
                "topic": "general",
            },
            policy=_GatePolicy(
                max_concurrency=4,
                max_calls_per_minute=settings.tavily_max_calls_per_minute,
                max_calls_per_month=settings.tavily_max_calls_per_month,
            ),
            builder=_build_tavily,
        ),
        _ProbeDefinition(
            provider="exa",
            tool="web.search_exa",
            credential=settings.exa_api_key,
            arguments={"query": "OpenAI official website"},
            policy=_GatePolicy(max_concurrency=4, max_calls_per_minute=10),
            builder=_build_exa,
        ),
        _ProbeDefinition(
            provider="coingecko",
            tool="market.get_crypto_snapshot_coingecko",
            credential=settings.coingecko_api_key,
            arguments={"asset_id": "bitcoin", "quote_currency": "USD"},
            policy=_GatePolicy(
                max_concurrency=2,
                max_calls_per_minute=settings.coingecko_max_calls_per_minute,
            ),
            builder=_build_coingecko,
        ),
        _ProbeDefinition(
            provider="coinmarketcap",
            tool="market.get_crypto_snapshot_coinmarketcap",
            credential=settings.coin_market_cap_api_key,
            arguments={"asset_id": "bitcoin", "quote_currency": "USD"},
            policy=_GatePolicy(
                max_concurrency=2,
                max_calls_per_minute=settings.coin_market_cap_max_calls_per_minute,
            ),
            builder=_build_coinmarketcap,
        ),
        _ProbeDefinition(
            provider="alpha_vantage",
            tool="market.get_quote_alpha_vantage",
            credential=settings.alpha_vantage_api_key,
            arguments={"symbol": "NVDA"},
            policy=_GatePolicy(
                max_concurrency=1,
                max_calls_per_minute=settings.alpha_vantage_max_calls_per_minute,
                max_calls_per_day=settings.alpha_vantage_max_calls_per_day,
            ),
            builder=_build_alpha_vantage,
        ),
        _ProbeDefinition(
            provider="massive",
            tool="market.search_symbols_massive",
            credential=settings.massive_api_key,
            arguments={"query": "NVDA", "limit": 1, "market": "US"},
            policy=_GatePolicy(
                max_concurrency=2,
                max_calls_per_minute=settings.massive_max_calls_per_minute,
            ),
            builder=_build_massive,
        ),
        _ProbeDefinition(
            provider="ticker_layer",
            tool="market.search_symbols_ticker_layer",
            credential=settings.ticker_layer_api_key,
            arguments={"query": "NVDA", "limit": 1, "market": "US"},
            policy=_GatePolicy(
                max_concurrency=2,
                max_calls_per_minute=settings.ticker_layer_max_calls_per_minute,
                max_calls_per_month=settings.ticker_layer_max_calls_per_month,
            ),
            builder=_build_ticker_layer,
        ),
    )


def _build_finnhub(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    return FinnhubQuoteTool(
        client=client,
        api_key=api_key,
        clock=clock,
        base_url=settings.finnhub_base_url,
        gate=gate,
    )


def _build_tavily(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    return TavilySearchTool(
        client=client,
        api_key=api_key,
        clock=clock,
        max_calls_per_minute=settings.tavily_max_calls_per_minute,
        max_calls_per_month=settings.tavily_max_calls_per_month,
        gate=gate,
    )


def _build_exa(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    del settings
    return ExaSearchTool(client=client, api_key=api_key, clock=clock, gate=gate)


def _build_coingecko(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    base_url = resolve_coingecko_rest_base_url(
        configured_base=settings.coingecko_base_url,
        configured_endpoint=settings.coingecko_endpoint,
    )
    return CoinGeckoMarketSnapshotTool(
        client=client,
        api_key=api_key,
        clock=clock,
        base_url=base_url,
        gate=gate,
        max_calls_per_minute=settings.coingecko_max_calls_per_minute,
    )


def _build_coinmarketcap(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    return CoinMarketCapMarketSnapshotTool(
        client=client,
        api_key=api_key,
        clock=clock,
        base_url=settings.coin_market_cap_base_url,
        gate=gate,
        max_calls_per_minute=settings.coin_market_cap_max_calls_per_minute,
    )


def _build_alpha_vantage(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    del settings
    return AlphaVantageQuoteTool(client=client, api_key=api_key, clock=clock, gate=gate)


def _build_massive(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    del settings
    # Reference ticker search is documented across Stocks plans, unlike snapshots.
    return MassiveSymbolSearchTool(client=client, api_key=api_key, clock=clock, gate=gate)


def _build_ticker_layer(
    client: httpx.AsyncClient,
    api_key: str,
    clock: Clock,
    gate: ProviderCallGate,
    settings: Settings,
) -> Tool:
    del settings
    # Symbol discovery avoids the separate Fundamentals permission and quote entitlement.
    return TickerLayerSymbolSearchTool(client=client, api_key=api_key, clock=clock, gate=gate)


def _execution_context(provider: ProviderName) -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(
            namespace=ScopeKey(
                organization_id="provider-smoke",
                strategy_id="provider-smoke",
            ),
            actor_id="trusted-provider-smoke-operator",
            roles=frozenset({"operator"}),
        ),
        run_id="provider-smoke-run",
        tool_call_id=f"provider-smoke-{provider}",
    )


def _credential_value(value: SecretStr | None) -> str | None:
    if value is None:
        return None
    secret = value.get_secret_value()
    return secret if secret.strip() else None


def _safe_failure_code(value: str) -> str:
    return value if _SAFE_CODE.fullmatch(value) is not None else "PROVIDER_FAILURE_CODE_REDACTED"


def _synthetic_case(
    definition: _ProbeDefinition,
    *,
    status: ProviderSmokeStatus,
    safe_code: str,
    started_at: datetime,
    completed_at: datetime,
) -> ProviderSmokeCase:
    projection = {
        "provider": definition.provider,
        "tool": definition.tool,
        "status": status.value,
        "safe_code": safe_code,
    }
    digest = _digest({"schema": "provider-smoke-synthetic-v1", **projection})
    return ProviderSmokeCase(
        **projection,
        started_at=started_at,
        completed_at=completed_at,
        network_attempt_count=0,
        outcome_digest=digest,
        health_digest=_digest({"schema": "provider-smoke-health-v1", **projection}),
    )


def _outcome_digest(
    *,
    provider: ProviderName,
    tool: str,
    outcome: ToolOutcome | None,
    status: ProviderSmokeStatus,
    safe_code: str,
) -> str:
    if isinstance(outcome, ToolSuccess | ToolFailure):
        outcome_value: object = outcome.model_dump(mode="json")
    else:
        outcome_value = None
    return _digest(
        {
            "schema": "provider-smoke-outcome-v1",
            "provider": provider,
            "tool": tool,
            "status": status.value,
            "safe_code": safe_code,
            "outcome": outcome_value,
        }
    )


async def _health_digest(
    gate: ProviderCallGate | None,
    *,
    provider: ProviderName,
    safe_code: str,
) -> str:
    if gate is None:
        health: object = {"provider": provider, "safe_code": safe_code, "registered": False}
    else:
        try:
            health = (await gate.snapshot()).model_dump(mode="json")
        except Exception:
            health = {"provider": provider, "safe_code": safe_code, "snapshot": "unavailable"}
    return _digest({"schema": "provider-smoke-health-v1", "health": health})


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_write(destination: Path, payload: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run content-free provider smoke reads")
    parser.add_argument("--output", required=True, type=Path)
    return parser


async def _run(arguments: argparse.Namespace) -> int:
    report = await collect_provider_smoke(Settings())
    export_provider_smoke(report, arguments.output)
    print(
        json.dumps(
            {
                "status": report.status,
                "provider_count": report.provider_count,
                "configured_provider_count": report.configured_provider_count,
                "success_count": report.success_count,
                "nonfatal_failure_count": report.nonfatal_failure_count,
                "skipped_count": report.skipped_count,
                "cohort_digest": report.cohort_digest,
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return asyncio.run(_run(_parser().parse_args(argv)))
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # This is the only terminal operator failure.  Details are intentionally
        # excluded because they may contain a credential, URL, or provider body.
        print("provider_smoke_collection_failed")
        return 2


if __name__ == "__main__":
    sys.exit(main())


__all__ = (
    "PROVIDER_ORDER",
    "PROVIDER_SMOKE_VERSION",
    "ProviderSmokeCase",
    "ProviderSmokeReport",
    "ProviderSmokeStatus",
    "collect_provider_smoke",
    "export_provider_smoke",
    "main",
)
