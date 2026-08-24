"""Bounded provider-neutral equity quote routing.

The router is part of Leo's custom harness composition, not an agent framework.
It calls providers sequentially in deterministic order, stops after two healthy
successes, and only spends calls on later providers when corroboration or failover
still needs them.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from leo.agent.contracts import (
    Clock,
    RunPhase,
    SourceRef,
    Tool,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRetryPolicy,
    ToolSpec,
    ToolSuccess,
)
from leo.providers.equity import (
    EQUITY_PROFILE_PROVIDERS,
    EQUITY_QUOTE_PROVIDERS,
    EQUITY_SEARCH_PROVIDERS,
    canonical_equity_profile_statements,
    canonical_equity_quote_disagreement_statement,
    canonical_equity_quote_statement,
    canonical_equity_quote_time_skew_statement,
    canonical_equity_search_statements,
    equity_query_hash,
    valid_equity_observed_at,
    valid_equity_profile_provenance,
    valid_equity_quote_provenance,
    valid_equity_search_provenance,
)
from leo.providers.health import ProviderHealthSnapshot


class _EquityQuoteArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")


class EquitySearchArguments(BaseModel):
    """Common bounded symbol-discovery arguments used by provider adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=120)
    limit: int = Field(default=5, ge=1, le=10)
    market: str = Field(default="US", min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")


class EquityProfileArguments(BaseModel):
    """Common company-profile lookup arguments used by provider adapters."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    symbol: str = Field(min_length=1, max_length=20, pattern=r"^[A-Z0-9.-]+$")
    market: str = Field(default="US", min_length=2, max_length=2, pattern=r"^[A-Z]{2}$")


@runtime_checkable
class _HealthAwareProvider(Protocol):
    async def provider_health(self) -> ProviderHealthSnapshot: ...


class _ProviderRoute(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def tool(self) -> Tool: ...


@dataclass(frozen=True)
class EquityQuoteRoute:
    """One named provider tool in deterministic failover order."""

    provider: str
    tool: Tool

    def __post_init__(self) -> None:
        if self.provider not in EQUITY_QUOTE_PROVIDERS:
            raise ValueError(f"unsupported equity quote provider: {self.provider}")


@dataclass(frozen=True)
class EquitySearchRoute:
    """One named symbol-search tool in deterministic failover order."""

    provider: str
    tool: Tool

    def __post_init__(self) -> None:
        if self.provider not in EQUITY_SEARCH_PROVIDERS:
            raise ValueError(f"unsupported equity search provider: {self.provider}")


@dataclass(frozen=True)
class EquityProfileRoute:
    """One named company-profile tool in deterministic failover order."""

    provider: str
    tool: Tool

    def __post_init__(self) -> None:
        if self.provider not in EQUITY_PROFILE_PROVIDERS:
            raise ValueError(f"unsupported equity profile provider: {self.provider}")


@dataclass(frozen=True)
class _SuccessfulQuote:
    route_index: int
    route: EquityQuoteRoute
    outcome: ToolSuccess
    price: int | float


class RedundantEquityQuoteTool:
    """Corroborate quotes when cheap and fail over without hiding provider failures."""

    def __init__(
        self,
        *,
        routes: tuple[EquityQuoteRoute, ...],
        clock: Clock,
        corroboration_target: int = 2,
        agreement_threshold_percent: float = 1.0,
        max_corroboration_skew_seconds: int = 900,
    ) -> None:
        if not routes or len(routes) > 4:
            raise ValueError("equity quote routing requires one to four providers")
        providers = tuple(route.provider for route in routes)
        if len(providers) != len(set(providers)):
            raise ValueError("equity quote routes must use distinct providers")
        if corroboration_target not in {1, 2}:
            raise ValueError("equity quote corroboration target must be one or two")
        if (
            not math.isfinite(agreement_threshold_percent)
            or not 0 <= agreement_threshold_percent <= 100
        ):
            raise ValueError("equity quote agreement threshold must be between 0 and 100")
        if max_corroboration_skew_seconds < 0:
            raise ValueError("equity quote corroboration skew must be nonnegative")
        self._routes = routes
        self._clock = clock
        self._corroboration_target = min(corroboration_target, len(routes))
        self._agreement_threshold_percent = agreement_threshold_percent
        self._max_corroboration_skew_seconds = max_corroboration_skew_seconds
        self._spec = ToolSpec(
            name="market.get_quote",
            version="2.0.0",
            description=(
                "Return one fresh normalized equity quote through bounded provider-neutral "
                "corroboration and failover. At most two successful feeds are sampled; later "
                "configured feeds are called only after an earlier failure or when a second "
                "corroborating quote is still needed."
            ),
            domain="MARKET",
            input_schema=_EquityQuoteArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=30.0,
            # The adapter owns bounded failover. Coordinator retries would multiply
            # external calls and weaken the explicit provider-attempt accounting.
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=16_384,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def provider_order(self) -> tuple[str, ...]:
        return tuple(route.provider for route in self._routes)

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        parsed = _EquityQuoteArguments.model_validate(arguments)
        return {"symbol": parsed.symbol}

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = _EquityQuoteArguments.model_validate(arguments)
        successes: list[_SuccessfulQuote] = []
        attempts: list[dict[str, JsonValue]] = []
        failures: list[tuple[str, ToolFailure]] = []
        unavailable: list[tuple[str, ToolFailure]] = []

        for index, route in enumerate(self._routes):
            if len(successes) >= self._corroboration_target:
                attempts.append(_skipped_attempt(route.provider, "CORROBORATION_TARGET_REACHED"))
                continue

            health_failure = await _health_skip(route)
            if health_failure is not None:
                attempts.append(_skipped_attempt(route.provider, health_failure.code))
                unavailable.append((route.provider, health_failure))
                continue

            outcome = await _contained_execute(
                route=route,
                arguments={"symbol": parsed.symbol},
                context=context,
            )
            if isinstance(outcome, ToolFailure):
                failures.append((route.provider, outcome))
                unavailable.append((route.provider, outcome))
                attempts.append(
                    {
                        "provider": route.provider,
                        "status": "failure",
                        "code": outcome.code,
                        "retryable": outcome.retryable,
                    }
                )
                continue

            price = _validated_quote_price(
                outcome,
                expected_provider=route.provider,
                expected_symbol=parsed.symbol,
                now=self._clock.now(),
            )
            if price is None:
                failure = ToolFailure(
                    code="EQUITY_PROVIDER_CONTRACT_VIOLATION",
                    safe_message=(
                        f"{route.provider} returned a quote that violated Leo's normalized "
                        "equity evidence contract."
                    ),
                )
                failures.append((route.provider, failure))
                unavailable.append((route.provider, failure))
                attempts.append(
                    {
                        "provider": route.provider,
                        "status": "failure",
                        "code": failure.code,
                        "retryable": False,
                    }
                )
                continue

            successes.append(
                _SuccessfulQuote(
                    route_index=index,
                    route=route,
                    outcome=outcome,
                    price=price,
                )
            )
            attempts.append(
                {
                    "provider": route.provider,
                    "status": "success",
                    "reference": outcome.source.reference,
                    "price": price,
                    "as_of": outcome.observed_at.isoformat(),
                }
            )

        if not successes:
            return _all_failed(unavailable)

        # Prefer the freshest market timestamp. Route order is the deterministic
        # tie-breaker so identical fixtures never change selection across runs.
        selected = max(
            successes,
            key=lambda item: (item.outcome.observed_at, -item.route_index),
        )
        data: dict[str, JsonValue] = dict(selected.outcome.data)
        statement = canonical_equity_quote_statement(data)
        if statement is None:  # guarded by _validated_quote_price; defensive only
            return ToolFailure(
                code="EQUITY_PROVIDER_CONTRACT_VIOLATION",
                safe_message="The selected quote could not produce a canonical statement.",
            )
        quote_rows: list[JsonValue] = [
            {
                "provider": item.route.provider,
                "reference": item.outcome.source.reference,
                "price": item.price,
                "as_of": item.outcome.observed_at.isoformat(),
                "expires_at": (
                    item.outcome.expires_at.isoformat()
                    if item.outcome.expires_at is not None
                    else None
                ),
            }
            for item in successes
        ]
        attempt_rows: list[JsonValue] = [item for item in attempts]
        data.update(
            {
                "statements": [statement],
                "selected_provider": selected.route.provider,
                "selected_reference": selected.outcome.source.reference,
                "selection_policy": "freshest_then_provider_order",
                "provider_order": list(self.provider_order),
                "corroboration_target": self._corroboration_target,
                "agreement_threshold_percent": self._agreement_threshold_percent,
                "corroboration_skew_threshold_seconds": (self._max_corroboration_skew_seconds),
                "provider_call_bound": len(self._routes),
                "provider_attempts": attempt_rows,
                "provider_attempt_count": sum(item.get("status") != "skipped" for item in attempts),
                "provider_success_count": len(successes),
                "provider_failure_count": len(failures),
                "provider_skipped_count": sum(item.get("status") == "skipped" for item in attempts),
                "provider_health_skip_count": sum(
                    item.get("status") == "skipped"
                    and item.get("code") != "CORROBORATION_TARGET_REACHED"
                    for item in attempts
                ),
                "fallback_used": bool(failures) or selected.route_index > 0,
                "provider_quotes": quote_rows,
            }
        )
        if len(successes) > 1:
            prices = [float(item.price) for item in successes]
            observed = [item.outcome.observed_at for item in successes]
            low = min(prices)
            data["price_disagreement_percent"] = (max(prices) - low) / low * 100
            data["freshness_spread_seconds"] = (max(observed) - min(observed)).total_seconds()
            disagreement = data["price_disagreement_percent"]
            assert isinstance(disagreement, float)
            agrees = disagreement <= self._agreement_threshold_percent
            freshness_spread = data["freshness_spread_seconds"]
            assert isinstance(freshness_spread, float)
            aligned = freshness_spread <= self._max_corroboration_skew_seconds
            data["temporally_aligned"] = aligned
            if not agrees:
                data["agreement_status"] = "disagree"
            elif not aligned:
                data["agreement_status"] = "time_skewed"
            else:
                data["agreement_status"] = "agree"
            data["corroborated"] = agrees and aligned
        else:
            data["agreement_status"] = "single_source"
            data["temporally_aligned"] = False
            data["corroborated"] = False
        disagreement_statement = canonical_equity_quote_disagreement_statement(data)
        time_skew_statement = canonical_equity_quote_time_skew_statement(data)
        diagnostic_statements = [
            item for item in (disagreement_statement, time_skew_statement) if item is not None
        ]
        data["statements"] = [statement, *diagnostic_statements]
        return ToolSuccess(
            data=data,
            source=SourceRef(
                provider=selected.outcome.source.provider,
                reference=selected.outcome.source.reference,
                url=selected.outcome.source.url,
            ),
            observed_at=selected.outcome.observed_at,
            expires_at=min(
                (
                    item.outcome.expires_at
                    for item in successes
                    if item.outcome.expires_at is not None
                ),
                default=None,
            ),
        )


class RedundantEquitySymbolSearchTool:
    """Fail over bounded symbol search to the first provider with useful matches."""

    def __init__(
        self,
        *,
        routes: tuple[EquitySearchRoute, ...],
        clock: Clock,
    ) -> None:
        _validate_read_routes(routes, maximum=3)
        self._routes = routes
        self._clock = clock
        self._spec = ToolSpec(
            name="market.search_equity_symbols",
            version="1.0.0",
            description=(
                "Search equity symbols through deterministic bounded provider failover. "
                "Stops at the first normalized provider result with matches; later providers "
                "are tried only after failure, ineligibility, or an empty result."
            ),
            domain="MARKET",
            input_schema=EquitySearchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=30.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=24_576,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def provider_order(self) -> tuple[str, ...]:
        return tuple(route.provider for route in self._routes)

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return EquitySearchArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = EquitySearchArguments.model_validate(arguments)
        routed_arguments = parsed.model_dump(mode="json")
        attempts: list[dict[str, JsonValue]] = []
        failures: list[tuple[str, ToolFailure]] = []
        unavailable: list[tuple[str, ToolFailure]] = []
        empty_successes: list[tuple[int, EquitySearchRoute, ToolSuccess]] = []

        for index, route in enumerate(self._routes):
            health_failure = await _health_skip(route)
            if health_failure is not None:
                attempts.append(_skipped_attempt(route.provider, health_failure.code))
                unavailable.append((route.provider, health_failure))
                continue
            outcome = await _contained_execute(
                route=route,
                arguments=routed_arguments,
                context=context,
            )
            if isinstance(outcome, ToolFailure):
                failures.append((route.provider, outcome))
                unavailable.append((route.provider, outcome))
                attempts.append(_failure_attempt(route.provider, outcome))
                continue
            result_count = _validated_search_result_count(
                outcome,
                expected_provider=route.provider,
                query=parsed.query,
                market=parsed.market,
                now=self._clock.now(),
            )
            if result_count is None:
                failure = _contract_failure(route.provider, "symbol search")
                failures.append((route.provider, failure))
                unavailable.append((route.provider, failure))
                attempts.append(_failure_attempt(route.provider, failure))
                continue
            attempts.append(_success_attempt(route.provider, outcome, result_count=result_count))
            if result_count == 0:
                empty_successes.append((index, route, outcome))
                continue
            _append_remaining_skips(
                attempts,
                self._routes[index + 1 :],
                code="USEFUL_RESULT_FOUND",
            )
            return _routed_read_success(
                outcome=outcome,
                provider_order=self.provider_order,
                attempts=attempts,
                failure_count=len(failures),
                selected_index=index,
            )

        if empty_successes:
            index, _route, outcome = empty_successes[0]
            return _routed_read_success(
                outcome=outcome,
                provider_order=self.provider_order,
                attempts=attempts,
                failure_count=len(failures),
                selected_index=index,
            )
        return _all_read_failed("EQUITY_SYMBOL_SEARCH", unavailable)


class RedundantEquityProfileTool:
    """Fail over company profile lookup to the first normalized success."""

    def __init__(
        self,
        *,
        routes: tuple[EquityProfileRoute, ...],
        clock: Clock,
    ) -> None:
        _validate_read_routes(routes, maximum=4)
        self._routes = routes
        self._clock = clock
        self._spec = ToolSpec(
            name="market.get_equity_profile",
            version="1.0.0",
            description=(
                "Return a canonical equity company profile through deterministic bounded "
                "provider failover. Permission denial or provider failure cannot abort while "
                "another configured profile source remains eligible."
            ),
            domain="MARKET",
            input_schema=EquityProfileArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            timeout_seconds=30.0,
            retry=ToolRetryPolicy(max_attempts=1),
            max_result_bytes=16_384,
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    @property
    def provider_order(self) -> tuple[str, ...]:
        return tuple(route.provider for route in self._routes)

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return EquityProfileArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        parsed = EquityProfileArguments.model_validate(arguments)
        routed_arguments = parsed.model_dump(mode="json")
        attempts: list[dict[str, JsonValue]] = []
        failures: list[tuple[str, ToolFailure]] = []
        unavailable: list[tuple[str, ToolFailure]] = []

        for index, route in enumerate(self._routes):
            health_failure = await _health_skip(route)
            if health_failure is not None:
                attempts.append(_skipped_attempt(route.provider, health_failure.code))
                unavailable.append((route.provider, health_failure))
                continue
            outcome = await _contained_execute(
                route=route,
                arguments=routed_arguments,
                context=context,
            )
            if isinstance(outcome, ToolFailure):
                failures.append((route.provider, outcome))
                unavailable.append((route.provider, outcome))
                attempts.append(_failure_attempt(route.provider, outcome))
                continue
            if not _valid_profile_result(
                outcome,
                expected_provider=route.provider,
                expected_symbol=parsed.symbol,
                now=self._clock.now(),
            ):
                failure = _contract_failure(route.provider, "company profile")
                failures.append((route.provider, failure))
                unavailable.append((route.provider, failure))
                attempts.append(_failure_attempt(route.provider, failure))
                continue
            attempts.append(_success_attempt(route.provider, outcome))
            _append_remaining_skips(
                attempts,
                self._routes[index + 1 :],
                code="NORMALIZED_RESULT_FOUND",
            )
            return _routed_read_success(
                outcome=outcome,
                provider_order=self.provider_order,
                attempts=attempts,
                failure_count=len(failures),
                selected_index=index,
            )
        return _all_read_failed("EQUITY_PROFILE", unavailable)


async def _health_skip(route: _ProviderRoute) -> ToolFailure | None:
    if not isinstance(route.tool, _HealthAwareProvider):
        return None
    try:
        snapshot = await route.tool.provider_health()
    except Exception:
        return ToolFailure(
            code="EQUITY_PROVIDER_HEALTH_CHECK_FAILED",
            retryable=True,
            safe_message=f"{route.provider} health could not be checked safely.",
        )
    if snapshot.provider.replace("_", "-") != route.provider:
        return ToolFailure(
            code="EQUITY_PROVIDER_HEALTH_CONTRACT_VIOLATION",
            safe_message=f"{route.provider} returned mismatched health authority.",
        )
    if snapshot.status != "rate_limited":
        return None
    return ToolFailure(
        code=snapshot.last_failure_code or f"{snapshot.provider.upper()}_RATE_LIMITED",
        retryable=True,
        safe_message=f"{route.provider} is locally ineligible because its quota is limited.",
    )


async def _contained_execute(
    *,
    route: _ProviderRoute,
    arguments: dict[str, JsonValue],
    context: ToolExecutionContext,
) -> ToolOutcome:
    try:
        validated = route.tool.validate(arguments)
        return await route.tool.execute(validated, context)
    except Exception:
        # Provider bugs are isolated just like transport failures. Never include the
        # exception text: it may carry a credential-bearing request representation.
        return ToolFailure(
            code="EQUITY_PROVIDER_UNEXPECTED_ERROR",
            retryable=True,
            safe_message=f"{route.provider} failed inside its adapter boundary.",
        )


def _validated_quote_price(
    outcome: ToolSuccess,
    *,
    expected_provider: str,
    expected_symbol: str,
    now: datetime,
) -> int | float | None:
    symbol = outcome.data.get("symbol")
    price = outcome.data.get("price")
    if (
        outcome.source.provider != expected_provider
        or outcome.data.get("provider") != expected_provider
        or symbol != expected_symbol
        or not isinstance(price, int | float)
        or isinstance(price, bool)
        or not math.isfinite(float(price))
        or price <= 0
        or not valid_equity_quote_provenance(
            provider=outcome.source.provider,
            reference=outcome.source.reference,
            symbol=expected_symbol,
            observed_at=outcome.observed_at,
        )
        or not valid_equity_observed_at(outcome.data, outcome.observed_at)
        or outcome.expires_at is None
        or outcome.expires_at <= now
    ):
        return None
    if canonical_equity_quote_statement(outcome.data) is None:
        return None
    return price


def _skipped_attempt(provider: str, code: str) -> dict[str, JsonValue]:
    return {
        "provider": provider,
        "status": "skipped",
        "code": code,
        "retryable": False,
    }


def _failure_attempt(provider: str, failure: ToolFailure) -> dict[str, JsonValue]:
    return {
        "provider": provider,
        "status": "failure",
        "code": failure.code,
        "retryable": failure.retryable,
    }


def _success_attempt(
    provider: str,
    outcome: ToolSuccess,
    *,
    result_count: int | None = None,
) -> dict[str, JsonValue]:
    attempt: dict[str, JsonValue] = {
        "provider": provider,
        "status": "success",
        "reference": outcome.source.reference,
        "as_of": outcome.observed_at.isoformat(),
    }
    if result_count is not None:
        attempt["result_count"] = result_count
    return attempt


def _all_failed(failures: list[tuple[str, ToolFailure]]) -> ToolFailure:
    accounting = ", ".join(f"{provider}:{failure.code}" for provider, failure in failures)
    return ToolFailure(
        code="EQUITY_QUOTE_ALL_PROVIDERS_FAILED",
        retryable=any(failure.retryable for _, failure in failures),
        safe_message=(
            "No eligible equity quote provider succeeded. Provider accounting: "
            f"{accounting or 'none-eligible'}."
        )[:1_000],
    )


def _validated_search_result_count(
    outcome: ToolSuccess,
    *,
    expected_provider: str,
    query: str,
    market: str,
    now: datetime,
) -> int | None:
    provider = outcome.data.get("provider")
    query_digest = outcome.data.get("query_hash")
    result_count = outcome.data.get("result_count")
    statements = outcome.data.get("statements")
    canonical = canonical_equity_search_statements(outcome.data)
    if (
        outcome.source.provider != expected_provider
        or provider != expected_provider
        or outcome.data.get("query") != query
        or outcome.data.get("requested_market") != market
        or not isinstance(query_digest, str)
        or query_digest != equity_query_hash(query)
        or not isinstance(result_count, int)
        or isinstance(result_count, bool)
        or canonical is None
        or statements != list(canonical)
        or not valid_equity_search_provenance(
            provider=expected_provider,
            reference=outcome.source.reference,
            query_hash=query_digest,
        )
        or (outcome.expires_at is not None and outcome.expires_at <= now)
    ):
        return None
    return result_count


def _valid_profile_result(
    outcome: ToolSuccess,
    *,
    expected_provider: str,
    expected_symbol: str,
    now: datetime,
) -> bool:
    provider_symbol = outcome.data.get("provider_symbol")
    canonical = canonical_equity_profile_statements(outcome.data)
    return bool(
        outcome.source.provider == expected_provider
        and outcome.data.get("provider") == expected_provider
        and outcome.data.get("symbol") == expected_symbol
        and isinstance(provider_symbol, str)
        and canonical is not None
        and outcome.data.get("statements") == list(canonical)
        and valid_equity_profile_provenance(
            provider=expected_provider,
            reference=outcome.source.reference,
            provider_symbol=provider_symbol,
        )
        and valid_equity_observed_at(outcome.data, outcome.observed_at)
        and (outcome.expires_at is None or outcome.expires_at > now)
    )


def _validate_read_routes(routes: tuple[_ProviderRoute, ...], *, maximum: int) -> None:
    if not routes or len(routes) > maximum:
        raise ValueError(f"equity read routing requires one to {maximum} providers")
    providers = tuple(route.provider for route in routes)
    if len(providers) != len(set(providers)):
        raise ValueError("equity read routes must use distinct providers")


def _append_remaining_skips(
    attempts: list[dict[str, JsonValue]],
    routes: tuple[_ProviderRoute, ...],
    *,
    code: str,
) -> None:
    attempts.extend(_skipped_attempt(route.provider, code) for route in routes)


def _routed_read_success(
    *,
    outcome: ToolSuccess,
    provider_order: tuple[str, ...],
    attempts: list[dict[str, JsonValue]],
    failure_count: int,
    selected_index: int,
) -> ToolSuccess:
    data: dict[str, JsonValue] = dict(outcome.data)
    attempt_rows: list[JsonValue] = [item for item in attempts]
    data.update(
        {
            "selected_provider": outcome.source.provider,
            "selected_reference": outcome.source.reference,
            "selection_policy": "first_normalized_success_in_provider_order",
            "provider_order": list(provider_order),
            "provider_attempts": attempt_rows,
            "provider_attempt_count": sum(item.get("status") != "skipped" for item in attempts),
            "provider_failure_count": failure_count,
            "provider_skipped_count": sum(item.get("status") == "skipped" for item in attempts),
            "provider_health_skip_count": sum(
                item.get("status") == "skipped"
                and item.get("code") not in {"NORMALIZED_RESULT_FOUND", "USEFUL_RESULT_FOUND"}
                for item in attempts
            ),
            "fallback_used": failure_count > 0 or selected_index > 0,
        }
    )
    return ToolSuccess(
        data=data,
        source=outcome.source,
        observed_at=outcome.observed_at,
        expires_at=outcome.expires_at,
    )


def _contract_failure(provider: str, kind: str) -> ToolFailure:
    return ToolFailure(
        code="EQUITY_PROVIDER_CONTRACT_VIOLATION",
        safe_message=(
            f"{provider} returned {kind} data that violated Leo's normalized evidence contract."
        ),
    )


def _all_read_failed(
    operation: str,
    failures: list[tuple[str, ToolFailure]],
) -> ToolFailure:
    accounting = ", ".join(f"{provider}:{failure.code}" for provider, failure in failures)
    return ToolFailure(
        code=f"{operation}_ALL_PROVIDERS_FAILED",
        retryable=any(failure.retryable for _, failure in failures),
        safe_message=(
            "No eligible equity provider returned a normalized result. Provider accounting: "
            f"{accounting or 'none-eligible'}."
        )[:1_000],
    )


def bounded_provider_text(value: object, *, limit: int) -> str | None:
    """Strip control characters and cap provider-controlled text."""

    if not isinstance(value, str) or limit < 1:
        return None
    cleaned = re.sub(r"[\x00-\x1f\x7f]", " ", value)
    normalized = " ".join(cleaned.split())[:limit]
    return normalized or None


def finite_provider_number(value: object, *, positive: bool = False) -> float | None:
    """Parse finite provider numbers without accepting booleans or sentinel text."""

    if isinstance(value, bool):
        return None
    try:
        number = float(value) if isinstance(value, str | int | float) else math.nan
    except ValueError:
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def safe_provider_request_id(value: object) -> str | None:
    """Return a bounded printable request identifier, never a request URL."""

    if not isinstance(value, str):
        return None
    normalized = "".join(character for character in value.strip() if character.isprintable())
    return normalized[:128] or None


__all__ = (
    "EquityProfileArguments",
    "EquityProfileRoute",
    "EquityQuoteRoute",
    "EquitySearchArguments",
    "EquitySearchRoute",
    "RedundantEquityProfileTool",
    "RedundantEquityQuoteTool",
    "RedundantEquitySymbolSearchTool",
    "bounded_provider_text",
    "finite_provider_number",
    "safe_provider_request_id",
)
