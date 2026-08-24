"""Provider-neutral equity evidence contracts and canonical statements.

This module is harness-owned so verification never needs to import a concrete
integration.  Adapters emit the small normalized shapes below; the verifier
rebuilds canonical statements and provenance from the same functions.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue

_SYMBOL = re.compile(r"[A-Z][A-Z0-9.-]{0,19}")
_QUALIFIED_SYMBOL = re.compile(r"[A-Z]{2}:[A-Z0-9][A-Z0-9.-]{0,19}")
_CURRENCY = re.compile(r"[A-Z]{3}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_INTEGER = re.compile(r"[1-9]\d*")

EQUITY_QUOTE_PROVIDERS = frozenset({"finnhub", "alpha-vantage", "massive", "ticker-layer"})
EQUITY_SEARCH_PROVIDERS = frozenset({"alpha-vantage", "massive", "ticker-layer"})
EQUITY_PROFILE_PROVIDERS = frozenset({"alpha-vantage", "finnhub", "massive", "ticker-layer"})

_PROVIDER_LABELS = {
    "alpha-vantage": "Alpha Vantage",
    "finnhub": "Finnhub",
    "massive": "Massive",
    "ticker-layer": "TickerLayer",
}


def equity_query_hash(query: str) -> str:
    """Return the stable digest used to bind search results to their query."""

    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def canonical_equity_quote_statement(data: dict[str, JsonValue]) -> str | None:
    """Rebuild the only canonical current-price statement from normalized data."""

    provider = data.get("provider")
    symbol = data.get("symbol")
    price = _finite_positive_number(data.get("price"))
    currency = data.get("currency")
    if (
        not isinstance(symbol, str)
        or _SYMBOL.fullmatch(symbol) is None
        or price is None
        or (currency is not None and not _valid_currency(currency))
    ):
        return None
    suffix = f" {currency}" if isinstance(currency, str) else ""
    if provider is not None:
        as_of = data.get("as_of")
        if (
            not isinstance(provider, str)
            or provider not in EQUITY_QUOTE_PROVIDERS
            or not isinstance(as_of, str)
            or _aware_datetime(as_of) is None
        ):
            return None
        # Finnhub's compatibility sentence remains compact, while its adapter and
        # verifier still bind the exact provider/reference/as_of tuple. Providers
        # with delayed, entitlement-dependent, or indicative semantics must surface
        # those caveats directly in the canonical answer text below.
        if provider == "finnhub":
            return f"{symbol} is quoted at {format(price, 'g')}{suffix}."
        label = _PROVIDER_LABELS[provider]
        statement = (
            f"{label} reports {symbol} quoted at {format(price, 'g')}{suffix} as of {as_of}."
        )
        if provider == "alpha-vantage":
            if not (
                data.get("data_freshness") == "end_of_day"
                and data.get("market_data_entitlement") == "historical"
            ):
                return None
            return (
                f"{statement} Its GLOBAL_QUOTE evidence is end-of-day historical data "
                "without a realtime entitlement."
            )
        if provider == "massive":
            if data.get("data_freshness") != "provider_plan_dependent":
                return None
            return f"{statement} Snapshot freshness is provider-plan-dependent."
        if provider == "ticker-layer":
            if data.get("data_provenance") != "derived_non_exchange_indicative":
                return None
            return f"{statement} This is derived, non-exchange, indicative data."
        return statement
    return f"{symbol} is quoted at {format(price, 'g')}{suffix}."


def equity_quote_agreement_status(
    data: dict[str, JsonValue],
) -> Literal["single_source", "agree", "disagree", "time_skewed"] | None:
    """Validate and return router agreement semantics, or ``None`` when malformed.

    Concrete single-provider adapter payloads have no ``selected_provider`` and are
    outside this router diagnostic contract; callers can distinguish that legacy
    shape by checking for the field before invoking this function.
    """

    success_count = data.get("provider_success_count")
    status = data.get("agreement_status")
    threshold = _finite_nonnegative_number(data.get("agreement_threshold_percent"))
    disagreement = _finite_nonnegative_number(data.get("price_disagreement_percent"))
    skew_threshold = _finite_nonnegative_number(data.get("corroboration_skew_threshold_seconds"))
    freshness_spread = _finite_nonnegative_number(data.get("freshness_spread_seconds"))
    temporally_aligned = data.get("temporally_aligned")
    corroborated = data.get("corroborated")
    if (
        not isinstance(success_count, int)
        or isinstance(success_count, bool)
        or not 1 <= success_count <= 2
        or threshold is None
        or skew_threshold is None
        or not isinstance(status, str)
        or not isinstance(temporally_aligned, bool)
        or not isinstance(corroborated, bool)
    ):
        return None
    if success_count == 1:
        if (
            status != "single_source"
            or temporally_aligned
            or corroborated
            or disagreement is not None
            or freshness_spread is not None
        ):
            return None
        return "single_source"
    if disagreement is None or freshness_spread is None:
        return None
    expected_alignment = freshness_spread <= skew_threshold
    if temporally_aligned is not expected_alignment:
        return None
    if disagreement > threshold:
        if status != "disagree" or corroborated:
            return None
        return "disagree"
    if not expected_alignment:
        if status != "time_skewed" or corroborated:
            return None
        return "time_skewed"
    if status != "agree" or not corroborated:
        return None
    return "agree"


def canonical_equity_quote_disagreement_statement(
    data: dict[str, JsonValue],
) -> str | None:
    """Rebuild the required caveat when redundant provider prices diverge."""

    if equity_quote_agreement_status(data) != "disagree":
        return None
    symbol = data.get("symbol")
    disagreement = _finite_nonnegative_number(data.get("price_disagreement_percent"))
    threshold = _finite_nonnegative_number(data.get("agreement_threshold_percent"))
    if (
        not isinstance(symbol, str)
        or _SYMBOL.fullmatch(symbol) is None
        or disagreement is None
        or threshold is None
    ):
        return None
    return (
        f"{symbol} provider quotes disagree by {format(disagreement, '.6g')}%, above "
        f"Leo's {format(threshold, 'g')}% agreement threshold."
    )


def canonical_equity_quote_time_skew_statement(
    data: dict[str, JsonValue],
) -> str | None:
    """Rebuild the caveat for quote rows observed too far apart to corroborate."""

    if data.get("provider_success_count") != 2 or data.get("temporally_aligned") is not False:
        return None
    symbol = data.get("symbol")
    freshness_spread = _finite_nonnegative_number(data.get("freshness_spread_seconds"))
    skew_threshold = _finite_nonnegative_number(data.get("corroboration_skew_threshold_seconds"))
    if (
        not isinstance(symbol, str)
        or _SYMBOL.fullmatch(symbol) is None
        or freshness_spread is None
        or skew_threshold is None
        or freshness_spread <= skew_threshold
    ):
        return None
    return (
        f"{symbol} provider quotes were observed {format(freshness_spread, '.6g')} seconds "
        f"apart, above Leo's {format(skew_threshold, 'g')}-second corroboration window."
    )


def canonical_equity_search_statements(
    data: dict[str, JsonValue],
) -> tuple[str, ...] | None:
    """Rebuild bounded provider-attributed symbol-search statements."""

    provider = data.get("provider")
    query = data.get("query")
    query_digest = data.get("query_hash")
    results = data.get("results")
    result_count = data.get("result_count")
    if (
        not isinstance(provider, str)
        or provider not in EQUITY_SEARCH_PROVIDERS
        or not isinstance(query, str)
        or not query.strip()
        or not isinstance(query_digest, str)
        or _SHA256.fullmatch(query_digest) is None
        or query_digest != equity_query_hash(query)
        or not isinstance(results, list)
        or not isinstance(result_count, int)
        or isinstance(result_count, bool)
        or result_count != len(results)
        or not 0 <= result_count <= 10
    ):
        return None
    label = _PROVIDER_LABELS[provider]
    statements: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in results:
        if not isinstance(item, dict):
            return None
        symbol = item.get("symbol")
        name = item.get("name")
        provider_symbol = item.get("provider_symbol")
        if (
            not isinstance(symbol, str)
            or _SYMBOL.fullmatch(symbol) is None
            or not isinstance(name, str)
            or not name.strip()
            or len(name) > 240
            or not isinstance(provider_symbol, str)
            or not _valid_provider_symbol(provider, provider_symbol)
        ):
            return None
        identity = (symbol, name)
        if identity in seen:
            return None
        seen.add(identity)
        statements.append(f"{label} symbol search matched {symbol} to {name}.")
    return tuple(statements)


def canonical_equity_profile_statements(
    data: dict[str, JsonValue],
) -> tuple[str, ...] | None:
    """Rebuild one exact provider-attributed, possibly partial profile statement."""

    provider = data.get("provider")
    symbol = data.get("symbol")
    provider_symbol = data.get("provider_symbol")
    name = data.get("name")
    exchange = data.get("exchange")
    industry = data.get("industry")
    if (
        not isinstance(provider, str)
        or provider not in EQUITY_PROFILE_PROVIDERS
        or not isinstance(symbol, str)
        or _SYMBOL.fullmatch(symbol) is None
        or not isinstance(provider_symbol, str)
        or not _valid_provider_symbol(provider, provider_symbol)
        or (name is not None and (not isinstance(name, str) or not name.strip() or len(name) > 240))
        or (
            exchange is not None
            and (not isinstance(exchange, str) or not exchange.strip() or len(exchange) > 120)
        )
        or (industry is not None and (not isinstance(industry, str) or len(industry) > 160))
        or not any(isinstance(value, str) and value.strip() for value in (name, exchange, industry))
        or not _valid_missing_fields(data, ("exchange", "industry", "name"))
    ):
        return None
    canonical = f"{_PROVIDER_LABELS[provider]} reports {symbol}"
    if isinstance(name, str) and name:
        canonical += f" as {name}"
    if isinstance(exchange, str) and exchange:
        if isinstance(name, str) and name:
            canonical += f", listed on {exchange}"
        else:
            canonical += f" listed on {exchange}"
    if isinstance(industry, str) and industry:
        if (isinstance(name, str) and name) or (isinstance(exchange, str) and exchange):
            canonical += f", in industry {industry}"
        else:
            canonical += f" in industry {industry}"
    return (canonical if canonical.endswith(".") else canonical + ".",)


def _valid_missing_fields(
    data: dict[str, JsonValue],
    fields: tuple[str, ...],
) -> bool:
    raw = data.get("missing_fields")
    if raw is None:
        return True
    expected = sorted(field for field in fields if data.get(field) is None)
    return bool(
        isinstance(raw, list)
        and len(raw) == len(set(item for item in raw if isinstance(item, str)))
        and all(isinstance(item, str) and item in fields for item in raw)
        and raw == expected
    )


def valid_equity_quote_provenance(
    *,
    provider: str,
    reference: str,
    symbol: str,
    observed_at: datetime | None = None,
) -> bool:
    """Validate provider-specific quote references without accepting request URLs."""

    if _SYMBOL.fullmatch(symbol) is None:
        return False
    if provider == "finnhub":
        prefix, separator, timestamp = reference.rpartition(":")
        return bool(
            prefix == f"quote:{symbol}"
            and separator
            and _integer(timestamp)
            and _reference_time_matches(timestamp, observed_at, units_per_second=1)
        )
    if provider == "alpha-vantage":
        prefix, separator, trading_day = reference.rpartition(":")
        return (
            prefix == f"global-quote:{symbol}"
            and bool(separator)
            and _ISO_DATE.fullmatch(trading_day) is not None
            and (observed_at is None or observed_at.date().isoformat() == trading_day)
        )
    if provider == "massive":
        prefix, separator, timestamp = reference.rpartition(":")
        return bool(
            prefix == f"snapshot:{symbol}"
            and separator
            and _integer(timestamp)
            and _reference_time_matches(timestamp, observed_at, units_per_second=1_000_000_000)
        )
    if provider == "ticker-layer":
        match = re.fullmatch(
            rf"stock-snapshot:(?P<provider_symbol>[A-Z]{{2}}:{re.escape(symbol)}):(?P<ts>[1-9]\d*)",
            reference,
        )
        return bool(
            match is not None
            and _reference_time_matches(match.group("ts"), observed_at, units_per_second=1_000)
        )
    return False


def valid_equity_search_provenance(
    *,
    provider: str,
    reference: str,
    query_hash: str,
) -> bool:
    if _SHA256.fullmatch(query_hash) is None:
        return False
    if provider == "alpha-vantage":
        return reference == f"symbol-search:{query_hash}"
    if provider == "massive":
        return reference == f"ticker-search:{query_hash}"
    if provider == "ticker-layer":
        return re.fullmatch(rf"stock-symbol-search:[A-Z]{{2}}:{query_hash}", reference) is not None
    return False


def valid_equity_profile_provenance(
    *,
    provider: str,
    reference: str,
    provider_symbol: str,
) -> bool:
    if not _valid_provider_symbol(provider, provider_symbol):
        return False
    if provider == "alpha-vantage":
        return reference == f"company-overview:{provider_symbol}"
    if provider == "finnhub":
        return reference == f"company-profile:{provider_symbol}"
    if provider == "massive":
        return reference == f"ticker-overview:{provider_symbol}"
    if provider == "ticker-layer":
        prefix = f"stock-fundamentals:{provider_symbol}:"
        return reference.startswith(prefix) and bool(reference.removeprefix(prefix))
    return False


def valid_equity_observed_at(data: dict[str, JsonValue], observed_at: datetime) -> bool:
    """Require normalized market timestamps to exactly match observation authority."""

    raw = data.get("as_of")
    if not isinstance(raw, str):
        return False
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed == observed_at


def valid_equity_quote_aggregate(
    data: dict[str, JsonValue],
    *,
    source_provider: str,
    source_reference: str,
    observed_at: datetime,
    expires_at: datetime | None,
) -> bool:
    """Validate deterministic provider routing, selection, and diagnostic accounting."""

    symbol = data.get("symbol")
    selected_provider = data.get("selected_provider")
    selected_reference = data.get("selected_reference")
    provider_order = data.get("provider_order")
    attempts = data.get("provider_attempts")
    quote_rows = data.get("provider_quotes")
    corroboration_target = data.get("corroboration_target")
    call_bound = data.get("provider_call_bound")
    if not (
        isinstance(symbol, str)
        and _SYMBOL.fullmatch(symbol) is not None
        and isinstance(selected_provider, str)
        and selected_provider in EQUITY_QUOTE_PROVIDERS
        and selected_provider == source_provider
        and data.get("provider") == selected_provider
        and isinstance(selected_reference, str)
        and selected_reference == source_reference
        and valid_equity_quote_provenance(
            provider=source_provider,
            reference=source_reference,
            symbol=symbol,
            observed_at=observed_at,
        )
        and valid_equity_observed_at(data, observed_at)
        and isinstance(provider_order, list)
        and 1 <= len(provider_order) <= 4
        and all(
            isinstance(provider, str) and provider in EQUITY_QUOTE_PROVIDERS
            for provider in provider_order
        )
        and len(set(provider_order)) == len(provider_order)
        and isinstance(attempts, list)
        and len(attempts) == len(provider_order)
        and isinstance(quote_rows, list)
        and isinstance(corroboration_target, int)
        and not isinstance(corroboration_target, bool)
        and 1 <= corroboration_target <= min(2, len(provider_order))
        and call_bound == len(provider_order)
        and data.get("selection_policy") == "freshest_then_provider_order"
        and equity_quote_agreement_status(data) is not None
    ):
        return False

    attempted_successes: list[tuple[int, str, str, int | float, datetime]] = []
    failure_count = 0
    skipped_count = 0
    health_skip_count = 0
    reached_target = False
    for index, (provider, attempt) in enumerate(zip(provider_order, attempts, strict=True)):
        if not isinstance(provider, str) or not isinstance(attempt, dict):
            return False
        if attempt.get("provider") != provider:
            return False
        attempt_status = attempt.get("status")
        if reached_target:
            if not _valid_attempt_skip(attempt, required_code="CORROBORATION_TARGET_REACHED"):
                return False
            skipped_count += 1
            continue
        if attempt_status == "success":
            reference = attempt.get("reference")
            price = _finite_positive_number(attempt.get("price"))
            as_of = _aware_datetime(attempt.get("as_of"))
            if (
                not isinstance(reference, str)
                or price is None
                or as_of is None
                or not valid_equity_quote_provenance(
                    provider=provider,
                    reference=reference,
                    symbol=symbol,
                    observed_at=as_of,
                )
            ):
                return False
            attempted_successes.append((index, provider, reference, price, as_of))
            reached_target = len(attempted_successes) == corroboration_target
        elif attempt_status == "failure":
            if not _valid_attempt_failure(attempt):
                return False
            failure_count += 1
        elif attempt_status == "skipped":
            if not _valid_attempt_skip(attempt):
                return False
            skipped_count += 1
            health_skip_count += 1
        else:
            return False

    if (
        not attempted_successes
        or len(attempted_successes) > 2
        or len(quote_rows) != len(attempted_successes)
    ):
        return False
    normalized_successes: list[tuple[int, str, str, int | float, datetime, datetime]] = []
    for row, (index, provider, reference, price, as_of) in zip(
        quote_rows, attempted_successes, strict=True
    ):
        if not isinstance(row, dict):
            return False
        raw_expiry = row.get("expires_at")
        row_expiry = _aware_datetime(raw_expiry)
        if row_expiry is None or row_expiry <= observed_at:
            return False
        if not (
            row.get("provider") == provider
            and row.get("reference") == reference
            and _same_number(row.get("price"), price)
            and _aware_datetime(row.get("as_of")) == as_of
        ):
            return False
        normalized_successes.append((index, provider, reference, price, as_of, row_expiry))

    successes = normalized_successes

    selected = max(successes, key=lambda item: (item[4], -item[0]))
    expected_expiry = min(item[5] for item in successes)
    selected_price = _finite_positive_number(data.get("price"))
    if not (
        selected[1] == selected_provider
        and selected[2] == selected_reference
        and selected[4] == observed_at
        and expires_at == expected_expiry
        and selected_price is not None
        and _same_number(selected_price, selected[3])
        and data.get("provider_attempt_count") == len(successes) + failure_count
        and data.get("provider_success_count") == len(successes)
        and data.get("provider_failure_count") == failure_count
        and data.get("provider_skipped_count") == skipped_count
        and data.get("provider_health_skip_count") == health_skip_count
        and data.get("fallback_used") is (failure_count > 0 or selected[0] > 0)
    ):
        return False

    canonical = canonical_equity_quote_statement(data)
    disagreement = canonical_equity_quote_disagreement_statement(data)
    time_skew = canonical_equity_quote_time_skew_statement(data)
    expected_statements = [
        item for item in (canonical, disagreement, time_skew) if item is not None
    ]
    statements = data.get("statements")
    if canonical is None or statements != expected_statements:
        return False

    if len(successes) == 1:
        return True
    prices = [float(item[3]) for item in successes]
    low = min(prices)
    expected_disagreement = (max(prices) - low) / low * 100
    observed_times = [item[4] for item in successes]
    expected_spread = (max(observed_times) - min(observed_times)).total_seconds()
    actual_disagreement = _finite_nonnegative_number(data.get("price_disagreement_percent"))
    actual_spread = _finite_nonnegative_number(data.get("freshness_spread_seconds"))
    return bool(
        actual_disagreement is not None
        and actual_spread is not None
        and math.isclose(actual_disagreement, expected_disagreement, rel_tol=1e-12, abs_tol=1e-12)
        and math.isclose(actual_spread, expected_spread, rel_tol=0, abs_tol=1e-9)
    )


def _valid_attempt_failure(value: dict[str, JsonValue]) -> bool:
    code = value.get("code")
    return bool(
        isinstance(code, str)
        and 1 <= len(code) <= 160
        and isinstance(value.get("retryable"), bool)
        and "reference" not in value
        and "price" not in value
    )


def _valid_attempt_skip(
    value: dict[str, JsonValue],
    *,
    required_code: str | None = None,
) -> bool:
    code = value.get("code")
    return bool(
        isinstance(code, str)
        and 1 <= len(code) <= 160
        and (required_code is None or code == required_code)
        and value.get("retryable") is False
        and "reference" not in value
        and "price" not in value
    )


def _aware_datetime(value: JsonValue | None) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _reference_time_matches(
    raw_timestamp: str,
    observed_at: datetime | None,
    *,
    units_per_second: int,
) -> bool:
    if observed_at is None:
        return True
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        return False
    try:
        referenced_at = datetime.fromtimestamp(
            int(raw_timestamp) / units_per_second,
            tz=UTC,
        )
    except (OverflowError, OSError, ValueError):
        return False
    return referenced_at == observed_at


def _same_number(left: JsonValue | None, right: int | float) -> bool:
    normalized = _finite_positive_number(left)
    return normalized is not None and float(normalized) == float(right)


def _valid_provider_symbol(provider: str, value: str) -> bool:
    if provider == "ticker-layer":
        return _QUALIFIED_SYMBOL.fullmatch(value) is not None
    return _SYMBOL.fullmatch(value) is not None


def _valid_currency(value: object) -> bool:
    return isinstance(value, str) and _CURRENCY.fullmatch(value) is not None


def _finite_positive_number(value: object) -> int | float | None:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    ):
        return value
    return None


def _finite_nonnegative_number(value: object) -> int | float | None:
    if (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value >= 0
    ):
        return value
    return None


def _integer(value: str) -> bool:
    return _INTEGER.fullmatch(value) is not None


__all__ = (
    "EQUITY_PROFILE_PROVIDERS",
    "EQUITY_QUOTE_PROVIDERS",
    "EQUITY_SEARCH_PROVIDERS",
    "canonical_equity_profile_statements",
    "canonical_equity_quote_disagreement_statement",
    "canonical_equity_quote_statement",
    "canonical_equity_quote_time_skew_statement",
    "canonical_equity_search_statements",
    "equity_query_hash",
    "equity_quote_agreement_status",
    "valid_equity_observed_at",
    "valid_equity_profile_provenance",
    "valid_equity_quote_aggregate",
    "valid_equity_quote_provenance",
    "valid_equity_search_provenance",
)
