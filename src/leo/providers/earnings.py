"""Canonical, payload-derived summaries for bounded earnings observations."""

from __future__ import annotations

import math
import re
from datetime import date

_TICKER = re.compile(r"[A-Z][A-Z0-9.-]{0,7}")


def canonical_earnings_statements(
    symbol: object,
    items: object,
) -> tuple[str, ...] | None:
    """Return one bounded trend summary followed by exact per-period statements.

    The summary describes only the normalized periods present in ``items``.  It
    deliberately makes no claim about periods outside that bounded provider window.
    Returning all statements from one function lets the adapter, deterministic
    completion path, and verifier independently agree on the same canonical form.
    """

    if (
        not isinstance(symbol, str)
        or _TICKER.fullmatch(symbol) is None
        or not isinstance(items, list)
        or not 1 <= len(items) <= 4
    ):
        return None

    normalized: list[tuple[date, str, float, float]] = []
    periods: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or item.get("symbol") != symbol:
            return None
        period = item.get("period")
        actual = _finite_number(item.get("actual"))
        estimate = _finite_number(item.get("estimate"))
        if not isinstance(period, str):
            return None
        try:
            period_date = date.fromisoformat(period)
        except ValueError:
            period_date = None
        if period_date is None or actual is None or estimate is None or period in periods:
            return None
        periods.add(period)
        normalized.append((period_date, period, actual, estimate))

    normalized.sort(key=lambda item: item[0], reverse=True)
    beats = sum(actual > estimate for _date, _period, actual, estimate in normalized)
    misses = sum(actual < estimate for _date, _period, actual, estimate in normalized)
    matches = len(normalized) - beats - misses
    period_list = ", ".join(period for _date, period, _actual, _estimate in normalized)
    summary = (
        f"Across {len(normalized)} normalized Finnhub earnings observations for periods "
        f"{period_list}, {symbol} beat the EPS estimate in {beats}, missed it in {misses}, "
        f"and matched it in {matches}."
    )
    details = tuple(
        f"{symbol} reported actual EPS {format(actual, 'g')} versus estimate "
        f"{format(estimate, 'g')} for period {period}."
        for _date, period, actual, estimate in normalized
    )
    return (summary, *details)


def _finite_number(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
