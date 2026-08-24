"""Legacy Finnhub profile canonicalization used by the Finnhub adapter."""

from __future__ import annotations

import re

from pydantic import JsonValue

_TICKER = re.compile(r"[A-Z][A-Z0-9.-]{0,19}")


def canonical_finnhub_profile_statements(
    data: dict[str, JsonValue],
) -> tuple[str, ...] | None:
    """Canonicalize the legacy Finnhub-only profile shape with partial facts."""

    symbol = data.get("symbol")
    name = data.get("name")
    exchange = data.get("exchange")
    industry = data.get("industry")
    if not (
        isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and (name is None or (isinstance(name, str) and bool(name.strip())))
        and (exchange is None or (isinstance(exchange, str) and bool(exchange.strip())))
        and (industry is None or (isinstance(industry, str) and bool(industry.strip())))
        and any(isinstance(value, str) and value.strip() for value in (name, exchange, industry))
    ):
        return None
    missing = data.get("missing_fields")
    expected_missing = sorted(
        key
        for key, value in (("name", name), ("exchange", exchange), ("industry", industry))
        if value is None
    )
    if missing is not None and missing != expected_missing:
        return None
    statement = f"{symbol}"
    if isinstance(name, str):
        statement += f" is {name}"
    if isinstance(exchange, str):
        if isinstance(name, str):
            statement += f", listed on {exchange}"
        else:
            statement += f" is listed on {exchange}"
    if isinstance(industry, str):
        if len(statement) > len(symbol):
            statement += f", in Finnhub industry {industry}"
        else:
            statement += f" is in Finnhub industry {industry}"
    return (statement if statement.endswith(".") else statement + ".",)
