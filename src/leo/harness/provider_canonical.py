"""Canonical completions from fresh, normalized provider observations.

This module is deliberately provider-neutral and harness-owned.  A successful HTTP
response is not enough: direct completion is available only after an allowlisted
adapter has emitted a bounded observation whose entity, timestamps, provenance, and
payload-derived canonical statements all agree.  Discovery-only search metadata and
arbitrary fetched text are intentionally excluded.
"""

from __future__ import annotations

import hashlib
import re
from datetime import date, datetime, timedelta

from pydantic import JsonValue

from leo.harness.crypto_market import canonical_crypto_evidence_statement
from leo.harness.earnings import canonical_earnings_statements
from leo.harness.equity_market import (
    EQUITY_PROFILE_PROVIDERS,
    EQUITY_SEARCH_PROVIDERS,
    canonical_equity_profile_statements,
    canonical_equity_quote_disagreement_statement,
    canonical_equity_quote_statement,
    canonical_equity_quote_time_skew_statement,
    canonical_equity_search_statements,
    valid_equity_observed_at,
    valid_equity_profile_provenance,
    valid_equity_quote_aggregate,
    valid_equity_quote_provenance,
    valid_equity_search_provenance,
)
from leo.harness.exa_search import canonical_exa_highlight_statements, exa_result_hash
from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EvidenceQuality,
    EvidenceToolRequirement,
    Observation,
    ObservationStatus,
    constrained_values_match,
)
from leo.harness.web_research import valid_verified_web_attempts
from leo.url_policy import is_public_https_url

_TICKER = re.compile(r"[A-Z][A-Z0-9.-]{0,19}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_FORM = re.compile(r"[A-Za-z0-9-]{1,24}")
_FILING_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_CIK = re.compile(r"\d{10}")
_PRIMARY_DOCUMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")

_DIRECT_QUALITIES = frozenset(
    {
        EvidenceQuality.PRIMARY_SOURCE,
        EvidenceQuality.PROVIDER_REPORTED,
        # Exa highlights remain attributed untrusted retrieval.  Their exact URL,
        # selected result, hash, and canonical excerpt are validated below.
        EvidenceQuality.UNTRUSTED_RETRIEVAL,
    }
)


def canonical_evidence_completion(
    observations: tuple[Observation, ...],
    requirements: tuple[EvidenceToolRequirement, ...],
    *,
    now: datetime,
    include_sec_document_url: bool = False,
) -> CompletionProposal | None:
    """Return a bounded direct answer once every required provider read is usable.

    This is a recovery path, not a general summarizer.  It selects one newest exact
    observation per constrained requirement and one canonical statement per
    observation.  Richer reasoning remains the model's job; this path prevents a
    model formatting/refusal error from turning already retrieved provider facts
    into a failed run.
    """

    if not requirements:
        return None
    selected: list[tuple[Observation, str]] = []
    for requirement in requirements:
        candidates = sorted(
            (
                observation
                for observation in observations
                if observation.kind == requirement.observation_kind
                and _eligible_direct_observation(observation, now=now)
                and constrained_values_match(
                    requirement.required_arguments,
                    observation.data,
                    exact=False,
                )
            ),
            key=lambda item: (item.observed_at, item.id),
            reverse=True,
        )
        chosen: tuple[Observation, str] | None = None
        for observation in candidates:
            statement = canonical_provider_statement(
                observation,
                include_sec_document_url=include_sec_document_url,
            )
            if statement is not None:
                chosen = (observation, statement)
                break
        if chosen is None:
            return None
        selected.append(chosen)

    return CompletionProposal(
        answer=" ".join(statement for _observation, statement in selected),
        claims=tuple(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(observation.id,),
            )
            for observation, statement in selected
        ),
    )


def _eligible_direct_observation(observation: Observation, *, now: datetime) -> bool:
    return bool(
        observation.schema_version == "observation-v2"
        and observation.status is ObservationStatus.RETRIEVED
        and observation.rejection_code is None
        and observation.quality in _DIRECT_QUALITIES
        and observation.observed_at <= now + timedelta(seconds=60)
        and observation.expires_at is not None
        and observation.expires_at > now
    )


def canonical_provider_statement(
    observation: Observation,
    *,
    include_sec_document_url: bool = False,
) -> str | None:
    """Return one canonical claim only after kind-specific provenance validation."""

    if observation.kind.startswith("market.get_crypto_snapshot"):
        return canonical_crypto_evidence_statement(observation)
    if observation.kind == "market.get_quote":
        return _canonical_equity_quote(observation)
    if observation.kind.startswith("market.get_quote_"):
        return _canonical_direct_equity_quote(observation)
    if observation.kind in {
        "market.search_equity_symbols",
        "market.search_symbols_alpha_vantage",
        "market.search_symbols_massive",
        "market.search_symbols_ticker_layer",
    }:
        return _canonical_equity_search(observation)
    if observation.kind in {
        "market.get_equity_profile",
        "market.get_company_profile_alpha_vantage",
        "market.get_company_profile_finnhub",
        "market.get_company_profile_massive",
        "market.get_company_profile_ticker_layer",
    }:
        return _canonical_equity_profile(observation)
    if observation.kind == "market.get_company_profile":
        return _canonical_legacy_finnhub_profile(observation)
    if observation.kind == "market.get_company_news":
        return _canonical_finnhub_news(observation)
    if observation.kind == "market.get_earnings_surprises":
        return _canonical_finnhub_earnings(observation)
    if observation.kind == "market.get_basic_financials":
        return _canonical_finnhub_basic_financials(observation)
    if observation.kind == "sec.get_recent_filings":
        return _canonical_sec_filing(
            observation,
            include_document_url=include_sec_document_url,
        )
    if observation.kind in {"web.search_exa", "web.research_verified"}:
        return _canonical_exa(observation)
    # Tavily search rows are discovery-only and fetched page text is untrusted raw
    # content.  Neither becomes a canonical provider claim here.
    return None


def _canonical_equity_quote(observation: Observation) -> str | None:
    if "selected_provider" in observation.data:
        if not valid_equity_quote_aggregate(
            observation.data,
            source_provider=observation.source.provider,
            source_reference=observation.source.reference,
            observed_at=observation.observed_at,
            expires_at=observation.expires_at,
        ):
            return None
        statements = (
            canonical_equity_quote_statement(observation.data),
            canonical_equity_quote_disagreement_statement(observation.data),
            canonical_equity_quote_time_skew_statement(observation.data),
        )
        canonical_statements = tuple(item for item in statements if item is not None)
        if not canonical_statements or observation.data.get("statements") != list(
            canonical_statements
        ):
            return None
        return " ".join(canonical_statements)

    # Compatibility for the official Finnhub quote tool used by the narrow CLI
    # path.  The provider-neutral Slack composition uses the stricter direct shape.
    if observation.source.provider != "finnhub":
        return None
    symbol = observation.data.get("symbol")
    canonical = canonical_equity_quote_statement(observation.data)
    if (
        not isinstance(symbol, str)
        or canonical is None
        or not valid_equity_quote_provenance(
            provider="finnhub",
            reference=observation.source.reference,
            symbol=symbol,
            observed_at=observation.observed_at,
        )
        or not valid_equity_observed_at(observation.data, observation.observed_at)
    ):
        return None
    return canonical


def _canonical_direct_equity_quote(observation: Observation) -> str | None:
    expected_provider = {
        "market.get_quote_alpha_vantage": "alpha-vantage",
        "market.get_quote_finnhub": "finnhub",
        "market.get_quote_massive": "massive",
        "market.get_quote_ticker_layer": "ticker-layer",
    }.get(observation.kind)
    provider = observation.data.get("provider")
    symbol = observation.data.get("symbol")
    canonical = canonical_equity_quote_statement(observation.data)
    if (
        expected_provider is None
        or provider != expected_provider
        or observation.source.provider != expected_provider
        or not isinstance(symbol, str)
        or canonical is None
        or observation.data.get("statements") != [canonical]
        or not valid_equity_quote_provenance(
            provider=expected_provider,
            reference=observation.source.reference,
            symbol=symbol,
            observed_at=observation.observed_at,
        )
        or not valid_equity_observed_at(observation.data, observation.observed_at)
    ):
        return None
    return canonical


def _canonical_equity_search(observation: Observation) -> str | None:
    provider = observation.data.get("provider")
    query_hash = observation.data.get("query_hash")
    statements = canonical_equity_search_statements(observation.data)
    if (
        not isinstance(provider, str)
        or provider not in EQUITY_SEARCH_PROVIDERS
        or provider != observation.source.provider
        or not isinstance(query_hash, str)
        or not valid_equity_search_provenance(
            provider=provider,
            reference=observation.source.reference,
            query_hash=query_hash,
        )
        or not statements
        or observation.data.get("statements") != list(statements)
        or (
            "selected_provider" in observation.data
            and (
                observation.data.get("selected_provider") != provider
                or observation.data.get("selected_reference") != observation.source.reference
            )
        )
    ):
        return None
    return statements[0]


def _canonical_equity_profile(observation: Observation) -> str | None:
    provider = observation.data.get("provider")
    provider_symbol = observation.data.get("provider_symbol")
    statements = canonical_equity_profile_statements(observation.data)
    if (
        not isinstance(provider, str)
        or provider not in EQUITY_PROFILE_PROVIDERS
        or provider != observation.source.provider
        or not isinstance(provider_symbol, str)
        or not valid_equity_profile_provenance(
            provider=provider,
            reference=observation.source.reference,
            provider_symbol=provider_symbol,
        )
        or not valid_equity_observed_at(observation.data, observation.observed_at)
        or not statements
        or observation.data.get("statements") != list(statements)
        or (
            "selected_provider" in observation.data
            and (
                observation.data.get("selected_provider") != provider
                or observation.data.get("selected_reference") != observation.source.reference
            )
        )
    ):
        return None
    return statements[0]


def _canonical_legacy_finnhub_profile(observation: Observation) -> str | None:
    data = observation.data
    symbol = data.get("symbol")
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"company-profile:{symbol}"
    ):
        return None
    statements = canonical_finnhub_profile_statements(data)
    return (
        statements[0]
        if statements is not None and data.get("statements") == list(statements)
        else None
    )


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


def _canonical_finnhub_news(observation: Observation) -> str | None:
    data = observation.data
    symbol = data.get("symbol")
    items = data.get("items")
    item_count = data.get("item_count")
    from_value = data.get("from_date")
    to_value = data.get("to_date")
    try:
        from_date = date.fromisoformat(from_value) if isinstance(from_value, str) else None
        to_date = date.fromisoformat(to_value) if isinstance(to_value, str) else None
    except ValueError:
        return None
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and from_date is not None
        and to_date is not None
        and from_date <= to_date
        and observation.source.reference == f"company-news:{symbol}:{from_value}:{to_value}"
        and isinstance(items, list)
        and 1 <= len(items) <= 10
        and isinstance(item_count, int)
        and not isinstance(item_count, bool)
        and item_count == len(items)
    ):
        return None
    canonical: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            return None
        published_value = item.get("published_at")
        provider = item.get("source")
        headline = item.get("headline")
        url = item.get("url")
        try:
            published = (
                datetime.fromisoformat(published_value)
                if isinstance(published_value, str)
                else None
            )
        except ValueError:
            return None
        if not (
            published is not None
            and published.tzinfo is not None
            and from_date <= published.date() <= to_date
            and published <= observation.observed_at + timedelta(seconds=60)
            and isinstance(provider, str)
            and bool(provider.strip())
            and isinstance(headline, str)
            and bool(headline.strip())
            and isinstance(url, str)
            and is_public_https_url(url)
        ):
            return None
        canonical.append(
            f"On {published_value}, {provider} reported for {symbol}: {headline} Source URL: {url}"
        )
    return canonical[0] if data.get("statements") == canonical else None


def _canonical_finnhub_earnings(observation: Observation) -> str | None:
    data = observation.data
    symbol = data.get("symbol")
    items = data.get("items")
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"earnings-surprises:{symbol}"
        and isinstance(items, list)
        and data.get("item_count") == len(items)
    ):
        return None
    canonical = canonical_earnings_statements(symbol, items)
    if canonical is None or not canonical or data.get("statements") != list(canonical):
        return None
    return canonical[0]


def _canonical_finnhub_basic_financials(observation: Observation) -> str | None:
    data = observation.data
    symbol = data.get("symbol")
    metrics = data.get("metrics")
    labels = {
        "beta": "beta",
        "52WeekHigh": "52-week high",
        "52WeekLow": "52-week low",
        "10DayAverageTradingVolume": "10-day average trading volume",
        "marketCapitalization": "market capitalization",
        "peBasicExclExtraTTM": "basic P/E excluding extraordinary items (TTM)",
    }
    if not (
        observation.source.provider == "finnhub"
        and isinstance(symbol, str)
        and _TICKER.fullmatch(symbol) is not None
        and observation.source.reference == f"basic-financials:{symbol}"
        and isinstance(metrics, dict)
        and 1 <= len(metrics) <= len(labels)
        and data.get("metric_count") == len(metrics)
        and set(metrics).issubset(labels)
    ):
        return None
    canonical: list[str] = []
    for key, value in metrics.items():
        if not (
            isinstance(key, str) and isinstance(value, int | float) and not isinstance(value, bool)
        ):
            return None
        canonical.append(f"{symbol} has Finnhub {labels[key]} {format(value, 'g')}.")
    return canonical[0] if data.get("statements") == canonical else None


def _canonical_sec_filing(
    observation: Observation,
    *,
    include_document_url: bool,
) -> str | None:
    data = observation.data
    ticker = data.get("ticker")
    cik = data.get("cik")
    filings = data.get("filings")
    if not (
        observation.source.provider == "sec-edgar"
        and isinstance(ticker, str)
        and _TICKER.fullmatch(ticker) is not None
        and isinstance(cik, str)
        and _CIK.fullmatch(cik) is not None
        and observation.source.reference == f"submissions:{cik}"
        and isinstance(filings, list)
        and filings
        and isinstance(filings[0], dict)
    ):
        return None
    filing = filings[0]
    form = filing.get("form")
    filing_date = filing.get("filing_date")
    accession = filing.get("accession")
    primary_document = filing.get("primary_document")
    if not (
        isinstance(form, str)
        and _SAFE_FORM.fullmatch(form) is not None
        and isinstance(filing_date, str)
        and _FILING_DATE.fullmatch(filing_date) is not None
        and isinstance(accession, str)
        and _ACCESSION.fullmatch(accession) is not None
        and isinstance(primary_document, str)
        and _PRIMARY_DOCUMENT.fullmatch(primary_document) is not None
    ):
        return None
    statement = f"{ticker} filed form {form} on {filing_date} under accession {accession}."
    if not include_document_url:
        return statement
    expected_url = (
        "https://www.sec.gov/Archives/edgar/data/"
        f"{int(cik)}/{accession.replace('-', '')}/{primary_document}"
    )
    if filing.get("filing_url") != expected_url:
        return None
    return f"{statement} Document URL: {expected_url}"


def _canonical_exa(observation: Observation) -> str | None:
    data = observation.data
    if observation.kind == "web.research_verified":
        if data.get("selected_provider") != "exa" or not valid_verified_web_attempts(data):
            return None
    elif data.get("selected_provider") not in {None, "exa"}:
        return None
    query = data.get("query")
    query_hash = data.get("query_hash")
    result_hash = data.get("result_hash")
    result = data.get("result")
    provider_result_count = data.get("provider_result_count")
    selected_result_rank = data.get("selected_result_rank")
    statements = canonical_exa_highlight_statements(data)
    calculated_hash = exa_result_hash(data)
    result_url = result.get("url") if isinstance(result, dict) else None
    if not (
        observation.source.provider == "exa"
        and isinstance(query, str)
        and bool(query.strip())
        and isinstance(query_hash, str)
        and _SHA256.fullmatch(query_hash) is not None
        and hashlib.sha256(query.encode("utf-8")).hexdigest() == query_hash
        and isinstance(result_hash, str)
        and _SHA256.fullmatch(result_hash) is not None
        and calculated_hash == result_hash
        and observation.source.reference == f"search:{query_hash}:{result_hash}"
        and isinstance(result_url, str)
        and is_public_https_url(result_url)
        and observation.source.url == result_url
        and isinstance(provider_result_count, int)
        and not isinstance(provider_result_count, bool)
        and 1 <= provider_result_count <= 100
        and isinstance(selected_result_rank, int)
        and not isinstance(selected_result_rank, bool)
        and 1 <= selected_result_rank <= provider_result_count
        and statements
        and data.get("highlight_count") == len(statements)
        and data.get("search_type") == "auto"
        and data.get("contents_mode") == "highlights"
        and data.get("statements") == list(statements)
    ):
        return None
    return statements[0]


def canonical_claims_present(
    proposal: CompletionProposal,
    canonical: CompletionProposal,
) -> bool:
    """Whether a model proposal already carries the exact canonical source claims."""

    expected = {
        (_normalize(claim.statement), claim.observation_ids)
        for claim in canonical.claims
        if claim.kind is ClaimKind.SOURCE_CLAIM
    }
    actual = {
        (_normalize(claim.statement), claim.observation_ids)
        for claim in proposal.claims
        if claim.kind is ClaimKind.SOURCE_CLAIM
    }
    if actual != expected:
        return False
    normalized_answer = _normalize(proposal.answer)
    return all(statement in normalized_answer for statement, _ids in expected)


def _normalize(value: str) -> str:
    return " ".join(value.split()).casefold()


__all__ = [
    "canonical_claims_present",
    "canonical_evidence_completion",
    "canonical_finnhub_profile_statements",
    "canonical_provider_statement",
]
