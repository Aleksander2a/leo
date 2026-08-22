"""Provider-neutral cryptocurrency snapshot and corroboration contracts."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Literal

from pydantic import ConfigDict, Field, field_validator, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, Observation
from leo.harness.provider_health import ProviderHealthSnapshot

CryptoCurrency = Literal["USD", "EUR", "GBP", "JPY"]
CryptoAgreementStatus = Literal[
    "single_provider",
    "agreement",
    "divergence",
    "time_skewed",
]

_PROVIDER_LABELS = {
    "coingecko": "CoinGecko",
    "coinmarketcap": "CoinMarketCap",
}


class CryptoSnapshotArguments(ContractModel):
    """A common provider-independent lookup keyed by canonical asset slug."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    asset_id: str = Field(
        min_length=1,
        max_length=80,
        pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        description=(
            "Canonical provider-common cryptocurrency slug, such as bitcoin, ethereum, or solana."
        ),
    )
    quote_currency: CryptoCurrency = "USD"

    @field_validator("asset_id", mode="before")
    @classmethod
    def normalize_asset_id(cls, value: object) -> object:
        return value.strip().casefold() if isinstance(value, str) else value

    @field_validator("quote_currency", mode="before")
    @classmethod
    def normalize_currency(cls, value: object) -> object:
        return value.strip().upper() if isinstance(value, str) else value


class CryptoProviderSnapshot(ContractModel):
    """The exact shared schema emitted by every cryptocurrency price provider."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provider: Literal["coingecko", "coinmarketcap"]
    asset_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    provider_asset_id: NonEmptyStr
    name: NonEmptyStr | None = Field(default=None, max_length=120)
    symbol: str | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        pattern=r"^[A-Z0-9$@.-]+$",
    )
    quote_currency: CryptoCurrency
    price: float = Field(gt=0)
    market_cap: float | None = Field(default=None, ge=0)
    volume_24h: float | None = Field(default=None, ge=0)
    percent_change_24h: float | None = None
    as_of: datetime
    evidence_expires_at: datetime
    provider_reference: NonEmptyStr = Field(max_length=256)
    provider_request_id: str | None = Field(default=None, max_length=128)
    provider_payload_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provider_credits_used: int = Field(default=0, ge=0)
    missing_fields: tuple[str, ...] = Field(default=(), max_length=8)
    health: ProviderHealthSnapshot

    @field_validator("as_of", "evidence_expires_at")
    @classmethod
    def require_aware_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("crypto snapshot timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def provider_health_matches(self) -> CryptoProviderSnapshot:
        if self.health.provider != self.provider:
            raise ValueError("crypto provider health must match the snapshot provider")
        if self.evidence_expires_at <= self.as_of:
            raise ValueError("crypto provider evidence must expire after it was observed")
        allowed_missing = {
            "market_cap",
            "name",
            "percent_change_24h",
            "provider_asset_id",
            "provider_credits_used",
            "provider_payload_sha256",
            "symbol",
            "volume_24h",
        }
        if (
            len(self.missing_fields) != len(set(self.missing_fields))
            or tuple(sorted(self.missing_fields)) != self.missing_fields
            or not set(self.missing_fields).issubset(allowed_missing)
        ):
            raise ValueError("crypto provider missing-field accounting is malformed")
        expected_optional_missing = {
            field
            for field in (
                "market_cap",
                "name",
                "percent_change_24h",
                "provider_payload_sha256",
                "symbol",
                "volume_24h",
            )
            if getattr(self, field) is None
        }
        # Compatibility rows predating explicit partial-payload accounting may omit
        # the ledger.  Any nonempty ledger emitted by a current adapter must be exact.
        if self.missing_fields and not expected_optional_missing.issubset(self.missing_fields):
            raise ValueError("crypto provider missing-field ledger is incomplete")
        return self


class CryptoAgreement(ContractModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    status: CryptoAgreementStatus
    providers_compared: int = Field(ge=1, le=2)
    agreement_threshold_bps: float = Field(ge=0, le=10_000)
    corroboration_skew_threshold_seconds: float = Field(ge=0, le=86_400)
    spread_bps: float | None = Field(default=None, ge=0)
    freshness_spread_seconds: float | None = Field(default=None, ge=0)
    temporally_aligned: bool
    corroborated: bool
    lowest_price: float = Field(gt=0)
    highest_price: float = Field(gt=0)

    @model_validator(mode="after")
    def status_matches_cardinality_and_spread(self) -> CryptoAgreement:
        if self.lowest_price > self.highest_price:
            raise ValueError("crypto agreement price range is inverted")
        if self.providers_compared == 1:
            if (
                self.status != "single_provider"
                or self.spread_bps is not None
                or self.freshness_spread_seconds is not None
                or self.temporally_aligned
                or self.corroborated
            ):
                raise ValueError("one provider must have single-provider agreement state")
            if self.lowest_price != self.highest_price:
                raise ValueError("one provider must have a degenerate price range")
        else:
            if self.spread_bps is None or self.freshness_spread_seconds is None:
                raise ValueError("two providers require measured price and freshness spreads")
            expected_alignment = (
                self.freshness_spread_seconds <= self.corroboration_skew_threshold_seconds
            )
            if self.temporally_aligned is not expected_alignment:
                raise ValueError("crypto temporal-alignment state is inconsistent")
            if not expected_alignment:
                expected_status = "time_skewed"
                expected_corroborated = False
            elif self.spread_bps <= self.agreement_threshold_bps:
                expected_status = "agreement"
                expected_corroborated = True
            else:
                expected_status = "divergence"
                expected_corroborated = False
            if self.status != expected_status or self.corroborated is not expected_corroborated:
                raise ValueError("crypto agreement state does not match its threshold")
        return self


class CryptoProviderPayload(ContractModel):
    """Exact evidence payload for one provider-specific snapshot tool."""

    snapshot: CryptoProviderSnapshot
    statements: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=1)

    @model_validator(mode="after")
    def statement_is_canonical(self) -> CryptoProviderPayload:
        if self.statements != (canonical_crypto_snapshot_statement(self.snapshot),):
            raise ValueError("crypto provider statement is not canonical")
        return self


class CryptoAggregatePayload(ContractModel):
    """One or two fresh snapshots plus failure and provenance accounting."""

    asset_id: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    quote_currency: CryptoCurrency
    snapshots: tuple[CryptoProviderSnapshot, ...] = Field(min_length=1, max_length=2)
    providers_succeeded: tuple[Literal["coingecko", "coinmarketcap"], ...]
    provider_failures: dict[str, str]
    agreement: CryptoAgreement
    provenance_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    statements: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=3)
    summary: NonEmptyStr

    @model_validator(mode="after")
    def aggregate_is_self_consistent(self) -> CryptoAggregatePayload:
        providers = tuple(item.provider for item in self.snapshots)
        if providers != self.providers_succeeded or len(providers) != len(set(providers)):
            raise ValueError("crypto aggregate providers must be unique and ordered")
        if providers != tuple(sorted(providers, key=lambda item: _PROVIDER_ORDER[item])):
            raise ValueError("crypto aggregate provider ordering is not canonical")
        if any(
            item.asset_id != self.asset_id or item.quote_currency != self.quote_currency
            for item in self.snapshots
        ):
            raise ValueError("crypto aggregate snapshots must cover the same asset and currency")
        if self.agreement.providers_compared != len(self.snapshots):
            raise ValueError("crypto aggregate agreement cardinality is inconsistent")
        if set(self.provider_failures).intersection(providers):
            raise ValueError("a crypto provider cannot both succeed and fail")
        if any(
            provider not in _PROVIDER_LABELS or not _safe_failure_code(code)
            for provider, code in self.provider_failures.items()
        ):
            raise ValueError("crypto aggregate failure accounting is malformed")
        expected_agreement = calculate_crypto_agreement(
            self.snapshots,
            agreement_threshold_bps=self.agreement.agreement_threshold_bps,
            max_corroboration_skew_seconds=(self.agreement.corroboration_skew_threshold_seconds),
        )
        if self.agreement != expected_agreement:
            raise ValueError("crypto aggregate spread calculation is inconsistent")
        expected_statements = [
            canonical_crypto_snapshot_statement(snapshot) for snapshot in self.snapshots
        ]
        agreement_statement = canonical_crypto_agreement_statement(
            asset_id=self.asset_id,
            quote_currency=self.quote_currency,
            agreement=self.agreement,
        )
        if agreement_statement is not None:
            expected_statements.append(agreement_statement)
        if self.statements != tuple(expected_statements):
            raise ValueError("crypto aggregate statements are not canonical")
        if self.summary != canonical_crypto_aggregate_summary(
            snapshots=self.snapshots,
            agreement_statement=agreement_statement,
        ):
            raise ValueError("crypto aggregate summary is not canonical")
        expected_digest = crypto_provenance_digest(
            asset_id=self.asset_id,
            quote_currency=self.quote_currency,
            snapshots=self.snapshots,
            provider_failures=self.provider_failures,
            agreement=self.agreement,
        )
        if self.provenance_digest != expected_digest:
            raise ValueError("crypto aggregate provenance digest is inconsistent")
        return self


def canonical_crypto_snapshot_statement(snapshot: CryptoProviderSnapshot) -> str:
    label = _PROVIDER_LABELS[snapshot.provider]
    asset_label = snapshot.name or snapshot.asset_id
    symbol = f" ({snapshot.symbol})" if snapshot.symbol is not None else ""
    return (
        f"{label} reports {asset_label}{symbol} at "
        f"{format(snapshot.price, 'g')} {snapshot.quote_currency} as of "
        f"{snapshot.as_of.isoformat()}."
    )


def canonical_crypto_agreement_statement(
    *,
    asset_id: str,
    quote_currency: CryptoCurrency,
    agreement: CryptoAgreement,
) -> str | None:
    if agreement.status == "single_provider" or agreement.spread_bps is None:
        return None
    if agreement.status == "time_skewed":
        if agreement.freshness_spread_seconds is None:
            return None
        return (
            f"CoinGecko and CoinMarketCap snapshots for {asset_id} in {quote_currency} "
            f"were observed {format(agreement.freshness_spread_seconds, 'g')} seconds apart, "
            f"above Leo's {format(agreement.corroboration_skew_threshold_seconds, 'g')}-second "
            "corroboration window; their prices are not treated as corroborating each other."
        )
    label = "within" if agreement.status == "agreement" else "above"
    return (
        f"CoinGecko and CoinMarketCap prices for {asset_id} in {quote_currency} have a "
        f"{format(agreement.spread_bps, 'g')} basis-point spread, {label} the "
        f"{format(agreement.agreement_threshold_bps, 'g')} basis-point agreement threshold."
    )


def calculate_crypto_agreement(
    snapshots: tuple[CryptoProviderSnapshot, ...],
    *,
    agreement_threshold_bps: float,
    max_corroboration_skew_seconds: float,
) -> CryptoAgreement:
    if not 1 <= len(snapshots) <= 2:
        raise ValueError("crypto agreement requires one or two snapshots")
    if not math.isfinite(agreement_threshold_bps) or not 0 <= agreement_threshold_bps <= 10_000:
        raise ValueError("crypto agreement threshold is invalid")
    if (
        not math.isfinite(max_corroboration_skew_seconds)
        or not 0 <= max_corroboration_skew_seconds <= 86_400
    ):
        raise ValueError("crypto corroboration skew threshold is invalid")
    prices = tuple(item.price for item in snapshots)
    low = min(prices)
    high = max(prices)
    if len(prices) == 1:
        return CryptoAgreement(
            status="single_provider",
            providers_compared=1,
            agreement_threshold_bps=agreement_threshold_bps,
            corroboration_skew_threshold_seconds=max_corroboration_skew_seconds,
            temporally_aligned=False,
            corroborated=False,
            lowest_price=low,
            highest_price=high,
        )
    midpoint = (low + high) / 2
    spread_bps = abs(high - low) / midpoint * 10_000
    newest_snapshot_at = max(item.as_of for item in snapshots)
    oldest_snapshot_at = min(item.as_of for item in snapshots)
    freshness_spread_seconds = abs((newest_snapshot_at - oldest_snapshot_at).total_seconds())
    temporally_aligned = freshness_spread_seconds <= max_corroboration_skew_seconds
    return CryptoAgreement(
        status=(
            "time_skewed"
            if not temporally_aligned
            else "agreement"
            if spread_bps <= agreement_threshold_bps
            else "divergence"
        ),
        providers_compared=2,
        agreement_threshold_bps=agreement_threshold_bps,
        corroboration_skew_threshold_seconds=max_corroboration_skew_seconds,
        spread_bps=spread_bps,
        freshness_spread_seconds=freshness_spread_seconds,
        temporally_aligned=temporally_aligned,
        corroborated=temporally_aligned and spread_bps <= agreement_threshold_bps,
        lowest_price=low,
        highest_price=high,
    )


def canonical_crypto_aggregate_summary(
    *,
    snapshots: tuple[CryptoProviderSnapshot, ...],
    agreement_statement: str | None,
) -> str:
    statements = [canonical_crypto_snapshot_statement(snapshot) for snapshot in snapshots]
    if agreement_statement is not None:
        statements.append(agreement_statement)
    return " ".join(statements)


def crypto_provenance_digest(
    *,
    asset_id: str,
    quote_currency: CryptoCurrency,
    snapshots: tuple[CryptoProviderSnapshot, ...],
    provider_failures: dict[str, str],
    agreement: CryptoAgreement,
) -> str:
    """Bind aggregate provenance without including mutable quota-health counters."""

    payload = {
        "asset_id": asset_id,
        "quote_currency": quote_currency,
        "snapshots": [
            {
                "provider": item.provider,
                "name": item.name,
                "symbol": item.symbol,
                "provider_asset_id": item.provider_asset_id,
                "provider_reference": item.provider_reference,
                "as_of": item.as_of.isoformat(),
                "evidence_expires_at": item.evidence_expires_at.isoformat(),
                "price": item.price,
                "market_cap": item.market_cap,
                "volume_24h": item.volume_24h,
                "percent_change_24h": item.percent_change_24h,
                "provider_payload_sha256": item.provider_payload_sha256,
                "missing_fields": item.missing_fields,
            }
            for item in snapshots
        ],
        "provider_failures": dict(sorted(provider_failures.items())),
        "agreement": agreement.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_failure_code(value: str) -> bool:
    return bool(value) and len(value) <= 96 and value.replace("_", "").isalnum()


def ground_crypto_provider_snapshot(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    expected_provider = {
        "market.get_crypto_snapshot_coingecko": "coingecko",
        "market.get_crypto_snapshot_coinmarketcap": "coinmarketcap",
    }.get(observation.kind)
    if expected_provider is None:
        return False, "Cryptocurrency provider observation kind is unsupported."
    try:
        payload = CryptoProviderPayload.model_validate(observation.data)
    except ValueError:
        return False, "Cryptocurrency provider observation is malformed."
    snapshot = payload.snapshot
    if not (
        snapshot.provider == expected_provider
        and observation.source.provider == expected_provider
        and observation.source.reference == snapshot.provider_reference
        and observation.observed_at == snapshot.as_of
        and observation.expires_at == snapshot.evidence_expires_at
    ):
        return False, "Cryptocurrency provider provenance or timestamp is inconsistent."
    canonical = payload.statements[0]
    if _same_statement(statement, canonical) and _contains_statement(answer, canonical):
        return True, "Claim exactly matches a normalized provider cryptocurrency snapshot."
    return False, "Claim must exactly copy the canonical cryptocurrency provider statement."


def ground_crypto_aggregate_snapshot(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    try:
        payload = CryptoAggregatePayload.model_validate(observation.data)
    except ValueError:
        return False, "Cryptocurrency corroboration observation is malformed."
    if not (
        observation.source.provider == "crypto-corroboration"
        and observation.source.reference
        == (f"snapshot:{payload.asset_id}:{payload.quote_currency}:{payload.provenance_digest}")
        and observation.observed_at == max(item.as_of for item in payload.snapshots)
        and observation.expires_at == min(item.evidence_expires_at for item in payload.snapshots)
    ):
        return False, "Cryptocurrency corroboration provenance or timestamp is inconsistent."
    if _same_statement(statement, payload.summary) and _contains_statement(answer, payload.summary):
        return True, "Claim exactly matches the complete cryptocurrency corroboration summary."
    if any(
        _same_statement(statement, canonical) and _contains_statement(answer, canonical)
        for canonical in payload.statements
    ):
        return True, "Claim exactly matches normalized cryptocurrency corroboration evidence."
    return False, "Claim must exactly copy a canonical cryptocurrency corroboration statement."


def canonical_crypto_evidence_statement(observation: Observation) -> str | None:
    """Return a canonical provider/aggregate statement only after provenance validation."""

    if observation.kind in {
        "market.get_crypto_snapshot_coingecko",
        "market.get_crypto_snapshot_coinmarketcap",
    }:
        expected_provider = (
            "coingecko"
            if observation.kind == "market.get_crypto_snapshot_coingecko"
            else "coinmarketcap"
        )
        try:
            provider_payload = CryptoProviderPayload.model_validate(observation.data)
        except ValueError:
            return None
        snapshot = provider_payload.snapshot
        if not (
            snapshot.provider == expected_provider
            and observation.source.provider == expected_provider
            and observation.source.reference == snapshot.provider_reference
            and observation.observed_at == snapshot.as_of
            and observation.expires_at == snapshot.evidence_expires_at
        ):
            return None
        return provider_payload.statements[0]
    if observation.kind == "market.get_crypto_snapshot":
        try:
            aggregate_payload = CryptoAggregatePayload.model_validate(observation.data)
        except ValueError:
            return None
        if not (
            observation.source.provider == "crypto-corroboration"
            and observation.source.reference
            == (
                f"snapshot:{aggregate_payload.asset_id}:{aggregate_payload.quote_currency}:"
                f"{aggregate_payload.provenance_digest}"
            )
            and observation.observed_at == max(item.as_of for item in aggregate_payload.snapshots)
            and observation.expires_at
            == min(item.evidence_expires_at for item in aggregate_payload.snapshots)
        ):
            return None
        return aggregate_payload.summary
    return None


def _contains_statement(text: str, statement: str) -> bool:
    normalized_text = " ".join(text.split()).casefold()
    normalized_statement = " ".join(statement.split()).casefold()
    return bool(normalized_statement) and normalized_statement in normalized_text


def _same_statement(actual: str, expected: str) -> bool:
    return " ".join(actual.split()).casefold() == " ".join(expected.split()).casefold()


_PROVIDER_ORDER = {"coingecko": 0, "coinmarketcap": 1}
