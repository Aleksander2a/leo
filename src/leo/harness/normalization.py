"""Bounded provider-result normalization at the harness evidence boundary."""

from __future__ import annotations

import hashlib
import json
import math
from typing import cast

from pydantic import JsonValue, ValidationError

from leo.harness.models import (
    EvidenceQuality,
    Observation,
    ObservationStatus,
    ScopeKey,
    ToolFailure,
    ToolSuccess,
)

NORMALIZATION_VERSION = "normalization-v1"


class NormalizationFailure(RuntimeError):
    """A provider result could not safely become durable evidence."""

    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def normalize_success(
    outcome: ToolSuccess | ToolFailure,
    *,
    observation_id: str,
    scope: ScopeKey,
    run_id: str,
    tool_call_id: str,
    observation_kind: str | None = None,
    quality: EvidenceQuality | None = None,
    max_bytes: int = 32_768,
) -> Observation:
    """Create evidence only from a successful, bounded, finite provider result."""

    if isinstance(outcome, ToolFailure):
        raise NormalizationFailure("tool_failure_is_not_evidence")
    if max_bytes < 1:
        raise NormalizationFailure("normalization_limit_invalid")
    try:
        _assert_finite(outcome.data)
    except NormalizationFailure:
        raise
    except (RecursionError, TypeError, ValueError) as exc:
        raise NormalizationFailure("observation_data_not_json") from exc
    try:
        canonical = json.dumps(
            outcome.data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        decoded = json.loads(canonical)
    except (RecursionError, TypeError, ValueError) as exc:
        raise NormalizationFailure("observation_data_not_json") from exc
    if not isinstance(decoded, dict):
        raise NormalizationFailure("observation_data_not_object")
    encoded = canonical.encode("utf-8")
    if len(encoded) > max_bytes:
        # Oversize evidence used to be discarded outright, so Leo would fetch a
        # long filing or article, throw the whole result away, and then report
        # that it had no reliable source. Several tools also declare result caps
        # above this ceiling (web.research_verified at 40KB, the subagent tools
        # far higher), which made those routes structurally unable to succeed.
        # Truncating preserves the leading, most relevant portion and marks the
        # elision so the model knows the payload is partial.
        decoded = _truncate_to_fit(decoded, max_bytes)
        canonical = json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        encoded = canonical.encode("utf-8")
        if len(encoded) > max_bytes:
            raise NormalizationFailure("observation_data_too_large")
    try:
        kind = observation_kind or outcome.source.reference.split(":", 1)[0]
        return Observation(
            id=observation_id,
            scope=scope,
            run_id=run_id,
            tool_call_id=tool_call_id,
            kind=kind,
            data=cast(dict[str, JsonValue], decoded),
            source=outcome.source,
            observed_at=outcome.observed_at,
            expires_at=outcome.expires_at,
            raw_hash=hashlib.sha256(encoded).hexdigest(),
            status=ObservationStatus.RETRIEVED,
            quality=quality or _quality_for_kind(kind),
            schema_version="observation-v2",
            normalization_version=NORMALIZATION_VERSION,
        )
    except (AttributeError, TypeError, ValueError, ValidationError) as exc:
        raise NormalizationFailure("observation_contract_invalid") from exc


_TRUNCATION_MARKER = "… [truncated by Leo's evidence boundary]"


def _encoded_size(value: object) -> int:
    return len(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    )


def _truncate_to_fit(data: dict[str, JsonValue], max_bytes: int) -> dict[str, JsonValue]:
    """Shrink a payload to the evidence ceiling, largest contributor first.

    Strings are cut (keeping their leading text, which is where the answer
    usually is) and lists are shortened from the tail. Applied repeatedly until
    the canonical encoding fits, so the model receives partial evidence rather
    than nothing at all.
    """

    working: dict[str, JsonValue] = dict(data)
    for _ in range(64):
        if _encoded_size(working) <= max_bytes:
            return working
        widest_key: str | None = None
        widest_size = 0
        for key, value in working.items():
            size = _encoded_size(value)
            if size > widest_size:
                widest_key, widest_size = key, size
        if widest_key is None:
            break
        value = working[widest_key]
        overshoot = _encoded_size(working) - max_bytes
        if isinstance(value, str) and len(value) > len(_TRUNCATION_MARKER):
            keep = max(0, len(value) - overshoot - len(_TRUNCATION_MARKER) - 8)
            if keep <= 0:
                working[widest_key] = _TRUNCATION_MARKER
            else:
                working[widest_key] = value[:keep] + _TRUNCATION_MARKER
        elif isinstance(value, list) and value:
            working[widest_key] = value[: max(1, len(value) // 2)] if len(value) > 1 else []
        elif isinstance(value, dict) and value:
            working[widest_key] = _truncate_to_fit(
                value, max(1, max_bytes - (_encoded_size(working) - widest_size))
            )
        else:
            working[widest_key] = _TRUNCATION_MARKER
    return working


def _quality_for_kind(kind: str) -> EvidenceQuality:
    if kind.startswith("sec."):
        return EvidenceQuality.PRIMARY_SOURCE
    if kind in {"agent.delegate_research", "agent.execute_research_plan"}:
        return EvidenceQuality.VERIFIED_CHILD
    if kind.startswith(("memory.", "thread_context.")):
        return EvidenceQuality.INTERNAL_CONTEXT
    if kind in {"web.search_public", "web.search_tavily"}:
        return EvidenceQuality.DISCOVERY_ONLY
    if kind in {"web.fetch_public_text", "web.search_exa", "web.research_verified"}:
        return EvidenceQuality.UNTRUSTED_RETRIEVAL
    return EvidenceQuality.PROVIDER_REPORTED


def _assert_finite(value: object, *, active_ids: frozenset[int] = frozenset()) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NormalizationFailure("observation_non_finite_number")
    if isinstance(value, dict):
        identity = id(value)
        if identity in active_ids:
            raise NormalizationFailure("observation_data_not_json")
        nested_ids = active_ids | {identity}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NormalizationFailure("observation_data_not_json")
            _assert_finite(nested, active_ids=nested_ids)
    elif isinstance(value, list):
        identity = id(value)
        if identity in active_ids:
            raise NormalizationFailure("observation_data_not_json")
        nested_ids = active_ids | {identity}
        for nested in value:
            _assert_finite(nested, active_ids=nested_ids)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise NormalizationFailure("observation_data_not_json")
