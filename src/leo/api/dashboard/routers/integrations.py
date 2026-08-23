"""Per-integration call volume and success rate, derived from durable state.

Provider identity is read from ``Observation.source.provider`` (see
``leo.harness.models.SourceRef``) rather than a hardcoded tool registry, so newly added
providers show up automatically. Tool calls that fail before ever producing an observation
(``tool_failed`` events) have no provider attribution and are reported separately, keyed by
raw tool id. Every provider row is also classified via ``leo.api.dashboard.provenance`` so
the dashboard can group/badge by MCP vs REST API vs internal call, not just by provider name.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from leo.api.dashboard.deps import get_session
from leo.api.dashboard.provenance import classify_call
from leo.persistence.schema import ObservationRow, RunEventRow

router = APIRouter()


@router.get("/integrations")
async def get_integrations(session: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    observation_rows = (
        await session.execute(
            select(ObservationRow.kind, ObservationRow.source, ObservationRow.status)
        )
    ).all()

    per_provider: dict[str, Counter[str]] = defaultdict(Counter)
    provenance_by_provider: dict[str, dict[str, str]] = {}
    for kind, source, status in observation_rows:
        provider = (source or {}).get("provider") or "unknown"
        per_provider[provider][status] += 1
        provenance_by_provider[provider] = classify_call(
            tool_name=kind, provider=source.get("provider") if source else None
        )

    providers = []
    for provider, counts in sorted(per_provider.items()):
        total = sum(counts.values())
        retrieved = counts.get("retrieved", 0)
        provenance = provenance_by_provider.get(
            provider, {"call_kind": "unknown", "integration": provider}
        )
        providers.append(
            {
                "provider": provider,
                "display_name": provenance["integration"],
                "call_kind": provenance["call_kind"],
                "total": total,
                "retrieved": retrieved,
                "stale": counts.get("stale", 0),
                "rejected": counts.get("rejected", 0),
                "success_rate": (retrieved / total) if total else None,
            }
        )

    failed_payloads = (
        (
            await session.execute(
                select(RunEventRow.payload).where(RunEventRow.type == "tool_failed")
            )
        )
        .scalars()
        .all()
    )
    tool_failures = Counter(
        (payload or {}).get("tool") or (payload or {}).get("tool_id") or "unknown"
        for payload in failed_payloads
    )

    return {
        "providers": providers,
        "tool_failures": [
            {
                "key": key,
                "count": count,
                **classify_call(tool_name=key if isinstance(key, str) else None, provider=None),
            }
            for key, count in tool_failures.most_common(20)
        ],
    }
