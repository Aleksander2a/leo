"""Deterministic adaptive routing over the already policy-eligible catalog."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import Field

from leo.capabilities.catalog import InMemoryToolCatalog
from leo.capabilities.discovery import DiscoveryBroker, DiscoveryQuery
from leo.capabilities.embeddings import EmbeddingVector
from leo.harness.models import ContractModel, NonEmptyStr, RunPhase, ScopeKey


class RequestComplexity(StrEnum):
    SIMPLE = "simple"
    MULTI_DOMAIN = "multi_domain"
    AMBIGUOUS = "ambiguous"


class RouteDecision(ContractModel):
    complexity: RequestComplexity
    mode: str = Field(pattern=r"^(direct|discovery|safe_stop)$")
    candidates: tuple[NonEmptyStr, ...] = ()
    selected: tuple[NonEmptyStr, ...] = ()
    reason: NonEmptyStr
    catalog_version: NonEmptyStr


class AdaptiveRouter:
    def __init__(self, catalog: InMemoryToolCatalog) -> None:
        self._catalog = catalog
        self._discovery = DiscoveryBroker(catalog)

    def route(
        self,
        objective: str,
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        shortlist_limit: int = 8,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
        tool_embeddings: dict[str, EmbeddingVector] | None = None,
        query_embedding: EmbeddingVector | None = None,
    ) -> RouteDecision:
        eligible = self._catalog.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        if not eligible:
            return RouteDecision(
                complexity=RequestComplexity.AMBIGUOUS,
                mode="direct",
                reason="no eligible capability; direct conversation remains available",
                catalog_version=self._catalog.version,
            )
        tokens = set(re.findall(r"[a-z0-9_.-]{2,64}", objective.lower()))
        direct = tuple(record.id for record in eligible if record.id.lower() in tokens)
        if len(direct) == 1:
            return RouteDecision(
                complexity=RequestComplexity.SIMPLE,
                mode="direct",
                candidates=direct,
                selected=direct,
                reason="exact eligible capability name matched",
                catalog_version=self._catalog.version,
            )
        summaries = self._discovery.search(
            DiscoveryQuery(query=objective, limit=shortlist_limit),
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
            tool_embeddings=tool_embeddings,
            query_embedding=query_embedding,
        )
        candidates = tuple(item.id for item in summaries)
        complexity = (
            RequestComplexity.MULTI_DOMAIN if len(candidates) > 1 else RequestComplexity.AMBIGUOUS
        )
        return RouteDecision(
            complexity=complexity,
            mode="discovery" if candidates else "direct",
            candidates=candidates,
            selected=candidates,
            reason="bounded lexical+semantic eligible shortlist"
            if candidates
            else "no eligible lexical or semantic match; direct conversation remains available",
            catalog_version=self._catalog.version,
        )
