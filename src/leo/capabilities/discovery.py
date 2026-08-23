"""Bounded lexical capability discovery over the already eligible catalog."""

from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import sha256

from pydantic import Field

from leo.capabilities.catalog import CatalogTool, InMemoryToolCatalog, ToolCatalogError
from leo.capabilities.embeddings import EmbeddingVector, cosine_similarity
from leo.harness.fusion import reciprocal_rank_fusion
from leo.harness.models import ContractModel, NonEmptyStr, RunPhase, ScopeKey

_BM25_K1 = 1.5
_BM25_B = 0.75

_STOP_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "can",
        "could",
        "for",
        "from",
        "get",
        "give",
        "help",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "one",
        "please",
        "read",
        "show",
        "the",
        "this",
        "to",
        "tool",
        "use",
        "what",
        "with",
        "would",
        "you",
    }
)
_ALIASES = (
    frozenset({"quote", "price", "pricing"}),
    frozenset(
        {
            "coin",
            "crypto",
            "cryptocurrency",
            "market-data",
            "token",
        }
    ),
    frozenset({"bitcoin", "btc"}),
    frozenset({"ethereum", "eth", "ether"}),
    frozenset(
        {
            "agreement",
            "compare",
            "corroborate",
            "corroboration",
            "cross-check",
            "divergence",
            "redundancy",
        }
    ),
    frozenset({"current", "latest", "recent", "now"}),
    frozenset({"filing", "filings", "disclosure", "disclosures", "10-k", "10-q"}),
    frozenset({"company", "profile", "identity", "listing", "exchange", "industry"}),
    frozenset(
        {
            "changed",
            "development",
            "developments",
            "happened",
            "headline",
            "headlines",
            "news",
            "story",
            "stories",
            "update",
            "updates",
        }
    ),
    frozenset({"earnings", "eps", "surprise", "surprises", "estimate", "actual"}),
    frozenset(
        {"beta", "financial", "financials", "fundamental", "fundamentals", "metric", "metrics"}
    ),
    frozenset({"web", "internet", "online", "url", "page", "website"}),
    frozenset({"delegate", "delegated", "subagent", "subagents"}),
    frozenset({"plan", "compare", "comparison", "multi-source", "synthesize"}),
)


class DiscoveryQuery(ContractModel):
    query: NonEmptyStr = Field(max_length=256)
    limit: int = Field(default=5, ge=1, le=20)
    max_bytes: int = Field(default=4096, ge=256, le=32_768)


class CapabilitySummary(ContractModel):
    id: NonEmptyStr
    version: NonEmptyStr
    short_description: NonEmptyStr
    tags: frozenset[NonEmptyStr]
    schema_fingerprint: NonEmptyStr
    score: float = Field(ge=0)


class DiscoveryBroker:
    def __init__(self, catalog: InMemoryToolCatalog) -> None:
        self._catalog = catalog

    def search(
        self,
        request: DiscoveryQuery,
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
        tool_embeddings: dict[str, EmbeddingVector] | None = None,
        query_embedding: EmbeddingVector | None = None,
    ) -> tuple[CapabilitySummary, ...]:
        tokens = _tokens(request.query)
        if not tokens and query_embedding is None:
            return ()
        eligible = self._catalog.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        documents = tuple(
            (
                record,
                _tokens(
                    " ".join(
                        (
                            record.id,
                            record.spec.domain,
                            record.short_description,
                            *record.tags,
                        )
                    )
                ),
            )
            for record in eligible
        )
        lexical_ranked = tuple(
            (record, score) for record, score in _bm25_rank(tokens, documents) if score > 0
        )
        semantic_ranked: tuple[tuple[CatalogTool, float], ...] = ()
        if query_embedding is not None and tool_embeddings:
            semantic_ranked = tuple(
                (record, cosine_similarity(query_embedding, tool_embeddings[record.id]))
                for record, _tokens_unused in documents
                if record.id in tool_embeddings
            )
        ranked = [
            CapabilitySummary(
                id=record.id,
                version=record.semantic_version,
                short_description=record.short_description,
                tags=record.tags,
                schema_fingerprint=record.schema_fingerprint,
                score=score,
            )
            for record, score in reciprocal_rank_fusion(
                lexical_ranked, semantic_ranked, key=lambda record: record.id
            )
        ]
        ranked.sort(key=lambda item: (-item.score, item.id, item.version))
        return tuple(ranked[: request.limit])

    def describe(
        self,
        capability_ids: tuple[str, ...],
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
        max_bytes: int = 16_384,
    ) -> tuple[CatalogTool, ...]:
        eligible = self._catalog.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        by_id = {record.id: record for record in eligible}
        records: list[CatalogTool] = []
        used_bytes = 0
        for capability_id in capability_ids:
            record = by_id.get(capability_id)
            if record is None:
                raise ToolCatalogError("capability_not_eligible")
            size = len(record.model_dump_json().encode("utf-8"))
            if used_bytes + size > max_bytes:
                raise ToolCatalogError("describe_budget_exceeded")
            used_bytes += size
            records.append(record)
        return tuple(records)


def _bm25_rank(
    query_tokens: tuple[str, ...],
    documents: tuple[tuple[CatalogTool, tuple[str, ...]], ...],
) -> tuple[tuple[CatalogTool, float], ...]:
    """Rank eligible tools by Okapi BM25 over their catalog summary tokens.

    Plain query/document token-overlap ratio (the prior scorer) weighs every
    matched term identically and ignores summary length, so a rare distinguishing
    term (e.g. a specific provider name) scores no higher than a common one (e.g.
    "market"), and a short, precise summary is not favored over a verbose one that
    happens to share a word. BM25 adds inverse-document-frequency weighting and
    length normalization, both well-understood, embedding-free improvements over
    raw overlap -- this stays deterministic and adds no new dependency.
    """

    document_count = len(documents)
    if document_count == 0:
        return ()
    document_lengths = tuple(len(doc_tokens) for _, doc_tokens in documents)
    average_length = sum(document_lengths) / document_count
    unique_query_tokens = tuple(dict.fromkeys(query_tokens))
    document_frequency = {
        token: sum(1 for _, doc_tokens in documents if token in doc_tokens)
        for token in unique_query_tokens
    }
    inverse_document_frequency = {
        token: math.log((document_count + 1) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
        if frequency > 0
    }
    scored: list[tuple[CatalogTool, float]] = []
    for (record, doc_tokens), length in zip(documents, document_lengths, strict=True):
        term_counts = Counter(doc_tokens)
        score = 0.0
        for token in unique_query_tokens:
            idf = inverse_document_frequency.get(token)
            term_frequency = term_counts.get(token, 0)
            if idf is None or term_frequency == 0:
                continue
            normalized_length = length / average_length if average_length else 1.0
            denominator = term_frequency + _BM25_K1 * (1 - _BM25_B + _BM25_B * normalized_length)
            score += idf * (term_frequency * (_BM25_K1 + 1)) / denominator
        scored.append((record, score))
    return tuple(scored)


def _tokens(value: str) -> tuple[str, ...]:
    raw = tuple(
        token
        for token in re.findall(r"[a-z0-9_.-]{2,64}", value.lower())
        if token not in _STOP_TOKENS
    )
    expanded: list[str] = list(raw)
    for token in raw:
        expanded.extend(
            part
            for part in re.split(r"[._-]+", token)
            if len(part) >= 2 and part not in _STOP_TOKENS
        )
    for token in tuple(dict.fromkeys(expanded)):
        singular = token[:-1] if token.endswith("s") and len(token) > 3 else token
        if singular != token:
            expanded.append(singular)
        for group in _ALIASES:
            if token in group or singular in group:
                expanded.extend(group)
    return tuple(dict.fromkeys(expanded))


def query_hash(query: str) -> str:
    return sha256(query.encode("utf-8")).hexdigest()


def search_tokens(value: str) -> tuple[str, ...]:
    """Return the deterministic lexical vocabulary shared by routing and skills."""

    return _tokens(value)
