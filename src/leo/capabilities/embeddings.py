"""Optional embedding-based semantic recall for tool/capability discovery.

Lexical (BM25) matching alone cannot bridge a vocabulary gap between a conceptual
query ("prognosis for the S&P 500") and a tool's literal description ("search the
public web"). This module adds a parallel semantic signal, fused with the lexical
score via reciprocal rank fusion -- exactly the "lexical + embedding recall ->
fusion" pipeline the harness design already specifies. It is a pure, optional
enhancement: if no gateway is configured, or a request fails, callers fall back to
lexical-only ranking with no error and no degraded availability.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

EmbeddingVector = tuple[float, ...]
CapabilityEmbeddingKey = tuple[str, str, str]
"""(capability_id, content_hash, model) -- identity for one persisted embedding."""

_DEFAULT_MODEL = "openai/text-embedding-3-small"

# Process-wide cache: a tool's catalog summary text is stable for the process
# lifetime (it only changes on a code deploy/restart), so repeated turns never
# re-embed the same tool. Keyed by (tool_id, content_hash) so a changed
# description naturally invalidates its own cache entry rather than serving a
# stale vector.
_TOOL_EMBEDDING_CACHE: dict[str, EmbeddingVector] = {}


class _EmbeddingDatum(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    embedding: tuple[float, ...] = ()
    index: int = 0


class _EmbeddingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: tuple[_EmbeddingDatum, ...] = Field(default=())


class OpenRouterEmbeddingGateway:
    """Batch text embeddings through OpenRouter's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 10.0,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        self._client = client
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout_seconds = timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector | None, ...]:
        """Embed a batch of texts. Returns a same-length tuple; None per item on failure."""

        if not texts:
            return ()
        try:
            response = await self._client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": list(texts)},
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            parsed = _EmbeddingResponse.model_validate(response.json())
        except (httpx.HTTPError, ValueError):
            return tuple(None for _ in texts)
        by_index = {item.index: item.embedding for item in parsed.data if item.embedding}
        return tuple(by_index.get(index) for index in range(len(texts)))


class CapabilityEmbeddingCache(Protocol):
    """Durable L2 cache behind the in-process ``_TOOL_EMBEDDING_CACHE`` L1.

    Kept as a narrow protocol (not a concrete SQL dependency) so this module
    stays free of persistence imports; ``persistence.capability_embeddings``
    supplies the real Postgres-backed implementation.
    """

    async def get_many(
        self, keys: tuple[CapabilityEmbeddingKey, ...]
    ) -> dict[CapabilityEmbeddingKey, EmbeddingVector]: ...

    async def put_many(
        self, items: tuple[tuple[CapabilityEmbeddingKey, EmbeddingVector], ...]
    ) -> None: ...


def cosine_similarity(a: EmbeddingVector, b: EmbeddingVector) -> float:
    if not a or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


async def ensure_tool_embeddings(
    gateway: OpenRouterEmbeddingGateway | None,
    tools: tuple[tuple[str, str], ...],
    *,
    cache: CapabilityEmbeddingCache | None = None,
) -> dict[str, EmbeddingVector]:
    """Return a tool_id -> embedding map for the given (tool_id, searchable_text) pairs.

    Three tiers, cheapest first: the in-process L1 cache (this run's/process's own
    prior lookups), an optional durable L2 cache (survives process restarts), and
    finally the embedding gateway itself for anything still missing. Only L2/gateway
    misses cost a request. Returns an empty map -- never raises -- when no gateway is
    configured or every request fails, so callers can unconditionally fall back to
    lexical-only ranking.
    """

    if gateway is None or not tools:
        return {}
    l1_missing = tuple(
        (tool_id, text)
        for tool_id, text in tools
        if _cache_key(tool_id, text) not in _TOOL_EMBEDDING_CACHE
    )
    if l1_missing and cache is not None:
        l2_hits = await cache.get_many(
            tuple(_capability_key(tool_id, text, gateway.model) for tool_id, text in l1_missing)
        )
        for tool_id, text in l1_missing:
            vector = l2_hits.get(_capability_key(tool_id, text, gateway.model))
            if vector is not None:
                _TOOL_EMBEDDING_CACHE[_cache_key(tool_id, text)] = vector
        l1_missing = tuple(
            (tool_id, text)
            for tool_id, text in l1_missing
            if _cache_key(tool_id, text) not in _TOOL_EMBEDDING_CACHE
        )
    if l1_missing:
        embeddings = await gateway.embed(tuple(text for _, text in l1_missing))
        newly_embedded: list[tuple[CapabilityEmbeddingKey, EmbeddingVector]] = []
        for (tool_id, text), vector in zip(l1_missing, embeddings, strict=True):
            if vector is not None:
                _TOOL_EMBEDDING_CACHE[_cache_key(tool_id, text)] = vector
                newly_embedded.append((_capability_key(tool_id, text, gateway.model), vector))
        if newly_embedded and cache is not None:
            await cache.put_many(tuple(newly_embedded))
    return {
        tool_id: vector
        for tool_id, text in tools
        if (vector := _TOOL_EMBEDDING_CACHE.get(_cache_key(tool_id, text))) is not None
    }


def _cache_key(tool_id: str, text: str) -> str:
    return f"{tool_id}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def _capability_key(tool_id: str, text: str, model: str) -> CapabilityEmbeddingKey:
    return (tool_id, hashlib.sha256(text.encode("utf-8")).hexdigest(), model)
