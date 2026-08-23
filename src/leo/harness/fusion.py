"""Reciprocal rank fusion for combining incomparable ranked signals.

Shared by tool/capability discovery and memory retrieval: both fuse a lexical
(BM25/FTS) ranking with a semantic (embedding cosine-similarity) ranking. The
two scores live on different scales, so summing or weighting them directly
would be arbitrary; RRF instead sums ``1 / (k + rank)`` per ranking an item
appears in -- a deterministic, parameter-light standard technique for
combining heterogeneous rankers.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion[T](
    *rankings: Sequence[tuple[T, float]],
    key: Callable[[T], Hashable],
    k: int = DEFAULT_RRF_K,
) -> tuple[tuple[T, float], ...]:
    """Combine any number of independently-scored rankings by rank position.

    Each input ranking is a sequence of (item, score) pairs; items are
    resorted internally by descending score to determine rank position, so
    callers do not need to pre-sort. An item need only appear in one ranking
    to be included in the result. Returns (item, fused_score) pairs in
    insertion order (by first appearance across rankings) -- callers sort by
    fused_score themselves.
    """

    combined: dict[Hashable, float] = {}
    items: dict[Hashable, T] = {}
    for ranking in rankings:
        ordered = sorted(ranking, key=lambda pair: -pair[1])
        for rank, (item, _score) in enumerate(ordered, start=1):
            item_key = key(item)
            combined[item_key] = combined.get(item_key, 0.0) + 1.0 / (k + rank)
            items[item_key] = item
    return tuple((items[item_key], score) for item_key, score in combined.items())
