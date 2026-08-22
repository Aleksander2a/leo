"""Immutable synthetic/public retrieval corpus and content-addressed labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime

from pydantic import Field

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.models import MemoryVisibility


class CorpusEntry(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    content: NonEmptyStr
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    relevant_queries: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    recorded_at: datetime


class FrozenCorpus(ContractModel):
    version: NonEmptyStr = "corpus-v1"
    digest: NonEmptyStr = Field(min_length=64, max_length=64)
    entries: tuple[CorpusEntry, ...]


def freeze_corpus(entries: tuple[CorpusEntry, ...]) -> FrozenCorpus:
    ordered = tuple(sorted(entries, key=lambda entry: entry.id))
    if len({entry.id for entry in ordered}) != len(ordered):
        raise ValueError("corpus entry IDs must be unique")
    payload = [entry.model_dump(mode="json") for entry in ordered]
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return FrozenCorpus(digest=digest, entries=ordered)
