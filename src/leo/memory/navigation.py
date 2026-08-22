"""Progressive, scope-bound memory cards, chunks, and opaque handle contracts."""

from __future__ import annotations

import hashlib
import re
from enum import StrEnum

from pydantic import Field, model_validator

from leo.domain.conversation import ConversationKind
from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.models import MemoryStatus, MemoryVisibility
from leo.memory.retrieval import (
    AuthorizedMemoryNamespace,
    channel_authorized_namespaces,
    dm_authorized_namespaces,
)

NAVIGATION_POLICY_VERSION = "memory-navigation-v1"


class MemoryNavigationError(RuntimeError):
    """A safe, typed denial or bounded navigation failure."""

    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class MemoryNavigationAuthority(ContractModel):
    """Sealed run authority derived from one admitted Slack access snapshot."""

    scope: ScopeKey
    team_id: NonEmptyStr
    destination_id: NonEmptyStr
    destination_kind: ConversationKind
    actor_id: NonEmptyStr
    task_id: NonEmptyStr
    run_id: NonEmptyStr
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    current_thread_namespace_id: NonEmptyStr

    @model_validator(mode="after")
    def validate_projection(self) -> MemoryNavigationAuthority:
        normalized = tuple(sorted(set(self.allowed_conversation_ids)))
        if normalized != self.allowed_conversation_ids:
            raise ValueError("memory navigation conversations must be sorted and unique")
        if self.destination_id not in normalized:
            raise ValueError("memory navigation destination is missing from the source set")
        if self.destination_kind is not ConversationKind.DM and normalized != (
            self.destination_id,
        ):
            raise ValueError("non-DM memory navigation must use the exact destination")
        if self.destination_kind not in {
            ConversationKind.CHANNEL,
            ConversationKind.DM,
            ConversationKind.GROUP_DM,
            ConversationKind.SHARED,
            ConversationKind.EXTERNAL,
        }:
            raise ValueError("memory navigation destination is not admitted")
        return self

    @property
    def authorized_namespaces(self) -> frozenset[AuthorizedMemoryNamespace]:
        if self.destination_kind is ConversationKind.DM:
            return dm_authorized_namespaces(
                self.allowed_conversation_ids,
                actor_id=self.actor_id,
                thread_namespace_id=self.current_thread_namespace_id,
            )
        return channel_authorized_namespaces(
            self.destination_id,
            thread_namespace_id=self.current_thread_namespace_id,
        )


class MemoryResultKind(StrEnum):
    INLINE = "inline"
    CARD = "card"


class ProgressiveMemoryItem(ContractModel):
    """Model-visible result; internal record and database IDs are deliberately absent."""

    kind: MemoryResultKind
    reference: NonEmptyStr
    content: str | None = Field(default=None, max_length=1_200)
    excerpt: str | None = Field(default=None, max_length=600)
    handle: str | None = Field(default=None, min_length=16, max_length=256)
    chunk_count: int = Field(default=0, ge=0, le=128)
    source_conversation: NonEmptyStr
    lifecycle_status: MemoryStatus
    contested: bool = False

    @model_validator(mode="after")
    def validate_shape(self) -> ProgressiveMemoryItem:
        if self.kind is MemoryResultKind.INLINE:
            if self.content is None or self.excerpt is not None or self.handle is not None:
                raise ValueError("inline memory result has an invalid shape")
            if self.chunk_count != 0:
                raise ValueError("inline memory result cannot declare chunks")
        elif self.content is not None or self.excerpt is None or self.handle is None:
            raise ValueError("memory card result has an invalid shape")
        return self


class ProgressiveMemorySearchResult(ContractModel):
    items: tuple[ProgressiveMemoryItem, ...]
    query_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    selected_count: int = Field(ge=0)
    cache_status: str = Field(pattern=r"^(hit|miss|disabled)$")
    policy_version: NonEmptyStr = NAVIGATION_POLICY_VERSION


class MemoryChunkView(ContractModel):
    ordinal: int = Field(ge=0)
    text: NonEmptyStr = Field(max_length=1_200)


class ProgressiveMemoryOpenResult(ContractModel):
    reference: NonEmptyStr
    handle: NonEmptyStr
    chunks: tuple[MemoryChunkView, ...] = Field(min_length=1, max_length=16)
    next_ordinal: int | None = Field(default=None, ge=0)
    source_conversation: NonEmptyStr
    revision: int = Field(ge=1)
    policy_version: NonEmptyStr = NAVIGATION_POLICY_VERSION


class AuthorizedMemoryDocument(ContractModel):
    """Internal document returned only after atomic handle reauthorization."""

    record_id: NonEmptyStr
    revision: int = Field(ge=1)
    content: NonEmptyStr = Field(max_length=16_384)
    content_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    status: MemoryStatus
    handle: NonEmptyStr
    reference: NonEmptyStr


def membership_snapshot_hash(conversation_ids: tuple[str, ...]) -> str:
    """Hash the exact active source set; optional domain labels never participate."""

    normalized = tuple(sorted(set(conversation_ids)))
    if not normalized or normalized != conversation_ids:
        raise ValueError("membership source set must be non-empty, sorted, and unique")
    return hashlib.sha256("\x1f".join(normalized).encode("utf-8")).hexdigest()


def opaque_memory_reference(record_id: str, revision: int, access_hash: str) -> str:
    material = f"{record_id}\x1f{revision}\x1f{access_hash}"
    return f"mem_{hashlib.sha256(material.encode()).hexdigest()[:24]}"


def source_conversation_label(visibility: MemoryVisibility, namespace_id: str) -> str:
    if visibility is MemoryVisibility.ACTOR_PRIVATE:
        return "actor-private"
    if visibility is MemoryVisibility.THREAD_LOCAL:
        return "current-thread"
    return namespace_id


def deterministic_memory_chunks(
    content: str,
    *,
    max_chars: int = 1_000,
    overlap_chars: int = 120,
) -> tuple[str, ...]:
    """Split long text deterministically while retaining modest local overlap."""

    if max_chars < 200 or max_chars > 1_200:
        raise ValueError("memory chunk size must be between 200 and 1200 characters")
    if overlap_chars < 0 or overlap_chars >= max_chars // 2:
        raise ValueError("memory chunk overlap is invalid")
    normalized = content.strip()
    if not normalized:
        raise ValueError("memory content cannot be empty")
    chunks: list[str] = []
    cursor = 0
    while cursor < len(normalized):
        end = min(len(normalized), cursor + max_chars)
        if end < len(normalized):
            window = normalized[cursor:end]
            boundaries = [match.end() for match in re.finditer(r"(?:[.!?]\s+|\n+)", window)]
            if boundaries and boundaries[-1] >= max_chars // 2:
                end = cursor + boundaries[-1]
        chunk = normalized[cursor:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_cursor = max(cursor + 1, end - overlap_chars)
        cursor = next_cursor
    if len(chunks) > 128:
        raise MemoryNavigationError("memory_document_exceeds_chunk_budget")
    return tuple(chunks)


def project_open_window(
    document: AuthorizedMemoryDocument,
    *,
    start_ordinal: int = 0,
    max_chunks: int = 4,
    query: str | None = None,
) -> ProgressiveMemoryOpenResult:
    if start_ordinal < 0:
        raise MemoryNavigationError("memory_chunk_ordinal_invalid")
    if max_chunks < 1 or max_chunks > 16:
        raise MemoryNavigationError("memory_open_budget_invalid")
    chunks = deterministic_memory_chunks(document.content)
    if query is not None:
        tokens = _navigation_tokens(query)
        if not tokens:
            raise MemoryNavigationError("empty_search_within_query")
        selected_ordinals = tuple(
            ordinal
            for ordinal, chunk in enumerate(chunks)
            if tokens.issubset(_navigation_tokens(chunk))
        )
        selected_ordinals = selected_ordinals[:max_chunks]
        if not selected_ordinals:
            return ProgressiveMemoryOpenResult(
                reference=document.reference,
                handle=document.handle,
                chunks=(MemoryChunkView(ordinal=0, text="No matching authorized chunk."),),
                next_ordinal=None,
                source_conversation=source_conversation_label(
                    document.visibility, document.namespace_id
                ),
                revision=document.revision,
            )
        views = tuple(
            MemoryChunkView(ordinal=ordinal, text=chunks[ordinal]) for ordinal in selected_ordinals
        )
        return ProgressiveMemoryOpenResult(
            reference=document.reference,
            handle=document.handle,
            chunks=views,
            next_ordinal=None,
            source_conversation=source_conversation_label(
                document.visibility, document.namespace_id
            ),
            revision=document.revision,
        )
    if start_ordinal >= len(chunks):
        raise MemoryNavigationError("memory_chunk_ordinal_out_of_range")
    end = min(len(chunks), start_ordinal + max_chunks)
    return ProgressiveMemoryOpenResult(
        reference=document.reference,
        handle=document.handle,
        chunks=tuple(
            MemoryChunkView(ordinal=ordinal, text=chunks[ordinal])
            for ordinal in range(start_ordinal, end)
        ),
        next_ordinal=end if end < len(chunks) else None,
        source_conversation=source_conversation_label(document.visibility, document.namespace_id),
        revision=document.revision,
    )


def _navigation_tokens(value: str) -> frozenset[str]:
    """Keep dots/dashes only inside tokens, never as trailing sentence punctuation."""

    return frozenset(
        match.group(0).lower() for match in re.finditer(r"[\w]{2,64}(?:[.-][\w]{1,64})*", value)
    )
