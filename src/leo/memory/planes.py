"""Typed boundaries between source messages, memory, evidence, and derived artifacts."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class DataPlane(StrEnum):
    MESSAGE = "message"
    DOMAIN = "domain"
    EVIDENCE = "evidence"
    MEMORY = "memory"
    SUMMARY = "summary"
    EMBEDDING = "embedding"
    CACHE = "cache"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SanitizedMessage(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    destination_id: NonEmptyStr
    external_event_id: NonEmptyStr
    text: NonEmptyStr = Field(max_length=8192)
    content_hash: str = Field(min_length=64, max_length=64)
    recorded_at: datetime
    conversation_id: str | None = None
    harness_thread_id: str | None = None
    actor_id: str | None = None
    role: MessageRole = MessageRole.USER
    provider_message_ts: str | None = None
    context_access_hash: str | None = Field(
        default=None,
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )

    @model_validator(mode="after")
    def content_hash_matches(self) -> SanitizedMessage:
        if hashlib.sha256(self.text.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("sanitized message content hash mismatch")
        return self

    @classmethod
    def from_text(
        cls,
        *,
        id: str,
        scope: ScopeKey,
        destination_id: str,
        external_event_id: str,
        text: str,
        recorded_at: datetime,
        conversation_id: str | None = None,
        harness_thread_id: str | None = None,
        actor_id: str | None = None,
        role: MessageRole = MessageRole.USER,
        provider_message_ts: str | None = None,
        context_access_hash: str | None = None,
    ) -> SanitizedMessage:
        sanitized = sanitize_message_text(text)
        return cls(
            id=id,
            scope=scope,
            destination_id=destination_id,
            external_event_id=external_event_id,
            text=sanitized,
            content_hash=hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
            recorded_at=recorded_at,
            conversation_id=conversation_id,
            harness_thread_id=harness_thread_id,
            actor_id=actor_id,
            role=role,
            provider_message_ts=provider_message_ts,
            context_access_hash=context_access_hash,
        )


class SummaryRevision(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    thread_id: NonEmptyStr
    source_message_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    revision: int = Field(ge=1)
    content: NonEmptyStr = Field(max_length=8192)
    content_hash: str = Field(min_length=64, max_length=64)
    created_at: datetime

    @model_validator(mode="after")
    def content_hash_matches(self) -> SummaryRevision:
        if hashlib.sha256(self.content.encode("utf-8")).hexdigest() != self.content_hash:
            raise ValueError("summary content hash mismatch")
        return self


class EmbeddingJob(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    source_plane: DataPlane = DataPlane.MEMORY
    source_id: NonEmptyStr
    content_hash: str = Field(min_length=64, max_length=64)
    model: NonEmptyStr
    dimensions: int = Field(default=1536, ge=1)
    status: str = Field(pattern=r"^(queued|retry|succeeded|dead)$")
    attempts: int = Field(default=0, ge=0)


class RetrievalCacheEntry(ContractModel):
    id: NonEmptyStr
    scope: ScopeKey
    key_hash: str = Field(min_length=64, max_length=64)
    generation: int = Field(ge=1)
    result_ids: tuple[NonEmptyStr, ...] = ()
    expires_at: datetime | None = None


class InMemoryDerivedPlaneStore:
    """Derived-plane operations never mutate source messages or authoritative domain state."""

    def __init__(self) -> None:
        self.messages: dict[str, SanitizedMessage] = {}
        self.summaries: dict[str, SummaryRevision] = {}
        self.embedding_jobs: dict[str, EmbeddingJob] = {}
        self.cache: dict[str, RetrievalCacheEntry] = {}

    def add_message(self, message: SanitizedMessage) -> None:
        if message.id in self.messages and self.messages[message.id] != message:
            raise ValueError("message ID is immutable")
        self.messages[message.id] = message

    def add_summary(self, summary: SummaryRevision) -> None:
        if not set(summary.source_message_ids).issubset(self.messages):
            raise ValueError("summary references an unknown source message")
        self.summaries[summary.id] = summary

    def drop_summary(self, summary_id: str) -> None:
        self.summaries.pop(summary_id, None)

    def add_embedding_job(self, job: EmbeddingJob) -> None:
        existing = self.embedding_jobs.get(job.id)
        if existing is not None and existing != job:
            raise ValueError("embedding job ID is immutable")
        self.embedding_jobs[job.id] = job

    def put_cache(self, entry: RetrievalCacheEntry) -> None:
        self.cache[entry.id] = entry


def sanitize_message_text(text: str) -> str:
    if not text.strip():
        raise ValueError("message text must be non-empty")
    sanitized = "".join(char for char in text if char in "\n\r\t" or ord(char) >= 32)
    sanitized = re.sub(
        r"(?i)\b(?:authorization|api[_-]?key|token|secret|password)\s*[:=]\s*\S+",
        "[REDACTED]",
        sanitized,
    ).strip()
    if not sanitized or len(sanitized) > 8192:
        raise ValueError("sanitized message text must be 1-8192 characters")
    return sanitized
