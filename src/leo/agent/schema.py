"""The agent's durable schema: six tables, one isolation key.

Everything a conversation owns -- its messages, its runs, the trace of each
run, and its memories -- hangs off ``scope_key``. Reads filter on it in SQL, so
isolation between a channel and a DM is a property of the query rather than of
a policy layer that can be bypassed or misconfigured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from leo.agent.llm import EMBEDDING_DIMENSIONS


class Base(DeclarativeBase):
    pass


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Conversation(Base):
    """One Slack channel, group, or DM."""

    __tablename__ = "agent_conversations"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(32), nullable=False, default="slack")
    team_id: Mapped[str | None] = mapped_column(String(64))
    channel_id: Mapped[str | None] = mapped_column(String(64))
    kind: Mapped[str] = mapped_column(String(16), nullable=False, default="channel")
    title: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = _now()
    last_active_at: Mapped[datetime] = _now()


class Message(Base):
    """Conversation history: what the user said, what Leo answered.

    Only the two roles the *next* turn needs to read. A run's internal tool
    traffic is a trace, not history, and lives in :class:`Step` -- replaying raw
    tool JSON into later prompts is how a context window fills with noise.
    """

    __tablename__ = "agent_messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False
    )
    run_id: Mapped[str | None] = mapped_column(String(64))
    thread_key: Mapped[str | None] = mapped_column(String(128))
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    author_id: Mapped[str | None] = mapped_column(String(64))
    external_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index("ix_agent_messages_scope_created", "scope_key", "created_at"),
        Index("ix_agent_messages_thread", "scope_key", "thread_key", "created_at"),
        # Slack redelivers events. A partial unique index makes ingest idempotent
        # without a lock or a dedupe table.
        Index(
            "uq_agent_messages_external",
            "scope_key",
            "external_id",
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        ),
    )


class Run(Base):
    """One user request handled end to end."""

    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_conversations.id", ondelete="CASCADE"), nullable=False
    )
    actor_id: Mapped[str | None] = mapped_column(String(64))
    thread_key: Mapped[str | None] = mapped_column(String(128))
    question: Mapped[str] = mapped_column(Text, nullable=False, default="")
    answer: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="running")
    error: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(String(128))
    turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    started_at: Mapped[datetime] = _now()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (Index("ix_agent_runs_scope_started", "scope_key", "started_at"),)


class Step(Base):
    """The replayable trace of a run: every model turn and every tool call."""

    __tablename__ = "agent_steps"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    arguments: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    ok: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (Index("ix_agent_steps_run_seq", "run_id", "seq"),)


class Memory(Base):
    """A durable fact, preference, or decision, owned by exactly one scope.

    ``superseded_by`` makes updates non-destructive: revising a memory writes a
    new row and points the old one at it, so the history of what Leo believed
    stays inspectable.
    """

    __tablename__ = "agent_memories"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    scope_key: Mapped[str] = mapped_column(String(255), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="fact")
    subject: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    source_run_id: Mapped[str | None] = mapped_column(String(64))
    author_id: Mapped[str | None] = mapped_column(String(64))
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    superseded_by: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index("ix_agent_memories_scope_active", "scope_key", "active"),
        Index(
            "ix_agent_memories_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )


class ToolIndex(Base):
    """Cached embeddings of tool descriptions, for semantic tool discovery."""

    __tablename__ = "agent_tool_index"

    name: Mapped[str] = mapped_column(String(128), primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSIONS))
    updated_at: Mapped[datetime] = _now()
