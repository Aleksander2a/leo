from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.domain.conversation import ConversationKind
from leo.harness.models import OriginRef, Run, ScopeKey, Task, Thread
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)
from leo.memory.navigation import (
    MemoryNavigationAuthority,
    MemoryNavigationError,
    MemoryResultKind,
    membership_snapshot_hash,
)
from leo.persistence.memory_navigation import PostgresProgressiveMemoryService
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import ConversationActorMembershipRow, ConversationRow

NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="navigation-org", strategy_id="destination-domain")
SOURCE_SCOPE = ScopeKey(organization_id=SCOPE.organization_id, strategy_id="source-domain")


@pytest_asyncio.fixture
async def navigation_harness(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[
    tuple[
        async_sessionmaker[AsyncSession],
        PostgresMemoryStore,
        PostgresProgressiveMemoryService,
    ]
]:
    sessions = preserved_postgres_sessions
    run_store = PostgresRunStore(sessions, FixedClock(), SequentialIdGenerator())
    thread = Thread(
        id="thread-navigation",
        scope=SCOPE,
        origin=OriginRef(
            provider="slack",
            external_thread_id="slack:T-navigation:D-navigation:100.1",
            external_event_id="event-navigation",
            external_channel_id="D-navigation",
        ),
    )
    task = Task(
        id="task-navigation",
        thread_id=thread.id,
        scope=SCOPE,
        objective="Recall the synthetic long memory.",
    )
    run = Run(id="run-navigation", task_id=task.id, scope=SCOPE)
    await run_store.seed(thread, task, run)
    access_hash = "a" * 64
    async with sessions() as session, session.begin():
        for external_id, kind, actor_id in (
            ("C-navigation", "channel", None),
            ("D-navigation", "dm", "U-navigation"),
        ):
            session.add(
                ConversationRow(
                    id=f"conversation-{external_id}",
                    provider="slack",
                    team_id="T-navigation",
                    external_id=external_id,
                    kind=kind,
                    actor_id=actor_id,
                    authority_source="slack_conversations_info",
                    bot_presence="present",
                    lifecycle="active",
                    external_provenance=("not_applicable" if kind == "dm" else "internal"),
                    membership_policy_version=1,
                    version=1,
                )
            )
            session.add(
                ConversationActorMembershipRow(
                    id=f"membership-{external_id}",
                    organization_id=SCOPE.organization_id,
                    team_id="T-navigation",
                    actor_id="U-navigation",
                    conversation_external_id=external_id,
                    status="active",
                    source_kind="dm_membership_intersection",
                    context_access_hash=access_hash,
                    version=1,
                    observed_at=NOW,
                )
            )
    memory = PostgresMemoryStore(sessions)
    yield sessions, memory, PostgresProgressiveMemoryService(sessions)


def _authority() -> MemoryNavigationAuthority:
    source_ids = ("C-navigation", "D-navigation")
    return MemoryNavigationAuthority(
        scope=SCOPE,
        team_id="T-navigation",
        destination_id="D-navigation",
        destination_kind=ConversationKind.DM,
        actor_id="U-navigation",
        task_id="task-navigation",
        run_id="run-navigation",
        allowed_conversation_ids=source_ids,
        access_hash="a" * 64,
        membership_hash=membership_snapshot_hash(source_ids),
        current_thread_namespace_id="slack:T-navigation:D-navigation:100.1",
    )


async def _seed_memory(
    store: PostgresMemoryStore,
    *,
    record_id: str,
    namespace_id: str,
    content: str,
    scope: ScopeKey = SOURCE_SCOPE,
    source_count: int = 1,
) -> None:
    sources = tuple(
        MemorySource(
            id=f"source-{record_id}-{index}",
            scope=scope,
            source_kind="synthetic",
            reference=f"fixture:{record_id}:{index}",
            visibility=MemoryVisibility.CONVERSATION_LOCAL,
            namespace_id=namespace_id,
        )
        for index in range(source_count)
    )
    record = MemoryRecord(
        id=record_id,
        scope=scope,
        kind=MemoryKind.NOTE,
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id=namespace_id,
        created_at=NOW,
    )
    revision = MemoryRevision.from_content(
        id=f"revision-{record_id}",
        record_id=record_id,
        number=1,
        content=content,
        source_ids=tuple(source.id for source in sources),
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id=namespace_id,
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="U-navigation",
        reason="synthetic navigation fixture",
    )
    await store.create(record, revision, sources)


@pytest.mark.asyncio
async def test_inline_hit_projection_preserves_multi_source_memory_shape(
    navigation_harness: tuple[
        async_sessionmaker[AsyncSession],
        PostgresMemoryStore,
        PostgresProgressiveMemoryService,
    ],
) -> None:
    _sessions, memory, service = navigation_harness
    expected = "Project Borealis's display preference is amber hexagons."
    await _seed_memory(
        memory,
        record_id="memory-navigation-borealis",
        namespace_id="C-navigation",
        content=expected,
        source_count=3,
    )

    result = await service.search(_authority(), query="project borealis", now=NOW)

    assert result.selected_count == 1
    assert result.items[0].kind is MemoryResultKind.INLINE
    assert result.items[0].content == expected
    assert result.items[0].source_conversation == "C-navigation"


@pytest.mark.asyncio
async def test_progressive_navigation_cache_parity_and_repeat_authorization(
    navigation_harness: tuple[
        async_sessionmaker[AsyncSession],
        PostgresMemoryStore,
        PostgresProgressiveMemoryService,
    ],
) -> None:
    sessions, memory, service = navigation_harness
    long_content = " ".join(
        f"Synthetic navigation evidence segment {index}." for index in range(120)
    )
    await _seed_memory(
        memory,
        record_id="memory-navigation-long",
        namespace_id="C-navigation",
        content=long_content,
    )
    await _seed_memory(
        memory,
        record_id="memory-navigation-short",
        namespace_id="D-navigation",
        content="Synthetic navigation evidence is also present in the current DM.",
    )
    await _seed_memory(
        memory,
        record_id="memory-navigation-cross-org",
        namespace_id="C-navigation",
        content="Synthetic navigation evidence from another organization must never appear.",
        scope=ScopeKey(organization_id="other-navigation-org", strategy_id="foreign-domain"),
    )

    first = await service.search(_authority(), query="Synthetic navigation evidence", now=NOW)
    second = await service.search(_authority(), query="Synthetic navigation evidence", now=NOW)
    assert first.cache_status == "miss"
    assert second.cache_status == "hit"
    assert tuple((item.kind, item.reference) for item in second.items) == tuple(
        (item.kind, item.reference) for item in first.items
    )
    projected_payload = first.model_dump_json()
    # Both authorized memories were written under source-domain while the admitted
    # DM run is destination-domain: strategy is provenance, not an authority gate.
    assert "current DM" in projected_payload
    assert "evidence segment" in projected_payload
    # Organization plus exact authorized conversation namespaces remain hard gates.
    assert "another organization" not in projected_payload
    card = next(item for item in first.items if item.kind is MemoryResultKind.CARD)
    assert card.handle is not None
    opened = await service.open(_authority(), handle=card.handle, now=NOW)
    assert opened.chunks[0].ordinal == 0
    assert "memory-navigation-long" not in opened.model_dump_json()

    with pytest.raises(MemoryNavigationError, match="not_authorized"):
        await service.open(
            _authority().model_copy(update={"run_id": "run-forged"}),
            handle=card.handle,
            now=NOW,
        )

    async with sessions() as session, session.begin():
        await session.execute(
            update(ConversationActorMembershipRow)
            .where(
                ConversationActorMembershipRow.organization_id == SCOPE.organization_id,
                ConversationActorMembershipRow.team_id == "T-navigation",
                ConversationActorMembershipRow.actor_id == "U-navigation",
                ConversationActorMembershipRow.conversation_external_id == "C-navigation",
            )
            .values(status="revoked", version=2)
        )
    with pytest.raises(MemoryNavigationError, match="revoked"):
        await service.open(_authority(), handle=card.handle, now=NOW)


@pytest.mark.asyncio
async def test_generation_change_invalidates_existing_progressive_handles(
    navigation_harness: tuple[
        async_sessionmaker[AsyncSession],
        PostgresMemoryStore,
        PostgresProgressiveMemoryService,
    ],
) -> None:
    _sessions, memory, service = navigation_harness
    await _seed_memory(
        memory,
        record_id="memory-navigation-forget",
        namespace_id="C-navigation",
        content=" ".join(
            f"Synthetic invalidation evidence segment {index}." for index in range(120)
        ),
    )
    result = await service.search(
        _authority(),
        query="Synthetic invalidation evidence",
        now=NOW,
    )
    card = next(item for item in result.items if item.kind is MemoryResultKind.CARD)
    assert card.handle is not None

    await memory.forget(SOURCE_SCOPE, "memory-navigation-forget", "confirmed demo forget")

    with pytest.raises(MemoryNavigationError, match="invalidated"):
        await service.open(_authority(), handle=card.handle, now=NOW)
