from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.domain.conversation import ConversationKind, ConversationRef
from leo.domain.conversation_store import ConversationStoreError
from leo.harness.models import ScopeKey
from leo.integrations.fake import SequentialIdGenerator
from leo.persistence.conversation_store import PostgresConversationStore
from leo.persistence.schema import OrganizationRow, StrategyRow


@dataclass(frozen=True)
class ConversationStoreHarness:
    store: PostgresConversationStore
    scope: ScopeKey
    team_id: str


@pytest_asyncio.fixture
async def conversation_store(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[ConversationStoreHarness]:
    suffix = uuid4().hex
    organization_id = f"conversation-org-{suffix}"
    strategy_id = f"conversation-strategy-{suffix}"
    async with preserved_postgres_sessions() as session, session.begin():
        session.add(OrganizationRow(id=organization_id, name="Conversation Demo"))
        session.add(
            StrategyRow(
                id=strategy_id,
                organization_id=organization_id,
                name="Conversation Strategy",
                slug=f"conversation-strategy-{suffix}",
            )
        )
    yield ConversationStoreHarness(
        store=PostgresConversationStore(
            preserved_postgres_sessions,
            SequentialIdGenerator(),
        ),
        scope=ScopeKey(organization_id=organization_id, strategy_id=strategy_id),
        team_id=f"T{suffix[:15].upper()}",
    )


@pytest.mark.asyncio
async def test_postgres_thread_keeps_organization_boundary_without_mapping_gate(
    conversation_store: ConversationStoreHarness,
) -> None:
    scope = conversation_store.scope
    destination = ConversationRef(
        provider="slack",
        team_id=conversation_store.team_id,
        external_id=f"C{conversation_store.team_id[1:]}",
        kind=ConversationKind.CHANNEL,
    )
    first = await conversation_store.store.pin_thread(
        scope,
        destination,
        root_ts="100.1",
        mapping_version=2,
    )
    remapped_scope = ScopeKey(
        organization_id=scope.organization_id,
        strategy_id=f"alternate-{scope.strategy_id}"[:64],
    )
    assert (
        await conversation_store.store.pin_thread(
            remapped_scope,
            destination,
            root_ts="100.1",
            mapping_version=3,
        )
        == first
    )
    assert (
        await conversation_store.store.load_thread(remapped_scope, destination, root_ts="100.1")
        == first
    )
    other_org = ScopeKey(
        organization_id=f"other-{scope.organization_id}"[:64],
        strategy_id="any",
    )
    assert (
        await conversation_store.store.load_thread(other_org, destination, root_ts="100.1") is None
    )
    with pytest.raises(ConversationStoreError, match="thread_organization_changed"):
        await conversation_store.store.pin_thread(other_org, destination, root_ts="100.1")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", (ConversationKind.SHARED, ConversationKind.EXTERNAL))
async def test_postgres_shared_and_external_conversations_are_eligible(
    conversation_store: ConversationStoreHarness,
    kind: ConversationKind,
) -> None:
    scope = conversation_store.scope
    destination = ConversationRef(
        provider="slack",
        team_id=conversation_store.team_id,
        external_id=f"{kind.value}-{conversation_store.team_id[1:]}",
        kind=kind,
    )
    pinned = await conversation_store.store.pin_thread(scope, destination, root_ts="200.1")
    assert pinned.conversation == destination
