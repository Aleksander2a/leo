from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.domain.conversation import ConversationKind
from leo.harness.models import OriginRef, Run, RunStatus, ScopeKey, Task, Thread, TrustedScope
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.live import _EMPTY_MEMORY_SCOPE_INFERENCE, run_live_conversation
from leo.memory.models import (
    MemoryKind,
    MemoryRecord,
    MemoryRevision,
    MemorySource,
    MemoryVisibility,
)
from leo.memory.navigation import MemoryNavigationAuthority, membership_snapshot_hash
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.run_store import PostgresRunStore
from leo.persistence.schema import ConversationActorMembershipRow, ConversationRow

NOW = datetime(2026, 8, 21, 12, tzinfo=UTC)
SCOPE = ScopeKey(organization_id="live-memory-org", strategy_id="live-memory-domain")


@pytest_asyncio.fixture
async def live_memory_sessions(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    sessions = preserved_postgres_sessions
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T-LIVE-MEM:C-LIVE-MEM:100.1",
        external_event_id="event-live-memory",
        external_channel_id="C-LIVE-MEM",
    )
    thread = Thread(id="thread-live-memory", scope=SCOPE, origin=origin)
    task = Task(
        id="task-live-memory",
        thread_id=thread.id,
        scope=SCOPE,
        objective="What do you remember about Project Borealis?",
    )
    run = Run(id="run-live-memory", task_id=task.id, scope=SCOPE)
    await PostgresRunStore(sessions, FixedClock(), SequentialIdGenerator()).seed(thread, task, run)
    async with sessions() as session, session.begin():
        session.add(
            ConversationRow(
                id="conversation-live-memory",
                provider="slack",
                team_id="T-LIVE-MEM",
                external_id="C-LIVE-MEM",
                kind="channel",
                actor_id=None,
                authority_source="slack_conversations_info",
                bot_presence="present",
                lifecycle="active",
                external_provenance="internal",
                membership_policy_version=1,
                version=1,
            )
        )
        session.add(
            ConversationActorMembershipRow(
                id="membership-live-memory",
                organization_id=SCOPE.organization_id,
                team_id="T-LIVE-MEM",
                actor_id="U-LIVE-MEM",
                conversation_external_id="C-LIVE-MEM",
                status="active",
                source_kind="exact_destination",
                context_access_hash="a" * 64,
                version=1,
                observed_at=NOW,
            )
        )
    source = MemorySource(
        id="source-live-memory",
        scope=SCOPE,
        source_kind="synthetic",
        reference="fixture:live-memory",
        visibility=MemoryVisibility.CONVERSATION_LOCAL,
        namespace_id="C-LIVE-MEM",
    )
    record = MemoryRecord(
        id="memory-live-borealis",
        scope=SCOPE,
        kind=MemoryKind.NOTE,
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        created_at=NOW,
    )
    revision = MemoryRevision.from_content(
        id="revision-live-borealis",
        record_id=record.id,
        number=1,
        content="Project Borealis launch window is October.",
        source_ids=(source.id,),
        visibility=source.visibility,
        namespace_id=source.namespace_id,
        sensitivity=0.2,
        valid_from=NOW,
        recorded_at=NOW,
        actor_id="U-LIVE-MEM",
        reason="confirmed synthetic memory",
    )
    await PostgresMemoryStore(sessions).create(record, revision, (source,))
    yield sessions


@pytest.mark.asyncio
async def test_live_memory_question_requires_search_and_completes_as_grounded_inference(
    live_memory_sessions: async_sessionmaker[AsyncSession],
) -> None:
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        tool_names = {item["function"]["name"] for item in payload["tools"]}
        assert {"memory_search", "memory_open", "memory_search_within"}.issubset(tool_names)
        assert not any(item["id"].startswith("skill:") for item in user_payload["scoped_context"])
        if not user_payload["observations"]:
            assert payload["tool_choice"] == {
                "type": "function",
                "function": {"name": "memory_search"},
            }
            return httpx.Response(
                200,
                json={
                    "id": "generation-memory-search",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-memory-search",
                                        "type": "function",
                                        "function": {
                                            "name": "memory_search",
                                            "arguments": '{"query":"Project Borealis"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        observation = next(
            item for item in user_payload["observations"] if item["kind"] == "memory.search"
        )
        assert observation["data"]["items"][0]["content"] == (
            "Project Borealis launch window is October."
        )
        completion_schema = payload["response_format"]["json_schema"]["schema"]
        assert completion_schema["properties"]["source_claims"]["maxItems"] == 0
        assert completion_schema["properties"]["inferences"]["minItems"] == 1
        statement = "The launch window for Project Borealis is October."
        answer = "I remember that **October** is the Project Borealis launch window."
        if model_calls == 2:
            return httpx.Response(
                200,
                json={
                    "id": "generation-memory-uncited-answer",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "answer": answer,
                                        "source_claims": [],
                                        "inferences": [
                                            {
                                                "statement": statement,
                                                "observation_ids": [],
                                            }
                                        ],
                                    }
                                ),
                                "tool_calls": [],
                            }
                        }
                    ],
                },
            )
        assert any(
            "must cite the executed orchestration result" in item
            for item in user_payload["verifier_feedback"]
        )
        return httpx.Response(
            200,
            json={
                "id": "generation-memory-answer",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [],
                                    "inferences": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation["id"]],
                                        }
                                    ],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        leo_model="test/model",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        finnhub_api_key=None,
        leo_max_model_turns=4,
    )
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T-LIVE-MEM:C-LIVE-MEM:100.1",
        external_event_id="event-live-memory",
        external_channel_id="C-LIVE-MEM",
    )
    authority = MemoryNavigationAuthority(
        scope=SCOPE,
        team_id="T-LIVE-MEM",
        destination_id="C-LIVE-MEM",
        destination_kind=ConversationKind.CHANNEL,
        actor_id="U-LIVE-MEM",
        task_id="task-live-memory",
        run_id="run-live-memory",
        allowed_conversation_ids=("C-LIVE-MEM",),
        access_hash="a" * 64,
        membership_hash=membership_snapshot_hash(("C-LIVE-MEM",)),
        current_thread_namespace_id=origin.external_thread_id,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="What do you remember about Project Borealis?",
            trusted_scope=TrustedScope(
                namespace=SCOPE,
                actor_id="U-LIVE-MEM",
                roles=frozenset({"researcher"}),
            ),
            origin=origin,
            sessions=live_memory_sessions,
            launch_ids=("thread-live-memory", "task-live-memory", "run-live-memory"),
            memory_navigation_authority=authority,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == (
        "I remember that **October** is the Project Borealis launch window."
    )
    assert model_calls == 3
    assert tuple(item.kind for item in result.observations) == ("memory.search",)
    assert len(result.claims) == 1


@pytest.mark.asyncio
async def test_cross_channel_memory_no_match_returns_only_canonical_scoped_negative(
    live_memory_sessions: async_sessionmaker[AsyncSession],
) -> None:
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T-LIVE-MEM:C-EMPTY-MEM:200.1",
        external_event_id="event-empty-memory",
        external_channel_id="C-EMPTY-MEM",
    )
    thread = Thread(id="thread-empty-memory", scope=SCOPE, origin=origin)
    task = Task(
        id="task-empty-memory",
        thread_id=thread.id,
        scope=SCOPE,
        objective="What do you remember about Project Borealis?",
    )
    run = Run(id="run-empty-memory", task_id=task.id, scope=SCOPE)
    await PostgresRunStore(
        live_memory_sessions,
        FixedClock(),
        SequentialIdGenerator(),
    ).seed(thread, task, run)
    async with live_memory_sessions() as session, session.begin():
        session.add(
            ConversationRow(
                id="conversation-empty-memory",
                provider="slack",
                team_id="T-LIVE-MEM",
                external_id="C-EMPTY-MEM",
                kind="channel",
                actor_id=None,
                authority_source="slack_conversations_info",
                bot_presence="present",
                lifecycle="active",
                external_provenance="internal",
                membership_policy_version=1,
                version=1,
            )
        )
        session.add(
            ConversationActorMembershipRow(
                id="membership-empty-memory",
                organization_id=SCOPE.organization_id,
                team_id="T-LIVE-MEM",
                actor_id="U-LIVE-MEM",
                conversation_external_id="C-EMPTY-MEM",
                status="active",
                source_kind="exact_destination",
                context_access_hash="b" * 64,
                version=1,
                observed_at=NOW,
            )
        )

    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        if not user_payload["observations"]:
            assert payload["tool_choice"] == {
                "type": "function",
                "function": {"name": "memory_search"},
            }
            return httpx.Response(
                200,
                json={
                    "id": "generation-empty-memory-search",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-empty-memory-search",
                                        "type": "function",
                                        "function": {
                                            "name": "memory_search",
                                            "arguments": '{"query":"Project Borealis"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        observation = next(
            item for item in user_payload["observations"] if item["kind"] == "memory.search"
        )
        assert observation["data"]["selected_count"] == 0
        assert observation["data"]["items"] == []
        assert "launch window" not in json.dumps(observation["data"])
        return httpx.Response(
            200,
            json={
                "id": "generation-empty-memory-answer",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": _EMPTY_MEMORY_SCOPE_INFERENCE,
                                    "source_claims": [],
                                    "inferences": [
                                        {
                                            "statement": _EMPTY_MEMORY_SCOPE_INFERENCE,
                                            "observation_ids": [observation["id"]],
                                        }
                                    ],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        leo_model="test/model",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        finnhub_api_key=None,
        leo_max_model_turns=4,
    )
    authority = MemoryNavigationAuthority(
        scope=SCOPE,
        team_id="T-LIVE-MEM",
        destination_id="C-EMPTY-MEM",
        destination_kind=ConversationKind.CHANNEL,
        actor_id="U-LIVE-MEM",
        task_id=task.id,
        run_id=run.id,
        allowed_conversation_ids=("C-EMPTY-MEM",),
        access_hash="b" * 64,
        membership_hash=membership_snapshot_hash(("C-EMPTY-MEM",)),
        current_thread_namespace_id=origin.external_thread_id,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=task.objective,
            trusted_scope=TrustedScope(
                namespace=SCOPE,
                actor_id="U-LIVE-MEM",
                roles=frozenset({"researcher"}),
            ),
            origin=origin,
            sessions=live_memory_sessions,
            launch_ids=(thread.id, task.id, run.id),
            memory_navigation_authority=authority,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == _EMPTY_MEMORY_SCOPE_INFERENCE
    assert model_calls == 2
    assert tuple(item.data["selected_count"] for item in result.observations) == (0,)
    assert tuple(item.statement for item in result.claims) == (_EMPTY_MEMORY_SCOPE_INFERENCE,)
