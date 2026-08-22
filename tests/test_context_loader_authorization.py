from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from leo.harness.models import ContextItemRetention, ScopeKey
from leo.integrations.slack.events import build_context_access_hash
from leo.persistence.context_loader import (
    ConversationContextAuthorizationError,
    ConversationContextOverflowError,
    ConversationContextRequest,
    PostgresConversationContextLoader,
)

SCOPE = ScopeKey(organization_id="org-context", strategy_id="strategy-context")


class _Result:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[object, ...]]:
        return self._rows


class _Session:
    def __init__(self, rows: list[tuple[object, ...]]) -> None:
        self.rows = rows
        self.statements: list[object] = []

    async def execute(self, statement: object) -> _Result:
        self.statements.append(statement)
        return _Result(self.rows)

    async def scalar(self, statement: object) -> object | None:
        self.statements.append(statement)
        if not self.rows:
            return None
        ingress = self.rows[0][0]
        return SimpleNamespace(
            id=ingress.conversation_id,
            provider="slack",
            team_id=ingress.team_id,
            external_id=ingress.channel_id,
            kind={
                "ordinary_internal": "channel",
                "dm": "dm",
                "mpim": "group_dm",
                "shared": "shared",
                "external": "external",
            }[ingress.conversation_kind],
            bot_presence="present",
            lifecycle="active",
            external_provenance=ingress.external_provenance,
            membership_policy_version=ingress.membership_policy_version,
        )

    async def scalars(self, statement: object) -> list[object]:
        self.statements.append(statement)
        if not self.rows:
            return []
        ingress = self.rows[0][0]
        if "FROM conversations" in str(statement):
            return [
                SimpleNamespace(
                    id=(
                        ingress.conversation_id
                        if conversation_id == ingress.channel_id
                        else f"conversation-{conversation_id}"
                    ),
                    external_id=conversation_id,
                    kind=(
                        {
                            "ordinary_internal": "channel",
                            "dm": "dm",
                            "mpim": "group_dm",
                            "shared": "shared",
                            "external": "external",
                        }[ingress.conversation_kind]
                        if conversation_id == ingress.channel_id
                        else "channel"
                    ),
                    bot_presence="present",
                    lifecycle="active",
                    external_provenance=(
                        ingress.external_provenance
                        if conversation_id == ingress.channel_id
                        else "internal"
                    ),
                    membership_policy_version=ingress.membership_policy_version,
                )
                for conversation_id in ingress.context_conversation_ids
            ]
        return [
            SimpleNamespace(
                conversation_external_id=conversation_id,
                status="active",
            )
            for conversation_id in ingress.context_conversation_ids
        ]


class _SessionContext:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Session:
        return self.session

    async def __aexit__(self, *args: object) -> None:
        del args


class _Sessions:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _SessionContext:
        return _SessionContext(self.session)


class _RecordingLoader(PostgresConversationContextLoader):
    def __init__(self, session: _Session) -> None:
        super().__init__(_Sessions(session))  # type: ignore[arg-type]
        self.content_queries: list[str] = []

    async def _load_turns(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.content_queries.append("turns")
        return ()

    async def _load_memories(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.content_queries.append("memories")
        return ()

    async def _load_summary(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.content_queries.append("summary")
        return ()

    async def _load_recent_thread_messages(self, *args: Any, **kwargs: Any) -> tuple[()]:
        del args, kwargs
        self.content_queries.append("recent")
        return ()


def _request(
    *,
    destination_id: str = "C1",
    destination_kind: str = "channel",
    projection: tuple[str, ...] = ("C1",),
) -> ConversationContextRequest:
    access_hash = build_context_access_hash(
        team_id="T1",
        user_id="U1",
        channel_id=destination_id,
        context_conversation_ids=projection,
    )
    return ConversationContextRequest(
        team_id="T1",
        destination_id=destination_id,
        destination_kind=destination_kind,  # type: ignore[arg-type]
        actor_id="U1",
        objective="What did we decide?",
        current_task_id="task-current",
        current_event_id="event-current",
        current_message_ts="1.1",
        thread_root_ts="1.0",
        allowed_conversation_ids=projection,
        access_hash=access_hash,
        current_thread_namespace_id=f"slack:T1:{destination_id}:1.0",
    )


def _authorized_rows(
    request: ConversationContextRequest,
) -> list[tuple[object, ...]]:
    conversation_kind = {
        "channel": "ordinary_internal",
        "dm": "dm",
        "group_dm": "mpim",
        "shared": "shared",
        "external": "external",
    }[request.destination_kind]
    source_kind = (
        "dm_membership_intersection"
        if request.destination_kind == "dm" and len(request.allowed_conversation_ids) > 1
        else "exact_destination"
    )
    ingress = SimpleNamespace(
        event_id="event-current",
        task_id=request.current_task_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        team_id=request.team_id,
        user_id=request.actor_id,
        channel_id=request.destination_id,
        message_ts=request.current_message_ts,
        thread_root_ts=request.thread_root_ts,
        conversation_key=request.current_thread_namespace_id,
        conversation_kind=conversation_kind,
        context_conversation_ids=list(request.allowed_conversation_ids),
        context_access_hash=request.access_hash,
        context_projection_source=source_kind,
        conversation_id="conversation-current",
        bot_presence="present",
        conversation_lifecycle="active",
        external_provenance=("external" if request.destination_kind == "external" else "internal"),
        membership_policy_version=1,
    )
    task = SimpleNamespace(
        id=request.current_task_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        thread_id="thread-current",
    )
    thread = SimpleNamespace(
        id=task.thread_id,
        organization_id=SCOPE.organization_id,
        strategy_id=SCOPE.strategy_id,
        origin_provider="slack",
        external_thread_id=request.current_thread_namespace_id,
        external_channel_id=request.destination_id,
        conversation_id=ingress.conversation_id,
    )
    return [
        (
            ingress,
            task,
            SimpleNamespace(
                id=f"snapshot-{position}",
                ingress_event_id=ingress.event_id,
                organization_id=SCOPE.organization_id,
                team_id=request.team_id,
                actor_id=request.actor_id,
                destination_external_id=request.destination_id,
                conversation_external_id=conversation_id,
                position=position,
                source_kind=source_kind,
                context_access_hash=request.access_hash,
            ),
            thread,
        )
        for position, conversation_id in enumerate(request.allowed_conversation_ids)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context_request",
    [
        _request(),
        _request(
            destination_id="D1",
            destination_kind="dm",
            projection=("C1", "D1", "G1"),
        ),
        _request(destination_id="G1", destination_kind="group_dm", projection=("G1",)),
        _request(destination_id="S1", destination_kind="shared", projection=("S1",)),
        _request(destination_id="E1", destination_kind="external", projection=("E1",)),
    ],
    ids=("channel", "dm-union", "group-local", "shared-local", "external-local"),
)
async def test_valid_snapshot_authorizes_before_content_queries(
    context_request: ConversationContextRequest,
) -> None:
    session = _Session(_authorized_rows(context_request))
    loader = _RecordingLoader(session)

    assert await loader.load(SCOPE, context_request) == ()
    assert len(session.statements) == 3
    assert loader.content_queries == ["turns", "memories", "summary", "recent"]


@pytest.mark.asyncio
async def test_missing_snapshot_fails_before_any_content_query() -> None:
    request = _request()
    session = _Session([])
    loader = _RecordingLoader(session)

    with pytest.raises(ConversationContextAuthorizationError, match="did not authorize"):
        await loader.load(SCOPE, request)

    assert len(session.statements) == 1
    assert loader.content_queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "attribute", "forged_value"),
    [
        ("task", "id", "task-forged"),
        ("ingress", "team_id", "T-forged"),
        ("ingress", "user_id", "U-forged"),
        ("ingress", "channel_id", "C-forged"),
        ("ingress", "conversation_key", "slack:T1:C1:forged"),
        ("ingress", "context_access_hash", "f" * 64),
        ("thread", "external_thread_id", "slack:T1:C1:forged"),
        ("thread", "conversation_id", "conversation-forged"),
    ],
)
async def test_mismatched_authority_fields_fail_before_retrieval(
    target: str,
    attribute: str,
    forged_value: str,
) -> None:
    request = _request()
    rows = _authorized_rows(request)
    selected = {"ingress": rows[0][0], "task": rows[0][1], "thread": rows[0][3]}[target]
    setattr(selected, attribute, forged_value)
    loader = _RecordingLoader(_Session(rows))

    with pytest.raises(ConversationContextAuthorizationError):
        await loader.load(SCOPE, request)

    assert loader.content_queries == []


@pytest.mark.asyncio
async def test_reordered_snapshot_positions_fail_before_retrieval() -> None:
    request = _request(
        destination_id="D1",
        destination_kind="dm",
        projection=("C1", "D1", "G1"),
    )
    rows = _authorized_rows(request)
    rows[0][2].position = 1  # type: ignore[attr-defined]
    rows[1][2].position = 0  # type: ignore[attr-defined]
    loader = _RecordingLoader(_Session(rows))

    with pytest.raises(ConversationContextAuthorizationError):
        await loader.load(SCOPE, request)

    assert loader.content_queries == []


@pytest.mark.asyncio
async def test_forged_projection_cannot_expand_the_durable_snapshot() -> None:
    durable_request = _request(
        destination_id="D1",
        destination_kind="dm",
        projection=("C1", "D1"),
    )
    forged_request = _request(
        destination_id="D1",
        destination_kind="dm",
        projection=("C1", "C9", "D1"),
    )
    loader = _RecordingLoader(_Session(_authorized_rows(durable_request)))

    with pytest.raises(ConversationContextAuthorizationError):
        await loader.load(SCOPE, forged_request)

    assert loader.content_queries == []


class _TurnRowsSession:
    def __init__(self, rows: list[tuple[object, object]]) -> None:
        self.rows = rows
        self.statement: object | None = None

    async def execute(self, statement: object) -> _Result:
        self.statement = statement
        return _Result(self.rows)


def _task_turn(
    suffix: str,
    *,
    task_id: str | None = None,
    thread_id: str = "thread-current",
    organization_id: str = SCOPE.organization_id,
    conversation_id: str = "D1",
) -> tuple[object, object]:
    return (
        SimpleNamespace(
            id=task_id or f"task-{suffix}",
            thread_id=thread_id,
            organization_id=organization_id,
            strategy_id=SCOPE.strategy_id,
            objective=f"Question {suffix}",
            final_output=f"Answer {suffix}",
            created_at=datetime(2026, 8, 22, tzinfo=UTC),
        ),
        SimpleNamespace(external_id=conversation_id),
    )


@pytest.mark.asyncio
async def test_task_turns_are_bound_to_current_thread_not_dm_membership_union() -> None:
    request = _request(
        destination_id="D1",
        destination_kind="dm",
        projection=("C1", "D1", "G1"),
    )
    session = _TurnRowsSession([_task_turn("prior-current-thread")])
    loader = PostgresConversationContextLoader(_Sessions(_Session([])))  # type: ignore[arg-type]

    items = await loader._load_turns(
        session,  # type: ignore[arg-type]
        SCOPE,
        request,
        harness_thread_id="thread-current",
    )

    assert [item.id for item in items] == ["turn:task-prior-current-thread"]
    assert [item.conversation_id for item in items] == [request.destination_id]
    assert session.statement is not None
    where = str(session.statement.whereclause)  # type: ignore[attr-defined]
    assert "tasks.thread_id" in where
    assert "threads.external_thread_id" in where
    assert "threads.external_channel_id" in where
    assert "conversations.external_id" in where
    assert "conversations.external_id IN" not in where


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_turn",
    [
        _task_turn("old-thread", thread_id="thread-old"),
        _task_turn("old-channel", conversation_id="C1"),
        _task_turn("old-organization", organization_id="org-foreign"),
        _task_turn("current-task", task_id="task-current"),
    ],
    ids=("foreign-thread", "foreign-destination", "foreign-organization", "current-task"),
)
async def test_task_turn_query_fails_closed_on_exact_thread_boundary_violation(
    forged_turn: tuple[object, object],
) -> None:
    request = _request(
        destination_id="D1",
        destination_kind="dm",
        projection=("C1", "D1", "G1"),
    )
    loader = PostgresConversationContextLoader(_Sessions(_Session([])))  # type: ignore[arg-type]

    with pytest.raises(ConversationContextAuthorizationError, match="exact thread boundary"):
        await loader._load_turns(
            _TurnRowsSession([forged_turn]),  # type: ignore[arg-type]
            SCOPE,
            request,
            harness_thread_id="thread-current",
        )


class _ThreadRowsSession:
    def __init__(self, rows: list[object]) -> None:
        self.rows = rows
        self.statement: object | None = None

    async def scalars(self, statement: object) -> list[object]:
        self.statement = statement
        return self.rows


def _thread_row(
    suffix: str,
    *,
    ts: str,
    text: str,
    role: str = "user",
    event_id: str | None = None,
    conversation_id: str = "conversation-current",
    provider_thread_root_ts: str | None = None,
    offset: int = 0,
) -> object:
    return SimpleNamespace(
        id=f"message-{suffix}",
        organization_id=SCOPE.organization_id,
        strategy_id=f"historical-strategy-{suffix}",
        conversation_id=conversation_id,
        external_event_id=event_id or f"event-{suffix}",
        provider_message_ts=ts,
        provider_thread_root_ts=provider_thread_root_ts,
        recorded_at=datetime(2026, 8, 22, tzinfo=UTC) + timedelta(seconds=offset),
        role=role,
        text=text,
        actor_id="leo" if role == "assistant" else "U1",
    )


@pytest.mark.asyncio
async def test_persisted_thread_excludes_current_and_pins_semantic_turns() -> None:
    # The fake scalar result follows the query's descending order.
    rows = [
        _thread_row(
            "outcome",
            ts="1.05",
            text="The verified run completed with option beta.",
            role="assistant",
            offset=5,
        ),
        _thread_row(
            "question",
            ts="1.04",
            text="Which dependency is still unresolved?",
            offset=4,
        ),
        _thread_row(
            "correction",
            ts="1.03",
            text="Correction: choose beta instead.",
            offset=3,
        ),
        _thread_row(
            "decision",
            ts="1.02",
            text="We decided to ship alpha.",
            offset=2,
        ),
        _thread_row("root", ts="1.0", text="Root objective", offset=1),
    ]
    session = _ThreadRowsSession(rows)
    request = _request()
    loader = PostgresConversationContextLoader(_Sessions(_Session([])))  # type: ignore[arg-type]

    items = await loader._load_recent_thread_messages(
        session,  # type: ignore[arg-type]
        SCOPE,
        request,
        harness_thread_id="thread-current",
        destination_conversation_id="conversation-current",
    )

    assert [item.id for item in items] == [
        "thread-message:message-root",
        "thread-message:message-decision",
        "thread-message:message-correction",
        "thread-message:message-question",
        "thread-message:message-outcome",
    ]
    assert [item.retention for item in items] == [
        ContextItemRetention.THREAD_ROOT,
        ContextItemRetention.DECISION,
        ContextItemRetention.CORRECTION,
        ContextItemRetention.UNRESOLVED_QUESTION,
        ContextItemRetention.PRIOR_OUTCOME,
    ]
    assert session.statement is not None
    where = str(session.statement.whereclause)  # type: ignore[attr-defined]
    assert "sanitized_messages.conversation_id" in where
    assert "sanitized_messages.external_event_id" in where
    assert "sanitized_messages.provider_message_ts" in where
    assert "sanitized_messages.strategy_id" not in where


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "forged_row",
    [
        _thread_row(
            "foreign",
            ts="1.01",
            text="cross-channel secret",
            conversation_id="conversation-foreign",
        ),
        _thread_row(
            "current",
            ts="1.1",
            text="current event duplicate",
            event_id="event-current",
        ),
        _thread_row("future", ts="1.2", text="future event"),
    ],
    ids=("foreign-conversation", "current-event", "future-event"),
)
async def test_persisted_thread_fails_closed_on_query_boundary_violation(
    forged_row: object,
) -> None:
    loader = PostgresConversationContextLoader(_Sessions(_Session([])))  # type: ignore[arg-type]
    with pytest.raises(ConversationContextAuthorizationError, match="exact event boundary"):
        await loader._load_recent_thread_messages(
            _ThreadRowsSession([forged_row]),  # type: ignore[arg-type]
            SCOPE,
            _request(),
            harness_thread_id="thread-current",
            destination_conversation_id="conversation-current",
        )


@pytest.mark.asyncio
async def test_persisted_thread_row_cap_fails_closed() -> None:
    request = _request().model_copy(update={"max_thread_messages": 2})
    rows = [
        _thread_row(str(index), ts=f"1.0{index}", text=f"turn {index}", offset=index)
        for index in range(3)
    ]
    loader = PostgresConversationContextLoader(_Sessions(_Session([])))  # type: ignore[arg-type]
    with pytest.raises(ConversationContextOverflowError, match="incomplete"):
        await loader._load_recent_thread_messages(
            _ThreadRowsSession(rows),  # type: ignore[arg-type]
            SCOPE,
            request,
            harness_thread_id="thread-current",
            destination_conversation_id="conversation-current",
        )
