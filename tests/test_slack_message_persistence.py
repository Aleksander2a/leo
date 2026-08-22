from __future__ import annotations

from collections.abc import Mapping

import pytest
from sqlalchemy.dialects import postgresql

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    SlackConversationKind,
    SlackPassiveMessage,
    SlackPassiveMessageRole,
)
from leo.persistence.schema import ConversationRow, SanitizedMessageRow
from leo.persistence.slack_messages import (
    PostgresSlackMessagePlane,
    SlackThreadCoverageReason,
    SlackThreadCoverageSource,
    _assess_thread_coverage,
    _validate_authoritative_root_snapshot,
)

_SNAPSHOT_HASH = "a" * 64


def _row(
    message_id: str,
    ts: str,
    text: str,
    *,
    actor_id: str = "U1",
    role: str = "user",
    event_id: str | None = None,
) -> SanitizedMessageRow:
    return SanitizedMessageRow(
        id=message_id,
        organization_id="org-demo",
        strategy_id="strategy-default",
        destination_id="C1",
        external_event_id=event_id or f"Ev-{message_id}",
        text=text,
        content_hash="b" * 64,
        conversation_id="conversation-1",
        actor_id=actor_id,
        role=role,
        provider_message_ts=ts,
        provider_thread_root_ts="100.000",
    )


def _assess(
    rows: tuple[SanitizedMessageRow, ...],
    *,
    reply_count: int = 2,
    latest_reply_ts: str | None = "120.000",
    current_message_ts: str = "120.000",
    current_actor_id: str = "U1",
    current_event_id: str = "Ev-current",
):  # type: ignore[no-untyped-def]
    return _assess_thread_coverage(
        team_id="T1",
        channel_id="C1",
        thread_root_ts="100.000",
        current_message_ts=current_message_ts,
        conversation_id="conversation-1",
        rows=rows,
        authoritative_reply_count=reply_count,
        authoritative_latest_reply_ts=latest_reply_ts,
        coverage_source=SlackThreadCoverageSource.BOT_HISTORY,
        coverage_snapshot_hash=_SNAPSHOT_HASH,
        max_messages=500,
        current_actor_id=current_actor_id,
        current_event_id=current_event_id,
    )


def test_exact_root_snapshot_proves_ordered_context_before_current_only() -> None:
    snapshot = _assess(
        (
            _row("previous", "110.000", "previous reply"),
            _row("root", "100.000", "root text"),
            _row("current", "120.000", "current message"),
        )
    )

    assert snapshot.complete is True
    assert snapshot.coverage_reason is SlackThreadCoverageReason.COMPLETE
    assert [message.message_ts for message in snapshot.messages] == ["100.000", "110.000"]
    assert all(message.text != "current message" for message in snapshot.messages)
    assert snapshot.coverage_source is SlackThreadCoverageSource.BOT_HISTORY
    assert snapshot.coverage_snapshot_hash == _SNAPSHOT_HASH
    assert len(snapshot.coverage_digest) == 64


@pytest.mark.parametrize(
    ("rows", "reply_count", "latest_reply_ts", "reason"),
    [
        (
            (_row("current", "120.000", "current"),),
            1,
            "120.000",
            SlackThreadCoverageReason.ROOT_MISSING,
        ),
        (
            (
                _row("root", "100.000", "root"),
                _row("current", "120.000", "current"),
            ),
            2,
            "120.000",
            SlackThreadCoverageReason.COUNT_MISMATCH,
        ),
        (
            (
                _row("root", "100.000", "root"),
                _row("duplicate", "100.000", "duplicate"),
                _row("current", "120.000", "current"),
            ),
            2,
            "120.000",
            SlackThreadCoverageReason.DUPLICATE_PROVIDER_TIMESTAMP,
        ),
    ],
)
def test_missing_forged_or_duplicate_persisted_rows_fail_closed(
    rows: tuple[SanitizedMessageRow, ...],
    reply_count: int,
    latest_reply_ts: str | None,
    reason: SlackThreadCoverageReason,
) -> None:
    snapshot = _assess(
        rows,
        reply_count=reply_count,
        latest_reply_ts=latest_reply_ts,
    )

    assert snapshot.complete is False
    assert snapshot.coverage_reason is reason


def test_lagged_or_raced_root_metadata_cannot_attest_current_boundary() -> None:
    rows = (
        _row("root", "100.000", "root"),
        _row("old", "110.000", "old reply"),
        _row("current", "120.000", "current"),
    )

    lagged = _assess(rows, reply_count=1, latest_reply_ts="110.000")
    raced = _assess(rows, reply_count=3, latest_reply_ts="130.000")

    assert lagged.coverage_reason is SlackThreadCoverageReason.BOUNDARY_NOT_ATTESTED
    assert raced.coverage_reason is SlackThreadCoverageReason.METADATA_AFTER_BOUNDARY
    assert lagged.complete is raced.complete is False


def test_persisted_future_row_prevents_older_boundary_fallback() -> None:
    snapshot = _assess(
        (
            _row("root", "100.000", "root"),
            _row("current", "120.000", "current"),
            _row("future", "130.000", "later reply"),
        ),
        reply_count=1,
        latest_reply_ts="120.000",
    )

    assert snapshot.complete is False
    assert snapshot.coverage_reason is SlackThreadCoverageReason.METADATA_AFTER_BOUNDARY


def test_exact_whole_snapshot_accepts_later_rows_but_exposes_only_boundary_prefix() -> None:
    snapshot = _assess(
        (
            _row("root", "100.000", "root"),
            _row("prior", "110.000", "prior reply"),
            _row("current", "120.000", "current ingress"),
            _row("later", "130.000", "progress posted before history load"),
        ),
        reply_count=3,
        latest_reply_ts="130.000",
    )

    assert snapshot.complete is True
    assert snapshot.coverage_reason is SlackThreadCoverageReason.COMPLETE
    assert snapshot.authoritative_latest_reply_ts == "130.000"
    assert snapshot.complete_through_ts == "120.000"
    assert snapshot.persisted_message_count == 4
    assert snapshot.boundary_attested is True
    assert [message.message_ts for message in snapshot.messages] == ["100.000", "110.000"]
    assert all(message.message_ts < "120.000" for message in snapshot.messages)


def test_later_snapshot_still_rejects_missing_current_and_unknown_gap() -> None:
    missing_current = _assess(
        (
            _row("root", "100.000", "root"),
            _row("prior", "110.000", "prior reply"),
            _row("later", "130.000", "later progress"),
        ),
        reply_count=2,
        latest_reply_ts="130.000",
    )
    missing_gap = _assess(
        (
            _row("root", "100.000", "root"),
            _row("prior", "110.000", "prior reply"),
            _row("current", "120.000", "current ingress"),
            _row("later", "130.000", "later progress"),
        ),
        reply_count=4,
        latest_reply_ts="130.000",
    )

    assert missing_current.coverage_reason is SlackThreadCoverageReason.BOUNDARY_NOT_ATTESTED
    assert missing_gap.coverage_reason is SlackThreadCoverageReason.COUNT_MISMATCH
    assert missing_current.complete is missing_gap.complete is False
    assert missing_current.messages == missing_gap.messages == ()


@pytest.mark.parametrize(
    "boundary_row",
    [
        _row("current", "120.000", "assistant collision", role="assistant"),
        _row("current", "120.000", "foreign actor", actor_id="U-other"),
        _row("current", "120.000", "foreign event", event_id="Ev-other"),
    ],
)
def test_boundary_requires_exact_user_actor_and_ingress_event(
    boundary_row: SanitizedMessageRow,
) -> None:
    snapshot = _assess(
        (
            _row("root", "100.000", "root"),
            _row("prior", "110.000", "prior"),
            boundary_row,
            _row("later", "130.000", "later progress"),
        ),
        reply_count=3,
        latest_reply_ts="130.000",
    )

    assert snapshot.complete is False
    assert snapshot.coverage_reason is SlackThreadCoverageReason.BOUNDARY_NOT_ATTESTED
    assert snapshot.boundary_attested is False
    assert snapshot.messages == ()


def test_authority_validator_accepts_only_exact_history_root_metadata() -> None:
    valid = _validate_authoritative_root_snapshot(
        team_id="T1",
        channel_id="C1",
        thread_root_ts="100.000",
        current_message_ts="120.000",
        raw_root={"ts": "100.000", "reply_count": 2, "latest_reply": "120.000"},
        source=SlackThreadCoverageSource.USER_HISTORY,
    )

    assert valid is not None
    assert valid[:2] == (2, "120.000")
    assert len(valid[2]) == 64


@pytest.mark.parametrize(
    "raw_root",
    [
        {"ts": "other", "reply_count": 1, "latest_reply": "120.000"},
        {"ts": "100.000", "thread_ts": "other", "reply_count": 1},
        {"ts": "100.000"},
        {"ts": "100.000", "reply_count": True},
        {"ts": "100.000", "reply_count": -1},
        {"ts": "100.000", "reply_count": 0, "latest_reply": "120.000"},
        {"ts": "100.000", "reply_count": 1, "latest_reply": None},
        {"ts": "100.000", "reply_count": 1, "latest_reply": "99.000"},
    ],
)
def test_malformed_or_non_root_history_metadata_never_creates_authority(
    raw_root: Mapping[str, object],
) -> None:
    assert (
        _validate_authoritative_root_snapshot(
            team_id="T1",
            channel_id="C1",
            thread_root_ts="100.000",
            current_message_ts="120.000",
            raw_root=raw_root,
            source=SlackThreadCoverageSource.BOT_HISTORY,
        )
        is None
    )


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        del args


class _Session:
    def __init__(self, scalar_results: list[object | None]) -> None:
        self.scalar_results = scalar_results
        self.statements: list[object] = []

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(self, *args: object) -> None:
        del args

    def begin(self) -> _Transaction:
        return _Transaction()

    async def scalar(self, statement: object) -> object | None:
        self.statements.append(statement)
        return self.scalar_results.pop(0)

    async def execute(self, statement: object) -> None:
        self.statements.append(statement)


class _Sessions:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.calls = 0

    def __call__(self) -> _Session:
        self.calls += 1
        return self.session


@pytest.mark.asyncio
async def test_passive_event_persists_sanitized_thread_identity_but_no_coverage() -> None:
    conversation = ConversationRow(id="conversation-1", kind="channel")
    session = _Session([conversation, "thread-1"])
    sessions = _Sessions(session)
    repository = PostgresSlackMessagePlane(sessions)  # type: ignore[arg-type]

    await repository.record_passive_message(
        SlackPassiveMessage(
            event_id="Ev-passive",
            team_id="T1",
            channel_id="C1",
            actor_id="U1",
            role=SlackPassiveMessageRole.USER,
            message_ts="120.000",
            thread_root_ts="100.000",
            text="sanitized passive context",
            conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        ),
        ScopeKey(organization_id="org-demo", strategy_id="strategy-default"),
    )

    compiled_statements = [
        statement.compile(dialect=postgresql.dialect())  # type: ignore[union-attr]
        for statement in session.statements
    ]
    message_insert = compiled_statements[-1]
    assert message_insert.params["provider_message_ts"] == "120.000"
    assert message_insert.params["provider_thread_root_ts"] == "100.000"
    assert message_insert.params["context_access_hash"] is None
    assert message_insert.params["role"] == "user"
    assert all("slack_thread_coverage" not in str(statement) for statement in compiled_statements)


@pytest.mark.asyncio
async def test_repository_records_history_authority_without_strategy_gating() -> None:
    conversation = ConversationRow(id="conversation-1")
    root = _row("root", "100.000", "root")
    session = _Session([conversation, root])
    sessions = _Sessions(session)
    repository = PostgresSlackMessagePlane(sessions)  # type: ignore[arg-type]

    recorded = await repository.record_root_coverage(
        team_id="T1",
        channel_id="C1",
        thread_root_ts="100.000",
        current_message_ts="120.000",
        raw_root={"ts": "100.000", "reply_count": 2, "latest_reply": "120.000"},
        source=SlackThreadCoverageSource.USER_HISTORY,
    )

    assert recorded is True
    insert = session.statements[-1]
    compiled = insert.compile(dialect=postgresql.dialect())  # type: ignore[union-attr]
    assert "ON CONFLICT" in str(compiled)
    assert compiled.params["authority_source"] == "slack_conversations_history_user"
    assert compiled.params["authoritative_reply_count"] == 2
    assert "organization_id" not in compiled.params
    assert "strategy_id" not in compiled.params


@pytest.mark.asyncio
async def test_invalid_history_root_fails_before_opening_a_database_session() -> None:
    sessions = _Sessions(_Session([]))
    repository = PostgresSlackMessagePlane(sessions)  # type: ignore[arg-type]

    recorded = await repository.record_root_coverage(
        team_id="T1",
        channel_id="C1",
        thread_root_ts="100.000",
        current_message_ts="120.000",
        raw_root={"ts": "forged", "reply_count": 0},
        source=SlackThreadCoverageSource.BOT_HISTORY,
    )

    assert recorded is False
    assert sessions.calls == 0
