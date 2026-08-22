from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    SlackBotPresence,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackMentionJob,
    SlackTriggerKind,
    build_context_access_hash,
)
from leo.persistence.schema import (
    ConversationRow,
    SlackChannelScopeRow,
    SlackIngressEventRow,
)
from leo.persistence.slack_ingress import (
    PostgresSlackIngressAdmission,
    _recoverable_linked_launch_predicate,
)
from leo.persistence.slack_scope import (
    PostgresSlackScopeResolver,
    SlackChannelScopeStatus,
    SlackScopeStoreInvariantError,
    resolution_from_row,
)


class _Result:
    def __init__(self, value: object | None = None) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object | None:
        return self.value


class _Transaction:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def __aenter__(self) -> _Transaction:
        self.session.transaction_entered += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc, traceback
        self.session.transaction_exited += 1
        self.session.rolled_back = exc_type is not None


class _Session:
    def __init__(
        self,
        *,
        execute_results: Sequence[_Result | BaseException] = (),
        scalar_results: Sequence[object | None] = (),
    ) -> None:
        self.execute_results = list(execute_results)
        self.scalar_results = list(scalar_results)
        self.statements: list[Any] = []
        self.transaction_entered = 0
        self.transaction_exited = 0
        self.rolled_back = False

    async def __aenter__(self) -> _Session:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        del exc_type, exc, traceback

    def begin(self) -> _Transaction:
        return _Transaction(self)

    async def execute(self, statement: object) -> _Result:
        self.statements.append(statement)
        outcome = self.execute_results.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def scalar(self, statement: object) -> object | None:
        self.statements.append(statement)
        return self.scalar_results.pop(0)


class _Sessions:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def __call__(self) -> _Session:
        return self.session


def _mapping(
    *,
    status: str = "active",
    organization_id: str = "org-original",
    strategy_id: str = "strategy-original",
    version: int = 3,
) -> SlackChannelScopeRow:
    return SlackChannelScopeRow(
        team_id="T1",
        channel_id="C1",
        organization_id=organization_id,
        strategy_id=strategy_id,
        status=status,
        provisioned_by_user_id="U-first",
        provisioned_via="first_valid_mention",
        version=version,
    )


def _job(event_id: str = "Ev1") -> SlackMentionJob:
    return SlackMentionJob(
        event_id=event_id,
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="1.2",
        thread_root_ts="1.0",
        conversation_key="slack:T1:C1:1.0",
        prompt="quote NVDA",
        conversation_kind=SlackConversationKind.ORDINARY_INTERNAL,
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=("C1",),
        conversation_authority_source="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
        context_access_hash=build_context_access_hash(
            team_id="T1",
            user_id="U1",
            channel_id="C1",
            context_conversation_ids=("C1",),
        ),
    )


def _eligibility() -> SlackConversationEligibility:
    return SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.INTERNAL,
    )


def _ingress(job: SlackMentionJob | None = None) -> SlackIngressEventRow:
    selected = job or _job()
    return SlackIngressEventRow(
        event_id=selected.event_id,
        team_id=selected.team_id,
        channel_id=selected.channel_id,
        user_id=selected.user_id,
        message_ts=selected.message_ts,
        thread_root_ts=selected.thread_root_ts,
        conversation_key=selected.conversation_key,
        prompt=selected.prompt,
        conversation_kind=selected.conversation_kind.value,
        trigger_kind=selected.trigger_kind.value,
        context_conversation_ids=list(selected.context_conversation_ids),
        context_access_hash=selected.context_access_hash,
        context_projection_source=selected.context_projection_source.value,
        conversation_authority_source=selected.conversation_authority_source,
        bot_presence=selected.bot_presence.value,
        conversation_lifecycle=selected.conversation_lifecycle.value,
        external_provenance=selected.external_provenance.value,
        membership_policy_version=selected.membership_policy_version,
        conversation_id="conversation-1",
        status="policy_rejected",
        attempt_count=1,
    )


def _conversation() -> ConversationRow:
    return ConversationRow(
        id="conversation-1",
        provider="slack",
        team_id="T1",
        external_id="C1",
        kind="channel",
        actor_id=None,
        authority_source="slack_conversations_info",
        bot_presence="present",
        lifecycle="active",
        external_provenance="internal",
        membership_policy_version=1,
        version=1,
    )


def _params(statement: object) -> dict[str, object]:
    compiled = statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call,attr-defined]
    return dict(compiled.params)


def test_startup_linked_recovery_uses_task_authority_not_ingress_diagnostic_status() -> None:
    statement = select(SlackIngressEventRow.event_id).where(_recoverable_linked_launch_predicate(3))
    compilation = statement.compile(dialect=postgresql.dialect())  # type: ignore[no-untyped-call]
    compiled = str(compilation).lower()
    parameters = tuple(compilation.params.values())

    assert "slack_ingress_events.launch_status =" in compiled
    assert "slack_ingress_events.status =" not in compiled
    assert "slack_ingress_events.task_id is not null" in compiled
    assert "tasks.status in" in compiled
    assert "tasks.lease_expires_at is null" in compiled
    assert "tasks.lease_expires_at <= now()" in compiled
    assert "tasks.attempt_count <" in compiled
    assert "tasks.retry_after is null" in compiled
    assert "tasks.retry_after <= now()" in compiled
    assert "tasks.attempt_count >=" in compiled
    assert "queued" in parameters
    assert ["queued", "active"] in parameters
    assert parameters.count(3) == 2


@pytest.mark.asyncio
async def test_configured_default_is_used_without_reading_legacy_mapping() -> None:
    session = _Session()
    resolver = PostgresSlackScopeResolver(_Sessions(session))  # type: ignore[arg-type]

    resolution = await resolver.resolve_or_provision(
        team_id="T1",
        channel_id="C1",
        user_id="U-new",
        default_scope=ScopeKey(organization_id="org-new", strategy_id="strategy-new"),
        eligibility=_eligibility(),
    )

    assert resolution.scope == ScopeKey(organization_id="org-new", strategy_id="strategy-new")
    assert resolution.mapping_version == 1
    assert resolution.provisioned is False
    assert session.transaction_exited == 1
    assert session.rolled_back is False
    assert session.statements == []


@pytest.mark.asyncio
async def test_resolution_does_not_provision_a_channel_mapping() -> None:
    session = _Session()
    resolver = PostgresSlackScopeResolver(_Sessions(session))  # type: ignore[arg-type]

    resolution = await resolver.resolve_or_provision(
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        default_scope=ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
        eligibility=_eligibility(),
    )

    assert resolution.scope == ScopeKey(
        organization_id="org-default", strategy_id="strategy-default"
    )
    assert resolution.provisioned is False
    assert session.statements == []


@pytest.mark.parametrize(
    "status",
    [
        SlackChannelScopeStatus.PENDING,
        SlackChannelScopeStatus.REVOKED,
        SlackChannelScopeStatus.CONFLICT,
    ],
)
def test_non_active_legacy_mapping_states_do_not_gate_availability(
    status: SlackChannelScopeStatus,
) -> None:
    resolution = resolution_from_row(_mapping(status=status.value), provisioned=False)

    assert resolution.scope == ScopeKey(
        organization_id="org-original", strategy_id="strategy-original"
    )
    assert resolution.mapping_version == 3


def test_unknown_legacy_mapping_state_is_also_non_gating() -> None:
    resolution = resolution_from_row(_mapping(status="unexpected"), provisioned=False)

    assert resolution.mapping_version == 3


@pytest.mark.asyncio
async def test_ingress_admission_snapshots_scope_in_same_transaction() -> None:
    session = _Session(
        execute_results=[
            _Result(),
            _Result(),
            _Result("Ev1"),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
        ],
        scalar_results=[None, _conversation()],
    )
    admission = PostgresSlackIngressAdmission(_Sessions(session))  # type: ignore[arg-type]

    admitted = await admission.admit(
        _job(),
        ScopeKey(organization_id="org-original", strategy_id="strategy-original"),
        eligibility=_eligibility(),
    )

    assert admitted is not None
    assert admitted.resolution.scope.organization_id == "org-original"
    assert admitted.resolution.mapping_version == 1
    assert session.transaction_entered == 1
    assert session.transaction_exited == 1
    assert session.rolled_back is False
    assert "pg_advisory_xact_lock" in str(session.statements[0])
    assert _params(session.statements[4])["status"] == "admitting"
    reservation = _params(session.statements[4])
    assert reservation["conversation_kind"] == "ordinary_internal"
    assert reservation["trigger_kind"] == "app_mention"
    assert reservation["context_conversation_ids"] == ["C1"]
    assert reservation["context_access_hash"] == _job().context_access_hash
    assert reservation["conversation_id"] == "conversation-1"
    access_snapshot = _params(session.statements[5])
    assert access_snapshot["organization_id_m0"] == "org-original"
    assert access_snapshot["conversation_external_id_m0"] == "C1"
    assert access_snapshot["context_access_hash_m0"] == _job().context_access_hash
    membership_snapshot = _params(session.statements[6])
    assert membership_snapshot["status_m0"] == "active"
    assert membership_snapshot["conversation_external_id_m0"] == "C1"
    assert "memory_retrieval_cache" in str(session.statements[7])
    assert "memory_capability_handles" in str(session.statements[8])
    message = _params(session.statements[9])
    assert message["role"] == "user"
    assert message["conversation_id"] == "conversation-1"
    snapshot = _params(session.statements[10])
    assert snapshot["organization_id"] == "org-original"
    assert snapshot["strategy_id"] == "strategy-original"
    assert snapshot["mapping_version"] == 1
    assert snapshot["status"] == "received"


@pytest.mark.asyncio
async def test_legacy_revoked_mapping_cannot_reject_ingress() -> None:
    session = _Session(
        execute_results=[
            _Result(),
            _Result(),
            _Result("Ev1"),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
            _Result(),
        ],
        scalar_results=[None, _conversation()],
    )
    admission = PostgresSlackIngressAdmission(_Sessions(session))  # type: ignore[arg-type]

    admitted = await admission.admit(
        _job(),
        ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
        eligibility=_eligibility(),
    )

    assert admitted is not None
    assert admitted.resolution.scope == ScopeKey(
        organization_id="org-default", strategy_id="strategy-default"
    )
    assert session.transaction_exited == 1
    assert session.rolled_back is False


@pytest.mark.asyncio
async def test_repeat_event_with_same_access_projection_is_duplicate() -> None:

    duplicate_session = _Session(
        execute_results=[_Result()],
        scalar_results=[_ingress()],
    )
    duplicate_admission = PostgresSlackIngressAdmission(
        _Sessions(duplicate_session)  # type: ignore[arg-type]
    )
    duplicate = await duplicate_admission.admit(
        _job(),
        ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
        eligibility=_eligibility(),
    )
    assert duplicate is None
    assert len(duplicate_session.statements) == 2
    assert "pg_advisory_xact_lock" in str(duplicate_session.statements[0])
    assert all(
        "INSERT INTO conversations" not in str(item) for item in duplicate_session.statements
    )


@pytest.mark.asyncio
async def test_duplicate_event_id_with_changed_envelope_fails_closed() -> None:
    conflicting = _job()
    conflicting = conflicting.model_copy(update={"prompt": "different command"})
    session = _Session(
        execute_results=[_Result()],
        scalar_results=[_ingress()],
    )
    admission = PostgresSlackIngressAdmission(_Sessions(session))  # type: ignore[arg-type]

    with pytest.raises(SlackScopeStoreInvariantError, match="different envelope"):
        await admission.admit(
            conflicting,
            ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
            eligibility=_eligibility(),
        )

    assert session.rolled_back is True


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["attach_task", "mark_failed"])
async def test_terminal_ingress_updates_fail_when_expected_state_does_not_match(
    method: str,
) -> None:
    session = _Session(execute_results=[_Result(None)])
    admission = PostgresSlackIngressAdmission(_Sessions(session))  # type: ignore[arg-type]

    with pytest.raises(SlackScopeStoreInvariantError, match="did not match"):
        if method == "attach_task":
            await admission.attach_task("Ev1", "task-1", "run_completed")
        else:
            await admission.mark_failed("Ev1", "runtime_error")


@pytest.mark.asyncio
async def test_operational_failure_rolls_back_event_transaction() -> None:
    session = _Session(
        execute_results=[
            _Result(),
            _Result(),
            _Result("Ev1"),
            RuntimeError("database disconnected"),
        ],
        scalar_results=[None, _conversation()],
    )
    admission = PostgresSlackIngressAdmission(_Sessions(session))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="database disconnected"):
        await admission.admit(
            _job(),
            ScopeKey(organization_id="org-default", strategy_id="strategy-default"),
            eligibility=_eligibility(),
        )

    assert session.transaction_exited == 1
    assert session.rolled_back is True
