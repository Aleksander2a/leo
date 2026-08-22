"""Compatibility domain defaults for Slack ingress.

Slack channel mappings are legacy metadata.  They are never consulted to decide whether
Leo is available in a conversation.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.integrations.slack.events import (
    SlackAdmissionPolicyRejected,
    SlackConversationEligibility,
    SlackConversationPolicyRejected,
    SlackScopeResolution,
)
from leo.persistence.schema import SlackChannelScopeRow


class SlackChannelScopeStatus(StrEnum):
    ACTIVE = "active"
    PENDING = "pending"
    REVOKED = "revoked"
    CONFLICT = "conflict"


class SlackScopePolicyRejected(SlackAdmissionPolicyRejected):
    """Deprecated compatibility exception; channel mapping status is no longer enforced."""

    def __init__(self, status: SlackChannelScopeStatus) -> None:
        self.status = status
        self.safe_code = f"scope_mapping_{status.value}"
        super().__init__(f"Slack channel scope mapping is {status.value}")


class SlackScopeStoreInvariantError(RuntimeError):
    """Persisted scope state is missing or malformed; callers must fail closed."""


class SlackScopeResolver(Protocol):
    async def preflight(self) -> None: ...

    async def resolve_or_provision(
        self,
        *,
        team_id: str,
        channel_id: str,
        user_id: str,
        default_scope: ScopeKey,
        eligibility: SlackConversationEligibility,
    ) -> SlackScopeResolution: ...


class PostgresSlackScopeResolver:
    """Bind configured domain defaults without consulting legacy channel mappings."""

    def __init__(self, sessions: async_sessionmaker[AsyncSession]) -> None:
        self._sessions = sessions

    async def preflight(self) -> None:
        async with self._sessions() as session:
            await session.execute(select(1))

    async def resolve_or_provision(
        self,
        *,
        team_id: str,
        channel_id: str,
        user_id: str,
        default_scope: ScopeKey,
        eligibility: SlackConversationEligibility,
    ) -> SlackScopeResolution:
        """Return the configured default; mappings cannot gate or redirect a run."""

        async with self._sessions() as session, session.begin():
            return await resolve_or_provision_in_session(
                session,
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
                default_scope=default_scope,
                eligibility=eligibility,
            )


async def resolve_or_provision_in_session(
    session: AsyncSession,
    *,
    team_id: str,
    channel_id: str,
    user_id: str,
    default_scope: ScopeKey,
    eligibility: SlackConversationEligibility,
) -> SlackScopeResolution:
    """Validate Slack authority and bind non-gating compatibility metadata."""

    del session
    if not eligibility.admissible:
        raise SlackConversationPolicyRejected(eligibility)
    _validate_slack_id(team_id, "team_id")
    _validate_slack_id(channel_id, "channel_id")
    _validate_slack_id(user_id, "user_id")
    return SlackScopeResolution(
        scope=default_scope,
        mapping_version=1,
        provisioned=False,
    )


def resolution_from_row(
    row: SlackChannelScopeRow,
    *,
    provisioned: bool,
) -> SlackScopeResolution:
    """Read legacy metadata without treating its status as an availability policy."""

    try:
        return SlackScopeResolution(
            scope=ScopeKey(
                organization_id=row.organization_id,
                strategy_id=row.strategy_id,
            ),
            mapping_version=row.version,
            provisioned=provisioned,
        )
    except (ValidationError, ValueError) as exc:
        raise SlackScopeStoreInvariantError("Slack channel mapping is malformed") from exc


def _validate_slack_id(value: str, name: str) -> None:
    if not value or value != value.strip() or len(value) > 32:
        raise ValueError(f"{name} must be a non-empty Slack ID of at most 32 characters")
