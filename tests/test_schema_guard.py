"""A worker must not serve Slack traffic against a schema it cannot use.

Deploy applies migrations as a separate step from starting the process, and
nothing orders the two. A build carrying a new migration can therefore come up
against the old schema and accept messages it cannot serve: every task write
fails on the missing column, each run ends in a generic terminal error, and the
user is told a source was unavailable when the deploy is simply half-applied.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from leo.persistence.database import (
    SchemaVersionError,
    build_alembic_head,
    require_schema_at_head,
)


class _FakeSessions:
    def __init__(self, applied: str | None) -> None:
        self._applied = applied

    def __call__(self) -> object:
        @asynccontextmanager
        async def _session() -> AsyncIterator[object]:
            yield self

        return _session()

    async def scalar(self, _statement: object) -> str | None:
        return self._applied


def test_the_build_declares_a_head() -> None:
    assert build_alembic_head() is not None


@pytest.mark.asyncio
async def test_a_matching_schema_starts() -> None:
    head = build_alembic_head()
    assert head is not None
    assert await require_schema_at_head(_FakeSessions(head)) == head


@pytest.mark.asyncio
async def test_a_stale_schema_refuses_to_start_and_names_both_revisions() -> None:
    with pytest.raises(SchemaVersionError) as refused:
        await require_schema_at_head(_FakeSessions("20260101_0001"))

    message = str(refused.value)
    # The logs must be enough to act on without reading the source.
    assert "20260101_0001" in message
    assert str(build_alembic_head()) in message
    assert "alembic upgrade head" in message


@pytest.mark.asyncio
async def test_an_unmigrated_database_refuses_to_start() -> None:
    with pytest.raises(SchemaVersionError):
        await require_schema_at_head(_FakeSessions(None))
