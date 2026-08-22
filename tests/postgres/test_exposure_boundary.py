from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.persistence.database import create_database_engine, create_session_factory

_LEO_TABLES = (
    "alembic_version",
    "threads",
    "tasks",
    "runs",
    "observations",
    "claims",
    "run_events",
    "slack_ingress_events",
    "slack_channel_scopes",
    "delivery_outbox",
    "organizations",
    "strategies",
    "organization_memberships",
    "assets",
    "portfolios",
    "positions",
    "mandates",
    "theses",
    "thesis_versions",
    "thesis_assumptions",
    "strategy_decisions",
    "risk_constraints",
    "memory_records",
    "memory_sources",
    "memory_revisions",
)


@pytest_asyncio.fixture
async def catalog_sessions(
    postgres_store: object,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    del postgres_store
    database_url = Settings().database_url
    if database_url is None:
        pytest.skip("DATABASE_URL is not configured")

    engine = create_database_engine(database_url.get_secret_value())
    sessions = create_session_factory(engine)
    try:
        yield sessions
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_exposure_migration_catalog_is_deny_by_default(
    catalog_sessions: async_sessionmaker[AsyncSession],
) -> None:
    table_literals = ", ".join(f"'{table}'" for table in _LEO_TABLES)
    async with catalog_sessions() as session:
        rls_rows = (
            (
                await session.execute(
                    text(
                        f"""
                    select c.relname as table_name, c.relrowsecurity as rls_enabled
                    from pg_class c
                    join pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = 'public' and c.relname in ({table_literals})
                    """
                    )
                )
            )
            .mappings()
            .all()
        )
        policies = (
            (
                await session.execute(
                    text(
                        f"""
                    select tablename, policyname, permissive, roles, qual, with_check
                    from pg_policies
                    where schemaname = 'public' and tablename in ({table_literals})
                    """
                    )
                )
            )
            .mappings()
            .all()
        )
        grants = (
            (
                await session.execute(
                    text(
                        f"""
                    select table_name, grantee,
                        has_table_privilege(grantee, format('public.%s', table_name), 'SELECT')
                            as can_select,
                        has_table_privilege(grantee, format('public.%s', table_name), 'INSERT')
                            as can_insert,
                        has_table_privilege(grantee, format('public.%s', table_name), 'UPDATE')
                            as can_update,
                        has_table_privilege(grantee, format('public.%s', table_name), 'DELETE')
                            as can_delete
                    from (values {", ".join(f"('{table}')" for table in _LEO_TABLES)})
                        as tables(table_name)
                    cross join (values ('anon'), ('authenticated')) as roles(grantee)
                    """
                    )
                )
            )
            .mappings()
            .all()
        )
        index_result = await session.execute(
            text(
                """
                select indexname, tablename
                from pg_indexes
                where schemaname = 'public'
                  and indexname in ('ix_tasks_thread_id', 'ix_slack_ingress_events_task_id')
                """
            )
        )
        indexes = index_result.mappings().all()

    assert {row["table_name"] for row in rls_rows} == set(_LEO_TABLES)
    assert all(row["rls_enabled"] for row in rls_rows)

    policy_by_table = {row["tablename"]: row for row in policies}
    assert set(policy_by_table) == set(_LEO_TABLES)
    for row in policy_by_table.values():
        assert row["policyname"] == "leo_client_deny"
        assert str(row["permissive"]).lower() in {"false", "restrictive"}
        assert set(row["roles"]) == {"anon", "authenticated"}
        assert row["qual"] == "false"
        assert row["with_check"] == "false"

    assert len(grants) == len(_LEO_TABLES) * 2
    assert all(
        not any(row[column] for column in ("can_select", "can_insert", "can_update", "can_delete"))
        for row in grants
    )
    assert {(row["indexname"], row["tablename"]) for row in indexes} == {
        ("ix_tasks_thread_id", "tasks"),
        ("ix_slack_ingress_events_task_id", "slack_ingress_events"),
    }


@pytest.mark.asyncio
async def test_catalog_queries_remain_safe_and_bounded(
    catalog_sessions: async_sessionmaker[AsyncSession],
) -> None:
    async with catalog_sessions() as session:
        row: Any = (
            (
                await session.execute(
                    text(
                        """
                    select count(*) as leo_tables
                    from pg_class c
                    join pg_namespace n on n.oid = c.relnamespace
                    where n.nspname = 'public'
                      and c.relname in ('threads', 'tasks', 'runs', 'observations', 'claims',
                                        'run_events', 'slack_ingress_events',
                                        'slack_channel_scopes', 'delivery_outbox', 'organizations',
                                        'strategies', 'organization_memberships', 'assets',
                                        'portfolios', 'positions', 'mandates', 'theses',
                                        'thesis_versions', 'thesis_assumptions',
                                        'strategy_decisions', 'risk_constraints', 'memory_records',
                                        'memory_sources', 'memory_revisions', 'alembic_version')
                    """
                    )
                )
            )
            .mappings()
            .one()
        )

    assert row["leo_tables"] == len(_LEO_TABLES)
