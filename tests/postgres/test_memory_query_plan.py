from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.harness.models import ScopeKey
from leo.memory.benchmark import BenchmarkDocument, load_frozen_retrieval_fixture
from leo.memory.models import MemoryStatus
from leo.persistence.memory_retrieval import (
    build_memory_search_statement,
    execute_memory_search,
)
from leo.persistence.schema import MemoryRecordRow, MemoryRevisionRow

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "evals/fixtures/memory-retrieval-v1"
DEMO_ORG = "pg-plan-workspace"
FOREIGN_ORG = "pg-plan-foreign"


@pytest.mark.asyncio
async def test_frozen_memory_corpus_postgres_metrics_and_query_plan(
    preserved_postgres_sessions: async_sessionmaker[AsyncSession],
) -> None:
    fixture = load_frozen_retrieval_fixture(FIXTURE_ROOT)
    documents = tuple(_database_document(item) for item in fixture.documents)
    async with preserved_postgres_sessions() as session, session.begin():
        await _assert_current_schema(session)
        await _seed_documents(session, documents)

    planning_times: list[float] = []
    execution_times: list[float] = []
    planner_costs: list[float] = []
    node_types: set[str] = set()
    relevant_selected = 0
    relevant_total = 0
    leakage_count = 0

    for query in fixture.queries:
        request = query.request().model_copy(
            update={
                "scope": ScopeKey(
                    organization_id=DEMO_ORG,
                    strategy_id=query.scope.strategy_id,
                )
            }
        )
        async with preserved_postgres_sessions() as session, session.begin():
            hits = await execute_memory_search(session, request)
            hit_ids = {item.record_id.removeprefix("pg-") for item in hits}
            relevant = set(query.relevance_grades)
            relevant_selected += len(hit_ids & relevant)
            relevant_total += len(relevant)
            leakage_count += len(hit_ids & set(query.forbidden_record_ids))
            assert relevant.issubset(hit_ids)
            assert not hit_ids.intersection(query.forbidden_record_ids)
            for record_id, expected_revision in query.expected_revisions.items():
                selected = next(item for item in hits if item.record_id == f"pg-{record_id}")
                assert selected.revision == expected_revision
            if query.expected_source_namespaces:
                assert query.expected_source_namespaces.issubset(
                    {item.namespace_id for item in hits}
                )

            plan = await _explain(session, build_memory_search_statement(request))
        planning_times.append(float(plan["Planning Time"]))
        execution_times.append(float(plan["Execution Time"]))
        root = plan["Plan"]
        planner_costs.append(float(root["Total Cost"]))
        node_types.update(_plan_node_types(root))

    recall = relevant_selected / relevant_total
    metrics = {
        "fixture_digest": fixture.manifest.fixture_digest,
        "measured_at": datetime.now(UTC).isoformat(),
        "query_count": len(fixture.queries),
        "corpus_revision_count": len(documents),
        "recall_at_k": recall,
        "leakage_count": leakage_count,
        "planning_ms": _latency_summary(planning_times),
        "execution_ms": _latency_summary(execution_times),
        "planner_total_cost": _latency_summary(planner_costs),
        "plan_node_types": sorted(node_types),
        "provider_cost_usd": None,
        "provider_cost_status": "not_applicable_database_only",
    }
    print("POSTGRES_MEMORY_PLAN=" + json.dumps(metrics, sort_keys=True))

    assert recall == 1
    assert leakage_count == 0
    assert max(execution_times) < 250
    assert max(planning_times) < 250
    assert "Limit" in node_types


async def _assert_current_schema(session: AsyncSession) -> None:
    revision = await session.scalar(text("SELECT version_num FROM public.alembic_version"))
    assert revision == "20260823_0028"
    constraints = set(
        await session.scalars(
            text(
                "SELECT conname FROM pg_constraint "
                "WHERE conrelid = 'public.memory_capability_handles'::regclass"
            )
        )
    )
    assert {
        "ck_memory_capability_handle_sources",
        "ck_memory_capability_handle_destination_source",
        "ck_memory_capability_handle_open_budget",
        "ck_memory_capability_handle_authority_hashes",
        "ck_memory_capability_handle_invalidation",
    }.issubset(constraints)
    search_vector = (
        await session.execute(
            text(
                "SELECT is_generated, generation_expression FROM information_schema.columns "
                "WHERE table_schema = 'public' AND table_name = 'memory_revisions' "
                "AND column_name = 'search_vector'"
            )
        )
    ).one()
    assert search_vector.is_generated == "ALWAYS"
    assert "to_tsvector" in search_vector.generation_expression
    index_definition = await session.scalar(
        text(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' "
            "AND indexname = 'ix_memory_revisions_search_vector'"
        )
    )
    assert index_definition is not None and "USING gin (search_vector)" in index_definition
    task_index_definition = await session.scalar(
        text(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = 'public' "
            "AND indexname = 'ix_memory_capability_handles_task'"
        )
    )
    assert task_index_definition is not None and "(task_id)" in task_index_definition


def _database_document(document: BenchmarkDocument) -> BenchmarkDocument:
    organization_id = (
        DEMO_ORG if document.scope.organization_id == "workspace-demo" else FOREIGN_ORG
    )
    return document.model_copy(
        update={
            "id": f"pg-{document.id}",
            "record_id": f"pg-{document.record_id}",
            "scope": ScopeKey(
                organization_id=organization_id,
                strategy_id=document.scope.strategy_id,
            ),
            "source_ids": tuple(f"pg-{item}" for item in document.source_ids),
        }
    )


async def _seed_documents(
    session: AsyncSession,
    documents: tuple[BenchmarkDocument, ...],
) -> None:
    by_record: dict[str, list[BenchmarkDocument]] = defaultdict(list)
    for document in documents:
        by_record[document.record_id].append(document)
    for record_id, revisions in by_record.items():
        current = next(item for item in revisions if item.revision == item.current_revision)
        session.add(
            MemoryRecordRow(
                id=record_id,
                organization_id=current.scope.organization_id,
                strategy_id=current.scope.strategy_id,
                kind="note",
                visibility=current.visibility.value,
                namespace_id=current.namespace_id,
                current_revision=current.current_revision,
                generation=current.current_revision,
                status=current.record_status.value,
                created_at=min(item.recorded_at for item in revisions),
            )
        )
    await session.flush()
    for document in documents:
        revision = document.candidate().revision
        session.add(
            MemoryRevisionRow(
                id=revision.id,
                record_id=revision.record_id,
                organization_id=document.scope.organization_id,
                strategy_id=document.scope.strategy_id,
                number=revision.number,
                content=revision.content,
                content_hash=revision.content_hash,
                source_ids=list(revision.source_ids),
                visibility=revision.visibility.value,
                namespace_id=revision.namespace_id,
                sensitivity=revision.sensitivity,
                valid_from=revision.valid_from,
                valid_until=revision.valid_until,
                recorded_at=revision.recorded_at,
                expires_at=revision.expires_at,
                status=revision.status.value,
                actor_id=revision.actor_id,
                reason=revision.reason,
                supersedes_revision=(
                    revision.supersedes_revision
                    if revision.status is MemoryStatus.SUPERSEDED
                    else None
                ),
            )
        )
    await session.flush()


async def _explain(session: AsyncSession, statement: Any) -> dict[str, Any]:
    compiled = statement.compile(
        dialect=session.get_bind().dialect,
        compile_kwargs={"render_postcompile": True},
    )
    connection = await session.connection()
    result = await connection.exec_driver_sql(
        f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {compiled}",
        compiled.params,
    )
    payload = result.scalar_one()
    if not isinstance(payload, list) or not payload or not isinstance(payload[0], dict):
        raise AssertionError("Postgres did not return a JSON query plan")
    return payload[0]


def _plan_node_types(plan: dict[str, Any]) -> set[str]:
    values = {str(plan["Node Type"])}
    for child in plan.get("Plans", ()):
        if isinstance(child, dict):
            values.update(_plan_node_types(child))
    return values


def _latency_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * 0.95)))
    return {
        "min": round(ordered[0], 6),
        "median": round(median(ordered), 6),
        "p95": round(ordered[p95_index], 6),
        "max": round(ordered[-1], 6),
    }
