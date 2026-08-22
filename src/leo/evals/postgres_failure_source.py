"""Exact-organization failure bundles derived from durable Postgres run events."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.evals.failure import (
    FailureBundle,
    FailureExportAuthority,
    FailureExportNotFound,
    classify_failure,
    make_bundle,
)
from leo.harness.events import event_contract_digest, normalize_run_timeline
from leo.harness.models import EventType, RunEvent, ScopeKey
from leo.persistence.schema import RunEventRow, RunRow, TaskRow

_FAILURE_STATUSES = frozenset({"failed", "cancelled", "timed_out", "budget_exhausted"})
_TERMINAL_FAILURE_EVENTS = frozenset(
    {
        EventType.RUN_CANCELLED,
        EventType.RUN_FAILED,
        EventType.RUN_TIMED_OUT,
        EventType.BUDGET_EXHAUSTED,
    }
)


class PostgresFailureEventSource:
    """Build a sanitized reproducible bundle from one exact durable run timeline."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        fixture_id: str = "durable-run-failure-v1",
        config_versions: dict[str, str] | None = None,
    ) -> None:
        self._sessions = sessions
        self._fixture_id = fixture_id
        self._config_versions = config_versions or {
            "event_contract": event_contract_digest(),
            "failure_source": "postgres-failure-source-v1",
        }

    async def load(
        self,
        *,
        authority: FailureExportAuthority,
        run_id: str,
    ) -> FailureBundle:
        if run_id not in authority.allowed_run_ids:
            raise FailureExportNotFound
        async with self._sessions() as session:
            run_row = await session.scalar(
                select(RunRow).where(
                    RunRow.id == run_id,
                    RunRow.organization_id == authority.organization_id,
                )
            )
            if run_row is None or run_row.status not in _FAILURE_STATUSES:
                raise FailureExportNotFound
            task_row = await session.scalar(
                select(TaskRow).where(
                    TaskRow.id == run_row.task_id,
                    TaskRow.organization_id == authority.organization_id,
                )
            )
            if task_row is None:
                raise FailureExportNotFound
            rows = tuple(
                (
                    await session.scalars(
                        select(RunEventRow)
                        .where(
                            RunEventRow.run_id == run_id,
                            RunEventRow.task_id == task_row.id,
                        )
                        .order_by(RunEventRow.sequence)
                    )
                ).all()
            )

        events = tuple(_event_model(row) for row in rows)
        scope = ScopeKey(
            organization_id=run_row.organization_id,
            strategy_id=run_row.strategy_id,
        )
        normalized = normalize_run_timeline(events, scope)
        terminal = next(
            (item for item in reversed(events) if item.type in _TERMINAL_FAILURE_EVENTS),
            None,
        )
        root_code = _root_code(terminal, run_row.terminal_reason, run_row.status)
        failure = classify_failure(
            run_id,
            root_code,
            reproduction_command=f"leo replay {run_id}",
            boundary=terminal.type.value if terminal is not None else "durable_terminal",
            terminal_reason=run_row.terminal_reason,
            event_ids=tuple(item.id for item in events),
        )
        sanitized_events = tuple(
            {
                "event_id": item.event_id,
                "run_id": item.run_id,
                "task_id": item.task_id,
                "sequence": item.sequence,
                "occurred_at": item.occurred_at.isoformat(),
                "kind": item.kind.value,
                "schema_version": item.schema_version,
                "correlation_id": item.correlation_id,
                "causation_id": item.causation_id,
                "payload": item.payload,
            }
            for item in normalized
        )
        return make_bundle(
            failure,
            fixture_id=self._fixture_id,
            sanitized_config=self._config_versions,
            events=sanitized_events,
        )


def _event_model(row: RunEventRow) -> RunEvent:
    return RunEvent(
        id=row.id,
        run_id=row.run_id,
        task_id=row.task_id,
        sequence=row.sequence,
        type=EventType(row.type),
        occurred_at=row.occurred_at,
        iteration=row.iteration,
        schema_version=row.schema_version,
        payload=row.payload,
    )


def _root_code(
    event: RunEvent | None,
    terminal_reason: str | None,
    status: str,
) -> str:
    if event is not None:
        for key in ("code", "reason"):
            value = event.payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return terminal_reason or status
