from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from leo.api.app import create_app
from leo.config import Settings
from leo.health import (
    HealthComponent,
    HealthState,
    SlackSocketReadinessRegistry,
    aggregate_status,
    config_snapshot,
    probe_database,
    probe_operational_metadata,
)


def test_config_health_never_claims_unprobed_database_is_healthy() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    snapshot = config_snapshot(
        Settings(_env_file=None, database_url="postgresql://demo"), observed_at=now
    )
    assert snapshot.status is HealthState.DEGRADED
    database = next(item for item in snapshot.components if item.name == "database")
    assert database.state is HealthState.UNKNOWN
    assert database.reason == "database_probe_not_run"


def test_health_aggregation_is_deterministic() -> None:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    components = (
        HealthComponent(name="process", state=HealthState.OK, observed_at=now, reason="alive"),
        HealthComponent(
            name="database", state=HealthState.DEGRADED, observed_at=now, reason="probe_failed"
        ),
    )
    assert aggregate_status(components) is HealthState.DEGRADED
    assert (
        aggregate_status(
            (components[0], components[0].model_copy(update={"state": HealthState.UNHEALTHY}))
        )
        is HealthState.UNHEALTHY
    )


def test_health_api_returns_the_same_safe_snapshot_shape() -> None:
    with TestClient(create_app(Settings(_env_file=None))) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 2
    assert payload["status"] == "degraded"
    assert all("reason" in component for component in payload["components"])


def test_health_api_deep_mode_uses_bounded_same_process_probes() -> None:
    settings = Settings(_env_file=None, database_url="postgresql://demo")
    with TestClient(
        create_app(
            settings,
            sessions=_HangingSessions(),  # type: ignore[arg-type]
            health_probe_timeout_seconds=0.01,
        )
    ) as client:
        response = client.get("/health", params={"deep": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 2
    components = {component["name"]: component for component in payload["components"]}
    assert components["database"]["reason"] == "database_probe_timeout"
    assert components["parent_child_orchestration"]["reason"] == "metadata_probe_timeout"
    assert components["task_queue"]["state"] == "unknown"
    assert not any("postgresql://" in str(component) for component in components.values())


def test_configured_model_is_unknown_until_a_runtime_result_is_recorded() -> None:
    snapshot = config_snapshot(
        Settings(_env_file=None, openrouter_api_key="test-key", leo_model="test/model")
    )
    model = next(item for item in snapshot.components if item.name == "model")
    assert model.state is HealthState.UNKNOWN
    assert model.reason == "model_result_not_registered"


def test_socket_readiness_uses_observed_connection_state_not_configuration() -> None:
    registry = SlackSocketReadinessRegistry()
    started = datetime(2026, 1, 1, tzinfo=UTC)

    assert registry.component(configured=True, observed_at=started).state is HealthState.UNKNOWN
    registry.record_starting(observed_at=started)
    registry.record_probe(False, observed_at=started)
    connecting = registry.component(configured=True, observed_at=started)
    assert connecting.state is HealthState.UNKNOWN
    assert connecting.reason == "socket_connecting"

    registry.record_probe(True, observed_at=started)
    assert registry.component(configured=True, observed_at=started).state is HealthState.OK
    registry.record_probe(False, observed_at=started)
    disconnected = registry.component(configured=True, observed_at=started)
    assert disconnected.state is HealthState.UNHEALTHY
    assert disconnected.reason == "socket_disconnected"

    registry.record_stopped(observed_at=started)
    assert registry.component(configured=True, observed_at=started).reason == "socket_stopped"


class _HangingSession:
    async def __aenter__(self) -> _HangingSession:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def execute(self, *_: object) -> object:
        await asyncio.Future()
        raise AssertionError("unreachable")


class _HangingSessions:
    def __call__(self) -> _HangingSession:
        return _HangingSession()


@pytest.mark.asyncio
async def test_deep_health_probes_have_hard_timeouts_and_safe_reasons() -> None:
    database, queue, outbox, last_success = await probe_database(  # type: ignore[arg-type]
        _HangingSessions(),
        timeout_seconds=0.01,
    )
    assert database.state is HealthState.DEGRADED
    assert database.reason == "database_probe_timeout"
    assert {queue.reason, outbox.reason, last_success.reason} == {"database_probe_timeout"}

    metadata = await probe_operational_metadata(  # type: ignore[arg-type]
        _HangingSessions(),
        timeout_seconds=0.01,
    )
    assert all(component.state is HealthState.UNKNOWN for component in metadata)
    assert {component.reason for component in metadata} == {"metadata_probe_timeout"}
