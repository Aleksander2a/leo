from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from leo.demo_world import load_demo_world

FIXTURE = Path("tests/fixtures/demo_world/v1.json")


def test_demo_world_has_two_stable_contradictory_strategies() -> None:
    world = load_demo_world(FIXTURE)

    assert world.version == "v1"
    assert world.organization_id == "demo-org"
    assert {strategy.id for strategy in world.strategies} == {"technology-ls", "conservative"}
    technology, conservative = world.strategies
    assert technology.asset_views[0].symbol == conservative.asset_views[0].symbol == "NVDA"
    assert technology.asset_views[0].stance != conservative.asset_views[0].stance
    assert technology.asset_views[0].target_weight > conservative.asset_views[0].target_weight


def test_demo_world_load_is_deterministic_and_text_has_no_authority() -> None:
    first = load_demo_world(FIXTURE)
    second = load_demo_world(json.loads(FIXTURE.read_text(encoding="utf-8")))

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert "instruction" not in first.strategy("technology-ls").thesis.lower()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.update({"version": "v2"}),
        lambda payload: payload["strategies"].append(payload["strategies"][0]),
        lambda payload: payload["strategies"][0]["asset_views"][0].update({"target_weight": 1.5}),
    ],
)
def test_demo_world_rejects_invalid_fixture_variants(mutator) -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutator(payload)
    with pytest.raises((ValidationError, ValueError)):
        load_demo_world(payload)


def test_demo_world_rejects_extra_authority_fields() -> None:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    payload["strategies"][0]["scope"] = {"organization_id": "foreign", "strategy_id": "x"}

    with pytest.raises(ValidationError):
        load_demo_world(payload)
