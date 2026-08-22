"""Safe JSON scenario loader with deterministic digests and duplicate rejection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import ValidationError

from leo.evals.models import Scenario


class ScenarioLoadError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def default_scenario_root() -> Path:
    """Resolve source-tree fixtures or their wheel-packaged equivalent."""

    packaged = Path(__file__).resolve().parent / "scenarios"
    if packaged.is_dir():
        return packaged
    source_tree = Path(__file__).resolve().parents[3] / "evals" / "scenarios"
    if source_tree.is_dir():
        return source_tree
    raise ScenarioLoadError("scenario_root_missing")


def load_scenarios(
    root: Path, *, scenario_ids: frozenset[str] | None = None
) -> tuple[Scenario, ...]:
    if not root.exists() or not root.is_dir():
        raise ScenarioLoadError("scenario_root_missing")
    loaded: list[Scenario] = []
    seen: set[str] = set()
    for path in sorted(root.glob("*.json")):
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            scenario = Scenario.model_validate(payload)
        except (OSError, ValueError, ValidationError) as exc:
            raise ScenarioLoadError("scenario_invalid") from exc
        if scenario.id in seen:
            raise ScenarioLoadError("scenario_duplicate_id")
        if scenario.provider_mode.value == "live":
            raise ScenarioLoadError("live_scenarios_are_not_allowed_in_offline_runner")
        canonical_payload = dict(payload)
        canonical_payload["fixture_digest"] = ""
        canonical = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
        if scenario.fixture_digest != hashlib.sha256(canonical.encode("utf-8")).hexdigest():
            raise ScenarioLoadError("scenario_digest_mismatch")
        seen.add(scenario.id)
        if scenario_ids is None or scenario.id in scenario_ids:
            loaded.append(scenario)
    if scenario_ids is not None and not {scenario.id for scenario in loaded}.issuperset(
        scenario_ids
    ):
        raise ScenarioLoadError("scenario_filter_not_found")
    return tuple(loaded)
