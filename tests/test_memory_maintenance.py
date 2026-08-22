from __future__ import annotations

import pytest

from leo.harness.models import ScopeKey
from leo.memory.maintenance import PurgeTarget, make_purge_plan, validate_confirmation

SCOPE = ScopeKey(organization_id="maintenance-org", strategy_id="demo-domain")


def test_purge_manifest_is_ordered_version_bound_and_explicit() -> None:
    targets = (
        PurgeTarget(record_id="memory-a", generation=2, current_revision=3),
        PurgeTarget(record_id="memory-b", generation=1, current_revision=2),
    )
    plan = make_purge_plan(SCOPE, ("memory-a", "memory-b"), targets=targets)
    validate_confirmation(plan, plan.confirmation_token, scope=SCOPE)

    changed_revision = make_purge_plan(
        SCOPE,
        ("memory-a", "memory-b"),
        targets=(
            targets[0].model_copy(update={"current_revision": 4}),
            targets[1],
        ),
    )
    reordered = make_purge_plan(
        SCOPE,
        ("memory-b", "memory-a"),
        targets=(targets[1], targets[0]),
    )
    assert changed_revision.confirmation_token != plan.confirmation_token
    assert reordered.confirmation_token != plan.confirmation_token
    with pytest.raises(ValueError, match="stale"):
        validate_confirmation(changed_revision, plan.confirmation_token, scope=SCOPE)


@pytest.mark.parametrize(
    "record_ids",
    [(), ("memory-*",), ("memory-?",), ("memory-a", "memory-a")],
)
def test_purge_manifest_rejects_unbounded_or_ambiguous_targets(
    record_ids: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError):
        make_purge_plan(SCOPE, record_ids)
