"""Provider-neutral, bounded, sanitized parent/plan/child replay timelines."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import Field, JsonValue, model_validator

from leo.harness.context import context_manifest_event_payload
from leo.harness.models import (
    BudgetUsage,
    ContextManifest,
    ContractModel,
    EventType,
    NonEmptyStr,
    OriginRef,
    RunBundle,
    RunStatus,
    ScopeKey,
)
from leo.harness.plan_models import PlanSnapshot

REPLAY_SCHEMA_VERSION = "replay-v1"
MAX_REPLAY_ENTRIES = 2_048
MAX_REPLAY_STRING_CHARS = 2_000
_MAX_COLLECTION_ITEMS = 64
_MAX_SANITIZE_DEPTH = 6
_SECRET_KEYS = re.compile(
    r"(?:authorization|cookie|credential|database_url|password|secret|token)",
    re.IGNORECASE,
)
_SECRET_VALUES = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{12,}"),
    re.compile(r"(?:sk|rk)-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\b[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    re.compile(r"(?:[A-Za-z]:\\|/(?:home|Users|var|tmp)/)[^\s]+"),
)


class ReplayLane(StrEnum):
    PARENT_EVENT = "parent_event"
    PLAN_REVISION = "plan_revision"
    PLAN_NODE = "plan_node"
    DELEGATION = "delegation"
    CHILD_EVENT = "child_event"
    OBSERVATION = "observation"
    CLAIM = "claim"


class ReplayFormat(StrEnum):
    JSON = "json"
    TEXT = "text"


class ReplaySourceManifest(ContractModel):
    schema_version: int = Field(ge=1)
    manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget_profile: NonEmptyStr = "legacy"
    estimator_version: NonEmptyStr = "legacy"
    included_source_ids: tuple[NonEmptyStr, ...] = Field(max_length=256)
    excluded_source_ids: tuple[NonEmptyStr, ...] = Field(max_length=256)
    omitted_source_id_count: int = Field(default=0, ge=0)
    included_estimated_tokens: int = Field(default=0, ge=0)
    excluded_estimated_tokens: int = Field(default=0, ge=0)
    included_estimated_bytes: int = Field(default=0, ge=0)
    excluded_estimated_bytes: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def source_sets_are_canonical(self) -> ReplaySourceManifest:
        for values in (self.included_source_ids, self.excluded_source_ids):
            if values != tuple(sorted(set(values))):
                raise ValueError("replay source IDs must be sorted and unique")
        if set(self.included_source_ids).intersection(self.excluded_source_ids):
            raise ValueError("replay included and excluded source IDs overlap")
        return self


class ReplayEntry(ContractModel):
    sequence: int = Field(ge=1, le=MAX_REPLAY_ENTRIES)
    lane: ReplayLane
    lane_sequence: int = Field(ge=0)
    occurred_at: datetime | None = None
    kind: NonEmptyStr
    plan_id: str | None = None
    revision_id: str | None = None
    node_id: str | None = None
    child_task_id: str | None = None
    child_run_id: str | None = None
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class NormalizedReplay(ContractModel):
    schema_version: str = Field(default=REPLAY_SCHEMA_VERSION, pattern=r"^replay-v[0-9]+$")
    scope: ScopeKey
    conversation: OriginRef
    thread_id: NonEmptyStr
    task_id: NonEmptyStr
    run_id: NonEmptyStr
    objective: NonEmptyStr
    status: RunStatus
    terminal_reason: str | None = None
    final_output: str | None = None
    usage: BudgetUsage
    source_manifest: ReplaySourceManifest | None = None
    entries: tuple[ReplayEntry, ...] = Field(max_length=MAX_REPLAY_ENTRIES)
    omitted_entry_count: int = Field(default=0, ge=0)
    omissions: tuple[NonEmptyStr, ...] = ()
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def timeline_and_digest_are_stable(self) -> NormalizedReplay:
        if tuple(entry.sequence for entry in self.entries) != tuple(
            range(1, len(self.entries) + 1)
        ):
            raise ValueError("normalized replay entries must be contiguous and ordered")
        if self.omissions != tuple(sorted(set(self.omissions))):
            raise ValueError("normalized replay omissions must be sorted and unique")
        expected = _digest(self.model_dump(mode="json", exclude={"digest"}))
        if self.digest != expected:
            raise ValueError("normalized replay digest mismatch")
        return self


@dataclass(frozen=True)
class _Candidate:
    lane: ReplayLane
    lane_sequence: int
    occurred_at: datetime | None
    kind: str
    stable_id: str
    payload: dict[str, JsonValue]
    plan_id: str | None = None
    revision_id: str | None = None
    node_id: str | None = None
    child_task_id: str | None = None
    child_run_id: str | None = None


_LANE_ORDER = {lane: index for index, lane in enumerate(ReplayLane)}


def normalize_replay(
    bundle: RunBundle,
    *,
    plan: PlanSnapshot | None = None,
    children: tuple[RunBundle, ...] = (),
    source_manifest: ContextManifest | ReplaySourceManifest | None = None,
    max_entries: int = MAX_REPLAY_ENTRIES,
) -> NormalizedReplay:
    """Build one exact-authority replay without serializing raw provider DTOs."""

    if not 1 <= max_entries <= MAX_REPLAY_ENTRIES:
        raise ValueError("replay entry bound is invalid")
    _validate_replay_authority(bundle, plan=plan, children=children)
    candidates = _parent_candidates(bundle)
    if plan is not None:
        candidates.extend(_plan_candidates(plan))
    for child in children:
        candidates.extend(_child_candidates(child))
    ordered = sorted(candidates, key=_candidate_sort_key)
    selected = ordered[:max_entries]
    entries = tuple(
        ReplayEntry(
            sequence=index,
            lane=item.lane,
            lane_sequence=item.lane_sequence,
            occurred_at=item.occurred_at,
            kind=item.kind,
            plan_id=item.plan_id,
            revision_id=item.revision_id,
            node_id=item.node_id,
            child_task_id=item.child_task_id,
            child_run_id=item.child_run_id,
            payload=item.payload,
        )
        for index, item in enumerate(selected, start=1)
    )
    resolved_source_manifest = source_manifest or _source_manifest_from_events(bundle)
    omissions: list[str] = []
    if resolved_source_manifest is None:
        omissions.append("source_manifest_not_persisted")
    if plan is None:
        omissions.append("plan_snapshot_not_bound")
    if not children:
        omissions.append("child_runs_not_bound")
    payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "scope": bundle.run.scope.model_dump(mode="json"),
        "conversation": bundle.thread.origin.model_dump(mode="json"),
        "thread_id": bundle.thread.id,
        "task_id": bundle.task.id,
        "run_id": bundle.run.id,
        "objective": _safe_text(bundle.task.objective),
        "status": bundle.run.status.value,
        "terminal_reason": bundle.run.terminal_reason,
        "final_output": (
            None if bundle.run.final_output is None else _safe_text(bundle.run.final_output)
        ),
        "usage": bundle.run.usage.model_dump(mode="json"),
        "source_manifest": (
            None
            if resolved_source_manifest is None
            else _source_manifest(resolved_source_manifest).model_dump(mode="json")
        ),
        "entries": [entry.model_dump(mode="json") for entry in entries],
        "omitted_entry_count": len(ordered) - len(selected),
        "omissions": sorted(omissions),
    }
    return NormalizedReplay.model_validate({**payload, "digest": _digest(payload)})


def render_replay_json(replay: NormalizedReplay) -> str:
    return (
        json.dumps(
            replay.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_replay_text(replay: NormalizedReplay) -> str:
    lines = [
        (f"Leo replay {replay.schema_version} run={replay.run_id} status={replay.status.value}")
    ]
    for entry in replay.entries:
        timestamp = entry.occurred_at.isoformat() if entry.occurred_at is not None else "unknown"
        payload = json.dumps(
            entry.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        lines.append(f"{entry.sequence:04d} {timestamp} {entry.lane.value}:{entry.kind} {payload}")
    lines.append(f"terminal_reason={replay.terminal_reason or 'none'}")
    lines.append(f"omissions={','.join(replay.omissions) or 'none'}")
    lines.append(f"digest={replay.digest}")
    return "\n".join(lines) + "\n"


def export_replay(
    replay: NormalizedReplay,
    destination: Path,
    *,
    output_format: ReplayFormat = ReplayFormat.JSON,
) -> Path:
    """Atomically export one already-sanitized normalized replay."""

    destination = destination.resolve()
    parent = destination.parent
    if not parent.is_dir():
        raise ValueError("replay export parent directory does not exist")
    rendered = (
        render_replay_json(replay)
        if output_format is ReplayFormat.JSON
        else render_replay_text(replay)
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=parent,
            delete=False,
        ) as temporary:
            temporary.write(rendered)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def _validate_replay_authority(
    parent: RunBundle,
    *,
    plan: PlanSnapshot | None,
    children: tuple[RunBundle, ...],
) -> None:
    if plan is not None and (
        plan.plan.scope.organization_id != parent.run.scope.organization_id
        or plan.plan.parent_task_id != parent.task.id
        or plan.plan.parent_run_id != parent.run.id
    ):
        raise ValueError("replay plan is outside the parent authority")
    child_run_ids = tuple(child.run.id for child in children)
    if len(child_run_ids) != len(set(child_run_ids)):
        raise ValueError("replay child runs must be unique")
    for child in children:
        if (
            child.task.parent_task_id != parent.task.id
            or child.run.scope != parent.run.scope
            or child.task.scope != parent.task.scope
        ):
            raise ValueError("replay child is outside the parent authority")
    if plan is not None:
        linked_child_runs = {
            child_run_id
            for child_run_id in (
                *(node.child_run_id for node in plan.nodes),
                *(delegation.child_run_id for delegation in plan.delegations),
            )
            if child_run_id is not None
        }
        if any(child.run.id not in linked_child_runs for child in children):
            raise ValueError("replay child is not linked by the durable plan")


def _parent_candidates(bundle: RunBundle) -> list[_Candidate]:
    candidates = [
        _Candidate(
            lane=ReplayLane.PARENT_EVENT,
            lane_sequence=event.sequence,
            occurred_at=event.occurred_at,
            kind=event.type.value,
            stable_id=event.id,
            payload=cast(
                dict[str, JsonValue],
                _sanitize(
                    {
                        "event_id": event.id,
                        "iteration": event.iteration,
                        "schema_version": event.schema_version,
                        "payload": event.payload,
                    }
                ),
            ),
        )
        for event in bundle.events
    ]
    candidates.extend(
        _Candidate(
            lane=ReplayLane.OBSERVATION,
            lane_sequence=index,
            occurred_at=observation.observed_at,
            kind=observation.kind,
            stable_id=observation.id,
            payload=cast(dict[str, JsonValue], _sanitize(observation.model_dump(mode="json"))),
        )
        for index, observation in enumerate(bundle.observations, start=1)
    )
    candidates.extend(
        _Candidate(
            lane=ReplayLane.CLAIM,
            lane_sequence=index,
            occurred_at=None,
            kind=claim.kind.value,
            stable_id=claim.id,
            payload=cast(dict[str, JsonValue], _sanitize(claim.model_dump(mode="json"))),
        )
        for index, claim in enumerate(bundle.claims, start=1)
    )
    return candidates


def _plan_candidates(snapshot: PlanSnapshot) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for revision in snapshot.revisions:
        candidates.append(
            _Candidate(
                lane=ReplayLane.PLAN_REVISION,
                lane_sequence=revision.number,
                occurred_at=revision.created_at,
                kind="plan_revision",
                stable_id=revision.id,
                plan_id=snapshot.plan.id,
                revision_id=revision.id,
                payload=cast(
                    dict[str, JsonValue],
                    _sanitize(revision.model_dump(mode="json")),
                ),
            )
        )
    for node in snapshot.nodes:
        candidates.append(
            _Candidate(
                lane=ReplayLane.PLAN_NODE,
                lane_sequence=node.revision_number,
                occurred_at=node.updated_at,
                kind=f"plan_node_{node.status.value}",
                stable_id=node.id,
                plan_id=snapshot.plan.id,
                revision_id=node.revision_id,
                node_id=node.id,
                child_task_id=node.child_task_id,
                child_run_id=node.child_run_id,
                payload=cast(dict[str, JsonValue], _sanitize(node.model_dump(mode="json"))),
            )
        )
    for delegation in snapshot.delegations:
        candidates.append(
            _Candidate(
                lane=ReplayLane.DELEGATION,
                lane_sequence=delegation.attempt,
                occurred_at=delegation.finished_at or delegation.created_at,
                kind=f"delegation_{delegation.status.value}",
                stable_id=delegation.id,
                plan_id=snapshot.plan.id,
                revision_id=delegation.revision_id,
                node_id=delegation.node_id,
                child_task_id=delegation.child_task_id,
                child_run_id=delegation.child_run_id,
                payload=cast(
                    dict[str, JsonValue],
                    _sanitize(delegation.model_dump(mode="json")),
                ),
            )
        )
    return candidates


def _child_candidates(bundle: RunBundle) -> list[_Candidate]:
    return [
        _Candidate(
            lane=ReplayLane.CHILD_EVENT,
            lane_sequence=event.sequence,
            occurred_at=event.occurred_at,
            kind=event.type.value,
            stable_id=f"{bundle.run.id}:{event.id}",
            child_task_id=bundle.task.id,
            child_run_id=bundle.run.id,
            payload=cast(
                dict[str, JsonValue],
                _sanitize(
                    {
                        "event_id": event.id,
                        "iteration": event.iteration,
                        "payload": event.payload,
                    }
                ),
            ),
        )
        for event in bundle.events
    ]


def _source_manifest(
    manifest: ContextManifest | ReplaySourceManifest,
) -> ReplaySourceManifest:
    if isinstance(manifest, ReplaySourceManifest):
        return manifest
    return ReplaySourceManifest.model_validate(context_manifest_event_payload(manifest))


def _source_manifest_from_events(bundle: RunBundle) -> ReplaySourceManifest | None:
    candidates = tuple(
        event.payload.get("source_manifest")
        for event in bundle.events
        if event.type is EventType.CONTEXT_BUILT and "source_manifest" in event.payload
    )
    if not candidates:
        return None
    latest = candidates[-1]
    if not isinstance(latest, dict):
        raise ValueError("persisted replay source manifest is malformed")
    try:
        return ReplaySourceManifest.model_validate(latest)
    except ValueError as exc:
        raise ValueError("persisted replay source manifest is malformed") from exc


def _candidate_sort_key(item: _Candidate) -> tuple[bool, datetime, int, int, str]:
    return (
        item.occurred_at is None,
        item.occurred_at or datetime.max.replace(tzinfo=UTC),
        _LANE_ORDER[item.lane],
        item.lane_sequence,
        item.stable_id,
    )


def _sanitize(value: object, *, depth: int = 0, key: str | None = None) -> JsonValue:
    if key is not None and _SECRET_KEYS.search(key):
        return "[REDACTED]"
    if depth >= _MAX_SANITIZE_DEPTH:
        return {"redacted": "depth_limit"}
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, dict):
        output: dict[str, JsonValue] = {}
        for raw_key, item in list(sorted(value.items(), key=lambda pair: str(pair[0])))[
            :_MAX_COLLECTION_ITEMS
        ]:
            safe_key = _safe_text(str(raw_key), max_chars=120)
            output[safe_key] = _sanitize(item, depth=depth + 1, key=str(raw_key))
        if len(value) > _MAX_COLLECTION_ITEMS:
            output["[TRUNCATED_ITEMS]"] = len(value) - _MAX_COLLECTION_ITEMS
        return output
    if isinstance(value, list | tuple):
        items = [_sanitize(item, depth=depth + 1) for item in value[:_MAX_COLLECTION_ITEMS]]
        if len(value) > _MAX_COLLECTION_ITEMS:
            items.append({"truncated_items": len(value) - _MAX_COLLECTION_ITEMS})
        return items
    return _safe_text(str(value))


def _safe_text(value: str, *, max_chars: int = MAX_REPLAY_STRING_CHARS) -> str:
    if any(pattern.search(value) for pattern in _SECRET_VALUES):
        return "[REDACTED]"
    if len(value) <= max_chars:
        return value
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"[TRUNCATED sha256={digest} chars={len(value)}]"


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
