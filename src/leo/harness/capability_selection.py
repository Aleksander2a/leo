"""Architecture-safe canonical fingerprints for model-visible capability selection."""

from __future__ import annotations

import hashlib
import json

from leo.harness.models import ToolSpec


def capability_selection_fingerprint(
    *,
    catalog_fingerprint: str,
    query_hash: str,
    tools: tuple[ToolSpec, ...],
    selected_skills: tuple[str, ...],
    mode: str,
) -> str:
    payload = {
        "catalog_fingerprint": catalog_fingerprint,
        "mode": mode,
        "query_hash": query_hash,
        "selected_skills": selected_skills,
        "tools": [tool.model_dump(mode="json") for tool in tools],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
