"""Paginated, escaped, regenerable read-only memory projection."""

from __future__ import annotations

import base64
import binascii
import hashlib
import html
import json

from pydantic import Field

from leo.harness.models import (
    ContextItem,
    ContextItemKind,
    ContractModel,
    NonEmptyStr,
    ScopeKey,
)
from leo.memory.models import MemoryRecord, MemoryRevision, MemoryStatus
from leo.memory.retrieval import AuthorizedMemoryNamespace


class ProjectionRequest(ContractModel):
    scope: ScopeKey
    authorized_namespaces: frozenset[AuthorizedMemoryNamespace] = Field(min_length=1)
    generated_at: NonEmptyStr
    policy_version: NonEmptyStr
    page_size: int = Field(default=25, ge=1, le=100)
    after: str | None = None


class MemoryProjectionPage(ContractModel):
    markdown: NonEmptyStr
    digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    item_count: int = Field(ge=0)
    next_cursor: str | None = None
    source_revisions: tuple[tuple[NonEmptyStr, int], ...] = ()


def render_memory_projection_page(
    records: tuple[tuple[MemoryRecord, MemoryRevision], ...],
    request: ProjectionRequest,
) -> MemoryProjectionPage:
    """Render one deterministic page after exact scope/visibility/current filtering."""

    authorized = {(item.visibility, item.namespace_id) for item in request.authorized_namespaces}
    eligible = tuple(
        sorted(
            (
                (record, revision)
                for record, revision in records
                if record.scope.organization_id == request.scope.organization_id
                and record.current_revision == revision.number
                and record.status in {MemoryStatus.ACTIVE, MemoryStatus.CONTESTED}
                and revision.status in {MemoryStatus.ACTIVE, MemoryStatus.CONTESTED}
                and (revision.visibility, revision.namespace_id) in authorized
            ),
            key=lambda item: (item[0].id, item[1].number),
        )
    )
    cursor = _decode_cursor(request.after, request.scope) if request.after is not None else None
    if cursor is not None:
        eligible = tuple(item for item in eligible if (item[0].id, item[1].number) > cursor)
    selected = eligible[: request.page_size]
    has_more = len(eligible) > len(selected)
    next_cursor = None
    if has_more and selected:
        next_cursor = _encode_cursor(request.scope, selected[-1][0].id, selected[-1][1].number)
    markdown = _render(
        selected,
        generated_at=request.generated_at,
        policy_version=request.policy_version,
        page_note=("more available" if has_more else "final page"),
    )
    return MemoryProjectionPage(
        markdown=markdown,
        digest=hashlib.sha256(markdown.encode()).hexdigest(),
        item_count=len(selected),
        next_cursor=next_cursor,
        source_revisions=tuple((record.id, revision.number) for record, revision in selected),
    )


def render_memory_projection(
    records: tuple[tuple[MemoryRecord, MemoryRevision], ...],
    *,
    generated_at: str,
    policy_version: str,
) -> tuple[str, str]:
    """Compatibility renderer for an already-authorized bounded record tuple."""

    selected = tuple(
        (record, revision)
        for record, revision in sorted(records, key=lambda item: item[0].id)
        if record.status is not MemoryStatus.RETRACTED
        and revision.status is not MemoryStatus.RETRACTED
    )
    text = _render(
        selected,
        generated_at=generated_at,
        policy_version=policy_version,
        page_note="provided authorized set",
    )
    return text, hashlib.sha256(text.encode()).hexdigest()


def projection_context_item(
    page: MemoryProjectionPage,
    *,
    scope: ScopeKey,
    conversation_id: str,
) -> ContextItem:
    """Expose only an already-authorized page as optional, derived model context."""

    return ContextItem(
        id=f"memory-projection:{page.digest}",
        kind=ContextItemKind.MEMORY,
        content=page.markdown,
        conversation_id=conversation_id,
        source_scope=scope,
    )


def _render(
    records: tuple[tuple[MemoryRecord, MemoryRevision], ...],
    *,
    generated_at: str,
    policy_version: str,
    page_note: str,
) -> str:
    lines = [
        "# Leo memory projection",
        "",
        "Derived/read-only; edits are ignored and cannot mutate memory.",
        f"Generated: {_escape(generated_at)}",
        f"Policy: {_escape(policy_version)}",
        f"Page: {_escape(page_note)}",
        "",
    ]
    if not records:
        lines.append("_No authorized current memory records on this page._")
    for record, revision in records:
        lines.append(
            f"- `{_escape(record.id)}` revision `{revision.number}` ({revision.status.value})"
        )
        lines.append(
            "  - Authority: "
            f"workspace `{_escape(record.scope.organization_id)}`, "
            f"{revision.visibility.value} `{_escape(revision.namespace_id)}`"
        )
        lines.append(
            "  - Optional domain provenance: "
            f"`{_escape(record.scope.strategy_id)}` (not disclosure authority)"
        )
        lines.append(f"  - Content: {_escape(revision.content)}")
        lines.append(
            "  - Sources: " + ", ".join(_escape(source_id) for source_id in revision.source_ids)
        )
    return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    normalized = " ".join(value.splitlines()).strip()
    escaped = html.escape(normalized, quote=True)
    escaped = escaped.replace("\\", "\\\\")
    for character in "`*_{}[]()#+-.!|":
        escaped = escaped.replace(character, f"\\{character}")
    return escaped.replace("@", "&#64;")


def _encode_cursor(scope: ScopeKey, record_id: str, revision: int) -> str:
    payload = {
        "organization_id": scope.organization_id,
        "strategy_id": scope.strategy_id,
        "record_id": record_id,
        "revision": revision,
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).decode()
    return encoded.rstrip("=")


def _decode_cursor(cursor: str, scope: ScopeKey) -> tuple[str, int]:
    try:
        padding = "=" * (-len(cursor) % 4)
        payload = json.loads(base64.urlsafe_b64decode(cursor + padding))
        if (
            payload["organization_id"] != scope.organization_id
            or payload["strategy_id"] != scope.strategy_id
            or not isinstance(payload["record_id"], str)
            or not payload["record_id"]
            or not isinstance(payload["revision"], int)
            or payload["revision"] < 1
        ):
            raise ValueError
        return payload["record_id"], payload["revision"]
    except (binascii.Error, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("projection cursor is invalid for this scope") from exc
