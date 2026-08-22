"""Content-free, read-only Slack topology evidence for trusted operators."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import Field, model_validator
from slack_sdk.web.async_client import AsyncWebClient

from leo.config import Settings
from leo.harness.models import ContractModel

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

SLACK_TOPOLOGY_VERSION: Literal["slack-topology-v1"] = "slack-topology-v1"
SLACK_CONVERSATION_TYPES = "public_channel,private_channel,mpim,im"
SLACK_PAGE_LIMIT = 200
MAX_SLACK_TOPOLOGY_PAGES = 1_000
_EMPTY_DIGEST = hashlib.sha256(b"[]").hexdigest()


class SlackTopologyError(ValueError):
    """Slack returned an incomplete, contradictory, or unauthorized topology."""


class TopologyConversationKind(StrEnum):
    PUBLIC_CHANNEL = "public_channel"
    PRIVATE_CHANNEL = "private_channel"
    MPIM = "mpim"
    DM = "dm"
    SHARED_EXTERNAL = "shared_external"


class BotPresence(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class SharedExternalPresence(StrEnum):
    PRESENT = "present"
    ABSENT = "absent"


class SlackTopologyConversation(ContractModel):
    """One redacted bot-visible conversation; no names, text, or raw IDs survive."""

    conversation_id_digest: Sha256
    kind: TopologyConversationKind
    bot_presence: BotPresence
    is_member: bool | None
    is_archived: bool
    is_shared: bool
    is_ext_shared: bool
    shared_team_count: int = Field(ge=0)
    external_team_count: int = Field(ge=0)

    @model_validator(mode="after")
    def flags_match_kind(self) -> SlackTopologyConversation:
        if self.external_team_count > self.shared_team_count:
            raise ValueError("external Slack team count exceeds shared team count")
        external = self.is_ext_shared or self.external_team_count > 0
        if (self.kind is TopologyConversationKind.SHARED_EXTERNAL) != external:
            raise ValueError("shared-external Slack classification is inconsistent")
        expected_presence = (
            BotPresence.PRESENT
            if self.is_member is True
            else BotPresence.ABSENT
            if self.is_member is False
            else (
                BotPresence.PRESENT
                if self.kind in {TopologyConversationKind.DM, TopologyConversationKind.MPIM}
                else BotPresence.UNKNOWN
            )
        )
        if self.bot_presence is not expected_presence:
            raise ValueError("Slack bot presence is inconsistent with membership metadata")
        return self


class SlackTopologyArtifact(ContractModel):
    """Versioned, self-digesting evidence that a complete topology read occurred."""

    version: Literal["slack-topology-v1"] = SLACK_TOPOLOGY_VERSION
    collected_at: datetime
    team_identity_digest: Sha256
    app_identity_digest: Sha256
    request_contract_digest: Sha256
    page_count: int = Field(ge=1, le=MAX_SLACK_TOPOLOGY_PAGES)
    pagination_manifest_digest: Sha256
    conversation_count: int = Field(ge=0)
    kind_counts: dict[TopologyConversationKind, int]
    present_kinds: tuple[TopologyConversationKind, ...]
    bot_presence_counts: dict[BotPresence, int]
    shared_external_presence: SharedExternalPresence
    conversations: tuple[SlackTopologyConversation, ...]
    manifest_digest: Sha256
    digest: Sha256

    @model_validator(mode="after")
    def exact_aggregate_and_digest(self) -> SlackTopologyArtifact:
        if self.collected_at.tzinfo is None:
            raise ValueError("Slack topology collection time must be timezone-aware")
        identifiers = tuple(item.conversation_id_digest for item in self.conversations)
        if identifiers != tuple(sorted(identifiers)) or len(identifiers) != len(set(identifiers)):
            raise ValueError("Slack topology conversations must be sorted and unique")
        expected_kind_counts = _kind_counts(self.conversations)
        expected_presence_counts = _presence_counts(self.conversations)
        expected_present = tuple(
            kind for kind in TopologyConversationKind if expected_kind_counts[kind]
        )
        expected_shared = (
            SharedExternalPresence.PRESENT
            if expected_kind_counts[TopologyConversationKind.SHARED_EXTERNAL]
            else SharedExternalPresence.ABSENT
        )
        if (
            self.conversation_count != len(self.conversations)
            or self.kind_counts != expected_kind_counts
            or self.bot_presence_counts != expected_presence_counts
            or self.present_kinds != expected_present
            or self.shared_external_presence is not expected_shared
        ):
            raise ValueError("Slack topology aggregate diverges from its manifest")
        expected_manifest = _manifest_digest(
            team_identity_digest=self.team_identity_digest,
            app_identity_digest=self.app_identity_digest,
            request_contract_digest=self.request_contract_digest,
            pagination_manifest_digest=self.pagination_manifest_digest,
            conversations=self.conversations,
        )
        if self.manifest_digest != expected_manifest:
            raise ValueError("Slack topology manifest digest does not match")
        if self.digest != _artifact_digest(self):
            raise ValueError("Slack topology artifact digest does not match")
        return self


class AsyncSlackTopologySource(Protocol):
    async def auth_test(self) -> Mapping[str, object]: ...

    async def list_conversations(self, *, cursor: str | None) -> Mapping[str, object]: ...


class SlackWebTopologySource:
    """Narrow read-only adapter over Slack's async Web API client."""

    def __init__(self, client: AsyncWebClient) -> None:
        self._client = client

    async def auth_test(self) -> Mapping[str, object]:
        return _response_payload(await self._client.auth_test())

    async def list_conversations(self, *, cursor: str | None) -> Mapping[str, object]:
        if cursor is None:
            response = await self._client.conversations_list(
                types=SLACK_CONVERSATION_TYPES,
                exclude_archived=False,
                limit=SLACK_PAGE_LIMIT,
            )
        else:
            response = await self._client.conversations_list(
                types=SLACK_CONVERSATION_TYPES,
                exclude_archived=False,
                limit=SLACK_PAGE_LIMIT,
                cursor=cursor,
            )
        return _response_payload(response)


async def collect_slack_topology(
    source: AsyncSlackTopologySource,
    *,
    expected_team_id: str,
    collected_at: datetime,
) -> SlackTopologyArtifact:
    """Fully paginate the bot-visible topology and return only redacted evidence."""

    if not expected_team_id.strip():
        raise SlackTopologyError("trusted Slack team authority is missing")
    if collected_at.tzinfo is None:
        raise SlackTopologyError("Slack topology collection time must be timezone-aware")
    auth = await source.auth_test()
    _require_ok(auth, "auth.test")
    team_id = _required_string(auth, "team_id", "auth.test")
    if team_id != expected_team_id:
        raise SlackTopologyError("Slack auth team does not match trusted runtime authority")
    app_identity = {
        key: value
        for key in ("app_id", "bot_id", "user_id")
        if isinstance((value := auth.get(key)), str) and value.strip()
    }
    if not app_identity:
        raise SlackTopologyError("Slack auth response lacks an app identity")
    team_identity_digest = _digest_text(team_id)
    app_identity_digest = _digest(app_identity)
    request_contract_digest = _digest(
        {
            "method": "conversations.list",
            "types": SLACK_CONVERSATION_TYPES,
            "exclude_archived": False,
            "limit": SLACK_PAGE_LIMIT,
            "pagination": "cursor_until_empty",
        }
    )

    conversations: list[SlackTopologyConversation] = []
    raw_conversation_ids: set[str] = set()
    page_digests: list[str] = []
    seen_cursors: set[str] = set()
    cursor: str | None = None
    for page_index in range(1, MAX_SLACK_TOPOLOGY_PAGES + 1):
        page = await source.list_conversations(cursor=cursor)
        _require_ok(page, "conversations.list")
        raw_channels = page.get("channels")
        if not isinstance(raw_channels, list):
            raise SlackTopologyError("Slack conversations page lacks a channel list")
        page_conversations: list[SlackTopologyConversation] = []
        for raw in raw_channels:
            if not isinstance(raw, Mapping):
                raise SlackTopologyError("Slack conversation metadata is malformed")
            conversation_id = _required_string(raw, "id", "conversation")
            if conversation_id in raw_conversation_ids:
                raise SlackTopologyError("Slack pagination returned a duplicate conversation")
            raw_conversation_ids.add(conversation_id)
            normalized = _normalize_conversation(raw, team_id=team_id)
            page_conversations.append(normalized)
            conversations.append(normalized)
        next_cursor = _next_cursor(page)
        page_digests.append(
            _digest(
                {
                    "page": page_index,
                    "conversation_digests": sorted(
                        item.conversation_id_digest for item in page_conversations
                    ),
                    "has_next": next_cursor is not None,
                    "next_cursor_digest": (
                        _digest_text(next_cursor) if next_cursor is not None else _EMPTY_DIGEST
                    ),
                }
            )
        )
        if next_cursor is None:
            break
        if next_cursor in seen_cursors:
            raise SlackTopologyError("Slack pagination cursor repeated")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    else:
        raise SlackTopologyError("Slack topology exceeded the hard pagination bound")

    normalized_conversations = tuple(
        sorted(conversations, key=lambda item: item.conversation_id_digest)
    )
    kind_counts = _kind_counts(normalized_conversations)
    presence_counts = _presence_counts(normalized_conversations)
    present_kinds = tuple(kind for kind in TopologyConversationKind if kind_counts[kind])
    shared_external_presence = (
        SharedExternalPresence.PRESENT
        if kind_counts[TopologyConversationKind.SHARED_EXTERNAL]
        else SharedExternalPresence.ABSENT
    )
    pagination_manifest_digest = _digest(page_digests)
    manifest_digest = _manifest_digest(
        team_identity_digest=team_identity_digest,
        app_identity_digest=app_identity_digest,
        request_contract_digest=request_contract_digest,
        pagination_manifest_digest=pagination_manifest_digest,
        conversations=normalized_conversations,
    )
    payload: dict[str, object] = {
        "version": SLACK_TOPOLOGY_VERSION,
        "collected_at": collected_at,
        "team_identity_digest": team_identity_digest,
        "app_identity_digest": app_identity_digest,
        "request_contract_digest": request_contract_digest,
        "page_count": len(page_digests),
        "pagination_manifest_digest": pagination_manifest_digest,
        "conversation_count": len(normalized_conversations),
        "kind_counts": kind_counts,
        "present_kinds": present_kinds,
        "bot_presence_counts": presence_counts,
        "shared_external_presence": shared_external_presence,
        "conversations": normalized_conversations,
        "manifest_digest": manifest_digest,
    }
    digest = _digest(_artifact_payload(payload))
    return SlackTopologyArtifact.model_validate({**payload, "digest": digest})


async def export_slack_topology(
    source: AsyncSlackTopologySource,
    *,
    expected_team_id: str,
    collected_at: datetime,
    destination: Path,
) -> SlackTopologyArtifact:
    artifact = await collect_slack_topology(
        source,
        expected_team_id=expected_team_id,
        collected_at=collected_at,
    )
    _atomic_write(destination, artifact.model_dump_json(indent=2) + "\n")
    return artifact


def _normalize_conversation(
    raw: Mapping[str, object],
    *,
    team_id: str,
) -> SlackTopologyConversation:
    conversation_id = _required_string(raw, "id", "conversation")
    is_im = _boolean(raw, "is_im")
    is_mpim = _boolean(raw, "is_mpim")
    is_channel = _boolean(raw, "is_channel")
    is_group = _boolean(raw, "is_group")
    # Slack may mark MPIMs as both ``is_mpim`` and ``is_channel`` (or legacy
    # ``is_group``).  ``is_mpim`` is the authoritative discriminator, while a
    # one-to-one IM collision remains contradictory.
    if (is_im and (is_mpim or is_channel or is_group)) or not (
        is_im or is_mpim or is_channel or is_group
    ):
        raise SlackTopologyError("Slack conversation type flags are contradictory")
    is_private = _boolean(raw, "is_private")
    is_shared = _boolean(raw, "is_shared")
    is_ext_shared = _boolean(raw, "is_ext_shared")
    is_archived = _boolean(raw, "is_archived")
    shared_team_ids = _string_list(raw, "shared_team_ids")
    external_team_count = len(set(shared_team_ids) - {team_id})
    external = is_ext_shared or external_team_count > 0
    if external and not (is_channel or is_group):
        raise SlackTopologyError("direct Slack conversation cannot be shared-external")
    kind = (
        TopologyConversationKind.SHARED_EXTERNAL
        if external
        else TopologyConversationKind.DM
        if is_im
        else TopologyConversationKind.MPIM
        if is_mpim
        else TopologyConversationKind.PRIVATE_CHANNEL
        if is_private or is_group
        else TopologyConversationKind.PUBLIC_CHANNEL
    )
    membership = _optional_boolean(raw, "is_member")
    presence = (
        BotPresence.PRESENT
        if membership is True
        else BotPresence.ABSENT
        if membership is False
        else (
            BotPresence.PRESENT
            if kind in {TopologyConversationKind.DM, TopologyConversationKind.MPIM}
            else BotPresence.UNKNOWN
        )
    )
    return SlackTopologyConversation(
        conversation_id_digest=_digest_text(conversation_id),
        kind=kind,
        bot_presence=presence,
        is_member=membership,
        is_archived=is_archived,
        is_shared=is_shared or is_ext_shared,
        is_ext_shared=is_ext_shared,
        shared_team_count=len(set(shared_team_ids)),
        external_team_count=external_team_count,
    )


def _next_cursor(page: Mapping[str, object]) -> str | None:
    metadata = page.get("response_metadata")
    if metadata is None:
        return None
    if not isinstance(metadata, Mapping):
        raise SlackTopologyError("Slack pagination metadata is malformed")
    value = metadata.get("next_cursor", "")
    if not isinstance(value, str):
        raise SlackTopologyError("Slack pagination cursor is malformed")
    normalized = value.strip()
    return normalized or None


def _response_payload(response: object) -> Mapping[str, object]:
    payload = getattr(response, "data", response)
    if not isinstance(payload, Mapping):
        raise SlackTopologyError("Slack Web API response is malformed")
    return payload


def _require_ok(payload: Mapping[str, object], method: str) -> None:
    if payload.get("ok") is not True:
        raise SlackTopologyError(f"Slack {method} did not return an authoritative result")


def _required_string(payload: Mapping[str, object], key: str, source: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SlackTopologyError(f"Slack {source} lacks {key}")
    return value


def _boolean(payload: Mapping[str, object], key: str) -> bool:
    value = payload.get(key, False)
    if not isinstance(value, bool):
        raise SlackTopologyError(f"Slack conversation field {key} is not boolean")
    return value


def _optional_boolean(payload: Mapping[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise SlackTopologyError(f"Slack conversation field {key} is not boolean")
    return value


def _string_list(payload: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise SlackTopologyError(f"Slack conversation field {key} is malformed")
    return tuple(value)


def _kind_counts(
    conversations: tuple[SlackTopologyConversation, ...],
) -> dict[TopologyConversationKind, int]:
    return {
        kind: sum(item.kind is kind for item in conversations) for kind in TopologyConversationKind
    }


def _presence_counts(
    conversations: tuple[SlackTopologyConversation, ...],
) -> dict[BotPresence, int]:
    return {
        presence: sum(item.bot_presence is presence for item in conversations)
        for presence in BotPresence
    }


def _manifest_digest(
    *,
    team_identity_digest: str,
    app_identity_digest: str,
    request_contract_digest: str,
    pagination_manifest_digest: str,
    conversations: tuple[SlackTopologyConversation, ...],
) -> str:
    return _digest(
        {
            "team_identity_digest": team_identity_digest,
            "app_identity_digest": app_identity_digest,
            "request_contract_digest": request_contract_digest,
            "pagination_manifest_digest": pagination_manifest_digest,
            "conversations": [item.model_dump(mode="json") for item in conversations],
        }
    )


def _artifact_payload(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        "version": payload["version"],
        "collected_at": _datetime_value(payload["collected_at"]),
        "team_identity_digest": payload["team_identity_digest"],
        "app_identity_digest": payload["app_identity_digest"],
        "request_contract_digest": payload["request_contract_digest"],
        "page_count": payload["page_count"],
        "pagination_manifest_digest": payload["pagination_manifest_digest"],
        "conversation_count": payload["conversation_count"],
        "kind_counts": {str(key): value for key, value in _mapping(payload["kind_counts"]).items()},
        "present_kinds": [str(item) for item in _sequence(payload["present_kinds"])],
        "bot_presence_counts": {
            str(key): value for key, value in _mapping(payload["bot_presence_counts"]).items()
        },
        "shared_external_presence": str(payload["shared_external_presence"]),
        "conversations": [
            item.model_dump(mode="json") if isinstance(item, SlackTopologyConversation) else item
            for item in _sequence(payload["conversations"])
        ],
        "manifest_digest": payload["manifest_digest"],
    }


def _artifact_digest(artifact: SlackTopologyArtifact) -> str:
    return _digest(
        _artifact_payload(
            artifact.model_dump(exclude={"digest"}),
        )
    )


def _mapping(value: object) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise ValueError("Slack topology aggregate is not a mapping")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("Slack topology aggregate is not a sequence")
    return value


def _datetime_value(value: object) -> str:
    if not isinstance(value, datetime):
        raise ValueError("Slack topology collection time is malformed")
    return value.isoformat()


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="leo-slack-topology",
        description=(
            "Collect a content-free Slack topology artifact using only the trusted "
            "SLACK_BOT_TOKEN and LEO_SLACK_TEAM_ID runtime settings."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


async def _run(arguments: argparse.Namespace) -> SlackTopologyArtifact:
    settings = Settings()
    if settings.slack_bot_token is None or settings.leo_slack_team_id is None:
        raise RuntimeError("slack_topology_configuration_missing")
    source = SlackWebTopologySource(
        AsyncWebClient(token=settings.slack_bot_token.get_secret_value())
    )
    return await export_slack_topology(
        source,
        expected_team_id=settings.leo_slack_team_id,
        collected_at=datetime.now(UTC),
        destination=arguments.output,
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        artifact = asyncio.run(_run(arguments))
    except Exception:
        # Slack errors can carry request metadata. Never expose them to operator output.
        print(
            json.dumps(
                {"code": "slack_topology_collection_failed", "status": "failed"},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "conversation_count": artifact.conversation_count,
                "digest": artifact.digest,
                "present_kinds": [str(item) for item in artifact.present_kinds],
                "shared_external_presence": str(artifact.shared_external_presence),
                "status": "ok",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - subprocess contract.
    raise SystemExit(main())
