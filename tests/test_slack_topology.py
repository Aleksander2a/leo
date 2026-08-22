from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

import leo.evals.slack_topology as topology_module
from leo.evals.slack_topology import (
    BotPresence,
    SharedExternalPresence,
    SlackTopologyArtifact,
    SlackTopologyError,
    TopologyConversationKind,
    _parse_arguments,
    collect_slack_topology,
    export_slack_topology,
)

NOW = datetime(2026, 8, 22, 12, tzinfo=UTC)
TEAM_ID = "T-TRUSTED"


class _FakeTopologySource:
    def __init__(
        self,
        *,
        auth: Mapping[str, object] | None = None,
        pages: Mapping[str | None, Mapping[str, object]] | None = None,
    ) -> None:
        self._auth = auth or _auth_payload()
        self._pages = pages or _complete_pages()
        self.auth_calls = 0
        self.cursors: list[str | None] = []

    async def auth_test(self) -> Mapping[str, object]:
        self.auth_calls += 1
        return self._auth

    async def list_conversations(self, *, cursor: str | None) -> Mapping[str, object]:
        self.cursors.append(cursor)
        try:
            return self._pages[cursor]
        except KeyError as exc:  # pragma: no cover - test fixture guard.
            raise AssertionError(f"unexpected cursor: {cursor!r}") from exc


def _auth_payload() -> dict[str, object]:
    return {
        "ok": True,
        "team_id": TEAM_ID,
        "app_id": "A-PRIVATE-APP",
        "bot_id": "B-PRIVATE-BOT",
        "user_id": "U-PRIVATE-BOT-USER",
        "team": "Secret workspace name",
        "user": "Secret bot display name",
        "url": "https://secret-workspace.example/",
        "token": "xox" + "b-must-never-survive",
    }


def _conversation(
    conversation_id: str,
    *,
    is_channel: bool = False,
    is_group: bool = False,
    is_im: bool = False,
    is_mpim: bool = False,
    is_private: bool = False,
    is_member: bool | None = None,
    is_shared: bool = False,
    is_ext_shared: bool = False,
    shared_team_ids: list[str] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": conversation_id,
        "is_channel": is_channel,
        "is_group": is_group,
        "is_im": is_im,
        "is_mpim": is_mpim,
        "is_private": is_private,
        "is_shared": is_shared,
        "is_ext_shared": is_ext_shared,
        "is_archived": False,
        "shared_team_ids": shared_team_ids or [],
        "name": f"secret-name-{conversation_id}",
        "topic": {"value": "secret topic text"},
        "purpose": {"value": "secret purpose text"},
        "latest": {"text": "secret message text"},
        "token": "xox" + "b-never-serialize",
    }
    if is_member is not None:
        payload["is_member"] = is_member
    return payload


def _complete_pages() -> dict[str | None, Mapping[str, object]]:
    return {
        None: {
            "ok": True,
            "channels": [
                _conversation("C-PUBLIC", is_channel=True, is_member=True),
                _conversation(
                    "C-PRIVATE",
                    is_channel=True,
                    is_private=True,
                    is_member=False,
                ),
            ],
            "response_metadata": {"next_cursor": "cursor-secret-page-2"},
        },
        "cursor-secret-page-2": {
            "ok": True,
            "channels": [
                _conversation("G-MPIM", is_channel=True, is_mpim=True),
                _conversation("D-DM", is_im=True),
                _conversation(
                    "C-SHARED",
                    is_channel=True,
                    is_private=True,
                    is_member=True,
                    is_shared=True,
                    is_ext_shared=True,
                    shared_team_ids=[TEAM_ID, "T-EXTERNAL"],
                ),
            ],
            "response_metadata": {"next_cursor": ""},
        },
    }


@pytest.mark.asyncio
async def test_topology_fully_paginates_and_classifies_all_actual_kinds() -> None:
    source = _FakeTopologySource()

    artifact = await collect_slack_topology(
        source,
        expected_team_id=TEAM_ID,
        collected_at=NOW,
    )

    assert source.auth_calls == 1
    assert source.cursors == [None, "cursor-secret-page-2"]
    assert artifact.page_count == 2
    assert artifact.conversation_count == 5
    assert artifact.present_kinds == tuple(TopologyConversationKind)
    assert artifact.kind_counts == {kind: 1 for kind in TopologyConversationKind}
    assert artifact.bot_presence_counts == {
        BotPresence.PRESENT: 4,
        BotPresence.ABSENT: 1,
        BotPresence.UNKNOWN: 0,
    }
    assert artifact.shared_external_presence is SharedExternalPresence.PRESENT
    assert SlackTopologyArtifact.model_validate_json(artifact.model_dump_json()) == artifact


@pytest.mark.asyncio
async def test_topology_is_deterministic_and_redacts_names_text_ids_cursors_and_tokens() -> None:
    first = await collect_slack_topology(
        _FakeTopologySource(),
        expected_team_id=TEAM_ID,
        collected_at=NOW,
    )
    second = await collect_slack_topology(
        _FakeTopologySource(),
        expected_team_id=TEAM_ID,
        collected_at=NOW,
    )

    assert first == second
    assert first.digest == second.digest
    serialized = first.model_dump_json()
    forbidden = (
        TEAM_ID,
        "T-EXTERNAL",
        "A-PRIVATE-APP",
        "B-PRIVATE-BOT",
        "U-PRIVATE-BOT-USER",
        "C-PUBLIC",
        "C-PRIVATE",
        "G-MPIM",
        "D-DM",
        "C-SHARED",
        "cursor-secret-page-2",
        "secret-name",
        "secret topic text",
        "secret purpose text",
        "secret message text",
        "xoxb-",
    )
    assert all(value not in serialized for value in forbidden)


@pytest.mark.asyncio
async def test_topology_explicitly_records_shared_external_absence_and_unknown_presence() -> None:
    source = _FakeTopologySource(
        pages={
            None: {
                "ok": True,
                "channels": [_conversation("C-PUBLIC", is_channel=True)],
                "response_metadata": {"next_cursor": ""},
            }
        }
    )

    artifact = await collect_slack_topology(
        source,
        expected_team_id=TEAM_ID,
        collected_at=NOW,
    )

    assert artifact.shared_external_presence is SharedExternalPresence.ABSENT
    assert artifact.present_kinds == (TopologyConversationKind.PUBLIC_CHANNEL,)
    assert artifact.bot_presence_counts[BotPresence.UNKNOWN] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth", "pages", "team_id", "match"),
    [
        ({"ok": False}, _complete_pages(), TEAM_ID, "authoritative result"),
        (_auth_payload(), _complete_pages(), "T-FORGED", "trusted runtime authority"),
        (
            _auth_payload(),
            {None: {"ok": True, "channels": "not-a-list"}},
            TEAM_ID,
            "channel list",
        ),
        (
            _auth_payload(),
            {
                None: {
                    "ok": True,
                    "channels": [_conversation("D-BROKEN", is_im=True, is_mpim=True)],
                }
            },
            TEAM_ID,
            "type flags",
        ),
        (
            _auth_payload(),
            {
                None: {
                    "ok": True,
                    "channels": [
                        {
                            **_conversation("C-BROKEN", is_channel=True),
                            "is_member": "yes",
                        }
                    ],
                }
            },
            TEAM_ID,
            "not boolean",
        ),
    ],
)
async def test_topology_fails_closed_on_malformed_or_untrusted_metadata(
    auth: Mapping[str, object],
    pages: Mapping[str | None, Mapping[str, object]],
    team_id: str,
    match: str,
) -> None:
    with pytest.raises(SlackTopologyError, match=match):
        await collect_slack_topology(
            _FakeTopologySource(auth=auth, pages=pages),
            expected_team_id=team_id,
            collected_at=NOW,
        )


@pytest.mark.asyncio
async def test_topology_rejects_repeated_cursor_and_duplicate_conversation() -> None:
    repeated = {
        None: {
            "ok": True,
            "channels": [],
            "response_metadata": {"next_cursor": "repeat"},
        },
        "repeat": {
            "ok": True,
            "channels": [],
            "response_metadata": {"next_cursor": "repeat"},
        },
    }
    with pytest.raises(SlackTopologyError, match="cursor repeated"):
        await collect_slack_topology(
            _FakeTopologySource(pages=repeated),
            expected_team_id=TEAM_ID,
            collected_at=NOW,
        )

    duplicate = {
        None: {
            "ok": True,
            "channels": [_conversation("C-SAME", is_channel=True)],
            "response_metadata": {"next_cursor": "next"},
        },
        "next": {
            "ok": True,
            "channels": [_conversation("C-SAME", is_channel=True)],
            "response_metadata": {"next_cursor": ""},
        },
    }
    with pytest.raises(SlackTopologyError, match="duplicate conversation"):
        await collect_slack_topology(
            _FakeTopologySource(pages=duplicate),
            expected_team_id=TEAM_ID,
            collected_at=NOW,
        )


@pytest.mark.asyncio
async def test_topology_export_is_atomic_parseable_json_with_real_newline(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "proof" / "slack-topology-v1.json"

    artifact = await export_slack_topology(
        _FakeTopologySource(),
        expected_team_id=TEAM_ID,
        collected_at=NOW,
        destination=destination,
    )

    raw = destination.read_bytes()
    assert raw.endswith(bytes([10]))
    assert not raw.endswith(bytes([92, 110]))
    assert json.loads(raw)["digest"] == artifact.digest
    assert SlackTopologyArtifact.model_validate_json(raw) == artifact


@pytest.mark.asyncio
async def test_topology_artifact_rejects_tampered_aggregate_or_digest() -> None:
    artifact = await collect_slack_topology(
        _FakeTopologySource(),
        expected_team_id=TEAM_ID,
        collected_at=NOW,
    )
    payload = artifact.model_dump(mode="json")

    with pytest.raises(ValidationError, match="aggregate diverges"):
        SlackTopologyArtifact.model_validate({**payload, "conversation_count": 99})
    with pytest.raises(ValidationError, match="artifact digest"):
        SlackTopologyArtifact.model_validate({**payload, "digest": "0" * 64})


def test_topology_cli_has_no_token_or_team_authority_flags() -> None:
    parsed = _parse_arguments(["--output", "topology.json"])
    assert parsed.output == Path("topology.json")

    with pytest.raises(SystemExit):
        _parse_arguments(
            [
                "--output",
                "topology.json",
                "--token",
                "xoxb-forged",
                "--team-id",
                "T-FORGED",
            ]
        )


def test_topology_cli_redacts_runtime_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    async def fail(_arguments: object) -> SlackTopologyArtifact:
        raise SlackTopologyError("xoxb-secret Slack response details")

    monkeypatch.setattr(topology_module, "_run", fail)

    assert topology_module.main(["--output", str(tmp_path / "unused.json")]) == 2
    output = capsys.readouterr().out
    assert "slack_topology_collection_failed" in output
    assert "xoxb-secret" not in output
