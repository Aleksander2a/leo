from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.integrations.slack.events import (
    SlackBotPresence,
    SlackContextProjectionSource,
    SlackConversationEligibility,
    SlackConversationKind,
    SlackConversationLifecycle,
    SlackExternalProvenance,
    SlackMentionJob,
    SlackPassiveMessageRole,
    SlackTriggerKind,
    build_context_access_hash,
    classify_slack_conversation,
    normalize_app_mention,
    normalize_message_im,
    normalize_passive_message,
)


def _body(
    *,
    thread_ts: str | None = None,
    channel: str = "C123",
    text: str = "<@UBOT>  quote NVDA ",
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "app_mention",
        "user": "U123",
        "text": text,
        "ts": "100.001",
        "channel": channel,
        "channel_type": "channel",
    }
    if thread_ts is not None:
        event["thread_ts"] = thread_ts
    return {
        "type": "event_callback",
        "team_id": "T123",
        "event_id": "Ev123",
        "event": event,
    }


def test_top_level_mention_creates_root_thread_key() -> None:
    job = normalize_app_mention(
        _body(),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )
    assert job is not None
    assert job.thread_root_ts == "100.001"
    assert job.conversation_key == "slack:T123:C123:100.001"
    assert job.prompt == "quote NVDA"
    assert job.conversation_kind is SlackConversationKind.ORDINARY_INTERNAL
    assert job.trigger_kind is SlackTriggerKind.APP_MENTION
    assert job.context_conversation_ids == ("C123",)
    assert job.context_projection_source is SlackContextProjectionSource.EXACT_DESTINATION


def test_arbitrary_mentioned_message_is_preserved_for_reasoning() -> None:
    job = normalize_app_mention(
        _body(text="<@UBOT> Compare our thesis with the latest filings and flag risks"),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )

    assert job is not None
    assert job.prompt == "Compare our thesis with the latest filings and flag risks"


def test_threaded_mention_preserves_root() -> None:
    job = normalize_app_mention(
        _body(thread_ts="99.999"),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )
    assert job is not None
    assert job.thread_root_ts == "99.999"


def test_any_channel_in_the_configured_team_is_allowed() -> None:
    job = normalize_app_mention(
        _body(channel="C999"),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )
    assert job is not None
    assert job.channel_id == "C999"


def test_connector_attribution_is_not_part_of_the_user_prompt() -> None:
    job = normalize_app_mention(
        _body(text="<@UBOT> quote NVDA *Sent using* <@UCONNECTOR>"),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )

    assert job is not None
    assert job.prompt == "quote NVDA"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body["event"].pop("user"),
        lambda body: body["event"].pop("channel"),
        lambda body: body["event"].update({"type": "message"}),
    ],
)
def test_malformed_mentions_are_rejected_before_admission(mutate) -> None:  # type: ignore[no-untyped-def]
    body = _body()
    mutate(body)

    with pytest.raises(ValidationError):
        normalize_app_mention(body, expected_team_id="T123", bot_user_id="UBOT")


def test_wrong_team_mentions_are_rejected_before_admission() -> None:
    with pytest.raises(ValueError, match="unconfigured Slack team"):
        normalize_app_mention(_body(), expected_team_id="T999", bot_user_id="UBOT")


@pytest.mark.parametrize(
    "event_update",
    [
        {"bot_id": "B123"},
        {"user": "UBOT"},
        {"subtype": "message_changed"},
    ],
)
def test_bot_self_and_unsupported_subtype_mentions_are_ignored(event_update) -> None:  # type: ignore[no-untyped-def]
    body = _body()
    body["event"].update(event_update)

    assert normalize_app_mention(body, expected_team_id="T123", bot_user_id="UBOT") is None


@pytest.mark.parametrize(
    ("kind", "eligible"),
    [
        (SlackConversationKind.ORDINARY_INTERNAL, True),
        (SlackConversationKind.DM, True),
        (SlackConversationKind.MPIM, True),
        (SlackConversationKind.SHARED, True),
        (SlackConversationKind.EXTERNAL, True),
        (SlackConversationKind.UNKNOWN, False),
    ],
)
def test_all_authoritative_recognized_conversations_are_admissible(
    kind: SlackConversationKind,
    eligible: bool,
) -> None:
    classification = SlackConversationEligibility(
        kind=kind,
        provenance="slack_conversations_info",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=(
            SlackExternalProvenance.SHARED
            if kind is SlackConversationKind.SHARED
            else SlackExternalProvenance.EXTERNAL
            if kind is SlackConversationKind.EXTERNAL
            else SlackExternalProvenance.NOT_APPLICABLE
            if kind in {SlackConversationKind.DM, SlackConversationKind.MPIM}
            else SlackExternalProvenance.INTERNAL
        ),
    )

    assert classification.eligible_for_scope_provision is eligible


def test_received_slack_event_is_authoritative_for_a_recognized_kind() -> None:
    classification = SlackConversationEligibility(
        kind=SlackConversationKind.ORDINARY_INTERNAL,
        provenance="slack_event",
        bot_presence=SlackBotPresence.PRESENT,
        lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.UNKNOWN,
    )

    assert classification.admissible is True


def test_conversation_eligibility_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="extra"):
        SlackConversationEligibility(
            kind=SlackConversationKind.UNKNOWN,
            provenance="unknown",
            channel_name="demo",
        )


@pytest.mark.parametrize(
    ("channel", "expected"),
    [
        ({"is_channel": True}, SlackConversationKind.ORDINARY_INTERNAL),
        ({"is_im": True}, SlackConversationKind.DM),
        ({"is_mpim": True}, SlackConversationKind.MPIM),
        ({"is_channel": True, "is_shared": True}, SlackConversationKind.SHARED),
        ({"is_channel": True, "is_ext_shared": True}, SlackConversationKind.EXTERNAL),
        ({}, SlackConversationKind.UNKNOWN),
    ],
)
def test_authoritative_conversation_metadata_maps_to_one_explicit_kind(
    channel: dict[str, object],
    expected: SlackConversationKind,
) -> None:
    body = {"ok": True, "channel": {"id": "C123", "is_member": True, **channel}}

    classification = classify_slack_conversation(body, expected_channel_id="C123")

    assert classification.kind is expected
    assert classification.provenance == "slack_conversations_info"
    if expected is not SlackConversationKind.UNKNOWN:
        assert classification.bot_presence is SlackBotPresence.PRESENT
        assert classification.lifecycle is SlackConversationLifecycle.ACTIVE
        assert classification.admissible is True


def test_real_slack_mpim_metadata_takes_precedence_over_is_channel() -> None:
    classification = classify_slack_conversation(
        {
            "ok": True,
            "channel": {
                "id": "C0BRV1MQMAS",
                "is_channel": True,
                "is_mpim": True,
                "is_private": True,
                "is_member": True,
            },
        },
        expected_channel_id="C0BRV1MQMAS",
    )

    assert classification.kind is SlackConversationKind.MPIM
    assert classification.bot_presence is SlackBotPresence.PRESENT
    assert classification.external_provenance is SlackExternalProvenance.NOT_APPLICABLE
    assert classification.admissible is True


@pytest.mark.parametrize(
    "channel",
    (
        {"is_im": True, "is_mpim": True},
        {"is_im": True, "is_group": True},
        {"is_mpim": True, "is_group": True},
        {"is_channel": True, "is_group": True},
    ),
)
def test_contradictory_conversation_kind_flags_fail_closed(
    channel: dict[str, object],
) -> None:
    classification = classify_slack_conversation(
        {"ok": True, "channel": {"id": "C123", "is_member": True, **channel}},
        expected_channel_id="C123",
    )

    assert classification.kind is SlackConversationKind.UNKNOWN
    assert classification.admissible is False


@pytest.mark.parametrize(
    ("metadata", "presence", "lifecycle"),
    [
        ({"is_member": False}, SlackBotPresence.ABSENT, SlackConversationLifecycle.ACTIVE),
        ({"is_archived": True}, SlackBotPresence.PRESENT, SlackConversationLifecycle.ARCHIVED),
    ],
)
def test_absent_or_archived_conversation_is_not_admissible(
    metadata: dict[str, object],
    presence: SlackBotPresence,
    lifecycle: SlackConversationLifecycle,
) -> None:
    classification = classify_slack_conversation(
        {
            "ok": True,
            "channel": {
                "id": "C123",
                "is_channel": True,
                "is_member": True,
                **metadata,
            },
        },
        expected_channel_id="C123",
    )

    assert classification.bot_presence is presence
    assert classification.lifecycle is lifecycle
    assert classification.admissible is False


def test_conversation_metadata_mismatch_is_unknown_and_ineligible() -> None:
    classification = classify_slack_conversation(
        {"ok": True, "channel": {"id": "C999", "is_channel": True}},
        expected_channel_id="C123",
    )

    assert classification.kind is SlackConversationKind.UNKNOWN
    assert classification.eligible_for_scope_provision is False


def _message_im_body(
    *,
    event_update: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "message",
        "channel_type": "im",
        "user": "U123",
        "text": "Can you summarize my portfolio context?",
        "ts": "200.001",
        "channel": "D123",
    }
    if event_update:
        event.update(event_update)
    return {
        "type": "event_callback",
        "team_id": "T123",
        "event_id": "Ev-im-1",
        "event": event,
    }


def test_human_message_im_is_a_dm_reasoning_job() -> None:
    job = normalize_message_im(
        _message_im_body(),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )

    assert job is not None
    assert job.prompt == "Can you summarize my portfolio context?"
    assert job.conversation_kind is SlackConversationKind.DM
    assert job.trigger_kind is SlackTriggerKind.MESSAGE_IM
    assert job.context_conversation_ids == ("D123",)


@pytest.mark.parametrize(
    "event_update",
    [
        {"bot_id": "B123", "subtype": "bot_message", "user": None},
        {"user": "UBOT"},
        {"subtype": "message_changed"},
    ],
)
def test_message_im_bot_self_and_subtype_events_are_ignored(
    event_update: dict[str, object],
) -> None:
    assert (
        normalize_message_im(
            _message_im_body(event_update=event_update),
            expected_team_id="T123",
            bot_user_id="UBOT",
        )
        is None
    )


def test_non_im_message_cannot_enter_the_dm_handler() -> None:
    with pytest.raises(ValidationError):
        normalize_message_im(
            _message_im_body(event_update={"channel_type": "channel"}),
            expected_team_id="T123",
            bot_user_id="UBOT",
        )


def _passive_message_body(
    *,
    channel_type: str = "channel",
    event_update: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "message",
        "channel_type": channel_type,
        "user": "U123",
        "text": "A useful thread fact.",
        "ts": "300.001",
        "channel": "C123",
    }
    if event_update:
        event.update(event_update)
    return {
        "type": "event_callback",
        "team_id": "T123",
        "event_id": "Ev-passive-1",
        "api_app_id": "A-LEO",
        "event": event,
    }


def test_passive_channel_root_is_context_only_and_cannot_attest_coverage() -> None:
    message = normalize_passive_message(
        _passive_message_body(),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )

    assert message is not None
    assert message.role is SlackPassiveMessageRole.USER
    assert message.thread_root_ts == message.message_ts == "300.001"
    assert "authoritative_reply_count" not in message.model_dump()
    assert message.conversation_kind is SlackConversationKind.ORDINARY_INTERNAL


@pytest.mark.parametrize(
    ("channel_type", "expected_kind"),
    [
        ("group", SlackConversationKind.ORDINARY_INTERNAL),
        ("mpim", SlackConversationKind.MPIM),
    ],
)
def test_passive_private_and_mpim_replies_preserve_exact_root(
    channel_type: str,
    expected_kind: SlackConversationKind,
) -> None:
    message = normalize_passive_message(
        _passive_message_body(
            channel_type=channel_type,
            event_update={"thread_ts": "299.900"},
        ),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )

    assert message is not None
    assert message.thread_root_ts == "299.900"
    assert message.conversation_kind is expected_kind


def test_passive_leo_bot_message_is_assistant_context_not_a_trigger() -> None:
    message = normalize_passive_message(
        _passive_message_body(
            event_update={
                "user": "UBOT",
                "bot_id": "B-LEO",
                "subtype": "bot_message",
                "thread_ts": "299.900",
                "text": "Leo's delivered answer.",
            }
        ),
        expected_team_id="T123",
        bot_user_id="UBOT",
        bot_id="B-LEO",
    )

    assert message is not None
    assert message.role is SlackPassiveMessageRole.ASSISTANT
    assert message.actor_id == "UBOT"
    assert message.text == "Leo's delivered answer."


@pytest.mark.parametrize(
    "event_update",
    [
        {
            "user": None,
            "bot_id": "B-LEO",
            "app_id": "A-LEO",
            "subtype": "bot_message",
        },
        {
            "user": None,
            "bot_id": None,
            "app_id": "A-LEO",
            "subtype": "bot_message",
        },
    ],
)
def test_passive_leo_bot_message_without_user_uses_exact_bot_or_app_identity(
    event_update: dict[str, object],
) -> None:
    message = normalize_passive_message(
        _passive_message_body(event_update=event_update),
        expected_team_id="T123",
        bot_user_id="UBOT",
        bot_id="B-LEO",
    )

    assert message is not None
    assert message.role is SlackPassiveMessageRole.ASSISTANT
    assert message.actor_id == "UBOT"


@pytest.mark.parametrize(
    "event_update",
    [
        {
            "user": None,
            "bot_id": "B-OTHER",
            "app_id": "A-OTHER",
            "subtype": "bot_message",
        },
        {
            "user": "UBOT",
            "bot_id": "B-OTHER",
            "app_id": "A-OTHER",
            "subtype": "bot_message",
        },
    ],
)
def test_passive_foreign_bot_cannot_spoof_leo_assistant_context(
    event_update: dict[str, object],
) -> None:
    assert (
        normalize_passive_message(
            _passive_message_body(event_update=event_update),
            expected_team_id="T123",
            bot_user_id="UBOT",
            bot_id="B-LEO",
        )
        is None
    )


@pytest.mark.parametrize(
    "event_update",
    [
        {"subtype": "message_changed"},
        {"subtype": "message_deleted", "text": ""},
        {"subtype": "thread_broadcast"},
        {"bot_id": "B-OTHER", "subtype": "bot_message", "user": "U-OTHER-BOT"},
        {"text": "<@UBOT> duplicate app mention"},
    ],
)
def test_passive_edits_deletes_other_bots_and_mention_duplicates_are_ignored(
    event_update: dict[str, object],
) -> None:
    assert (
        normalize_passive_message(
            _passive_message_body(event_update=event_update),
            expected_team_id="T123",
            bot_user_id="UBOT",
        )
        is None
    )


def test_dm_message_cannot_enter_passive_context_handler() -> None:
    with pytest.raises(ValidationError):
        normalize_passive_message(
            _passive_message_body(channel_type="im"),
            expected_team_id="T123",
            bot_user_id="UBOT",
        )


def test_non_dm_projection_rejects_dm_membership_provenance() -> None:
    job = normalize_app_mention(
        _body(),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )
    assert job is not None

    with pytest.raises(ValidationError, match="exact-destination provenance"):
        SlackMentionJob.model_validate(
            {
                **job.model_dump(mode="python"),
                "context_projection_source": (
                    SlackContextProjectionSource.DM_MEMBERSHIP_INTERSECTION
                ),
            }
        )


def test_dm_only_fallback_cannot_claim_an_expanded_projection() -> None:
    job = normalize_message_im(
        _message_im_body(),
        expected_team_id="T123",
        bot_user_id="UBOT",
    )
    assert job is not None
    projection = ("C123", "D123")

    with pytest.raises(ValidationError, match="current DM"):
        SlackMentionJob.model_validate(
            {
                **job.model_dump(mode="python"),
                "context_conversation_ids": projection,
                "context_projection_source": SlackContextProjectionSource.DM_ONLY_FALLBACK,
                "context_access_hash": build_context_access_hash(
                    team_id=job.team_id,
                    user_id=job.user_id,
                    channel_id=job.channel_id,
                    context_conversation_ids=projection,
                ),
            }
        )
