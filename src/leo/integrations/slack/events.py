"""Normalize Slack events before they enter Leo-owned routing and persistence."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from leo.harness.models import ScopeKey


class SlackPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class AppMentionEvent(SlackPayload):
    type: Literal["app_mention"]
    user: str = Field(min_length=1)
    text: str
    ts: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    channel_type: str | None = None
    thread_ts: str | None = None
    bot_id: str | None = None
    subtype: str | None = None


class EventCallback(SlackPayload):
    type: Literal["event_callback"]
    team_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event: AppMentionEvent


class MessageImEvent(SlackPayload):
    """The human-message subset of Slack's ``message.im`` event shape."""

    type: Literal["message"]
    channel_type: Literal["im"]
    user: str | None = None
    text: str = ""
    ts: str = Field(min_length=1)
    channel: str = Field(min_length=1)
    thread_ts: str | None = None
    bot_id: str | None = None
    subtype: str | None = None


class MessageImCallback(SlackPayload):
    type: Literal["event_callback"]
    team_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    event: MessageImEvent


class PassiveMessageEvent(SlackPayload):
    """A channel/private-channel/MPIM message observed without launching Leo."""

    type: Literal["message", "app_mention"]
    # app_mention payloads from some Slack connector paths omit channel_type;
    # conversation eligibility is re-derived from conversations.info before launch.
    channel_type: Literal["channel", "group", "mpim"] | None = None
    user: str | None = None
    text: str = ""
    ts: str = Field(min_length=1, max_length=64)
    channel: str = Field(min_length=1)
    thread_ts: str | None = None
    bot_id: str | None = None
    app_id: str | None = None
    subtype: str | None = None


class PassiveMessageCallback(SlackPayload):
    type: Literal["event_callback"]
    team_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    api_app_id: str | None = None
    event: PassiveMessageEvent


class SlackAdmissionPolicyRejected(RuntimeError):
    """Expected channel-policy rejection that must not invoke Leo's runtime."""

    safe_code: str


class SlackConversationKind(StrEnum):
    ORDINARY_INTERNAL = "ordinary_internal"
    DM = "dm"
    MPIM = "mpim"
    SHARED = "shared"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class SlackTriggerKind(StrEnum):
    APP_MENTION = "app_mention"
    MESSAGE_IM = "message_im"


class SlackPassiveMessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SlackPassiveMessage(SlackPayload):
    """One non-triggering Slack message safe to persist in the conversation plane."""

    event_id: str
    team_id: str
    channel_id: str
    actor_id: str
    role: SlackPassiveMessageRole
    message_ts: str
    thread_root_ts: str
    text: str
    conversation_kind: SlackConversationKind


class SlackContextProjectionSource(StrEnum):
    """Trusted provenance for the exact context conversation projection."""

    EXACT_DESTINATION = "exact_destination"
    DM_MEMBERSHIP_INTERSECTION = "dm_membership_intersection"
    DM_ONLY_FALLBACK = "dm_only_fallback"


class SlackBotPresence(StrEnum):
    """Authoritative observation of Leo's presence in a Slack conversation."""

    PRESENT = "present"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class SlackConversationLifecycle(StrEnum):
    """Lifecycle state relevant to accepting new work."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    LEFT = "left"
    UNKNOWN = "unknown"


class SlackExternalProvenance(StrEnum):
    """Slack-derived shared/external provenance, independent of domain metadata."""

    INTERNAL = "internal"
    SHARED = "shared"
    EXTERNAL = "external"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


SLACK_MEMBERSHIP_POLICY_VERSION = 1


class SlackConversationEligibility(SlackPayload):
    """Provider-neutral conversation classification with explicit provenance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: SlackConversationKind
    provenance: Literal["slack_conversations_info", "slack_event", "unknown"]
    bot_presence: SlackBotPresence = SlackBotPresence.UNKNOWN
    lifecycle: SlackConversationLifecycle = SlackConversationLifecycle.UNKNOWN
    external_provenance: SlackExternalProvenance = SlackExternalProvenance.UNKNOWN
    membership_policy_version: int = Field(
        default=SLACK_MEMBERSHIP_POLICY_VERSION,
        ge=1,
    )

    @property
    def admissible(self) -> bool:
        """Whether Slack authoritatively established a conversation Leo can receive."""

        if (
            self.kind is SlackConversationKind.UNKNOWN
            or self.provenance not in {"slack_conversations_info", "slack_event"}
            or self.bot_presence is not SlackBotPresence.PRESENT
            or self.lifecycle is not SlackConversationLifecycle.ACTIVE
        ):
            return False
        if self.kind is SlackConversationKind.SHARED:
            return self.external_provenance is SlackExternalProvenance.SHARED
        if self.kind is SlackConversationKind.EXTERNAL:
            return self.external_provenance is SlackExternalProvenance.EXTERNAL
        if self.kind is SlackConversationKind.ORDINARY_INTERNAL:
            return self.external_provenance in {
                SlackExternalProvenance.INTERNAL,
                SlackExternalProvenance.UNKNOWN,
            }
        return self.external_provenance in {
            SlackExternalProvenance.INTERNAL,
            SlackExternalProvenance.SHARED,
            SlackExternalProvenance.EXTERNAL,
            SlackExternalProvenance.NOT_APPLICABLE,
            SlackExternalProvenance.UNKNOWN,
        }

    @property
    def eligible_for_scope_provision(self) -> bool:
        """Compatibility alias; scope mappings no longer control availability."""

        return self.admissible


class _SlackConversationInfo(SlackPayload):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str = Field(min_length=1)
    is_channel: bool = False
    is_group: bool = False
    is_im: bool = False
    is_mpim: bool = False
    is_shared: bool = False
    is_org_shared: bool = False
    is_ext_shared: bool = False
    is_archived: bool = False
    is_member: bool | None = None


class _SlackConversationInfoResponse(SlackPayload):
    model_config = ConfigDict(extra="ignore", frozen=True)

    ok: bool
    channel: _SlackConversationInfo | None = None


class SlackConversationPolicyRejected(SlackAdmissionPolicyRejected):
    """Authoritative Slack metadata does not permit scope provisioning."""

    safe_code = "conversation_ineligible"

    def __init__(self, eligibility: SlackConversationEligibility) -> None:
        self.eligibility = eligibility
        super().__init__(f"Slack conversation is not eligible: {eligibility.kind.value}")


def classify_slack_conversation(
    body: object,
    *,
    expected_channel_id: str,
) -> SlackConversationEligibility:
    """Classify only a matching authoritative conversations.info response."""

    try:
        response = _SlackConversationInfoResponse.model_validate(body)
    except (ValidationError, ValueError):
        return SlackConversationEligibility(
            kind=SlackConversationKind.UNKNOWN,
            provenance="unknown",
        )
    channel = response.channel
    if not response.ok or channel is None or channel.id != expected_channel_id:
        return SlackConversationEligibility(
            kind=SlackConversationKind.UNKNOWN,
            provenance="unknown",
        )

    # Slack's real MPIM payloads may set both is_mpim and is_channel.  Treat
    # is_im/is_mpim as the higher-precedence conversation types while retaining
    # fail-closed rejection for genuinely contradictory flags.
    if channel.is_im:
        flags_are_consistent = not (channel.is_channel or channel.is_group or channel.is_mpim)
    elif channel.is_mpim:
        flags_are_consistent = not channel.is_group
    else:
        flags_are_consistent = channel.is_channel is not channel.is_group
    if not flags_are_consistent:
        return SlackConversationEligibility(
            kind=SlackConversationKind.UNKNOWN,
            provenance="slack_conversations_info",
            bot_presence=(
                SlackBotPresence.ABSENT if channel.is_member is False else SlackBotPresence.UNKNOWN
            ),
            lifecycle=(
                SlackConversationLifecycle.ARCHIVED
                if channel.is_archived
                else SlackConversationLifecycle.ACTIVE
            ),
        )
    if channel.is_im:
        kind = SlackConversationKind.DM
    elif channel.is_mpim:
        kind = SlackConversationKind.MPIM
    elif channel.is_ext_shared:
        kind = SlackConversationKind.EXTERNAL
    elif channel.is_shared or channel.is_org_shared:
        kind = SlackConversationKind.SHARED
    else:
        kind = SlackConversationKind.ORDINARY_INTERNAL
    external_provenance = (
        SlackExternalProvenance.EXTERNAL
        if channel.is_ext_shared
        else SlackExternalProvenance.SHARED
        if channel.is_shared or channel.is_org_shared
        else SlackExternalProvenance.NOT_APPLICABLE
        if kind in {SlackConversationKind.DM, SlackConversationKind.MPIM}
        else SlackExternalProvenance.INTERNAL
    )
    return SlackConversationEligibility(
        kind=kind,
        provenance="slack_conversations_info",
        bot_presence=(
            SlackBotPresence.PRESENT
            if channel.is_member is True
            else SlackBotPresence.ABSENT
            if channel.is_member is False
            else SlackBotPresence.UNKNOWN
        ),
        lifecycle=(
            SlackConversationLifecycle.ARCHIVED
            if channel.is_archived
            else SlackConversationLifecycle.ACTIVE
        ),
        external_provenance=external_provenance,
    )


class SlackMentionJob(BaseModel):
    """Normalized Slack envelope; it is not runnable until admission adds trusted scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    team_id: str
    channel_id: str
    user_id: str
    message_ts: str
    thread_root_ts: str
    conversation_key: str
    prompt: str
    conversation_kind: SlackConversationKind
    trigger_kind: SlackTriggerKind
    context_conversation_ids: tuple[str, ...]
    context_projection_source: SlackContextProjectionSource = (
        SlackContextProjectionSource.EXACT_DESTINATION
    )
    conversation_authority_source: Literal["slack_conversations_info", "slack_event", "unknown"] = (
        "unknown"
    )
    bot_presence: SlackBotPresence = SlackBotPresence.UNKNOWN
    conversation_lifecycle: SlackConversationLifecycle = SlackConversationLifecycle.UNKNOWN
    external_provenance: SlackExternalProvenance = SlackExternalProvenance.UNKNOWN
    membership_policy_version: int = Field(
        default=SLACK_MEMBERSHIP_POLICY_VERSION,
        ge=1,
    )
    context_access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_context_projection(self) -> SlackMentionJob:
        for name, value in (
            ("team_id", self.team_id),
            ("channel_id", self.channel_id),
            ("user_id", self.user_id),
        ):
            if not value or value != value.strip() or len(value) > 32:
                raise ValueError(f"{name} must be a valid Slack ID")
        projection = self.context_conversation_ids
        if not projection:
            raise ValueError("context_conversation_ids must not be empty")
        if any(
            not conversation_id
            or conversation_id != conversation_id.strip()
            or len(conversation_id) > 32
            for conversation_id in projection
        ):
            raise ValueError("context_conversation_ids must contain valid Slack IDs")
        if projection != tuple(sorted(set(projection))):
            raise ValueError("context_conversation_ids must be sorted and unique")
        if self.channel_id not in projection:
            raise ValueError("context_conversation_ids must include the current conversation")
        if self.conversation_kind is not SlackConversationKind.DM and projection != (
            self.channel_id,
        ):
            raise ValueError("non-DM context must be restricted to the current conversation")
        if self.conversation_kind is not SlackConversationKind.DM:
            if self.context_projection_source is not SlackContextProjectionSource.EXACT_DESTINATION:
                raise ValueError("non-DM context must use exact-destination provenance")
        elif self.context_projection_source is SlackContextProjectionSource.DM_ONLY_FALLBACK:
            if projection != (self.channel_id,):
                raise ValueError("DM-only fallback must be restricted to the current DM")
        if (
            self.trigger_kind is SlackTriggerKind.MESSAGE_IM
            and self.conversation_kind is not SlackConversationKind.DM
        ):
            raise ValueError("message_im triggers must belong to a DM")
        expected_hash = build_context_access_hash(
            team_id=self.team_id,
            user_id=self.user_id,
            channel_id=self.channel_id,
            context_conversation_ids=projection,
        )
        if self.context_access_hash != expected_hash:
            raise ValueError("context_access_hash does not match the access projection")
        return self


class SlackScopeResolution(BaseModel):
    """Immutable authority snapshot committed during Slack ingress admission."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scope: ScopeKey
    mapping_version: int = Field(ge=1)
    provisioned: bool


@dataclass(frozen=True, slots=True)
class AdmittedSlackMention:
    """The only Slack mention contract that a Leo runtime may execute."""

    job: SlackMentionJob
    resolution: SlackScopeResolution
    launch: SlackLaunchRef | None = None


@dataclass(frozen=True, slots=True)
class SlackLaunchRef:
    """Durable identity assigned before an admitted mention enters a wake-up queue."""

    thread_id: str
    task_id: str
    run_id: str


class SlackEventRejected(ValueError):
    pass


def normalize_app_mention(
    body: object,
    *,
    expected_team_id: str,
    bot_user_id: str,
) -> SlackMentionJob | None:
    callback = EventCallback.model_validate(body)
    event = callback.event
    if callback.team_id != expected_team_id:
        raise SlackEventRejected("event came from an unconfigured Slack team")
    if event.bot_id is not None or event.subtype is not None or event.user == bot_user_id:
        return None

    thread_root_ts = event.thread_ts or event.ts
    prompt = _strip_connector_attribution(_strip_bot_mention(event.text, bot_user_id))
    if not prompt:
        prompt = "help"
    conversation_key = f"slack:{callback.team_id}:{event.channel}:{thread_root_ts}"
    return SlackMentionJob(
        event_id=callback.event_id,
        team_id=callback.team_id,
        channel_id=event.channel,
        user_id=event.user,
        message_ts=event.ts,
        thread_root_ts=thread_root_ts,
        conversation_key=conversation_key,
        prompt=prompt,
        conversation_kind=_event_conversation_kind(event.channel_type),
        trigger_kind=SlackTriggerKind.APP_MENTION,
        context_conversation_ids=(event.channel,),
        conversation_authority_source="slack_event",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=_event_external_provenance(event.channel_type),
        context_access_hash=build_context_access_hash(
            team_id=callback.team_id,
            user_id=event.user,
            channel_id=event.channel,
            context_conversation_ids=(event.channel,),
        ),
    )


def normalize_message_im(
    body: object,
    *,
    expected_team_id: str,
    bot_user_id: str,
) -> SlackMentionJob | None:
    """Normalize one human DM message; bot, self, and subtype events are ignored."""

    callback = MessageImCallback.model_validate(body)
    event = callback.event
    if callback.team_id != expected_team_id:
        raise SlackEventRejected("event came from an unconfigured Slack team")
    if (
        event.user is None
        or event.bot_id is not None
        or event.subtype is not None
        or event.user == bot_user_id
    ):
        return None

    thread_root_ts = event.thread_ts or event.ts
    prompt = _strip_connector_attribution(event.text)
    if not prompt:
        prompt = "help"
    conversation_key = f"slack:{callback.team_id}:{event.channel}:{thread_root_ts}"
    return SlackMentionJob(
        event_id=callback.event_id,
        team_id=callback.team_id,
        channel_id=event.channel,
        user_id=event.user,
        message_ts=event.ts,
        thread_root_ts=thread_root_ts,
        conversation_key=conversation_key,
        prompt=prompt,
        conversation_kind=SlackConversationKind.DM,
        trigger_kind=SlackTriggerKind.MESSAGE_IM,
        context_conversation_ids=(event.channel,),
        conversation_authority_source="slack_event",
        bot_presence=SlackBotPresence.PRESENT,
        conversation_lifecycle=SlackConversationLifecycle.ACTIVE,
        external_provenance=SlackExternalProvenance.NOT_APPLICABLE,
        context_access_hash=build_context_access_hash(
            team_id=callback.team_id,
            user_id=event.user,
            channel_id=event.channel,
            context_conversation_ids=(event.channel,),
        ),
    )


def normalize_passive_message(
    body: object,
    *,
    expected_team_id: str,
    bot_user_id: str,
    bot_id: str | None = None,
    allow_mentioned_user_message: bool = False,
) -> SlackPassiveMessage | None:
    """Normalize passive channel/group/MPIM messages without creating a run trigger."""

    callback = PassiveMessageCallback.model_validate(body)
    event = callback.event
    if callback.team_id != expected_team_id:
        raise SlackEventRejected("event came from an unconfigured Slack team")
    if event.channel_type is None and not allow_mentioned_user_message:
        raise SlackEventRejected("passive Slack message omitted channel type")

    # Edits, deletes, thread broadcasts, and other Slack subtypes never become a
    # second immutable message-plane row. Other bots are also excluded. Slack may
    # omit ``user`` from bot_message events, so Leo is recognized by any exact,
    # trusted identity available on the Socket Mode envelope: bot user, auth.test
    # bot ID, or the event app ID matching this callback's api_app_id.
    app_identity_matches = bool(
        event.app_id and callback.api_app_id and event.app_id == callback.api_app_id
    )
    bot_identity_matches = bool(bot_id and event.bot_id and event.bot_id == bot_id)
    user_identity_matches = event.user == bot_user_id
    if user_identity_matches:
        if bot_id and event.bot_id and event.bot_id != bot_id:
            return None
        if callback.api_app_id and event.app_id and not app_identity_matches:
            return None
    is_leo_message = user_identity_matches or (
        event.subtype == "bot_message" and (bot_identity_matches or app_identity_matches)
    )
    if is_leo_message:
        if event.subtype not in {None, "bot_message"}:
            return None
        role = SlackPassiveMessageRole.ASSISTANT
    else:
        if event.user is None or event.bot_id is not None or event.subtype is not None:
            return None
        if re.search(rf"<@{re.escape(bot_user_id)}>", event.text) and not (
            allow_mentioned_user_message
        ):
            # The app_mention callback is the single persistence/launch authority for
            # mentioned messages, preventing duplicate rows across subscriptions.
            return None
        role = SlackPassiveMessageRole.USER

    text = _strip_connector_attribution(event.text).strip()
    if not text:
        return None
    thread_root_ts = event.thread_ts or event.ts
    return SlackPassiveMessage(
        event_id=callback.event_id,
        team_id=callback.team_id,
        channel_id=event.channel,
        actor_id=event.user or bot_user_id,
        role=role,
        message_ts=event.ts,
        thread_root_ts=thread_root_ts,
        text=text,
        conversation_kind=(
            SlackConversationKind.MPIM
            if event.channel_type == "mpim"
            else SlackConversationKind.ORDINARY_INTERNAL
        ),
    )


def build_context_access_hash(
    *,
    team_id: str,
    user_id: str,
    channel_id: str,
    context_conversation_ids: tuple[str, ...],
) -> str:
    """Fingerprint the actor/destination/projection tuple for caching and audit."""

    material = "\x1f".join(
        ("slack-context-v1", team_id, user_id, channel_id, *context_conversation_ids)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _event_conversation_kind(channel_type: str | None) -> SlackConversationKind:
    if channel_type == "im":
        return SlackConversationKind.DM
    if channel_type == "mpim":
        return SlackConversationKind.MPIM
    if channel_type in {"channel", "group"}:
        return SlackConversationKind.ORDINARY_INTERNAL
    return SlackConversationKind.UNKNOWN


def _event_external_provenance(channel_type: str | None) -> SlackExternalProvenance:
    if channel_type in {"im", "mpim"}:
        return SlackExternalProvenance.NOT_APPLICABLE
    return SlackExternalProvenance.UNKNOWN


def _strip_bot_mention(text: str, bot_user_id: str) -> str:
    mention = re.compile(rf"<@{re.escape(bot_user_id)}>")
    return " ".join(mention.sub(" ", text).split())


_CONNECTOR_ATTRIBUTION = re.compile(
    r"\s+\*Sent using\*\s+<@[A-Z0-9]+>\s*$",
    re.IGNORECASE,
)


def _strip_connector_attribution(text: str) -> str:
    """Remove Slack's connector attribution while retaining strict command text."""

    return _CONNECTOR_ATTRIBUTION.sub("", text).strip()
