"""Provider-neutral conversation identity and disclosure policy contracts."""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey


class ConversationKind(StrEnum):
    CHANNEL = "channel"
    DM = "dm"
    GROUP_DM = "group_dm"
    SHARED = "shared"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


class VisibilityNamespace(StrEnum):
    THREAD_LOCAL = "thread_local"
    CONVERSATION_LOCAL = "conversation_local"
    # Deprecated compatibility spelling for persisted pre-D-054 memory rows.
    CHANNEL_LOCAL = "channel_local"
    ACTOR_PRIVATE = "actor_private"
    STRATEGY_SHARED = "strategy_shared"
    ORGANIZATION_SHARED = "organization_shared"


class ConversationRef(ContractModel):
    """Server-normalized destination identity; names and message text are absent by design."""

    provider: NonEmptyStr
    team_id: NonEmptyStr
    external_id: NonEmptyStr
    kind: ConversationKind
    actor_id: str | None = None

    @model_validator(mode="after")
    def validate_destination_shape(self) -> ConversationRef:
        if self.kind is ConversationKind.DM and not self.actor_id:
            raise ValueError("DM destinations require the server-derived actor ID")
        if self.kind is not ConversationKind.DM and self.actor_id is not None:
            raise ValueError("only DM destinations may carry an actor ID")
        if self.kind is ConversationKind.UNKNOWN:
            raise ValueError("unknown destinations cannot be persisted")
        return self


class ThreadRef(ContractModel):
    """A thread pinned to one conversation; domain/mapping data is provenance only."""

    conversation: ConversationRef
    root_ts: NonEmptyStr
    scope: ScopeKey
    mapping_version: int | None = Field(default=None, ge=1)
    version: int = Field(default=1, ge=1)

    @property
    def namespace_id(self) -> str:
        destination = self.conversation
        return (
            f"{destination.provider}:{destination.team_id}:{destination.external_id}:{self.root_ts}"
        )


class VisibilityDecision(ContractModel):
    """Exact server-derived destination projection for one conversation turn."""

    policy_version: NonEmptyStr
    destination: ConversationRef
    actor_id: NonEmptyStr
    selected_scope: ScopeKey | None = None
    allowed_conversation_ids: tuple[NonEmptyStr, ...] = Field(min_length=1, max_length=500)
    current_thread_namespace_id: NonEmptyStr
    allowed_namespaces: frozenset[VisibilityNamespace]

    @model_validator(mode="after")
    def validate_visibility(self) -> VisibilityDecision:
        normalized = tuple(sorted(set(self.allowed_conversation_ids)))
        if normalized != self.allowed_conversation_ids:
            raise ValueError("allowed conversation IDs must be sorted and unique")
        thread_prefix = (
            f"{self.destination.provider}:{self.destination.team_id}:"
            f"{self.destination.external_id}:"
        )
        if not self.current_thread_namespace_id.startswith(thread_prefix) or len(
            self.current_thread_namespace_id
        ) <= len(thread_prefix):
            raise ValueError("current thread namespace must belong to the exact destination")
        if self.destination.kind is ConversationKind.DM:
            if self.destination.actor_id != self.actor_id:
                raise ValueError("DM actor does not match the destination actor")
            expected = {
                VisibilityNamespace.THREAD_LOCAL,
                VisibilityNamespace.CONVERSATION_LOCAL,
                VisibilityNamespace.ACTOR_PRIVATE,
            }
            if self.destination.external_id not in normalized:
                raise ValueError("DM projection must include its exact destination")
        elif self.destination.kind in {
            ConversationKind.CHANNEL,
            ConversationKind.GROUP_DM,
            ConversationKind.SHARED,
            ConversationKind.EXTERNAL,
        }:
            expected = {
                VisibilityNamespace.THREAD_LOCAL,
                VisibilityNamespace.CONVERSATION_LOCAL,
            }
            if normalized != (self.destination.external_id,):
                raise ValueError("shared conversation projection must use the exact destination")
        else:
            raise ValueError("destination kind is not eligible for visibility")
        if self.allowed_namespaces != expected:
            raise ValueError("visibility set does not match the destination policy")
        return self


class ConversationPolicyError(RuntimeError):
    """Safe fail-closed conversation or thread policy outcome."""

    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


def normalize_conversation_kind(raw_kind: str) -> ConversationKind:
    """Map an authoritative adapter classification without accepting provider names as policy."""

    mapping = {
        "ordinary_internal": ConversationKind.CHANNEL,
        "channel": ConversationKind.CHANNEL,
        "dm": ConversationKind.DM,
        "mpim": ConversationKind.GROUP_DM,
        "group_dm": ConversationKind.GROUP_DM,
        "shared": ConversationKind.SHARED,
        "external": ConversationKind.EXTERNAL,
    }
    try:
        kind = mapping[raw_kind]
    except KeyError as exc:
        raise ConversationPolicyError("unknown_destination") from exc
    return kind


def derive_visibility(
    destination: ConversationRef,
    *,
    actor_id: str,
    current_thread_namespace_id: str,
    allowed_conversation_ids: tuple[str, ...] | None = None,
    selected_scope: ScopeKey | None = None,
) -> VisibilityDecision:
    if destination.kind is ConversationKind.UNKNOWN:
        raise ConversationPolicyError("conversation_ineligible")
    try:
        return VisibilityDecision(
            policy_version="conversation-visibility-v2",
            destination=destination,
            actor_id=actor_id,
            selected_scope=selected_scope,
            allowed_conversation_ids=(
                (destination.external_id,)
                if allowed_conversation_ids is None
                else allowed_conversation_ids
            ),
            current_thread_namespace_id=current_thread_namespace_id,
            allowed_namespaces=_default_visibility(destination),
        )
    except ValueError as exc:
        raise ConversationPolicyError("visibility_context_invalid") from exc


def validate_pinned_thread_event(
    thread: ThreadRef,
    destination: ConversationRef,
    *,
    root_ts: str,
    mapping_version: int | None = None,
) -> None:
    del mapping_version
    if thread.conversation != destination or thread.root_ts != root_ts:
        raise ConversationPolicyError("thread_identity_mismatch")


def _default_visibility(
    destination: ConversationRef,
) -> frozenset[VisibilityNamespace]:
    if destination.kind is ConversationKind.DM:
        if destination.actor_id is None:
            raise ValueError("DM actor is missing")
        return frozenset(
            {
                VisibilityNamespace.ACTOR_PRIVATE,
                VisibilityNamespace.CONVERSATION_LOCAL,
                VisibilityNamespace.THREAD_LOCAL,
            }
        )
    if destination.kind in {
        ConversationKind.CHANNEL,
        ConversationKind.GROUP_DM,
        ConversationKind.SHARED,
        ConversationKind.EXTERNAL,
    }:
        return frozenset({VisibilityNamespace.THREAD_LOCAL, VisibilityNamespace.CONVERSATION_LOCAL})
    raise ValueError("destination kind is not eligible")
