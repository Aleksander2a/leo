from __future__ import annotations

import pytest
from pydantic import ValidationError

from leo.domain.conversation import (
    ConversationKind,
    ConversationPolicyError,
    ConversationRef,
    VisibilityNamespace,
    derive_visibility,
    normalize_conversation_kind,
    validate_pinned_thread_event,
)
from leo.domain.conversation_store import ConversationStoreError, InMemoryConversationStore
from leo.harness.models import ScopeKey
from leo.memory.models import MemoryVisibility

SCOPE = ScopeKey(organization_id="org", strategy_id="strategy-a")


def _destination(
    external_id: str,
    kind: ConversationKind = ConversationKind.CHANNEL,
    *,
    actor_id: str | None = None,
) -> ConversationRef:
    return ConversationRef(
        provider="slack",
        team_id="team",
        external_id=external_id,
        kind=kind,
        actor_id=actor_id,
    )


def _thread_namespace(destination: ConversationRef, root_ts: str = "100.1") -> str:
    return f"{destination.provider}:{destination.team_id}:{destination.external_id}:{root_ts}"


def test_channel_a_visibility_is_exact_and_cannot_widen_to_channel_b() -> None:
    channel_a = _destination("A")
    decision = derive_visibility(
        channel_a,
        actor_id="actor",
        current_thread_namespace_id=_thread_namespace(channel_a),
        selected_scope=SCOPE,
    )
    assert decision.allowed_conversation_ids == ("A",)
    assert decision.allowed_namespaces == {
        VisibilityNamespace.THREAD_LOCAL,
        VisibilityNamespace.CONVERSATION_LOCAL,
    }
    with pytest.raises(ConversationPolicyError, match="visibility_context_invalid"):
        derive_visibility(
            channel_a,
            actor_id="actor",
            current_thread_namespace_id=_thread_namespace(channel_a),
            allowed_conversation_ids=("A", "B"),
        )


def test_dm_carries_an_exact_sorted_membership_union_plus_private_and_thread_scope() -> None:
    dm = _destination("D", ConversationKind.DM, actor_id="actor")
    decision = derive_visibility(
        dm,
        actor_id="actor",
        current_thread_namespace_id=_thread_namespace(dm),
        allowed_conversation_ids=("A", "B", "D"),
    )
    assert decision.allowed_conversation_ids == ("A", "B", "D")
    assert decision.allowed_namespaces == {
        VisibilityNamespace.CONVERSATION_LOCAL,
        VisibilityNamespace.ACTOR_PRIVATE,
        VisibilityNamespace.THREAD_LOCAL,
    }
    with pytest.raises(ConversationPolicyError, match="visibility_context_invalid"):
        derive_visibility(
            dm,
            actor_id="actor",
            current_thread_namespace_id=_thread_namespace(dm),
            allowed_conversation_ids=("B", "A", "D"),
        )
    with pytest.raises(ConversationPolicyError, match="visibility_context_invalid"):
        derive_visibility(
            dm,
            actor_id="actor",
            current_thread_namespace_id=_thread_namespace(dm),
            allowed_conversation_ids=("A", "B"),
        )


def test_group_dm_is_exact_and_never_accepts_membership_aggregation() -> None:
    group = _destination("G", ConversationKind.GROUP_DM)
    decision = derive_visibility(
        group,
        actor_id="actor",
        current_thread_namespace_id=_thread_namespace(group),
    )
    assert decision.allowed_conversation_ids == ("G",)
    assert VisibilityNamespace.ACTOR_PRIVATE not in decision.allowed_namespaces
    with pytest.raises(ConversationPolicyError, match="visibility_context_invalid"):
        derive_visibility(
            group,
            actor_id="actor",
            current_thread_namespace_id=_thread_namespace(group),
            allowed_conversation_ids=("A", "G"),
        )


@pytest.mark.parametrize(("raw_kind", "kind"), (("shared", "shared"), ("external", "external")))
def test_shared_and_external_conversations_are_eligible_exact_destinations(
    raw_kind: str,
    kind: str,
) -> None:
    normalized = normalize_conversation_kind(raw_kind)
    assert normalized.value == kind
    destination = _destination(kind.upper(), normalized)
    decision = derive_visibility(
        destination,
        actor_id="actor",
        current_thread_namespace_id=_thread_namespace(destination),
    )
    assert decision.allowed_conversation_ids == (kind.upper(),)
    assert decision.allowed_namespaces == {
        VisibilityNamespace.CONVERSATION_LOCAL,
        VisibilityNamespace.THREAD_LOCAL,
    }


def test_forged_thread_projection_and_destination_shapes_fail_closed() -> None:
    channel_a = _destination("A")
    channel_b = _destination("B")
    with pytest.raises(ConversationPolicyError, match="visibility_context_invalid"):
        derive_visibility(
            channel_a,
            actor_id="actor",
            current_thread_namespace_id=_thread_namespace(channel_b),
        )
    with pytest.raises(ConversationPolicyError, match="unknown_destination"):
        normalize_conversation_kind("not-a-slack-kind")
    with pytest.raises(ValidationError, match="server-derived actor"):
        _destination("D", ConversationKind.DM)


def test_mapping_and_strategy_changes_are_non_gating_but_organization_is_pinned() -> None:
    destination = _destination("A")
    store = InMemoryConversationStore()
    first = store.pin_thread(SCOPE, destination, root_ts="100.1", mapping_version=3)
    remapped_scope = ScopeKey(organization_id="org", strategy_id="strategy-b")
    remapped = store.pin_thread(
        remapped_scope,
        destination,
        root_ts="100.1",
        mapping_version=99,
    )
    assert remapped == first
    assert store.load_thread(remapped_scope, destination, root_ts="100.1") == first
    validate_pinned_thread_event(first, destination, root_ts="100.1", mapping_version=99)

    with pytest.raises(ConversationStoreError, match="thread_organization_changed"):
        store.pin_thread(
            ScopeKey(organization_id="other-org", strategy_id="strategy-a"),
            destination,
            root_ts="100.1",
        )


def test_legacy_kind_and_channel_visibility_values_remain_parseable() -> None:
    assert normalize_conversation_kind("ordinary_internal") is ConversationKind.CHANNEL
    assert VisibilityNamespace("channel_local") is VisibilityNamespace.CHANNEL_LOCAL
    assert MemoryVisibility("channel_local") is MemoryVisibility.CHANNEL_LOCAL
    assert MemoryVisibility("conversation_local") is MemoryVisibility.CONVERSATION_LOCAL
