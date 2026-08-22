"""Server-bound explicit memory mutation tools for admitted Slack turns."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum

from pydantic import Field, JsonValue, ValidationError, model_validator

from leo.domain.conversation import ConversationKind, ConversationRef
from leo.harness.models import (
    ContractModel,
    NonEmptyStr,
    RunPhase,
    ScopeKey,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRetryPolicy,
    ToolSpec,
    ToolSuccess,
)
from leo.harness.ports import Clock
from leo.harness.store_errors import StoreError
from leo.memory.models import MemoryKind, MemorySource, MemoryVisibility
from leo.memory.service import ExplicitMemoryService, MemoryCandidate, MemoryCommandRejected

_RECORD_ID = r"[A-Za-z0-9](?:[A-Za-z0-9._:-]{0,126}[A-Za-z0-9])?"
_MEMORY_SCOPE_PHRASE = (
    r"(?:for|in)\s+(?:(?:this|the\s+current|our|my)\s+)?"
    r"(?:conversation|channel|group(?:\s+dm)?|dm|direct\s+message|chat|thread)"
)
_REMEMBER_PATTERN = re.compile(
    rf"(?:"
    rf"(?P<imperative>(?:please\s+)?remember)"
    rf"|(?P<modal>can|could|would)\s+you\s+(?:please\s+)?remember"
    rf")"
    rf"(?:\s+(?P<scope>{_MEMORY_SCOPE_PHRASE}))?"
    rf"(?:\s+(?P<that>that))?"
    rf"\s*[:,]?\s+(?P<content>.+)",
    re.IGNORECASE | re.DOTALL,
)
_RECALL_COMPLEMENT_PATTERN = re.compile(
    r"^(?:what|when|where|who|why|how|whether|if)\b",
    re.IGNORECASE,
)
_DECLARATIVE_COMPLEMENT_PATTERN = re.compile(
    r"\b(?:is|are|was|were|has|have|will|should|must|uses?|prefers?|equals?|means?)\b",
    re.IGNORECASE,
)
_CORRECT_PATTERN = re.compile(
    rf"correct\s+memory\s+({_RECORD_ID})\s+(?:to|with)\s+(.+)",
    re.IGNORECASE | re.DOTALL,
)
_FORGET_PATTERN = re.compile(
    rf"forget\s+memory\s+({_RECORD_ID})(?:\s+because\s+(.+))?",
    re.IGNORECASE | re.DOTALL,
)


class MemoryMutationCommand(StrEnum):
    REMEMBER = "remember"
    CORRECT = "correct"
    FORGET = "forget"


class ExplicitMemoryIntent(ContractModel):
    """Deterministically parsed user intent, never model-supplied authority."""

    command: MemoryMutationCommand
    content: str | None = Field(default=None, max_length=16_384)
    record_id: str | None = Field(default=None, max_length=128)
    reason: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def validate_command_shape(self) -> ExplicitMemoryIntent:
        if self.command is MemoryMutationCommand.REMEMBER:
            if not self.content or self.record_id is not None or self.reason is not None:
                raise ValueError("remember requires content only")
        elif self.command is MemoryMutationCommand.CORRECT:
            if not self.content or not self.record_id or self.reason is not None:
                raise ValueError("correct requires a record and replacement content")
        elif self.command is MemoryMutationCommand.FORGET:
            if self.content is not None or not self.record_id or not self.reason:
                raise ValueError("forget requires a record and reason")
        return self


class MemoryMutationAuthority(ContractModel):
    """Sealed destination and provenance authority derived from an admitted Slack event."""

    scope: ScopeKey
    destination: ConversationRef
    actor_id: NonEmptyStr
    event_id: NonEmptyStr
    task_id: NonEmptyStr
    run_id: NonEmptyStr
    message_reference: NonEmptyStr
    intent: ExplicitMemoryIntent

    @model_validator(mode="after")
    def validate_slack_destination(self) -> MemoryMutationAuthority:
        if self.destination.provider != "slack":
            raise ValueError("memory mutation authority requires a Slack destination")
        if (
            self.destination.kind is ConversationKind.DM
            and self.destination.actor_id != self.actor_id
        ):
            raise ValueError("DM destination actor does not match mutation actor")
        return self

    @property
    def visibility(self) -> MemoryVisibility:
        if self.destination.kind is ConversationKind.DM:
            return MemoryVisibility.ACTOR_PRIVATE
        return MemoryVisibility.CONVERSATION_LOCAL

    @property
    def namespace_id(self) -> str:
        if self.destination.kind is ConversationKind.DM:
            return self.actor_id
        return self.destination.external_id

    def sources(self) -> tuple[MemorySource, ...]:
        """Bind every revision to the current event, task, and Slack message."""

        return tuple(
            MemorySource(
                id=_source_id(kind, reference, self),
                scope=self.scope,
                source_kind=kind,
                reference=reference,
                visibility=self.visibility,
                namespace_id=self.namespace_id,
            )
            for kind, reference in (
                ("slack_event", self.event_id),
                ("leo_task", self.task_id),
                ("slack_message", self.message_reference),
            )
        )


def parse_explicit_memory_intent(objective: str) -> ExplicitMemoryIntent | None:
    """Accept unambiguous direct requests while conversational mentions fail closed."""

    text = objective.strip()
    match = _REMEMBER_PATTERN.fullmatch(text)
    if match is not None:
        content = match.group("content").strip()
        if match.group("modal") is not None and content.endswith("?"):
            content = content[:-1].rstrip()
        has_explicit_clause = match.group("that") is not None
        if _RECALL_COMPLEMENT_PATTERN.match(content) is not None and not has_explicit_clause:
            return None
        if (
            match.group("modal") is not None
            and match.group("scope") is None
            and not has_explicit_clause
            and _DECLARATIVE_COMPLEMENT_PATTERN.search(content) is None
        ):
            return None
        return _validated_intent(
            command=MemoryMutationCommand.REMEMBER,
            content=content,
        )
    match = _CORRECT_PATTERN.fullmatch(text)
    if match is not None:
        return _validated_intent(
            command=MemoryMutationCommand.CORRECT,
            record_id=match.group(1),
            content=match.group(2).strip(),
        )
    match = _FORGET_PATTERN.fullmatch(text)
    if match is not None:
        reason = (match.group(2) or "explicit user request").strip()
        return _validated_intent(
            command=MemoryMutationCommand.FORGET,
            record_id=match.group(1),
            reason=reason,
        )
    return None


def bind_memory_mutation_authority(
    *,
    scope: ScopeKey,
    team_id: str,
    conversation_id: str,
    conversation_kind: ConversationKind,
    actor_id: str,
    event_id: str,
    task_id: str,
    run_id: str,
    message_reference: str,
    objective: str,
) -> MemoryMutationAuthority | None:
    """Bind a recognized command to exact server-admitted destination authority."""

    intent = parse_explicit_memory_intent(objective)
    if intent is None:
        return None
    destination = ConversationRef(
        provider="slack",
        team_id=team_id,
        external_id=conversation_id,
        kind=conversation_kind,
        actor_id=actor_id if conversation_kind is ConversationKind.DM else None,
    )
    return MemoryMutationAuthority(
        scope=scope,
        destination=destination,
        actor_id=actor_id,
        event_id=event_id,
        task_id=task_id,
        run_id=run_id,
        message_reference=message_reference,
        intent=intent,
    )


class _NoArguments(ContractModel):
    pass


class ExplicitMemoryMutationTool:
    """One zero-argument tool whose complete mutation is sealed in server authority."""

    def __init__(
        self,
        *,
        service: ExplicitMemoryService,
        authority: MemoryMutationAuthority,
        clock: Clock,
    ) -> None:
        self._service = service
        self._authority = authority
        self._clock = clock
        self._outcome: ToolOutcome | None = None
        command = authority.intent.command
        self._spec = ToolSpec(
            name=f"memory.{command.value}",
            description=(
                "Commit the user's already-confirmed explicit memory command to the "
                "server-bound Slack destination. This tool accepts no arguments."
            ),
            domain="memory",
            input_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            effect=ToolEffect.STATE_MUTATION,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            retry=ToolRetryPolicy(max_attempts=1),
            timeout_seconds=10,
            max_result_bytes=2_048,
            required_roles=frozenset({"researcher"}),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _NoArguments.model_validate(arguments).model_dump()

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        try:
            self.validate(arguments)
        except (ValidationError, ValueError, TypeError):
            return ToolFailure(
                code="MEMORY_ARGUMENTS_NOT_ALLOWED",
                safe_message="The server-bound memory command accepts no model arguments.",
            )
        mismatch = _authority_mismatch(self._authority, context)
        if mismatch is not None:
            return ToolFailure(
                code="MEMORY_AUTHORITY_MISMATCH",
                safe_message=f"The explicit memory command was not committed: {mismatch}.",
            )
        if self._outcome is not None:
            return self._outcome
        authority = self._authority
        sources = authority.sources()
        intent = authority.intent
        try:
            if intent.command is MemoryMutationCommand.REMEMBER:
                assert intent.content is not None
                record = await self._service.remember(
                    authority.scope,
                    _candidate(authority, intent.content, sources, self._clock.now()),
                    actor_id=authority.actor_id,
                    sources=sources,
                    confirmed=True,
                )
            elif intent.command is MemoryMutationCommand.CORRECT:
                assert intent.content is not None and intent.record_id is not None
                record = await self._service.correct(
                    authority.scope,
                    intent.record_id,
                    _candidate(authority, intent.content, sources, self._clock.now()),
                    actor_id=authority.actor_id,
                    sources=sources,
                    confirmed=True,
                )
            else:
                assert intent.record_id is not None and intent.reason is not None
                record = await self._service.forget(
                    authority.scope,
                    intent.record_id,
                    actor_id=authority.actor_id,
                    visibility=authority.visibility,
                    namespace_id=authority.namespace_id,
                    sources=sources,
                    confirmed=True,
                    reason=intent.reason,
                )
        except MemoryCommandRejected as exc:
            return ToolFailure(
                code="MEMORY_COMMAND_REJECTED",
                safe_message=f"The explicit memory command was not committed: {exc.safe_code}.",
            )
        except StoreError:
            return ToolFailure(
                code="MEMORY_STORE_ERROR",
                safe_message="The explicit memory command could not be committed safely.",
            )
        outcome = ToolSuccess(
            data={
                "operation": intent.command.value,
                "record_id": record.id,
                "revision": record.current_revision,
                "status": record.status.value,
            },
            source=SourceRef(provider="leo_memory", reference=record.id),
            observed_at=self._clock.now(),
        )
        self._outcome = outcome
        return outcome


def build_explicit_memory_tools(
    *,
    service: ExplicitMemoryService,
    authority: MemoryMutationAuthority,
    clock: Clock,
) -> tuple[ExplicitMemoryMutationTool, ...]:
    return (ExplicitMemoryMutationTool(service=service, authority=authority, clock=clock),)


def _validated_intent(**values: object) -> ExplicitMemoryIntent | None:
    try:
        return ExplicitMemoryIntent.model_validate(values)
    except ValidationError:
        return None


def _source_id(kind: str, reference: str, authority: MemoryMutationAuthority) -> str:
    material = "\x1f".join(
        (
            authority.scope.organization_id,
            authority.scope.strategy_id,
            authority.destination.team_id,
            authority.destination.external_id,
            authority.actor_id,
            authority.task_id,
            kind,
            reference,
        )
    )
    return f"memory-source-{hashlib.sha256(material.encode('utf-8')).hexdigest()}"


def _candidate(
    authority: MemoryMutationAuthority,
    content: str,
    sources: tuple[MemorySource, ...],
    now: datetime,
) -> MemoryCandidate:
    return MemoryCandidate(
        kind=MemoryKind.NOTE,
        content=content,
        source_ids=tuple(source.id for source in sources),
        visibility=authority.visibility,
        namespace_id=authority.namespace_id,
        sensitivity=0.2,
        valid_from=now,
        reason=f"explicit Slack {authority.intent.command.value}",
    )


def _authority_mismatch(
    authority: MemoryMutationAuthority,
    context: ToolExecutionContext,
) -> str | None:
    if context.trusted_scope.namespace != authority.scope:
        return "scope mismatch"
    if context.trusted_scope.actor_id != authority.actor_id:
        return "actor mismatch"
    if context.run_id != authority.run_id:
        return "run mismatch"
    return None
