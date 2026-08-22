"""Versioned policy-first tool catalog; discovery never grants permission."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from leo.harness.models import (
    ContractModel,
    NonEmptyStr,
    RunPhase,
    ScopeKey,
    ToolEffect,
    ToolSpec,
)


class CapabilityHealth(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class CapabilitySensitivity(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"


class CapabilityLatency(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CatalogTool(ContractModel):
    id: NonEmptyStr
    semantic_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    provider: NonEmptyStr
    spec: ToolSpec
    short_description: NonEmptyStr = Field(max_length=240)
    long_description: NonEmptyStr | None = Field(default=None, max_length=2_000)
    tags: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    capability_tags: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    entity_tags: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    profiles: frozenset[NonEmptyStr] = Field(default_factory=lambda: frozenset({"research"}))
    authorized_roles: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    allowed_namespaces: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    allowed_conversation_kinds: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    authenticated: bool = True
    health: CapabilityHealth = CapabilityHealth.HEALTHY
    sensitivity: CapabilitySensitivity = CapabilitySensitivity.PUBLIC
    freshness_seconds: int | None = Field(default=None, ge=0)
    rate_limit_per_minute: int | None = Field(default=None, ge=1)
    latency: CapabilityLatency = CapabilityLatency.MEDIUM
    observation_kind: NonEmptyStr | None = None
    normalization_version: NonEmptyStr = "normalization-v1"
    verification_expectations: frozenset[NonEmptyStr] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def identity_matches_executable_spec(self) -> CatalogTool:
        if self.id != self.spec.name:
            raise ValueError("catalog capability ID must match executable tool name")
        return self

    @property
    def schema_fingerprint(self) -> str:
        encoded = json.dumps(self.spec.input_schema, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class EligibilityDecision(ContractModel):
    capability_id: NonEmptyStr
    eligible: bool
    reason: NonEmptyStr
    catalog_version: NonEmptyStr


class ToolCatalogError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class InMemoryToolCatalog:
    def __init__(self, *, version: str = "catalog-v1") -> None:
        self.version = version
        self._records: dict[str, CatalogTool] = {}

    def register(self, record: CatalogTool) -> None:
        existing = self._records.get(record.id)
        if existing is not None and existing != record:
            raise ToolCatalogError("catalog_version_conflict")
        self._records[record.id] = record

    @property
    def fingerprint(self) -> str:
        payload = [
            record.model_dump(mode="json")
            for record in sorted(
                self._records.values(),
                key=lambda item: (item.id, item.semantic_version),
            )
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def records(self) -> tuple[CatalogTool, ...]:
        return tuple(
            sorted(
                self._records.values(),
                key=lambda item: (item.id, item.semantic_version),
            )
        )

    def get(self, capability_id: str) -> CatalogTool:
        try:
            return self._records[capability_id]
        except KeyError as exc:
            raise ToolCatalogError("unknown_capability") from exc

    def eligible(
        self,
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
    ) -> tuple[CatalogTool, ...]:
        effective_roles = _effective_roles(role=role, roles=roles)
        records = tuple(
            record
            for record in self._records.values()
            if record.spec.effect is ToolEffect.READ
            and phase in record.spec.allowed_phases
            and profile in record.profiles
            and (
                not record.authorized_roles
                or bool(record.authorized_roles.intersection(effective_roles))
            )
            and record.spec.required_roles.issubset(effective_roles)
            and _namespace_allowed(record, namespace)
            and _conversation_allowed(record, conversation_kind)
            and record.authenticated
            and record.health in {CapabilityHealth.HEALTHY, CapabilityHealth.DEGRADED}
            and record.spec.estimated_cost <= remaining_cost
        )
        return tuple(sorted(records, key=lambda record: (record.id, record.semantic_version)))

    def eligibility(
        self,
        record: CatalogTool,
        *,
        phase: RunPhase,
        profile: str,
        role: str | None = None,
        roles: frozenset[str] | None = None,
        remaining_cost: float,
        namespace: ScopeKey | None = None,
        conversation_kind: str | None = None,
    ) -> EligibilityDecision:
        eligible = record in self.eligible(
            phase=phase,
            profile=profile,
            role=role,
            roles=roles,
            remaining_cost=remaining_cost,
            namespace=namespace,
            conversation_kind=conversation_kind,
        )
        effective_roles = _effective_roles(role=role, roles=roles)
        reason = (
            "eligible"
            if eligible
            else _rejection_reason(
                record,
                phase,
                profile,
                effective_roles,
                remaining_cost,
                namespace,
                conversation_kind,
            )
        )
        return EligibilityDecision(
            capability_id=record.id,
            eligible=eligible,
            reason=reason,
            catalog_version=self.version,
        )


def _rejection_reason(
    record: CatalogTool,
    phase: RunPhase,
    profile: str,
    roles: frozenset[str],
    remaining_cost: float,
    namespace: ScopeKey | None,
    conversation_kind: str | None,
) -> str:
    if record.spec.effect is not ToolEffect.READ:
        return "effect_not_allowed"
    if phase not in record.spec.allowed_phases:
        return "phase_not_allowed"
    if profile not in record.profiles:
        return "profile_not_allowed"
    if record.authorized_roles and not record.authorized_roles.intersection(roles):
        return "role_not_allowed"
    if not record.spec.required_roles.issubset(roles):
        return "required_role_missing"
    if not _namespace_allowed(record, namespace):
        return "namespace_not_allowed"
    if not _conversation_allowed(record, conversation_kind):
        return "conversation_not_allowed"
    if not record.authenticated:
        return "authentication_unavailable"
    if record.health not in {CapabilityHealth.HEALTHY, CapabilityHealth.DEGRADED}:
        return "health_unavailable"
    if record.spec.estimated_cost > remaining_cost:
        return "budget_exceeded"
    return "policy_denied"


def _effective_roles(*, role: str | None, roles: frozenset[str] | None) -> frozenset[str]:
    if roles is not None:
        if role is not None and role not in roles:
            raise ValueError("role must be included in roles when both are supplied")
        return roles
    return frozenset({role}) if role else frozenset()


def _namespace_allowed(record: CatalogTool, namespace: ScopeKey | None) -> bool:
    if not record.allowed_namespaces:
        return True
    return namespace is not None and _namespace_key(namespace) in record.allowed_namespaces


def _conversation_allowed(record: CatalogTool, conversation_kind: str | None) -> bool:
    if not record.allowed_conversation_kinds:
        return True
    return conversation_kind is not None and conversation_kind in record.allowed_conversation_kinds


def _namespace_key(namespace: ScopeKey) -> str:
    return f"{namespace.organization_id}/{namespace.strategy_id}"
