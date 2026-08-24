"""Harness tool wrapper for the bounded public-text fetch policy."""

from __future__ import annotations

from datetime import timedelta

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from leo.agent.contracts import (
    Clock,
    RunPhase,
    SourceRef,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolSpec,
    ToolSuccess,
)
from leo.integrations.safe_fetch import FetchPolicy, FetchPolicyError, fetch_public_text


class _FetchArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: str = Field(min_length=8, max_length=2_048)
    fallback_urls: tuple[str, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def candidates_are_bounded_and_unique(self) -> _FetchArguments:
        candidates = (self.url, *self.fallback_urls)
        if any(len(candidate) < 8 or len(candidate) > 2_048 for candidate in candidates):
            raise ValueError("public fetch candidate length is invalid")
        if len(set(candidates)) != len(candidates):
            raise ValueError("public fetch candidates must be unique")
        return self


class PublicTextFetchTool:
    """Fetch one explicit public URL as capped, sanitized, untrusted research text."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient,
        clock: Clock,
        policy: FetchPolicy | None = None,
    ) -> None:
        self._client = client
        self._clock = clock
        self._policy = policy or FetchPolicy()
        self._spec = ToolSpec(
            name="web.fetch_public_text",
            version="1.3.0",
            description=(
                "Read one explicit public HTTP(S) URL as sanitized untrusted text, with up to "
                "four ordered discovery fallbacks when supplied by the harness. "
                "Private-network hosts, unsafe redirects, active HTML, unsupported content, "
                "and oversized responses are denied or bounded."
            ),
            domain="WEB",
            input_schema=_FetchArguments.model_json_schema(),
            effect=ToolEffect.READ,
            allowed_phases=frozenset({RunPhase.RESEARCH}),
            # One harness call may try five bounded candidates serially.  Preserve
            # each candidate's network timeout without letting the registry cancel
            # the composite before its admitted fallbacks can run.
            timeout_seconds=min(60.0, self._policy.timeout_seconds * 5 + 5.0),
            max_result_bytes=max(8_192, self._policy.max_bytes + 4_096),
        )

    @property
    def spec(self) -> ToolSpec:
        return self._spec

    def validate(self, arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _FetchArguments.model_validate(arguments).model_dump(mode="json")

    async def execute(
        self,
        arguments: dict[str, JsonValue],
        context: ToolExecutionContext,
    ) -> ToolOutcome:
        del context
        parsed = _FetchArguments.model_validate(arguments)
        candidates = (parsed.url, *parsed.fallback_urls)
        failed_attempts: list[dict[str, JsonValue]] = []
        artifact = None
        for candidate in candidates:
            try:
                fetched = await fetch_public_text(
                    self._client,
                    candidate,
                    policy=self._policy,
                )
            except FetchPolicyError as exc:
                if len(candidates) == 1:
                    return _fetch_failure(exc)
                failed_attempts.append({"url": candidate, "code": exc.safe_code})
                continue
            if fetched.truncated and len(candidates) > 1:
                failed_attempts.append({"url": candidate, "code": "fetch_truncated"})
                continue
            artifact = fetched
            break
        if artifact is None:
            return ToolFailure(
                code="FETCH_CANDIDATES_EXHAUSTED",
                safe_message=(
                    "The bounded public-source candidates did not yield complete readable text."
                ),
            )
        now = self._clock.now()
        return ToolSuccess(
            data={
                "requested_url": artifact.requested_url,
                "url": artifact.final_url,
                "redirect_count": artifact.redirect_count,
                "content_type": artifact.content_type,
                "text": artifact.text,
                "content_sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
                "truncated": artifact.truncated,
                "peer_ip": artifact.peer_ip,
                "dns_pin_sha256": artifact.dns_pin_sha256,
                "candidate_attempt_count": len(failed_attempts) + 1,
                "failed_candidates": failed_attempts,
            },
            source=SourceRef(
                provider="public-web",
                reference=artifact.sha256,
                url=artifact.final_url,
            ),
            observed_at=now,
            expires_at=now + timedelta(minutes=15),
        )


def _fetch_failure(exc: FetchPolicyError) -> ToolFailure:
    return ToolFailure(
        code=exc.safe_code.upper(),
        retryable=exc.safe_code
        in {
            "fetch_timeout",
            "fetch_transport_error",
            "fetch_dns_failed",
            "fetch_rate_limited",
            "fetch_upstream_unavailable",
        },
        safe_message=f"Public fetch stopped safely ({exc.safe_code}).",
    )
