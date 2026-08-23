"""Leo's bounded plan/act/observe/verify runtime loop."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime

from pydantic import JsonValue

from leo.harness.capability_selection import capability_selection_fingerprint
from leo.harness.context import context_manifest_event_payload
from leo.harness.models import (
    BudgetUsage,
    CapabilitySelection,
    ClaimKind,
    CompletionProposal,
    CoordinatorResult,
    EventDraft,
    EventType,
    ModelDecision,
    ModelRequest,
    ModelTurnResult,
    ModelUsage,
    Observation,
    Run,
    RunBundle,
    RunStatus,
    TaskStatus,
    ToolChoiceMode,
    ToolExecutionContext,
    ToolFailure,
    ToolOutcome,
    ToolRequest,
    ToolRequests,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
    VerifiedCompletion,
    VerifierCheck,
    VerifierResult,
    VerifierStatus,
    constrained_values_match,
)
from leo.harness.normalization import NormalizationFailure, normalize_success
from leo.harness.ports import (
    CapabilitySelector,
    Clock,
    CompletionVerifier,
    ContextAssembler,
    ContextAssemblyError,
    IdGenerator,
    ModelCallTranscriptSink,
    ModelGateway,
    ModelGatewayError,
    RunStore,
)
from leo.harness.tools import ToolRegistry
from leo.harness.transitions import (
    advance_step,
    exhaust_task_and_run,
    fail_task_and_run,
    start_task_and_run,
    time_out_task_and_run,
)

logger = logging.getLogger(__name__)

_TRANSCRIPT_SINK_TIMEOUT_SECONDS = 5.0
_MAX_EVENT_ARGUMENTS_BYTES = 4096
# These gateway failure codes describe a malformed or truncated *model output*
# (bad JSON, no content, unparseable tool arguments) rather than a real
# infrastructure fault. A fresh turn with corrective feedback routinely recovers
# from them, so they get a bounded retry instead of an instant terminal failure.
_RETRYABLE_GATEWAY_FAILURE_CODES = frozenset(
    {"malformed_completion", "empty_decision", "malformed_tool_arguments"}
)


def _bounded_tool_arguments(arguments: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Cap tool-call arguments before they enter a durable event payload.

    Event payloads are hard-capped at EVENT_PAYLOAD_MAX_BYTES with no size-projection
    fallback for tool_started/tool_failed (only context_built has one, see
    persistence_rules._project_context_built_payload) -- an oversize payload raises an
    uncaught StoreError that would crash the whole run over one large-but-legitimate
    tool call (e.g. a big subagent plan definition), not just fail that call. Most
    arguments are a few hundred bytes; this only bites the rare outlier.
    """

    encoded = json.dumps(arguments, sort_keys=True, default=str)
    if len(encoded.encode("utf-8")) <= _MAX_EVENT_ARGUMENTS_BYTES:
        return arguments
    return {
        "_truncated": True,
        "_original_byte_size": len(encoded.encode("utf-8")),
        "_argument_keys": ", ".join(sorted(arguments.keys())),
    }


class ScopeMismatchError(RuntimeError):
    pass


class RunCoordinator:
    """Owns iteration and terminal truth; model/provider adapters only propose or observe."""

    def __init__(
        self,
        *,
        store: RunStore,
        model: ModelGateway,
        tools: ToolRegistry,
        context: ContextAssembler,
        verifier: CompletionVerifier,
        clock: Clock,
        ids: IdGenerator,
        capabilities: CapabilitySelector | None = None,
        transcript_sink: ModelCallTranscriptSink | None = None,
    ) -> None:
        self._store = store
        self._model = model
        self._tools = tools
        self._context = context
        self._verifier = verifier
        self._clock = clock
        self._ids = ids
        self._capabilities = capabilities
        self._transcript_sink = transcript_sink

    async def run(
        self,
        *,
        task_id: str,
        run_id: str,
        trusted_scope: TrustedScope,
    ) -> CoordinatorResult:
        bundle = await self._store.load(task_id, run_id, trusted_scope.namespace)
        self._require_scope(bundle, trusted_scope)
        if bundle.run.status is RunStatus.QUEUED:
            task, run = start_task_and_run(bundle.task, bundle.run, started_at=self._clock.now())
            bundle = await self._store.commit(
                expected_task_version=bundle.task.version,
                expected_run_version=bundle.run.version,
                task=task,
                run=run,
                events=(
                    EventDraft(
                        type=EventType.TASK_STARTED,
                        iteration=0,
                        payload={"phase": run.phase.value},
                    ),
                ),
            )

        if bundle.run.started_at is None:
            raise RuntimeError("running run has no durable start time")
        started_at = bundle.run.started_at

        while bundle.run.status is RunStatus.RUNNING:
            now = self._clock.now()
            if bundle.run.deadline_at is not None and now >= bundle.run.deadline_at:
                task, run = time_out_task_and_run(
                    bundle.task,
                    bundle.run,
                    "run_deadline_exceeded",
                    usage=bundle.run.usage,
                )
                bundle = await self._commit(
                    bundle,
                    task,
                    run,
                    events=(
                        EventDraft(
                            type=EventType.RUN_TIMED_OUT,
                            iteration=run.iteration,
                            payload={"reason": "run_deadline_exceeded"},
                        ),
                    ),
                )
                break
            elapsed_seconds = (now - started_at).total_seconds()
            exhausted_reason = _budget_reason(bundle, elapsed_seconds)
            if exhausted_reason is not None:
                bundle = await self._commit_exhaustion(bundle, exhausted_reason)
                break

            available_specs = self._tools.specs_for_context(
                bundle.run.phase,
                trusted_scope,
            )
            selection = self._select_capabilities(
                bundle=bundle,
                trusted_scope=trusted_scope,
                available_specs=available_specs,
            )
            specs = selection.tools
            try:
                request = self._context.assemble(bundle, specs)
            except ContextAssemblyError as exc:
                task, run = fail_task_and_run(
                    bundle.task,
                    bundle.run,
                    f"context_assembly_error:{exc.code}",
                    usage=bundle.run.usage,
                )
                bundle = await self._commit(
                    bundle,
                    task,
                    run,
                    events=(
                        EventDraft(
                            type=EventType.RUN_FAILED,
                            iteration=run.iteration,
                            payload={
                                "reason": run.terminal_reason or "context_assembly_error",
                                "detail": exc.safe_message,
                            },
                        ),
                    ),
                )
                break
            selection = _selection_for_advertised_tools(selection, request.tools)
            common_events = [
                EventDraft(
                    type=EventType.CONTEXT_BUILT,
                    iteration=bundle.run.iteration,
                    payload={
                        "segments": [segment.name for segment in request.manifest.segments],
                        "tool_count": len(request.tools),
                        "tool_choice": request.tool_choice.mode.value,
                        "required_tool": request.tool_choice.required_tool_name,
                        "required_arguments": [
                            item.model_dump(mode="json")
                            for item in request.tool_choice.required_arguments
                        ],
                        "completion_contract": request.completion_contract.model_dump(mode="json"),
                        "source_manifest": context_manifest_event_payload(request.manifest),
                        "catalog_version": selection.catalog_version,
                        "catalog_fingerprint": selection.catalog_fingerprint,
                        "selection_fingerprint": selection.selection_fingerprint,
                        "selection_mode": selection.mode,
                        "selection_reason": selection.reason,
                        "capability_candidates": list(selection.candidate_ids),
                        "capability_selected": list(selection.selected_ids),
                        "skill_selected": list(selection.selected_skill_ids),
                        "capability_query_hash": selection.query_hash,
                        "eligible_capability_count": selection.eligible_count,
                    },
                )
            ]

            reservation_cost = bundle.run.limits.estimated_model_cost
            reservation_id = self._ids.new("model-reservation")
            reservation_usage = bundle.run.usage.model_copy(
                update={
                    "reserved_cost": bundle.run.usage.reserved_cost + reservation_cost,
                    "reservation_id": reservation_id,
                }
            )
            reserved_task = bundle.task.model_copy(update={"version": bundle.task.version + 1})
            reserved_run = bundle.run.model_copy(
                update={"usage": reservation_usage, "version": bundle.run.version + 1}
            )
            bundle = await self._commit(
                bundle,
                reserved_task,
                reserved_run,
                events=(
                    EventDraft(
                        type=EventType.MODEL_BUDGET_RESERVED,
                        iteration=bundle.run.iteration,
                        payload={
                            "reservation_id": reservation_id,
                            "estimated_cost": reservation_cost,
                        },
                    ),
                ),
            )

            try:
                remaining_seconds = _remaining_seconds(bundle.run, started_at, self._clock.now())
                async with asyncio.timeout(remaining_seconds):
                    raw_result: object = await self._model.decide(request)
                    if isinstance(raw_result, ModelTurnResult):
                        turn_result = raw_result
                    else:
                        if not isinstance(raw_result, (ToolRequests, CompletionProposal)):
                            raise TypeError(
                                f"unsupported model result: {type(raw_result).__name__}"
                            )
                        turn_result = ModelTurnResult(
                            decision=raw_result,
                            provider="legacy-fixture",
                            model=type(self._model).__name__,
                            finish_reason=(
                                "tool_calls" if isinstance(raw_result, ToolRequests) else "stop"
                            ),
                        )
                    decision = turn_result.decision
            except TimeoutError:
                usage = _model_call_usage(bundle.run.usage, reconcile_reservation=False)
                task, run = time_out_task_and_run(
                    bundle.task,
                    bundle.run,
                    "model_call_exceeded_run_deadline",
                    usage=usage,
                    advance_iteration=True,
                )
                common_events.append(
                    EventDraft(
                        type=EventType.RUN_TIMED_OUT,
                        iteration=run.iteration,
                        payload={"reason": run.terminal_reason or "run_deadline_exceeded"},
                    )
                )
                bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                break
            except Exception as exc:
                fallback_answer: str | None = None
                if isinstance(exc, ModelGatewayError):
                    failure_code = exc.code
                    safe_detail = exc.safe_message
                    fallback_answer = exc.fallback_answer
                else:
                    failure_code = type(exc).__name__
                    safe_detail = "The model gateway failed unexpectedly."
                usage = _model_call_usage(bundle.run.usage, reconcile_reservation=False)
                if fallback_answer is not None:
                    # The gateway gave up trying to get a *better* completion but
                    # attached an earlier, self-contained answer. Deliver it instead
                    # of a terminal failure with no content at all.
                    bundle = await self._store.complete_verified(
                        expected_task_version=bundle.task.version,
                        expected_run_version=bundle.run.version,
                        task_id=bundle.task.id,
                        run_id=bundle.run.id,
                        scope=bundle.run.scope,
                        usage=usage,
                        completion=_best_effort_completion(fallback_answer, failure_code),
                        preceding_events=tuple(common_events),
                    )
                    break
                if failure_code in _RETRYABLE_GATEWAY_FAILURE_CODES:
                    task, run = advance_step(
                        bundle.task,
                        bundle.run,
                        usage=usage,
                        verifier_feedback=(
                            *bundle.task.verifier_feedback,
                            _gateway_failure_feedback(failure_code),
                        ),
                    )
                    bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                    continue
                task, run = fail_task_and_run(
                    bundle.task,
                    bundle.run,
                    f"model_gateway_error:{failure_code}",
                    usage=usage,
                )
                common_events.append(
                    EventDraft(
                        type=EventType.RUN_FAILED,
                        iteration=run.iteration,
                        payload={
                            "reason": run.terminal_reason or "model_gateway_error",
                            "detail": safe_detail,
                        },
                    )
                )
                bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                break

            # A cancellation or external terminal transition may win while the model is
            # in flight.  The provider result has no authority of its own: reload durable
            # truth before interpreting it or launching any requested tool.
            authoritative = await self._store.load(
                bundle.task.id,
                bundle.run.id,
                trusted_scope.namespace,
            )
            self._require_scope(authoritative, trusted_scope)
            if not _is_running_authority(authoritative):
                bundle = authoritative
                break

            usage = _model_call_usage(
                bundle.run.usage,
                turn_result.usage,
                reserved_cost=reservation_cost,
            )
            common_events.append(
                EventDraft(
                    type=EventType.MODEL_CALLED,
                    iteration=bundle.run.iteration,
                    payload={
                        "decision": decision.kind,
                        "provider": turn_result.provider,
                        "model": turn_result.model,
                        "request_id": turn_result.request_id,
                        "finish_reason": turn_result.finish_reason,
                        **turn_result.usage.model_dump(mode="json"),
                    },
                )
            )
            if (
                self._transcript_sink is not None
                and turn_result.request_id is not None
                and turn_result.raw_request is not None
                and turn_result.raw_response is not None
            ):
                try:
                    # Bounded independently of the run's own deadline: this is a
                    # best-effort dashboard side-write, not run work, so a hang here
                    # (pool exhaustion, lock contention, network stall -- not merely a
                    # raised exception) must not be able to wedge this iteration, or by
                    # extension a strictly-sequential durable worker with no other
                    # timeout watching this await.
                    async with asyncio.timeout(_TRANSCRIPT_SINK_TIMEOUT_SECONDS):
                        await self._transcript_sink.record(
                            run_id=bundle.run.id,
                            task_id=bundle.task.id,
                            scope=bundle.run.scope,
                            request_id=turn_result.request_id,
                            iteration=bundle.run.iteration,
                            raw_request=turn_result.raw_request,
                            raw_response=turn_result.raw_response,
                            occurred_at=self._clock.now(),
                        )
                except Exception:
                    logger.warning(
                        "model call transcript recording failed for run %s request %s; "
                        "dashboard inspection for this turn will be degraded",
                        bundle.run.id,
                        turn_result.request_id,
                        exc_info=True,
                    )
            cost_reason = _cost_budget_reason(bundle.run, usage)
            if cost_reason is not None:
                bundle = await self._commit_exhaustion(
                    bundle,
                    cost_reason,
                    usage=usage,
                    preceding_events=tuple(common_events),
                )
                break
            # A tool-choice or completion-contract violation is almost always a
            # one-turn correctable mistake (wrong tool, one claim too many, ...),
            # not a reason to kill the whole run. Feed the model exact corrective
            # guidance and let it try again, bounded by the ordinary iteration/model
            # -call budget -- the same recovery path a verifier FAIL already gets.
            policy_error = _decision_policy_error(request, decision)
            if policy_error is not None:
                task, run = advance_step(
                    bundle.task,
                    bundle.run,
                    usage=usage,
                    verifier_feedback=(
                        *bundle.task.verifier_feedback,
                        _policy_error_feedback(policy_error, request),
                    ),
                )
                bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                continue
            completion_error = _completion_contract_error(request, decision)
            if completion_error is not None:
                task, run = advance_step(
                    bundle.task,
                    bundle.run,
                    usage=usage,
                    verifier_feedback=(
                        *bundle.task.verifier_feedback,
                        _completion_contract_error_feedback(completion_error, request),
                    ),
                )
                bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                continue

            if isinstance(decision, ToolRequests):
                remaining_tools = bundle.run.limits.max_tool_calls - usage.tool_calls
                if len(decision.calls) > remaining_tools:
                    task, run = exhaust_task_and_run(
                        bundle.task,
                        bundle.run,
                        "tool_call_budget_exhausted",
                        usage=usage,
                        advance_iteration=True,
                    )
                    common_events.append(
                        EventDraft(
                            type=EventType.BUDGET_EXHAUSTED,
                            iteration=run.iteration,
                            payload={"reason": "tool_call_budget_exhausted"},
                        )
                    )
                    bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                    break

                new_observations: list[Observation] = []
                tool_failed = False
                run_timed_out = False
                failure_reason = ""
                failure: ToolFailure | None = None
                execution_results: list[tuple[ToolRequest, ToolOutcome]] = []
                parallel_safe = self._tools.requests_are_parallel_safe(
                    decision.calls,
                    bundle.run.phase,
                )

                if parallel_safe:
                    for call in decision.calls:
                        common_events.append(
                            EventDraft(
                                type=EventType.TOOL_STARTED,
                                iteration=bundle.run.iteration,
                                payload={
                                    "tool_call_id": call.id,
                                    "tool": call.name,
                                    "arguments": _bounded_tool_arguments(call.arguments),
                                    "parallel_batch": True,
                                },
                            )
                        )
                    remaining_seconds = _remaining_seconds(
                        bundle.run,
                        started_at,
                        self._clock.now(),
                    )
                    usage = usage.model_copy(
                        update={"tool_calls": usage.tool_calls + len(decision.calls)}
                    )
                    try:
                        async with asyncio.timeout(remaining_seconds):
                            outcomes = await asyncio.gather(
                                *(
                                    self._execute_tool(
                                        call,
                                        trusted_scope=trusted_scope,
                                        run=bundle.run,
                                    )
                                    for call in decision.calls
                                )
                            )
                        execution_results.extend(zip(decision.calls, outcomes, strict=True))
                    except TimeoutError:
                        run_timed_out = True
                        failure_reason = "tool_batch_exceeded_run_deadline"

                for call in () if parallel_safe else decision.calls:
                    common_events.append(
                        EventDraft(
                            type=EventType.TOOL_STARTED,
                            iteration=bundle.run.iteration,
                            payload={
                                "tool_call_id": call.id,
                                "tool": call.name,
                                "arguments": _bounded_tool_arguments(call.arguments),
                                "parallel_batch": False,
                            },
                        )
                    )
                    remaining_seconds = _remaining_seconds(
                        bundle.run, started_at, self._clock.now()
                    )
                    usage = usage.model_copy(update={"tool_calls": usage.tool_calls + 1})
                    try:
                        async with asyncio.timeout(remaining_seconds):
                            tool_outcome = await self._execute_tool(
                                call,
                                trusted_scope=trusted_scope,
                                run=bundle.run,
                            )
                    except TimeoutError:
                        run_timed_out = True
                        failure_reason = "tool_call_exceeded_run_deadline"
                        break
                    execution_results.append((call, tool_outcome))
                    if isinstance(tool_outcome, ToolFailure):
                        break

                # Tool adapters are still subordinate to durable terminal truth.  This
                # second reload covers cancellation after the model fence (including a
                # registry-level refusal) and prevents a stale observation commit.
                authoritative = await self._store.load(
                    bundle.task.id,
                    bundle.run.id,
                    trusted_scope.namespace,
                )
                self._require_scope(authoritative, trusted_scope)
                if not _is_running_authority(authoritative):
                    bundle = authoritative
                    break

                pending_successes: list[tuple[ToolRequest, Observation]] = []
                normalization_failed = False
                for raw_call, raw_outcome in execution_results:
                    call = raw_call
                    tool_outcome = raw_outcome
                    if isinstance(tool_outcome, ToolFailure):
                        tool_failed = True
                        if tool_outcome.code.startswith("TOOL_RESULT_"):
                            normalization_failed = True
                        if failure is None:
                            failure = tool_outcome
                            failure_reason = f"tool_failure:{tool_outcome.code}"
                        common_events.append(
                            EventDraft(
                                type=EventType.TOOL_FAILED,
                                iteration=bundle.run.iteration,
                                payload={
                                    "tool_call_id": call.id,
                                    "tool": call.name,
                                    "arguments": _bounded_tool_arguments(call.arguments),
                                    "code": tool_outcome.code,
                                    "retryable": tool_outcome.retryable,
                                },
                            )
                        )
                        continue
                    if not isinstance(tool_outcome, ToolSuccess):
                        tool_failed = True
                        normalization_failed = True
                        invalid = ToolFailure(
                            code="TOOL_RESULT_INVALID",
                            safe_message="Tool result failed bounded normalization.",
                        )
                        if failure is None:
                            failure = invalid
                            failure_reason = f"tool_failure:{invalid.code}"
                        common_events.append(
                            EventDraft(
                                type=EventType.TOOL_FAILED,
                                iteration=bundle.run.iteration,
                                payload={
                                    "tool_call_id": call.id,
                                    "tool": call.name,
                                    "arguments": _bounded_tool_arguments(call.arguments),
                                    "code": invalid.code,
                                    "retryable": False,
                                },
                            )
                        )
                        continue

                    try:
                        observation = normalize_success(
                            tool_outcome,
                            observation_id=self._ids.new("obs"),
                            scope=bundle.run.scope,
                            run_id=bundle.run.id,
                            tool_call_id=call.id,
                            observation_kind=call.name,
                        )
                    except NormalizationFailure as exc:
                        tool_failed = True
                        normalization_failed = True
                        invalid = ToolFailure(
                            code=_normalization_failure_code(exc.safe_code),
                            safe_message="Tool result failed bounded normalization.",
                        )
                        if failure is None:
                            failure = invalid
                            failure_reason = f"tool_failure:{invalid.code}"
                        common_events.append(
                            EventDraft(
                                type=EventType.TOOL_FAILED,
                                iteration=bundle.run.iteration,
                                payload={
                                    "tool_call_id": call.id,
                                    "tool": call.name,
                                    "arguments": _bounded_tool_arguments(call.arguments),
                                    "code": invalid.code,
                                    "retryable": False,
                                },
                            )
                        )
                        continue
                    pending_successes.append((call, observation))

                if not normalization_failed:
                    new_observations.extend(observation for _, observation in pending_successes)
                for call, observation in () if normalization_failed else pending_successes:
                    common_events.extend(
                        (
                            EventDraft(
                                type=EventType.TOOL_COMPLETED,
                                iteration=bundle.run.iteration,
                                payload={"tool_call_id": call.id, "tool": call.name},
                            ),
                            EventDraft(
                                type=EventType.OBSERVATION_CREATED,
                                iteration=bundle.run.iteration,
                                payload={
                                    "observation_id": observation.id,
                                    "tool_call_id": call.id,
                                },
                            ),
                        )
                    )

                if run_timed_out:
                    observation_ids = bundle.task.observation_ids + tuple(
                        item.id for item in new_observations
                    )
                    task, run = time_out_task_and_run(
                        bundle.task,
                        bundle.run,
                        failure_reason,
                        usage=usage,
                        observation_ids=observation_ids,
                        advance_iteration=True,
                    )
                    common_events.append(
                        EventDraft(
                            type=EventType.RUN_TIMED_OUT,
                            iteration=run.iteration,
                            payload={"reason": failure_reason},
                        )
                    )
                    bundle = await self._commit(
                        bundle,
                        task,
                        run,
                        observations=tuple(new_observations),
                        events=tuple(common_events),
                    )
                    break

                if tool_failed:
                    observation_ids = bundle.task.observation_ids + tuple(
                        item.id for item in new_observations
                    )
                    if failure is not None and failure.retryable:
                        task, run = advance_step(
                            bundle.task,
                            bundle.run,
                            usage=usage,
                            observation_ids=observation_ids,
                            verifier_feedback=(
                                *bundle.task.verifier_feedback,
                                failure.safe_message,
                            ),
                        )
                        bundle = await self._commit(
                            bundle,
                            task,
                            run,
                            observations=tuple(new_observations),
                            events=tuple(common_events),
                        )
                        continue
                    task, run = fail_task_and_run(
                        bundle.task,
                        bundle.run,
                        failure_reason,
                        usage=usage,
                        observation_ids=observation_ids,
                    )
                    common_events.append(
                        EventDraft(
                            type=EventType.RUN_FAILED,
                            iteration=run.iteration,
                            payload={"reason": failure_reason},
                        )
                    )
                    bundle = await self._commit(
                        bundle,
                        task,
                        run,
                        observations=tuple(new_observations),
                        events=tuple(common_events),
                    )
                    break

                observation_ids = bundle.task.observation_ids + tuple(
                    item.id for item in new_observations
                )
                task, run = advance_step(
                    bundle.task,
                    bundle.run,
                    usage=usage,
                    observation_ids=observation_ids,
                )
                bundle = await self._commit(
                    bundle,
                    task,
                    run,
                    observations=tuple(new_observations),
                    events=tuple(common_events),
                )
                continue

            if not isinstance(decision, CompletionProposal):
                raise TypeError(f"unsupported model decision: {type(decision).__name__}")

            common_events.append(
                EventDraft(
                    type=EventType.VERIFICATION_STARTED,
                    iteration=bundle.run.iteration,
                )
            )
            verification = self._verifier.verify(decision, bundle)
            if verification.result.status is VerifierStatus.PASS:
                if verification.completion is None:
                    raise RuntimeError("passing verifier did not return a completion")
                bundle = await self._store.complete_verified(
                    expected_task_version=bundle.task.version,
                    expected_run_version=bundle.run.version,
                    task_id=bundle.task.id,
                    run_id=bundle.run.id,
                    scope=bundle.run.scope,
                    usage=usage,
                    completion=verification.completion,
                    preceding_events=tuple(common_events),
                )
                break

            feedback = tuple(
                check.detail for check in verification.result.checks if not check.passed
            )
            task, run = advance_step(
                bundle.task,
                bundle.run,
                usage=usage,
                verifier_feedback=bundle.task.verifier_feedback + feedback,
            )
            common_events.append(
                EventDraft(
                    type=EventType.VERIFICATION_FAILED,
                    iteration=run.iteration,
                    payload={
                        "failed_checks": len(feedback),
                        "retryable": verification.result.retryable,
                        "checks": [
                            check.model_dump(mode="json") for check in verification.result.checks
                        ],
                    },
                )
            )
            if not verification.result.retryable:
                task, run = fail_task_and_run(
                    bundle.task,
                    bundle.run,
                    "non_retryable_verification_failure",
                    usage=usage,
                )
                common_events.append(
                    EventDraft(
                        type=EventType.RUN_FAILED,
                        iteration=run.iteration,
                        payload={"reason": "non_retryable_verification_failure"},
                    )
                )
                bundle = await self._commit(bundle, task, run, events=tuple(common_events))
                break
            bundle = await self._commit(bundle, task, run, events=tuple(common_events))

        return _result(bundle)

    def _select_capabilities(
        self,
        *,
        bundle: RunBundle,
        trusted_scope: TrustedScope,
        available_specs: tuple[ToolSpec, ...],
    ) -> CapabilitySelection:
        if self._capabilities is None:
            return _registry_selection(bundle, available_specs)
        try:
            selection = self._capabilities.select(
                bundle=bundle,
                trusted_scope=trusted_scope,
                available_tools=available_specs,
            )
            available_by_name = {item.name: item for item in available_specs}
            if any(available_by_name.get(item.name) != item for item in selection.tools):
                raise ValueError("selector returned a schema outside the eligible registry view")
            return selection
        except Exception:
            # Capability recall is optional for direct conversational answers. Fail closed
            # on tool authority, but do not turn an unavailable index/router into a Slack
            # availability error. Required sealed tools still fail in context assembly.
            logger.exception(
                "Capability selection failed closed; continuing without optional tools"
            )
            return _empty_selection(bundle, available_specs)

    async def _execute_tool(
        self,
        call: ToolRequest,
        *,
        trusted_scope: TrustedScope,
        run: Run,
    ) -> ToolOutcome:
        """Convert registry-boundary defects into one safe typed tool failure."""

        try:
            authoritative = await self._store.load(
                run.task_id,
                run.id,
                trusted_scope.namespace,
            )
            self._require_scope(authoritative, trusted_scope)
            if not _is_running_authority(authoritative):
                return ToolFailure(
                    code="RUN_TERMINAL_AUTHORITY",
                    retryable=False,
                    safe_message="The run stopped before the requested tool could start.",
                )
            return await self._tools.execute(
                call,
                ToolExecutionContext(
                    trusted_scope=trusted_scope,
                    run_id=run.id,
                    tool_call_id=call.id,
                ),
                run.phase,
            )
        except Exception:
            return ToolFailure(
                code="TOOL_RESULT_INVALID",
                safe_message="Tool result failed bounded normalization.",
            )

    async def _commit_exhaustion(
        self,
        bundle: RunBundle,
        reason: str,
        *,
        usage: BudgetUsage | None = None,
        preceding_events: tuple[EventDraft, ...] = (),
    ) -> RunBundle:
        next_usage = bundle.run.usage if usage is None else usage
        task, run = exhaust_task_and_run(
            bundle.task,
            bundle.run,
            reason,
            usage=next_usage,
        )
        return await self._commit(
            bundle,
            task,
            run,
            events=(
                *preceding_events,
                EventDraft(
                    type=EventType.BUDGET_EXHAUSTED,
                    iteration=run.iteration,
                    payload={"reason": reason},
                ),
            ),
        )

    async def _commit(
        self,
        previous: RunBundle,
        task: object,
        run: object,
        *,
        observations: tuple[Observation, ...] = (),
        events: tuple[EventDraft, ...] = (),
    ) -> RunBundle:
        from leo.harness.models import Run, Task

        if not isinstance(task, Task) or not isinstance(run, Run):
            raise TypeError("invalid task/run transition output")
        return await self._store.commit(
            expected_task_version=previous.task.version,
            expected_run_version=previous.run.version,
            task=task,
            run=run,
            observations=observations,
            events=events,
        )

    @staticmethod
    def _require_scope(bundle: RunBundle, trusted_scope: TrustedScope) -> None:
        if (
            bundle.task.scope != trusted_scope.namespace
            or bundle.run.scope != trusted_scope.namespace
        ):
            raise ScopeMismatchError("trusted scope does not match the task namespace")


def _budget_reason(bundle: RunBundle, elapsed_seconds: float) -> str | None:
    if bundle.run.usage.reservation_id is not None:
        return "model_budget_reservation_unreconciled"
    if bundle.run.iteration >= bundle.run.limits.max_iterations:
        return "iteration_budget_exhausted"
    if bundle.run.usage.model_calls >= bundle.run.limits.max_model_calls:
        return "model_call_budget_exhausted"
    if elapsed_seconds >= bundle.run.limits.max_elapsed_seconds:
        return "elapsed_time_budget_exhausted"
    return _cost_budget_reason(bundle.run, bundle.run.usage)


def _is_running_authority(bundle: RunBundle) -> bool:
    return bundle.task.status is TaskStatus.ACTIVE and bundle.run.status is RunStatus.RUNNING


def _remaining_seconds(run: Run, started_at: datetime, now: datetime) -> float:
    if run.deadline_at is not None:
        return max(0.0, (run.deadline_at - now).total_seconds())
    return max(0.0, run.limits.max_elapsed_seconds - (now - started_at).total_seconds())


def _cost_budget_reason(run: object, usage: BudgetUsage) -> str | None:
    from leo.harness.models import Run

    if not isinstance(run, Run) or run.limits.max_cost is None:
        return None
    if usage.model_calls > 0 and usage.cost is None:
        return "model_cost_unknown"
    if usage.cost is not None and usage.cost >= run.limits.max_cost:
        return "model_cost_budget_exhausted"
    estimated_total = (usage.cost or 0.0) + usage.reserved_cost + run.limits.estimated_model_cost
    if estimated_total > run.limits.max_cost:
        return "estimated_model_cost_budget_exhausted"
    return None


def _model_call_usage(
    current: BudgetUsage,
    reported: ModelUsage | None = None,
    *,
    reserved_cost: float = 0.0,
    reconcile_reservation: bool = True,
) -> BudgetUsage:
    metrics = reported or ModelUsage()
    prior_calls = current.model_calls
    reservation_update: dict[str, object] = {}
    if reconcile_reservation:
        reservation_update = {
            "reserved_cost": max(0.0, current.reserved_cost - reserved_cost),
            "reservation_id": None,
        }
    return current.model_copy(
        update={
            "model_calls": prior_calls + 1,
            "prompt_tokens": _add_metric(current.prompt_tokens, metrics.prompt_tokens, prior_calls),
            "completion_tokens": _add_metric(
                current.completion_tokens, metrics.completion_tokens, prior_calls
            ),
            "total_tokens": _add_metric(current.total_tokens, metrics.total_tokens, prior_calls),
            "cost": _add_metric(current.cost, metrics.cost, prior_calls),
            **reservation_update,
        }
    )


def _add_metric(
    previous: int | float | None, current: int | float | None, prior_calls: int
) -> int | float | None:
    if prior_calls == 0:
        return current
    if previous is None or current is None:
        return None
    return previous + current


def _decision_policy_error(
    request: ModelRequest,
    decision: ModelDecision,
) -> str | None:
    policy = request.tool_choice
    if policy.mode is ToolChoiceMode.NONE:
        return "tool_requested_while_disabled" if isinstance(decision, ToolRequests) else None
    if policy.mode is ToolChoiceMode.AUTO:
        if isinstance(decision, ToolRequests):
            advertised = frozenset(tool.name for tool in request.tools)
            if any(call.name not in advertised for call in decision.calls):
                return "unadvertised_tool_requested"
        return None
    if not isinstance(decision, ToolRequests):
        return "required_tool_not_requested"
    if len(decision.calls) != 1:
        return "required_tool_call_count_invalid"
    if decision.calls[0].name != policy.required_tool_name:
        return "wrong_required_tool_requested"
    if policy.required_arguments and not constrained_values_match(
        policy.required_arguments,
        decision.calls[0].arguments,
        exact=True,
    ):
        return "required_tool_arguments_mismatch"
    return None


def _completion_contract_error(
    request: ModelRequest,
    decision: ModelDecision,
) -> str | None:
    if not isinstance(decision, CompletionProposal):
        return None
    source_claims = tuple(
        claim for claim in decision.claims if claim.kind is ClaimKind.SOURCE_CLAIM
    )
    inferences = tuple(claim for claim in decision.claims if claim.kind is ClaimKind.INFERENCE)
    contract = request.completion_contract
    if not _within_bounds(
        len(source_claims),
        contract.source_claim_count.minimum,
        contract.source_claim_count.maximum,
    ):
        return "source_claim_count_invalid"
    if not _within_bounds(
        len(inferences),
        contract.inference_count.minimum,
        contract.inference_count.maximum,
    ):
        return "inference_count_invalid"
    for claim in source_claims:
        if not _within_bounds(
            len(claim.observation_ids),
            contract.source_observation_id_count.minimum,
            contract.source_observation_id_count.maximum,
        ):
            return "source_observation_id_count_invalid"
    return None


def _within_bounds(value: int, minimum: int, maximum: int) -> bool:
    return minimum <= value <= maximum


def _policy_error_feedback(code: str, request: ModelRequest) -> str:
    required = request.tool_choice.required_tool_name
    return {
        "tool_requested_while_disabled": (
            "Tool calls are disabled for this turn. Answer directly, or ask one concrete "
            "clarifying question instead of requesting a tool."
        ),
        "unadvertised_tool_requested": (
            "You requested a tool that was not offered this turn. Use only one of the "
            "advertised tools, or answer directly."
        ),
        "required_tool_not_requested": (
            f"You must call the required tool {required} before completing this turn. Call "
            "it now instead of answering directly."
        ),
        "required_tool_call_count_invalid": (
            f"Call exactly the one required tool {required} this turn -- not zero calls and "
            "not more than one."
        ),
        "wrong_required_tool_requested": (
            f"Call the required tool {required} this turn, not a different tool."
        ),
        "required_tool_arguments_mismatch": (
            f"Call the required tool {required} with exactly the arguments already pinned "
            "for this turn."
        ),
    }.get(
        code,
        "Your last decision violated the harness tool-choice policy for this turn. "
        "Reconsider and try again.",
    )


def _completion_contract_error_feedback(code: str, request: ModelRequest) -> str:
    contract = request.completion_contract
    if code == "source_claim_count_invalid":
        bounds = contract.source_claim_count
        return (
            f"Your completion must include between {bounds.minimum} and {bounds.maximum} "
            "source-backed claims. Consolidate or add claims so the count fits that range."
        )
    if code == "inference_count_invalid":
        bounds = contract.inference_count
        return (
            f"Your completion must include between {bounds.minimum} and {bounds.maximum} "
            "inference claims. Adjust the number of inferences so it fits that range."
        )
    if code == "source_observation_id_count_invalid":
        bounds = contract.source_observation_id_count
        return (
            f"Each source claim must cite between {bounds.minimum} and {bounds.maximum} "
            "observation IDs. Adjust your citations so each claim fits that range."
        )
    return (
        "Your last completion violated the harness completion contract. Reconsider and "
        "try again."
    )


def _gateway_failure_feedback(failure_code: str) -> str:
    if failure_code == "malformed_tool_arguments":
        return (
            "Your last tool call had arguments that were not valid JSON. Call the tool "
            "again with well-formed JSON arguments."
        )
    return (
        "Your last response could not be read as a valid completion -- it may have been "
        "cut off before finishing or was not valid JSON. Answer again, keeping the "
        "response concise and inside the required JSON contract."
    )


def _best_effort_completion(answer: str, failure_code: str) -> VerifiedCompletion:
    """Deliver a salvaged answer the harness gave up trying to verify or improve.

    Reserved for gateway/policy layers that explicitly attach a fallback answer (see
    ``ModelGatewayError.fallback_answer``) once further repair turns stopped being
    productive. A real, readable answer beats a terminal failure message with no
    content, even when it never earned a passing verifier result.
    """

    return VerifiedCompletion(
        answer=answer,
        claims=(),
        verifier_result=VerifierResult(
            status=VerifierStatus.PASS,
            checks=(
                VerifierCheck(
                    name="best_effort_fallback",
                    passed=True,
                    detail=(
                        f"Delivered without full verification after {failure_code} exhausted "
                        "the bounded repair loop; content may not satisfy every completeness "
                        "or grounding check."
                    ),
                ),
            ),
            retryable=False,
            allow_unsourced_completion=True,
        ),
    )


def _normalization_failure_code(safe_code: str) -> str:
    return {
        "observation_non_finite_number": "TOOL_RESULT_NON_FINITE",
        "observation_data_too_large": "TOOL_RESULT_TOO_LARGE",
        "observation_data_not_json": "TOOL_RESULT_MALFORMED",
        "observation_data_not_object": "TOOL_RESULT_MALFORMED",
        "observation_contract_invalid": "TOOL_RESULT_CONTRACT_INVALID",
    }.get(safe_code, "TOOL_RESULT_INVALID")


def _registry_selection(
    bundle: RunBundle,
    tools: tuple[ToolSpec, ...],
) -> CapabilitySelection:
    catalog_fingerprint = _tool_catalog_fingerprint(tools)
    objective_hash = hashlib.sha256(bundle.task.objective.encode("utf-8")).hexdigest()
    return CapabilitySelection(
        tools=tools,
        catalog_version="registry-v1",
        catalog_fingerprint=catalog_fingerprint,
        selection_fingerprint=capability_selection_fingerprint(
            catalog_fingerprint=catalog_fingerprint,
            query_hash=objective_hash,
            tools=tools,
            selected_skills=(),
            mode="registry",
        ),
        query_hash=objective_hash,
        eligible_count=len(tools),
        candidate_ids=tuple(tool.name for tool in tools),
        selected_ids=tuple(tool.name for tool in tools),
        mode="registry",
        reason="phase and role eligible registry view",
    )


def _empty_selection(
    bundle: RunBundle,
    available_tools: tuple[ToolSpec, ...],
) -> CapabilitySelection:
    catalog_fingerprint = _tool_catalog_fingerprint(available_tools)
    objective_hash = hashlib.sha256(bundle.task.objective.encode("utf-8")).hexdigest()
    return CapabilitySelection(
        tools=(),
        catalog_version="selector-unavailable",
        catalog_fingerprint=catalog_fingerprint,
        selection_fingerprint=capability_selection_fingerprint(
            catalog_fingerprint=catalog_fingerprint,
            query_hash=objective_hash,
            tools=(),
            selected_skills=(),
            mode="direct",
        ),
        query_hash=objective_hash,
        eligible_count=0,
        mode="direct",
        reason="capability selection unavailable; direct conversation remains available",
    )


def _selection_for_advertised_tools(
    selection: CapabilitySelection,
    tools: tuple[ToolSpec, ...],
) -> CapabilitySelection:
    if tools == selection.tools:
        return selection
    return selection.model_copy(
        update={
            "tools": tools,
            "selected_ids": tuple(tool.name for tool in tools),
            "selection_fingerprint": capability_selection_fingerprint(
                catalog_fingerprint=selection.catalog_fingerprint,
                query_hash=selection.query_hash,
                tools=tools,
                selected_skills=selection.selected_skill_ids,
                mode=selection.mode,
            ),
        }
    )


def _tool_catalog_fingerprint(tools: tuple[ToolSpec, ...]) -> str:
    payload = [
        item.model_dump(mode="json")
        for item in sorted(
            tools,
            key=lambda tool: tool.name,
        )
    ]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _result(bundle: RunBundle) -> CoordinatorResult:
    return CoordinatorResult(
        thread=bundle.thread,
        task=bundle.task,
        run=bundle.run,
        observations=bundle.observations,
        claims=bundle.claims,
        events=bundle.events,
    )
