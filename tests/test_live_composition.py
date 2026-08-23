from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import cast

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.config import Settings
from leo.domain.conversation import ConversationKind
from leo.harness.models import (
    ClaimKind,
    CompletionContract,
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    EventType,
    EvidenceToolRequirement,
    Observation,
    OriginRef,
    Run,
    RunStatus,
    ScopeKey,
    SourceRef,
    Task,
    Thread,
    ToolArgumentConstraint,
    TrustedScope,
)
from leo.harness.storage import InMemoryRunStore
from leo.harness.subagents import canonical_evidence_completion
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.slack.render import render_verified_result, verified_result_from_coordinator
from leo.live import (
    _EMPTY_MEMORY_SCOPE_INFERENCE,
    _child_evidence_requirements,
    _conversation_completion_guidance,
    _effective_tool_free_request,
    _ground_memory_observation,
    _requires_current_equity_screening_research,
    _requires_external_evidence,
    _requires_memory_search,
    _select_verified_web_provider,
    _thread_intent_routing_authority_ids,
    _thread_intent_routing_objective,
    run_live_conversation,
    run_live_quote,
)
from leo.memory.navigation import MemoryNavigationAuthority, membership_snapshot_hash
from leo.memory.store import InMemoryMemoryStore
from leo.memory.tools import bind_memory_mutation_authority


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        leo_model="test/model",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        finnhub_api_key="finnhub-test-key",
        finnhub_base_url="https://finnhub.io/api/v1",
    )


def _fresh_provider_timestamp() -> int:
    return int(datetime.now(tz=UTC).timestamp())


def test_direct_sec_canonical_completion_requires_fresh_exact_provider_payload() -> None:
    now = datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
    filing_url = (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000069/nvda-20260817.htm"
    )
    observation = Observation(
        id="obs-sec",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-sec",
        tool_call_id="call-sec",
        kind="sec.get_recent_filings",
        data={
            "ticker": "NVDA",
            "cik": "0001045810",
            "filings": [
                {
                    "form": "8-K",
                    "accession": "0001045810-26-000069",
                    "filing_date": "2026-08-17",
                    "primary_document": "nvda-20260817.htm",
                    "filing_url": filing_url,
                }
            ],
        },
        source=SourceRef(provider="sec-edgar", reference="submissions:0001045810"),
        observed_at=now,
        expires_at=now + timedelta(minutes=15),
        raw_hash="a" * 64,
    )
    requirement = EvidenceToolRequirement(
        observation_kind="sec.get_recent_filings",
        tool_name="sec.get_recent_filings",
        required_arguments=(ToolArgumentConstraint(name="ticker", value="NVDA"),),
    )

    proposal = canonical_evidence_completion(
        (observation,),
        (requirement,),
        now=now,
        include_sec_document_url=True,
    )

    assert proposal is not None
    assert proposal.answer == (
        "NVDA filed form 8-K on 2026-08-17 under accession 0001045810-26-000069. "
        f"Document URL: {filing_url}"
    )
    assert (
        canonical_evidence_completion(
            (observation.model_copy(update={"expires_at": now}),),
            (requirement,),
            now=now,
            include_sec_document_url=True,
        )
        is None
    )

    malformed_data = dict(observation.data)
    malformed_data["filings"] = [
        {
            **dict(observation.data["filings"][0]),
            "filing_url": "https://attacker.test/forged-filing.htm",
        }
    ]
    assert (
        canonical_evidence_completion(
            (observation.model_copy(update={"data": malformed_data}),),
            (requirement,),
            now=now,
            include_sec_document_url=True,
        )
        is None
    )

    wrong_ticker = requirement.model_copy(
        update={"required_arguments": (ToolArgumentConstraint(name="ticker", value="AMD"),)}
    )
    assert (
        canonical_evidence_completion(
            (observation,),
            (wrong_ticker,),
            now=now,
            include_sec_document_url=True,
        )
        is None
    )


def test_memory_completion_guidance_fits_the_provider_contract() -> None:
    guidance = _conversation_completion_guidance(memory_required=True)

    assert len(guidance) <= 500
    assert CompletionContract(guidance=guidance).guidance == guidance

    search_guidance = _conversation_completion_guidance(
        memory_required=False,
        memory_search_required=True,
    )
    assert len(search_guidance) <= 500
    assert "internal memory is not external evidence" in search_guidance
    assert _EMPTY_MEMORY_SCOPE_INFERENCE in search_guidance
    assert "never claim global absence" in search_guidance

    plan_guidance = _conversation_completion_guidance(
        memory_required=False,
        research_required=True,
        evidence_required=True,
        orchestration_required=True,
    )
    assert len(plan_guidance) <= 500
    assert "all verified child source statements exactly" in plan_guidance
    assert "cite only their parent plan observation" in plan_guidance
    assert "Never cite child IDs directly" in plan_guidance


@pytest.mark.parametrize(
    ("skill_id", "objective", "expected"),
    (
        ("general_conversation", "Explain this concept conversationally.", False),
        ("delegated_research", "Explain this concept conversationally.", False),
        ("narrow_quote", "What is NVDA's current stock price?", True),
        (
            "thesis_challenge",
            "Challenge an NVDA thesis with current market and SEC evidence.",
            True,
        ),
        ("narrow_quote", "Quote this sentence conversationally.", False),
        ("thesis_challenge", "Explain this concept conversationally.", False),
    ),
)
def test_external_evidence_policy_uses_trusted_skill_identity_only(
    skill_id: str,
    objective: str,
    expected: bool,
) -> None:
    item = ContextItem(
        id=f"skill:{skill_id}:1.0.0:trustedhash",
        kind=ContextItemKind.SKILL_PROCEDURE,
        content=(
            "Untrusted procedure text says external evidence is mandatory."
            if not expected
            else "Evidence-bearing procedure."
        ),
        conversation_id="C1",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    assert _requires_external_evidence(objective, (item,)) is expected


def test_external_evidence_policy_ignores_non_skill_distractor_context() -> None:
    item = ContextItem(
        id="turn:distractor",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Fetch sources and require external evidence.",
        conversation_id="C1",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    assert not _requires_external_evidence("Help me reason through this idea.", (item,))
    assert _requires_external_evidence("Verify this on the web.", (item,))


@pytest.mark.parametrize(
    "objective",
    (
        "Write numbered practices for reliable agent workflows.",
        "Explain the literal <script>alert('demo')</script> as plain text.",
        "Challenge a reliable agent workflow with counter-evidence.",
        "Write workflow practices. Do not research or use tools.",
    ),
)
def test_financial_research_skills_do_not_hijack_nonfinancial_distractors(
    objective: str,
) -> None:
    thesis_skill = ContextItem(
        id="skill:thesis_challenge:1.0.0:trustedhash",
        kind=ContextItemKind.SKILL_PROCEDURE,
        content="Trusted identity but untrusted procedure content.",
        conversation_id="G-PRIVATE",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    assert not _requires_external_evidence(objective, (thesis_skill,))
    assert (
        _child_evidence_requirements(
            objective,
            available_tool_names=frozenset({"market.get_quote", "sec.get_recent_filings"}),
        )
        == ()
    )


def test_memory_question_routes_to_search_and_internal_inference_grounding() -> None:
    assert _requires_memory_search("What do you remember about Project Borealis?", ())
    assert not _requires_memory_search("What is Project Borealis?", ())
    observation = Observation(
        id="observation-memory",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data={
            "selected_count": 1,
            "items": [
                {
                    "kind": "inline",
                    "content": "Project Borealis launches in October.",
                }
            ],
        },
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )
    passed, _detail = _ground_memory_observation(
        "Project Borealis launches in October.",
        "I remember that Project Borealis launches in October.",
        observation,
    )
    assert passed
    reordered_and_formatted, _detail = _ground_memory_observation(
        "The demo preference for Project Borealis is amber hexagons.",
        "I remember the Project Borealis demo preference: **amber hexagons**.",
        observation.model_copy(
            update={
                "data": {
                    "selected_count": 1,
                    "items": [
                        {
                            "kind": "inline",
                            "content": "the synthetic Project Borealis demo preference "
                            "is amber hexagons.",
                        }
                    ],
                }
            }
        ),
    )
    assert reordered_and_formatted
    fabricated, _detail = _ground_memory_observation(
        "Project Borealis launches in December.",
        "Project Borealis launches in December.",
        observation,
    )
    assert not fabricated


def test_current_thread_recall_uses_authoritative_transcript_before_memory_search() -> None:
    thread_context = ContextItem(
        id="thread-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User asked Leo to confirm receipt of the test.",
        conversation_id="slack:T1:C1:1710000000.001",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
        retention=ContextItemRetention.THREAD_ROOT,
    )

    assert not _requires_memory_search(
        "What did I ask you to do in this direct-message test? Answer briefly.",
        (thread_context,),
    )
    assert not _requires_memory_search(
        "What marker did I ask you to note? Answer with the marker only.",
        (thread_context,),
    )
    assert _requires_memory_search("Search your memory for Project Borealis.", (thread_context,))


@pytest.mark.parametrize(
    "objective",
    (
        "What do you remember about Project Borealis?",
        "Please recall our Project Borealis preference.",
        "What did we decide earlier about Project Borealis?",
        "What was our prior decision?",
        "Search your memory for Project Borealis.",
        "What did I ask you to remember about Borealis?",
        "What did we discuss?",
        "Show our stored memory about Borealis.",
        "What color was it again, and which conversation did that come from?",
    ),
)
def test_explicit_recall_intents_require_memory_search(objective: str) -> None:
    assert _requires_memory_search(objective, ())


@pytest.mark.parametrize(
    "objective",
    (
        "In one sentence, explain why exact conversation-local memory boundaries matter.",
        "Explain how computer memory works.",
        "Compare cache memory architectures.",
        "Write a poem about memories.",
        "Can you explain memory recall in psychology?",
        "Ask me one clarifying question before you answer.",
        "What is a prior probability?",
        "Design a decision tree.",
        "Why does memory safety matter?",
    ),
)
def test_conceptual_memory_discussion_never_forces_recall_search(objective: str) -> None:
    assert not _requires_memory_search(objective, ())


@pytest.mark.parametrize(
    "objective",
    (
        "I have not told you what either option is.",
        "I haven't told you which option I prefer.",
        "I did not tell you the constraints.",
        "I have not yet told you the two choices.",
        "I have not provided the missing details.",
        "I have not shared the alternatives with you.",
        "I have not said what the deadline is.",
        "The two options have not been provided yet.",
    ),
)
def test_current_missing_information_negations_never_force_memory_search(
    objective: str,
) -> None:
    assert not _requires_memory_search(objective, ())


def test_explicit_recall_still_wins_when_a_separate_clause_notes_missing_information() -> None:
    assert _requires_memory_search(
        "I have not told you today's choice, but search your memory for our prior decision.",
        (),
    )


def test_injected_memory_context_does_not_turn_a_conceptual_prompt_into_recall() -> None:
    memory_context = ContextItem(
        id="memory-conceptual",
        kind=ContextItemKind.MEMORY,
        content="An authorized memory item is already present.",
        conversation_id="C1",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    assert not _requires_memory_search(
        "Explain why exact conversation-local memory boundaries matter.",
        (memory_context,),
    )


@pytest.mark.asyncio
async def test_polite_remember_request_pins_confirmed_mutation_before_memory_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = (
        "Please remember for this conversation that Project Borealis's display preference "
        "is amber hexagons."
    )
    scope = ScopeKey(organization_id="org", strategy_id="strategy")
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T1:C1:1710000000.001",
        external_event_id="Ev1",
        external_channel_id="C1",
    )
    launch_ids = ("thread-1", "task-1", "run-1")
    mutation_authority = bind_memory_mutation_authority(
        scope=scope,
        team_id="T1",
        conversation_id="C1",
        conversation_kind=ConversationKind.CHANNEL,
        actor_id="U1",
        event_id="Ev1",
        task_id=launch_ids[1],
        run_id=launch_ids[2],
        message_reference="1710000000.001",
        objective=objective,
    )
    assert mutation_authority is not None
    navigation_authority = MemoryNavigationAuthority(
        scope=scope,
        team_id="T1",
        destination_id="C1",
        destination_kind=ConversationKind.CHANNEL,
        actor_id="U1",
        task_id=launch_ids[1],
        run_id=launch_ids[2],
        allowed_conversation_ids=("C1",),
        access_hash="a" * 64,
        membership_hash=membership_snapshot_hash(("C1",)),
        current_thread_namespace_id=origin.external_thread_id,
    )
    clock = FixedClock(datetime(2026, 8, 21, tzinfo=UTC))
    run_store = InMemoryRunStore(clock, SequentialIdGenerator())
    await run_store.seed(
        Thread(id=launch_ids[0], scope=scope, origin=origin),
        Task(id=launch_ids[1], thread_id=launch_ids[0], scope=scope, objective=objective),
        Run(id=launch_ids[2], task_id=launch_ids[1], scope=scope),
    )
    memory_store = InMemoryMemoryStore()
    monkeypatch.setattr("leo.live.PostgresRunStore", lambda *_args: run_store)
    monkeypatch.setattr("leo.live.PostgresMemoryStore", lambda *_args: memory_store)
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        advertised = {tool["function"]["name"] for tool in payload["tools"]}
        assert {"memory_remember", "memory_search"}.issubset(advertised)
        if model_calls == 1:
            assert user_payload["observations"] == []
            assert user_payload["tool_choice_policy"] == {
                "mode": "required",
                "required_tool_name": "memory.remember",
                "required_arguments": [],
            }
            assert (
                "explicit memory command is confirmed"
                in user_payload["completion_contract"]["guidance"]
            )
            assert payload["tool_choice"] == {
                "type": "function",
                "function": {"name": "memory_remember"},
            }
            return httpx.Response(
                200,
                json={
                    "id": "gen-memory-tool",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-memory",
                                        "type": "function",
                                        "function": {
                                            "name": "memory_remember",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )

        assert payload["tool_choice"] == "auto"
        assert [item["kind"] for item in user_payload["observations"]] == ["memory.remember"]
        return httpx.Response(
            200,
            json={
                "id": "gen-memory-confirmation",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "I remembered that for this conversation.",
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    sessions = cast(async_sessionmaker[AsyncSession], object())
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings().model_copy(update={"finnhub_api_key": None}),
            client=client,
            objective=objective,
            trusted_scope=TrustedScope(
                namespace=scope,
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
            origin=origin,
            sessions=sessions,
            launch_ids=launch_ids,
            memory_authority=mutation_authority,
            memory_navigation_authority=navigation_authority,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.tool_calls == 1
    assert [observation.kind for observation in result.observations] == ["memory.remember"]
    assert model_calls == 2
    record_ids = tuple(memory_store._records)
    assert len(record_ids) == 1
    revision = await memory_store.current(scope, record_ids[0])
    assert revision is not None
    assert revision.content == "Project Borealis's display preference is amber hexagons."


def test_mpim_tell_us_prompt_is_direct_conversation_not_past_exchange_recall() -> None:
    mpim_context = ContextItem(
        id="turn:G-MPIM",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="The group is having a light conversational exchange.",
        conversation_id="G-MPIM",
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    assert not _requires_memory_search(
        "Tell us a friendly one-sentence joke about two robots organizing a bookshelf.",
        (mpim_context,),
    )


@pytest.mark.parametrize(
    ("statement", "answer"),
    (
        (
            "The Project Borealis demo preference is blue hexagons.",
            "I remember that the Project Borealis demo preference is blue hexagons.",
        ),
        (
            "The Project Borealis demo preference is not amber hexagons.",
            "I remember that the Project Borealis demo preference is not amber hexagons.",
        ),
        (
            "The Project Borealis demo preference is always amber hexagons.",
            "I remember that the Project Borealis demo preference is always amber hexagons.",
        ),
    ),
)
def test_memory_grounding_rejects_wrong_value_negation_and_added_fact(
    statement: str,
    answer: str,
) -> None:
    observation = Observation(
        id="observation-borealis",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data={
            "selected_count": 1,
            "items": [
                {
                    "kind": "inline",
                    "content": "the synthetic Project Borealis demo preference is amber hexagons.",
                }
            ],
        },
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )

    passed, _detail = _ground_memory_observation(statement, answer, observation)

    assert not passed


def test_memory_grounding_never_combines_content_across_results() -> None:
    observation = Observation(
        id="observation-split-memory",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data={
            "selected_count": 2,
            "items": [
                {"kind": "inline", "content": "Project Borealis uses amber."},
                {"kind": "inline", "content": "Project Orion uses hexagons."},
            ],
        },
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )

    passed, _detail = _ground_memory_observation(
        "Project Borealis uses amber hexagons.",
        "I remember that Project Borealis uses amber hexagons.",
        observation,
    )

    assert not passed


def test_empty_memory_search_grounds_only_the_canonical_scoped_negative() -> None:
    observation = Observation(
        id="observation-empty-memory",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data={"items": [], "selected_count": 0},
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )

    passed, _detail = _ground_memory_observation(
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        observation,
    )
    global_absence, _detail = _ground_memory_observation(
        "No memory about Project Borealis exists anywhere.",
        "No memory about Project Borealis exists anywhere.",
        observation,
    )

    assert passed
    assert not global_absence


@pytest.mark.parametrize(
    "data",
    (
        {"items": [], "selected_count": 1},
        {
            "items": [{"kind": "inline", "content": "Project Borealis is amber."}],
            "selected_count": 0,
        },
        {"items": [], "selected_count": "0"},
    ),
)
def test_empty_memory_search_rejects_malformed_count_item_combinations(
    data: dict[str, object],
) -> None:
    observation = Observation(
        id="observation-malformed-memory",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data=data,
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )

    passed, detail = _ground_memory_observation(
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        observation,
    )

    assert not passed
    assert "malformed" in detail.lower()


def test_positive_memory_search_cannot_ground_the_canonical_negative() -> None:
    observation = Observation(
        id="observation-positive-memory",
        scope=ScopeKey(organization_id="org", strategy_id="domain"),
        run_id="run-memory",
        tool_call_id="call-memory",
        kind="memory.search",
        data={
            "items": [{"kind": "inline", "content": "Project Borealis is amber."}],
            "selected_count": 1,
        },
        source=SourceRef(provider="leo_memory", reference="query-hash"),
        observed_at=datetime(2026, 8, 21, tzinfo=UTC),
        raw_hash="a" * 64,
    )

    passed, _detail = _ground_memory_observation(
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        _EMPTY_MEMORY_SCOPE_INFERENCE,
        observation,
    )

    assert not passed


@pytest.mark.asyncio
async def test_live_composition_runs_two_turn_fresh_context_contract() -> None:
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "finnhub.io":
            assert request.url.params["symbol"] == "NVDA"
            return httpx.Response(
                200,
                json={
                    "c": 181.25,
                    "d": 1.5,
                    "dp": 0.83,
                    "h": 183,
                    "l": 178,
                    "o": 179,
                    "pc": 179.75,
                    "t": _fresh_provider_timestamp(),
                },
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["provider"]["require_parameters"] is True
        completion_schema = payload["response_format"]["json_schema"]["schema"]
        assert {"answer", "source_claims", "inferences"}.issubset(completion_schema["required"])
        source_claim_schema = completion_schema["properties"]["source_claims"]
        inference_schema = completion_schema["properties"]["inferences"]
        source_ids_schema = completion_schema["$defs"]["_SourceClaimPayload"]["properties"][
            "observation_ids"
        ]
        assert source_claim_schema["minItems"] == source_claim_schema["maxItems"] == 1
        assert source_ids_schema["minItems"] == source_ids_schema["maxItems"] == 1
        assert inference_schema["minItems"] == inference_schema["maxItems"] == 0
        trusted_guidance = (
            "In both the answer and source claim, copy NVDA and the exact numeric "
            "observation.data.price without rounding. Do not make separate change, high, low, "
            "open, or previous-close claims."
        )
        system = "".join(block["text"] for block in payload["messages"][0]["content"])
        assert trusted_guidance in system
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["completion_contract"]["guidance"] == trusted_guidance
        if not user_payload["observations"]:
            assert payload["tool_choice"] == {
                "type": "function",
                "function": {"name": "market_get_quote"},
            }
            assert user_payload["tool_choice_policy"]["required_arguments"] == [
                {"name": "symbol", "value": "NVDA"}
            ]
            assert "enum" not in source_ids_schema["items"]
            return httpx.Response(
                200,
                json={
                    "id": "gen-tool",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-quote",
                                        "type": "function",
                                        "function": {
                                            "name": "market_get_quote",
                                            "arguments": '{"symbol":"NVDA"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )

        observation_id = user_payload["observations"][0]["id"]
        assert source_ids_schema["items"]["enum"] == [observation_id]
        assert payload["tool_choice"] == "auto"
        statement = "NVDA is quoted at 181.25 in the current Finnhub observation."
        return httpx.Response(
            200,
            json={
                "id": "gen-completion",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation_id],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    admitted_scope = TrustedScope(
        namespace=ScopeKey(
            organization_id="org-from-slack-mapping",
            strategy_id="strategy-from-slack-mapping",
        ),
        actor_id="U-SLACK",
        roles=frozenset({"researcher"}),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_quote(
            settings=_settings(),
            client=client,
            symbol="nvda",
            objective="Report the current NVDA quote using an allowed market tool.",
            trusted_scope=admitted_scope,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.thread.scope == admitted_scope.namespace
    assert result.task.scope == admitted_scope.namespace
    assert result.run.scope == admitted_scope.namespace
    assert model_calls == 2
    assert len(result.observations) == 1
    assert len(result.claims) == 1


@pytest.mark.asyncio
async def test_live_composition_preserves_safe_openrouter_failure_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(401, headers={"x-request-id": "req-safe"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_quote(
            settings=_settings(),
            client=client,
            symbol="NVDA",
            objective="Report the current NVDA quote using an allowed market tool.",
        )

    assert result.run.status is RunStatus.FAILED
    failed = next(event for event in result.events if event.type is EventType.RUN_FAILED)
    assert failed.payload["reason"] == "model_gateway_error:http_401"
    assert failed.payload["detail"] == "OpenRouter returned HTTP 401; request_id=req-safe"


@pytest.mark.asyncio
async def test_live_composition_rejects_wrong_symbol_before_finnhub_execution() -> None:
    finnhub_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal finnhub_calls
        if request.url.host == "finnhub.io":
            finnhub_calls += 1
            return httpx.Response(500)
        return httpx.Response(
            200,
            json={
                "id": "gen-wrong-symbol",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-aapl",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_quote",
                                        "arguments": '{"symbol":"AAPL"}',
                                    },
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_quote(
            settings=_settings(),
            client=client,
            symbol="NVDA",
            objective="Report the current NVDA quote using an allowed market tool.",
        )

    # Wrong required arguments are retried with corrective feedback instead of
    # instantly failing the run; this fixture handler always answers with the
    # wrong symbol, so the bounded retry loop exhausts its budget without ever
    # executing the tool.
    assert result.run.status is RunStatus.BUDGET_EXHAUSTED
    assert result.run.usage.model_calls > 1
    assert result.run.usage.tool_calls == 0
    assert finnhub_calls == 0
    assert any("argument" in feedback.lower() for feedback in result.task.verifier_feedback)


@pytest.mark.asyncio
async def test_conversation_answers_arbitrary_contextual_prompt_without_finnhub() -> None:
    settings = _settings().model_copy(update={"finnhub_api_key": None})
    context_item = ContextItem(
        id="turn:prior",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: Call the demo Helios.\nLeo: I will remember that in this channel.",
        conversation_id="C1",
        source_scope=ScopeKey(
            organization_id="org-from-slack",
            strategy_id="default-domain",
        ),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["objective"] == "What did we call the demo?"
        assert user_payload["scoped_context"][0]["content"].startswith("User: Call")
        tool_names = {tool["function"]["name"] for tool in payload["tools"]}
        assert tool_names == {
            "agent_delegate_research",
            "agent_execute_research_plan",
            "tool_search",
            "tool_describe",
        }
        return httpx.Response(
            200,
            json={
                "id": "gen-chat",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "You called the demo Helios.",
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="What did we call the demo?",
            context_items=(context_item,),
            trusted_scope=TrustedScope(
                namespace=context_item.source_scope,
                actor_id="U1",
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == "You called the demo Helios."
    assert result.run.usage.tool_calls == 0
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["selection_mode"] == "direct"
    assert context_event.payload["capability_candidates"] == []
    assert len(context_event.payload["catalog_fingerprint"]) == 64
    assert len(context_event.payload["selection_fingerprint"]) == 64


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("conversation_id", "objective", "answer"),
    (
        (
            "G-PRIVATE",
            "Answer this conversational question from our private channel context: "
            "why do scoped boundaries matter?",
            "Scoped boundaries keep one private channel's discussion separate from another.",
        ),
        (
            "G-MPIM",
            "Answer this conversational question for our group from this exact context: "
            "what should we discuss next?",
            "You could decide which demo workflow the group wants to explore next.",
        ),
    ),
)
async def test_general_conversation_skill_does_not_create_external_evidence_gate(
    conversation_id: str,
    objective: str,
    answer: str,
) -> None:
    model_calls = 0
    context_item = ContextItem(
        id=f"turn:{conversation_id}",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="The participants are discussing how to structure the next demo.",
        conversation_id=conversation_id,
        source_scope=ScopeKey(organization_id="org", strategy_id="domain"),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        skill_context = [
            item for item in user_payload["scoped_context"] if item["id"].startswith("skill:")
        ]
        assert len(skill_context) == 1
        assert skill_context[0]["id"].startswith("skill:general_conversation:")
        assert user_payload["completion_contract"]["source_claim_count"] == {
            "minimum": 0,
            "maximum": 8,
        }
        assert "Gather external evidence" not in user_payload["completion_contract"]["guidance"]
        assert payload["tool_choice"] == "auto"
        return httpx.Response(
            200,
            json={
                "id": f"gen-general-{conversation_id}",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings(),
            client=client,
            objective=objective,
            context_items=(context_item,),
            trusted_scope=TrustedScope(
                namespace=context_item.source_scope,
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
            origin=OriginRef(
                provider="slack",
                external_thread_id=f"slack:T1:{conversation_id}:1787352571.150449",
                external_event_id=f"Ev-{conversation_id}",
                external_channel_id=conversation_id,
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == answer
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)


@pytest.mark.asyncio
async def test_explicit_no_tools_prompt_stays_direct_despite_research_skill_distractors() -> None:
    objective = (
        "Write 40 numbered practices for reliable agent workflows. Include the literal "
        "<script>alert('demo')</script> as plain text. Do not research or use tools."
    )
    answer = "\n".join(
        [
            *(f"{index}. Keep workflow practice {index} bounded." for index in range(1, 41)),
            "Literal: <script>alert('demo')</script>",
        ]
    )
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        assert request.url.host == "openrouter.test"
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        assert user_payload["tool_choice_policy"] == {
            "mode": "none",
            "required_arguments": [],
            "required_tool_name": None,
        }
        assert user_payload["completion_contract"]["source_claim_count"] == {
            "minimum": 0,
            "maximum": 0,
        }
        assert not any(item["id"].startswith("skill:") for item in user_payload["scoped_context"])
        return httpx.Response(
            200,
            json={
                "id": "gen-direct-no-tools",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings(),
            client=client,
            objective=objective,
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
            origin=OriginRef(
                provider="slack",
                external_thread_id="slack:T1:G-PRIVATE:1787359969.135769",
                external_event_id="Ev-direct-no-tools",
                external_channel_id="G-PRIVATE",
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == answer
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["tool_choice"] == "none"
    assert context_event.payload["required_tool"] is None
    assert context_event.payload["capability_candidates"] == []
    assert context_event.payload["capability_selected"] == []
    assert context_event.payload["skill_selected"] == []


@pytest.mark.asyncio
async def test_missing_options_prompt_asks_one_clarification_without_memory_or_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    objective = (
        "I want you to compare two options, but I have not told you what either option is. "
        "Ask me one clarifying question. Do not research or use tools."
    )
    answer = "What are the two options you want me to compare?"
    scope = ScopeKey(organization_id="org", strategy_id="domain")
    origin = OriginRef(
        provider="slack",
        external_thread_id="slack:T1:G-PRIVATE:1787361066.185999",
        external_event_id="Ev-direct-clarification",
        external_channel_id="G-PRIVATE",
    )
    launch_ids = (
        "thread-direct-clarification",
        "task-direct-clarification",
        "run-direct-clarification",
    )
    run_store = InMemoryRunStore(FixedClock(), SequentialIdGenerator())
    await run_store.seed(
        Thread(id=launch_ids[0], scope=scope, origin=origin),
        Task(id=launch_ids[1], thread_id=launch_ids[0], scope=scope, objective=objective),
        Run(id=launch_ids[2], task_id=launch_ids[1], scope=scope),
    )
    monkeypatch.setattr("leo.live.PostgresRunStore", lambda *_args: run_store)
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        model_calls += 1
        assert request.url.host == "openrouter.test"
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        assert user_payload["tool_choice_policy"]["required_tool_name"] is None
        assert user_payload["completion_contract"]["source_claim_count"] == {
            "minimum": 0,
            "maximum": 0,
        }
        return httpx.Response(
            200,
            json={
                "id": "gen-missing-options",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings(),
            client=client,
            objective=objective,
            memory_navigation_authority=MemoryNavigationAuthority(
                scope=scope,
                actor_id="U1",
                team_id="T1",
                destination_id="G-PRIVATE",
                destination_kind=ConversationKind.CHANNEL,
                access_hash="a" * 64,
                membership_hash=membership_snapshot_hash(("G-PRIVATE",)),
                task_id=launch_ids[1],
                run_id=launch_ids[2],
                allowed_conversation_ids=("G-PRIVATE",),
                current_thread_namespace_id=origin.external_thread_id,
            ),
            trusted_scope=TrustedScope(
                namespace=scope,
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
            origin=origin,
            sessions=cast(async_sessionmaker[AsyncSession], object()),
            launch_ids=launch_ids,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == answer
    assert result.run.usage.model_calls == 1
    assert result.run.usage.tool_calls == 0
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["tool_choice"] == "none"
    assert context_event.payload["required_tool"] is None


@pytest.mark.asyncio
async def test_live_conversation_recalls_paraphrased_tool_and_preserves_direct_reasoning() -> None:
    model_calls = 0
    seen_tool_names: list[set[str]] = []
    seen_context_kinds: list[set[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "finnhub.io":
            assert request.url.params["symbol"] == "NVDA"
            return httpx.Response(
                200,
                json={
                    "c": 181.25,
                    "d": 1.5,
                    "dp": 0.83,
                    "h": 183,
                    "l": 178,
                    "o": 179,
                    "pc": 179.75,
                    "t": _fresh_provider_timestamp(),
                },
            )

        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        tool_names = {tool["function"]["name"] for tool in payload["tools"]}
        seen_tool_names.append(tool_names)
        seen_context_kinds.append({item["kind"] for item in user_payload["scoped_context"]})
        if not user_payload["observations"]:
            return httpx.Response(
                200,
                json={
                    "id": "gen-paraphrase-tool",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-quote",
                                        "type": "function",
                                        "function": {
                                            "name": "market_get_quote",
                                            "arguments": '{"symbol":"NVDA"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        observation_id = user_payload["observations"][0]["id"]
        statement = "NVDA is quoted at 181.25."
        return httpx.Response(
            200,
            json={
                "id": "gen-paraphrase-answer",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation_id],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings(),
            client=client,
            objective="Could you look up NVDA's latest stock price?",
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert "market_get_quote" in seen_tool_names[0]
    assert "web_fetch_public_text" not in seen_tool_names[0]
    assert "skill_procedure" in seen_context_kinds[0]
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == "NVDA is quoted at 181.25."
    assert result.run.usage.tool_calls == 1
    # The model selects the trusted route, then receives the fresh observation
    # and produces the conversational completion in a second model turn.
    assert model_calls == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "objective",
    [
        "What's NVDA trading at right now?",
        "What's NVDA's share price right now?",
        "Give me the current NVDA stock price.",
    ],
    ids=["trading-at", "share-price", "stock-price"],
)
async def test_live_current_quote_pins_tool_and_stops_repeated_fabricated_citations(
    objective: str,
) -> None:
    model_calls = 0
    finnhub_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal finnhub_calls, model_calls
        if request.url.host == "finnhub.io":
            finnhub_calls += 1
            raise AssertionError("the ignored required-tool policy must not call Finnhub")
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["observations"] == []
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "market_get_quote"},
        }
        assert user_payload["tool_choice_policy"]["required_arguments"] == [
            {"name": "symbol", "value": "NVDA"}
        ]
        assert user_payload["completion_contract"]["source_claim_count"] == {
            "minimum": 1,
            "maximum": 1,
        }
        statement = "NVDA is quoted at a fabricated value."
        return httpx.Response(
            200,
            json={
                "id": f"gen-fabricated-{model_calls}",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": ["obs-fabricated"],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = _settings().model_copy(update={"leo_max_model_turns": 4})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    # Ignoring the required tool and fabricating a quote gets one retry with
    # corrective feedback instead of instantly failing on the first turn -- but
    # the core safety property holds either way: a decision carrying an
    # unverified source claim is never eligible for the bounded-loop's
    # best-effort fallback (only claim-free prose is), so once this fixture
    # handler repeats the identical fabricated claim, the run still fails
    # closed. The fabricated citation is never delivered, no observation is
    # ever fabricated, and Finnhub is never reached.
    assert result.run.status is RunStatus.FAILED
    assert result.run.terminal_reason == "model_gateway_error:deliberation_repeated_decision"
    assert result.run.final_output is None
    assert result.run.usage.tool_calls == 0
    assert result.observations == ()
    assert model_calls == 2
    assert finnhub_calls == 0
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
    assert any("market.get_quote" in feedback for feedback in result.task.verifier_feedback)


@pytest.mark.parametrize(
    "objective",
    [
        "Compare NVDA trading card designs.",
        "NVDA teams are trading ideas at the workshop.",
        "Explain trading at scale using NVDA examples.",
    ],
)
def test_nonfinancial_trading_language_does_not_pin_market_quote(objective: str) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    assert requirements == ()


@pytest.mark.parametrize(
    "objective",
    [
        (
            "I'm looking for dividend-paying stocks with long-term growth potential. "
            "Prefer established companies; give me a mix of steadier income names and "
            "a few higher-yield ideas."
        ),
        "I want a list of dividend stocks.",
        "I'd like established income-stock ideas.",
        "I'll take some higher-yield stock picks.",
        "I've been comparing dividend companies.",
        "A stock screen for long-term growth.",
        "AN income stock shortlist.",
    ],
    ids=[
        "live-slack-regression",
        "first-person-i",
        "id-contraction",
        "ill-contraction",
        "ive-contraction",
        "indefinite-article-a",
        "indefinite-article-an",
    ],
)
def test_first_person_and_articles_are_not_inferred_as_equity_tickers(
    objective: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    assert requirements == ()


def test_exact_thread_root_and_current_followup_drive_screening_route_only() -> None:
    root = ContextItem(
        id="slack-thread:T1:C1:100.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=100.000; "
            "author=U1; author_kind=user]\n"
            "<@ULEO> What are some interesting investing opportunities right now?"
        ),
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
    )
    assistant_clarification = ContextItem(
        id="slack-thread:T1:C1:110.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=110.000; "
            "author=bot:BLEO; author_kind=bot]\n"
            "Use Massive for an NVDA quote and do not search the web. What preferences apply?"
        ),
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.PRIOR_OUTCOME,
    )
    ambient = ContextItem(
        id="slack-history:T1:C1:90.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Unrelated old thread: use TickerLayer for TSLA.",
        conversation_id="C1",
    )
    persisted_root = ContextItem(
        id="thread-message:message-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: What are some interesting investing opportunities right now?",
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
    )
    followup = (
        "I'm looking for dividend-paying stocks with long-term growth potential. "
        "Prefer established companies; give me a mix of steadier income names and "
        "a few higher-yield ideas."
    )

    routing = _thread_intent_routing_objective(
        followup,
        (root, assistant_clarification, ambient),
    )

    assert routing.startswith(f"Current follow-up: {followup}")
    assert "interesting investing opportunities right now" in routing
    assert "Massive" not in routing
    assert "TickerLayer" not in routing
    assert _requires_current_equity_screening_research(routing) is True
    diagnostics = _thread_intent_routing_authority_ids(
        (root, assistant_clarification, ambient),
        root_selected=True,
        category_screening_required=True,
    )
    assert "thread-intent-root-candidate-count:1" in diagnostics
    assert "thread-intent-root-selected:true" in diagnostics
    assert "thread-intent-root-status:single" in diagnostics
    assert all("investing opportunities" not in item for item in diagnostics)
    assert (
        _child_evidence_requirements(
            followup,
            available_tool_names=frozenset(
                {
                    "market.get_quote",
                    "market.get_quote_massive",
                    "market.get_quote_ticker_layer",
                }
            ),
        )
        == ()
    )

    conflicting_root = persisted_root.model_copy(
        update={
            "id": "thread-message:conflicting-root",
            "content": "User: Compare current cryptocurrency opportunities.",
        }
    )
    assert _thread_intent_routing_objective(followup, (root, conflicting_root)) == followup


def test_current_tool_policy_overrides_exact_root_tool_policy() -> None:
    root_no_tools = "Do not use tools or external research for this request."

    assert (
        _effective_tool_free_request(
            "Give me a concise answer.",
            thread_root_objective=root_no_tools,
        )
        is True
    )
    assert (
        _effective_tool_free_request(
            "Search the web with Tavily and verify the current result.",
            thread_root_objective=root_no_tools,
        )
        is False
    )
    assert (
        _effective_tool_free_request(
            "Do not use tools; answer only from the admitted thread.",
            thread_root_objective="Search Exa for current stock ideas.",
        )
        is True
    )


@pytest.mark.asyncio
async def test_live_exact_thread_dividend_screen_forces_verified_research_without_quote() -> None:
    from leo.worker.slack_conversation import _merge_authorized_context

    followup = (
        "I'm looking for dividend-paying stocks with long-term growth potential. "
        "Prefer established companies; give me a mix of steadier income names and "
        "a few higher-yield ideas."
    )
    root = ContextItem(
        id="slack-thread:T1:C1:100.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=100.000; "
            "author=U1; author_kind=user]\n"
            "<@ULEO> What are some interesting investing opportunities right now?"
        ),
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
    )
    assistant_clarification = ContextItem(
        id="slack-thread:T1:C1:110.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=110.000; "
            "author=bot:BLEO; author_kind=bot]\n"
            "Which goals, risk tolerance, and time horizon should I use?"
        ),
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.UNRESOLVED_QUESTION,
    )
    durable_prior_turn = ContextItem(
        id="turn:prior-task",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "User: What are some interesting investing opportunities right now?\n"
            "Leo: Which goals, risk tolerance, and time horizon should I use?"
        ),
        conversation_id="C1",
        retention=ContextItemRetention.RECENT,
    )
    durable_root = ContextItem(
        id="thread-message:durable-root",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="User: What are some interesting investing opportunities right now?",
        conversation_id="C1",
        source_actor_id="U1",
        retention=ContextItemRetention.THREAD_ROOT,
        budget_priority=100,
    )
    durable_progress = ContextItem(
        id="thread-message:durable-progress",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Assistant: Leo is working...",
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.RECENT,
    )
    durable_clarification = ContextItem(
        id="thread-message:durable-clarification",
        kind=ContextItemKind.CONVERSATION_TURN,
        content="Assistant: Which goals, risk tolerance, and time horizon should I use?",
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.UNRESOLVED_QUESTION,
    )
    slack_progress = ContextItem(
        id="slack-thread:T1:C1:105.000",
        kind=ContextItemKind.CONVERSATION_TURN,
        content=(
            "[Slack exact thread; team=T1; conversation=C1; message_ts=105.000; "
            "author=bot:BLEO; author_kind=bot]\nLeo is working..."
        ),
        conversation_id="C1",
        source_actor_id="bot:BLEO",
        retention=ContextItemRetention.RECENT,
    )
    combined_loader_context = (
        durable_prior_turn,
        durable_root,
        durable_progress,
        durable_clarification,
        root,
        slack_progress,
        assistant_clarification,
    )
    assert len(combined_loader_context) == 7
    assert (
        sum(item.retention is ContextItemRetention.THREAD_ROOT for item in combined_loader_context)
        == 2
    )
    production_context = _merge_authorized_context(
        combined_loader_context,
        allowed_conversation_ids=frozenset({"C1"}),
        destination_id="C1",
        team_id="T1",
        thread_root_ts="100.000",
        actor_id="U1",
    )
    assert len(production_context) == 6
    assert (
        sum(item.retention is ContextItemRetention.THREAD_ROOT for item in production_context) == 1
    )
    result_url = "https://example.com/dividend-research"
    title = "Established dividend companies"
    highlight = "Established dividend companies can combine income with long-term growth."
    statement = f'Exa highlight from "{title}" ({result_url}): {highlight}'
    exa_calls = 0
    model_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exa_calls, model_calls
        if request.url.host == "api.exa.ai":
            exa_calls += 1
            assert json.loads(request.content) == {
                "query": followup,
                "type": "auto",
                "contents": {"highlights": True},
            }
            return httpx.Response(
                200,
                json={
                    "requestId": "exa-dividend-screen",
                    "results": [
                        {
                            "title": title,
                            "url": result_url,
                            "id": result_url,
                            "highlights": [highlight],
                            "highlightScores": [0.9],
                        }
                    ],
                },
            )

        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["objective"] == followup
        observation = next(
            item for item in user_payload["observations"] if item["kind"] == "web.search_exa"
        )
        advertised = {item["function"]["name"] for item in payload["tools"]}
        assert "web_search_exa" in advertised
        return httpx.Response(
            200,
            json={
                "id": "dividend-screen-completion",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation["id"]],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        leo_model="test/model",
        openrouter_api_key="openrouter-test-key",
        openrouter_base_url="https://openrouter.test/api/v1",
        exa_api_key="exa-test-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=followup,
            context_items=production_context,
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == statement
    assert exa_calls == 1
    assert model_calls == 1
    assert result.run.usage.tool_calls == 1
    assert tuple(item.kind for item in result.observations) == ("web.search_exa",)
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["tool_choice"] == "required"
    assert context_event.payload["required_tool"] == "web.search_exa"
    source_manifest = context_event.payload["source_manifest"]
    assert isinstance(source_manifest, dict)
    included_source_ids = source_manifest["included_source_ids"]
    assert "thread-intent-routing-version:v1" in included_source_ids
    assert "thread-intent-root-candidate-count:1" in included_source_ids
    assert "thread-intent-root-selected:true" in included_source_ids
    assert "thread-intent-root-status:single" in included_source_ids
    assert "thread-intent-category-screening:true" in included_source_ids


@pytest.mark.parametrize(
    "objective",
    [
        "Show me UK dividend stocks with growth potential.",
        "Find EU equities with steady income.",
        "Compare Apple and Tesla investment opportunities.",
        "Compare NVDA and AAPL prices.",
    ],
)
def test_plural_equity_screening_and_multi_entity_prompts_do_not_pin_one_quote(
    objective: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    assert requirements == ()


@pytest.mark.parametrize(
    ("objective", "symbol"),
    [
        ("What's $I's price?", "I"),
        ("Look up ticker I price.", "I"),
        ("What is $F trading at?", "F"),
        ("Look up symbol F price.", "F"),
        ("What is PLTR's price?", "PLTR"),
    ],
)
def test_safely_disambiguated_symbols_pin_exactly_one_equity_quote(
    objective: str,
    symbol: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    assert len(requirements) == 1
    assert requirements[0].tool_name == "market.get_quote"
    assert requirements[0].required_arguments == (
        ToolArgumentConstraint(name="symbol", value=symbol),
    )


@pytest.mark.parametrize(
    "objective",
    [
        "I price dividend stocks for a living.",
        "F price?",
        "$I or $F price?",
        "Compare ticker I with ticker F prices.",
    ],
)
def test_bare_or_ambiguous_one_letter_symbols_do_not_pin_an_equity_quote(
    objective: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset({"market.get_quote"}),
    )

    assert requirements == ()


@pytest.mark.asyncio
async def test_live_latest_sec_lookup_pins_one_tool_without_thesis_hijack() -> None:
    model_calls = 0
    sec_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, sec_calls
        if request.url.host != "openrouter.test":
            sec_calls += 1
            raise AssertionError("the ignored required-tool policy must not call SEC")
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        assert user_payload["observations"] == []
        assert payload["tool_choice"] == {
            "type": "function",
            "function": {"name": "sec_get_recent_filings"},
        }
        assert user_payload["tool_choice_policy"]["required_arguments"] == [
            {"name": "ticker", "value": "NVDA"}
        ]
        assert user_payload["completion_contract"]["source_claim_count"] == {
            "minimum": 1,
            "maximum": 1,
        }
        assert all(
            not item["id"].startswith("skill:thesis_challenge:")
            for item in user_payload["scoped_context"]
        )
        statement = "NVDA filed fabricated metadata."
        return httpx.Response(
            200,
            json={
                "id": f"gen-sec-fabricated-{model_calls}",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": ["obs-fabricated"],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = _settings().model_copy(
        update={
            "sec_user_agent": "Leo demo leo-test@example.com",
            "leo_max_model_turns": 4,
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="What is NVIDIA's latest SEC filing metadata?",
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    # Ignoring the required tool and fabricating filing metadata gets one retry
    # with corrective feedback instead of instantly failing on the first turn --
    # but the core safety property holds either way: a decision carrying an
    # unverified source claim is never eligible for the bounded-loop's
    # best-effort fallback (only claim-free prose is), so once this fixture
    # handler repeats the identical fabricated claim, the run still fails
    # closed. The fabricated citation is never delivered, no observation is
    # ever fabricated, and SEC EDGAR is never reached.
    assert result.run.status is RunStatus.FAILED
    assert result.run.terminal_reason == "model_gateway_error:deliberation_repeated_decision"
    assert result.run.final_output is None
    assert result.run.usage.tool_calls == 0
    assert result.observations == ()
    assert model_calls == 2
    assert sec_calls == 0
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
    assert any("sec.get_recent_filings" in feedback for feedback in result.task.verifier_feedback)


@pytest.mark.asyncio
async def test_live_latest_sec_lookup_canonicalizes_fresh_tuple_and_document_url() -> None:
    model_calls = 0
    statement = (
        "The latest SEC filing for NVDA is form 8-K, dated 2026-08-17, "
        "with accession 0001045810-26-000069."
    )
    filing_url = (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000069/nvda-20260817.htm"
    )
    answer = (
        "NVDA filed form 8-K on 2026-08-17 under accession 0001045810-26-000069. "
        f"Document URL: {filing_url}"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "data.sec.gov":
            return httpx.Response(
                200,
                json={
                    "name": "NVIDIA CORP",
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "accessionNumber": ["0001045810-26-000069"],
                            "filingDate": ["2026-08-17"],
                            "primaryDocument": ["nvda-20260817.htm"],
                        }
                    },
                },
            )
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        if not user_payload["observations"]:
            return httpx.Response(
                200,
                json={
                    "id": "gen-sec-tool",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-sec",
                                        "type": "function",
                                        "function": {
                                            "name": "sec_get_recent_filings",
                                            "arguments": '{"ticker":"NVDA"}',
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        system = "".join(block["text"] for block in payload["messages"][0]["content"])
        assert (
            "ticker=NVDA; form=8-K; filing_date=2026-08-17; accession=0001045810-26-000069"
        ) in system
        observation_id = user_payload["observations"][0]["id"]
        return httpx.Response(
            200,
            json={
                "id": "gen-sec-answer",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": answer,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation_id],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = _settings().model_copy(update={"sec_user_agent": "Leo demo leo-test@example.com"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=(
                "Give me NVIDIA's latest SEC form, filing date, accession, and exact document URL."
            ),
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == answer
    assert len(result.observations) == 1
    assert len(result.claims) == 1
    assert result.run.usage.tool_calls == 1
    # Routing remains model-selected, and the second model turn receives the
    # typed SEC observation before producing the final answer.
    assert model_calls == 2


@pytest.mark.asyncio
async def test_live_multi_source_research_corrects_missing_sec_claim() -> None:
    model_calls = 0
    correction_feedback: list[str] = []
    quote_statement = "NVDA is quoted at 181.25."
    sec_statement = "NVDA filed form 10-Q on 2026-08-20 under accession 0001045810-26-000123."

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "finnhub.io":
            return httpx.Response(
                200,
                json={
                    "c": 181.25,
                    "d": 1.5,
                    "dp": 0.83,
                    "h": 183,
                    "l": 178,
                    "o": 179,
                    "pc": 179.75,
                    "t": _fresh_provider_timestamp(),
                },
            )
        if request.url.host == "data.sec.gov":
            return httpx.Response(
                200,
                json={
                    "name": "NVIDIA CORP",
                    "filings": {
                        "recent": {
                            "form": ["10-Q"],
                            "accessionNumber": ["0001045810-26-000123"],
                            "filingDate": ["2026-08-20"],
                            "primaryDocument": ["nvda-20260726.htm"],
                        }
                    },
                },
            )

        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        observations = user_payload["observations"]
        if not observations:
            content = None
            tool_calls: list[dict[str, object]] = [
                {
                    "id": "call-market",
                    "type": "function",
                    "function": {
                        "name": "market_get_quote",
                        "arguments": '{"symbol":"NVDA"}',
                    },
                },
                {
                    "id": "call-sec",
                    "type": "function",
                    "function": {
                        "name": "sec_get_recent_filings",
                        "arguments": '{"ticker":"NVDA","limit":1}',
                    },
                },
            ]
        else:
            by_kind = {item["kind"]: item["id"] for item in observations}
            if not user_payload["verifier_feedback"]:
                content = json.dumps(
                    {
                        "answer": quote_statement,
                        "source_claims": [
                            {
                                "statement": quote_statement,
                                "observation_ids": [by_kind["market.get_quote"]],
                            }
                        ],
                        "inferences": [],
                    }
                )
            else:
                correction_feedback.extend(user_payload["verifier_feedback"])
                content = json.dumps(
                    {
                        "answer": f"{quote_statement} {sec_statement}",
                        "source_claims": [
                            {
                                "statement": quote_statement,
                                "observation_ids": [by_kind["market.get_quote"]],
                            },
                            {
                                "statement": sec_statement,
                                "observation_ids": [by_kind["sec.get_recent_filings"]],
                            },
                        ],
                        "inferences": [],
                        "affected_assumption": "Demand remains durable.",
                        "uncertainty": "Market and filing evidence cover different time windows.",
                    }
                )
            tool_calls = []
        return httpx.Response(
            200,
            json={
                "id": f"gen-multi-source-{model_calls}",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
            },
        )

    settings = _settings().model_copy(
        update={
            "sec_user_agent": "Leo demo leo-test@example.com",
            "leo_max_model_turns": 4,
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=(
                "Compare an NVDA thesis with current market and SEC primary-source evidence "
                "and counter-evidence."
            ),
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == f"{quote_statement} {sec_statement}"
    assert len(result.observations) == 2
    assert [claim.kind for claim in result.claims] == [
        ClaimKind.SOURCE_CLAIM,
        ClaimKind.SOURCE_CLAIM,
        ClaimKind.AFFECTED_ASSUMPTION,
        ClaimKind.UNCERTAINTY,
    ]
    view = verified_result_from_coordinator(result)
    assert view.affected_assumption == "Demand remains durable."
    assert view.uncertainty == "Market and filing evidence cover different time windows."
    rendered = "".join(render_verified_result(view).chunks)
    assert "Facts\n" in rendered
    assert "Affected assumption: Demand remains durable." in rendered
    assert "Uncertainty: Market and filing evidence cover different time windows." in rendered
    assert "Research evidence, not financial advice." in rendered
    assert model_calls == 3
    assert correction_feedback
    failed = [event for event in result.events if event.type is EventType.VERIFICATION_FAILED]
    assert len(failed) == 1
    assert any(
        check["name"] == "research_required_kind_sec.get_recent_filings"
        and check["passed"] is False
        for check in failed[0].payload["checks"]
    )


@pytest.mark.asyncio
async def test_live_conversation_selects_parallel_market_sec_and_parent_tools() -> None:
    """The live Slack research phrasing must not collapse to a tool-less fallback."""

    seen_tool_names: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen_tool_names
        assert request.url.host == "openrouter.test"
        payload = json.loads(request.content)
        seen_tool_names = {tool["function"]["name"] for tool in payload["tools"]}
        return httpx.Response(
            200,
            json={
                "id": "gen-parallel-plan-intake",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "I need current provider evidence to answer.",
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = _settings().model_copy(
        update={
            "sec_user_agent": "Leo demo leo-test@example.com",
            "leo_max_model_turns": 2,
        }
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=(
                "Build and execute a two-step research plan: obtain NVDA's current market "
                "quote and latest SEC filing metadata in parallel where possible, delegate "
                "bounded read-only work, reconcile the evidence, and answer with sources."
            ),
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
            ),
        )

    # The required-orchestration-tool policy is still correctly enforced on both
    # turns (see the tool_choice/required_tool assertions below); the model
    # simply never complies. The first miss is retried with corrective feedback;
    # once it repeats the identical claim-free answer, the bounded loop's
    # best-effort fallback delivers that honest "I need evidence" text instead
    # of a terminal failure -- it never fabricates a claim, so this is safe.
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.final_output == "I need current provider evidence to answer."
    assert any(
        "agent.execute_research_plan" in feedback for feedback in result.task.verifier_feedback
    )
    assert {"market_get_quote", "sec_get_recent_filings"}.issubset(seen_tool_names)
    assert {
        "agent_delegate_research",
        "agent_execute_research_plan",
        "tool_search",
        "tool_describe",
    }.issubset(seen_tool_names)
    assert not any(event.type is EventType.VERIFICATION_FAILED for event in result.events)
    assert result.run.usage.tool_calls == 0
    context_event = next(event for event in result.events if event.type is EventType.CONTEXT_BUILT)
    assert context_event.payload["catalog_version"] != "selector-unavailable"
    assert context_event.payload["tool_choice"] == "required"
    assert context_event.payload["required_tool"] == "agent.execute_research_plan"


@pytest.mark.asyncio
async def test_live_conversation_progressively_searches_describes_then_executes() -> None:
    advertised_by_turn: list[set[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "finnhub.io":
            return httpx.Response(
                200,
                json={
                    "c": 181.25,
                    "d": 0,
                    "dp": 0,
                    "h": 181.25,
                    "l": 181.25,
                    "o": 181.25,
                    "pc": 181.25,
                    "t": _fresh_provider_timestamp(),
                },
            )

        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        advertised = {tool["function"]["name"] for tool in payload["tools"]}
        advertised_by_turn.append(advertised)
        observation_kinds = [item["kind"] for item in user_payload["observations"]]
        if observation_kinds == []:
            name = "tool_search"
            arguments = '{"query":"latest stock quote price","limit":2}'
        elif observation_kinds == ["tool.search"]:
            name = "tool_describe"
            arguments = '{"capability_ids":["market.get_quote"]}'
        elif observation_kinds == ["tool.search", "tool.describe"]:
            name = "market_get_quote"
            arguments = '{"symbol":"NVDA"}'
        else:
            observation_id = next(
                item["id"]
                for item in user_payload["observations"]
                if item["kind"] == "market.get_quote"
            )
            statement = "NVDA is quoted at 181.25."
            return httpx.Response(
                200,
                json={
                    "id": "gen-progressive-answer",
                    "model": "test/model",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "answer": statement,
                                        "source_claims": [
                                            {
                                                "statement": statement,
                                                "observation_ids": [observation_id],
                                            }
                                        ],
                                        "inferences": [],
                                    }
                                ),
                                "tool_calls": [],
                            }
                        }
                    ],
                },
            )
        return httpx.Response(
            200,
            json={
                "id": f"gen-progressive-{len(advertised_by_turn)}",
                "model": "test/model",
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(advertised_by_turn)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": arguments},
                                }
                            ],
                        }
                    }
                ],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=_settings(),
            client=client,
            objective="Investigate this with an appropriate available capability.",
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert "market_get_quote" not in advertised_by_turn[0]
    assert "market_get_quote" not in advertised_by_turn[1]
    assert "market_get_quote" in advertised_by_turn[2]
    assert result.run.status is RunStatus.COMPLETED
    assert result.run.usage.tool_calls == 3


@pytest.mark.asyncio
async def test_live_web_search_then_fetch_produces_grounded_source_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "leo.integrations.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    model_calls = 0
    advertised: set[str] = set()
    statement = "Leo is a constellation of the zodiac."

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls, advertised
        if request.url.host == "en.wikipedia.org" and request.url.path == "/w/api.php":
            return httpx.Response(
                200,
                json=[
                    "Leo constellation",
                    ["Leo (constellation)"],
                    [statement],
                    ["https://en.wikipedia.org/wiki/Leo_(constellation)"],
                ],
            )
        if request.url.host == "en.wikipedia.org":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=statement,
                extensions={"leo_peer_ip": "93.184.216.34"},
            )

        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        advertised = {tool["function"]["name"] for tool in payload["tools"]}
        observations = user_payload["observations"]
        kinds = [item["kind"] for item in observations]
        if not observations:
            content = None
            tool_calls: list[dict[str, object]] = [
                {
                    "id": "call-public-search",
                    "type": "function",
                    "function": {
                        "name": "web_search_public",
                        "arguments": '{"query":"Leo constellation","limit":1}',
                    },
                }
            ]
        elif kinds == ["web.search_public"]:
            result_url = observations[0]["data"]["results"][0]["url"]
            content = None
            tool_calls = [
                {
                    "id": "call-public-fetch",
                    "type": "function",
                    "function": {
                        "name": "web_fetch_public_text",
                        "arguments": json.dumps({"url": result_url}),
                    },
                }
            ]
        else:
            fetch_id = next(
                item["id"] for item in observations if item["kind"] == "web.fetch_public_text"
            )
            content = json.dumps(
                {
                    "answer": statement,
                    "source_claims": [{"statement": statement, "observation_ids": [fetch_id]}],
                    "inferences": [],
                }
            )
            tool_calls = []
        return httpx.Response(
            200,
            json={
                "id": f"gen-web-research-{model_calls}",
                "model": "test/model",
                "choices": [{"message": {"content": content, "tool_calls": tool_calls}}],
            },
        )

    settings = _settings().model_copy(update={"finnhub_api_key": None, "leo_max_model_turns": 4})
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="Search the web for Leo constellation, fetch a result, and cite the source.",
            trusted_scope=TrustedScope(
                namespace=ScopeKey(organization_id="org", strategy_id="domain"),
                actor_id="U1",
                roles=frozenset({"researcher"}),
            ),
        )

    assert {"web_search_public", "web_fetch_public_text"}.issubset(advertised)
    assert result.run.status is RunStatus.COMPLETED
    assert tuple(item.kind for item in result.observations) == (
        "web.search_public",
        "web.fetch_public_text",
    )
    assert result.run.final_output == statement
    assert result.run.usage.tool_calls == 2
    assert model_calls == 3


@pytest.mark.asyncio
async def test_live_conversation_rejects_fabricated_memory_authority_before_store_access() -> None:
    objective = "remember that this belongs to the admitted actor"
    authority = bind_memory_mutation_authority(
        scope=ScopeKey(organization_id="org", strategy_id="strategy"),
        team_id="T1",
        conversation_id="C1",
        conversation_kind=ConversationKind.CHANNEL,
        actor_id="U-ADMITTED",
        event_id="Ev1",
        task_id="task-1",
        run_id="run-1",
        message_reference="1710000000.001",
        objective=objective,
    )
    assert authority is not None
    sessions = cast(async_sessionmaker[AsyncSession], object())

    async with httpx.AsyncClient() as client:
        with pytest.raises(ValueError, match="actor does not match"):
            await run_live_conversation(
                settings=_settings().model_copy(update={"finnhub_api_key": None}),
                client=client,
                objective=objective,
                trusted_scope=TrustedScope(
                    namespace=authority.scope,
                    actor_id="U-ATTACKER",
                ),
                origin=OriginRef(
                    provider="slack",
                    external_thread_id="slack:T1:C1:1710000000.001",
                    external_event_id="Ev1",
                    external_channel_id="C1",
                ),
                sessions=sessions,
                launch_ids=("thread-1", "task-1", "run-1"),
                memory_authority=authority,
            )


@pytest.mark.parametrize(
    ("objective", "expected_tool"),
    [
        ("Alpha Vantage NVDA quote", "market.get_quote_alpha_vantage"),
        ("Use Massive for the NVDA quote", "market.get_quote_massive"),
        ("TickerLayer NVDA quote", "market.get_quote_ticker_layer"),
        ("Finnhub NVDA quote", "market.get_quote_finnhub"),
        (
            "Alpha Vantage PLTR company profile",
            "market.get_company_profile_alpha_vantage",
        ),
    ],
)
def test_named_equity_provider_precedes_provider_neutral_family(
    objective: str,
    expected_tool: str,
) -> None:
    available = frozenset(
        {
            "market.get_quote",
            "market.get_quote_alpha_vantage",
            "market.get_quote_finnhub",
            "market.get_quote_massive",
            "market.get_quote_ticker_layer",
            "market.get_equity_profile",
            "market.get_company_profile_alpha_vantage",
        }
    )

    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=available,
    )

    assert len(requirements) == 1
    assert requirements[0].tool_name == expected_tool


def test_named_equity_provider_combined_quote_and_profile_keeps_both_operations() -> None:
    requirements = _child_evidence_requirements(
        "Use Alpha Vantage for PLTR price and company profile",
        available_tool_names=frozenset(
            {
                "market.get_quote",
                "market.get_quote_alpha_vantage",
                "market.get_equity_profile",
                "market.get_company_profile_alpha_vantage",
            }
        ),
    )

    assert tuple(requirement.tool_name for requirement in requirements) == (
        "market.get_quote_alpha_vantage",
        "market.get_company_profile_alpha_vantage",
    )
    assert all(
        tuple((argument.name, argument.value) for argument in requirement.required_arguments)
        == (("symbol", "PLTR"),)
        for requirement in requirements
    )


@pytest.mark.parametrize(
    "objective",
    [
        "Massive NVDA price move today?",
        "What did we say earlier about CoinGecko's Bitcoin price?",
        "Compare CoinGecko and CoinMarketCap Bitcoin price",
        "Compare Alpha Vantage and Massive NVDA quote",
    ],
)
def test_vague_memory_and_multi_provider_mentions_do_not_force_one_direct_provider(
    objective: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset(
            {
                "market.get_quote",
                "market.get_quote_alpha_vantage",
                "market.get_quote_massive",
                "market.get_crypto_snapshot",
                "market.get_crypto_snapshot_coingecko",
                "market.get_crypto_snapshot_coinmarketcap",
            }
        ),
    )

    assert all(
        requirement.tool_name
        not in {
            "market.get_quote_alpha_vantage",
            "market.get_quote_massive",
            "market.get_crypto_snapshot_coingecko",
            "market.get_crypto_snapshot_coinmarketcap",
        }
        for requirement in requirements
    )


@pytest.mark.parametrize(
    ("objective", "expected_tool"),
    [
        ("CoinGecko BTC price", "market.get_crypto_snapshot_coingecko"),
        ("CoinMarketCap BTC price", "market.get_crypto_snapshot_coinmarketcap"),
    ],
)
def test_named_crypto_provider_precedes_snapshot_family(
    objective: str,
    expected_tool: str,
) -> None:
    requirements = _child_evidence_requirements(
        objective,
        available_tool_names=frozenset(
            {
                "market.get_crypto_snapshot",
                "market.get_crypto_snapshot_coingecko",
                "market.get_crypto_snapshot_coinmarketcap",
            }
        ),
    )

    assert len(requirements) == 1
    assert requirements[0].tool_name == expected_tool


@pytest.mark.parametrize(
    ("objective", "expected_tool"),
    [
        ("Search Exa for official release notes on Python 3.14", "web.search_exa"),
        ("Use Tavily to search the web for Python 3.14 release notes", "web.search_tavily"),
    ],
)
def test_named_web_provider_precedes_verified_provider_family(
    objective: str,
    expected_tool: str,
) -> None:
    route = _select_verified_web_provider(
        objective,
        frozenset(
            {
                "web.search_exa",
                "web.search_tavily",
                "web.fetch_public_text",
                "web.research_verified",
            }
        ),
    )

    assert route is not None
    assert route.search_tool == expected_tool


def test_multi_provider_web_comparison_uses_verified_family() -> None:
    route = _select_verified_web_provider(
        "Compare Exa and Tavily search results for Python 3.14",
        frozenset(
            {
                "web.search_exa",
                "web.search_tavily",
                "web.fetch_public_text",
                "web.research_verified",
            }
        ),
    )

    assert route is not None
    assert route.search_tool == "web.research_verified"


@pytest.mark.asyncio
async def test_blank_optional_provider_keys_register_nothing_and_never_make_provider_calls() -> (
    None
):
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.host)
        assert request.url.host == "openrouter.test"
        return httpx.Response(
            200,
            json={
                "id": "gen-blank-optional-keys",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "Hello.",
                                    "source_claims": [],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        tavily_api_key="   ",
        exa_api_key="",
        coingecko_api_key=" ",
        coin_market_cap_api_key="",
        alpha_vantage_api_key=" ",
        massive_api_key="",
        ticker_layer_api_key=" ",
        finnhub_api_key="",
    )
    registry = ProviderGateRegistry(FixedClock(datetime(2026, 8, 22, 12, 0, tzinfo=UTC)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="Say hello in one word.",
            provider_gates=registry,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert registry.registered_providers == ()
    assert requests == ["openrouter.test"]


@pytest.mark.asyncio
async def test_unavailable_named_provider_replies_without_substituting_healthy_peer() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unavailable named-provider turn made request to {request.url.host}")

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        coingecko_api_key=" ",
        coin_market_cap_api_key="cmc-key",
        tavily_api_key="",
        exa_api_key="",
        alpha_vantage_api_key="",
        massive_api_key="",
        ticker_layer_api_key="",
        finnhub_api_key="",
    )
    registry = ProviderGateRegistry(FixedClock(datetime(2026, 8, 22, 12, 0, tzinfo=UTC)))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="CoinGecko BTC price",
            provider_gates=registry,
        )

    assert result.run.status is RunStatus.COMPLETED
    assert "CoinGecko's direct route is unavailable" in (result.run.final_output or "")
    assert result.run.usage.tool_calls == 0
    assert result.observations == ()
    assert registry.registered_providers == ("coinmarketcap",)


@pytest.mark.asyncio
async def test_all_keys_named_alpha_vantage_quote_uses_model_completion() -> None:
    model_calls = 0
    provider_hosts: list[str] = []
    tool_choices: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "www.alphavantage.co":
            provider_hosts.append(request.url.host)
            assert request.url.params["function"] == "GLOBAL_QUOTE"
            assert request.url.params["symbol"] == "PLTR"
            return httpx.Response(
                200,
                json={
                    "Global Quote": {
                        "01. symbol": "PLTR",
                        "05. price": "180.00",
                        "07. latest trading day": "2026-08-21",
                    }
                },
            )
        assert request.url.host == "openrouter.test"
        model_calls += 1
        payload = json.loads(request.content)
        user_payload = json.loads(payload["messages"][1]["content"])
        tool_choices.append(payload["tool_choice"])
        observations = user_payload["observations"]
        if not observations:
            return httpx.Response(
                200,
                json={
                    "id": "gen-alpha-direct-call",
                    "model": "fixture/model",
                    "choices": [
                        {
                            "message": {
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call-alpha-direct",
                                        "type": "function",
                                        "function": {
                                            "name": "market_get_quote_alpha_vantage",
                                            "arguments": json.dumps({"symbol": "PLTR"}),
                                        },
                                    }
                                ],
                            }
                        }
                    ],
                },
            )
        observation = observations[0]
        statement = observation["data"]["statements"][0]
        return httpx.Response(
            200,
            json={
                "id": "gen-alpha-direct-completion",
                "model": "fixture/model",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": statement,
                                    "source_claims": [
                                        {
                                            "statement": statement,
                                            "observation_ids": [observation["id"]],
                                        }
                                    ],
                                    "inferences": [],
                                }
                            ),
                            "tool_calls": [],
                        }
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        alpha_vantage_api_key="alpha-key",
        massive_api_key="massive-key",
        ticker_layer_api_key="ticker-key",
        finnhub_api_key="finnhub-key",
        coingecko_api_key="coingecko-key",
        coin_market_cap_api_key="cmc-key",
        tavily_api_key="tavily-key",
        exa_api_key="exa-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective="Use Alpha Vantage for PLTR price",
        )

    assert result.run.status is RunStatus.COMPLETED
    assert model_calls == 2
    assert tool_choices[0] == {
        "type": "function",
        "function": {"name": "market_get_quote_alpha_vantage"},
    }
    assert provider_hosts == ["www.alphavantage.co"]
    assert tuple(observation.kind for observation in result.observations) == (
        "market.get_quote_alpha_vantage",
    )
