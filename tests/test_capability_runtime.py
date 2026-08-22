from __future__ import annotations

from pathlib import Path

import pytest

from leo.capabilities.catalog import CapabilityHealth, CatalogTool, InMemoryToolCatalog
from leo.capabilities.runtime import CapabilityRuntime
from leo.capabilities.skills import SkillCatalog
from leo.capabilities.tools import ToolDescribeTool, ToolSearchTool
from leo.harness.models import (
    OriginRef,
    Run,
    RunBundle,
    RunPhase,
    ScopeKey,
    Task,
    Thread,
    ToolEffect,
    ToolExecutionContext,
    ToolFailure,
    ToolSpec,
    ToolSuccess,
    TrustedScope,
)
from leo.integrations.fake import FixedClock

SCOPE = ScopeKey(organization_id="org", strategy_id="domain")
TRUSTED = TrustedScope(
    namespace=SCOPE,
    actor_id="actor",
    roles=frozenset({"researcher"}),
)


def _spec(
    name: str,
    description: str,
    *,
    effect: ToolEffect = ToolEffect.READ,
    required_roles: frozenset[str] = frozenset(),
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description,
        domain=name.split(".", 1)[0],
        input_schema={
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "additionalProperties": False,
        },
        effect=effect,
        allowed_phases=frozenset(
            {RunPhase.RESEARCH if effect is not ToolEffect.WRITE else RunPhase.EXECUTION}
        ),
        required_roles=required_roles,
    )


def _record(
    spec: ToolSpec,
    *,
    tags: frozenset[str],
    roles: frozenset[str] = frozenset(),
    health: CapabilityHealth = CapabilityHealth.HEALTHY,
) -> CatalogTool:
    return CatalogTool(
        id=spec.name,
        semantic_version="1.0.0",
        provider="fixture",
        spec=spec,
        short_description=spec.description,
        tags=tags,
        authorized_roles=roles,
        health=health,
    )


def _bundle(objective: str, *, run_id: str = "run-1") -> RunBundle:
    thread = Thread(
        id="thread-1",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation-1"),
    )
    task = Task(id="task-1", thread_id=thread.id, scope=SCOPE, objective=objective)
    run = Run(id=run_id, task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run)


def _catalog() -> tuple[InMemoryToolCatalog, dict[str, ToolSpec]]:
    quote = _spec("market.get_quote", "Return a current market quote and stock price.")
    weather = _spec("weather.forecast", "Return a local weather forecast.")
    forbidden = _spec(
        "finance.private_ledger",
        "Read a private ledger.",
        required_roles=frozenset({"operator"}),
    )
    unhealthy = _spec("market.unhealthy_quote", "Return a current stock price.")
    write = _spec("market.place_order", "Place a market order.", effect=ToolEffect.WRITE)
    catalog = InMemoryToolCatalog(version="fixture-v3")
    catalog.register(_record(quote, tags=frozenset({"market", "quote", "price", "stock"})))
    catalog.register(_record(weather, tags=frozenset({"weather", "forecast"})))
    catalog.register(
        _record(
            forbidden,
            tags=frozenset({"ledger", "private"}),
            roles=frozenset({"operator"}),
        )
    )
    catalog.register(
        _record(
            unhealthy,
            tags=frozenset({"market", "price"}),
            health=CapabilityHealth.UNHEALTHY,
        )
    )
    catalog.register(_record(write, tags=frozenset({"market", "order"})))
    return catalog, {item.name: item for item in (quote, weather, forbidden, unhealthy, write)}


def test_policy_first_runtime_recalls_paraphrase_without_forbidden_or_distractors() -> None:
    catalog, specs = _catalog()
    search = _spec("tool.search", "Search eligible tools.")
    describe = _spec("tool.describe", "Describe searched tools.")
    runtime = CapabilityRuntime(
        catalog,
        always_available_tool_names=frozenset({search.name, describe.name}),
    )

    selection = runtime.select(
        bundle=_bundle("Could you look up NVDA's latest stock price?"),
        trusted_scope=TRUSTED,
        available_tools=(*specs.values(), search, describe),
    )

    assert selection.candidate_ids == ("market.get_quote",)
    assert selection.selected_ids == ("market.get_quote", "tool.search", "tool.describe")
    assert "weather.forecast" not in selection.selected_ids
    assert "finance.private_ledger" not in selection.candidate_ids
    assert "market.unhealthy_quote" not in selection.candidate_ids
    assert "market.place_order" not in selection.candidate_ids
    assert selection.eligible_count == 2
    assert len(selection.catalog_fingerprint) == 64
    assert len(selection.selection_fingerprint) == 64


def test_empty_recall_keeps_direct_answer_and_required_sealed_tool_available() -> None:
    catalog, specs = _catalog()
    memory = _spec(
        "memory.remember",
        "Commit confirmed memory.",
        effect=ToolEffect.STATE_MUTATION,
        required_roles=frozenset({"researcher"}),
    )
    runtime = CapabilityRuntime(
        catalog,
        required_tool_names=frozenset({memory.name}),
    )

    selection = runtime.select(
        bundle=_bundle("What did we call the demo?"),
        trusted_scope=TRUSTED,
        available_tools=(*specs.values(), memory),
    )

    assert selection.mode == "direct"
    assert selection.candidate_ids == ()
    assert selection.selected_ids == ("memory.remember",)


def test_required_catalog_tool_cannot_bypass_rate_limited_health_policy() -> None:
    direct = _spec("market.get_quote_direct", "Return one provider's current stock price.")
    catalog = InMemoryToolCatalog(version="health-required-v1")
    catalog.register(
        _record(
            direct,
            tags=frozenset({"market", "quote", "price"}),
            health=CapabilityHealth.RATE_LIMITED,
        )
    )
    runtime = CapabilityRuntime(
        catalog,
        required_tool_names=frozenset({direct.name}),
    )

    selection = runtime.select(
        bundle=_bundle("Use the direct quote provider."),
        trusted_scope=TRUSTED,
        available_tools=(direct,),
    )

    assert selection.eligible_count == 0
    assert selection.selected_ids == ()


@pytest.mark.asyncio
async def test_search_describe_is_run_bound_bounded_and_schema_exact() -> None:
    catalog, specs = _catalog()
    clock = FixedClock()
    runtime = CapabilityRuntime(catalog, shortlist_limit=2, max_search_calls=1)
    bundle = _bundle("Tell me something conversationally")
    runtime.select(
        bundle=bundle,
        trusted_scope=TRUSTED,
        available_tools=tuple(specs.values()),
    )
    context = ToolExecutionContext(
        trusted_scope=TRUSTED,
        run_id=bundle.run.id,
        tool_call_id="call-1",
    )
    search = ToolSearchTool(runtime, clock)
    describe = ToolDescribeTool(runtime, clock)

    guessed = await describe.execute({"capability_ids": ["market.get_quote"]}, context)
    assert isinstance(guessed, ToolFailure)
    assert guessed.code == "CAPABILITY_NOT_DISCOVERED"

    found = await search.execute({"query": "latest share price", "limit": 2}, context)
    assert isinstance(found, ToolSuccess)
    assert [item["id"] for item in found.data["capabilities"]] == ["market.get_quote"]
    described = await describe.execute({"capability_ids": ["market.get_quote"]}, context)
    assert isinstance(described, ToolSuccess)
    payload = described.data["capabilities"][0]
    assert payload["spec"] == specs["market.get_quote"].model_dump(mode="json")
    assert payload["schema_fingerprint"] == catalog.get("market.get_quote").schema_fingerprint

    repeated = await search.execute({"query": "different query", "limit": 1}, context)
    assert isinstance(repeated, ToolFailure)
    assert repeated.code == "DISCOVERY_SEARCH_BUDGET_EXHAUSTED"


def test_selected_skill_procedure_is_hash_verified_and_unselected_skill_is_absent() -> None:
    catalog, specs = _catalog()
    runtime = CapabilityRuntime(
        catalog,
        skill_catalog=SkillCatalog(Path("resources/leo-skills")),
    )
    objective = "Please retrieve the latest stock price for NVDA."

    selection = runtime.select(
        bundle=_bundle(objective),
        trusted_scope=TRUSTED,
        available_tools=tuple(specs.values()),
    )
    items = runtime.skill_context_items(
        objective,
        scope=SCOPE,
        conversation_id="conversation-1",
        phase=RunPhase.RESEARCH,
        roles=TRUSTED.roles,
    )

    assert selection.selected_skill_ids[0].startswith("leo-skill-v1:narrow_quote@1.0.0:")
    assert len(items) == 1
    assert items[0].kind.value == "skill_procedure"
    assert "narrow_quote" in items[0].content
    assert "thesis_challenge" not in items[0].content


def test_general_conversation_skill_can_project_without_tool_recall() -> None:
    catalog, specs = _catalog()
    runtime = CapabilityRuntime(
        catalog,
        skill_catalog=SkillCatalog(Path("resources/leo-skills")),
    )
    objective = "Answer this conversational question from our context."

    selection = runtime.select(
        bundle=_bundle(objective),
        trusted_scope=TRUSTED,
        available_tools=tuple(specs.values()),
    )
    items = runtime.skill_context_items(
        objective,
        scope=SCOPE,
        conversation_id="conversation-1",
        phase=RunPhase.RESEARCH,
        roles=TRUSTED.roles,
    )

    assert selection.mode == "direct"
    assert selection.selected_skill_ids[0].startswith("leo-skill-v1:general_conversation@1.0.0:")
    assert len(items) == 1
    assert '"child_compatible":false' in items[0].content
    assert "delegated_research" not in items[0].content


def test_policy_first_recall_with_one_thousand_distractors_has_zero_forbidden_exposure() -> None:
    quote = _spec("market.get_quote", "Return the latest equity quote and current stock price.")
    forbidden = _spec(
        "market.private_quote",
        "Return the latest equity quote and current stock price.",
        required_roles=frozenset({"operator"}),
    )
    unhealthy = _spec(
        "market.unhealthy_quote", "Return the latest equity quote and current stock price."
    )
    write = _spec(
        "market.place_order",
        "Return the latest equity quote then place an order.",
        effect=ToolEffect.WRITE,
    )
    catalog = InMemoryToolCatalog(version="benchmark-1000-v1")
    catalog.register(
        _record(quote, tags=frozenset({"market", "quote", "price", "equity", "latest"}))
    )
    catalog.register(
        _record(
            forbidden,
            tags=frozenset({"market", "quote", "price"}),
            roles=frozenset({"operator"}),
        )
    )
    catalog.register(
        _record(
            unhealthy,
            tags=frozenset({"market", "quote", "price"}),
            health=CapabilityHealth.UNHEALTHY,
        )
    )
    catalog.register(_record(write, tags=frozenset({"market", "quote", "price"})))
    distractors: list[ToolSpec] = []
    for index in range(1_000):
        spec = _spec(
            f"utility.fixture_{index:04d}",
            f"Return unrelated deterministic fixture number {index}.",
        )
        distractors.append(spec)
        catalog.register(_record(spec, tags=frozenset({"utility", f"fixture-{index:04d}"})))
    runtime = CapabilityRuntime(catalog, shortlist_limit=3)

    selection = runtime.select(
        bundle=_bundle("Could you retrieve NVDA's latest equity quote and stock price?"),
        trusted_scope=TRUSTED,
        available_tools=(quote, forbidden, unhealthy, write, *distractors),
    )

    assert selection.eligible_count == 1_001
    assert selection.candidate_ids == ("market.get_quote",)
    assert selection.selected_ids == ("market.get_quote",)
    assert not {
        "market.private_quote",
        "market.unhealthy_quote",
        "market.place_order",
    }.intersection(selection.candidate_ids)
