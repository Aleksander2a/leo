"""Local composition root for real OpenRouter + Finnhub smoke runs."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import httpx
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import (
    CapabilityHealth,
    CapabilityLatency,
    CapabilitySensitivity,
    CatalogTool,
    InMemoryToolCatalog,
)
from leo.capabilities.crypto_descriptors import CRYPTO_CAPABILITY_DESCRIPTORS
from leo.capabilities.discovery import search_tokens
from leo.capabilities.embeddings import OpenRouterEmbeddingGateway, ensure_tool_embeddings
from leo.capabilities.equity_descriptors import EQUITY_CAPABILITY_DESCRIPTORS
from leo.capabilities.runtime import CapabilityRuntime
from leo.capabilities.skills import SkillCatalog
from leo.capabilities.tools import build_capability_discovery_tools
from leo.config import Settings, is_configured_secret
from leo.harness.context import DefaultContextAssembler
from leo.harness.coordinator import RunCoordinator
from leo.harness.deliberation import (
    ElasticDeliberationGateway,
    ElasticDeliberationPolicy,
    apply_deliberation_guidance,
)
from leo.harness.models import (
    BudgetLimits,
    CardinalityBounds,
    CompletionContract,
    ContextItem,
    ContextItemKind,
    ContextItemRetention,
    CoordinatorResult,
    EvidenceToolRequirement,
    Observation,
    OriginRef,
    Run,
    ScopeKey,
    Task,
    Thread,
    ToolArgumentConstraint,
    TrustedScope,
)
from leo.harness.ports import ModelGateway, RunStore, Tool
from leo.harness.provider_health import ProviderHealthProjection
from leo.harness.research import ResearchRequirement
from leo.harness.storage import InMemoryRunStore
from leo.harness.subagents import (
    SubagentPlanTool,
    SubagentResearchTool,
)
from leo.harness.tools import ToolRegistry
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.crypto_composition import build_crypto_market_tools
from leo.integrations.equity_composition import build_equity_market_tools
from leo.integrations.exa import EXA_CAPABILITY_DESCRIPTOR, ExaSearchTool
from leo.integrations.finnhub import (
    FinnhubBasicFinancialsTool,
    FinnhubCompanyNewsTool,
    FinnhubEarningsSurprisesTool,
    FinnhubQuoteTool,
    normalize_quote_symbol,
)
from leo.integrations.mcp_tools import (
    build_alpha_vantage_mcp_tools,
    build_coingecko_mcp_tools,
    build_tavily_mcp_tools,
)
from leo.integrations.openrouter import OpenRouterGateway
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.sec_edgar import SecEdgarRecentFilingsTool
from leo.integrations.system import SystemClock, UuidIdGenerator
from leo.integrations.tavily import TAVILY_FREE_TIER_MONTHLY_CREDITS, TavilySearchTool
from leo.integrations.verified_web import VerifiedWebResearchTool
from leo.integrations.web_fetch import PublicTextFetchTool
from leo.integrations.web_search import PublicWebSearchTool
from leo.memory.navigation import MemoryNavigationAuthority
from leo.memory.navigation_tools import build_memory_navigation_tools
from leo.memory.service import ExplicitMemoryService
from leo.memory.tools import (
    MemoryMutationAuthority,
    build_autonomous_memory_tools,
    build_explicit_memory_tools,
    parse_explicit_memory_intent,
)
from leo.packaging import require_data_directory
from leo.persistence.capability_embeddings import PostgresCapabilityEmbeddingStore
from leo.persistence.memory_embeddings import PostgresMemoryEmbeddingIndexer
from leo.persistence.memory_navigation import PostgresProgressiveMemoryService
from leo.persistence.memory_store import PostgresMemoryStore
from leo.persistence.model_call_transcripts import PostgresModelCallTranscriptSink
from leo.persistence.plan_store import PostgresPlanStore
from leo.persistence.run_store import LeaseBoundRunStore, PostgresRunStore
from leo.persistence.task_leases import TaskLease

_DEMO_SEC_TICKER_MAP = {
    "AAPL": "320193",
    "AMD": "2488",
    "AMZN": "1018724",
    "GOOGL": "1652044",
    "META": "1326801",
    "MSFT": "789019",
    "NVDA": "1045810",
    "TSLA": "1318605",
}
_DEMO_ENTITY_ALIASES = {
    "apple": "AAPL",
    "amazon": "AMZN",
    "google": "GOOGL",
    "meta": "META",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "tesla": "TSLA",
}
_CRYPTO_ASSET_ALIASES = {
    "ada": "cardano",
    "bitcoin": "bitcoin",
    "btc": "bitcoin",
    "cardano": "cardano",
    "doge": "dogecoin",
    "dogecoin": "dogecoin",
    "eth": "ethereum",
    "ether": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "tether": "tether",
    "usd-coin": "usd-coin",
    "usdc": "usd-coin",
    "usdt": "tether",
    "xrp": "ripple",
}
_NON_EQUITY_SYMBOL_TOKENS = frozenset(
    {
        "AI",
        "AN",
        "API",
        "CEO",
        "CIK",
        "COMPANY",
        "CURRENT",
        "ETF",
        "EPS",
        "EUR",
        "GBP",
        "IPO",
        "JPY",
        "LATEST",
        "LLM",
        "NASDAQ",
        "NOW",
        "NYSE",
        "PE",
        "PRICE",
        "PROFILE",
        "QUOTE",
        "SEC",
        "STOCK",
        "THE",
        "TODAY",
        "USD",
        "US",
        "WHAT",
    }
    | {alias.upper() for alias in _CRYPTO_ASSET_ALIASES}
)
# Resolved by layout, not by this module's position: installed into
# site-packages the old expression pointed at the interpreter's lib
# directory, and a glob over a missing directory returns nothing, so the
# deployed agent ran with an empty skill catalogue and never said so.
_SKILL_ROOT = require_data_directory("resources/leo-skills", anchor=Path(__file__))
_EMPTY_MEMORY_SCOPE_INFERENCE = (
    "No matching authorized memory was found in this conversation scope."
)
# Kept in sync with `deliberation._SEARCH_LADDER`, which walks the same routes as
# a deterministic failover order when the model itself stops making progress.
_WEB_SEARCH_LADDER = frozenset(
    {
        "web.research_verified",
        "web.search_exa",
        "web.search_tavily",
        "web.search_public",
        # Discovery without retrieval is a dead end. Search tools return URLs and
        # snippets that are explicitly *not* citable evidence, so a model that
        # can search but cannot open a result has no way to finish the job: it
        # either cites discovery metadata (which the verifier rejects) or gives
        # up. The fetch tool used to be reachable only because a deterministic
        # repair called it directly, bypassing advertisement entirely -- so when
        # that repair was removed, the chain broke. It belongs on the ladder.
        "web.fetch_public_text",
    }
)
_PARENT_ORCHESTRATION_TOOL_NAMES = frozenset(
    {
        "agent.delegate_research",
        "agent.execute_research_plan",
        "tool.search",
        "tool.describe",
    }
)


@dataclass(frozen=True)
class _VerifiedWebProviderRoute:
    provider: str
    search_tool: str
    required_tools: frozenset[str]
    direct_highlight_evidence: bool


@dataclass(frozen=True)
class _ExplicitProviderIntent:
    provider: str
    display_name: str
    primary_tool: str
    required_tools: frozenset[str]


_EXA_VERIFIED_WEB_ROUTE = _VerifiedWebProviderRoute(
    provider="exa",
    search_tool="web.search_exa",
    required_tools=frozenset({"web.search_exa"}),
    direct_highlight_evidence=True,
)
_PROVIDER_FAMILY_VERIFIED_WEB_ROUTE = _VerifiedWebProviderRoute(
    provider="provider-family",
    search_tool="web.research_verified",
    required_tools=frozenset({"web.research_verified"}),
    direct_highlight_evidence=True,
)
_TAVILY_VERIFIED_WEB_ROUTE = _VerifiedWebProviderRoute(
    provider="tavily",
    search_tool="web.search_tavily",
    required_tools=frozenset({"web.search_tavily", "web.fetch_public_text"}),
    direct_highlight_evidence=False,
)

_EXPLICIT_PROVIDER_INTENTS = {
    "alpha_vantage": _ExplicitProviderIntent(
        provider="alpha_vantage",
        display_name="Alpha Vantage",
        primary_tool="market.get_quote_alpha_vantage",
        required_tools=frozenset({"market.get_quote_alpha_vantage"}),
    ),
    "coingecko": _ExplicitProviderIntent(
        provider="coingecko",
        display_name="CoinGecko",
        primary_tool="market.get_crypto_snapshot_coingecko",
        required_tools=frozenset({"market.get_crypto_snapshot_coingecko"}),
    ),
    "coinmarketcap": _ExplicitProviderIntent(
        provider="coinmarketcap",
        display_name="CoinMarketCap",
        primary_tool="market.get_crypto_snapshot_coinmarketcap",
        required_tools=frozenset({"market.get_crypto_snapshot_coinmarketcap"}),
    ),
    "exa": _ExplicitProviderIntent(
        provider="exa",
        display_name="Exa",
        primary_tool="web.search_exa",
        required_tools=frozenset({"web.search_exa"}),
    ),
    "finnhub": _ExplicitProviderIntent(
        provider="finnhub",
        display_name="Finnhub",
        primary_tool="market.get_quote_finnhub",
        required_tools=frozenset({"market.get_quote_finnhub"}),
    ),
    "massive": _ExplicitProviderIntent(
        provider="massive",
        display_name="Massive",
        primary_tool="market.get_quote_massive",
        required_tools=frozenset({"market.get_quote_massive"}),
    ),
    "tavily": _ExplicitProviderIntent(
        provider="tavily",
        display_name="Tavily",
        primary_tool="web.search_tavily",
        required_tools=frozenset({"web.search_tavily", "web.fetch_public_text"}),
    ),
    "ticker_layer": _ExplicitProviderIntent(
        provider="ticker_layer",
        display_name="TickerLayer",
        primary_tool="market.get_quote_ticker_layer",
        required_tools=frozenset({"market.get_quote_ticker_layer"}),
    ),
}


_DIRECT_CANONICAL_EVIDENCE_KINDS = frozenset(
    {
        "market.get_crypto_snapshot",
        "market.get_quote",
        "market.get_earnings_surprises",
    }
)


async def run_live_quote(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    symbol: str,
    objective: str,
    actor_id: str | None = None,
    trusted_scope: TrustedScope | None = None,
    origin: OriginRef | None = None,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    launch_ids: tuple[str, str, str] | None = None,
    lease: TaskLease | None = None,
) -> CoordinatorResult:
    normalized_symbol = normalize_quote_symbol(symbol)
    missing = settings.missing_for_live_providers()
    if missing:
        raise RuntimeError(f"missing provider configuration names: {', '.join(missing)}")
    assert settings.openrouter_api_key is not None
    assert settings.finnhub_api_key is not None
    assert settings.leo_model is not None

    clock = SystemClock()
    ids = UuidIdGenerator()
    if trusted_scope is not None:
        if actor_id is not None:
            raise ValueError("actor_id cannot be supplied with trusted_scope")
        execution_scope = trusted_scope
    else:
        execution_scope = TrustedScope(
            namespace=ScopeKey(
                organization_id=settings.leo_organization_id,
                strategy_id=settings.leo_strategy_id,
            ),
            actor_id=actor_id or "local-user",
            roles=frozenset({"researcher"}),
        )
    scope = execution_scope.namespace
    if launch_ids is not None and sessions is None:
        raise ValueError("launch_ids require a durable session store")
    if lease is not None and sessions is None:
        raise ValueError("lease requires a durable session store")
    thread_id, task_id, run_id = launch_ids or (
        ids.new("thread"),
        ids.new("task"),
        ids.new("run"),
    )
    thread = Thread(
        id=thread_id,
        scope=scope,
        origin=origin or OriginRef(provider="cli", external_thread_id=ids.new("conversation")),
    )
    task = Task(
        id=task_id,
        thread_id=thread.id,
        scope=scope,
        objective=objective,
    )
    run = Run(
        id=run_id,
        task_id=task.id,
        scope=scope,
        limits=BudgetLimits(
            max_iterations=settings.leo_max_model_turns,
            max_model_calls=settings.leo_max_model_turns,
            max_tool_calls=settings.leo_max_tool_calls,
            max_elapsed_seconds=settings.leo_max_run_seconds,
        ),
    )
    store: RunStore
    if sessions is None:
        store = InMemoryRunStore(clock, ids)
    else:
        durable_store = PostgresRunStore(sessions, clock, ids)
        store = durable_store if lease is None else LeaseBoundRunStore(durable_store, lease)
    if launch_ids is None:
        await store.seed(thread, task, run)
    model = OpenRouterGateway(
        client=client,
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.leo_model,
        base_url=settings.openrouter_base_url,
        max_output_tokens=settings.leo_max_output_tokens,
        parallel_tool_calls=False,
    )
    tools = ToolRegistry(
        (
            FinnhubQuoteTool(
                client=client,
                api_key=settings.finnhub_api_key.get_secret_value(),
                clock=clock,
                base_url=settings.finnhub_base_url,
            ),
        )
    )
    quote_requirement = EvidenceToolRequirement(
        observation_kind="market.get_quote",
        tool_name="market.get_quote",
        required_arguments=(ToolArgumentConstraint(name="symbol", value=normalized_symbol),),
    )
    quote_completion_contract = CompletionContract(
        source_claim_count=CardinalityBounds(minimum=1, maximum=1),
        source_observation_id_count=CardinalityBounds(minimum=1, maximum=1),
        inference_count=CardinalityBounds(minimum=0, maximum=0),
        guidance=(
            f"In both the answer and source claim, copy {normalized_symbol} and the exact numeric "
            "observation.data.price without rounding. Do not make separate change, high, low, "
            "open, or previous-close claims."
        ),
    )
    coordinator = RunCoordinator(
        store=store,
        model=model,
        tools=tools,
        context=DefaultContextAssembler(
            evidence_requirements=(quote_requirement,),
            clock=clock,
            completion_contract=quote_completion_contract,
        ),
        transcript_sink=(
            PostgresModelCallTranscriptSink(sessions, ids=ids) if sessions is not None else None
        ),
        verifier=DeterministicCompletionVerifier(
            ids,
            clock,
            evidence_requirements=(quote_requirement,),
            relax_integration_grounding=True,
        ),
        clock=clock,
        ids=ids,
    )
    return await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=execution_scope,
    )


async def run_live_conversation(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    objective: str,
    context_items: tuple[ContextItem, ...] = (),
    context_authority_ids: tuple[str, ...] = (),
    actor_id: str | None = None,
    trusted_scope: TrustedScope | None = None,
    origin: OriginRef | None = None,
    sessions: async_sessionmaker[AsyncSession] | None = None,
    launch_ids: tuple[str, str, str] | None = None,
    lease: TaskLease | None = None,
    memory_authority: MemoryMutationAuthority | None = None,
    memory_navigation_authority: MemoryNavigationAuthority | None = None,
    autonomous_memory_authority: MemoryMutationAuthority | None = None,
    thread_context_tools: tuple[Tool, ...] = (),
    provider_gates: ProviderGateRegistry | None = None,
    embedding_gateway: OpenRouterEmbeddingGateway | None = None,
) -> CoordinatorResult:
    """Run an open-ended conversational turn through Leo's bounded harness.

    Unlike ``run_live_quote``, this route does not impose a phrase grammar or mandatory
    evidence tool. The model may answer from the selected scoped context, ask for clarity,
    or use any advertised read tool. Runtime scope, budgets, tool execution, and terminal
    truth remain harness-owned.
    """

    missing = settings.missing_for_conversation_providers()
    if missing:
        raise RuntimeError(f"missing provider configuration names: {', '.join(missing)}")
    assert settings.openrouter_api_key is not None
    assert settings.leo_model is not None

    clock = SystemClock()
    provider_gate_registry = provider_gates or ProviderGateRegistry(clock)
    ids = UuidIdGenerator()
    if trusted_scope is not None:
        if actor_id is not None:
            raise ValueError("actor_id cannot be supplied with trusted_scope")
        execution_scope = trusted_scope
    else:
        execution_scope = TrustedScope(
            namespace=ScopeKey(
                organization_id=settings.leo_organization_id,
                strategy_id=settings.leo_strategy_id,
            ),
            actor_id=actor_id or "local-user",
            roles=frozenset({"researcher"}),
        )
    scope = execution_scope.namespace
    if launch_ids is not None and sessions is None:
        raise ValueError("launch_ids require a durable session store")
    if lease is not None and sessions is None:
        raise ValueError("lease requires a durable session store")
    thread_id, task_id, run_id = launch_ids or (
        ids.new("thread"),
        ids.new("task"),
        ids.new("run"),
    )
    if memory_authority is not None:
        _validate_memory_authority(
            authority=memory_authority,
            objective=objective,
            trusted_scope=execution_scope,
            origin=origin,
            sessions=sessions,
            launch_ids=launch_ids,
        )
    if memory_navigation_authority is not None:
        _validate_memory_navigation_authority(
            authority=memory_navigation_authority,
            trusted_scope=execution_scope,
            origin=origin,
            sessions=sessions,
            launch_ids=launch_ids,
        )
    if autonomous_memory_authority is not None:
        _validate_autonomous_memory_authority(
            authority=autonomous_memory_authority,
            trusted_scope=execution_scope,
            origin=origin,
            sessions=sessions,
            launch_ids=launch_ids,
        )
    thread_root_objective = _trusted_thread_root_objective(context_items)
    routing_objective = _thread_intent_routing_objective(objective, context_items)
    thread = Thread(
        id=thread_id,
        scope=scope,
        origin=origin or OriginRef(provider="cli", external_thread_id=ids.new("conversation")),
    )
    task = Task(id=task_id, thread_id=thread.id, scope=scope, objective=objective)
    run = Run(
        id=run_id,
        task_id=task.id,
        scope=scope,
        limits=BudgetLimits(
            max_iterations=settings.leo_max_model_turns,
            max_model_calls=settings.leo_max_model_turns,
            max_tool_calls=settings.leo_max_tool_calls,
            max_elapsed_seconds=settings.leo_max_run_seconds,
        ),
    )
    store: RunStore
    child_run_store: RunStore | None = None
    durable_plan_store: PostgresPlanStore | None = None
    if sessions is None:
        store = InMemoryRunStore(clock, ids)
    else:
        durable_store = PostgresRunStore(sessions, clock, ids)
        store = durable_store if lease is None else LeaseBoundRunStore(durable_store, lease)
        child_run_store = durable_store
        durable_plan_store = PostgresPlanStore(sessions, clock, ids)
    if launch_ids is None:
        await store.seed(thread, task, run)

    model = OpenRouterGateway(
        client=client,
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.leo_model,
        base_url=settings.openrouter_base_url,
        max_output_tokens=settings.leo_max_output_tokens,
        parallel_tool_calls=True,
    )
    public_fetch_tool = PublicTextFetchTool(client=client, clock=clock)
    research_tools: list[Tool] = [
        PublicWebSearchTool(
            client=client,
            clock=clock,
            # SEC_USER_AGENT is already a required "contact" identity for public
            # data sources; reuse it so Wikimedia sees a real contact too.
            user_agent=settings.sec_user_agent,
        ),
        public_fetch_tool,
    ]
    tavily_tool: TavilySearchTool | None = None
    if is_configured_secret(settings.tavily_api_key):
        assert settings.tavily_api_key is not None
        tavily_tool = TavilySearchTool(
            client=client,
            api_key=settings.tavily_api_key.get_secret_value(),
            clock=clock,
            max_calls_per_minute=settings.tavily_max_calls_per_minute,
            max_calls_per_month=settings.tavily_max_calls_per_month,
            gate=provider_gate_registry.get(
                provider="tavily",
                max_concurrency=4,
                max_calls_per_minute=settings.tavily_max_calls_per_minute,
                max_calls_per_month=settings.tavily_max_calls_per_month,
                max_provider_credits_per_month=TAVILY_FREE_TIER_MONTHLY_CREDITS,
            ),
        )
        research_tools.append(tavily_tool)
    exa_tool: ExaSearchTool | None = None
    if is_configured_secret(settings.exa_api_key):
        assert settings.exa_api_key is not None
        exa_tool = ExaSearchTool(
            client=client,
            api_key=settings.exa_api_key.get_secret_value(),
            clock=clock,
            max_calls_per_minute=EXA_CAPABILITY_DESCRIPTOR.max_calls_per_minute,
            gate=provider_gate_registry.get(
                provider="exa",
                max_concurrency=4,
                max_calls_per_minute=EXA_CAPABILITY_DESCRIPTOR.max_calls_per_minute,
            ),
        )
        research_tools.append(exa_tool)
        if tavily_tool is not None:
            # The family tool exists to fail over *between* providers, so it is
            # only meaningful when both routes are configured. A single-provider
            # deployment is still guaranteed a web fallback: every registered
            # search route is advertised via `_WEB_SEARCH_LADDER`, and
            # `deliberation._next_untried_search` walks the same ladder. The
            # instances share their credential-level gates with the direct tools.
            research_tools.append(
                VerifiedWebResearchTool(
                    exa=exa_tool,
                    tavily=tavily_tool,
                    fetch=public_fetch_tool,
                )
            )
    research_tools.extend(
        build_crypto_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=provider_gate_registry,
        )
    )
    research_tools.extend(
        build_equity_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=provider_gate_registry,
        )
    )
    # MCP-sourced tools are additive redundancy alongside the REST adapters above,
    # not a replacement: the model may call both for the same fact and reconcile
    # them itself. See leo.integrations.mcp_tools for why each endpoint is (or
    # isn't) wired up this way.
    if is_configured_secret(settings.tavily_endpoint):
        assert settings.tavily_endpoint is not None
        research_tools.extend(
            build_tavily_mcp_tools(
                endpoint=settings.tavily_endpoint.get_secret_value(),
                clock=clock,
                gate=provider_gate_registry.get(
                    provider="tavily_mcp",
                    max_concurrency=2,
                    max_calls_per_minute=20,
                ),
            )
        )
    if is_configured_secret(settings.alpha_vantage_endpoint_legacy):
        assert settings.alpha_vantage_endpoint_legacy is not None
        research_tools.extend(
            build_alpha_vantage_mcp_tools(
                endpoint=settings.alpha_vantage_endpoint_legacy.get_secret_value(),
                clock=clock,
                gate=provider_gate_registry.get(
                    provider="alpha_vantage_mcp",
                    max_concurrency=2,
                    max_calls_per_minute=settings.alpha_vantage_max_calls_per_minute,
                ),
            )
        )
    if is_configured_secret(settings.coingecko_endpoint):
        research_tools.extend(
            build_coingecko_mcp_tools(
                clock=clock,
                gate=provider_gate_registry.get(
                    provider="coingecko_mcp",
                    max_concurrency=2,
                    max_calls_per_minute=30,
                ),
            )
        )
    if is_configured_secret(settings.finnhub_api_key):
        assert settings.finnhub_api_key is not None
        finnhub_api_key = settings.finnhub_api_key.get_secret_value()
        finnhub_gate = provider_gate_registry.get(
            provider="finnhub",
            max_concurrency=4,
            max_calls_per_minute=60,
        )
        research_tools.extend(
            (
                FinnhubCompanyNewsTool(
                    client=client,
                    api_key=finnhub_api_key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=finnhub_gate,
                ),
                FinnhubEarningsSurprisesTool(
                    client=client,
                    api_key=finnhub_api_key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=finnhub_gate,
                ),
                FinnhubBasicFinancialsTool(
                    client=client,
                    api_key=finnhub_api_key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=finnhub_gate,
                ),
            )
        )
    if settings.sec_user_agent is not None and settings.sec_user_agent.strip():
        research_tools.append(
            SecEdgarRecentFilingsTool(
                client=client,
                clock=clock,
                ticker_to_cik=_DEMO_SEC_TICKER_MAP,
                user_agent=settings.sec_user_agent,
                base_url=settings.sec_edgar_base_url,
            )
        )
    child_tools = ToolRegistry(research_tools)
    child_tool_names = frozenset(tool.spec.name for tool in research_tools)
    provider_health = await provider_gate_registry.snapshot_all()
    explicit_provider_intent = _explicit_provider_intent(objective)
    explicit_provider_unavailable = (
        explicit_provider_intent is not None
        and not _explicit_provider_is_admitted(
            explicit_provider_intent,
            available_tool_names=child_tool_names,
            provider_health=provider_health,
        )
    )

    def select_child_requirements(
        child_objective: str,
    ) -> tuple[EvidenceToolRequirement, ...]:
        return _child_evidence_requirements(
            child_objective,
            available_tool_names=child_tool_names,
        )

    memory_embedding_indexer = (
        PostgresMemoryEmbeddingIndexer(sessions, embedding_gateway, ids=ids)
        if sessions is not None and embedding_gateway is not None
        else None
    )
    memory_tools: tuple[Tool, ...] = ()
    if memory_authority is not None:
        assert sessions is not None
        memory_tools = build_explicit_memory_tools(
            service=ExplicitMemoryService(
                PostgresMemoryStore(sessions),
                clock,
                ids,
                embedding_indexer=memory_embedding_indexer,
            ),
            authority=memory_authority,
            clock=clock,
        )
    autonomous_memory_tools: tuple[Tool, ...] = ()
    if autonomous_memory_authority is not None:
        assert sessions is not None
        autonomous_memory_tools = build_autonomous_memory_tools(
            service=ExplicitMemoryService(
                PostgresMemoryStore(sessions),
                clock,
                ids,
                embedding_indexer=memory_embedding_indexer,
            ),
            authority=autonomous_memory_authority,
            clock=clock,
        )
    navigation_tools: tuple[Tool, ...] = ()
    if memory_navigation_authority is not None:
        assert sessions is not None
        navigation_tools = build_memory_navigation_tools(
            service=PostgresProgressiveMemoryService(sessions, embedding_gateway=embedding_gateway),
            authority=memory_navigation_authority,
            clock=clock,
        )
    required_memory_tool = memory_tools[0].spec.name if memory_tools else None
    required_memory_read_tool = (
        "memory.search"
        if required_memory_tool is None
        and navigation_tools
        and _requires_memory_search(objective, context_items)
        else None
    )
    memory_bound_turn = required_memory_tool is not None or required_memory_read_tool is not None
    explicitly_tool_free_requested = _effective_tool_free_request(
        objective,
        thread_root_objective=thread_root_objective,
    )
    direct_tool_free_turn = not memory_bound_turn and (
        explicitly_tool_free_requested or explicit_provider_unavailable
    )
    detected_evidence_requirements = (
        ()
        if memory_bound_turn or direct_tool_free_turn
        else _child_evidence_requirements(
            objective,
            available_tool_names=child_tool_names,
        )
    )
    plain_single_evidence_lookup = _is_plain_single_evidence_lookup(
        objective,
        detected_evidence_requirements,
    )
    category_screening_research_required = (
        not memory_bound_turn
        and not direct_tool_free_turn
        and _requires_current_equity_screening_research(routing_objective)
    )
    direct_external_evidence_required = (
        not memory_bound_turn
        and not direct_tool_free_turn
        and (
            _requires_external_evidence(objective, ())
            or _requires_external_evidence(routing_objective, ())
            or category_screening_research_required
        )
    )
    # Availability is decided before capability selection so the search ladder can
    # be advertised even when neither lexical nor semantic ranking produces a
    # confident match. "No obvious tool" must degrade to "search the web", never
    # to "no tools at all".
    # A memory turn no longer *forbids* research. "Remember what we said about
    # NVDA, then check where it's trading" is one question, and treating memory
    # and integrations as mutually exclusive made it unanswerable by
    # construction. The memory obligation still runs first via REQUIRED tool
    # choice; once it is satisfied, later AUTO turns may also read the web.
    research_tools_available = not direct_tool_free_turn and (
        direct_external_evidence_required
        or _research_is_available(objective, ())
        or _research_is_available(routing_objective, ())
    )
    verified_web_objective = (
        objective if explicit_provider_intent is not None else routing_objective
    )
    verified_web_route = (
        _select_verified_web_provider(verified_web_objective, child_tool_names)
        if direct_external_evidence_required
        and not detected_evidence_requirements
        and (
            category_screening_research_required
            or _requires_verified_web_chain(verified_web_objective)
        )
        and not re.search(r"https?://", objective, flags=re.IGNORECASE)
        else None
    )
    capability_catalog = _conversation_capability_catalog(
        [*research_tools, *navigation_tools, *thread_context_tools],
        provider_health=provider_health,
    )
    # Semantic recall: lexical/tag matching alone cannot bridge a vocabulary gap
    # between a conceptual query ("prognosis for the S&P 500") and a tool's literal
    # description. Embedding the catalog's summaries and the turn's own objective
    # lets discovery match by meaning, fused with the lexical score. This is a pure
    # best-effort addition -- an absent embedding_gateway (the default) leaves
    # discovery exactly as lexical-only as before, with zero extra HTTP calls; it
    # is an explicit opt-in parameter rather than auto-constructed from settings so
    # that callers using a mocked/test client never see an unexpected request.
    tool_embeddings = await ensure_tool_embeddings(
        embedding_gateway,
        tuple(
            (record.id, _capability_embedding_text(record))
            for record in capability_catalog.records()
        ),
        cache=(
            PostgresCapabilityEmbeddingStore(sessions, ids=ids) if sessions is not None else None
        ),
    )
    (query_embedding,) = (
        await embedding_gateway.embed((routing_objective,))
        if embedding_gateway is not None
        else (None,)
    )
    required_capability_names = frozenset(
        name
        for name in (
            required_memory_tool,
            required_memory_read_tool,
            *(requirement.tool_name for requirement in detected_evidence_requirements),
            *(sorted(verified_web_route.required_tools) if verified_web_route is not None else ()),
        )
        if name is not None
    )
    capabilities = CapabilityRuntime(
        capability_catalog,
        # A direct/no-tool instruction is authoritative model-input policy. Do not
        # let lexical skill recall re-introduce a research procedure through words
        # such as "workflow" or a negated "research" mention.
        skill_catalog=None if direct_tool_free_turn else SkillCatalog(_SKILL_ROOT),
        always_available_tool_names=(
            _PARENT_ORCHESTRATION_TOOL_NAMES
            # Every registered web-search route stays advertised whenever research
            # is available -- not just the combined family tool. Ranking decides
            # what Leo reaches for *first*; it must never decide whether searching
            # is possible at all. Names for unregistered tools are inert here, so
            # a single-provider deployment simply advertises fewer of them.
            | (_WEB_SEARCH_LADDER if research_tools_available else frozenset())
            | frozenset(tool.spec.name for tool in navigation_tools)
            | frozenset(tool.spec.name for tool in thread_context_tools)
        )
        - required_capability_names,
        required_tool_names=required_capability_names,
        embedding_gateway=embedding_gateway,
        tool_embeddings=tool_embeddings,
        query_embedding=query_embedding,
    )
    selected_skill_items = (
        ()
        if memory_bound_turn or direct_tool_free_turn
        else capabilities.skill_context_items(
            routing_objective,
            scope=scope,
            conversation_id=thread.origin.external_thread_id,
            phase=run.phase,
            roles=execution_scope.roles,
        )
    )
    skill_items = tuple(
        item
        for item in selected_skill_items
        if not (plain_single_evidence_lookup and item.id.startswith("skill:thesis_challenge:"))
    )
    runtime_context_items = (
        *context_items,
        *skill_items[: max(0, 128 - len(context_items))],
    )
    parent_tools: tuple[Tool, ...] = (
        SubagentResearchTool(
            model=model,
            tools=child_tools,
            context_items=runtime_context_items,
            clock=clock,
            ids=ids,
            run_store=child_run_store,
            parent_task_id=task.id if child_run_store is not None else None,
            requirement_selector=select_child_requirements,
        ),
        SubagentPlanTool(
            model=model,
            tools=child_tools,
            context_items=runtime_context_items,
            clock=clock,
            ids=ids,
            run_store=child_run_store,
            plan_store=durable_plan_store,
            parent_task_id=task.id if child_run_store is not None else None,
            parent_run_id=run.id if durable_plan_store is not None else None,
            plan_owner=(lease.owner if lease is not None else None),
            requirement_selector=select_child_requirements,
        ),
    )
    discovery_tools = build_capability_discovery_tools(capabilities, clock)
    tools = ToolRegistry(
        ()
        if direct_tool_free_turn
        else (
            *research_tools,
            *memory_tools,
            *autonomous_memory_tools,
            *navigation_tools,
            *thread_context_tools,
            *parent_tools,
            *discovery_tools,
        )
    )
    research_requirement = (
        None
        if memory_bound_turn or direct_tool_free_turn or plain_single_evidence_lookup
        else _selected_research_requirement(objective, skill_items)
    )
    # Obligation: raises the deliberation floor so the model must actually read
    # something. Reserved for the confident lexical signal so a trivial turn is
    # not forced into a pointless tool call.
    external_evidence_required = (
        not memory_bound_turn
        and not direct_tool_free_turn
        and (
            direct_external_evidence_required
            or _requires_external_evidence(objective, skill_items)
            or _requires_external_evidence(routing_objective, skill_items)
        )
    )
    deliberation = ElasticDeliberationPolicy().assess(
        objective if memory_bound_turn else routing_objective,
        # Skill procedures are untrusted instructions, not an antecedent that can
        # make an otherwise underspecified conversational follow-up sufficient.
        context_item_count=len(context_items),
        memory_recall_required=required_memory_read_tool is not None,
        state_mutation_required=required_memory_tool is not None,
        evidence_tool_names=tuple(
            requirement.tool_name for requirement in detected_evidence_requirements
        ),
        external_evidence_required=external_evidence_required,
        explicit_tool_free=explicitly_tool_free_requested,
        available_tool_names=child_tool_names | _PARENT_ORCHESTRATION_TOOL_NAMES,
    )
    if deliberation.hard_disable_tools and not memory_bound_turn:
        runtime_context_items = context_items
        tools = ToolRegistry(())
    # Recommendations remain advisory. Only an explicit user request to perform
    # orchestration becomes a required parent effect inside the trusted envelope.
    required_orchestration_tool = deliberation.required_parent_tool
    orchestration_required = required_orchestration_tool is not None
    forced_evidence_requirements = (
        detected_evidence_requirements
        if len(detected_evidence_requirements) == 1
        and research_requirement is None
        and not orchestration_required
        else ()
    )
    # The model decides; the harness does not decide for it.
    #
    # Four gateways used to sit here, each intercepting the model's turn and
    # substituting a decision the harness had authored:
    #
    #   _VerifiedWebResearchGateway       forced a search on turn 0 using the raw
    #                                     objective text as the query
    #   _DirectEvidenceCompletionGateway  wrote the answer itself from a provider
    #                                     payload, in provider-shaped prose
    #   _RequiredMemorySearchGateway      forced memory.search
    #   _ProviderUnavailableGateway       failed the run when one named provider
    #                                     was down, rather than letting the model
    #                                     re-route to another
    #
    # Together they routinely seized every turn of a run. A live two-ticker
    # comparison spent twelve consecutive iterations having its decision replaced
    # by a scraped page dump that then failed verification, the model's own
    # reasoning discarded each time -- the trace reads "no plan stated" precisely
    # because no model authored those decisions. The user got the raw scrape.
    #
    # What replaces them is the loop itself: the model plans its steps, and the
    # committed plan holds the run open until those steps have real observations
    # behind them. Tools get called because the plan requires them, not because a
    # keyword matched. ElasticDeliberationGateway stays for the one job that is
    # genuinely the harness's: noticing a loop that has stopped making progress.
    coordinator_model: ModelGateway = ElasticDeliberationGateway(model, deliberation)
    # `external_evidence_required` makes research *available* and raises the
    # deliberation floor; it deliberately does NOT make the verifier demand a
    # source claim. Those were previously the same flag, so widening research
    # availability would have simultaneously tightened completion -- the exact
    # combination that turns "no confident tool match" into a refusal. The
    # verifier stays strict only where a concrete evidence obligation was
    # actually detected.
    evidence_required = (
        research_requirement is not None
        or orchestration_required
        or bool(detected_evidence_requirements)
    )
    completion_guidance = _conversation_completion_guidance(
        memory_required=required_memory_tool is not None,
        memory_search_required=required_memory_read_tool is not None,
        research_required=research_requirement is not None,
        evidence_required=evidence_required,
        orchestration_required=orchestration_required,
    )
    completion_guidance = apply_deliberation_guidance(completion_guidance, deliberation)
    completion_contract = CompletionContract(
        source_claim_count=CardinalityBounds(
            minimum=1 if forced_evidence_requirements else 0,
            # A ceiling of one or two was set from prompt-shaped route detection
            # before the model had planned anything, and it capped how much
            # evidence an answer was allowed to cite. A run that read six sources
            # for a two-ticker comparison then could not cite them: the model
            # reported that "the completion contract restricts output to exactly
            # two source claims" and dropped the rest of its own work.
            #
            # Citing more of what was actually retrieved is never the failure mode
            # worth guarding against -- citing things that were *not* retrieved
            # is, and the verifier checks every claim against a real observation
            # regardless of how many there are. Only a no-tools turn still caps at
            # zero, because there is nothing legitimate to cite.
            maximum=0 if (memory_bound_turn or direct_tool_free_turn) else 8,
        ),
        source_observation_id_count=CardinalityBounds(minimum=1, maximum=8),
        inference_count=CardinalityBounds(
            minimum=1 if required_memory_read_tool is not None else 0,
            maximum=8,
        ),
        require_affected_assumption=research_requirement is not None,
        require_uncertainty=research_requirement is not None,
        guidance=completion_guidance,
    )
    coordinator = RunCoordinator(
        store=store,
        model=coordinator_model,
        tools=tools,
        transcript_sink=(
            PostgresModelCallTranscriptSink(sessions, ids=ids) if sessions is not None else None
        ),
        context=DefaultContextAssembler(
            evidence_requirements=forced_evidence_requirements,
            clock=clock,
            completion_contract=completion_contract,
            context_items=runtime_context_items,
            authority_snapshot_ids=(
                *context_authority_ids,
                *_thread_intent_routing_authority_ids(
                    context_items,
                    root_selected=thread_root_objective is not None,
                    category_screening_required=category_screening_research_required,
                ),
                deliberation.audit_source_id(),
            ),
            required_state_mutation_tool=required_memory_tool,
            required_read_tool=(
                required_memory_read_tool
                or required_orchestration_tool
                or (verified_web_route.search_tool if verified_web_route is not None else None)
            ),
        ),
        verifier=DeterministicCompletionVerifier(
            ids,
            clock,
            require_source_claim=evidence_required,
            evidence_requirements=detected_evidence_requirements,
            relax_integration_grounding=True,
            required_observation_kinds=frozenset(
                name
                for name in (required_memory_tool, required_memory_read_tool)
                if name is not None
            ),
            research_requirement=research_requirement,
            completion_contract=completion_contract,
            required_any_observation_kinds=(
                frozenset({"memory.search"})
                if required_memory_read_tool is not None
                else (
                    frozenset({"agent.execute_research_plan", "agent.delegate_research"})
                    if orchestration_required
                    else frozenset()
                )
            ),
            grounding_rules=(
                {
                    "memory.search": _ground_memory_observation,
                    "memory.open": _ground_memory_observation,
                    "memory.search_within": _ground_memory_observation,
                }
                if required_memory_read_tool is not None
                else None
            ),
        ),
        clock=clock,
        ids=ids,
        capabilities=capabilities,
    )
    return await coordinator.run(
        task_id=task.id,
        run_id=run.id,
        trusted_scope=execution_scope,
    )


def _validate_memory_authority(
    *,
    authority: MemoryMutationAuthority,
    objective: str,
    trusted_scope: TrustedScope,
    origin: OriginRef | None,
    sessions: async_sessionmaker[AsyncSession] | None,
    launch_ids: tuple[str, str, str] | None,
) -> None:
    """Reject memory capabilities unless every durable Slack boundary agrees."""

    if sessions is None or launch_ids is None or origin is None:
        raise ValueError("memory authority requires a durable admitted Slack launch")
    if origin.provider != "slack":
        raise ValueError("memory authority requires a Slack origin")
    _, task_id, run_id = launch_ids
    if authority.scope != trusted_scope.namespace:
        raise ValueError("memory authority scope does not match trusted scope")
    if authority.actor_id != trusted_scope.actor_id:
        raise ValueError("memory authority actor does not match trusted scope")
    if authority.task_id != task_id or authority.run_id != run_id:
        raise ValueError("memory authority launch does not match durable launch")
    if (
        authority.event_id != origin.external_event_id
        or authority.destination.external_id != origin.external_channel_id
    ):
        raise ValueError("memory authority origin does not match Slack event")
    expected_thread_prefix = (
        f"slack:{authority.destination.team_id}:{authority.destination.external_id}:"
    )
    if not origin.external_thread_id.startswith(expected_thread_prefix):
        raise ValueError("memory authority workspace does not match Slack origin")
    if parse_explicit_memory_intent(objective) != authority.intent:
        raise ValueError("memory authority intent does not match objective")


def _validate_autonomous_memory_authority(
    *,
    authority: MemoryMutationAuthority,
    trusted_scope: TrustedScope,
    origin: OriginRef | None,
    sessions: async_sessionmaker[AsyncSession] | None,
    launch_ids: tuple[str, str, str] | None,
) -> None:
    """Same durable-boundary checks as an explicit command, minus the intent match.

    An autonomous proposal is not triggered by a parsed command phrase, so there is
    no ``ExplicitMemoryIntent`` to cross-check the objective against -- every other
    scope/actor/launch/origin boundary still applies.
    """

    if sessions is None or launch_ids is None or origin is None:
        raise ValueError("memory authority requires a durable admitted Slack launch")
    if origin.provider != "slack":
        raise ValueError("memory authority requires a Slack origin")
    if authority.intent is not None:
        raise ValueError("autonomous memory authority must not carry a parsed intent")
    _, task_id, run_id = launch_ids
    if authority.scope != trusted_scope.namespace:
        raise ValueError("memory authority scope does not match trusted scope")
    if authority.actor_id != trusted_scope.actor_id:
        raise ValueError("memory authority actor does not match trusted scope")
    if authority.task_id != task_id or authority.run_id != run_id:
        raise ValueError("memory authority launch does not match durable launch")
    if (
        authority.event_id != origin.external_event_id
        or authority.destination.external_id != origin.external_channel_id
    ):
        raise ValueError("memory authority origin does not match Slack event")
    expected_thread_prefix = (
        f"slack:{authority.destination.team_id}:{authority.destination.external_id}:"
    )
    if not origin.external_thread_id.startswith(expected_thread_prefix):
        raise ValueError("memory authority workspace does not match Slack origin")


def _validate_memory_navigation_authority(
    *,
    authority: MemoryNavigationAuthority,
    trusted_scope: TrustedScope,
    origin: OriginRef | None,
    sessions: async_sessionmaker[AsyncSession] | None,
    launch_ids: tuple[str, str, str] | None,
) -> None:
    """Expose progressive memory only for one durable admitted Slack run."""

    if sessions is None or launch_ids is None or origin is None:
        raise ValueError("memory navigation requires a durable admitted Slack launch")
    if origin.provider != "slack":
        raise ValueError("memory navigation requires a Slack origin")
    _, task_id, run_id = launch_ids
    if authority.scope != trusted_scope.namespace:
        raise ValueError("memory navigation scope does not match trusted scope")
    if authority.actor_id != trusted_scope.actor_id:
        raise ValueError("memory navigation actor does not match trusted scope")
    if authority.task_id != task_id or authority.run_id != run_id:
        raise ValueError("memory navigation launch does not match durable launch")
    if authority.destination_id != origin.external_channel_id:
        raise ValueError("memory navigation destination does not match Slack origin")
    expected_thread_prefix = f"slack:{authority.team_id}:{authority.destination_id}:"
    if not origin.external_thread_id.startswith(expected_thread_prefix):
        raise ValueError("memory navigation workspace does not match Slack origin")


def _conversation_completion_guidance(
    *,
    memory_required: bool,
    memory_search_required: bool = False,
    research_required: bool = False,
    evidence_required: bool = False,
    orchestration_required: bool = False,
) -> str:
    if memory_required:
        return (
            "Answer conversationally. The explicit memory command is confirmed: call the required "
            "zero-argument memory tool before answering. Report its receipt plainly with no "
            "source_claims; internal memory state is not external evidence. Never claim an "
            "unobserved action or fact, and use only scoped context."
        )
    if memory_search_required:
        return (
            "Search authorized memory with concise terms and use only exact-conversation/DM "
            "results. For a match, copy one returned sentence into an inference and answer; cite "
            "that observation and add no facts. If selected_count=0 and items=[], use exactly "
            f"'{_EMPTY_MEMORY_SCOPE_INFERENCE}' as both inference and whole answer; never claim "
            "global absence. Keep source_claims empty: internal memory is not external evidence."
        )
    if orchestration_required:
        if research_required:
            return (
                "Execute the bounded parent plan before answering. Copy all verified child source "
                "statements exactly: one market quote and one SEC filing claim. Put both exact "
                "statements in the answer, cite only their parent plan observation, and add the "
                "affected assumption plus bounded uncertainty. Never cite child IDs directly."
            )
        return (
            "Execute a bounded parent plan or delegation before answering; do not merely promise "
            "future work. Copy every verified child source statement needed by the completed plan "
            "into separate claims and the final answer, citing only the parent observation. Never "
            "invent or cite child IDs; correct every verifier-reported grounding gap."
        )
    if research_required:
        return (
            "After current market and SEC evidence, return exactly two source claims, no extras: "
            "one exact symbol/current-price sentence and one exact canonical filings[0] sentence. "
            "Cite one matching observation per claim. Name the affected thesis assumption and "
            "state bounded uncertainty from counter-evidence; correct every verifier gap."
        )
    if evidence_required:
        return (
            "Gather external evidence now; never promise future work. Copy one exact canonical "
            "observation statement into claim and answer, citing that observation. Web-search "
            "snippets are discovery-only: fetch a selected URL before claiming its text."
        )
    return (
        "Answer conversationally; resolve follow-up antecedents from admitted thread context. If "
        "reliable information is missing, use an eligible tool now or ask one specific input "
        "question; never promise future work. Claims must copy supported observation text."
    )


def _requires_memory_search(objective: str, _context_items: tuple[ContextItem, ...]) -> bool:
    """Recognize explicit recall intent without hijacking conceptual memory discussion."""

    normalized = " ".join(objective.casefold().split())
    tokens = set(search_tokens(objective))
    has_exact_thread_context = any(
        item.kind is ContextItemKind.CONVERSATION_TURN
        and item.retention
        in {
            ContextItemRetention.THREAD_ROOT,
            ContextItemRetention.RECENT,
            ContextItemRetention.DECISION,
            ContextItemRetention.PRIOR_OUTCOME,
        }
        for item in _context_items
    )
    current_thread_recall = has_exact_thread_context and bool(
        re.search(
            r"\b(?:this|same|current)\s+(?:thread|conversation|dm|direct[- ]message|test)\b",
            normalized,
        )
    )
    current_thread_exchange_recall = has_exact_thread_context and bool(
        re.search(
            r"\b(?:what|which|where)\b.{0,120}\b(?:ask|asked|tell|told|say|said|"
            r"mention|mentioned|discuss|discussed|request|requested|marker|test|role)\b",
            normalized,
        )
    )
    direct_recall = bool(
        re.search(
            r"(?:^|[.!?]\s+)(?:please\s+)?(?:recall|remember)\b"
            r"|\b(?:do|did|can|could|would|will)\s+you\s+(?:still\s+)?"
            r"(?:recall|remember)\b"
            r"|\b(?:you|we)\s+(?:recall|recalled|remember|remembered)\b"
            r"|\b(?:recalled|remembered)\b.{0,40}\babout\b",
            normalized,
        )
    )
    memory_lookup = bool(
        re.search(
            r"\b(?:search|check|query|retrieve|find)\s+"
            r"(?:(?:your|our|the|stored|saved)\s+)?memor(?:y|ies)\b"
            r"|\blook\s+(?:in|through)\s+(?:(?:your|our|the)\s+)?memor(?:y|ies)\b",
            normalized,
        )
    )
    asked_to_remember = bool(
        re.search(r"\b(?:asked|told)\b.{0,80}\b(?:recall|remember)\b", normalized)
    )
    stored_personal_memory = bool(
        re.search(r"\b(?:our|your)\s+(?:saved|stored)\s+memor(?:y|ies)\b", normalized)
    )
    contextual_recall = bool(
        re.search(
            r"\b(?:again|which conversation did (?:it|that|this) come from|"
            r"where did (?:it|that|this) come from)\b",
            normalized,
        )
        and re.search(
            r"\b(?:again|color|conversation|source|it|that|this|where|which)\b",
            normalized,
        )
    )
    negated_current_exchange = bool(
        re.search(
            r"\b(?:i|we)\s+(?:(?:have|had|did)\s+not|haven't|hadn't|didn't|never)\s+"
            r"(?:yet\s+)?(?:tell|told|provide|provided|share|shared|say|said|mention|mentioned)\b",
            normalized,
        )
        or re.search(
            r"\b(?:has|have|had)\s+not\s+(?:yet\s+)?been\s+"
            r"(?:provided|shared|stated|described)\b",
            normalized,
        )
    )
    # A direct/no-tool request and an explicit statement that information has not
    # been supplied are present-turn clarification signals, not historical recall.
    # Preserve genuinely explicit recall/search commands even when both clauses occur.
    explicit_memory_intent = direct_recall or memory_lookup or asked_to_remember
    # The admitted thread transcript is the authoritative source for a question
    # explicitly scoped to this conversation.  Do not replace that exact context
    # with a cross-turn memory search (which can return an empty result and hide the
    # request the user just made).  Explicit memory commands still win.
    if (current_thread_recall or current_thread_exchange_recall) and not (
        memory_lookup or stored_personal_memory
    ):
        return False
    if _explicitly_requests_tool_free_answer(objective) and not (
        explicit_memory_intent or stored_personal_memory
    ):
        return False
    if negated_current_exchange and not (direct_recall or memory_lookup or stored_personal_memory):
        return False
    personal_reference = bool(re.search(r"\b(?:i|you|we|our|us)\b", normalized))
    explicit_did_exchange = bool(
        re.search(
            r"\bdid\s+(?:i|you|we)\s+"
            r"(?:agree|ask|decide|discuss|mention|say|tell)\b",
            normalized,
        )
    )
    past_exchange = (
        bool(
            tokens.intersection(
                {
                    "agreed",
                    "asked",
                    "decided",
                    "discussed",
                    "mentioned",
                    "said",
                    "told",
                }
            )
        )
        and personal_reference
    ) or explicit_did_exchange
    temporal_reference = bool(
        tokens.intersection({"before", "earlier", "previous", "previously", "prior"})
    ) and bool(
        tokens.intersection(
            {
                "agreement",
                "conversation",
                "decision",
                "discussion",
                "message",
                "thread",
            }
        )
    )
    return bool(
        direct_recall
        or memory_lookup
        or asked_to_remember
        or past_exchange
        or temporal_reference
        or stored_personal_memory
        or contextual_recall
    )


def _ground_memory_observation(
    statement: str,
    answer: str,
    observation: Observation,
) -> tuple[bool, str]:
    """Ground an inference only in model-visible text from an authorized memory result."""

    shape_error = _memory_observation_shape_error(observation)
    if shape_error is not None:
        return False, shape_error

    items = observation.data.get("items")
    if (
        observation.kind == "memory.search"
        and isinstance(items, list)
        and not items
        and observation.data.get("selected_count") == 0
    ):
        normalized_empty = _normalize_canonical_memory_inference(_EMPTY_MEMORY_SCOPE_INFERENCE)
        passed = (
            _normalize_canonical_memory_inference(statement) == normalized_empty
            and _normalize_canonical_memory_inference(answer) == normalized_empty
        )
        if passed:
            return (
                True,
                "Scoped no-match inference is grounded in an empty authorized memory search.",
            )
        return (
            False,
            f"For an empty authorized search, use exactly: {_EMPTY_MEMORY_SCOPE_INFERENCE}",
        )

    visible_text: list[str] = []
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("content", "excerpt"):
                value = item.get(key)
                if isinstance(value, str):
                    visible_text.append(value)
    chunks = observation.data.get("chunks")
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            text_value = chunk.get("text")
            if isinstance(text_value, str):
                visible_text.append(text_value)
    supported = any(_content_tokens_are_grounded(statement, value) for value in visible_text)
    carried = _content_tokens_are_grounded(statement, answer)
    if supported and carried:
        return True, "Inference is grounded in one authorized memory and carried by the answer."
    return (
        False,
        "Copy one complete returned memory sentence into both the inference and final answer; "
        "cite that observation, do not combine memories, paraphrase, negate, or add facts.",
    )


def _memory_observation_shape_error(observation: Observation) -> str | None:
    if observation.kind == "memory.search":
        items = observation.data.get("items")
        selected_count = observation.data.get("selected_count")
        if not isinstance(items, list):
            return "Memory search result is malformed: items must be a list."
        if type(selected_count) is not int or selected_count < 0:  # bool is not a valid count
            return (
                "Memory search result is malformed: selected_count must be a nonnegative integer."
            )
        if selected_count != len(items):
            return "Memory search result is malformed: selected_count does not match items."
    return None


def _normalize_canonical_memory_inference(value: str) -> str:
    return " ".join(value.casefold().split())


_MEMORY_GROUNDING_SCAFFOLD = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "according",
        "be",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "memory",
        "of",
        "on",
        "or",
        "recall",
        "recalled",
        "remember",
        "remembered",
        "says",
        "stored",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
    }
)


def _content_tokens_are_grounded(statement: str, visible_text: str) -> bool:
    """Accept formatting/reordering only when every substantive token stays in one text."""

    statement_tokens = _memory_grounding_tokens(statement)
    visible_tokens = _memory_grounding_tokens(visible_text)
    if not statement_tokens or not _token_multiset_is_subset(statement_tokens, visible_tokens):
        return False
    minimum_span = 1 if len(statement_tokens) == 1 else 2
    return _has_shared_contiguous_span(statement_tokens, visible_tokens, minimum_span)


def _memory_grounding_tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[^\W_]+", value.casefold())
        if token not in _MEMORY_GROUNDING_SCAFFOLD
    )


def _token_multiset_is_subset(needles: tuple[str, ...], haystack: tuple[str, ...]) -> bool:
    required = Counter(needles)
    available = Counter(haystack)
    return all(available[token] >= count for token, count in required.items())


def _has_shared_contiguous_span(
    needles: tuple[str, ...],
    haystack: tuple[str, ...],
    minimum_span: int,
) -> bool:
    for width in range(len(needles), minimum_span - 1, -1):
        for start in range(len(needles) - width + 1):
            candidate = needles[start : start + width]
            if any(
                haystack[offset : offset + width] == candidate
                for offset in range(len(haystack) - width + 1)
            ):
                return True
    return False


_LEGACY_RUNTIME_HEALTH_PROVIDERS: dict[str, frozenset[str]] = {
    "web.search_tavily": frozenset({"tavily"}),
    "web.search_exa": frozenset({"exa"}),
    "web.research_verified": frozenset({"exa", "tavily"}),
    "market.get_company_profile": frozenset({"finnhub"}),
    "market.get_company_news": frozenset({"finnhub"}),
    "market.get_earnings_surprises": frozenset({"finnhub"}),
    "market.get_basic_financials": frozenset({"finnhub"}),
}


def _capability_embedding_text(record: CatalogTool) -> str:
    """Describe a capability in prose, because that is what embeddings encode.

    This previously reused the lexical scoring basis verbatim, which appended the
    hand-maintained keyword tag bag ("outlook forecast prognosis trend ...") to
    every vector. Embedding a keyword salad makes the semantic channel a blurry
    copy of the lexical one and re-imports its blind spots -- a synonym missing
    from the tag list stayed missing from both signals, which is exactly how a
    stock-forecast question matched no research tool.

    The two channels are now genuinely independent: BM25 keeps the tags (it is
    supposed to be literal), while the vector encodes the tool's own
    natural-language description of what it does and what it returns.
    """

    return " ".join(
        part
        for part in (
            record.id.replace(".", " ").replace("_", " "),
            record.spec.domain,
            record.short_description,
            record.long_description or record.spec.description,
        )
        if part
    )


def _conversation_capability_catalog(
    tools: list[Tool],
    *,
    provider_health: tuple[ProviderHealthProjection, ...] = (),
) -> InMemoryToolCatalog:
    catalog = InMemoryToolCatalog(version="live-conversation-v1")
    health_by_provider = {item.provider: item for item in provider_health}
    metadata: dict[
        str,
        tuple[
            str,
            frozenset[str],
            CapabilitySensitivity,
            int | None,
            int | None,
            CapabilityLatency,
            frozenset[str],
        ],
    ] = {
        "web.search_public": (
            "wikipedia-opensearch",
            frozenset({"web", "internet", "search", "find", "page", "website", "public"}),
            CapabilitySensitivity.PUBLIC,
            None,
            None,
            CapabilityLatency.MEDIUM,
            frozenset({"discovery_only", "selected_result_requires_fetch"}),
        ),
        "web.search_tavily": (
            "tavily",
            frozenset(
                {
                    "web",
                    "internet",
                    "search",
                    "find",
                    "news",
                    "finance",
                    "public",
                    "change",
                    "changes",
                    "documentation",
                    "feature",
                    "features",
                    "language",
                    "programming",
                    "release",
                    "software",
                    "version",
                    "outlook",
                    "forecast",
                    "forecasts",
                    "prediction",
                    "predictions",
                    "prognosis",
                    "analysis",
                    "opinion",
                    "sentiment",
                    "index",
                    "indices",
                    "trend",
                    "trends",
                }
            ),
            CapabilitySensitivity.PUBLIC,
            600,
            None,
            CapabilityLatency.MEDIUM,
            frozenset({"discovery_only", "selected_result_requires_fetch"}),
        ),
        "web.search_exa": (
            EXA_CAPABILITY_DESCRIPTOR.provider,
            EXA_CAPABILITY_DESCRIPTOR.tags,
            CapabilitySensitivity.PUBLIC,
            EXA_CAPABILITY_DESCRIPTOR.freshness_seconds,
            None,
            CapabilityLatency.MEDIUM,
            EXA_CAPABILITY_DESCRIPTOR.verification_expectations,
        ),
        "web.research_verified": (
            "verified-web-provider-family",
            frozenset(
                {
                    "web",
                    "internet",
                    "search",
                    "research",
                    "source",
                    "news",
                    "current",
                    "comparison",
                    "landscape",
                    "verify",
                    "outlook",
                    "forecast",
                    "forecasts",
                    "prediction",
                    "predictions",
                    "prognosis",
                    "analysis",
                    "opinion",
                    "sentiment",
                    "index",
                    "indices",
                    "trend",
                    "trends",
                }
            ),
            CapabilitySensitivity.PUBLIC,
            600,
            None,
            CapabilityLatency.MEDIUM,
            frozenset(
                {
                    "bounded_provider_failover",
                    "exact_url_bound_highlight_or_retained_text",
                    "untrusted_content",
                }
            ),
        ),
        "web.fetch_public_text": (
            "web",
            frozenset(
                {
                    "web",
                    "internet",
                    "url",
                    "page",
                    "website",
                    "public",
                    "fetch",
                    "change",
                    "changes",
                    "documentation",
                    "feature",
                    "features",
                    "language",
                    "programming",
                    "release",
                    "software",
                    "version",
                }
            ),
            CapabilitySensitivity.PUBLIC,
            None,
            None,
            CapabilityLatency.MEDIUM,
            frozenset({"retained_text_grounding", "untrusted_content"}),
        ),
        "market.get_quote": (
            "finnhub",
            frozenset({"market", "stock", "quote", "price", "current", "latest"}),
            CapabilitySensitivity.PUBLIC,
            345_600,
            None,
            CapabilityLatency.LOW,
            frozenset({"exact_numeric_grounding", "as_of_freshness"}),
        ),
        "market.get_company_profile": (
            "finnhub",
            frozenset(
                {"market", "company", "profile", "identity", "industry", "exchange", "listing"}
            ),
            CapabilitySensitivity.PUBLIC,
            86_400,
            None,
            CapabilityLatency.LOW,
            frozenset({"canonical_statement", "provider_reported"}),
        ),
        "market.get_company_news": (
            "finnhub",
            frozenset({"market", "company", "news", "headline", "recent", "latest"}),
            CapabilitySensitivity.PUBLIC,
            900,
            None,
            CapabilityLatency.LOW,
            frozenset({"canonical_statement", "provider_reported", "as_of_freshness"}),
        ),
        "market.get_earnings_surprises": (
            "finnhub",
            frozenset({"market", "earnings", "eps", "actual", "estimate", "surprise", "quarter"}),
            CapabilitySensitivity.PUBLIC,
            21_600,
            None,
            CapabilityLatency.LOW,
            frozenset({"canonical_statement", "provider_reported"}),
        ),
        "market.get_basic_financials": (
            "finnhub",
            frozenset(
                {
                    "market",
                    "fundamentals",
                    "financials",
                    "metrics",
                    "beta",
                    "valuation",
                    "pe",
                }
            ),
            CapabilitySensitivity.PUBLIC,
            21_600,
            None,
            CapabilityLatency.LOW,
            frozenset({"canonical_statement", "provider_reported"}),
        ),
        "sec.get_recent_filings": (
            "sec-edgar",
            frozenset({"sec", "filing", "filings", "disclosure", "company", "primary-source"}),
            CapabilitySensitivity.PUBLIC,
            900,
            480,
            CapabilityLatency.MEDIUM,
            frozenset({"canonical_filing_metadata", "primary_source"}),
        ),
        "thread_context.open": (
            "leo-thread-context",
            frozenset({"thread", "history", "context", "earlier", "previous", "follow-up", "open"}),
            CapabilitySensitivity.INTERNAL,
            None,
            None,
            CapabilityLatency.LOW,
            frozenset({"current_authority", "internal_context", "bounded_handle"}),
        ),
    }
    for tool in tools:
        resolved: tuple[
            str,
            frozenset[str],
            CapabilitySensitivity,
            int | None,
            int | None,
            CapabilityLatency,
            frozenset[str],
        ]
        if tool.spec.name.startswith("memory."):
            resolved = (
                "leo-memory",
                frozenset({"memory", "internal", "retrieval"}),
                CapabilitySensitivity.INTERNAL,
                None,
                None,
                CapabilityLatency.LOW,
                frozenset({"current_authority", "internal_context"}),
            )
        elif tool.spec.name in CRYPTO_CAPABILITY_DESCRIPTORS:
            descriptor = CRYPTO_CAPABILITY_DESCRIPTORS[tool.spec.name]
            resolved = (
                descriptor.provider,
                descriptor.tags,
                descriptor.sensitivity,
                descriptor.freshness_seconds,
                descriptor.rate_limit_per_minute,
                descriptor.latency,
                descriptor.verification_expectations,
            )
        elif tool.spec.name in EQUITY_CAPABILITY_DESCRIPTORS:
            descriptor = EQUITY_CAPABILITY_DESCRIPTORS[tool.spec.name]
            resolved = (
                descriptor.provider,
                descriptor.tags,
                descriptor.sensitivity,
                descriptor.freshness_seconds,
                descriptor.rate_limit_per_minute,
                descriptor.latency,
                descriptor.verification_expectations,
            )
        else:
            resolved = metadata.get(
                tool.spec.name,
                (
                    "runtime",
                    frozenset({tool.spec.domain.lower()}),
                    CapabilitySensitivity.PUBLIC,
                    None,
                    None,
                    CapabilityLatency.MEDIUM,
                    frozenset({"scoped_observation"}),
                ),
            )
        provider, tags, sensitivity, freshness, rate, latency, verification = resolved
        catalog.register(
            catalog_tool_from_spec(
                tool.spec,
                provider=provider,
                tags=tags,
                health=_capability_health(tool.spec.name, health_by_provider),
                sensitivity=sensitivity,
                freshness_seconds=freshness,
                rate_limit_per_minute=rate,
                latency=latency,
                verification_expectations=verification,
            )
        )
    return catalog


def _capability_health(
    tool_name: str,
    health_by_provider: dict[str, ProviderHealthProjection],
) -> CapabilityHealth:
    """Project process-local provider state without exposing provider content.

    Provider identities are declared explicitly beside capability metadata. Virtual
    families are healthy when any currently registered route is healthy, degraded
    when only retryable/degraded routes remain, and rate-limited only when every
    registered route is locally unavailable.
    """

    descriptor = CRYPTO_CAPABILITY_DESCRIPTORS.get(tool_name)
    if descriptor is None:
        descriptor = EQUITY_CAPABILITY_DESCRIPTORS.get(tool_name)
    runtime_providers = (
        descriptor.runtime_health_providers
        if descriptor is not None
        else _LEGACY_RUNTIME_HEALTH_PROVIDERS.get(tool_name, frozenset())
    )
    observed = tuple(
        health_by_provider[provider]
        for provider in sorted(runtime_providers)
        if provider in health_by_provider
    )
    if not observed or any(item.status == "healthy" for item in observed):
        return CapabilityHealth.HEALTHY
    if any(item.status == "degraded" for item in observed):
        return CapabilityHealth.DEGRADED
    return CapabilityHealth.RATE_LIMITED


def _selected_research_requirement(
    objective: str,
    skill_items: tuple[ContextItem, ...],
) -> ResearchRequirement | None:
    if not _has_explicit_thesis_evidence_intent(objective) or not any(
        item.id.startswith("skill:thesis_challenge:") for item in skill_items
    ):
        return None
    return ResearchRequirement(
        required_kinds=frozenset({"market.get_quote", "sec.get_recent_filings"}),
        minimum_source_claims=2,
        minimum_distinct_sources=2,
        counter_evidence_kinds=frozenset({"market.get_quote"}),
        freshness_seconds=345_600,
        require_uncertainty_on_conflict=True,
        require_affected_assumption_on_conflict=True,
    )


def _is_plain_single_evidence_lookup(
    objective: str,
    requirements: tuple[EvidenceToolRequirement, ...],
) -> bool:
    """Keep an explicit one-provider lookup out of the multi-source thesis workflow."""

    if len(requirements) != 1:
        return False
    tokens = set(search_tokens(objective))
    return not tokens.intersection(
        {
            "assumption",
            "challenge",
            "compare",
            "counter",
            "counter-evidence",
            "reconcile",
            "thesis",
        }
    )


def _requires_verified_web_chain(objective: str) -> bool:
    """Return whether a natural objective specifically needs web discovery + fetch.

    Open-ended market/event prompts keep their semantic choice among admitted
    first-party/provider tools.  Explicit web research and versioned-change
    questions use the deterministic discovery-to-fetch admission path.
    """

    lowered = objective.casefold()
    tokens = set(search_tokens(objective))
    explicit_web_research = bool(
        tokens.intersection({"internet", "online", "source", "sources", "web", "website"})
        and tokens.intersection(
            {"browse", "find", "investigate", "lookup", "research", "search", "verify"}
        )
    )
    versioned_change = bool(
        re.search(r"\b[a-z][a-z0-9_.+-]{1,31}\s+v?\d+(?:\.\d+){1,3}\b", lowered)
        and tokens.intersection(
            {
                "change",
                "changed",
                "changes",
                "deprecation",
                "deprecations",
                "different",
                "feature",
                "features",
                "new",
                "noteworthy",
                "release",
                "released",
                "version",
            }
        )
    )
    return explicit_web_research or versioned_change


def _select_verified_web_provider(
    objective: str,
    available_tool_names: frozenset[str],
) -> _VerifiedWebProviderRoute | None:
    """Choose one admitted provider family without widening tool authority.

    Versioned software/documentation questions deliberately retain the observable
    Tavily discovery -> public fetch contract. Other open web-research objectives
    use the provider family when admitted, where Exa is attempted once and every
    typed failure can fall through to Tavily + fetch without spending model turns.
    """

    explicit_intent = _explicit_provider_intent(objective)
    if explicit_intent is not None and explicit_intent.provider == "exa":
        return (
            _EXA_VERIFIED_WEB_ROUTE
            if _EXA_VERIFIED_WEB_ROUTE.required_tools.issubset(available_tool_names)
            else None
        )
    if explicit_intent is not None and explicit_intent.provider == "tavily":
        return (
            _TAVILY_VERIFIED_WEB_ROUTE
            if _TAVILY_VERIFIED_WEB_ROUTE.required_tools.issubset(available_tool_names)
            else None
        )

    tavily_available = _TAVILY_VERIFIED_WEB_ROUTE.required_tools.issubset(available_tool_names)
    if tavily_available and _prefer_tavily_official_software_route(objective):
        return _TAVILY_VERIFIED_WEB_ROUTE
    family_available = _PROVIDER_FAMILY_VERIFIED_WEB_ROUTE.required_tools.issubset(
        available_tool_names
    )
    if family_available and tavily_available:
        return _PROVIDER_FAMILY_VERIFIED_WEB_ROUTE
    if tavily_available:
        return _TAVILY_VERIFIED_WEB_ROUTE
    if _EXA_VERIFIED_WEB_ROUTE.required_tools.issubset(available_tool_names):
        return _EXA_VERIFIED_WEB_ROUTE
    if family_available:
        return _PROVIDER_FAMILY_VERIFIED_WEB_ROUTE
    return None


def _explicit_provider_intent(objective: str) -> _ExplicitProviderIntent | None:
    """Recognize only unambiguous named-provider requests in a matching task family."""

    normalized = " ".join(objective.casefold().replace("-", " ").split())
    mentioned_web_providers = tuple(
        provider
        for provider in ("exa", "tavily")
        if re.search(rf"\b{provider}\b", normalized) is not None
    )
    # A comparison or other multi-provider request must never collapse to whichever
    # provider happens to satisfy the more specific positional phrase grammar.
    if len(mentioned_web_providers) > 1:
        return None
    explicit_web_providers = tuple(
        provider
        for provider in ("exa", "tavily")
        if any(
            re.search(pattern.format(provider=provider), normalized) is not None
            for pattern in (
                r"\b(?:use|query)\s+(?:the\s+)?{provider}\b",
                r"\bsearch\s+(?:the\s+web\s+)?(?:the\s+)?{provider}\b",
                r"\bsearch\s+(?:the\s+web\s+)?(?:with|using|via)\s+(?:the\s+)?{provider}\b",
                r"\b{provider}\s+(?:web\s+)?search\b",
            )
        )
    )
    if len(explicit_web_providers) == 1:
        return _EXPLICIT_PROVIDER_INTENTS[explicit_web_providers[0]]
    if len(explicit_web_providers) > 1:
        return None

    tokens = set(search_tokens(objective))
    quote_language = (
        bool(
            tokens.intersection(
                {
                    "current",
                    "latest",
                    "market",
                    "price",
                    "quote",
                    "rate",
                    "stock",
                    "today",
                    "trading",
                    "value",
                    "worth",
                }
            )
        )
        or _looks_like_trading_at_quote(objective)
        or _looks_like_current_trading_level(objective)
    )
    if _crypto_asset_from_tokens(tokens) is not None and quote_language:
        crypto_provider_patterns = (
            ("coingecko", r"coin\s*gecko"),
            ("coinmarketcap", r"coin\s*market\s*cap"),
        )
        mentioned_crypto_providers = tuple(
            (provider, pattern)
            for provider, pattern in crypto_provider_patterns
            if re.search(rf"\b{pattern}\b", normalized)
        )
        if len(mentioned_crypto_providers) > 1:
            return None
        if len(mentioned_crypto_providers) == 1:
            provider, pattern = mentioned_crypto_providers[0]
            if _explicit_named_source_request(normalized, pattern):
                return _EXPLICIT_PROVIDER_INTENTS[provider]

    if _equity_ticker_from_objective(objective, tokens) is not None:
        provider_patterns = (
            ("alpha_vantage", r"alpha\s*vantage"),
            ("ticker_layer", r"ticker\s*layer"),
            ("finnhub", r"finnhub"),
            ("massive", r"massive(?:\.com)?"),
        )
        mentioned_equity_providers = tuple(
            (provider_name, pattern)
            for provider_name, pattern in provider_patterns
            if re.search(rf"\b{pattern}\b", normalized)
        )
        if len(mentioned_equity_providers) > 1:
            return None
        equity_provider: str | None = None
        if len(mentioned_equity_providers) == 1:
            provider_name, pattern = mentioned_equity_providers[0]
            if _explicit_named_source_request(
                normalized,
                pattern,
                allow_leading=provider_name != "massive",
            ):
                equity_provider = provider_name
        if equity_provider is not None and tokens.intersection(
            {"company", "exchange", "industry", "listed", "listing", "profile"}
        ):
            return _provider_intent_for_tool(
                equity_provider,
                {
                    "alpha_vantage": "market.get_company_profile_alpha_vantage",
                    "finnhub": "market.get_company_profile_finnhub",
                    "massive": "market.get_company_profile_massive",
                    "ticker_layer": "market.get_company_profile_ticker_layer",
                }[equity_provider],
            )
        if equity_provider is not None and quote_language:
            return _EXPLICIT_PROVIDER_INTENTS[equity_provider]
    return None


def _explicit_named_source_request(
    normalized: str,
    provider_pattern: str,
    *,
    allow_leading: bool = True,
) -> bool:
    leading = (
        allow_leading and re.search(rf"^(?:please\s+)?{provider_pattern}\b", normalized) is not None
    )
    relational = (
        re.search(
            rf"\b(?:according\s+to|from|query|use|using|via|with)\s+"
            rf"(?:the\s+)?{provider_pattern}\b",
            normalized,
        )
        is not None
    )
    return (
        leading or relational or re.search(rf"\b{provider_pattern}\.com\b", normalized) is not None
    )


def _provider_intent_for_tool(provider: str, tool_name: str) -> _ExplicitProviderIntent:
    base = _EXPLICIT_PROVIDER_INTENTS[provider]
    return _ExplicitProviderIntent(
        provider=base.provider,
        display_name=base.display_name,
        primary_tool=tool_name,
        required_tools=frozenset({tool_name}),
    )


def _resolved_explicit_provider_tool(
    intent: _ExplicitProviderIntent,
    available_tool_names: frozenset[str],
) -> str | None:
    if intent.primary_tool in available_tool_names:
        return intent.primary_tool
    if (
        intent.provider == "finnhub"
        and intent.primary_tool == "market.get_company_profile_finnhub"
        and "market.get_company_profile" in available_tool_names
    ):
        return "market.get_company_profile"
    return None


def _explicit_provider_is_admitted(
    intent: _ExplicitProviderIntent,
    *,
    available_tool_names: frozenset[str],
    provider_health: tuple[ProviderHealthProjection, ...],
) -> bool:
    if intent.provider == "tavily":
        tools_admitted = intent.required_tools.issubset(available_tool_names)
    else:
        tools_admitted = _resolved_explicit_provider_tool(intent, available_tool_names) is not None
    if not tools_admitted:
        return False
    health = next((item for item in provider_health if item.provider == intent.provider), None)
    return health is None or health.status in {"healthy", "degraded"}


def _prefer_tavily_official_software_route(objective: str) -> bool:
    lowered = objective.casefold()
    tokens = set(search_tokens(objective))
    return bool(
        re.search(r"\b[a-z][a-z0-9_.+-]{1,31}\s+v?\d+(?:\.\d+){1,3}\b", lowered)
        and tokens.intersection(
            {
                "change",
                "changed",
                "changes",
                "deprecation",
                "deprecations",
                "different",
                "feature",
                "features",
                "new",
                "noteworthy",
                "release",
                "released",
                "version",
            }
        )
    )


_THREAD_ROOT_ROUTING_MAX_CHARS = 2_048
_CURRENT_ROUTING_MAX_CHARS = 4_096


def _trusted_thread_root_objective(context_items: tuple[ContextItem, ...]) -> str | None:
    """Return one bounded exact root selected by trusted retention metadata.

    Background DM continuity, compacted summaries, assistant outcomes, and ordinary
    conversation turns are deliberately ineligible. Multiple roots fail closed to the
    current objective because the live policy cannot prove one authoritative antecedent.
    """

    roots = tuple(
        item
        for item in context_items
        if item.kind is ContextItemKind.CONVERSATION_TURN
        and item.retention is ContextItemRetention.THREAD_ROOT
    )
    if len(roots) != 1:
        return None
    content = roots[0].content.strip()
    header, separator, body = content.partition("\n")
    if separator and header.startswith("[Slack ") and header.endswith("]"):
        content = body.strip()
    normalized = " ".join(content.split())
    return normalized[:_THREAD_ROOT_ROUTING_MAX_CHARS].rstrip() or None


def _thread_intent_routing_authority_ids(
    context_items: tuple[ContextItem, ...],
    *,
    root_selected: bool,
    category_screening_required: bool,
) -> tuple[str, ...]:
    """Project content-free routing diagnostics into the context authority audit."""

    roots = tuple(
        item
        for item in context_items
        if item.kind is ContextItemKind.CONVERSATION_TURN
        and item.retention is ContextItemRetention.THREAD_ROOT
    )
    status = "none" if not roots else "single" if len(roots) == 1 and root_selected else "conflict"
    return (
        "thread-intent-routing-version:v1",
        f"thread-intent-root-candidate-count:{len(roots)}",
        f"thread-intent-root-selected:{str(root_selected).lower()}",
        f"thread-intent-root-status:{status}",
        f"thread-intent-category-screening:{str(category_screening_required).lower()}",
    )


def _thread_intent_routing_objective(
    objective: str,
    context_items: tuple[ContextItem, ...],
) -> str:
    """Combine only current text and one exact root for semantic route selection.

    The current follow-up stays first and is never replaced on ``Task`` or
    ``ModelRequest``. This projection may select capabilities or evidence depth; it is
    never passed to tool-argument binders or provider calls.
    """

    current = " ".join(objective.split())[:_CURRENT_ROUTING_MAX_CHARS].rstrip()
    root = _trusted_thread_root_objective(context_items)
    if root is None:
        return current or objective
    return f"Current follow-up: {current}\nExact thread root: {root}"


def _effective_tool_free_request(
    objective: str,
    *,
    thread_root_objective: str | None,
) -> bool:
    """Apply current-turn tool policy first, then carry an unopposed root constraint."""

    if _explicitly_requests_tool_free_answer(objective):
        return True
    if thread_root_objective is None or not _explicitly_requests_tool_free_answer(
        thread_root_objective
    ):
        return False
    current_explicitly_needs_tools = bool(
        _explicit_provider_intent(objective) is not None
        or _requires_external_evidence(objective, ())
        or _requires_verified_web_chain(objective)
        or re.search(
            r"\b(?:browse|research|search|verify|look\s+(?:it|this|that)\s+up)\b",
            objective.casefold(),
        )
        is not None
    )
    return not current_explicitly_needs_tools


def _requires_current_equity_screening_research(objective: str) -> bool:
    """Recognize current plural equity screens without inventing a ticker argument."""

    tokens = set(search_tokens(objective))
    equity_domain = bool(
        tokens.intersection(
            {
                "companies",
                "dividend",
                "dividends",
                "equities",
                "income",
                "investing",
                "investment",
                "investments",
                "stock",
                "stocks",
                "yield",
            }
        )
    )
    plural_universe = _looks_like_equity_category_request(tokens) or bool(
        tokens.intersection({"companies", "dividends", "equities", "stocks"})
    )
    screening_intent = bool(
        tokens.intersection(
            {
                "ideas",
                "mix",
                "names",
                "opportunities",
                "picks",
                "recommend",
                "recommendations",
                "screen",
                "screening",
                "shortlist",
                "suggest",
            }
        )
    )
    current_intent = bool(
        tokens.intersection({"current", "currently", "latest", "now", "recent", "today"})
    )
    selection_criteria = bool(
        tokens.intersection(
            {
                "dividend",
                "dividends",
                "established",
                "growth",
                "income",
                "long-term",
                "risk",
                "steady",
                "steadier",
                "yield",
            }
        )
    )
    return all(
        (
            equity_domain,
            plural_universe,
            screening_intent,
            current_intent,
            selection_criteria,
        )
    )


def _requires_external_evidence(
    objective: str,
    skill_items: tuple[ContextItem, ...],
) -> bool:
    if _explicitly_requests_tool_free_answer(objective):
        return False
    # Skill procedures are context, not an availability or completion gate. A
    # harness-known skill may add an obligation only when the user's own objective
    # independently expresses the matching evidence intent. This prevents a lexical
    # distractor from turning arbitrary conversation into financial research.
    trusted_skill_ids = tuple(
        item.id for item in skill_items if item.kind is ContextItemKind.SKILL_PROCEDURE
    )
    if any(item.startswith("skill:thesis_challenge:") for item in trusted_skill_ids) and (
        _has_explicit_thesis_evidence_intent(objective)
    ):
        return True
    if any(item.startswith("skill:narrow_quote:") for item in trusted_skill_ids) and bool(
        _child_evidence_requirements(
            objective,
            available_tool_names=frozenset({"market.get_quote", "sec.get_recent_filings"}),
        )
    ):
        return True
    lowered = objective.lower()
    if "http://" in lowered or "https://" in lowered:
        return True
    tokens = set(search_tokens(objective))
    market_domain = tokens.intersection(
        {
            "company",
            "coin",
            "crypto",
            "cryptocurrency",
            "earnings",
            "eps",
            "financial",
            "financials",
            "filing",
            "filings",
            "headline",
            "headlines",
            "fundamental",
            "fundamentals",
            "market",
            "news",
            "price",
            "profile",
            "quote",
            "sec",
            "stock",
            "token",
        }
    )
    freshness_or_research = tokens.intersection(
        {"current", "investigate", "latest", "recent", "research", "today", "verify"}
    )
    web_domain = tokens.intersection(
        {"internet", "online", "source", "sources", "url", "web", "website"}
    )
    web_action = tokens.intersection(
        {"browse", "find", "investigate", "lookup", "research", "search", "verify"}
    )
    open_current_event = bool(
        re.search(
            r"\b(?:what happened|what changed|what(?:'s| is) new|latest developments?)\b",
            objective.casefold(),
        )
        and tokens.intersection({"current", "latest", "now", "recent", "today"})
    )
    versioned_change = bool(
        re.search(r"\b[a-z][a-z0-9_.+-]{1,31}\s+v?\d+(?:\.\d+){1,3}\b", lowered)
        and tokens.intersection(
            {
                "change",
                "changed",
                "changes",
                "deprecation",
                "deprecations",
                "different",
                "feature",
                "features",
                "new",
                "noteworthy",
                "release",
                "released",
                "version",
            }
        )
    )
    return bool(
        (market_domain and freshness_or_research)
        or (web_domain and web_action)
        or open_current_event
        or versioned_change
    )


def _research_is_available(objective: str, skill_items: tuple[ContextItem, ...]) -> bool:
    """Return whether research tools should be advertised for this turn.

    `_requires_external_evidence` above is a fast, high-confidence *yes*. Its
    absence was previously treated as a *no*, which is what broke the reported
    turn: "what's the year end forecast for GOOG?" contains no freshness word
    and no market word, so the envelope collapsed to depth-0 `direct`, no tool
    was ever advertised, and the run ended in a canned "I'm missing a reliable
    source" reply.

    Burden of proof is therefore inverted here. Research stays *available* for
    anything not recognizably self-contained, while remaining *obligatory* only
    under the confident signal -- so an unnecessary search costs a few cents and
    a trivial question still answers in one turn.
    """

    if _requires_external_evidence(objective, skill_items):
        return True
    return not _is_self_contained_conversational_turn(objective)


_CONVERSATIONAL_ONLY = re.compile(
    r"^(?:"
    r"(?:hi|hey|hello|yo|thanks|thank you|thx|ty|ok|okay|got it|nice|cool|great|"
    r"good morning|good afternoon|good evening|gm|gn)\b"
    r"|(?:who are you|what are you|what can you do|what do you do|help|"
    r"how do you work|what tools do you have)\b"
    r")",
    re.IGNORECASE,
)
_SELF_REFERENTIAL_TASK = re.compile(
    r"\b(?:rewrite|rephrase|reword|summari[sz]e|shorten|translate|proofread|"
    r"format|reformat|bullet|tidy up|clean up)\b\s+(?:this|that|the above|it|my)\b",
    re.IGNORECASE,
)
# A question *about the conversation* ("what did we call the demo?") is answered
# from context or memory. The public web does not know what you and Leo agreed,
# so advertising search for it is pure noise.
_CONVERSATION_REFERENCE = re.compile(
    # Both the plain past ("we called it X") and the auxiliary-plus-base form
    # produced by questions ("what did we call it?").
    r"\b(?:we|you|i)\s+(?:just\s+|already\s+|previously\s+)?"
    r"(?:say|said|call|called|name|named|discuss|discussed|mention|mentioned|"
    r"agree|agreed|decide|decided|choose|chose|pick|picked|"
    r"talk(?:ed)? about|went with|go with|settle[d]? on)\b",
    re.IGNORECASE,
)


def _is_self_contained_conversational_turn(objective: str) -> bool:
    """Recognize turns that genuinely need no external lookup.

    Deliberately small and high-precision. Anything not matched here keeps
    research available -- a false negative costs one search, a false positive
    costs an unanswered question.
    """

    normalized = " ".join(objective.split())
    if not normalized:
        return True
    if _explicitly_requests_tool_free_answer(objective):
        return True
    if _CONVERSATIONAL_ONLY.match(normalized) and len(normalized.split()) <= 8:
        return True
    # "summarize this", "rewrite the above" operate on supplied material.
    if _SELF_REFERENTIAL_TASK.search(normalized):
        return True
    if _CONVERSATION_REFERENCE.search(normalized):
        return True
    # Pure arithmetic with no prose subject.
    return bool(re.fullmatch(r"[\d\s+\-*/().,%^=]+\??", normalized))


def _child_evidence_requirements(
    objective: str,
    *,
    available_tool_names: frozenset[str],
) -> tuple[EvidenceToolRequirement, ...]:
    """Bind explicit child research objectives to deterministic evidence tools."""

    if _explicitly_requests_tool_free_answer(objective):
        return ()
    tokens = set(search_tokens(objective))
    literal_tokens = set(re.findall(r"[a-z0-9.-]{2,64}", objective.casefold()))
    explicit_provider_intent = _explicit_provider_intent(objective)
    crypto_asset = _crypto_asset_from_tokens(tokens)
    crypto_quote_currency = _crypto_quote_currency(tokens)
    crypto_quote_language = (
        bool(
            tokens.intersection(
                {
                    "current",
                    "latest",
                    "market",
                    "price",
                    "quote",
                    "rate",
                    "today",
                    "trading",
                    "value",
                    "worth",
                }
            )
        )
        or _looks_like_trading_at_quote(objective)
        or _looks_like_current_trading_level(objective)
    )
    if crypto_asset is not None and crypto_quote_currency is not None and crypto_quote_language:
        if explicit_provider_intent is not None and explicit_provider_intent.provider in {
            "coingecko",
            "coinmarketcap",
        }:
            direct_crypto_tool = _resolved_explicit_provider_tool(
                explicit_provider_intent, available_tool_names
            )
            if direct_crypto_tool is None:
                return ()
            crypto_tool_name = direct_crypto_tool
        elif "market.get_crypto_snapshot" in available_tool_names:
            crypto_tool_name = "market.get_crypto_snapshot"
        else:
            crypto_tool_name = ""
        if not crypto_tool_name:
            return ()
        return (
            EvidenceToolRequirement(
                observation_kind=crypto_tool_name,
                tool_name=crypto_tool_name,
                required_arguments=(
                    ToolArgumentConstraint(name="asset_id", value=crypto_asset),
                    ToolArgumentConstraint(
                        name="quote_currency",
                        value=crypto_quote_currency,
                    ),
                ),
            ),
        )
    ticker = _equity_ticker_from_objective(objective, tokens)
    if ticker is None:
        return ()

    requirements: list[EvidenceToolRequirement] = []
    market_quote_language = (
        bool(tokens.intersection({"market", "price", "quote", "stock"}))
        or _looks_like_trading_at_quote(objective)
        or _looks_like_current_trading_level(objective)
    )
    if market_quote_language:
        if explicit_provider_intent is not None and explicit_provider_intent.provider in {
            "alpha_vantage",
            "finnhub",
            "massive",
            "ticker_layer",
        }:
            # Provider intent and operation intent are independent. A combined
            # quote/profile request makes the provider intent's primary tool the
            # profile route, but the quote obligation must still use that provider's
            # direct quote tool rather than duplicating the profile requirement.
            direct_quote_intent = _EXPLICIT_PROVIDER_INTENTS[explicit_provider_intent.provider]
            quote_tool_name = (
                _resolved_explicit_provider_tool(direct_quote_intent, available_tool_names) or ""
            )
        elif "market.get_quote" in available_tool_names:
            quote_tool_name = "market.get_quote"
        else:
            quote_tool_name = ""
    else:
        quote_tool_name = ""
    if quote_tool_name:
        requirements.append(
            EvidenceToolRequirement(
                observation_kind=quote_tool_name,
                tool_name=quote_tool_name,
                required_arguments=(ToolArgumentConstraint(name="symbol", value=ticker),),
            )
        )
    explicit_profile_tool = (
        _resolved_explicit_provider_tool(explicit_provider_intent, available_tool_names)
        if explicit_provider_intent is not None
        and "company_profile" in explicit_provider_intent.primary_tool
        else None
    )
    if explicit_provider_intent is not None and "company_profile" in (
        explicit_provider_intent.primary_tool
    ):
        profile_tool_name = explicit_profile_tool or ""
    else:
        profile_tool_name = (
            "market.get_equity_profile"
            if "market.get_equity_profile" in available_tool_names
            else "market.get_company_profile"
        )
    if profile_tool_name in available_tool_names and literal_tokens.intersection(
        {"company", "exchange", "industry", "listed", "listing", "profile"}
    ):
        requirements.append(
            EvidenceToolRequirement(
                observation_kind=profile_tool_name,
                tool_name=profile_tool_name,
                required_arguments=(ToolArgumentConstraint(name="symbol", value=ticker),),
            )
        )
    if "market.get_company_news" in available_tool_names and literal_tokens.intersection(
        {"headline", "headlines", "news"}
    ):
        requirements.append(
            EvidenceToolRequirement(
                observation_kind="market.get_company_news",
                tool_name="market.get_company_news",
                required_arguments=(ToolArgumentConstraint(name="symbol", value=ticker),),
            )
        )
    if "market.get_earnings_surprises" in available_tool_names and literal_tokens.intersection(
        {"earnings", "eps", "surprise", "surprises"}
    ):
        requirements.append(
            EvidenceToolRequirement(
                observation_kind="market.get_earnings_surprises",
                tool_name="market.get_earnings_surprises",
                required_arguments=(ToolArgumentConstraint(name="symbol", value=ticker),),
            )
        )
    if "market.get_basic_financials" in available_tool_names and literal_tokens.intersection(
        {"beta", "financial", "financials", "fundamental", "fundamentals", "metrics", "pe"}
    ):
        requirements.append(
            EvidenceToolRequirement(
                observation_kind="market.get_basic_financials",
                tool_name="market.get_basic_financials",
                required_arguments=(ToolArgumentConstraint(name="symbol", value=ticker),),
            )
        )
    if (
        ticker in _DEMO_SEC_TICKER_MAP
        and "sec.get_recent_filings" in available_tool_names
        and tokens.intersection({"disclosure", "filing", "filings", "metadata", "sec"})
    ):
        requirements.append(
            EvidenceToolRequirement(
                observation_kind="sec.get_recent_filings",
                tool_name="sec.get_recent_filings",
                required_arguments=(ToolArgumentConstraint(name="ticker", value=ticker),),
            )
        )
    return tuple(requirements)


def _crypto_asset_from_tokens(tokens: set[str]) -> str | None:
    matches = {canonical for alias, canonical in _CRYPTO_ASSET_ALIASES.items() if alias in tokens}
    return next(iter(matches)) if len(matches) == 1 else None


def _crypto_quote_currency(tokens: set[str]) -> str | None:
    aliases = {
        "dollar": "USD",
        "dollars": "USD",
        "eur": "EUR",
        "euro": "EUR",
        "euros": "EUR",
        "gbp": "GBP",
        "jpy": "JPY",
        "pound": "GBP",
        "pounds": "GBP",
        "usd": "USD",
        "yen": "JPY",
    }
    matches = {currency for alias, currency in aliases.items() if alias in tokens}
    if len(matches) > 1:
        return None
    return next(iter(matches), "USD")


def _explicitly_requests_tool_free_answer(objective: str) -> bool:
    """Honor an explicit direct-answer/no-research instruction before lexical routing."""

    normalized = " ".join(objective.casefold().replace("\u2019", "'").split())
    patterns = (
        r"\b(?:do\s+not|don't|dont|never)\s+(?:do\s+)?(?:any\s+)?"
        r"(?:external\s+)?(?:research|browse|search|look\s+anything\s+up)\b",
        r"\b(?:do\s+not|don't|dont|never)\s+(?:use|call|invoke)\s+"
        r"(?:any\s+)?tools?\b",
        r"\bwithout\s+(?:doing\s+)?(?:any\s+)?(?:external\s+)?research\b",
        r"\bwithout\s+(?:using\s+)?(?:any\s+)?tools?\b",
    )
    return any(re.search(pattern, normalized) is not None for pattern in patterns)


def _has_explicit_thesis_evidence_intent(objective: str) -> bool:
    """Require an entity plus thesis and evidence language before deep research."""

    if _explicitly_requests_tool_free_answer(objective):
        return False
    tokens = set(search_tokens(objective))
    if _equity_ticker_from_objective(objective, tokens) is None:
        return False
    thesis_intent = tokens.intersection(
        {"assumption", "challenge", "compare", "counter", "counter-evidence", "thesis"}
    )
    evidence_intent = tokens.intersection(
        {
            "current",
            "evidence",
            "filing",
            "filings",
            "market",
            "primary-source",
            "quote",
            "research",
            "sec",
            "source",
            "sources",
            "verify",
        }
    )
    return bool(thesis_intent and evidence_intent)


def _demo_ticker_from_tokens(tokens: set[str]) -> str | None:
    matches = {candidate for candidate in _DEMO_SEC_TICKER_MAP if candidate.lower() in tokens} | {
        symbol for name, symbol in _DEMO_ENTITY_ALIASES.items() if name in tokens
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _equity_ticker_from_objective(objective: str, tokens: set[str]) -> str | None:
    """Resolve one bounded equity symbol while rejecting acronyms and ambiguity.

    The fixed demo map remains the only ticker-to-CIK authority for SEC reads. Equity
    provider tools can accept one explicit uppercase ticker or cashtag beyond that map.
    """

    cashtags = {
        match.upper()
        for match in re.findall(
            r"(?<![A-Za-z0-9.-])\$([A-Za-z][A-Za-z0-9]{0,9}(?:[.-][A-Za-z0-9]{1,4})?)"
            r"(?![A-Za-z0-9.-])",
            objective,
        )
    }
    labelled_symbols = {
        match.upper()
        for match in re.findall(
            r"(?i:\bticker|\bsymbol)\s*(?:(?i:is)\s+|[:=]\s*)?"
            r"([A-Z][A-Z0-9]{0,9}(?:[.-][A-Z0-9]{1,4})?)\b",
            objective,
        )
    }
    explicit_symbols = cashtags | labelled_symbols
    if explicit_symbols:
        return next(iter(explicit_symbols)) if len(explicit_symbols) == 1 else None

    # A category or screening request is not a request for one security. In
    # particular, capitalized geography/strategy acronyms in prompts such as
    # "UK dividend stocks" must not be guessed to be ticker symbols.
    if _looks_like_equity_category_request(tokens):
        return None

    known = _demo_ticker_from_tokens(tokens)
    if known is not None:
        return known
    uppercase_symbols = set(
        re.findall(
            r"(?<![A-Za-z0-9.-])([A-Z][A-Z0-9]{0,9}(?:[.-][A-Z0-9]{1,4})?)"
            r"(?![A-Za-z0-9.-])",
            objective,
        )
    )
    candidates = {
        candidate
        for candidate in uppercase_symbols
        if len(candidate) > 1 and candidate not in _NON_EQUITY_SYMBOL_TOKENS
    }
    return next(iter(candidates)) if len(candidates) == 1 else None


def _looks_like_equity_category_request(tokens: set[str]) -> bool:
    """Return whether the objective describes a plural investment universe."""

    return bool(
        tokens.intersection(
            {
                "companies",
                "equities",
                "ideas",
                "investments",
                "names",
                "opportunities",
                "picks",
                "stocks",
            }
        )
    )


def _looks_like_trading_at_quote(objective: str) -> bool:
    """Recognize the price idiom without treating generic trading prose as a quote."""

    normalized = " ".join(objective.casefold().split())
    return (
        re.search(
            r"\b(?:trade|trades|trading)\s+at"
            r"(?:\s+(?:(?:right\s+)?now|currently|today|the\s+(?:moment|open|close)))?"
            r"\s*[?!.]*\s*$",
            normalized,
        )
        is not None
    )


def _looks_like_current_trading_level(objective: str) -> bool:
    """Recognize natural market-level phrasing without treating generic trading as price."""

    normalized = " ".join(objective.casefold().split())
    return bool(
        re.search(
            r"\b(?:current|latest|live|today(?:'s)?)\s+(?:market|share|stock|trading)\s+"
            r"(?:level|price|value)\b|\b(?:market|share|stock|trading)\s+"
            r"(?:level|price)\s+(?:right\s+)?now\b",
            normalized,
        )
    )
