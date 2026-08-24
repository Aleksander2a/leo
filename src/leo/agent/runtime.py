"""Composition root: build the tool set, wire the loop, run one turn end to end."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from leo.agent.contracts import Clock, Scope, Tool
from leo.agent.db import create_engine, create_sessions
from leo.agent.discovery import ToolDiscovery, ToolFinderTool
from leo.agent.llm import LLM
from leo.agent.loop import Agent, AgentResult, LoopLimits, StepCallback
from leo.agent.memory import MemoryService, build_memory_tools
from leo.agent.store import AgentStore
from leo.agent.tools import ToolRegistry
from leo.config import Settings, has_value, is_configured_secret
from leo.integrations.crypto_composition import build_crypto_market_tools
from leo.integrations.equity_composition import build_equity_market_tools
from leo.integrations.exa import EXA_CAPABILITY_DESCRIPTOR, ExaSearchTool
from leo.integrations.finnhub import (
    FinnhubBasicFinancialsTool,
    FinnhubCompanyNewsTool,
    FinnhubEarningsSurprisesTool,
)
from leo.integrations.mcp_tools import (
    build_alpha_vantage_mcp_tools,
    build_coingecko_mcp_tools,
    build_tavily_mcp_tools,
)
from leo.integrations.provider_runtime import ProviderGateRegistry
from leo.integrations.sec_edgar import SecEdgarRecentFilingsTool
from leo.integrations.tavily import TAVILY_FREE_TIER_MONTHLY_CREDITS, TavilySearchTool
from leo.integrations.verified_web import VerifiedWebResearchTool
from leo.integrations.web_fetch import PublicTextFetchTool
from leo.integrations.web_search import PublicWebSearchTool

logger = logging.getLogger(__name__)


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def build_tools(
    *,
    settings: Settings,
    client: httpx.AsyncClient,
    clock: Clock,
    gates: ProviderGateRegistry,
) -> list[Tool]:
    """Every capability the deployment is credentialed for.

    Nothing here is conditional on the *question* -- only on whether a provider
    is configured. Which tools a given turn sees is decided semantically at
    turn time by :mod:`leo.agent.discovery`, not by a keyword table.
    """

    fetch = PublicTextFetchTool(client=client, clock=clock)
    tools: list[Tool] = [
        PublicWebSearchTool(client=client, clock=clock, user_agent=settings.sec_user_agent),
        fetch,
    ]

    tavily: TavilySearchTool | None = None
    if is_configured_secret(settings.tavily_api_key):
        assert settings.tavily_api_key is not None
        tavily = TavilySearchTool(
            client=client,
            api_key=settings.tavily_api_key.get_secret_value(),
            clock=clock,
            max_calls_per_minute=settings.tavily_max_calls_per_minute,
            max_calls_per_month=settings.tavily_max_calls_per_month,
            gate=gates.get(
                provider="tavily",
                max_concurrency=4,
                max_calls_per_minute=settings.tavily_max_calls_per_minute,
                max_calls_per_month=settings.tavily_max_calls_per_month,
                max_provider_credits_per_month=TAVILY_FREE_TIER_MONTHLY_CREDITS,
            ),
        )
        tools.append(tavily)

    exa: ExaSearchTool | None = None
    if is_configured_secret(settings.exa_api_key):
        assert settings.exa_api_key is not None
        exa = ExaSearchTool(
            client=client,
            api_key=settings.exa_api_key.get_secret_value(),
            clock=clock,
            max_calls_per_minute=EXA_CAPABILITY_DESCRIPTOR.max_calls_per_minute,
            gate=gates.get(
                provider="exa",
                max_concurrency=4,
                max_calls_per_minute=EXA_CAPABILITY_DESCRIPTOR.max_calls_per_minute,
            ),
        )
        tools.append(exa)

    if exa is not None and tavily is not None:
        # Search-then-fetch across two providers, so one outage does not remove
        # Leo's ability to read a page.
        tools.append(VerifiedWebResearchTool(exa=exa, tavily=tavily, fetch=fetch))

    tools.extend(
        build_crypto_market_tools(
            settings=settings, client=client, clock=clock, provider_gates=gates
        )
    )
    tools.extend(
        build_equity_market_tools(
            settings=settings, client=client, clock=clock, provider_gates=gates
        )
    )

    if is_configured_secret(settings.tavily_endpoint):
        assert settings.tavily_endpoint is not None
        tools.extend(
            build_tavily_mcp_tools(
                endpoint=settings.tavily_endpoint.get_secret_value(),
                clock=clock,
                gate=gates.get(provider="tavily_mcp", max_concurrency=2, max_calls_per_minute=20),
            )
        )
    if is_configured_secret(settings.alpha_vantage_endpoint_legacy):
        assert settings.alpha_vantage_endpoint_legacy is not None
        tools.extend(
            build_alpha_vantage_mcp_tools(
                endpoint=settings.alpha_vantage_endpoint_legacy.get_secret_value(),
                clock=clock,
                gate=gates.get(
                    provider="alpha_vantage_mcp",
                    max_concurrency=2,
                    max_calls_per_minute=settings.alpha_vantage_max_calls_per_minute,
                ),
            )
        )
    if is_configured_secret(settings.coingecko_endpoint):
        tools.extend(
            build_coingecko_mcp_tools(
                clock=clock,
                gate=gates.get(
                    provider="coingecko_mcp", max_concurrency=2, max_calls_per_minute=30
                ),
            )
        )

    if is_configured_secret(settings.finnhub_api_key):
        assert settings.finnhub_api_key is not None
        key = settings.finnhub_api_key.get_secret_value()
        gate = gates.get(provider="finnhub", max_concurrency=4, max_calls_per_minute=60)
        tools.extend(
            (
                FinnhubCompanyNewsTool(
                    client=client,
                    api_key=key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=gate,
                ),
                FinnhubEarningsSurprisesTool(
                    client=client,
                    api_key=key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=gate,
                ),
                FinnhubBasicFinancialsTool(
                    client=client,
                    api_key=key,
                    clock=clock,
                    base_url=settings.finnhub_base_url,
                    gate=gate,
                ),
            )
        )

    if settings.sec_user_agent and settings.sec_user_agent.strip():
        tools.append(
            SecEdgarRecentFilingsTool(
                client=client,
                clock=clock,
                user_agent=settings.sec_user_agent,
                base_url=settings.sec_edgar_base_url,
            )
        )
    return tools


@dataclass
class TurnRequest:
    question: str
    scope: Scope
    thread_key: str | None = None
    external_id: str | None = None
    scope_description: str = "a Slack conversation"
    conversation_kind: str = "channel"
    team_id: str | None = None
    channel_id: str | None = None
    on_step: StepCallback | None = None


class LeoRuntime:
    """A long-lived process's agent: shared HTTP client, engine, and tool set."""

    def __init__(
        self,
        *,
        settings: Settings,
        client: httpx.AsyncClient,
        engine: AsyncEngine,
        sessions: async_sessionmaker[AsyncSession],
    ) -> None:
        self._settings = settings
        self._client = client
        self._engine = engine
        self._sessions = sessions
        self._clock = SystemClock()
        self._gates = ProviderGateRegistry(self._clock)

        assert settings.openrouter_api_key is not None
        assert settings.leo_model is not None
        self._llm = LLM(
            client=client,
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.leo_model,
            base_url=settings.openrouter_base_url,
            max_output_tokens=settings.leo_max_output_tokens,
        )
        self._store = AgentStore(sessions)
        self._memory = MemoryService(sessions=sessions, llm=self._llm)
        self._provider_tools = build_tools(
            settings=settings, client=client, clock=self._clock, gates=self._gates
        )
        self._limits = LoopLimits(
            max_turns=settings.leo_max_model_turns,
            max_tool_calls=settings.leo_max_tool_calls,
            max_seconds=settings.leo_max_run_seconds,
        )
        logger.info(
            "runtime ready: model=%s tools=%d", settings.leo_model, len(self._provider_tools)
        )

    @property
    def store(self) -> AgentStore:
        return self._store

    @property
    def memory(self) -> MemoryService:
        return self._memory

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(tool.spec.name for tool in self._provider_tools))

    async def handle(self, request: TurnRequest) -> AgentResult:
        """Run one user request: load context, reason and act, persist everything."""

        scope = request.scope
        conversation_id = await self._store.ensure_conversation(
            scope,
            team_id=request.team_id,
            channel_id=request.channel_id,
            kind=request.conversation_kind,
            title=request.scope_description,
        )
        await self._store.record_message(
            scope,
            conversation_id,
            role="user",
            content=request.question,
            thread_key=request.thread_key,
            author_id=scope.actor_id,
            external_id=request.external_id,
        )
        history = await self._store.history(
            scope,
            thread_key=request.thread_key,
            exclude_external_id=request.external_id,
        )
        run_id = await self._store.start_run(
            scope,
            conversation_id,
            question=request.question,
            model=self._llm.model,
            thread_key=request.thread_key,
        )

        registry = ToolRegistry(list(self._provider_tools))
        for tool in build_memory_tools(self._memory, scope, run_id):
            registry.add(tool)
        discovery = ToolDiscovery(registry=registry, llm=self._llm, sessions=self._sessions)
        finder = ToolFinderTool(discovery)
        registry.add(finder)

        recalled = await self._memory.recall(scope, request.question)
        memories = "\n".join(f"- {item.render()} (id: {item.id})" for item in recalled)

        agent = Agent(
            llm=self._llm,
            registry=registry,
            discovery=discovery,
            finder=finder,
            limits=self._limits,
            on_step=request.on_step,
        )
        result = await agent.run(
            question=request.question,
            scope=scope,
            history=history,
            memories=memories,
            run_id=run_id,
            scope_description=request.scope_description,
            on_model_step=lambda seq, completion, ms: self._store.record_model_step(
                run_id,
                seq=seq,
                tool_names=[c.name for c in completion.tool_calls],
                content_preview=completion.content,
                finish_reason=completion.finish_reason,
                usage=completion.usage,
                duration_ms=ms,
            ),
            on_tool_step=lambda seq, tool_result: self._store.record_tool_step(
                run_id, seq=seq, result=tool_result
            ),
        )

        await self._store.finish_run(
            run_id,
            status=result.status,
            answer=result.answer or None,
            error=result.error,
            turns=result.turns,
            tool_calls=result.tool_calls,
            usage=result.usage,
        )
        if result.answered:
            await self._store.record_message(
                scope,
                conversation_id,
                role="assistant",
                content=result.answer,
                thread_key=request.thread_key,
                run_id=run_id,
            )
        return result


@asynccontextmanager
async def runtime(settings: Settings) -> AsyncIterator[LeoRuntime]:
    """Open every shared resource the agent needs, and close them all after."""

    missing = [
        name
        for name, value in (
            ("OPENROUTER_API_KEY", settings.openrouter_api_key),
            ("LEO_MODEL", settings.leo_model),
            ("DATABASE_URL", settings.database_url),
        )
        if not has_value(value)
    ]
    if missing:
        raise RuntimeError(f"missing required configuration: {', '.join(missing)}")
    assert settings.database_url is not None

    engine = create_engine(settings.database_url.get_secret_value())
    sessions = create_sessions(engine)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(180.0, connect=15.0),
        limits=httpx.Limits(max_connections=40, max_keepalive_connections=20),
    ) as client:
        try:
            yield LeoRuntime(settings=settings, client=client, engine=engine, sessions=sessions)
        finally:
            await engine.dispose()
