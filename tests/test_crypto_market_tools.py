from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from leo.capabilities.adapters import catalog_tool_from_spec
from leo.capabilities.catalog import InMemoryToolCatalog
from leo.capabilities.crypto_descriptors import CRYPTO_CAPABILITY_DESCRIPTORS
from leo.capabilities.discovery import DiscoveryBroker, DiscoveryQuery
from leo.config import Settings
from leo.harness.models import (
    CandidateClaim,
    ClaimKind,
    CompletionProposal,
    EvidenceQuality,
    EvidenceToolRequirement,
    OriginRef,
    Run,
    RunBundle,
    RunPhase,
    RunStatus,
    ScopeKey,
    Task,
    Thread,
    ToolArgumentConstraint,
    ToolExecutionContext,
    ToolFailure,
    ToolSuccess,
    TrustedScope,
    VerifierStatus,
)
from leo.harness.normalization import normalize_success
from leo.harness.subagents import canonical_evidence_completion
from leo.harness.verifier import DeterministicCompletionVerifier
from leo.integrations.crypto_composition import (
    build_crypto_market_tools,
    resolve_coingecko_rest_base_url,
)
from leo.integrations.crypto_market import (
    CoinGeckoMarketSnapshotTool,
    CoinMarketCapMarketSnapshotTool,
    CryptoMarketSnapshotTool,
)
from leo.integrations.fake import FixedClock, SequentialIdGenerator
from leo.integrations.provider_runtime import (
    ProviderCallGate,
    ProviderGateRegistry,
    ProviderGateRejected,
)
from leo.live import (
    _child_evidence_requirements,
    _conversation_capability_catalog,
    run_live_conversation,
)

NOW = datetime(2026, 8, 22, 12, 0, tzinfo=UTC)
AS_OF = NOW - timedelta(seconds=30)
SCOPE = ScopeKey(organization_id="org", strategy_id="strategy")


def _context() -> ToolExecutionContext:
    return ToolExecutionContext(
        trusted_scope=TrustedScope(namespace=SCOPE, actor_id="actor"),
        run_id="run",
        tool_call_id="call",
    )


def _coingecko_payload(*, price: float = 64_000.25) -> list[dict[str, object]]:
    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "current_price": price,
            "market_cap": 1_270_000_000_000.5,
            "total_volume": 31_000_000_000.25,
            "price_change_percentage_24h": 1.25,
            "last_updated": AS_OF.isoformat().replace("+00:00", "Z"),
        }
    ]


def _coinmarketcap_payload(*, price: float = 64_010.5) -> dict[str, object]:
    return {
        "data": [
            {
                "id": 1,
                "name": "Bitcoin",
                "symbol": "BTC",
                "slug": "bitcoin",
                "quotes": [
                    {
                        "symbol": "USD",
                        "price": price,
                        "market_cap": 1_271_000_000_000.5,
                        "volume_24h": 30_500_000_000.25,
                        "percent_change_24h": 1.20,
                        "last_updated": AS_OF.isoformat().replace("+00:00", "Z"),
                    }
                ],
            }
        ],
        "status": {
            "timestamp": NOW.isoformat().replace("+00:00", "Z"),
            "error_code": 0,
            "error_message": "",
            "elapsed": 10,
            "credit_count": 1,
        },
    }


def _bundle(observation_kind: str, outcome: ToolSuccess) -> tuple[RunBundle, str]:
    observation = normalize_success(
        outcome,
        observation_id="obs-crypto",
        scope=SCOPE,
        run_id="run",
        tool_call_id="call",
        observation_kind=observation_kind,
    )
    thread = Thread(
        id="thread",
        scope=SCOPE,
        origin=OriginRef(provider="fixture", external_thread_id="conversation"),
    )
    task = Task(id="task", thread_id=thread.id, scope=SCOPE, objective="Bitcoin price")
    run = Run(id="run", task_id=task.id, scope=SCOPE)
    return RunBundle(thread=thread, task=task, run=run, observations=(observation,)), observation.id


def _verify(observation_kind: str, outcome: ToolSuccess, statement: str) -> VerifierStatus:
    bundle, observation_id = _bundle(observation_kind, outcome)
    proposal = CompletionProposal(
        answer=statement,
        claims=(
            CandidateClaim(
                kind=ClaimKind.SOURCE_CLAIM,
                statement=statement,
                observation_ids=(observation_id,),
            ),
        ),
    )
    result = DeterministicCompletionVerifier(
        SequentialIdGenerator(),
        FixedClock(NOW),
    ).verify(proposal, bundle)
    return result.result.status


@pytest.mark.asyncio
async def test_coingecko_demo_snapshot_uses_header_and_normalized_fresh_payload() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "api.coingecko.com"
        assert request.url.path == "/api/v3/coins/markets"
        assert request.url.params["ids"] == "bitcoin"
        assert request.url.params["vs_currency"] == "usd"
        assert request.headers["x-cg-demo-api-key"] == "cg-key"
        assert "x-cg-pro-api-key" not in request.headers
        return httpx.Response(200, json=_coingecko_payload(), headers={"x-request-id": "cg-1"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
            base_url="https://api.coingecko.com/api/v3",
        ).execute({"asset_id": " BITCOIN ", "quote_currency": "usd"}, _context())

    assert isinstance(outcome, ToolSuccess)
    snapshot = outcome.data["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["provider"] == "coingecko"
    assert snapshot["asset_id"] == "bitcoin"
    assert snapshot["symbol"] == "BTC"
    assert snapshot["price"] == 64_000.25
    assert snapshot["as_of"] == AS_OF.isoformat().replace("+00:00", "Z")
    assert snapshot["provider_request_id"] == "cg-1"
    assert outcome.source.reference == snapshot["provider_reference"]
    assert outcome.observed_at == AS_OF
    assert outcome.expires_at == NOW + timedelta(minutes=3)
    normalized_bundle, _normalized_id = _bundle(
        "market.get_crypto_snapshot_coingecko",
        outcome,
    )
    assert normalized_bundle.observations[0].quality is EvidenceQuality.PROVIDER_REPORTED
    statement = outcome.data["statements"][0]  # type: ignore[index]
    assert isinstance(statement, str)
    assert (
        _verify("market.get_crypto_snapshot_coingecko", outcome, statement) is VerifierStatus.PASS
    )
    assert (
        _verify(
            "market.get_crypto_snapshot_coingecko",
            outcome,
            statement.replace("64000.2", "64000.3"),
        )
        is VerifierStatus.FAIL
    )


@pytest.mark.asyncio
async def test_coinmarketcap_snapshot_uses_header_and_accounts_for_provider_credit() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "pro-api.coinmarketcap.com"
        assert request.url.path == "/v2/simple/price"
        assert request.url.params["slug"] == "bitcoin"
        assert request.url.params["convert"] == "USD"
        assert request.url.params["include_all"] == "true"
        assert request.headers["X-CMC_PRO_API_KEY"] == "cmc-key"
        return httpx.Response(200, json=_coinmarketcap_payload())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            base_url="https://pro-api.coinmarketcap.com",
        ).execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(outcome, ToolSuccess)
    snapshot = outcome.data["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["provider"] == "coinmarketcap"
    assert snapshot["provider_asset_id"] == "1"
    assert snapshot["provider_credits_used"] == 1
    health = snapshot["health"]
    assert isinstance(health, dict)
    assert health["provider_credits_used"] == 1
    statement = outcome.data["statements"][0]  # type: ignore[index]
    assert isinstance(statement, str)
    assert (
        _verify("market.get_crypto_snapshot_coinmarketcap", outcome, statement)
        is VerifierStatus.PASS
    )


@pytest.mark.asyncio
async def test_coingecko_partial_alternate_nesting_keeps_exact_price_and_marks_missing() -> None:
    payload = {
        "data": [
            {
                "id": "bitcoin",
                "current_price": 64_000.25,
                "last_updated": AS_OF.isoformat().replace("+00:00", "Z"),
                "new_optional_provider_field": {"ignored": True},
            }
        ],
        "provider_metadata": {"shape_version": "future"},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    ) as client:
        outcome = await CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
        ).execute({"asset_id": "bitcoin", "quote_currency": "USD"}, _context())

    assert isinstance(outcome, ToolSuccess)
    snapshot = outcome.data["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["price"] == 64_000.25
    assert snapshot["name"] is None
    assert snapshot["symbol"] is None
    assert snapshot["missing_fields"] == [
        "market_cap",
        "name",
        "percent_change_24h",
        "symbol",
        "volume_24h",
    ]
    assert isinstance(snapshot["provider_payload_sha256"], str)
    statement = outcome.data["statements"][0]  # type: ignore[index]
    assert statement == (f"CoinGecko reports bitcoin at 64000.2 USD as of {AS_OF.isoformat()}.")
    assert (
        _verify("market.get_crypto_snapshot_coingecko", outcome, statement) is VerifierStatus.PASS
    )


@pytest.mark.asyncio
async def test_coinmarketcap_partial_dict_shape_uses_slug_identity_and_keyed_quote() -> None:
    payload = {
        "data": {
            "bitcoin": {
                "slug": "bitcoin",
                "quote": {
                    "USD": {
                        "price": 64_010.5,
                        "last_updated": AS_OF.isoformat().replace("+00:00", "Z"),
                    }
                },
                "future_field": "ignored",
            }
        },
        "status": {"error_code": 0},
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    ) as client:
        outcome = await CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
        ).execute({"asset_id": "bitcoin", "quote_currency": "USD"}, _context())

    assert isinstance(outcome, ToolSuccess)
    snapshot = outcome.data["snapshot"]
    assert isinstance(snapshot, dict)
    assert snapshot["provider_asset_id"] == "bitcoin"
    assert snapshot["missing_fields"] == [
        "market_cap",
        "name",
        "percent_change_24h",
        "provider_asset_id",
        "provider_credits_used",
        "symbol",
        "volume_24h",
    ]
    statement = outcome.data["statements"][0]  # type: ignore[index]
    assert statement == (f"CoinMarketCap reports bitcoin at 64010.5 USD as of {AS_OF.isoformat()}.")
    assert (
        _verify("market.get_crypto_snapshot_coinmarketcap", outcome, statement)
        is VerifierStatus.PASS
    )


@pytest.mark.asyncio
async def test_coinmarketcap_accounts_for_provider_credit_on_failed_http_response() -> None:
    gate = ProviderCallGate(provider="coinmarketcap", clock=FixedClock(NOW))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"Retry-After": "60"},
            json={
                "status": {
                    "timestamp": NOW.isoformat().replace("+00:00", "Z"),
                    "error_code": 1008,
                    "error_message": "rate limit",
                    "credit_count": 2,
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            gate=gate,
        ).execute({"asset_id": "bitcoin"}, _context())

    health = await gate.snapshot()
    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "COINMARKETCAP_RATE_LIMITED"
    assert health.provider_credits_used == 2
    assert health.failures == 1
    assert health.status == "rate_limited"


@pytest.mark.asyncio
async def test_oversized_coinmarketcap_429_activates_cooldown_without_json_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden_json(_response: httpx.Response, **_kwargs: object) -> object:
        raise AssertionError("an oversized provider response must not be JSON-decoded")

    monkeypatch.setattr(httpx.Response, "json", forbidden_json)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "999999"},
            content=b"x" * 1_048_577,
        )

    clock = FixedClock(NOW)
    gate = ProviderCallGate(provider="coinmarketcap", clock=clock)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=clock,
            gate=gate,
        )
        outcome = await tool.execute({"asset_id": "bitcoin"}, _context())
        cooldown = await tool.execute({"asset_id": "bitcoin"}, _context())

    health = await gate.snapshot()
    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "COINMARKETCAP_RATE_LIMITED"
    assert isinstance(cooldown, ToolFailure)
    assert cooldown.code == "COINMARKETCAP_COOLDOWN_ACTIVE"
    assert calls == 1
    assert health.calls_in_window == 1
    assert health.failures == 1
    assert health.rate_limit_count == 1
    assert health.provider_credits_used == 0
    assert health.cooldown_until == NOW + timedelta(seconds=300)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "base_url"),
    (
        ("coingecko", "https://api.coingecko.com.attacker.invalid/api/v3"),
        ("coingecko", "https://attacker.invalid/api/v3"),
        ("coingecko", "https://api.coingecko.com/not-api-v3"),
        ("coinmarketcap", "https://pro-api.coinmarketcap.com.attacker.invalid"),
        ("coinmarketcap", "https://attacker.invalid"),
        ("coinmarketcap", "https://pro-api.coinmarketcap.com/untrusted-base"),
    ),
)
async def test_crypto_adapters_reject_untrusted_credential_destinations(
    provider: str,
    base_url: str,
) -> None:
    network_calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal network_calls
        network_calls += 1
        return httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(ValueError, match="official REST API host and path"):
            if provider == "coingecko":
                CoinGeckoMarketSnapshotTool(
                    client=client,
                    api_key="must-not-egress",
                    clock=FixedClock(NOW),
                    base_url=base_url,
                )
            else:
                CoinMarketCapMarketSnapshotTool(
                    client=client,
                    api_key="must-not-egress",
                    clock=FixedClock(NOW),
                    base_url=base_url,
                )

    assert network_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "official_host", "expected_code"),
    (
        ("coingecko", "api.coingecko.com", "COINGECKO_SCHEMA_DRIFT"),
        (
            "coinmarketcap",
            "pro-api.coinmarketcap.com",
            "COINMARKETCAP_SCHEMA_DRIFT",
        ),
    ),
)
async def test_crypto_adapter_never_forwards_credential_across_redirect(
    provider: str,
    official_host: str,
    expected_code: str,
) -> None:
    requested_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_hosts.append(request.url.host)
        return httpx.Response(302, headers={"Location": "https://attacker.invalid/collect"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        follow_redirects=True,
    ) as client:
        if provider == "coingecko":
            tool = CoinGeckoMarketSnapshotTool(
                client=client,
                api_key="must-not-egress",
                clock=FixedClock(NOW),
            )
        else:
            tool = CoinMarketCapMarketSnapshotTool(
                client=client,
                api_key="must-not-egress",
                clock=FixedClock(NOW),
            )
        outcome = await tool.execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == expected_code
    assert requested_hosts == [official_host]


@pytest.mark.asyncio
async def test_crypto_adapters_fail_closed_on_stale_or_mismatched_provider_identity() -> None:
    stale_payload = _coingecko_payload()
    stale_payload[0]["last_updated"] = (NOW - timedelta(seconds=901)).isoformat()
    mismatched_cmc = _coinmarketcap_payload()
    data = mismatched_cmc["data"]
    assert isinstance(data, list) and isinstance(data[0], dict)
    data[0]["slug"] = "wrapped-bitcoin"

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=stale_payload))
    ) as client:
        stale = await CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
            base_url="https://api.coingecko.com/api/v3",
        ).execute({"asset_id": "bitcoin"}, _context())
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, json=mismatched_cmc))
    ) as client:
        mismatched = await CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            base_url="https://pro-api.coinmarketcap.com",
        ).execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(stale, ToolFailure) and stale.code == "COINGECKO_STALE_SNAPSHOT"
    assert isinstance(mismatched, ToolFailure)
    assert mismatched.code == "COINMARKETCAP_SCHEMA_DRIFT"


@pytest.mark.asyncio
async def test_crypto_adapters_reject_oversized_provider_responses_before_json_parsing() -> None:
    oversized_body = b"x" * 1_048_577
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=oversized_body))
    ) as client:
        coingecko = await CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
        ).execute({"asset_id": "bitcoin"}, _context())
        coinmarketcap = await CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
        ).execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(coingecko, ToolFailure)
    assert coingecko.code == "COINGECKO_RESPONSE_TOO_LARGE"
    assert isinstance(coinmarketcap, ToolFailure)
    assert coinmarketcap.code == "COINMARKETCAP_RESPONSE_TOO_LARGE"


@pytest.mark.asyncio
async def test_crypto_corroboration_records_exact_agreement_and_provenance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.coingecko.com":
            return httpx.Response(200, json=_coingecko_payload(price=64_000.0))
        assert request.url.host == "pro-api.coinmarketcap.com"
        return httpx.Response(200, json=_coinmarketcap_payload(price=64_100.0))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cg = CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
            base_url="https://api.coingecko.com/api/v3",
        )
        cmc = CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            base_url="https://pro-api.coinmarketcap.com",
        )
        outcome = await CryptoMarketSnapshotTool(
            (cmc, cg),
            agreement_threshold_bps=250,
        ).execute({"asset_id": "bitcoin", "quote_currency": "USD"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["providers_succeeded"] == ["coingecko", "coinmarketcap"]
    agreement = outcome.data["agreement"]
    assert isinstance(agreement, dict)
    assert agreement["status"] == "agreement"
    assert 15 < agreement["spread_bps"] < 16  # type: ignore[operator]
    digest = outcome.data["provenance_digest"]
    assert isinstance(digest, str) and len(digest) == 64
    assert outcome.source.reference == f"snapshot:bitcoin:USD:{digest}"
    statements = outcome.data["statements"]
    assert isinstance(statements, list) and len(statements) == 3
    summary = outcome.data["summary"]
    assert isinstance(summary, str)
    assert _verify("market.get_crypto_snapshot", outcome, summary) is VerifierStatus.PASS
    bundle, _observation_id = _bundle("market.get_crypto_snapshot", outcome)
    canonical = canonical_evidence_completion(
        bundle.observations,
        (
            EvidenceToolRequirement(
                observation_kind="market.get_crypto_snapshot",
                tool_name="market.get_crypto_snapshot",
                required_arguments=(
                    ToolArgumentConstraint(name="asset_id", value="bitcoin"),
                    ToolArgumentConstraint(name="quote_currency", value="USD"),
                ),
            ),
        ),
        now=NOW,
    )
    assert canonical is not None
    assert canonical.answer == summary
    assert all(
        _verify("market.get_crypto_snapshot", outcome, statement) is VerifierStatus.PASS
        for statement in statements
        if isinstance(statement, str)
    )

    forged = outcome.model_copy(
        update={
            "data": {
                **outcome.data,
                "provenance_digest": "0" * 64,
            }
        }
    )
    assert isinstance(statements[0], str)
    assert _verify("market.get_crypto_snapshot", forged, statements[0]) is VerifierStatus.FAIL

    coherently_renamed_data = deepcopy(outcome.data)
    snapshots = coherently_renamed_data["snapshots"]
    assert isinstance(snapshots, list) and isinstance(snapshots[0], dict)
    snapshots[0]["name"] = "Forged Bitcoin"
    renamed_statements = coherently_renamed_data["statements"]
    assert isinstance(renamed_statements, list) and isinstance(renamed_statements[0], str)
    renamed_statements[0] = renamed_statements[0].replace(
        "Bitcoin (BTC)",
        "Forged Bitcoin (BTC)",
    )
    renamed_summary = coherently_renamed_data["summary"]
    assert isinstance(renamed_summary, str)
    coherently_renamed_data["summary"] = renamed_summary.replace(
        "Bitcoin (BTC)",
        "Forged Bitcoin (BTC)",
        1,
    )
    coherently_renamed = outcome.model_copy(update={"data": coherently_renamed_data})
    assert (
        _verify(
            "market.get_crypto_snapshot",
            coherently_renamed,
            renamed_statements[0],
        )
        is VerifierStatus.FAIL
    )


@pytest.mark.asyncio
async def test_crypto_corroboration_rejects_time_skew_and_requires_exact_caveat() -> None:
    freshness_spread_seconds = 270

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.coingecko.com":
            return httpx.Response(200, json=_coingecko_payload(price=64_000.0))
        assert request.url.host == "pro-api.coinmarketcap.com"
        payload = _coinmarketcap_payload(price=64_010.0)
        data = payload["data"]
        assert isinstance(data, list) and isinstance(data[0], dict)
        quotes = data[0]["quotes"]
        assert isinstance(quotes, list) and isinstance(quotes[0], dict)
        quotes[0]["last_updated"] = (
            AS_OF - timedelta(seconds=freshness_spread_seconds)
        ).isoformat()
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        outcome = await CryptoMarketSnapshotTool(
            (
                CoinGeckoMarketSnapshotTool(
                    client=client,
                    api_key="cg-key",
                    clock=FixedClock(NOW),
                ),
                CoinMarketCapMarketSnapshotTool(
                    client=client,
                    api_key="cmc-key",
                    clock=FixedClock(NOW),
                ),
            ),
            agreement_threshold_bps=250,
            max_corroboration_skew_seconds=60,
        ).execute({"asset_id": "bitcoin", "quote_currency": "USD"}, _context())

    assert isinstance(outcome, ToolSuccess)
    agreement = outcome.data["agreement"]
    assert isinstance(agreement, dict)
    assert agreement["status"] == "time_skewed"
    assert agreement["providers_compared"] == 2
    assert agreement["agreement_threshold_bps"] == 250.0
    assert agreement["corroboration_skew_threshold_seconds"] == 60.0
    assert 1 < agreement["spread_bps"] < 2  # type: ignore[operator]
    assert agreement["freshness_spread_seconds"] == 270.0
    assert agreement["temporally_aligned"] is False
    assert agreement["corroborated"] is False
    assert agreement["lowest_price"] == 64_000.0
    assert agreement["highest_price"] == 64_010.0
    statements = outcome.data["statements"]
    assert isinstance(statements, list)
    caveat = (
        "CoinGecko and CoinMarketCap snapshots for bitcoin in USD were observed 270 seconds "
        "apart, above Leo's 60-second corroboration window; their prices are not treated as "
        "corroborating each other."
    )
    assert statements[-1] == caveat
    summary = outcome.data["summary"]
    assert isinstance(summary, str) and summary.endswith(caveat)
    assert _verify("market.get_crypto_snapshot", outcome, summary) is VerifierStatus.PASS
    assert _verify("market.get_crypto_snapshot", outcome, caveat) is VerifierStatus.PASS

    caveat_omitted_data = deepcopy(outcome.data)
    caveat_omitted_statements = caveat_omitted_data["statements"]
    assert isinstance(caveat_omitted_statements, list)
    caveat_omitted_data["statements"] = caveat_omitted_statements[:-1]
    caveat_omitted_data["summary"] = " ".join(caveat_omitted_statements[:-1])
    caveat_omitted = outcome.model_copy(update={"data": caveat_omitted_data})
    assert (
        _verify("market.get_crypto_snapshot", caveat_omitted, caveat_omitted_data["summary"])
        is VerifierStatus.FAIL
    )

    forged_alignment_data = deepcopy(outcome.data)
    forged_agreement = forged_alignment_data["agreement"]
    assert isinstance(forged_agreement, dict)
    forged_agreement["status"] = "agreement"
    forged_agreement["temporally_aligned"] = True
    forged_agreement["corroborated"] = True
    forged_alignment = outcome.model_copy(update={"data": forged_alignment_data})
    assert _verify("market.get_crypto_snapshot", forged_alignment, caveat) is VerifierStatus.FAIL


@pytest.mark.asyncio
async def test_crypto_corroboration_survives_rate_limited_provider_and_gate_cools_down() -> None:
    coingecko_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal coingecko_calls
        if request.url.host == "api.coingecko.com":
            coingecko_calls += 1
            return httpx.Response(429, headers={"Retry-After": "60"})
        return httpx.Response(200, json=_coinmarketcap_payload())

    cg_gate = ProviderCallGate(provider="coingecko", clock=FixedClock(NOW))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        cg = CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=FixedClock(NOW),
            base_url="https://api.coingecko.com/api/v3",
            gate=cg_gate,
        )
        cmc = CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            base_url="https://pro-api.coinmarketcap.com",
        )
        outcome = await CryptoMarketSnapshotTool((cg, cmc)).execute(
            {"asset_id": "bitcoin"},
            _context(),
        )
        cooldown = await cg.execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["providers_succeeded"] == ["coinmarketcap"]
    assert outcome.data["provider_failures"] == {"coingecko": "COINGECKO_RATE_LIMITED"}
    agreement = outcome.data["agreement"]
    assert isinstance(agreement, dict) and agreement["status"] == "single_provider"
    assert isinstance(cooldown, ToolFailure)
    assert cooldown.code == "COINGECKO_COOLDOWN_ACTIVE"
    assert cooldown.retryable
    assert coingecko_calls == 1


@pytest.mark.asyncio
async def test_crypto_corroboration_returns_safe_failure_only_when_every_provider_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503 if request.url.host == "api.coingecko.com" else 401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = CryptoMarketSnapshotTool(
            (
                CoinGeckoMarketSnapshotTool(
                    client=client,
                    api_key="cg-key",
                    clock=FixedClock(NOW),
                    base_url="https://api.coingecko.com/api/v3",
                ),
                CoinMarketCapMarketSnapshotTool(
                    client=client,
                    api_key="cmc-key",
                    clock=FixedClock(NOW),
                    base_url="https://pro-api.coinmarketcap.com",
                ),
            )
        )
        outcome = await tool.execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(outcome, ToolFailure)
    assert outcome.code == "CRYPTO_PROVIDERS_UNAVAILABLE"
    assert not outcome.retryable
    assert "key" not in outcome.safe_message.casefold()


@pytest.mark.asyncio
async def test_crypto_corroboration_contains_unexpected_provider_exception() -> None:
    class ExplodingCoinGecko:
        provider_name = "coingecko"

        async def execute(
            self,
            arguments: dict[str, object],
            context: ToolExecutionContext,
        ) -> ToolSuccess:
            del arguments, context
            raise RuntimeError("must remain content-free")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=_coinmarketcap_payload())
        )
    ) as client:
        cmc = CoinMarketCapMarketSnapshotTool(
            client=client,
            api_key="cmc-key",
            clock=FixedClock(NOW),
            base_url="https://pro-api.coinmarketcap.com",
        )
        outcome = await CryptoMarketSnapshotTool((ExplodingCoinGecko(), cmc)).execute(  # type: ignore[arg-type]
            {"asset_id": "bitcoin"},
            _context(),
        )

    assert isinstance(outcome, ToolSuccess)
    assert outcome.data["providers_succeeded"] == ["coinmarketcap"]
    assert outcome.data["provider_failures"] == {"coingecko": "COINGECKO_UNEXPECTED_FAILURE"}
    assert "must remain content-free" not in outcome.model_dump_json()


@pytest.mark.asyncio
async def test_shared_provider_gate_caps_calls_without_additional_network_request() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_coingecko_payload())

    clock = FixedClock(NOW)
    gate = ProviderCallGate(
        provider="coingecko",
        clock=clock,
        max_calls_per_minute=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = CoinGeckoMarketSnapshotTool(
            client=client,
            api_key="cg-key",
            clock=clock,
            base_url="https://api.coingecko.com/api/v3",
            gate=gate,
        )
        first = await tool.execute({"asset_id": "bitcoin"}, _context())
        limited = await tool.execute({"asset_id": "bitcoin"}, _context())
        clock.advance(seconds=61)
        after_reset = await tool.execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(first, ToolSuccess)
    assert isinstance(limited, ToolFailure)
    assert limited.code == "COINGECKO_LOCAL_RATE_LIMIT"
    assert isinstance(after_reset, ToolSuccess)
    assert calls == 2


def test_crypto_settings_are_optional_bounded_and_secret() -> None:
    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-never-print",
        coin_market_cap_api_key="cmc-never-print",
        coingecko_max_calls_per_minute=7,
        coin_market_cap_max_calls_per_minute=9,
    )

    assert settings.missing_for_conversation_providers() == ("OPENROUTER_API_KEY", "LEO_MODEL")
    assert settings.coingecko_max_calls_per_minute == 7
    assert settings.coin_market_cap_max_calls_per_minute == 9
    assert "cg-never-print" not in repr(settings)
    assert "cmc-never-print" not in repr(settings)


def test_crypto_composition_is_optional_extensible_and_keeps_mcp_endpoint_separate() -> None:
    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-key",
        coingecko_endpoint="https://mcp.api.coingecko.com/mcp",
        coingecko_base_url="https://api.coingecko.com/api/v3",
        coin_market_cap_api_key="cmc-key",
        coin_market_cap_base_url="https://pro-api.coinmarketcap.com",
        crypto_max_corroboration_skew_seconds=17,
    )

    async def build() -> tuple[tuple[str, ...], float]:
        async with httpx.AsyncClient() as client:
            tools = build_crypto_market_tools(
                settings=settings,
                client=client,
                clock=FixedClock(NOW),
            )
            aggregate = next(item for item in tools if isinstance(item, CryptoMarketSnapshotTool))
            return (
                tuple(item.spec.name for item in tools),
                aggregate.max_corroboration_skew_seconds,
            )

    import asyncio

    tool_names, skew_seconds = asyncio.run(build())
    assert tool_names == (
        "market.get_crypto_snapshot_coingecko",
        "market.get_crypto_snapshot_coinmarketcap",
        "market.get_crypto_snapshot",
    )
    assert skew_seconds == 17
    assert (
        resolve_coingecko_rest_base_url(
            configured_base="https://pro-api.coingecko.com/api/v3",
            configured_endpoint="https://api.coingecko.com/api/v3",
        )
        == "https://pro-api.coingecko.com/api/v3"
    )
    assert (
        resolve_coingecko_rest_base_url(
            configured_base="https://pro-api.coingecko.com/api/v3",
            configured_endpoint="https://mcp.api.coingecko.com/mcp",
        )
        == "https://pro-api.coingecko.com/api/v3"
    )
    with pytest.raises(ValueError, match="official /api/v3"):
        resolve_coingecko_rest_base_url(
            configured_base="https://attacker.invalid/api/v3",
            configured_endpoint="https://mcp.api.coingecko.com/mcp",
        )


def test_invalid_optional_crypto_provider_config_does_not_break_other_tools() -> None:
    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-key",
        coingecko_base_url="http://not-https.invalid/api/v3",
        coin_market_cap_api_key="cmc-key",
        coin_market_cap_base_url="https://pro-api.coinmarketcap.com",
    )

    async def build() -> tuple[str, ...]:
        async with httpx.AsyncClient() as client:
            return tuple(
                tool.spec.name
                for tool in build_crypto_market_tools(
                    settings=settings,
                    client=client,
                    clock=FixedClock(NOW),
                )
            )

    import asyncio

    assert asyncio.run(build()) == (
        "market.get_crypto_snapshot_coinmarketcap",
        "market.get_crypto_snapshot",
    )


@pytest.mark.asyncio
async def test_provider_gate_enforces_optional_utc_daily_budget_and_resets_next_day() -> None:
    clock = FixedClock(NOW)
    gate = ProviderCallGate(
        provider="alpha_vantage",
        clock=clock,
        max_calls_per_minute=20,
        max_calls_per_day=1,
    )
    async with gate.slot():
        await gate.record_failure("ALPHA_FAILURE", provider_credits_used=1)

    with pytest.raises(ProviderGateRejected) as blocked:
        async with gate.slot():
            pass
    assert blocked.value.code == "ALPHA_VANTAGE_LOCAL_DAILY_RATE_LIMIT"
    exhausted = await gate.snapshot()
    assert exhausted.calls_in_day == 1
    assert exhausted.remaining_local_daily_calls == 0
    assert exhausted.provider_credits_used == 1
    assert exhausted.status == "rate_limited"

    clock.advance(seconds=86_400)
    async with gate.slot():
        await gate.record_success()
    reset = await gate.snapshot()
    assert reset.calls_in_day == 1
    assert reset.remaining_local_daily_calls == 0


@pytest.mark.asyncio
async def test_provider_gate_enforces_optional_utc_monthly_budget_and_resets_next_month() -> None:
    clock = FixedClock(datetime(2026, 8, 31, 23, 59, tzinfo=UTC))
    gate = ProviderCallGate(
        provider="ticker_layer",
        clock=clock,
        max_calls_per_minute=20,
        max_calls_per_month=1,
    )
    async with gate.slot():
        await gate.record_failure("TICKER_LAYER_FAILURE", provider_credits_used=1)

    with pytest.raises(ProviderGateRejected) as blocked:
        async with gate.slot():
            pass
    assert blocked.value.code == "TICKER_LAYER_LOCAL_MONTHLY_RATE_LIMIT"
    exhausted = await gate.snapshot()
    assert exhausted.calls_in_month == 1
    assert exhausted.remaining_local_monthly_calls == 0
    assert exhausted.provider_credits_used == 1
    assert exhausted.status == "rate_limited"

    clock.advance(seconds=120)
    async with gate.slot():
        await gate.record_success()
    reset = await gate.snapshot()
    assert reset.calls_in_month == 1
    assert reset.remaining_local_monthly_calls == 0


def test_provider_gate_registry_reuses_health_and_rejects_policy_drift() -> None:
    registry = ProviderGateRegistry(FixedClock(NOW))
    first = registry.get(
        provider="coingecko",
        max_calls_per_minute=7,
    )
    second = registry.get(
        provider="CoinGecko",
        max_calls_per_minute=7,
    )

    assert first is second
    assert registry.registered_providers == ("coingecko",)
    with pytest.raises(ValueError, match="policy changed"):
        registry.get(provider="coingecko", max_calls_per_minute=8)


@pytest.mark.asyncio
async def test_crypto_composition_registry_preserves_quota_across_run_rebuilds() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=_coingecko_payload())

    clock = FixedClock(NOW)
    registry = ProviderGateRegistry(clock)
    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-key",
        coingecko_base_url="https://api.coingecko.com/api/v3",
        coingecko_max_calls_per_minute=1,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first_tools = build_crypto_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=registry,
        )
        first = await first_tools[0].execute({"asset_id": "bitcoin"}, _context())
        second_tools = build_crypto_market_tools(
            settings=settings,
            client=client,
            clock=clock,
            provider_gates=registry,
        )
        second = await second_tools[0].execute({"asset_id": "bitcoin"}, _context())

    assert isinstance(first, ToolSuccess)
    assert isinstance(second, ToolFailure)
    assert second.code == "COINGECKO_LOCAL_RATE_LIMIT"
    assert calls == 1


def test_crypto_capability_descriptors_drive_relevant_progressive_discovery() -> None:
    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-key",
        coin_market_cap_api_key="cmc-key",
    )

    async def build_catalog() -> InMemoryToolCatalog:
        catalog = InMemoryToolCatalog(version="crypto-test-v1")
        async with httpx.AsyncClient() as client:
            tools = build_crypto_market_tools(
                settings=settings,
                client=client,
                clock=FixedClock(NOW),
            )
            for tool in tools:
                descriptor = CRYPTO_CAPABILITY_DESCRIPTORS[tool.spec.name]
                catalog.register(
                    catalog_tool_from_spec(
                        tool.spec,
                        provider=descriptor.provider,
                        tags=descriptor.tags,
                        sensitivity=descriptor.sensitivity,
                        freshness_seconds=descriptor.freshness_seconds,
                        rate_limit_per_minute=descriptor.rate_limit_per_minute,
                        latency=descriptor.latency,
                        verification_expectations=descriptor.verification_expectations,
                    )
                )
        return catalog

    import asyncio

    catalog = asyncio.run(build_catalog())
    results = DiscoveryBroker(catalog).search(
        DiscoveryQuery(query="cross-check bitcoin crypto price"),
        phase=RunPhase.RESEARCH,
        profile="research",
        remaining_cost=10,
    )

    assert results
    assert results[0].id == "market.get_crypto_snapshot"
    assert set(CRYPTO_CAPABILITY_DESCRIPTORS) == {
        "market.get_crypto_snapshot",
        "market.get_crypto_snapshot_coingecko",
        "market.get_crypto_snapshot_coinmarketcap",
    }


def test_xrp_price_selects_the_crypto_aggregate_not_generic_equity() -> None:
    requirements = _child_evidence_requirements(
        "XRP price?",
        available_tool_names=frozenset({"market.get_crypto_snapshot", "market.get_quote"}),
    )

    assert len(requirements) == 1
    assert requirements[0].tool_name == "market.get_crypto_snapshot"
    assert tuple((item.name, item.value) for item in requirements[0].required_arguments) == (
        ("asset_id", "ripple"),
        ("quote_currency", "USD"),
    )


def test_live_crypto_routing_and_catalog_use_provider_owned_descriptors() -> None:
    requirements = _child_evidence_requirements(
        "What's BTC trading at in euros?",
        available_tool_names=frozenset({"market.get_crypto_snapshot"}),
    )

    assert len(requirements) == 1
    assert requirements[0].tool_name == "market.get_crypto_snapshot"
    assert requirements[0].required_arguments == (
        ToolArgumentConstraint(name="asset_id", value="bitcoin"),
        ToolArgumentConstraint(name="quote_currency", value="EUR"),
    )

    settings = Settings(
        _env_file=None,
        coingecko_api_key="cg-key",
        coin_market_cap_api_key="cmc-key",
    )

    async def build_catalog() -> InMemoryToolCatalog:
        async with httpx.AsyncClient() as client:
            tools = build_crypto_market_tools(
                settings=settings,
                client=client,
                clock=FixedClock(NOW),
            )
            return _conversation_capability_catalog(list(tools))

    import asyncio

    catalog = asyncio.run(build_catalog())
    aggregate = catalog.get("market.get_crypto_snapshot")
    coingecko = catalog.get("market.get_crypto_snapshot_coingecko")
    assert aggregate.provider == "crypto-corroboration"
    assert "provider_agreement_measurement" in aggregate.verification_expectations
    assert coingecko.provider == "coingecko"
    assert coingecko.rate_limit_per_minute == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("objective", ["Bitcoin price?", "What's BTC trading at now?"])
async def test_live_short_crypto_prompt_corroborates_then_answers_from_the_snapshot(
    objective: str,
) -> None:
    model_calls = 0
    provider_calls: list[str] = []
    tool_choices: list[object] = []
    required_argument_policies: list[object] = []
    observed_at = datetime.now(tz=UTC) - timedelta(seconds=5)

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal model_calls
        if request.url.host == "api.coingecko.com":
            provider_calls.append("coingecko")
            payload = _coingecko_payload(price=64_000.0)
            payload[0]["last_updated"] = observed_at.isoformat().replace("+00:00", "Z")
            return httpx.Response(200, json=payload)
        if request.url.host == "pro-api.coinmarketcap.com":
            provider_calls.append("coinmarketcap")
            payload = _coinmarketcap_payload(price=64_100.0)
            data = payload["data"]
            assert isinstance(data, list) and isinstance(data[0], dict)
            quotes = data[0]["quotes"]
            assert isinstance(quotes, list) and isinstance(quotes[0], dict)
            quotes[0]["last_updated"] = observed_at.isoformat().replace("+00:00", "Z")
            return httpx.Response(200, json=payload)

        assert request.url.host == "openrouter.test"
        model_calls += 1
        request_payload = json.loads(request.content)
        user_payload = json.loads(request_payload["messages"][1]["content"])
        observations = user_payload["observations"]
        if observations:
            # The model writes the answer from the corroborated snapshot. The
            # harness used to emit the snapshot's summary itself and never call
            # the model a second time.
            summary = observations[-1]["data"]["summary"]
            return httpx.Response(
                200,
                json={
                    "id": "crypto-answer",
                    "model": "fixture/model",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "answer": summary,
                                        "source_claims": [
                                            {
                                                "statement": summary,
                                                "observation_ids": [observations[-1]["id"]],
                                            }
                                        ],
                                        "inferences": [],
                                    }
                                ),
                                "tool_calls": [],
                            },
                        }
                    ],
                },
            )
        tool_choices.append(request_payload["tool_choice"])
        required_argument_policies.append(
            user_payload.get("tool_choice_policy", {}).get("required_arguments")
        )
        return httpx.Response(
            200,
            json={
                "id": "crypto-tool-call",
                "model": "fixture/model",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-crypto",
                                    "type": "function",
                                    "function": {
                                        "name": "market_get_crypto_snapshot",
                                        "arguments": json.dumps(
                                            {
                                                "asset_id": "bitcoin",
                                                "quote_currency": "USD",
                                            }
                                        ),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    settings = Settings(
        _env_file=None,
        openrouter_api_key="router-key",
        openrouter_base_url="https://openrouter.test/v1",
        leo_model="fixture/model",
        coingecko_api_key="cg-key",
        coin_market_cap_api_key="cmc-key",
        leo_max_model_turns=4,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await run_live_conversation(
            settings=settings,
            client=client,
            objective=objective,
        )

    assert result.run.status is RunStatus.COMPLETED
    # Two turns: fetch the corroborated snapshot, then answer from it.
    assert model_calls == 2
    assert tool_choices == [
        {
            "type": "function",
            "function": {"name": "market_get_crypto_snapshot"},
        }
    ], (provider_calls, result.run.final_output, required_argument_policies)
    assert required_argument_policies == [
        [
            {"name": "asset_id", "value": "bitcoin"},
            {"name": "quote_currency", "value": "USD"},
        ]
    ]
    assert sorted(provider_calls) == ["coingecko", "coinmarketcap"]
    assert len(result.observations) == 1
    observation = result.observations[0]
    assert observation.kind == "market.get_crypto_snapshot"
    assert result.run.final_output == observation.data["summary"]
    assert observation.data["providers_succeeded"] == ["coingecko", "coinmarketcap"]
