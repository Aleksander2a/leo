from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

import leo.evals.provider_smoke_operator as smoke_operator
from leo.config import Settings
from leo.evals.provider_smoke_operator import (
    PROVIDER_ORDER,
    ProviderSmokeReport,
    ProviderSmokeStatus,
    collect_provider_smoke,
    export_provider_smoke,
)


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 22, 12, 0, tzinfo=UTC)


def _settings(**overrides: object) -> Settings:
    provider_values: dict[str, object] = {
        "finnhub_api_key": None,
        "tavily_api_key": None,
        "exa_api_key": None,
        "coingecko_api_key": None,
        "coin_market_cap_api_key": None,
        "alpha_vantage_api_key": None,
        "massive_api_key": None,
        "ticker_layer_api_key": None,
    }
    provider_values.update(overrides)
    return Settings(_env_file=None, **provider_values)


@pytest.mark.asyncio
async def test_missing_credentials_are_content_free_skips_without_transport() -> None:
    factory_calls: list[str] = []

    def transport_factory(provider: str) -> httpx.AsyncBaseTransport:
        factory_calls.append(provider)
        raise AssertionError("missing credentials must not construct a transport")

    report = await collect_provider_smoke(
        _settings(),
        clock=_Clock(),
        transport_factory=transport_factory,
    )

    assert factory_calls == []
    assert tuple(item.provider for item in report.cases) == PROVIDER_ORDER
    assert report.provider_count == 8
    assert report.configured_provider_count == 0
    assert report.skipped_count == 8
    assert all(item.status is ProviderSmokeStatus.SKIPPED for item in report.cases)
    assert all(item.safe_code == "CREDENTIAL_MISSING" for item in report.cases)
    assert all(item.network_attempt_count == 0 for item in report.cases)


@pytest.mark.asyncio
async def test_each_configured_real_adapter_dispatches_exactly_one_request_and_failures_continue(
    tmp_path: Path,
) -> None:
    secret = "provider-secret-must-never-appear"
    counts: Counter[str] = Counter()

    def transport_factory(provider: str) -> httpx.AsyncBaseTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            counts[provider] += 1
            return httpx.Response(
                401,
                json={"error": secret, "raw_content": f"{secret}:{request.method}"},
                request=request,
            )

        return httpx.MockTransport(handle)

    report = await collect_provider_smoke(
        _settings(
            finnhub_api_key=secret,
            tavily_api_key=secret,
            exa_api_key=secret,
            coingecko_api_key=secret,
            coin_market_cap_api_key=secret,
            alpha_vantage_api_key=secret,
            massive_api_key=secret,
            ticker_layer_api_key=secret,
        ),
        clock=_Clock(),
        transport_factory=transport_factory,
    )
    destination = tmp_path / "nested" / "provider-smoke-v1.json"
    export_provider_smoke(report, destination)

    assert counts == Counter({provider: 1 for provider in PROVIDER_ORDER})
    assert report.configured_provider_count == len(PROVIDER_ORDER)
    assert report.nonfatal_failure_count == len(PROVIDER_ORDER)
    assert report.success_count == 0
    assert all(item.network_attempt_count == 1 for item in report.cases)
    assert all(item.status is ProviderSmokeStatus.NONFATAL_FAILURE for item in report.cases)
    assert all(not item.request_bound_violation for item in report.cases)

    raw = destination.read_text(encoding="utf-8")
    assert secret not in raw
    assert "raw_content" not in raw
    assert "https://" not in raw
    assert "safe_message" not in raw
    assert raw.endswith("\n")
    assert ProviderSmokeReport.model_validate_json(raw) == report
    assert tuple(destination.parent.glob(f".{destination.name}.*.tmp")) == ()


@pytest.mark.asyncio
async def test_success_hashes_provider_content_without_exporting_it() -> None:
    credential = "tavily-credential-sentinel"
    provider_content = "raw-provider-content-sentinel"
    provider_url = "https://example.com/private-smoke-result"
    counts: Counter[str] = Counter()

    def transport_factory(provider: str) -> httpx.AsyncBaseTransport:
        def handle(request: httpx.Request) -> httpx.Response:
            counts[provider] += 1
            return httpx.Response(
                200,
                json={
                    "request_id": provider_content,
                    "results": [
                        {
                            "title": provider_content,
                            "url": provider_url,
                            "content": provider_content,
                            "score": 0.99,
                        }
                    ],
                },
                request=request,
            )

        return httpx.MockTransport(handle)

    report = await collect_provider_smoke(
        _settings(tavily_api_key=credential),
        clock=_Clock(),
        transport_factory=transport_factory,
    )

    tavily = report.cases[1]
    assert tavily.provider == "tavily"
    assert tavily.status is ProviderSmokeStatus.SUCCESS
    assert tavily.safe_code == "OK"
    assert tavily.network_attempt_count == 1
    assert counts == Counter({"tavily": 1})
    serialized = report.model_dump_json()
    assert credential not in serialized
    assert provider_content not in serialized
    assert provider_url not in serialized
    assert "https://" not in serialized


@pytest.mark.asyncio
async def test_provider_exception_text_is_discarded_and_does_not_abort_cohort() -> None:
    secret = "exception-secret-and-url-https://secret.invalid/path"
    counts: Counter[str] = Counter()

    def transport_factory(provider: str) -> httpx.AsyncBaseTransport:
        if provider == "exa":
            raise RuntimeError(secret)

        def handle(request: httpx.Request) -> httpx.Response:
            counts[provider] += 1
            return httpx.Response(429, json={"error": secret}, request=request)

        return httpx.MockTransport(handle)

    report = await collect_provider_smoke(
        _settings(exa_api_key=secret, tavily_api_key=secret),
        clock=_Clock(),
        transport_factory=transport_factory,
    )

    tavily = report.cases[1]
    exa = report.cases[2]
    assert tavily.safe_code == "TAVILY_RATE_LIMITED"
    assert tavily.retryable is True
    assert exa.safe_code == "PROVIDER_SMOKE_UNEXPECTED_FAILURE"
    assert exa.network_attempt_count == 0
    assert report.nonfatal_failure_count == 2
    assert report.skipped_count == 6
    assert secret not in report.model_dump_json()
    assert counts == Counter({"tavily": 1})


@pytest.mark.asyncio
async def test_transport_hard_stops_a_second_network_attempt() -> None:
    dispatched = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal dispatched
        dispatched += 1
        return httpx.Response(200, json={}, request=request)

    bounded = smoke_operator._SingleAttemptTransport(httpx.MockTransport(handle))
    async with httpx.AsyncClient(transport=bounded) as client:
        response = await client.get("https://example.com/first")
        assert response.status_code == 200
        with pytest.raises(httpx.TransportError, match="request bound exceeded"):
            await client.get("https://example.com/second")

    assert dispatched == 1
    assert bounded.network_attempt_count == 1
    assert bounded.request_bound_violation is True


def test_cli_redacts_terminal_exception_details(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    secret = "credential=https://secret.invalid/api?key=do-not-print"

    async def fail(_arguments: object) -> int:
        raise RuntimeError(secret)

    monkeypatch.setattr(smoke_operator, "_run", fail)
    result = smoke_operator.main(["--output", str(tmp_path / "unused.json")])

    output = capsys.readouterr().out
    assert result == 2
    assert output.strip() == "provider_smoke_collection_failed"
    assert secret not in output


def test_artifact_schema_has_no_raw_content_url_or_message_fields() -> None:
    schema = json.dumps(ProviderSmokeReport.model_json_schema(), sort_keys=True)

    assert '"raw"' not in schema
    assert '"url"' not in schema
    assert '"message"' not in schema
    assert '"credential"' not in schema
    assert '"request_url"' not in schema
