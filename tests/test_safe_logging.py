from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import pytest

from leo.config import Settings
from leo.safe_logging import SensitiveDataRedactor, configure_safe_logging

_ALPHA_SECRET = "sentinel-alpha-credential"
_TAVILY_SECRET = "sentinel-tavily-credential"
_EXA_SECRET = "sentinel-exa-credential"
_PATH_SECRET = "sentinel-path-credential"
_ROUTER_SECRET = "sentinel-router-" + "credential"


@dataclass(frozen=True)
class _LoggerState:
    level: int
    handlers: tuple[logging.Handler, ...]
    propagate: bool
    disabled: bool


@pytest.fixture
def restore_logging_state() -> Iterator[None]:
    root = logging.getLogger()
    root_state = _LoggerState(
        level=root.level,
        handlers=tuple(root.handlers),
        propagate=root.propagate,
        disabled=root.disabled,
    )
    factory = logging.getLogRecordFactory()
    names = (
        "httpx",
        "httpcore",
        "httpcore.connection",
        "leo.test.safe_logging",
        "vendor.provider",
    )
    states = {
        name: _LoggerState(
            level=logging.getLogger(name).level,
            handlers=tuple(logging.getLogger(name).handlers),
            propagate=logging.getLogger(name).propagate,
            disabled=logging.getLogger(name).disabled,
        )
        for name in names
    }
    yield
    logging.setLogRecordFactory(factory)
    _restore_logger(root, root_state)
    for name, state in states.items():
        _restore_logger(logging.getLogger(name), state)


def test_redactor_removes_known_secrets_and_all_url_queries() -> None:
    redactor = SensitiveDataRedactor.from_values(
        (_ALPHA_SECRET, _TAVILY_SECRET, _EXA_SECRET, _PATH_SECRET)
    )
    unsafe = (
        "HTTP Request: GET "
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&apikey={_ALPHA_SECRET} "
        f"Authorization: Bearer {_TAVILY_SECRET} "
        f"x-api-key={_EXA_SECRET} "
        f"https://mcp.example.test/v1/{_PATH_SECRET}/search?token=unconfigured-secret"
    )

    rendered = redactor.redact(unsafe)

    for secret in (_ALPHA_SECRET, _TAVILY_SECRET, _EXA_SECRET, _PATH_SECRET):
        assert secret not in rendered
    assert "?" not in rendered
    assert "apikey=" not in rendered.casefold()
    assert "unconfigured-secret" not in rendered
    assert "Authorization: [REDACTED]" in rendered
    assert "x-api-key=[REDACTED]" in rendered
    assert "https://www.alphavantage.co/query" in rendered


def test_settings_collect_every_secret_field_without_nonsecret_metadata() -> None:
    settings = Settings(
        _env_file=None,
        openrouter_api_key=_ROUTER_SECRET,
        alpha_vantage_api_key=_ALPHA_SECRET,
        tavily_api_key=_TAVILY_SECRET,
        exa_api_key=_EXA_SECRET,
        database_url="postgresql://sentinel-user:sentinel-password@example.test/db",
        leo_slack_team_id="T_NOT_A_SECRET",
    )

    values = settings.sensitive_values_for_logging()

    assert _ROUTER_SECRET in values
    assert _ALPHA_SECRET in values
    assert _TAVILY_SECRET in values
    assert _EXA_SECRET in values
    assert "postgresql://sentinel-user:sentinel-password@example.test/db" in values
    assert "T_NOT_A_SECRET" not in values


@pytest.mark.usefixtures("restore_logging_state")
def test_debug_logging_suppresses_http_wire_urls_and_redacts_application_records() -> None:
    output = io.StringIO()
    configure_safe_logging(
        level_name="DEBUG",
        sensitive_values=(_ALPHA_SECRET, _TAVILY_SECRET, _EXA_SECRET, _PATH_SECRET),
        stream=output,
    )
    alpha_url = (
        f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey={_ALPHA_SECRET}"
    )
    logging.getLogger("httpx").info('HTTP Request: GET "%s" "HTTP/1.1 200 OK"', alpha_url)
    logging.getLogger("httpcore.connection").debug(
        "connect_tcp.started url=%s authorization=Bearer %s", alpha_url, _TAVILY_SECRET
    )

    request = httpx.Request("GET", alpha_url, headers={"x-api-key": _EXA_SECRET})
    try:
        raise httpx.ConnectError(f"provider connection failed for {request.url}", request=request)
    except httpx.ConnectError:
        logging.getLogger("leo.test.safe_logging").exception(
            "Provider call failed url=%s Authorization: Bearer %s",
            alpha_url,
            _TAVILY_SECRET,
            extra={
                "provider_context": {
                    "url": alpha_url,
                    "path": f"/v1/{_PATH_SECRET}/search",
                    "api_key": _EXA_SECRET,
                }
            },
        )

    rendered = output.getvalue()
    for secret in (_ALPHA_SECRET, _TAVILY_SECRET, _EXA_SECRET, _PATH_SECRET):
        assert secret not in rendered
    assert "HTTP Request:" not in rendered
    assert "connect_tcp.started" not in rendered
    assert "apikey=" not in rendered.casefold()
    assert "function=GLOBAL_QUOTE" not in rendered
    assert "Provider call failed" in rendered
    assert "https://www.alphavantage.co/query" in rendered


@pytest.mark.usefixtures("restore_logging_state")
def test_log_record_factory_sanitizes_before_library_owned_handler() -> None:
    output = io.StringIO()
    configure_safe_logging(
        level_name="DEBUG",
        sensitive_values=(_ALPHA_SECRET,),
        stream=output,
    )
    library_output = io.StringIO()
    library_handler = logging.StreamHandler(library_output)
    library_logger = logging.getLogger("vendor.provider")
    library_logger.handlers = [library_handler]
    library_logger.propagate = False
    library_logger.setLevel(logging.DEBUG)

    unsafe_url = f"https://provider.example.test/v1?q=IBM&apikey={_ALPHA_SECRET}"
    try:
        raise RuntimeError(f"request failed at {unsafe_url}")
    except RuntimeError:
        library_logger.exception("provider=%s", unsafe_url)

    rendered = library_output.getvalue()
    assert _ALPHA_SECRET not in rendered
    assert "apikey=" not in rendered.casefold()
    assert "q=IBM" not in rendered
    assert "https://provider.example.test/v1" in rendered


def _restore_logger(logger: logging.Logger, state: _LoggerState) -> None:
    logger.handlers = list(state.handlers)
    logger.setLevel(state.level)
    logger.propagate = state.propagate
    logger.disabled = state.disabled
