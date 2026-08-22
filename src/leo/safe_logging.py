"""Process-wide logging guards for credential-bearing live integrations.

Provider clients should return typed, bounded failures to the harness.  Their HTTP
transport diagnostics are neither user-facing evidence nor a safe operator surface:
some providers, notably Alpha Vantage, require an API key in the request query string.
This module therefore suppresses ``httpx``/``httpcore`` wire logs and redacts every
remaining application log at both record and formatter boundaries.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import TextIO
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

_REDACTION = "[REDACTED]"
_UNRENDERABLE = "[UNRENDERABLE LOG MESSAGE]"
_URL_PATTERN = re.compile(r"\b(?:https?|wss?)://[^\s<>\"']+", flags=re.IGNORECASE)
_CREDENTIAL_ASSIGNMENT_PATTERN = re.compile(
    r"(?ix)"
    r"\b("
    r"authorization|proxy-authorization|"
    r"api[-_]?key|apikey|access[-_]?key|access[-_]?token|"
    r"client[-_]?secret|password|passwd|token|"
    r"x-api-key|x-finnhub-token|x-cmc_pro_api_key|"
    r"x-cg-demo-api-key|x-cg-pro-api-key"
    r")\b"
    r"(\s*[:=]\s*)"
    r"(?:bearer\s+|basic\s+)?"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;}\]]+)",
)
_HTTP_LOGGER_ROOTS = ("httpx", "httpcore")
_SUPPRESSED_LOG_LEVEL = logging.CRITICAL + 1

LogRecordFactory = Callable[..., logging.LogRecord]


@dataclass(frozen=True, slots=True)
class SensitiveDataRedactor:
    """Remove configured secrets and credential-shaped URL/header material."""

    _needles: tuple[str, ...]

    @classmethod
    def from_values(cls, values: Iterable[str]) -> SensitiveDataRedactor:
        needles: set[str] = set()
        for value in values:
            if not value:
                continue
            needles.add(value)
            stripped = value.strip()
            if stripped:
                needles.add(stripped)
                needles.add(quote(stripped, safe=""))
                needles.add(quote_plus(stripped, safe=""))
        # Longer values must be removed first when one credential is a prefix of another.
        return cls(tuple(sorted(needles, key=lambda item: (-len(item), item))))

    def merged(self, values: Iterable[str]) -> SensitiveDataRedactor:
        return SensitiveDataRedactor.from_values((*self._needles, *values))

    def redact(self, value: object) -> str:
        """Return a bounded-safe textual representation of one log value."""

        text = str(value)
        for needle in self._needles:
            text = text.replace(needle, _REDACTION)
        text = _URL_PATTERN.sub(self._safe_url, text)
        return _CREDENTIAL_ASSIGNMENT_PATTERN.sub(
            lambda match: f"{match.group(1)}{match.group(2)}{_REDACTION}",
            text,
        )

    def _safe_url(self, match: re.Match[str]) -> str:
        raw_url = match.group(0)
        try:
            parsed = urlsplit(raw_url)
            hostname = parsed.hostname
            if hostname is None:
                return _REDACTION
            host = f"[{hostname}]" if ":" in hostname else hostname
            try:
                port = parsed.port
            except ValueError:
                port = None
            if port is not None:
                host = f"{host}:{port}"
            # Userinfo, query strings, and fragments never belong in operator logs.
            # Known secret values are additionally removed from credential-bearing paths.
            path = parsed.path
            for needle in self._needles:
                path = path.replace(needle, _REDACTION)
            return urlunsplit((parsed.scheme, host, path, "", ""))
        except (TypeError, ValueError):
            return _REDACTION


class _RedactingLogRecordFactory:
    """Sanitize messages before any library-owned handler can observe a record."""

    def __init__(self, delegate: LogRecordFactory, redactor: SensitiveDataRedactor) -> None:
        self._delegate = delegate
        self._redactor = redactor

    def add_sensitive_values(self, values: Iterable[str]) -> None:
        # Logging configuration can be called more than once in an embedded/test process.
        # Keep prior values protected instead of briefly making an old credential loggable.
        self._redactor = self._redactor.merged(values)

    @property
    def redactor(self) -> SensitiveDataRedactor:
        return self._redactor

    def __call__(self, *args: object, **kwargs: object) -> logging.LogRecord:
        record = self._delegate(*args, **kwargs)
        _redact_record(record, self._redactor)
        return record


class _RedactingFilter(logging.Filter):
    """Sanitize late-bound ``extra`` fields before the configured handler runs."""

    def __init__(self, redactor: SensitiveDataRedactor) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        _redact_record(record, self._redactor)
        for key, value in tuple(record.__dict__.items()):
            if key in {"exc_info", "args"}:
                continue
            record.__dict__[key] = _redact_structured_value(value, self._redactor)
        return True


class _RedactingFormatter(logging.Formatter):
    """Final fail-closed pass, including formatted traceback text."""

    def __init__(self, redactor: SensitiveDataRedactor) -> None:
        super().__init__("%(asctime)s %(levelname)s %(name)s %(message)s")
        self._redactor = redactor

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.redact(super().format(record))


def configure_safe_logging(
    *,
    level_name: str,
    sensitive_values: Iterable[str],
    stream: TextIO | None = None,
) -> None:
    """Configure Leo's process log surface without HTTP request metadata or secrets."""

    values = tuple(value for value in sensitive_values if value)
    redactor = _install_record_redaction(SensitiveDataRedactor.from_values(values), values)

    handler = logging.StreamHandler(stream)
    handler.addFilter(_RedactingFilter(redactor))
    handler.setFormatter(_RedactingFormatter(redactor))
    level = logging.getLevelNamesMapping().get(level_name.upper(), logging.INFO)
    logging.basicConfig(level=level, handlers=(handler,), force=True)
    _suppress_http_wire_logging()


def _install_record_redaction(
    redactor: SensitiveDataRedactor,
    sensitive_values: tuple[str, ...],
) -> SensitiveDataRedactor:
    current = logging.getLogRecordFactory()
    if isinstance(current, _RedactingLogRecordFactory):
        current.add_sensitive_values(sensitive_values)
        return current.redactor
    logging.setLogRecordFactory(_RedactingLogRecordFactory(current, redactor))
    return redactor


def _suppress_http_wire_logging() -> None:
    """Drop httpx/httpcore wire records at DEBUG, INFO, and higher levels.

    A provider may legitimately authenticate in a URL.  Redaction is defense in depth,
    but the safest transport diagnostic is one that never reaches a log handler.
    """

    manager = logging.root.manager
    known_names = tuple(
        name
        for name, candidate in manager.loggerDict.items()
        if isinstance(candidate, logging.Logger)
        and any(name == root or name.startswith(f"{root}.") for root in _HTTP_LOGGER_ROOTS)
    )
    for name in (*_HTTP_LOGGER_ROOTS, *known_names):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.disabled = False
        if name in _HTTP_LOGGER_ROOTS:
            logger.setLevel(_SUPPRESSED_LOG_LEVEL)
            logger.propagate = False
            logger.addHandler(logging.NullHandler())
        else:
            logger.setLevel(logging.NOTSET)
            logger.propagate = True


def _redact_record(record: logging.LogRecord, redactor: SensitiveDataRedactor) -> None:
    if getattr(record, "_leo_redacted", False):
        return
    try:
        rendered = record.getMessage()
    except Exception:
        rendered = _UNRENDERABLE
    record.msg = redactor.redact(rendered)
    record.args = ()
    if record.stack_info is not None:
        record.stack_info = redactor.redact(record.stack_info)
    if record.exc_info is not None:
        exc_type, exc, traceback = record.exc_info
        safe_type = getattr(exc_type, "__name__", "Exception")
        safe_detail = redactor.redact(exc)
        # Preserve the useful traceback frames while replacing the exception object;
        # chained provider exceptions cannot then reintroduce an unredacted request URL.
        safe_exception = Exception(f"{safe_type}: {safe_detail}")
        record.exc_info = (Exception, safe_exception, traceback)
        record.exc_text = None
    record._leo_redacted = True


def _redact_structured_value(
    value: object,
    redactor: SensitiveDataRedactor,
    *,
    depth: int = 0,
) -> object:
    if depth >= 4:
        return _REDACTION
    if isinstance(value, str):
        return redactor.redact(value)
    if isinstance(value, Mapping):
        return {
            redactor.redact(key): _redact_structured_value(item, redactor, depth=depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_redact_structured_value(item, redactor, depth=depth + 1) for item in value)
    if isinstance(value, list):
        return [_redact_structured_value(item, redactor, depth=depth + 1) for item in value]
    return value
