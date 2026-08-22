"""Production clock and identifier adapters."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


class UuidIdGenerator:
    def new(self, prefix: str) -> str:
        return f"{prefix}-{uuid4()}"
