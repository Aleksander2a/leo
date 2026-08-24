"""Test doubles shared across the suite."""

from __future__ import annotations

from datetime import datetime, timedelta


class FixedClock:
    """A clock that only moves when a test tells it to."""

    def __init__(self, moment: datetime, step_seconds: float = 0.0) -> None:
        self._moment = moment
        self._step = timedelta(seconds=step_seconds)

    def now(self) -> datetime:
        current = self._moment
        self._moment = current + self._step
        return current


class SequentialIdGenerator:
    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def new(self, prefix: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        return f"{prefix}-{self._counts[prefix]}"
