from __future__ import annotations

import pytest

from leo.persistence.database import normalize_database_url


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("postgres://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        ("postgresql://user:pass@host/db", "postgresql+psycopg://user:pass@host/db"),
        (
            "postgresql+psycopg://user:pass@host/db",
            "postgresql+psycopg://user:pass@host/db",
        ),
    ],
)
def test_normalize_database_url(source: str, expected: str) -> None:
    assert normalize_database_url(source) == expected


def test_reject_non_postgres_url() -> None:
    with pytest.raises(ValueError, match="PostgreSQL"):
        normalize_database_url("sqlite:///leo.db")
