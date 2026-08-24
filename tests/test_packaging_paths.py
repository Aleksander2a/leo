"""Data that ships beside the code must be found in both layouts.

A source checkout keeps `resources/` and `migrations/` at the repository root,
two levels above `src/leo/`. An installed package puts the code in site-packages
and those directories elsewhere -- the container copies them next to the app.

Deriving a path from a module's own position encodes the checkout layout. The
deployed worker resolved its skill catalogue to
`/usr/local/lib/python3.12/resources/leo-skills`, and because `glob` over a
missing directory returns an empty list rather than raising, it ran with no
skills at all and reported nothing. The same expression in the schema guard did
raise, and took production down on startup.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from leo.live import _SKILL_ROOT
from leo.packaging import find_data_directory, require_data_directory


def test_the_skill_catalogue_resolves_to_a_real_directory() -> None:
    assert _SKILL_ROOT.is_dir()
    assert sorted(item.parent.name for item in _SKILL_ROOT.glob("*/metadata.json"))


def test_data_is_found_next_to_the_app_when_the_package_lives_elsewhere(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container's layout: code installed away from its data."""

    app = tmp_path / "app"
    (app / "resources" / "leo-skills" / "example").mkdir(parents=True)
    monkeypatch.chdir(app)

    # Anchor somewhere with no data above it, as site-packages would be.
    elsewhere = tmp_path / "site-packages" / "leo" / "live.py"
    elsewhere.parent.mkdir(parents=True)

    assert (
        find_data_directory("resources/leo-skills", anchor=elsewhere)
        == app / "resources" / "leo-skills"
    )


def test_a_missing_directory_is_reported_rather_than_silently_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silence is the failure mode that hid an empty skill catalogue in production."""

    monkeypatch.chdir(tmp_path)
    anchor = tmp_path / "site-packages" / "leo" / "live.py"
    anchor.parent.mkdir(parents=True)

    assert find_data_directory("resources/leo-skills", anchor=anchor) is None
    with caplog.at_level(logging.WARNING):
        require_data_directory("resources/leo-skills", anchor=anchor)
    assert any("resources/leo-skills" in record.getMessage() for record in caplog.records)
