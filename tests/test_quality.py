from __future__ import annotations

import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.quality import (
    _inspect_artifacts,
    _offline_environment,
    _run,
    architecture_violations,
    database_test_safety_violations,
    dependency_violations,
    scan_public_files,
)


def test_public_secret_scan_ignores_local_env_and_finds_no_real_tokens() -> None:
    assert scan_public_files(Path(".")) == ()


def test_public_secret_scan_reports_seeded_token_without_echoing_it(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    seeded_token = "xox" + "b-" + "12345678901234567890"
    source.write_text(f"value = {seeded_token!r}\n", encoding="utf-8")

    assert scan_public_files(tmp_path) == ("example.py:1",)


def test_public_secret_scan_detects_opaque_provider_key_assignment_without_echoing_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configuration.md"
    seeded_token = "opaque" + "0123456789abcdef0123456789"
    source.write_text(f"MASSIVE_API_KEY={seeded_token}\n", encoding="utf-8")

    assert scan_public_files(tmp_path) == ("configuration.md:1",)


def test_public_secret_scan_detects_credential_bearing_provider_endpoint_without_echoing_it(
    tmp_path: Path,
) -> None:
    source = tmp_path / "configuration.md"
    variable_name = "TAVILY" + "_ENDPOINT"
    scheme = "https" + "://"
    opaque_path = "opaque" + "0123456789abcdef0123456789"
    source.write_text(
        f"{variable_name}={scheme}mcp.vendor.invalid/{opaque_path}?transport=http\n",
        encoding="utf-8",
    )

    assert scan_public_files(tmp_path) == ("configuration.md:1",)


def test_public_secret_scan_allows_empty_and_explicit_provider_placeholders(
    tmp_path: Path,
) -> None:
    source = tmp_path / ".env.example"
    source.write_text(
        "TAVILY_API_KEY=\n"
        "EXA_API_KEY=YOUR_API_KEY_PLACEHOLDER\n"
        "COINGECKO_API_KEY=replace-with-your-key\n"
        "TAVILY_ENDPOINT=\n"
        "COINGECKO_ENDPOINT=https://example.invalid/replace-with-your-endpoint\n",
        encoding="utf-8",
    )

    assert scan_public_files(tmp_path) == ()


def test_architecture_scan_reports_forbidden_harness_import(tmp_path: Path) -> None:
    harness = tmp_path / "src" / "leo" / "harness"
    harness.mkdir(parents=True)
    (harness / "bad.py").write_text("from sqlalchemy import select\n", encoding="utf-8")

    assert architecture_violations(tmp_path) == ("src/leo/harness/bad.py:1:sqlalchemy",)


def test_database_test_safety_scan_rejects_destructive_lifecycle_helpers(
    tmp_path: Path,
) -> None:
    postgres_tests = tmp_path / "tests" / "postgres"
    postgres_tests.mkdir(parents=True)
    (postgres_tests / "conftest.py").write_text(
        "from alembic import command\n"
        "command.downgrade(config, 'base')\n"
        "metadata.drop_all(engine)\n"
        "sql = 'DROP SCHEMA public CASCADE'\n",
        encoding="utf-8",
    )

    assert database_test_safety_violations(tmp_path) == (
        "tests/postgres/conftest.py:2:alembic_downgrade",
        "tests/postgres/conftest.py:3:metadata_drop_all",
        "tests/postgres/conftest.py:4:drop_schema_or_database",
    )


def test_repository_postgres_tests_have_no_destructive_lifecycle_path() -> None:
    assert database_test_safety_violations(Path(".")) == ()


def test_public_scan_prunes_local_test_environments_but_scans_env_example(
    tmp_path: Path,
) -> None:
    seeded_token = "xox" + "b-" + "12345678901234567890"
    ignored = tmp_path / ".pytest-tmp-cleanroom" / "installed" / "secret.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text(f"token = {seeded_token!r}\n", encoding="utf-8")
    uv_cached = tmp_path / ".uv-cache-python-3.13" / "archive" / "dependency.py"
    uv_cached.parent.mkdir(parents=True)
    uv_cached.write_text(f"token = {seeded_token!r}\n", encoding="utf-8")
    (tmp_path / ".env").write_text(f"TOKEN={seeded_token}\n", encoding="utf-8")
    (tmp_path / ".env.example").write_text(f"TOKEN={seeded_token}\n", encoding="utf-8")

    assert scan_public_files(tmp_path) == (".env.example:1",)


def test_architecture_scan_reports_syntax_and_agent_framework_dependency(
    tmp_path: Path,
) -> None:
    harness = tmp_path / "src" / "leo" / "harness"
    harness.mkdir(parents=True)
    (harness / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "1"\ndependencies = []\n'
        '[project.optional-dependencies]\nagent = ["crewai>=1"]\n'
        '[dependency-groups]\ndev = ["langgraph>=1"]\n',
        encoding="utf-8",
    )

    assert architecture_violations(tmp_path) == ("src/leo/harness/broken.py:1:syntax_error",)
    assert dependency_violations(tmp_path) == (
        "pyproject.toml:forbidden_dependency:crewai",
        "pyproject.toml:forbidden_dependency:langgraph",
    )


def test_labelled_subprocess_gate_fails_fast_with_the_gate_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args=args, returncode=7),
    )

    with pytest.raises(SystemExit, match=r"quality gate failed: injected gate \(exit 7"):
        _run("injected gate", "does-not-run")


def test_artifact_inspection_requires_one_clean_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "leo_portfolio_agent-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("leo/__init__.py", "")
        archive.writestr("leo/evals/scenarios/control.json", "{}")
        archive.writestr("leo_portfolio_agent-0.1.0.dist-info/METADATA", "Name: leo")
    source = tmp_path / "leo_portfolio_agent-0.1.0.tar.gz"
    content = tmp_path / "README.md"
    content.write_text("demo", encoding="utf-8")
    with tarfile.open(source, "w:gz") as archive:
        archive.add(content, arcname="leo_portfolio_agent-0.1.0/README.md")
        archive.add(
            content,
            arcname="leo_portfolio_agent-0.1.0/evals/scenarios/control.json",
        )

    assert _inspect_artifacts(tmp_path) == wheel

    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    bad_wheel = forbidden / wheel.name
    with zipfile.ZipFile(bad_wheel, "w") as archive:
        archive.writestr("leo/__init__.py", "")
        archive.writestr("leo/evals/scenarios/control.json", "{}")
        archive.writestr(".env", "SECRET=value")
    bad_source = forbidden / source.name
    with tarfile.open(bad_source, "w:gz") as archive:
        archive.add(content, arcname="leo_portfolio_agent-0.1.0/README.md")
        archive.add(
            content,
            arcname="leo_portfolio_agent-0.1.0/evals/scenarios/control.json",
        )
    with pytest.raises(SystemExit, match="forbidden build content"):
        _inspect_artifacts(forbidden)


def test_artifact_inspection_returns_absolute_wheel_for_outside_repo_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    wheel = artifact_root / "leo_portfolio_agent-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("leo/__init__.py", "")
        archive.writestr("leo/evals/scenarios/control.json", "{}")
    source = artifact_root / "leo_portfolio_agent-0.1.0.tar.gz"
    content = tmp_path / "scenario.json"
    content.write_text("{}", encoding="utf-8")
    with tarfile.open(source, "w:gz") as archive:
        archive.add(
            content,
            arcname="leo_portfolio_agent-0.1.0/evals/scenarios/control.json",
        )
    monkeypatch.chdir(tmp_path)

    inspected = _inspect_artifacts(Path("artifacts"))

    assert inspected == wheel.resolve()


def test_clean_install_environment_excludes_credentials_and_import_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DATABASE_URL", "synthetic")
    monkeypatch.setenv("SLACK_BOT_TOKEN", "synthetic")
    provider_variables = (
        "ALPHA_VANTAGE_API_KEY",
        "COINGECKO_API_KEY",
        "COIN_MARKET_CAP_API_KEY",
        "EXA_API_KEY",
        "FINNHUB_API_KEY",
        "MASSIVE_API_KEY",
        "OPENROUTER_API_KEY",
        "TAVILY_API_KEY",
        "TICKER_LAYER_API_KEY",
    )
    provider_endpoint_variables = (
        "ALPHA_VANTAGE_ENDPOINT",
        "ALPHA_VANTAGE_ENDPOINT_LEGACY",
        "COINGECKO_ENDPOINT",
        "MASSIVE_ENDPOINT",
        "TAVILY_ENDPOINT",
    )
    for variable in provider_variables:
        monkeypatch.setenv(variable, "synthetic")
    for variable in provider_endpoint_variables:
        monkeypatch.setenv(variable, "https://mcp.example.invalid/opaque")
    monkeypatch.setenv("SLACK_USER_TOKEN", "synthetic")
    monkeypatch.setenv("PYTHONPATH", "synthetic")

    environment = _offline_environment()

    assert "DATABASE_URL" not in environment
    assert "SLACK_BOT_TOKEN" not in environment
    assert "SLACK_USER_TOKEN" not in environment
    assert all(variable not in environment for variable in provider_variables)
    assert all(variable not in environment for variable in provider_endpoint_variables)
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_ci_and_docs_use_the_single_cross_platform_quality_command() -> None:
    workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    readme = Path("README.md").read_text(encoding="utf-8")
    web_readme = Path("web/README.md").read_text(encoding="utf-8")

    # Leo supports exactly one Python version (matching the Dockerfile's runtime and
    # mypy's configured target), so CI runs a single quality job rather than a matrix.
    assert "uv python pin 3.12" in workflow
    assert "matrix" not in workflow
    assert "uv run python scripts/quality.py" in workflow
    assert "uv run python scripts/quality.py" in readme
    # M6's dashboard is real now, not the earlier placeholder; it must still declare the
    # same boundary the placeholder asserted: read-only, no privileged DB credential of its own.
    assert "no privileged database credential of its own" in web_readme
