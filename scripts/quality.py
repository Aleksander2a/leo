"""Single deterministic local/CI quality gate for the demo repository."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
import zipfile
from collections.abc import Callable, Iterator
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SECRET_PATTERNS = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{12,}"),
    re.compile(r"(?:sk|rk)-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
)
_PROVIDER_SECRET_VARIABLES = frozenset(
    {
        "ALPHA_VANTAGE_API_KEY",
        "COINGECKO_API_KEY",
        "COIN_MARKET_CAP_API_KEY",
        "EXA_API_KEY",
        "FINNHUB_API_KEY",
        "MASSIVE_API_KEY",
        "OPENROUTER_API_KEY",
        "TAVILY_API_KEY",
        "TICKER_LAYER_API_KEY",
    }
)
_PROVIDER_ENDPOINT_VARIABLES = frozenset(
    {
        "ALPHA_VANTAGE_ENDPOINT",
        "ALPHA_VANTAGE_ENDPOINT_LEGACY",
        "COINGECKO_ENDPOINT",
        "MASSIVE_ENDPOINT",
        "TAVILY_ENDPOINT",
    }
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    rf"[\"']?\b(?:{'|'.join(sorted(_PROVIDER_SECRET_VARIABLES))})\b[\"']?"
    r"\s*[:=]\s*[\"']?(?P<value>[A-Za-z0-9][A-Za-z0-9._~+/=-]{15,})",
    re.IGNORECASE,
)
_SECRET_ENDPOINT_ASSIGNMENT_PATTERN = re.compile(
    rf"[\"']?\b(?:{'|'.join(sorted(_PROVIDER_ENDPOINT_VARIABLES))})\b[\"']?"
    r"\s*[:=]\s*[\"']?(?P<value>https?://[^\s\"']+)",
)
_SECRET_PLACEHOLDER_MARKERS = frozenset(
    {
        "api_key",
        "example",
        "placeholder",
        "replace",
        "sample",
        "synthetic",
        "test",
        "your",
    }
)
_SKIP_PARTS = {
    ".git",
    ".leo-planning",
    ".venv",
    ".uv-cache",
    ".uv-python",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
    "__pycache__",
}
_SCAN_SUFFIXES = {".py", ".md", ".toml", ".ini", ".json", ".yml", ".yaml", ".ps1"}
_FORBIDDEN_AGENT_FRAMEWORKS = {
    "autogen",
    "crewai",
    "langchain",
    "langgraph",
    "openai-agents",
}
_OFFLINE_SECRET_VARIABLES = {
    *_PROVIDER_SECRET_VARIABLES,
    *_PROVIDER_ENDPOINT_VARIABLES,
    "DATABASE_URL",
    "SLACK_APP_TOKEN",
    "SLACK_BOT_TOKEN",
    "SLACK_USER_TOKEN",
}
_FORBIDDEN_DATABASE_TEST_PATTERNS = (
    ("alembic_downgrade", re.compile(r"\bcommand\s*\.\s*downgrade\s*\(")),
    ("alembic_cli_downgrade", re.compile(r"\balembic\b[^\r\n]*\bdowngrade\b", re.IGNORECASE)),
    ("migration_to_base", re.compile(r"\b(?:_migrate|upgrade|downgrade)\s*\(\s*['\"]base['\"]")),
    ("metadata_drop_all", re.compile(r"\.\s*drop_all\s*\(")),
    ("drop_schema_or_database", re.compile(r"\bDROP\s+(?:SCHEMA|DATABASE)\b", re.IGNORECASE)),
)


def _skip_directory(name: str) -> bool:
    return name in _SKIP_PARTS or name.startswith((".pytest-tmp", ".uv-cache"))


def _public_files(root: Path) -> Iterator[Path]:
    for directory, names, files in os.walk(root):
        names[:] = sorted(name for name in names if not _skip_directory(name))
        directory_path = Path(directory)
        for name in sorted(files):
            path = directory_path / name
            if (
                path.suffix.lower() in _SCAN_SUFFIXES or path.name == ".env.example"
            ) and path.name != ".env":
                yield path


def scan_public_files(root: Path = ROOT) -> tuple[str, ...]:
    """Return file/line findings for credentials, excluding ignored local state."""

    findings: list[str] = []
    for path in _public_files(root):
        relative = path.relative_to(root)
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if any(pattern.search(line) for pattern in _SECRET_PATTERNS) or (
                _contains_nonplaceholder_secret_assignment(line)
            ):
                findings.append(f"{relative}:{line_number}")
    return tuple(findings)


def _contains_nonplaceholder_secret_assignment(line: str) -> bool:
    for pattern in (_SECRET_ASSIGNMENT_PATTERN, _SECRET_ENDPOINT_ASSIGNMENT_PATTERN):
        match = pattern.search(line)
        if match is None:
            continue
        normalized = match.group("value").casefold()
        if not any(marker in normalized for marker in _SECRET_PLACEHOLDER_MARKERS):
            return True
    return False


def architecture_violations(root: Path = ROOT) -> tuple[str, ...]:
    """Return forbidden infrastructure imports from the harness core."""

    import ast

    harness_root = root / "src" / "leo" / "harness"
    forbidden_prefixes = (
        "fastapi",
        "httpx",
        "slack_bolt",
        "slack_sdk",
        "sqlalchemy",
        "leo.integrations",
        "leo.persistence",
    )
    findings: list[str] = []
    for path in sorted(harness_root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            relative = path.relative_to(root).as_posix()
            findings.append(f"{relative}:{exc.lineno or 0}:syntax_error")
            continue
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                modules = (node.module,)
            for module in modules:
                if module.startswith(forbidden_prefixes):
                    relative = path.relative_to(root).as_posix()
                    findings.append(f"{relative}:{getattr(node, 'lineno', 0)}:{module}")
    return tuple(findings)


def dependency_violations(root: Path = ROOT) -> tuple[str, ...]:
    """Reject dependencies that would own Leo's agent loop or authority."""

    pyproject = root / "pyproject.toml"
    if not pyproject.exists():
        return ("pyproject.toml:missing",)
    payload = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = payload.get("project")
    dependency_sets: list[object] = []
    if isinstance(project, dict):
        dependency_sets.append(project.get("dependencies", ()))
        optional = project.get("optional-dependencies", {})
        if isinstance(optional, dict):
            dependency_sets.extend(optional.values())
    groups = payload.get("dependency-groups", {})
    if isinstance(groups, dict):
        dependency_sets.extend(groups.values())
    findings: list[str] = []
    for dependencies in dependency_sets:
        if not isinstance(dependencies, list):
            findings.append("pyproject.toml:invalid_dependency_set")
            continue
        for raw in dependencies:
            if not isinstance(raw, str):
                findings.append("pyproject.toml:invalid_dependency")
                continue
            normalized = re.split(r"[<>=!~;\[ @]", raw, maxsplit=1)[0].strip().lower()
            if normalized in _FORBIDDEN_AGENT_FRAMEWORKS:
                findings.append(f"pyproject.toml:forbidden_dependency:{normalized}")
    return tuple(sorted(set(findings)))


def database_test_safety_violations(root: Path = ROOT) -> tuple[str, ...]:
    """Reject destructive lifecycle operations from shared-DB test code.

    Migration SQL is intentionally outside this scan. Contract tests may verify
    forward migrations on an explicitly disposable database, but no test helper
    may downgrade/drop the configured ``DATABASE_URL`` to manufacture a clean
    state. Current-head savepoints and targeted cleanup are the only accepted
    shared-database patterns.
    """

    postgres_tests = root / "tests" / "postgres"
    if not postgres_tests.exists():
        return ()
    findings: list[str] = []
    for path in sorted(postgres_tests.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        relative = path.relative_to(root).as_posix()
        for label, pattern in _FORBIDDEN_DATABASE_TEST_PATTERNS:
            match = pattern.search(source)
            if match is None:
                continue
            line = source.count("\n", 0, match.start()) + 1
            findings.append(f"{relative}:{line}:{label}")
    return tuple(findings)


def _check(label: str, operation: Callable[[], tuple[str, ...]]) -> None:
    print(f"[quality] {label}")
    started = time.perf_counter()
    findings = operation()
    elapsed = time.perf_counter() - started
    if findings:
        raise SystemExit(f"quality gate failed: {label}: {', '.join(findings)} ({elapsed:.2f}s)")
    print(f"[quality] {label}: PASS ({elapsed:.2f}s)")


def _run(
    label: str,
    *args: str,
    env: dict[str, str] | None = None,
    cwd: Path = ROOT,
    quiet: bool = False,
) -> None:
    print(f"[quality] {label}")
    started = time.perf_counter()
    completed = subprocess.run(
        args,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL if quiet else None,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        raise SystemExit(
            f"quality gate failed: {label} (exit {completed.returncode}, {elapsed:.2f}s)"
        )
    print(f"[quality] {label}: PASS ({elapsed:.2f}s)")


def _inspect_artifacts(artifact_root: Path) -> Path:
    artifact_root = artifact_root.resolve()
    artifacts = sorted(artifact_root.glob("*.whl")) + sorted(artifact_root.glob("*.tar.gz"))
    if not artifacts:
        raise SystemExit("quality gate failed: build produced no artifacts")
    for artifact in artifacts:
        names: tuple[str, ...]
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                names = tuple(archive.namelist())
        else:
            with tarfile.open(artifact) as archive:
                names = tuple(member.name for member in archive.getmembers())
        forbidden = tuple(
            name
            for name in names
            if ".env" in name
            or ".leo-planning" in name
            or "/tests/" in f"/{name}"
            or "leo_vision.md" in name
        )
        if forbidden:
            raise SystemExit(f"quality gate failed: forbidden build content in {artifact.name}")
        scenario_paths = tuple(
            name for name in names if "/evals/scenarios/" in f"/{name}" and name.endswith(".json")
        )
        if not scenario_paths:
            raise SystemExit(
                f"quality gate failed: deterministic scenarios missing from {artifact.name}"
            )
        digest = sha256(artifact.read_bytes()).hexdigest()
        print(f"[quality] artifact {artifact.name} sha256={digest}")
    wheels = tuple(artifact for artifact in artifacts if artifact.suffix == ".whl")
    if len(wheels) != 1:
        raise SystemExit("quality gate failed: build must produce exactly one wheel")
    source_distributions = tuple(
        artifact for artifact in artifacts if artifact.name.endswith(".tar.gz")
    )
    if len(source_distributions) != 1:
        raise SystemExit("quality gate failed: build must produce exactly one source distribution")
    return wheels[0]


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _offline_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in _OFFLINE_SECRET_VARIABLES | {"PYTHONHOME", "PYTHONPATH"}
    }
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _clean_install_smoke(wheel: Path) -> None:
    """Install all wheel dependencies into a fresh venv and smoke outside the repository."""

    with tempfile.TemporaryDirectory(prefix="leo clean install ") as temporary:
        temporary_path = Path(temporary).resolve()
        venv = temporary_path / "isolated environment"
        uv = shutil.which("uv")
        if uv is None:
            raise SystemExit("quality gate failed: uv executable is unavailable")
        offline_environment = _offline_environment()
        _run(
            "clean venv create",
            uv,
            "--cache-dir",
            str(ROOT / ".uv-cache"),
            "venv",
            str(venv),
            "--python",
            sys.executable,
            env=offline_environment,
            cwd=temporary_path,
        )
        clean_python = _venv_python(venv)
        _run(
            "clean wheel install",
            uv,
            "--cache-dir",
            str(ROOT / ".uv-cache"),
            "pip",
            "install",
            "--offline",
            "--python",
            str(clean_python),
            str(wheel),
            env=offline_environment,
            cwd=temporary_path,
        )
        probe = (
            "from pathlib import Path; import leo; "
            f"root=Path({str(venv)!r}).resolve(); "
            "loaded=Path(leo.__file__).resolve(); "
            "assert loaded.is_relative_to(root), (loaded, root); print(loaded)"
        )
        _run(
            "clean wheel import",
            str(clean_python),
            "-I",
            "-c",
            probe,
            env=offline_environment,
            cwd=temporary_path,
        )
        _run(
            "clean wheel smoke",
            str(clean_python),
            "-I",
            "-m",
            "leo",
            "smoke",
            env=offline_environment,
            cwd=temporary_path,
            quiet=True,
        )
        _run(
            "clean wheel eval",
            str(clean_python),
            "-I",
            "-m",
            "leo",
            "eval",
            env=offline_environment,
            cwd=temporary_path,
            quiet=True,
        )
        _run(
            "clean wheel fixture",
            str(clean_python),
            "-I",
            "-m",
            "leo",
            "run-fixture",
            "one-tool",
            "--format",
            "text",
            env=offline_environment,
            cwd=temporary_path,
            quiet=True,
        )


def main() -> int:
    _check("public secret scan", scan_public_files)
    _check("database test safety", database_test_safety_violations)
    _check(
        "architecture boundary",
        lambda: architecture_violations() + dependency_violations(),
    )

    python = sys.executable
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("quality gate failed: uv executable is unavailable")
    uv_version = subprocess.check_output((uv, "--version"), text=True).strip()
    print(f"[quality] python={sys.version.split()[0]} uv={uv_version}")
    _run("ruff format", python, "-m", "ruff", "format", "--check", ".")
    _run("ruff lint", python, "-m", "ruff", "check", ".")
    _run("mypy", python, "-m", "mypy", "src")
    _run(
        "offline tests",
        python,
        "-m",
        "pytest",
        "tests",
        "--ignore=tests/postgres",
        "-q",
    )
    _run("deterministic smoke", python, "-m", "leo", "smoke")
    _run(
        "migration compile",
        python,
        "-m",
        "alembic",
        "upgrade",
        "head",
        "--sql",
        env={**os.environ, "DATABASE_URL": "postgresql://user:pass@localhost/leo"},
    )
    with tempfile.TemporaryDirectory(prefix="leo build artifacts ") as temporary:
        artifact_root = Path(temporary).resolve()
        _run(
            "build",
            uv,
            "--cache-dir",
            str(ROOT / ".uv-cache"),
            "build",
            "--offline",
            "--out-dir",
            str(artifact_root),
        )
        wheel = _inspect_artifacts(artifact_root)
        _clean_install_smoke(wheel)
    print("[quality] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
