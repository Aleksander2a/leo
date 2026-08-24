"""The single quality gate, run locally and in CI.

Everything here either protects the repository from something irreversible
(leaked secrets, a destructive migration) or proves the package still builds and
imports from a clean install. Correctness lives in the test suite.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SECRET_PATTERNS = (
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{12,}"),
    re.compile(r"(?:sk|rk)-[A-Za-z0-9]{20,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]+ PRIVATE KEY-----"),
)

#: A migration that can destroy data must be written deliberately, not reached
#: by a helper that happens to call it.
_DESTRUCTIVE_PATTERNS = (
    ("alembic_downgrade_to_base", re.compile(r"\bdowngrade\s*\(\s*['\"]base['\"]")),
    ("metadata_drop_all", re.compile(r"\.\s*drop_all\s*\(")),
    ("drop_schema_or_database", re.compile(r"\bDROP\s+(?:SCHEMA|DATABASE)\b", re.IGNORECASE)),
)

_SKIP_DIRS = frozenset(
    {
        ".git",
        ".venv",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "node_modules",
        ".next",
        "dist",
        "artifacts",
        ".tmp",
    }
)

_TEXT_SUFFIXES = frozenset(
    {".py", ".md", ".toml", ".yml", ".yaml", ".json", ".ts", ".tsx", ".ini", ".ps1", ".sh"}
)


def _tracked_files() -> Iterator[Path]:
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        parts = set(path.relative_to(ROOT).parts)
        if parts & _SKIP_DIRS or any(p.startswith((".pytest-tmp", ".uv-cache")) for p in parts):
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        yield path


def scan_for_secrets() -> list[str]:
    findings: list[str] = []
    for path in _tracked_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: matches {pattern.pattern}")
    return findings


def scan_for_destructive_helpers() -> list[str]:
    findings: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in _DESTRUCTIVE_PATTERNS:
            if pattern.search(text):
                findings.append(f"{path.relative_to(ROOT)}: {name}")
    return findings


def check_layering() -> list[str]:
    """The agent must not depend on a transport, and providers must not depend on the agent loop.

    This is the boundary that lets the loop be reasoned about on its own, and
    the one whose absence let the previous runtime's routing leak everywhere.
    """

    findings: list[str] = []
    for path in (ROOT / "src" / "leo" / "agent").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("leo.slack", "leo.api", "slack_bolt", "fastapi"):
            if f"import {forbidden}" in text or f"from {forbidden}" in text:
                findings.append(f"{path.relative_to(ROOT)} imports {forbidden}")
    for path in (ROOT / "src" / "leo" / "providers").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("leo.agent.loop", "leo.agent.runtime", "leo.integrations"):
            if forbidden in text:
                findings.append(f"{path.relative_to(ROOT)} depends on {forbidden}")
    return findings


def _check(label: str, probe: Callable[[], list[str]]) -> None:
    findings = probe()
    if findings:
        print(f"[quality] FAIL {label}")
        for finding in findings:
            print(f"          {finding}")
        raise SystemExit(1)
    print(f"[quality] ok   {label}")


def _run(label: str, *command: str, env: dict[str, str] | None = None) -> None:
    print(f"[quality] run  {label}")
    result = subprocess.run(command, cwd=ROOT, env=env or os.environ.copy(), check=False)
    if result.returncode != 0:
        raise SystemExit(f"quality gate failed: {label}")


def _clean_install_smoke(wheel: Path) -> None:
    """Install the built wheel into a bare venv and import the runtime from it."""

    with tempfile.TemporaryDirectory(prefix="leo-wheel-") as temporary:
        venv = Path(temporary) / "venv"
        subprocess.run((sys.executable, "-m", "venv", str(venv)), check=True)
        python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run((str(python), "-m", "pip", "install", "--quiet", str(wheel)), check=True)
        subprocess.run(
            (
                str(python),
                "-c",
                "import leo.agent.runtime, leo.slack.app, leo.cli; print('wheel import ok')",
            ),
            check=True,
        )


def main() -> int:
    _check("secret scan", scan_for_secrets)
    _check("destructive migration helpers", scan_for_destructive_helpers)
    _check("layering", check_layering)

    python = sys.executable
    uv = shutil.which("uv")
    if uv is None:
        raise SystemExit("quality gate failed: uv is not on PATH")

    _run("ruff format", python, "-m", "ruff", "format", "--check", "src", "tests", "scripts")
    _run("ruff lint", python, "-m", "ruff", "check", "src", "tests", "scripts")
    _run("mypy", python, "-m", "mypy")
    _run("tests", python, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider")
    _run(
        "migration compiles",
        python,
        "-m",
        "alembic",
        "upgrade",
        "head",
        "--sql",
        env={**os.environ, "DATABASE_URL": "postgresql://user:pass@localhost/leo"},
    )

    with tempfile.TemporaryDirectory(prefix="leo-build-") as temporary:
        out = Path(temporary)
        _run("build", uv, "build", "--out-dir", str(out))
        wheels = sorted(out.glob("*.whl"))
        if len(wheels) != 1:
            raise SystemExit("quality gate failed: expected exactly one wheel")
        _clean_install_smoke(wheels[0])

    print("[quality] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
