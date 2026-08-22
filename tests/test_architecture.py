from __future__ import annotations

import ast
from pathlib import Path


def test_harness_core_does_not_import_infrastructure_adapters() -> None:
    harness_root = Path("src/leo/harness")
    forbidden_prefixes = (
        "fastapi",
        "httpx",
        "slack_bolt",
        "slack_sdk",
        "sqlalchemy",
        "leo.integrations",
        "leo.persistence",
    )
    violations: list[str] = []
    for path in sorted(harness_root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(forbidden_prefixes):
                        violations.append(f"{path}:{node.lineno}:{alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                if module is not None and module.startswith(forbidden_prefixes):
                    violations.append(f"{path}:{node.lineno}:{module}")
    assert violations == []
