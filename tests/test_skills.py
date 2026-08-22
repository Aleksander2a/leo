from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from leo.capabilities.skills import SkillCatalog, SkillLoadError


def test_skill_catalog_discovers_summaries_then_loads_only_selected_procedure() -> None:
    catalog = SkillCatalog(Path("resources/leo-skills"))
    summaries = catalog.discover()
    assert {summary.id for summary in summaries} == {
        "delegated_research",
        "general_conversation",
        "narrow_quote",
        "thesis_challenge",
    }
    assert catalog.load("narrow_quote").summary.required_evidence
    assert all(summary.schema_version == "leo-skill-v1" for summary in summaries)
    assert catalog.load("delegated_research").summary.child_compatible
    assert not catalog.load("general_conversation").summary.child_compatible
    assert catalog.discover() == summaries
    with pytest.raises(SkillLoadError, match="not_discovered"):
        catalog.load("unknown")


def test_skill_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = root / "demo"
    skill.mkdir(parents=True)
    (skill / "metadata.json").write_text(
        '{"schema_version":"leo-skill-v1","id":"demo","version":"1.0.0",'
        '"summary":"demo","domains":["test"],'
        '"required_evidence":["evidence"],"stop_rules":["stop"],'
        '"child_compatible":false,"procedure_trust":"untrusted_instruction_data",'
        '"procedure_file":"procedure.md","procedure_sha256":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )
    (skill / "procedure.md").write_text("untrusted procedure", encoding="utf-8")
    catalog = SkillCatalog(root)
    catalog.discover()
    with pytest.raises(SkillLoadError, match="hash_mismatch"):
        catalog.load("demo")


def _write_skill(
    root: Path,
    directory: str,
    *,
    skill_id: str = "demo",
    schema_version: str = "leo-skill-v1",
    procedure_file: str = "procedure.md",
    procedure: str = "bounded procedure",
) -> None:
    skill = root / directory
    skill.mkdir(parents=True)
    digest = hashlib.sha256(procedure.encode()).hexdigest()
    payload = {
        "schema_version": schema_version,
        "id": skill_id,
        "version": "1.0.0",
        "summary": "Bounded demo skill.",
        "domains": ["test"],
        "required_evidence": ["evidence"],
        "stop_rules": ["stop"],
        "allowed_capabilities": [],
        "compatible_profiles": ["research"],
        "child_compatible": False,
        "procedure_trust": "untrusted_instruction_data",
        "procedure_file": procedure_file,
        "procedure_sha256": digest,
    }
    (skill / "metadata.json").write_text(json.dumps(payload), encoding="utf-8")
    if procedure_file == "procedure.md":
        (skill / procedure_file).write_text(procedure, encoding="utf-8")


def test_skill_unknown_schema_and_duplicate_identity_fail_closed(tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported"
    _write_skill(unsupported, "one", schema_version="leo-skill-v99")
    with pytest.raises(SkillLoadError, match="schema_version_unsupported"):
        SkillCatalog(unsupported).discover()

    duplicate = tmp_path / "duplicate"
    _write_skill(duplicate, "one", skill_id="same")
    _write_skill(duplicate, "two", skill_id="same")
    with pytest.raises(SkillLoadError, match="identity_conflict"):
        SkillCatalog(duplicate).discover()


def test_skill_procedure_is_confined_and_size_bounded(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    _write_skill(root, "escape", procedure_file="../outside.md")
    (root / "outside.md").write_text("bounded procedure", encoding="utf-8")
    catalog = SkillCatalog(root)
    catalog.discover()
    with pytest.raises(SkillLoadError, match="path_denied"):
        catalog.load("demo")

    bounded = tmp_path / "bounded"
    _write_skill(bounded, "large", procedure="x" * 300)
    catalog = SkillCatalog(bounded, max_procedure_bytes=256)
    catalog.discover()
    with pytest.raises(SkillLoadError, match="too_large"):
        catalog.load("demo")
