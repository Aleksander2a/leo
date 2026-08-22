"""Progressive, versioned skill metadata and full-procedure loader."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError

from leo.harness.models import ContractModel, NonEmptyStr


class SkillSummary(ContractModel):
    schema_version: Literal["leo-skill-v1"]
    id: NonEmptyStr
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    summary: NonEmptyStr = Field(max_length=400)
    domains: frozenset[NonEmptyStr]
    required_evidence: tuple[NonEmptyStr, ...] = Field(min_length=1)
    stop_rules: tuple[NonEmptyStr, ...] = Field(min_length=1)
    allowed_capabilities: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    compatible_profiles: frozenset[NonEmptyStr] = Field(
        default_factory=lambda: frozenset({"research"})
    )
    child_compatible: bool
    procedure_trust: Literal["untrusted_instruction_data"]
    procedure_file: NonEmptyStr
    procedure_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class LoadedSkill(ContractModel):
    summary: SkillSummary
    procedure: NonEmptyStr = Field(max_length=32_768)
    content_hash: str = Field(min_length=64, max_length=64)


class SkillLoadError(RuntimeError):
    def __init__(self, safe_code: str) -> None:
        super().__init__(safe_code)
        self.safe_code = safe_code


class SkillCatalog:
    def __init__(
        self,
        root: Path,
        *,
        max_metadata_bytes: int = 16_384,
        max_procedure_bytes: int = 32_768,
    ) -> None:
        if not 256 <= max_metadata_bytes <= 65_536:
            raise ValueError("skill metadata byte limit is invalid")
        if not 256 <= max_procedure_bytes <= 131_072:
            raise ValueError("skill procedure byte limit is invalid")
        self._root = root.resolve()
        self._max_metadata_bytes = max_metadata_bytes
        self._max_procedure_bytes = max_procedure_bytes
        self._summaries: dict[str, SkillSummary] = {}
        self._directories: dict[str, Path] = {}

    def discover(self) -> tuple[SkillSummary, ...]:
        summaries: list[SkillSummary] = []
        discovered: dict[str, SkillSummary] = {}
        directories: dict[str, Path] = {}
        for metadata_path in sorted(self._root.glob("*/metadata.json")):
            try:
                raw = metadata_path.read_bytes()
                if len(raw) > self._max_metadata_bytes:
                    raise SkillLoadError("skill_metadata_too_large")
                payload = json.loads(raw.decode("utf-8"))
                if not isinstance(payload, dict):
                    raise SkillLoadError("skill_metadata_malformed")
                if payload.get("schema_version") != "leo-skill-v1":
                    raise SkillLoadError("skill_schema_version_unsupported")
                summary = SkillSummary.model_validate(payload)
            except SkillLoadError:
                raise
            except (OSError, UnicodeError, ValueError, ValidationError) as exc:
                raise SkillLoadError("skill_metadata_malformed") from exc
            if summary.id in discovered:
                raise SkillLoadError("skill_identity_conflict")
            discovered[summary.id] = summary
            directories[summary.id] = metadata_path.parent.resolve()
            summaries.append(summary)
        self._summaries = discovered
        self._directories = directories
        return tuple(sorted(summaries, key=lambda item: (item.id, item.version)))

    def load(self, skill_id: str) -> LoadedSkill:
        summary = self._summaries.get(skill_id)
        if summary is None:
            raise SkillLoadError("skill_not_discovered")
        skill_directory = self._directories[skill_id]
        procedure_path = (skill_directory / summary.procedure_file).resolve()
        if not procedure_path.is_relative_to(skill_directory):
            raise SkillLoadError("skill_procedure_path_denied")
        try:
            raw = procedure_path.read_bytes()
            if len(raw) > self._max_procedure_bytes:
                raise SkillLoadError("skill_procedure_too_large")
            procedure = raw.decode("utf-8")
        except SkillLoadError:
            raise
        except (OSError, UnicodeError, ValueError) as exc:
            raise SkillLoadError("skill_procedure_unavailable") from exc
        digest = hashlib.sha256(procedure.encode("utf-8")).hexdigest()
        if digest != summary.procedure_sha256:
            raise SkillLoadError("skill_hash_mismatch")
        return LoadedSkill(summary=summary, procedure=procedure, content_hash=digest)
