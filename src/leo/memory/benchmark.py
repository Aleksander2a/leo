"""Frozen synthetic/public memory retrieval fixture and deterministic D-058 report."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from leo.harness.models import ContractModel, NonEmptyStr, ScopeKey
from leo.memory.models import MemoryRevision, MemoryStatus, MemoryVisibility
from leo.memory.retrieval import (
    AuthorizedMemoryNamespace,
    MemorySearchRequest,
    ScopedMemoryCandidate,
    search_memory,
)


class FixtureAccessState(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"
    NO_LEO = "no_leo"
    CROSS_WORKSPACE = "cross_workspace"
    GROUP_ISOLATED = "group_isolated"


class QueryMode(StrEnum):
    CHANNEL = "channel"
    DM = "dm"


class BenchmarkDocument(ContractModel):
    id: NonEmptyStr
    record_id: NonEmptyStr
    revision: int = Field(ge=1)
    current_revision: int = Field(ge=1)
    scope: ScopeKey
    visibility: MemoryVisibility
    namespace_id: NonEmptyStr
    source_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)
    content: str | None = None
    chunks: tuple[NonEmptyStr, ...] = ()
    sensitivity: float = Field(default=0.2, ge=0, le=1)
    valid_from: datetime
    valid_until: datetime | None = None
    recorded_at: datetime
    expires_at: datetime | None = None
    revision_status: MemoryStatus = MemoryStatus.ACTIVE
    record_status: MemoryStatus = MemoryStatus.ACTIVE
    reason: NonEmptyStr = "synthetic_public_fixture"
    access_state: FixtureAccessState

    @model_validator(mode="after")
    def content_is_unambiguous(self) -> BenchmarkDocument:
        if (self.content is None) == (not self.chunks):
            raise ValueError("benchmark document requires exactly one of content or chunks")
        if self.current_revision < self.revision:
            raise ValueError("current revision cannot precede the fixture revision")
        return self

    def searchable_content(self) -> str:
        return self.content if self.content is not None else "\n\n".join(self.chunks)

    def candidate(self) -> ScopedMemoryCandidate:
        revision = MemoryRevision.from_content(
            id=self.id,
            record_id=self.record_id,
            number=self.revision,
            content=self.searchable_content(),
            source_ids=self.source_ids,
            visibility=self.visibility,
            namespace_id=self.namespace_id,
            sensitivity=self.sensitivity,
            valid_from=self.valid_from,
            valid_until=self.valid_until,
            recorded_at=self.recorded_at,
            expires_at=self.expires_at,
            actor_id="fixture-actor",
            reason=self.reason,
            status=self.revision_status,
            supersedes_revision=(
                self.revision - 1
                if self.revision_status is MemoryStatus.SUPERSEDED and self.revision > 1
                else 1
                if self.revision_status is MemoryStatus.SUPERSEDED
                else None
            ),
        )
        return ScopedMemoryCandidate(
            scope=self.scope,
            revision=revision,
            record_status=self.record_status,
            current_revision=self.current_revision,
        )


class BenchmarkQuery(ContractModel):
    id: NonEmptyStr
    mode: QueryMode
    scope: ScopeKey
    query: NonEmptyStr
    authorized_namespaces: frozenset[AuthorizedMemoryNamespace] = Field(min_length=1)
    access_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    membership_hash: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
    as_of: datetime
    k: int = Field(default=5, ge=1, le=100)
    per_namespace_limit: int = Field(default=3, ge=1, le=50)
    relevance_grades: dict[NonEmptyStr, int] = Field(default_factory=dict)
    expected_revisions: dict[NonEmptyStr, int] = Field(default_factory=dict)
    forbidden_record_ids: frozenset[NonEmptyStr] = Field(default_factory=frozenset)
    expected_source_namespaces: frozenset[NonEmptyStr] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def labels_are_consistent(self) -> BenchmarkQuery:
        if any(grade < 1 or grade > 3 for grade in self.relevance_grades.values()):
            raise ValueError("relevance grades must be between one and three")
        if set(self.relevance_grades) & set(self.forbidden_record_ids):
            raise ValueError("a relevant record cannot also be forbidden")
        if not set(self.expected_revisions).issubset(self.relevance_grades):
            raise ValueError("expected revisions must reference relevant records")
        return self

    def request(self) -> MemorySearchRequest:
        return MemorySearchRequest(
            scope=self.scope,
            query=self.query,
            authorized_namespaces=self.authorized_namespaces,
            access_hash=self.access_hash,
            membership_hash=self.membership_hash,
            as_of=self.as_of,
            limit=self.k,
            per_namespace_limit=self.per_namespace_limit,
        )


class BenchmarkManifest(ContractModel):
    version: NonEmptyStr
    retrieval_policy_version: NonEmptyStr
    synthetic_public_only: bool
    fixed_clock: datetime
    recall_at_k_threshold: float = Field(ge=0, le=1)
    coverage_threshold: float = Field(ge=0, le=1)
    corpus_file: NonEmptyStr
    queries_file: NonEmptyStr
    fixture_digest: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")


class FrozenRetrievalFixture(ContractModel):
    manifest: BenchmarkManifest
    documents: tuple[BenchmarkDocument, ...]
    queries: tuple[BenchmarkQuery, ...]


class RetrievalMetrics(ContractModel):
    recall_at_k: float | None = Field(default=None, ge=0, le=1)
    full_query_coverage: float | None = Field(default=None, ge=0, le=1)
    ndcg_at_k: float | None = Field(default=None, ge=0, le=1)
    expected_dm_source_coverage: float | None = Field(default=None, ge=0, le=1)
    current_revision_recall: float | None = Field(default=None, ge=0, le=1)
    conflict_recall: float | None = Field(default=None, ge=0, le=1)
    leakage_count: int = Field(ge=0)
    relevant_selected: int = Field(ge=0)
    relevant_total: int = Field(ge=0)
    query_count: int = Field(ge=1)
    selected_count: int = Field(ge=0)
    selected_content_bytes: int = Field(ge=0)
    estimated_context_tokens: int = Field(ge=0)
    context_cost_status: NonEmptyStr
    latency_ms: float | None = None
    latency_status: NonEmptyStr


class VariantStatus(StrEnum):
    COMPLETED = "completed"
    NOT_REQUIRED = "not_required"
    REQUIRED = "required"


class RetrievalVariantOutcome(ContractModel):
    variant: NonEmptyStr
    status: VariantStatus
    reason: NonEmptyStr
    metrics: RetrievalMetrics | None = None


class RetrievalBenchmarkReport(ContractModel):
    version: NonEmptyStr = "memory-retrieval-report-v1"
    fixture_version: NonEmptyStr
    fixture_digest: str = Field(min_length=64, max_length=64)
    retrieval_policy_version: NonEmptyStr
    fixed_clock: datetime
    outcomes: tuple[RetrievalVariantOutcome, ...]
    selected_default: NonEmptyStr
    fallback: NonEmptyStr
    report_digest: str = Field(min_length=64, max_length=64)


def load_frozen_retrieval_fixture(directory: Path) -> FrozenRetrievalFixture:
    manifest = BenchmarkManifest.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    documents = tuple(
        BenchmarkDocument.model_validate(item)
        for item in json.loads((directory / manifest.corpus_file).read_text(encoding="utf-8"))
    )
    queries = tuple(
        BenchmarkQuery.model_validate(item)
        for item in json.loads((directory / manifest.queries_file).read_text(encoding="utf-8"))
    )
    fixture = FrozenRetrievalFixture(
        manifest=manifest,
        documents=documents,
        queries=queries,
    )
    validate_frozen_retrieval_fixture(fixture)
    return fixture


def validate_committed_retrieval_report(directory: Path) -> RetrievalBenchmarkReport:
    """Re-run the frozen baseline and reject any stale or hand-edited report."""

    generated = run_retrieval_benchmark(load_frozen_retrieval_fixture(directory))
    committed = RetrievalBenchmarkReport.model_validate_json(
        (directory / "report.json").read_text(encoding="utf-8")
    )
    if committed != generated:
        raise ValueError("committed retrieval benchmark report is stale")
    return committed


def validate_frozen_retrieval_fixture(fixture: FrozenRetrievalFixture) -> None:
    manifest = fixture.manifest
    if not manifest.synthetic_public_only:
        raise ValueError("retrieval fixture must contain only synthetic/public content")
    document_ids = tuple(item.id for item in fixture.documents)
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("retrieval fixture document IDs must be unique")
    query_ids = tuple(item.id for item in fixture.queries)
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("retrieval fixture query IDs must be unique")
    access_states = {item.access_state for item in fixture.documents}
    required_negative_states = {
        FixtureAccessState.REVOKED,
        FixtureAccessState.NO_LEO,
        FixtureAccessState.CROSS_WORKSPACE,
        FixtureAccessState.GROUP_ISOLATED,
    }
    if not required_negative_states.issubset(access_states):
        raise ValueError("retrieval fixture is missing a required access negative")
    if {item.mode for item in fixture.queries} != {QueryMode.CHANNEL, QueryMode.DM}:
        raise ValueError("retrieval fixture must cover channel and DM modes")
    if not any(item.chunks for item in fixture.documents):
        raise ValueError("retrieval fixture must contain a bounded long-card example")
    if not any(item.revision_status is MemoryStatus.SUPERSEDED for item in fixture.documents):
        raise ValueError("retrieval fixture must contain a corrected prior revision")
    if sum(item.record_status is MemoryStatus.CONTESTED for item in fixture.documents) < 2:
        raise ValueError("retrieval fixture must contain both sides of a conflict")
    record_ids = {item.record_id for item in fixture.documents}
    for query in fixture.queries:
        labelled = set(query.relevance_grades) | set(query.forbidden_record_ids)
        if not labelled.issubset(record_ids):
            raise ValueError("retrieval fixture labels reference an unknown record")
        if query.as_of != manifest.fixed_clock:
            raise ValueError("retrieval fixture queries must use the fixed clock")
    expected = retrieval_fixture_digest(fixture.documents, fixture.queries)
    if manifest.fixture_digest != expected:
        raise ValueError("retrieval fixture digest mismatch")


def retrieval_fixture_digest(
    documents: tuple[BenchmarkDocument, ...], queries: tuple[BenchmarkQuery, ...]
) -> str:
    payload = {
        "documents": [item.model_dump(mode="json") for item in documents],
        "queries": [_stable_query_payload(item) for item in queries],
    }
    return hashlib.sha256(_canonical(payload).encode()).hexdigest()


def _stable_query_payload(query: BenchmarkQuery) -> dict[str, object]:
    payload: dict[str, object] = query.model_dump(mode="json")
    payload["authorized_namespaces"] = [
        item.model_dump(mode="json")
        for item in sorted(
            query.authorized_namespaces,
            key=lambda item: (item.visibility.value, item.namespace_id),
        )
    ]
    payload["forbidden_record_ids"] = sorted(query.forbidden_record_ids)
    payload["expected_source_namespaces"] = sorted(query.expected_source_namespaces)
    return payload


def run_retrieval_benchmark(fixture: FrozenRetrievalFixture) -> RetrievalBenchmarkReport:
    validate_frozen_retrieval_fixture(fixture)
    candidates = tuple(document.candidate() for document in fixture.documents)
    document_by_record_revision = {
        (item.record_id, item.revision): item for item in fixture.documents
    }
    total_relevant = 0
    selected_relevant = 0
    queries_with_relevant = 0
    fully_covered_queries = 0
    ndcg_values: list[float] = []
    dm_sources_expected = 0
    dm_sources_selected = 0
    expected_current = 0
    selected_current = 0
    expected_conflicts = 0
    selected_conflicts = 0
    leakage_count = 0
    selected_count = 0
    selected_bytes = 0
    for query in fixture.queries:
        hits = search_memory(candidates, query.request())
        hit_ids = tuple(item.record_id for item in hits)
        if len(hit_ids) != len(set(hit_ids)):
            raise ValueError("retrieval benchmark returned duplicate record IDs")
        relevant = set(query.relevance_grades)
        selected = set(hit_ids)
        total_relevant += len(relevant)
        selected_relevant += len(selected & relevant)
        selected_count += len(hits)
        selected_bytes += sum(len(item.content.encode("utf-8")) for item in hits)
        leakage_count += len(selected & set(query.forbidden_record_ids))
        if relevant:
            queries_with_relevant += 1
            if relevant.issubset(selected):
                fully_covered_queries += 1
            ndcg_values.append(_ndcg(hit_ids, query.relevance_grades, query.k))
        expected_sources = set(query.expected_source_namespaces)
        if expected_sources:
            dm_sources_expected += len(expected_sources)
            dm_sources_selected += len(expected_sources & {item.namespace_id for item in hits})
        for record_id, revision in query.expected_revisions.items():
            expected_current += 1
            if any(hit.record_id == record_id and hit.revision == revision for hit in hits):
                selected_current += 1
        conflict_records = {
            record_id
            for record_id in relevant
            if any(
                document.record_id == record_id and document.record_status is MemoryStatus.CONTESTED
                for document in fixture.documents
            )
        }
        expected_conflicts += len(conflict_records)
        selected_conflicts += len(conflict_records & selected)
        for hit in hits:
            if (hit.record_id, hit.revision) not in document_by_record_revision:
                raise ValueError("retrieval benchmark returned an unknown revision")
    metrics = RetrievalMetrics(
        recall_at_k=_ratio(selected_relevant, total_relevant),
        full_query_coverage=_ratio(fully_covered_queries, queries_with_relevant),
        ndcg_at_k=(sum(ndcg_values) / len(ndcg_values) if ndcg_values else None),
        expected_dm_source_coverage=_ratio(dm_sources_selected, dm_sources_expected),
        current_revision_recall=_ratio(selected_current, expected_current),
        conflict_recall=_ratio(selected_conflicts, expected_conflicts),
        leakage_count=leakage_count,
        relevant_selected=selected_relevant,
        relevant_total=total_relevant,
        query_count=len(fixture.queries),
        selected_count=selected_count,
        selected_content_bytes=selected_bytes,
        estimated_context_tokens=(selected_bytes + 3) // 4,
        context_cost_status="content_only_utf8_bytes_div4_v1",
        latency_ms=None,
        latency_status="not_measured_deterministic_offline",
    )
    threshold_passed = (
        metrics.leakage_count == 0
        and metrics.recall_at_k is not None
        and metrics.recall_at_k >= fixture.manifest.recall_at_k_threshold
        and metrics.full_query_coverage is not None
        and metrics.full_query_coverage >= fixture.manifest.coverage_threshold
        and metrics.expected_dm_source_coverage == 1
        and metrics.current_revision_recall == 1
        and metrics.conflict_recall == 1
    )
    vector_status = VariantStatus.NOT_REQUIRED if threshold_passed else VariantStatus.REQUIRED
    vector_reason = (
        "D-058 deterministic FTS threshold passed; no vector metrics were generated."
        if threshold_passed
        else "D-058 deterministic FTS threshold missed; vector work is required before closure."
    )
    progressive_metrics = metrics.model_copy(
        update={
            "context_cost_status": "progressive_full-content-upper-bound-v1",
        }
    )
    compaction_metrics = metrics.model_copy(
        update={
            "context_cost_status": "compaction-selection-parity-v1",
        }
    )
    outcomes = (
        RetrievalVariantOutcome(
            variant="deterministic_fts",
            status=VariantStatus.COMPLETED,
            reason="Frozen scope-first lexical baseline executed.",
            metrics=metrics,
        ),
        RetrievalVariantOutcome(
            variant="progressive_navigation",
            status=VariantStatus.COMPLETED,
            reason=(
                "The same authorized FTS records are exposed through bounded inline/card/open "
                "contracts; the content token count is a conservative full-content upper bound."
            ),
            metrics=progressive_metrics,
        ),
        RetrievalVariantOutcome(
            variant="compaction_context",
            status=VariantStatus.COMPLETED,
            reason=(
                "The same frozen retrieval selection is preserved; the content-addressed M3 "
                "report separately executes the 100-message compaction/context workload."
            ),
            metrics=compaction_metrics,
        ),
        RetrievalVariantOutcome(
            variant="vector",
            status=vector_status,
            reason=vector_reason,
            metrics=None,
        ),
        RetrievalVariantOutcome(
            variant="hybrid_reranking",
            status=vector_status,
            reason=vector_reason,
            metrics=None,
        ),
    )
    selected_default = "deterministic_fts+progressive_navigation+compaction_context"
    fallback = "deterministic_fts+bounded_inline_context"
    payload = {
        "version": "memory-retrieval-report-v1",
        "fixture_version": fixture.manifest.version,
        "fixture_digest": fixture.manifest.fixture_digest,
        "retrieval_policy_version": fixture.manifest.retrieval_policy_version,
        "fixed_clock": fixture.manifest.fixed_clock.isoformat(),
        "outcomes": [item.model_dump(mode="json") for item in outcomes],
        "selected_default": selected_default,
        "fallback": fallback,
    }
    return RetrievalBenchmarkReport(
        fixture_version=fixture.manifest.version,
        fixture_digest=fixture.manifest.fixture_digest,
        retrieval_policy_version=fixture.manifest.retrieval_policy_version,
        fixed_clock=fixture.manifest.fixed_clock,
        outcomes=outcomes,
        selected_default=selected_default,
        fallback=fallback,
        report_digest=hashlib.sha256(_canonical(payload).encode()).hexdigest(),
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ndcg(selected_ids: tuple[str, ...], grades: dict[str, int], k: int) -> float:
    def dcg(values: list[int]) -> float:
        return math.fsum(
            (2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(values)
        )

    actual = [grades.get(record_id, 0) for record_id in selected_ids[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    ideal_score = dcg(ideal)
    return dcg(actual) / ideal_score if ideal_score else 0


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
