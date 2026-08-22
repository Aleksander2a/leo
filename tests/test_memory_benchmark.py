from __future__ import annotations

from pathlib import Path

import pytest

from leo.memory.benchmark import (
    FrozenRetrievalFixture,
    RetrievalBenchmarkReport,
    VariantStatus,
    load_frozen_retrieval_fixture,
    retrieval_fixture_digest,
    run_retrieval_benchmark,
    validate_committed_retrieval_report,
    validate_frozen_retrieval_fixture,
)

FIXTURE = Path(__file__).resolve().parents[1] / "evals/fixtures/memory-retrieval-v1"


def test_frozen_fts_report_passes_isolation_and_d058_without_fake_vector_metrics() -> None:
    report = validate_committed_retrieval_report(FIXTURE)

    fts, progressive, compaction, vector, hybrid = report.outcomes
    assert fts.status is VariantStatus.COMPLETED
    assert fts.metrics is not None
    assert fts.metrics.recall_at_k == 1
    assert fts.metrics.full_query_coverage == 1
    assert fts.metrics.ndcg_at_k == 1
    assert fts.metrics.expected_dm_source_coverage == 1
    assert fts.metrics.current_revision_recall == 1
    assert fts.metrics.conflict_recall == 1
    assert fts.metrics.leakage_count == 0
    assert fts.metrics.estimated_context_tokens == 258
    assert fts.metrics.context_cost_status == "content_only_utf8_bytes_div4_v1"
    assert fts.metrics.latency_ms is None
    assert progressive.status is VariantStatus.COMPLETED
    assert progressive.metrics is not None
    assert progressive.metrics.recall_at_k == fts.metrics.recall_at_k
    assert progressive.metrics.leakage_count == 0
    assert progressive.metrics.context_cost_status == "progressive_full-content-upper-bound-v1"
    assert compaction.status is VariantStatus.COMPLETED
    assert compaction.metrics is not None
    assert compaction.metrics.recall_at_k == fts.metrics.recall_at_k
    assert compaction.metrics.context_cost_status == "compaction-selection-parity-v1"
    assert vector.status is VariantStatus.NOT_REQUIRED
    assert hybrid.status is VariantStatus.NOT_REQUIRED
    assert vector.metrics is None
    assert hybrid.metrics is None


def test_fixture_digest_rejects_corpus_drift_and_report_is_replayable() -> None:
    fixture = load_frozen_retrieval_fixture(FIXTURE)
    changed_documents = (
        fixture.documents[0].model_copy(update={"content": "Changed synthetic content."}),
        *fixture.documents[1:],
    )
    changed = fixture.model_copy(update={"documents": changed_documents})

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_frozen_retrieval_fixture(changed)

    replayed = run_retrieval_benchmark(fixture)
    committed = RetrievalBenchmarkReport.model_validate_json(
        (FIXTURE / "report.json").read_text(encoding="utf-8")
    )
    assert replayed == committed


def test_d058_requires_vector_work_when_declared_fts_threshold_is_missed() -> None:
    fixture = load_frozen_retrieval_fixture(FIXTURE)
    query = fixture.queries[0]
    missed_query = query.model_copy(
        update={
            "relevance_grades": {
                **query.relevance_grades,
                "mem-a-atlas": 3,
                "mem-a-northstar-long": 3,
            }
        }
    )
    changed_queries = (missed_query, *fixture.queries[1:])
    digest = retrieval_fixture_digest(fixture.documents, changed_queries)
    changed = FrozenRetrievalFixture(
        manifest=fixture.manifest.model_copy(update={"fixture_digest": digest}),
        documents=fixture.documents,
        queries=changed_queries,
    )

    report = run_retrieval_benchmark(changed)

    assert report.outcomes[0].metrics is not None
    assert report.outcomes[0].metrics.full_query_coverage < 1
    assert report.outcomes[3].status is VariantStatus.REQUIRED
    assert report.outcomes[4].status is VariantStatus.REQUIRED
    assert report.outcomes[3].metrics is None
    assert report.outcomes[4].metrics is None
