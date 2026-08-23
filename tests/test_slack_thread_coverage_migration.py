from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/versions/20260822_0026_slack_passive_thread_coverage.py"
EMBEDDINGS_MIGRATION = ROOT / "migrations/versions/20260823_0028_embeddings.py"
MODEL_CALL_TRANSCRIPTS_MIGRATION = (
    ROOT / "migrations/versions/20260823_0029_model_call_transcripts.py"
)


def test_slack_thread_coverage_migration_is_the_single_forward_head() -> None:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("20260822_0026")

    assert revision is not None
    assert revision.down_revision == "20260821_0025"


def test_memory_fts_english_migration_is_the_single_forward_head() -> None:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("20260823_0027")

    assert revision is not None
    assert revision.down_revision == "20260822_0026"


def test_embeddings_migration_is_the_single_forward_head() -> None:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("20260823_0028")

    assert revision is not None
    assert revision.down_revision == "20260823_0027"


def test_model_call_transcripts_migration_is_the_single_forward_head() -> None:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("20260823_0029")

    assert scripts.get_current_head() == "20260823_0029"
    assert revision is not None
    assert revision.down_revision == "20260823_0028"


def test_model_call_transcripts_migration_has_client_deny() -> None:
    source = MODEL_CALL_TRANSCRIPTS_MIGRATION.read_text(encoding="utf-8")

    assert "model_call_transcripts" in source
    assert "ENABLE ROW LEVEL SECURITY" in source
    assert "CREATE POLICY leo_client_deny" in source
    assert "FROM anon, authenticated" in source


def test_embeddings_migration_has_pgvector_and_client_deny() -> None:
    source = EMBEDDINGS_MIGRATION.read_text(encoding="utf-8")

    assert "CREATE EXTENSION IF NOT EXISTS vector" in source
    assert "memory_embeddings" in source
    assert "capability_embeddings" in source
    assert "ck_memory_revisions_source_type" in source
    assert "ENABLE ROW LEVEL SECURITY" in source
    assert "CREATE POLICY leo_client_deny" in source
    assert "FROM anon, authenticated" in source


def test_slack_thread_coverage_migration_has_topology_authority_and_client_deny() -> None:
    source = MIGRATION.read_text(encoding="utf-8")

    assert 'sa.Column("provider_thread_root_ts", sa.String(64))' in source
    assert "ck_sanitized_messages_provider_thread_root_ts" in source
    assert "slack_conversations_history_bot" in source
    assert "slack_conversations_history_user" in source
    assert "ck_slack_thread_coverage_reply_shape" in source
    assert "ck_slack_thread_coverage_root_ts" in source
    assert "ck_slack_thread_coverage_latest_ts" in source
    assert "ENABLE ROW LEVEL SECURITY" in source
    assert "CREATE POLICY leo_client_deny" in source
    assert "FROM anon, authenticated" in source
    assert 'sa.Column("organization_id"' not in source
    assert 'sa.Column("strategy_id"' not in source
