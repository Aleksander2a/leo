from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "migrations/versions/20260822_0026_slack_passive_thread_coverage.py"


def test_slack_thread_coverage_migration_is_the_single_forward_head() -> None:
    config = Config(ROOT / "alembic.ini")
    config.set_main_option("script_location", str(ROOT / "migrations"))
    scripts = ScriptDirectory.from_config(config)
    revision = scripts.get_revision("20260822_0026")

    assert scripts.get_current_head() == "20260822_0026"
    assert revision is not None
    assert revision.down_revision == "20260821_0025"


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
