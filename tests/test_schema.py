from __future__ import annotations

from leo.persistence.schema import Base


def test_initial_schema_covers_durable_harness_records() -> None:
    tables = Base.metadata.tables
    assert {
        "threads",
        "tasks",
        "runs",
        "observations",
        "claims",
        "run_events",
        "slack_channel_scopes",
        "slack_ingress_events",
        "delivery_outbox",
        "organizations",
        "strategies",
        "organization_memberships",
        "assets",
        "portfolios",
        "positions",
        "mandates",
        "theses",
        "thesis_versions",
        "thesis_assumptions",
        "strategy_decisions",
        "risk_constraints",
        "memory_records",
        "memory_sources",
        "memory_revisions",
        "conversations",
        "conversation_threads",
    }.issubset(tables)
    assert "started_at" in tables["runs"].c
    assert "deadline_at" in tables["runs"].c
    assert "mapping_version" in tables["threads"].c
    assert {"parent_task_id", "continuation_kind", "mapping_version"}.issubset(
        tables["tasks"].c.keys()
    )
    assert "event_sequence" in tables["runs"].c
    assert {
        "status",
        "quality",
        "schema_version",
        "normalization_version",
        "rejection_code",
    }.issubset(tables["observations"].c.keys())
    assert {
        "ck_observations_status",
        "ck_observations_quality",
        "ck_observations_schema_version",
        "ck_observations_rejection_state",
    }.issubset(constraint.name for constraint in tables["observations"].constraints)
    assert "external_event_id" in tables["threads"].c
    assert "conversation_id" in tables["threads"].c
    assert "lease_expires_at" in tables["slack_ingress_events"].c
    assert "attempt_count" in tables["slack_ingress_events"].c
    assert {
        "lease_owner",
        "lease_token",
        "lease_expires_at",
        "heartbeat_at",
        "attempt_count",
        "retry_after",
        "last_error",
    }.issubset(tables["tasks"].c.keys())
    assert "organization_id" in tables["slack_ingress_events"].c
    assert "strategy_id" in tables["slack_ingress_events"].c
    assert "mapping_version" in tables["slack_ingress_events"].c
    assert "conversation_id" in tables["slack_ingress_events"].c
    assert {
        "conversation_authority_source",
        "bot_presence",
        "conversation_lifecycle",
        "external_provenance",
        "membership_policy_version",
    }.issubset(tables["slack_ingress_events"].c.keys())
    assert {
        "authority_source",
        "bot_presence",
        "lifecycle",
        "external_provenance",
        "membership_policy_version",
    }.issubset(tables["conversations"].c.keys())
    assert {
        "launch_status",
        "launch_attempt_count",
        "launch_error",
        "launch_updated_at",
    }.issubset(tables["slack_ingress_events"].c.keys())


def test_slack_channel_scope_has_composite_identity_and_no_strategy_foreign_key() -> None:
    table = Base.metadata.tables["slack_channel_scopes"]
    assert {column.name for column in table.primary_key.columns} == {"team_id", "channel_id"}
    assert {"provisioned_by_user_id", "provisioned_via", "version", "status"}.issubset(
        table.c.keys()
    )
    assert not table.foreign_keys


def test_missing_foreign_key_indexes_are_in_metadata() -> None:
    assert "ix_tasks_thread_id" in {index.name for index in Base.metadata.tables["tasks"].indexes}
    assert "ix_slack_ingress_events_task_id" in {
        index.name for index in Base.metadata.tables["slack_ingress_events"].indexes
    }
    assert "uq_slack_ingress_task_id" in {
        index.name for index in Base.metadata.tables["slack_ingress_events"].indexes
    }
    assert "ix_tasks_claim_eligibility" in {
        index.name for index in Base.metadata.tables["tasks"].indexes
    }
    assert "ix_delivery_outbox_claim_eligibility" in {
        index.name for index in Base.metadata.tables["delivery_outbox"].indexes
    }
    assert "ix_delivery_outbox_task" in {
        index.name for index in Base.metadata.tables["delivery_outbox"].indexes
    }
    assert "uq_tasks_one_active_per_thread" in {
        index.name for index in Base.metadata.tables["tasks"].indexes
    }
    assert "ix_threads_conversation" in {
        index.name for index in Base.metadata.tables["threads"].indexes
    }
    assert "ix_memory_capability_handles_task" in {
        index.name for index in Base.metadata.tables["memory_capability_handles"].indexes
    }
    assert "uq_conversation_threads_harness_thread" in {
        constraint.name for constraint in Base.metadata.tables["conversation_threads"].constraints
    }


def test_slack_thread_coverage_is_conversation_local_and_fail_closed() -> None:
    messages = Base.metadata.tables["sanitized_messages"]
    coverage = Base.metadata.tables["slack_thread_coverage"]

    assert "provider_thread_root_ts" in messages.c
    assert "ix_sanitized_messages_provider_thread" in {index.name for index in messages.indexes}
    assert "ck_sanitized_messages_provider_thread_root_ts" in {
        constraint.name for constraint in messages.constraints
    }
    assert {
        "conversation_id",
        "team_id",
        "channel_id",
        "thread_root_ts",
        "authoritative_reply_count",
        "authoritative_latest_reply_ts",
        "authority_source",
        "authority_snapshot_hash",
        "metadata_observed_at",
    }.issubset(coverage.c.keys())
    assert "organization_id" not in coverage.c
    assert "strategy_id" not in coverage.c
    assert {
        "uq_slack_thread_coverage_root",
        "ck_slack_thread_coverage_reply_count",
        "ck_slack_thread_coverage_reply_shape",
        "ck_slack_thread_coverage_root_ts",
        "ck_slack_thread_coverage_latest_ts",
        "ck_slack_thread_coverage_authority_source",
        "ck_slack_thread_coverage_snapshot_hash",
    }.issubset(constraint.name for constraint in coverage.constraints)
    assert {index.name for index in coverage.indexes} == {"ix_slack_thread_coverage_conversation"}
