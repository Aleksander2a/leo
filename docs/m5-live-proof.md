# M5 live-proof collection

The M5 collector reconciles Slack-visible acceptance with current durable Postgres truth. It has no
Slack client and its database implementation issues only `SELECT` statements. The proof output
contains stable IDs, statuses, counts, and SHA-256 digests; prompts, answers, memory content, provider
payloads, connection strings, and credentials are not exported.

Run it only after the acceptance cohort has been exercised against the current database epoch:

```powershell
.venv\Scripts\python.exe -m leo.evals.live_proof_operator `
  --request artifacts\fresh-m5-live-request.json `
  --proof artifacts\m5-offline-proof-v2.json `
  --output artifacts\m5-live-proof-v2.json `
  --actor-id trusted-demo-operator `
  --not-before-received-at 2026-08-22T12:00:00+00:00 `
  --not-before-message-ts 1788000000.000000
```

`DATABASE_URL`, `LEO_SLACK_TEAM_ID`, and `LEO_ORGANIZATION_ID` come from trusted runtime settings;
they cannot be selected through the command. The proof input can be a direct `proof-v2` manifest or
a frozen report containing `proof_manifest`. A missing, stale, ambiguous, or inconsistent row causes
exit code 2 and no output. A valid partial cohort is written with exit code 1 and cannot pass the live
gate; the exact complete cohort exits 0.

## Request contract

The request must exactly partition these nine independently required evidence IDs between sorted
`bindings` and sorted `pending_evidence_ids`:

- `cross_channel_negative`
- `delegated_replanning`
- `dm_membership_union`
- `group_dm`
- `memory_recall`
- `memory_write`
- `private_channel`
- `quote`
- `sec`

Every binding supplies a full `run-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx` ID, exact Slack request TS,
exact destination, conversation kind, exact sorted context-conversation source set, its server-derived
access hash, plan expectation, and optionally the expected Slack reply TS. Run prefixes and bindings
at or before either post-restore boundary are rejected.

The 1:1-DM and group-DM cases are intentionally different. A `dm_membership_union` binding must use
conversation kind `dm`, contain the current DM destination plus at least one current shared channel,
and name a non-DM `expected_recall_source_conversation_id`. Collection succeeds only if a retrieved
`memory.search` observation has a positive item from that exact channel and a persisted inference
claim cites the observation. Its exact source-set digest and access hash are included in the proof.
The collector also re-selects the actor's current durable membership rows: the active set, projection
source, and access hash must still exactly equal the immutable ingress snapshot. A stale, revoked, or
extra active source fails closed and only a digest/count of this current projection is exported.

A `group_dm` binding must use `mpim` and the singleton source set containing only its destination.
Any attempted channel aggregation fails before database collection. `memory_recall` remains a separate
same-channel proof whose recall source must equal its destination.

## Observed case contracts

The collector does not treat a completed task/run or a fixture assertion as acceptance evidence.
Each exact binding must also satisfy its case-specific durable contract:

- `memory_write` selects one active current memory record and its first revision plus the exact
  `slack_event`, `leo_task`, and `slack_message` sources. Visibility and namespace must match the
  admitted destination, the rows must fall within the bound run, and the observation must have the
  matching `memory.remember` start/completion/observation event path. Retracted, superseded,
  foreign-scope, or widened provenance fails closed.
- `memory_recall` and `dm_membership_union` require one normalized `memory.search` result, one
  persisted inference carried by the final answer, and only selected items from the binding's exact
  expected source conversation. The completed tool-event path is mandatory.
- `cross_channel_negative` requires a normalized empty `memory.search` result (`selected_count=0`)
  and the exact scoped-negative inference in both the persisted claim and final answer.
- `quote` and `sec` require exactly one fresh `observation-v2` from Finnhub or SEC EDGAR,
  respectively, the exact completed provider-tool path, and persisted source claims carried by the
  parent answer. Extra observations, stale evidence, discovery-only/foreign providers, or uncited
  claims fail closed.
- `private_channel` and `group_dm` require a direct verified answer with explicit zero tool calls,
  no observation/claim rows, and no plan. Group-DM context remains the exact singleton destination.
- `delegated_replanning` requires a completed durable parent plan with at least two exact completed
  node/delegation pairs. Child tasks/runs must be durable subagents, their verified evidence
  envelopes must exactly match persisted claims and observations, market and SEC evidence must come
  from distinct overlapping child runs, no child may own a Slack outbox row, and the parent's source
  claims must cite its verified-child plan observation.

Only content-free identifiers, counts, status values, and canonical SHA-256 digests enter the proof
artifact. Provider payloads, memory content, prompts, answers, and source references are represented
only by digests.

After a schema/data reset, prior Slack timestamps are historical references only. All nine bindings
must be recollected with fresh post-restore Slack requests and durable rows before the proof can be
marked complete.
