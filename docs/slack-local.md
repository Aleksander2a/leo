# Local Slack verification

Leo uses Slack Socket Mode, so the demo runs locally over outbound WebSocket traffic and needs no
public callback URL.

## 1. Install or refresh the app

Create or update the app from [`slack/manifest.yml`](../slack/manifest.yml), enable Socket Mode, and
create an `xapp-` app token with `connections:write`.

The manifest subscribes to:

- `app_mention` for channels and group DMs;
- `message.im` for human messages in 1:1 DMs;
- passive `message.channels`, `message.groups`, and `message.mpim` events for the sanitized context
  plane. Passive events never launch work or produce a reply.

Its bot history/read scopes allow bounded background loading from public channels, private channels,
group DMs, and 1:1 DMs. Slack permits bot-token `conversations.replies` only for DMs/MPIMs, so a
separate optional user OAuth token with `channels:history` and `groups:history` is used only for exact
public/private/shared thread reads. The bot token remains the sole admission, membership, and write
authority. Reinstall the app after any manifest scope or event change. If Slack rotates a token,
replace only the ignored local `.env` value. Never paste a token into chat or logs.

Invite Leo to each public/private/shared channel or group DM where it should be available. A valid
event from the pinned workspace plus Leo's presence is sufficient admission; no strategy mapping,
channel configuration, or activation command exists.

## 2. Configure the local process

Set the following names in `.env`:

```text
SLACK_APP_TOKEN=xapp-...
SLACK_BOT_TOKEN=xoxb-...
SLACK_USER_TOKEN=xoxp-...   # optional but needed for historical channel-thread reads
LEO_SLACK_TEAM_ID=T...
LEO_ORGANIZATION_ID=demo-org
LEO_STRATEGY_ID=technology-ls
OPENROUTER_API_KEY=...
LEO_MODEL=...
DATABASE_URL=postgresql://...
```

`LEO_ORGANIZATION_ID` and `LEO_STRATEGY_ID` are non-gating demo-domain defaults. Event text,
channel metadata, and model output cannot select or expand authority. The process calls `auth.test`
and rejects a token from the wrong workspace.

Check names without printing values:

```powershell
uv run leo check-config
```

`check-config` reports `thread_history` separately from `live_slack`. A missing
`SLACK_USER_TOKEN` never makes Leo unavailable for top-level turns: the status is the non-blocking
`persisted_coverage_fallback_required` mode. In that mode, a public/private/shared thread reply can
run only when the passive context plane has an exact bot-attested root snapshot covering the current
boundary; otherwise the loader fails closed. `direct_user_history` avoids that cold-start limitation.

## 3. Transport and offline-harness checks

Run `uv run leo slack-smoke`, then mention Leo with arbitrary text. This checks normalization,
deduplication, thread routing, and Slack write transport without providers or Postgres.

Run `uv run leo slack-harness-smoke` for a deterministic custom-loop response through the same
transport. These commands are diagnostics, not the live persistence acceptance proof.

## 4. Live conversational path

Apply the current schema, then start the long-running listener/worker:

```powershell
uv run alembic upgrade head
uv run leo slack-live
```

`slack-live` admits every recognized Leo-present conversation kind, upserts a canonical
conversation, snapshots the exact context source set and its digest, persists the Task/Run, loads
authorized history and memory, and runs the generic coordinator. For a thread reply, the context
source is the exact destination thread from its root through every message before the current event,
including prior Leo replies. Root/recent turns, unresolved questions, decisions, corrections, and
prior tool/assistant outcomes are protected; older supporting turns may be source-linked and
compacted under the whole-request budget. The legacy channel-strategy table may remain for migration
compatibility, but admission never reads it.

The current Alembic head records the trusted authority behind
each admitted turn:
conversation kind, authority source, Leo presence, lifecycle, shared/external provenance, and
membership-policy version. Apply the complete forward migration chain to current head before
restarting a listener built from this source; running the new ORM against an older database is
intentionally rejected by preflight.

If a thread already has active work, later turns remain durably queued in arrival order instead of
being discarded. Startup recovery reclaims materialized or pending turns. Progress and final output
stay in the originating Slack thread.

## 5. Required live acceptance matrix

Use synthetic messages and verify both Slack output and Supabase rows:

1. Ping Leo with a non-quote conversational question in two different invited channels. Each reply
   must use only its own destination history.
2. Put different synthetic facts in those channels, then ask for both in a 1:1 DM. The DM should
   retrieve relevant facts only from the user's current shared source set, with source provenance.
3. Repeat from a group DM. It must stay group-local and must not receive the 1:1-DM union.
4. Ask for a current fact that requires a research tool and inspect grounded observations/claims.
5. Ask Leo to split a complex request into independent research nodes, delegate them, and synthesize
   a parent-verified answer.
6. Send a second turn while the first is working; it must queue, then run once.
7. Restart around a queued/materialized run and verify recovery plus a single final delivery.
8. Start a thread with a fact/question, let Leo answer, then ask a short follow-up such as `Why?`.
   Verify the context manifest covers the root through the prior Leo reply and records any compacted
   source range/handle; do not accept a recent-history sample as proof.
9. Ask an underspecified short question. Leo should ask one useful clarification with no tool call.
   Ask a short current-information question separately; Leo should autonomously choose the needed
   search/fetch or market tools without requiring the word `research` or a prescribed workflow.
10. Exercise Tavily discovery followed by a selected public-text fetch, and the configured Finnhub
    quote/profile/news/earnings/basic-financials tools. Search snippets alone must not ground a claim.
11. Induce a bounded provider, verification, context, and budget failure. Slack must receive a safe
    conversational explanation, any verifier-approved partial result, and a useful next step—never a
    bare terminal enum, run ID, generic “unverified work” boilerplate, or raw exception.

Expected operational degradation is explicit: the model is never called with a thread presented as
complete unless cursor-to-empty Slack history or the persisted coverage row proves root, reply count,
latest reply, and current boundary. Missing history permission, an ambiguous cursor, a protected-turn
overflow, or incomplete passive-event coverage produces a conversational context-unavailable reply;
it never silently substitutes the latest N messages or broadens to global context. Provider/database
failures similarly produce a useful explanation and next step rather than a bare `failed`,
`budget_exhausted`, or internal reason. Bot/self/subtype messages do not trigger loops.

Persisted coverage intentionally cannot serve an older queued turn after Slack's root metadata has
advanced to a later reply: that raced boundary fails closed. Only a direct, cursor-complete history
read that is explicitly bounded before the older turn may supply that context.

When authorized loaders produce the same authoritative root, Leo collapses them only if their
normalized identity/content agree; a conflict fails before the model. A Leo progress/final row after
the current user event is never exposed to that turn. Exact persisted coverage may still attest and
load only the prefix before that boundary.

At the 2026-08-22 handoff the listener is stopped. Public-channel and 1:1-DM conversational probes
are positive after the latest isolation/terminal-quality fixes. Private-channel and MPIM constrained-
format probes require a fresh rerun after their false-clarification/format fix, and the original
dividend-stock research thread still needs a successful current-data rerun. These are open live-
acceptance rows, not completed claims.

## Conversation semantics

- Public/private/shared/external channel: exact destination only.
- Group DM/MPIM: exact group only.
- 1:1 DM: DM-local plus the relevance-budgeted current user-and-Leo conversation intersection.
- Archived, inaccessible, revoked, or lookup-failed sources are excluded. If membership projection
  fails, Leo safely falls back to the current DM only.

Slack Connect is accepted in this synthetic demo when Slack delivers the event and Leo is present;
it remains isolated by exact Slack conversation ID. Its content can enter a 1:1-DM union only when
both the user and Leo still share that exact conversation.

## Operator lifecycle and recovery

These are demo-safe procedures; none of them changes conversation or memory authority.

- **Rotate an app-level token:** create a replacement `xapp-` token with only
  `connections:write`, update the ignored local `SLACK_APP_TOKEN`, restart Leo, and confirm
  `auth.test` plus a synthetic mention before revoking the old token.
- **Refresh bot scopes or rotate the bot token:** apply the checked-in manifest, reinstall the app,
  update only `SLACK_BOT_TOKEN`, restart, and verify one public-channel mention, one private/group
  mention, and a 1:1 DM. Reinstallation is required after scope/event changes.
- **Suspected compromise:** stop every listener/worker first, revoke both Slack tokens in the app
  console, replace them, then inspect sanitized ingress/event timestamps from the suspected window.
  Do not replay unknown Slack delivery effects automatically.
- **Remove Leo from one conversation:** remove the app from that channel/group and refresh the
  actor/Leo membership projection. New events must not arrive, and the removed destination must
  disappear from subsequent 1:1-DM source sets and cache keys.
- **Uninstall:** stop Leo, uninstall the Slack app, and revoke any remaining app-level token. Durable
  demo traces remain for audit but confer no Slack access. A readiness check must report the socket
  and admission boundary unavailable.
- **Reinstall/rollback:** apply the desired manifest version, reinstall, recreate the app-level
  token if necessary, invite Leo only to the intended conversations, restart, then repeat the live
  acceptance matrix. If a manifest refresh fails, reapply the last checked-in version and reinstall;
  never restore availability by inserting a strategy mapping.

After any lifecycle action, verify the Slack receipt and the matching Supabase ingress row. A Slack
reply without a persisted authority snapshot, or a persisted terminal intent without an unambiguous
Slack receipt, is a failed check rather than proof of recovery.

## Operator readiness and durable recovery

`uv run leo health` is cheap liveness/config output. `uv run leo health --deep` adds bounded,
read-only database aggregates for queued/expired work, unmaterialized Slack launches, outbox lag,
unknown effects, conversation/membership freshness, parent/child leases, the latest model result,
and the latest successful parent run. A timeout becomes a typed degraded/unknown component; health
never calls a model, research provider, or Slack write method.

Socket readiness comes from the running Socket Mode client's actual ping/session state. A separate
one-shot CLI process cannot observe another process's in-memory socket and must report it as
`unknown`, not infer readiness from environment variables. Use the listener's registered snapshot
when health is hosted in the same process, plus a fresh synthetic ping for acceptance evidence.

On restart, `slack-live` scans unmaterialized/claimable launches in immutable admission order,
reclaims expired task leases, reconciles missing terminal delivery intents, and drains safely
retryable outbox rows before accepting new wake-up hints. An outbox row in `unknown_effect` is never
blindly reposted: reconcile its Slack receipt manually, then confirm or reject the durable intent.
Message edits, deletions, bot messages, and unsupported subtypes launch no work. Each new explicit
mention is an immutable new turn; a first-class user cancellation command remains a separate
workflow and must not be simulated by rewriting or deleting an accepted event.

### Current-head Postgres acceptance packet

Stop every Slack listener and standalone worker before running the committed-row concurrency
packet. Then explicitly acknowledge that condition:

```powershell
.\scripts\run_m2_postgres_acceptance.ps1 -ListenerStopped
```

The script runs the 63 reviewed M2 cases plus two rollback-safe M5 durable-recovery cases. Ordinary
cases use one outer transaction and roll back; the genuine two-connection races commit
UUID-namespaced synthetic rows, delete only that exact organization/team namespace in
foreign-key-safe order, and verify that no scoped rows remain. It then validates the observed pytest
artifacts and writes `artifacts/m5-postgres-reliability-v1.json` plus a refreshed final-evidence
aggregate. Do not run it beside a listener because production startup scanners do not participate in
the test advisory lock. A host kill or network loss during teardown can still leave its uniquely
prefixed synthetic namespace; identify and clean only that exact namespace before retrying rather
than broadening the cleanup predicate.
