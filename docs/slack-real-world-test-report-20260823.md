# Leo Slack real-world test report — 2026-08-23

## Scope

The live Socket Mode listener was exercised through the Slack connector across
public channels, a private channel, a 1:1 DM, exact-thread follow-ups, a
multi-step conversation, current-thread recall, explicit memory isolation, and
passive non-triggering messages. All test values were synthetic.

## Bugs found and fixed

### 1. Stale thread roots could become top-level channel posts

The reported high-yield crypto request had one correct in-thread run:

- Channel: `#all-leo-demo` (`C0BRFU0LQF8`)
- Root: `1787488782.490989`
- Replies: `1787488786.054989` (progress), `1787488821.523609` (final)
- Durable ingress: `Ev0BSXU7ETGQ`

The apparent duplicate was a separate delayed event whose parent had been
deleted. Slack returned `thread_not_found` for root `1787488534.667199`, but
accepted a later `chat.postMessage` with that stale `thread_ts` as a top-level
post.

Fixed in `src/leo/persistence/outbox.py` and
`src/leo/integrations/slack/socket_mode.py`:

- The outbox probes the exact root immediately before posting.
- Missing roots are dead-lettered as `slack_thread_root_missing` without a
  Slack post.
- Probe failures retry without posting.
- Admission now rejects definitively stale threaded events before durable/model
  work, avoiding wasted runtime budget.

Regression coverage is in `tests/test_outbox_reliability.py` and
`tests/test_slack_transport.py`.

### 2. Context-only follow-ups could enter deliberation retry loops

The public three-word follow-up produced a model answer, but its optional
source claims cited context-item IDs that were not observation IDs. Verification
rejected those claims repeatedly and ended with `deliberation_no_progress`.

Fixed in `src/leo/harness/deliberation.py` and
`src/leo/integrations/openrouter.py`:

- Context-only conversational turns now drop unsourced optional provider claims.
- The provider prompt explicitly distinguishes context-item IDs from
  observation IDs and instructs the model to omit optional source claims when
  no observations exist.
- Required evidence turns remain strict.

### 3. DM and channel thread roots could fail actor reconciliation

Slack connector messages are posted by the ChatGPT app at the provider layer,
while the durable Slack ingress records the human actor. The exact durable root
and Slack root therefore appeared to have different actors. This previously
caused DM runtime errors and, for channel follow-ups, context-unavailable
failures.

Fixed in `src/leo/worker/slack_conversation.py`:

- Forwarded app roots are accepted in DMs and channels only when the exact
  destination, root ID, normalized mention, and normalized content reconcile.
- The durable human ingress remains authoritative; this does not broaden the
  authorized conversation scope.

### 4. Mentioned channel messages raced passive persistence

For channel `app_mention` events, the listener could launch the run before the
parallel passive-message handler persisted the current user message. The exact
thread snapshot then appeared incomplete and the run ended with
`context_unavailable:slack_thread_history_unavailable`.

Fixed in `src/leo/integrations/slack/socket_mode.py` and
`src/leo/integrations/slack/events.py`:

- Mentioned messages are persisted synchronously before launch.
- Duplicate persistence remains idempotent when the normal `message` callback
  also arrives.
- Connector `app_mention` payloads that omit `channel_type` are accepted for
  this pre-persistence path; final conversation eligibility is still derived
  from Slack conversation metadata before launch.

### 5. Current-thread recall was incorrectly routed to global memory search

Questions such as “What marker did I ask you to note?” were treated as
cross-conversation memory searches even though the exact thread transcript was
already available. An empty authorized search then hid the answer.

Fixed in `src/leo/live.py`:

- Current-thread exchange recall uses the exact admitted transcript.
- Explicit commands such as “search your memory” still invoke scoped memory.
- Context-only source claims remain sanitized as described above.

## Final live verification

The final listener instance was started at 18:02 CEST. Its log is
`artifacts/slack-live-20260823-181000.err.log`; after startup and Socket Mode
connection it contains no warnings or errors.

| Surface / test | Result | Live evidence |
|---|---|---|
| Public root + exact three-word follow-up | Pass | `#all-leo-demo`; root `1787500999.956749`; follow-up `1787501051.665059`; Leo replied in the same thread with three words. |
| Public marker recall on final build | Pass | `#social`; root `1787501132.310689`; follow-up `1787501180.055759`; Leo replied exactly `violet-comet-31`. |
| Public multi-step thread | Pass | `#new-channel`; root `1787500362.568019`; step two `1787500439.498989` → `Step two recorded.`; step three `1787500626.780049` → `42`. |
| Private-channel root + five-word follow-up | Pass | `#qa-private-20260823` (`C0BS795NTEG`); root `1787500366.013009`; follow-up `1787500443.227929` → `Private channel test confirmed successfully.` |
| 1:1 DM root + three-word follow-up | Pass | `D0BRMS9FSG1`; root `1787500370.519249`; follow-up `1787500446.647419` → `Your AI assistant`. |
| DM explicit memory isolation | Pass | Follow-up `1787500720.496729`; Leo returned `No matching authorized memory was found in this conversation scope.` |
| Passive private-channel message | Pass | `1787500682.097769` received no Leo reply. |
| Original generic DM runtime failure retest | Pass | Follow-up `1787499191.554789` answered the actual request: `You asked me to state, in one concise sentence, what my role is in this direct message.` |

## Observed Slack behavior / remaining caveats

1. A public-channel continuation without a fresh `@Leo` mention does not emit
   an `app_mention` event, so the listener correctly does nothing. DMs emit
   `message_im` events and do not require a fresh mention. Channel follow-ups
   should therefore mention Leo explicitly. This is Slack event semantics, not
   a delivery duplication.

2. A true MPIM/group DM was not available for this identity. Creating a DM with
   Leo and the current user deduplicated to the existing 1:1 DM. A second human
   Slack participant is required to cover true MPIM behavior.

3. Historical failed QA messages remain visible in the test channels for
   inspection; they were not deleted. The final clean runs above are the
   post-fix evidence.

## Verification

The focused Slack, context, deliberation, outbox, and live-composition tests
pass. Ruff reports no issues on changed files, and mypy reports no issues in the
changed live/Slack runtime modules.
