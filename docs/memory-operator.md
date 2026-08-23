# Memory operator runbook

Leo's durable memory is conversation-native. A channel, group DM, shared channel, or external
channel can read and write only its exact conversation namespace. A 1:1 DM may additionally read
the current user's same-workspace conversations where Leo is currently present. Group DMs remain
isolated. Optional strategy labels are provenance and never grant access.

## Autonomous capture

Alongside the explicit `remember`/`correct`/`forget` commands below, Leo can also save a note on its
own initiative during an ordinary conversational turn via the `memory.note` tool
(`src/leo/memory/tools.py:AutonomousMemoryTool`). The model supplies only free-text content, a coarse
kind, and a short reason; scope/visibility/namespace/provenance are still entirely harness-derived
from the sealed Slack destination authority, exactly as with the explicit commands. Every proposal is
routed through `ExplicitMemoryService.propose_autonomous`, which runs it against
`memory.policy.assess_candidate` for duplicate/contradiction governance before committing: an exact
duplicate is a no-op, a single conflicting active record in the same namespace is updated via
`correct` (a new revision superseding it) rather than creating a disconnected duplicate, and
anything else becomes a new record. There is no separate interactive confirmation step for this
path — it is designed to make memory accumulate as a side effect of conversation rather than only
through an explicit command.

## Semantic recall

Retrieval is hybrid, not lexical-only. Every remembered/corrected/autonomously-captured revision is
embedded (`openai/text-embedding-3-small` via OpenRouter) and persisted into the `memory_embeddings`
pgvector table (`src/leo/persistence/memory_embeddings.py`), alongside a `source_type` tag
(`explicit` vs `autonomous`) recorded directly on the revision. A search with a query embedding runs
a parallel vector-KNN pool (cosine distance) against a lexical full-text pool and fuses the two with
reciprocal rank fusion (`src/leo/harness/fusion.py`, the same technique used for tool discovery) --
so a conceptual query with no shared vocabulary with the stored content can still surface it. Both
pools apply the identical scope/visibility/sensitivity/temporal authorization filter
(`_authorized_hard_filters` in `src/leo/persistence/memory_retrieval.py`); the vector query never
sees a row the lexical query would not have been authorized to see either.

Embedding is best-effort and asynchronous to the write it accompanies: a gateway failure or missing
API key degrades that one revision to lexical-only recall, never blocks or rolls back the memory
write itself. A revision with no embedding row simply never enters the vector pool -- it stays fully
findable through FTS.

## Read-only projection

`memory-project` renders escaped Markdown from explicit visibility/namespace pairs. It never reads
wildcards and never uses the optional strategy label as disclosure authority.

```powershell
leo memory-project --namespace conversation_local=C012345 --page-size 25
```

For a current 1:1-DM projection, repeat only the namespaces authorized by the same server-derived
membership snapshot used for that run, including the actor-private namespace when appropriate:

```powershell
leo memory-project `
  --namespace conversation_local=D012345 `
  --namespace conversation_local=C012345 `
  --namespace actor_private=U012345
```

The command prints the page followed by a small JSON manifest. Pass its `next_cursor` back with
`--after`. A cursor is bound to the configured workspace/domain scope and fails if altered. Every
page lists the exact source record/revision pairs; it is a regenerable, read-only view and must not
be edited as a memory write path.

## Logical forget

Normal user-facing deletion is the confirmed `memory.forget` tool. It appends a retracted revision
and immediately excludes the record from retrieval, context, projections, cached results, and
previously issued progressive-open handles. This is the default operation.

## Manual physical purge

Physical deletion exists only for synthetic/demo cleanup. It accepts one to 100 explicit record IDs,
rejects globs, and deletes only records already logically retracted.

First create a dry-run manifest:

```powershell
leo memory-purge memory-123 memory-456
```

Review the scope, ordered record IDs, generation/revision snapshots, manifest hash, and confirmation
token. Then repeat the exact request with that token:

```powershell
leo memory-purge memory-123 memory-456 --confirm confirm:0123456789abcdef
```

The confirm invocation prepares the plan again. A concurrent correction, new revision, changed
status, scope mismatch, missing target, reordered target list, or stale token stops the purge. The
execute transaction locks every surviving target, rechecks its generation/current revision/status,
deletes dependent revisions and unreferenced sources/jobs, and invalidates workspace retrieval
caches. Repeating a fully completed plan is safe at the repository level and reports already-absent
records; the CLI deliberately asks for a fresh dry-run before each execution.

Never use physical purge for active records or as an automatic retention job. Do not run purge while
investigating a memory lifecycle/retrieval race; retain the append-only revisions until the incident
is understood.

## Recovery and checks

- A failed/retried embedding job can be reclaimed after its lease; completion is attempt-versioned.
- Compaction summaries are derived, incrementally versioned, and invalidated when a covered message
  changes. The latest exact-thread summary plus the recent un-compacted window is used for context.
- Membership, Leo presence, lifecycle, record generation, or revision changes invalidate caches and
  progressive handles. Every `memory.open` repeats authorization inside its transaction.
- Run `leo health --deep` for aggregate database/queue health. Run the frozen memory benchmark and
  the Postgres memory tests before treating a retrieval-policy change as accepted evidence.

