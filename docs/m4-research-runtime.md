# M4 research runtime

## Executable design

Leo uses one policy and evidence path for native tools, delegated children, plans, and MCP tools:

```text
trusted conversation authority + objective
  -> policy eligibility (before recall)
  -> short skill/tool recall; selected procedures/schemas only
  -> bounded READ execution or durable parent-owned plan/delegation
  -> atomic finite/size-bounded normalizer
  -> scoped Observation v2 with status, quality, provenance, time, and hash
  -> deterministic grounding + intention-specific research verifier
  -> verified parent completion or bounded correction/replan/safe stop
  -> renderer-v2 immutable multipart Slack outbox intents
```

Provider text, fetched pages, MCP metadata, skill procedures, and child prose are untrusted data.
None can grant a tool, widen conversation authority, alter a budget, or declare completion.
Conversational answers remain available when no tool is recalled; tool recall is not an admission or
answer-availability gate.

## Progressive skills and tool catalog

Skill metadata uses `leo-skill-v1` and fixes the skill ID/version, schema and procedure hashes,
domains/capabilities, required evidence, fallback/stop rules, compatible profiles/tools, child
compatibility, and procedure trust. Only a selected, compatible procedure enters context. The four
checked-in fixtures cover narrow quote, thesis challenge, general conversation, and delegated
research. Unknown versions, duplicate identities, hash mismatch, path escape, and size overflow fail
closed.

Every recalled executable is represented by a catalog record with semantic version and schema
fingerprint; short/long descriptions; domain/capability/entity tags; effect/phase/profile/role and
optional namespace/conversation policy; auth/health; sensitivity; freshness; cost/latency/rate and
result caps; observation kind/normalization version; and verification expectations. Policy removes
unauthorized, unhealthy, effectful, or over-budget entries before their descriptions are ranked.
`tool.search` and `tool.describe` use run-bound eligible handles, exact call/item/byte budgets, stable
tie-breaking, and no-progress limits. A 1,000-distractor executable regression retains the required
tool and exposes zero forbidden schemas.

## Provider contracts

| Capability | Evidence projection | Fail-closed controls | Verification expectation |
| --- | --- | --- | --- |
| `market.get_quote` | exact numeric quote fields, symbol, provider as-of/reference/request ID, expiry | schema, non-finite/zero, stale/future time, timeout, 429/4xx/5xx, concurrency | exact numeric token and as-of grounding |
| `sec.get_recent_filings` | trusted ticker/CIK, capped recent filing metadata, canonical accession URL, request ID, 15-minute cache | identity map, equal arrays, safe document path, no filings, timeout, 403/429/5xx, global 8 req/s scheduler | canonical primary-source filing metadata |
| `web.search_public` | at most five English Wikipedia result URLs and query hash | HTTPS host/path projection, empty/schema, timeout/rate/4xx/5xx, concurrency | discovery only; cannot ground a fact |
| `web.fetch_public_text` | capped sanitized text, URL chain, content/hash/bytes, DNS-set fingerprint and actual peer | credential/private address, redirect/type/byte/time caps, active-content removal, unverifiable or changed peer | retained-text grounding; truncated text cannot ground |

The SEC tool intentionally returns bounded filing metadata and references instead of putting filing
bodies into the prompt. A later `web.fetch_public_text` read can retrieve a selected public filing
under the same untrusted-text cap. This is the demo-appropriate interpretation of the filing-text
invariant: oversized raw filings remain out of context rather than being silently truncated into
primary-source claims.

## MCP decision

Leo pins the official Python MCP SDK `mcp==2.0.0` and uses its `Client` in `auto` protocol mode. Tool
semantic versions and the server/transport/allowlist are trusted configuration because MCP tool
metadata does not define Leo's semantic version. Discovery is paginated and capped; schemas use
JSON Schema Draft 2020-12 and reject reserved authority keys. Calls accept complete structured
content only and retain Leo's timeout, cancellation, concurrency, result-size, policy, normalization,
and verifier boundaries. Health drift degrades the catalog until reconnect and rediscovery.

Server prompts, roots, sampling, elicitation, instructions, unstructured content, recursive calls,
and WRITE/ADMIN tools are unsupported by design. No remote MCP server is required for M4; the
official in-memory server contract is the deterministic compatibility proof. See the
[official SDK repository](https://github.com/modelcontextprotocol/python-sdk) and
[client documentation](https://py.sdk.modelcontextprotocol.io/client/).

## DNS rebinding decision

For this test demo, each URL/redirect is resolved immediately before the request; all resolved
addresses must be public, and the actual connected peer reported by the HTTP transport must belong
to that exact set. Missing peer metadata or a mismatch fails closed before any response content can
become evidence. Deterministic tests cover private IPv4/IPv6, private redirect, changed peer, missing
peer, active HTML, type/byte/redirect caps, and malformed content.

This is evidence/disclosure isolation, not a claim that the injected generic HTTP client never opens
a socket before post-connect peer inspection. A production service should own a resolver-pinned
transport or enforce the destination in an egress proxy/network sandbox. Proxies that hide the peer
are intentionally unavailable rather than silently weakening the policy.

## Observation and verification semantics

| State/quality | May support a current source claim? |
| --- | --- |
| `retrieved` + eligible provider/primary-source/verified-child quality | yes, after kind/scope/time/value grounding |
| `retrieved` + `discovery_only` | no |
| `stale` | no |
| `rejected` with explicit rejection code | no |
| tool failure or normalizer failure | no Observation is created |

All successful native and MCP results pass the same atomic `normalization-v1` boundary. The boundary
accepts bounded documented wrappers, aliases, missing optional metadata, and safe primitive coercion
before requiring a finite canonical JSON object. It caps canonical bytes, hashes the canonical projection, assigns an
explicit quality/status, and either emits one complete Observation or a typed failure with zero
observations. Observation v1 rows remain replayable; new rows use `observation-v2`.

Research requirements are intention-sensitive. Ordinary contextual conversation can complete
without tools. A selected generic-conversation or procedural skill never creates an external-evidence
gate; only explicit current/source-seeking intent or a harness-known evidence-bearing skill can do so.
Current market/web claims require eligible cited observations. The thesis challenge requires distinct
market and SEC evidence plus counter-evidence. Explicit plan/delegation requests
must cite a completed parent orchestration observation; a promise to start does not pass. Conflict
requires verifier-owned typed `affected_assumption` and `uncertainty` claims. Missing evidence enters
the bounded correction/replan loop; deterministic failures cannot be overridden by model prose or an
optional semantic judge.

## Durable plans and Slack rendering

Complex work uses append-only plan revisions, DAG/effect/depth/fan-out/budget validation, leased child
nodes, bounded parallel independent reads, dependency context, stale-lease reclaim, bounded replan,
durable partial results, parent cancellation, and parent-only final authority. Replay reuses verified
child evidence instead of repeating provider/model calls.

Renderer v4 consumes only the completed coordinator result. It separates facts, inferences, affected
assumption, uncertainty, safe source links, and disclaimer; strips controls/bidi characters;
neutralizes Slack mentions and markup; redacts credential shapes; and never splits trusted link
markup. Every part uses an immutable outbox key `renderer_version * 1000 + part_index`. All parts are
materialized before the first Slack post. Startup dispatch recovers pending parts, and terminal
reconciliation treats an existing final intent for the same ingress/run as satisfied across renderer
upgrades, preventing cross-version duplicate delivery. Renderer v4 adds the research/financial disclaimer
only when the verified result cites external evidence or contains financial content; a pure internal-
memory answer does not acquire unrelated financial-advice language. Internal run/plan IDs and durable
terminal reasons remain in trace/replay/outbox correlation and are not appended to Slack. Terminal
failures render category-specific conversational recovery without raw status, generic verification
boilerplate, or another exhausted-budget call.

## Executable evidence

The deterministic threat report executes production-boundary regression node IDs, then emits each
threat's expected/actual result and absolute safety counters:

```powershell
.venv\Scripts\python.exe scripts\m4_adversarial_report.py
```

At this checkpoint it executes 17 threat groups (39 parametrized test instances) and reports zero
false-success, forbidden-exposure, scope-leak, unsafe-fetch, and unsafe-delivery counters. It covers
malformed/non-finite/oversized results, stale/rejected/discovery evidence, forged child and Slack
authority, no-tool quote fabrication, unresolved conflicts, 1,000-tool policy filtering,
skill/MCP authority injection, DNS
rebinding/private redirect/active HTML, SEC identity/schema attacks, effectful/cyclic plans, child
completion escalation, and hostile Slack output.

Focused offline gate:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_capabilities.py `
  tests/test_capability_runtime.py tests/test_skills.py tests/test_mcp_adapter.py `
  tests/test_http_integrations.py tests/test_research_adapters.py `
  tests/test_observation_semantics.py tests/test_tool_result_normalization.py `
  tests/test_planning.py tests/test_plan_store.py tests/test_subagent_durable.py `
  tests/test_verifier_grounding.py tests/test_live_composition.py `
  tests/test_slack_render.py tests/test_slack_transport.py `
  tests/test_m4_adversarial_report.py
```

## External acceptance still owned by the live proof

Recorded/local suites prove contracts, not current third-party availability. The current checkpoint
includes a sanitized real Finnhub/Slack read (`run-09e2b402-c0dd-4e1c-93dd-b3997f5599a1`, NVDA
`214.72`, provider as-of `2026-08-21T20:00:00Z`) with an exact cited observation, plus a real SEC/Slack
read (`run-95143cca-afac-4142-aa39-fc69425b2f9b`) whose answer and claim carry the exact primary-source
tuple NVDA, form `8-K`, filing date `2026-08-17`, accession `0001045810-26-000069`. Before declaring
the M4 exit gate complete, retain one fresh Slack parent run that combines both providers with a bounded
child/delegated branch and reaches verified parent completion, plus one non-NVDA delegated/replan Slack
run. Do not store API keys, full provider responses, or inaccessible source content in the evidence.
