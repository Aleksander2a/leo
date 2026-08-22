# Provider extension guide

Leo's provider integrations are small typed ports behind the custom harness. A provider SDK or MCP
transport may simplify HTTP connectivity, but it never owns deliberation, iteration, context,
memory, authorization, budgets, subagents, verification, or completion.

## Runtime layers

| Layer | Responsibility | Must not do |
| --- | --- | --- |
| Native adapter | Validate one request, call one official origin, bound the response, tolerantly extract usable documented variants, normalize one provider result, and return a typed failure | Choose a workflow, retry without a bound, expose credentials, or require one brittle external JSON shape |
| Harness contract | Rebuild canonical statements, validate timestamps/references/digests, and classify evidence quality | Perform network I/O or trust an adapter merely because it returned success |
| Provider-family tool | Select configured healthy providers, fail over or read peers concurrently, retain attempt accounting, and calculate agreement/freshness | Hide disagreement, silently broaden authority, or let one provider exception abort the family |
| Capability descriptor/catalog | Advertise identity, schema, tags, health, freshness, cost, role, phase, and verification expectations | Grant permissions or advertise an unavailable credential/tier |
| Coordinator and verifier | Admit calls inside the run envelope, persist observations, enforce evidence, and decide terminal truth | Delegate loop ownership to a provider SDK, MCP, or agent framework |

The executable tool name, catalog ID, normalization kind, and verifier rule form one versioned
identity. Changing a request/response contract or evidence meaning requires a semantic-version bump
and fixture update; changing only a provider's health does not.

## Add a native provider

1. Add a `pydantic-settings` credential field and conservative local limits. Keep configurable MCP
   URLs separate from REST bases. Secrets use `SecretStr` and never enter model context, events,
   artifacts, exceptions, or logs.
2. Implement one adapter in `src/leo/integrations/`. Pin credential-bearing requests to the exact
   official HTTPS host, port, and base path; reject user-info, query/fragment-bearing bases, and
   redirects. Set finite connect/read/write/pool timeouts and a pre-JSON byte cap.
3. Validate bounded arguments before acquiring a call slot. Use the runtime-owned
   `ProviderGateRegistry` so every endpoint backed by the same credential shares concurrency,
   call/credit accounting, cooldown, and health across Slack turns in that listener process.
4. Treat the wire response as a tolerant boundary: accept documented wrappers/aliases, absent
   optional metadata, and safe string/number coercion; ignore unknown bounded fields. Return
   `ToolSuccess` when enough finite, fresh data exists to build the canonical internal observation.
   Map authentication, entitlement, quota, rate-limit, timeout, oversize, materially unusable schema,
   stale, and unavailable outcomes to
   content-free typed `ToolFailure` codes. Contain unexpected provider exceptions at the adapter or
   provider-family boundary; cancellation must still propagate.
5. Put stable canonicalization and provenance validation in `src/leo/harness/`, not brittle wire-shape
   assumptions. A connected integration is accepted as provider-attributed evidence after origin,
   authority, size, and safety checks; its content still cannot grant authority or instructions.
   Bind every claim to provider identity, normalized request identity, exact reference, observed/as-of
   time, expiry, and a deterministic digest. Provider-generated summaries are untrusted unless a
   dedicated contract says otherwise.
6. Export a `ProviderCapabilityDescriptor`, register the executable tool, and add objective-friendly
   tags and aliases. Direct tools become ineligible during their provider's cooldown/exhaustion;
   aggregate tools stay eligible while at least one admitted alternate remains.
7. Add the tool's observation normalization and deterministic verifier rule. A discovery-only search
   result cannot support a claim; require a URL-bound extract or a complete validated fetch.
8. Add it to the content-free provider smoke only after a single, entitlement-conservative probe is
   available. One provider smoke failure is a report row, never a cohort abort.

## Add redundancy

Use a provider-family tool when multiple providers answer the same normalized question. The family
contract should state:

- deterministic provider order and a hard call/concurrency bound;
- whether it stops after one success or seeks a fixed corroboration target;
- which typed failures permit fallback and which argument/policy failures terminate the family;
- exact attempted, failed, health-skipped, quota-skipped, and uncalled rows;
- selection policy and the source/as-of/expiry retained from every successful peer;
- a numerical agreement rule, maximum timestamp skew, and canonical disagreement/time-skew caveat;
- partial-success behavior and one typed all-providers-unavailable result.

Do not average unlike observations. A delayed close and a live quote can both be useful, but they are
not corroboration unless their timestamps fall inside the declared comparison window. The aggregate
expiry is no later than the earliest contributing peer expiry.

## Add an MCP capability

MCP is a transport and discovery protocol, not an authority boundary. Admit a server/tool only after
pinning its configured identity and transport, completing protocol discovery, and mapping the remote
schema into the same local `ToolSpec`, effect, role, phase, cost, timeout, normalization, freshness,
and verifier contracts used by native tools. Treat server annotations and returned content as
untrusted. Server prompts, sampling, roots, elicitation, or write effects remain disabled unless a
separate product decision and adversarial test explicitly admits them.

## Required tests

Every provider addition needs deterministic tests for:

- official-origin credential egress, redirect refusal, malformed base URLs, and secret redaction;
- exact request shape and argument bounds;
- success, empty/malformed/oversized payload, non-finite values, stale/future time, and schema drift;
- 401/402/403 entitlement, 429 cooldown, bounded `Retry-After`, timeout, and 5xx handling;
- shared endpoint health/quota state, limit rollover, policy drift, and process-restart disclosure;
- unexpected-exception containment, partial success, all-failed behavior, and no hidden retry;
- normalization quality, canonical statement reconstruction, provenance tamper, and claim rejection;
- catalog discovery for short natural prompts and exclusion when unavailable or unhealthy;
- provider-family agreement, disagreement, timestamp skew, deterministic selection, and exact expiry;
- one-request credentialed smoke with a content-free atomic artifact.

Run provider tests without credentials by passing `Settings(_env_file=None, ...)` or explicitly
clearing every optional provider field in the fixture. Tests must never inherit a developer `.env`.

## Operational limits

The current call gates are listener-process scoped. They share health across turns and stop retries
during cooldown, but their counters reset on listener restart. That is an explicit soft safety layer,
not a billing ledger. Provider-reported 402/403/429 outcomes remain fail-closed. If a future plan
requires a hard cross-restart allowance, add a transactional server-side quota ledger keyed by
provider, credential digest, UTC window, and charge unit before advertising that guarantee.
