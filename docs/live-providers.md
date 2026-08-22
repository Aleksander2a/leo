# Local provider and database verification

Leo separates transport, provider, database, and full Slack checks so failures remain diagnosable.
No command should print secret values.

## Provider-neutral conversational runtime

`OPENROUTER_API_KEY` and `LEO_MODEL` are required for live conversation. The chosen model endpoint
must support tool calling and strict structured output. Leo sends one decision request at a time;
OpenRouter neither executes tools nor owns the loop. The harness validates tool names and arguments,
enforces effects/roles/phases/budgets, records observations, and decides whether completion is true.

Default demo budgets allow multi-step work while staying bounded:

```text
LEO_MAX_MODEL_TURNS=12
LEO_MAX_TOOL_CALLS=24
LEO_MAX_RUN_SECONDS=600
LEO_MAX_OUTPUT_TOKENS=2000
```

Before the first model turn, Leo builds a typed deliberation envelope from context sufficiency,
ambiguity, freshness/evidence needs, dependency structure, effect risk, eligible tool health, and
the remaining budget. The model can answer, clarify, retrieve, call one tool, request independent
reads, or invoke a bounded durable plan/delegation inside that envelope. Prompt length and words such
as `workflow` are not dispatch rules. Independent reads may run concurrently; dependency edges are
executed by the plan/subagent layer. Repeated decisions without a new observation, verifier rejection
without progress, deadline, or budget exhaustion stop safely rather than claiming success.

## Research capabilities

- Wikipedia OpenSearch discovery and public-text fetch are always eligible read capabilities.
  Search results are untrusted URL metadata and deliberately have no source-claim grounding rule;
  the model must fetch a selected result before it can cite retained text.
- Public fetch revalidates each redirect destination, streams only the configured byte cap, allows
  text/HTML/JSON, strips active HTML, records the requested/final URL and redirect count, and marks
  every result untrusted. Empty, truncated, private, unsupported, rate-limited, and unavailable
  responses are typed; truncated text cannot support a completed claim.
- Tavily Search is exposed only when `TAVILY_API_KEY` is configured. It supports bounded basic or
  advanced searches, general/news/finance topics, time or explicit date windows, and small
  include/exclude-domain sets. `include_answer`, raw content, images, and automatic parameter
  expansion are disabled. Tavily titles/snippets are untrusted discovery metadata; Leo must fetch a
  selected public URL before any external source claim can rely on its contents.
- Exa Search is exposed only when `EXA_API_KEY` is configured. Leo calls the raw Search endpoint
  directly with the fixed request shape `query`, `type: auto`, and nested
  `contents: {highlights: true}`. It requests no generated output, second content mode, agent run,
  category, domain, result-count, or freshness override. Only the first structurally complete public
  result is retained; its capped canonical highlights are bound to the exact result URL and digest.
  Incomplete highlights, quota/credit exhaustion, rate limits, timeouts, and schema drift fail with
  typed safe outcomes and never weaken grounding.
- When Exa and Tavily are both configured, open-ended web research can use one provider-family read:
  a complete Exa URL-bound highlight wins; every typed or contained Exa failure falls through once
  to Tavily discovery and complete public fetch. The same Exa call gate is shared across direct and
  family use, so rate-limit cooldowns fail over immediately instead of spending model turns. Natural
  versioned-software questions retain the explicit `web.search_tavily` then
  `web.fetch_public_text` route for primary-document discovery and observable verification.
- Finnhub is exposed only when `FINNHUB_API_KEY` is configured. The catalog includes a current quote,
  Company Profile 2, bounded recent company news, recent reported-versus-estimated earnings, and a
  small whitelist from Basic Financials; additional history reads are advertised only when their
  typed adapter and configured plan support them. Every result preserves endpoint-specific fields, a canonical provider statement or
  source reference, provider/as-of time where available, finite/size validation, and a short evidence
  expiry. The quote adapter's default latest-quote age window is 96 hours so market-close and weekend
  quotes remain representable with an explicit as-of time; callers can tighten it.
- CoinGecko and CoinMarketCap are exposed independently when their keys are configured. The
  provider-neutral `market.get_crypto_snapshot` reads configured peers concurrently, succeeds with
  one fresh exact result, and reports agreement/divergence only when two provider timestamps are
  close enough to corroborate. A time-skewed pair carries an exact non-corroboration caveat. The
  CoinGecko MCP endpoint setting is never treated as a REST root; native reads stay pinned to an
  official `/api/v3` origin. CoinMarketCap's `status.credit_count` is retained as bounded health
  telemetry, not assumed to describe a particular account plan.
- Alpha Vantage, Massive, TickerLayer, and Finnhub participate in the provider-neutral equity
  routes when configured. `market.get_quote` seeks at most two successes across at most four
  providers, preserves exact source/as-of/expiry rows, and reports price disagreement separately
  from timestamp misalignment. Symbol search and company profile use deterministic sequential
  failover. Alpha Vantage is last in general failover to conserve its local 25/UTC-day allowance;
  Massive snapshot and TickerLayer fundamentals permission failures remain provider-local. Every
  TickerLayer quote is labelled derived, indicative, and non-exchange.
- SEC recent filings is exposed only when `SEC_USER_AGENT` is configured. It uses Leo's trusted
  demo ticker-to-CIK map, validates filing arrays and document paths, returns canonical accession
  URLs, and caches each exact ticker/limit read for 15 minutes.
- Delegation and plan tools receive only read capabilities and the parent's pre-authorized context.

Completion remains intention-sensitive without narrowing conversational availability. Ordinary
contextual questions may answer with no tool. Clearly current market/web research cannot complete
without a grounded source claim. A selected thesis-challenge procedure requires distinct current
market and SEC source claims. An explicit execute/parallel/delegate request must execute and cite a
completed parent plan or delegation; a zero-call promise is verifier feedback, not success.

`uv run leo live-quote NVDA` remains a narrow provider/verifier diagnostic. It deliberately requires
one Finnhub observation and an exact grounded claim, but it is not the general Slack dispatcher.

The deterministic offline provider/runtime gate is:

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_http_integrations.py `
  tests/test_research_adapters.py tests/test_exa_integration.py `
  tests/test_crypto_market_tools.py tests/test_equity_provider_expansion.py `
  tests/test_provider_runtime_health.py tests/test_provider_smoke_operator.py `
  tests/test_verifier_grounding.py tests/test_capability_runtime.py `
  tests/test_live_composition.py
```

Recorded fixtures cover Tavily, multiple Finnhub endpoints, SEC submissions, and Wikipedia
OpenSearch. They prove normalized contract parity, not third-party availability. Live public
provider smoke references remain required before calling an individual adapter acceptance case
complete.

After the offline gate, the credentialed operator performs exactly one bounded, independent read per
configured provider and writes only typed status/counters/timestamps/digests. A provider failure does
not abort the cohort, and the artifact has no response text, URL, request, exception, or credential
field:

```powershell
.venv\Scripts\python.exe scripts\run_provider_smoke.py `
  --output artifacts\provider-smoke-v1.json
```

The 2026-08-22 bounded run reached all eight configured providers. Finnhub, Tavily, CoinGecko,
CoinMarketCap, Alpha Vantage, Massive, and TickerLayer returned valid typed results. Exa returned no
structurally complete URL-bound highlights, so Leo recorded the provider-local
`EXA_NO_COMPLETE_HIGHLIGHTS` failure and completed the cohort without admitting that response as
evidence. The content-free artifact SHA-256 is
`905ae752fed83054e3c02e8c7631b84da3b7e11fe8fcee69aeb3bfaead9a7cb7`.

The listener owns one `ProviderGateRegistry`, so every endpoint backed by the same credential shares
health, cooldown, and local call/credit windows across turns. Direct capabilities are not advertised
while their provider is locally exhausted; provider-family tools remain eligible if an admitted
alternate is healthy. The gate is deliberately fail-fast and never sleeps inside an agent run. Its
counters are process-lifetime safeguards and reset on listener restart; do not use them as a billing
ledger or claim cross-restart quota enforcement.

Demo network decision: hostname addresses are checked before every request and redirect, and the
actual transport peer must match that exact public address set. Changed or missing peer metadata
fails closed before response content can become evidence. This prevents evidence/disclosure through
DNS rebinding in the demo, but a generic injected HTTP transport may open its socket before the
post-connect peer check. Production-grade SSRF isolation therefore still needs a resolver-pinned
transport or egress proxy/network sandbox; proxies that hide peer metadata are unavailable.

## Supabase Postgres

Set `DATABASE_URL` to the direct connection or session pooler on port 5432 with TLS, then run:

```powershell
uv run alembic upgrade head
uv run leo durable-quote NVDA
uv run leo replay RUN_ID
```

The forward migrations preserve old compatibility columns while making canonical conversation
identity, exact context-source snapshots, plan/delegation state, and conversation-local memory the
active contracts. Applied historical migrations are never rewritten.

The Postgres contract suite intentionally mutates and cleans Leo-owned synthetic tables:

```powershell
uv run pytest -q -rs tests/postgres
uv run pytest -q
```

Use it only against the designated demo project. The privileged repository worker is server-only;
client roles have no Leo table privileges. That defense remains useful, but production security and
real-user data are outside this demo's scope.

## Full live proof

Start `uv run leo slack-live`, drive the conversation matrix in
[`docs/slack-local.md`](slack-local.md), and inspect Supabase for:

- canonical conversation and Slack ingress rows;
- immutable context conversation IDs plus access hash;
- linked Thread/Task/Run and ordered events;
- observations, verifier results, claims, and parent/child plan state where applicable;
- terminal ingress/run/delivery state.

A provider call, a migration file, or a historical quote reply alone is not acceptance evidence for
the revised design. Primary acceptance is a fresh arbitrary Slack ping whose persisted trace shows
the same exact authority that was injected into the model request.
