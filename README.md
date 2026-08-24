<p align="center">
  <img src="assets/leo_logo.png" alt="Leo logo" width="180">
</p>

<h1 align="center">Leo</h1>

<p align="center">
  An AI portfolio-research agent that lives in Slack
</p>

<p align="center">
  <a href="LICENSE">Apache-2.0</a> · Python 3.12 · Slack Socket Mode · PostgreSQL + pgvector
</p>

Leo answers questions in Slack. Ask it about a market, a company, a strategy, or anything
else, and it reasons about what it needs, calls the tools that will get it, reads what came
back, and writes you an answer.

That is the whole design, and it is deliberate. An earlier version of this repository had a
runtime that could refuse to speak: a verifier that graded answers, a "deliberation
envelope" that vetoed the model's chosen depth, a completion contract the model had to
satisfy before it was allowed to reply, and keyword tables that decided which tools a
question "really" needed. Asked *"what high yield crypto strategy should I adopt now?"*, that
system called five different equity-quote tools for the symbol "BTC", failed its own
no-progress check, and told the user the reasoning service had stopped unexpectedly.

None of those layers exist any more. The model is the agent; the harness serves it.

## Contents

- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Slack setup](#slack-setup)
- [Memory](#memory)
- [Tools](#tools)
- [Database](#database)
- [Dashboard](#dashboard)
- [Configuration](#configuration)
- [Testing](#testing)
- [Repository layout](#repository-layout)
- [Operating boundaries](#operating-boundaries)
- [License](#license)

## How it works

One turn, start to finish:

1. **Context.** Leo loads the conversation's recent history and the memories most relevant
   to the question, both filtered to this channel or DM.
2. **Tool selection.** The question is embedded and compared against every tool's own
   description. The most relevant tools open the turn, alongside a small always-available
   core (memory, web search, web fetch). Nothing is hidden permanently — `tools.find`
   searches the same index and adds what it finds to the live set.
3. **Reason and act.** The model either calls tools or writes its answer. Tool calls run in
   parallel, results come back as ordinary messages, and the model decides what to do next.
4. **Answer.** When the model has enough, it writes the reply. Leo converts it to Slack's
   mrkdwn, splits it if it is long, and posts it in the thread.

Three rules hold the whole thing together:

**No tool outcome ends a run.** A missing tool, bad arguments, a provider outage, a timeout,
an oversize payload — each comes back to the model as a `tool` message describing what went
wrong, so it can try another source or answer with what it has. `leo/agent/tools.py` has no
code path from a tool problem to a failed turn.

**The harness never writes an answer.** When the turn budget runs out, the model gets one
final call with tools withdrawn and an instruction to answer with what it gathered. That
answer is still the model's. There is no fallback text that asserts facts, because a harness
that authors claims is a harness that fabricates them.

**Isolation is a WHERE clause.** Every durable row carries a `scope_key` — `slack:T123:C456`
for a channel, `slack:T123:D789` for a DM. Reads filter on it in SQL, so a DM's memories are
not ranked below a channel's, they are absent from the query.

## Quick start

### Requirements

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- PostgreSQL with the `pgvector` extension (Supabase works out of the box)
- An OpenRouter API key and a tool-capable model
- Slack app credentials, for the Slack surface

Provider credentials (web search, market data, SEC) are all optional. Leo advertises whatever
is configured and works without the rest.

### Install

```bash
uv python install 3.12
uv python pin 3.12
uv sync --locked --dev
cp .env.example .env
```

Fill in `.env`, then create the schema:

```bash
uv run alembic upgrade head
```

### Try it

```bash
uv run leo ask "what high yield crypto strategy should I adopt now?" --trace
```

`--trace` prints each tool call as it runs. For a back-and-forth session:

```bash
uv run leo chat
```

And to check that everything is wired up:

```bash
uv run leo health
```

## Slack setup

Create a Slack app from [`slack/manifest.yml`](slack/manifest.yml) — it declares the scopes
and the two events Leo listens for (`app_mention` and `message.im`). Install it to your
workspace, then put the bot token (`xoxb-…`) and app-level token (`xapp-…`) in `.env`.

```bash
uv run leo slack
```

Leo replies when mentioned in a channel and to any direct message. Channel replies open a
thread on the message so long answers do not flood the channel; replies inside a thread stay
in that thread. While it works, it posts a placeholder and updates it with the tools it is
calling, then replaces it with the answer.

## Memory

Leo remembers per conversation. What it learns in a DM stays in that DM.

Three tools, all scoped:

| Tool | What it does |
| --- | --- |
| `memory.search` | Finds memories by meaning, using pgvector cosine similarity. |
| `memory.write` | Stores a durable fact, preference, decision, or constraint. |
| `memory.forget` | Retires a memory that is no longer true. |

Updates supersede rather than overwrite: `memory.write` with `supersedes` set marks the old
row inactive and points it at its replacement, so what Leo used to believe stays inspectable.
Recall is semantic, so "what's my risk tolerance?" finds "never wants more than 15%
drawdown" without sharing a keyword. If the embedding provider is down, recall falls back to
the scope's most important memories rather than reporting amnesia.

Inspect or prune from the terminal:

```bash
uv run leo memory list --scope slack:T123:D456
```

## Tools

Leo carries around thirty tools, depending on which credentials are configured:

- **Web** — Tavily and Exa search, a public-page fetcher, Wikipedia lookup, and a
  search-then-fetch route that fails over between providers.
- **Crypto** — CoinGecko and CoinMarketCap snapshots, plus a corroborated aggregate that
  reports agreement and time skew between them.
- **Equities** — quotes, profiles, and symbol search across Finnhub, Alpha Vantage, Massive,
  and TickerLayer, with provider-neutral routes that fail over.
- **Fundamentals** — company news, earnings surprises, basic financials.
- **SEC** — recent EDGAR filings for any registered ticker, resolved from SEC's own index.
- **MCP** — Tavily, Alpha Vantage, and CoinGecko over MCP, alongside the REST adapters.

Adding one is a single class and one line in `build_tools` — see
[docs/adding-a-tool.md](docs/adding-a-tool.md). There is no catalogue to register with, no
keyword tags to maintain, and no routing table to update: the tool's own description is what
makes it discoverable.

## Database

Six tables, all keyed by `scope_key`:

| Table | Holds |
| --- | --- |
| `agent_conversations` | One row per channel or DM. |
| `agent_messages` | What the user asked and what Leo answered. |
| `agent_runs` | One row per request: status, answer, tokens, cost. |
| `agent_steps` | The ReAct trace — every model turn and tool call, in order. |
| `agent_memories` | Durable facts with their embeddings. |
| `agent_tool_index` | Cached tool-description embeddings for discovery. |

A run's raw tool traffic lives in `agent_steps`, not in `agent_messages`. Replaying old tool
JSON into later prompts is how a context window fills with noise; the next turn reads prior
*answers*.

## Dashboard

A read-only FastAPI surface over the same tables:

```bash
uv run python scripts/run_dashboard_api.py
```

- `GET /health` — configuration and (with `?deep=true`) database reachability
- `GET /dashboard/overview` — run counts, answer rate, tokens, cost, tool usage
- `GET /dashboard/runs` and `/dashboard/runs/{id}` — runs and their full step trace
- `GET /dashboard/conversations`, `/dashboard/memory`, `/dashboard/tools`, `/dashboard/failures`

> The Next.js app in `web/` still targets the previous runtime's API shape and has not been
> ported to these endpoints.

## Configuration

Everything is environment variables; see [`.env.example`](.env.example) for the full list.

| Variable | Purpose |
| --- | --- |
| `LEO_MODEL` | Any tool-capable OpenRouter model. |
| `OPENROUTER_API_KEY` | Model and embedding access. |
| `DATABASE_URL` | PostgreSQL with `pgvector`, direct or session-pooler (port 5432). |
| `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN` | Required only for the Slack surface. |
| `LEO_MAX_MODEL_TURNS` | Reason/act rounds per question (default 12). |
| `LEO_MAX_TOOL_CALLS` | Tool calls per question (default 24). |
| `LEO_MAX_RUN_SECONDS` | Wall-clock budget per question (default 600). |

Provider keys — `TAVILY_API_KEY`, `EXA_API_KEY`, `FINNHUB_API_KEY`, `COINGECKO_API_KEY`,
`COIN_MARKET_CAP_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `MASSIVE_API_KEY`,
`TICKER_LAYER_API_KEY`, `SEC_USER_AGENT` — are each optional and additive.

## Testing

```bash
uv run pytest tests -q          # unit and integration
uv run python scripts/quality.py  # the full gate: lint, types, tests, build
```

Tests that need a database skip themselves when `DATABASE_URL` is unset. The quality gate
also runs a secret scan, a layering check (the agent may not import a transport; providers
may not import the loop), and a clean-install smoke test of the built wheel.

## Repository layout

```
src/leo/
  agent/          the agent
    loop.py       reason, act, observe, answer
    tools.py      the registry -- every outcome returns to the model
    llm.py        OpenAI-compatible chat and embeddings
    memory.py     scope-isolated recall, write, supersede
    discovery.py  semantic tool selection and tools.find
    prompts.py    the system prompt
    store.py      conversations, history, runs, traces
    db.py         engine, sessions, and the event loop psycopg needs
    schema.py     six tables
    runtime.py    composition root
  slack/          Socket Mode transport and mrkdwn rendering
  integrations/   provider tool adapters (HTTP and MCP)
  providers/      provider-domain normalization, pure functions
  api/            read-only dashboard API
migrations/       one baseline migration
tests/            the suite
```

## Operating boundaries

Leo's market tools are **read-only research**. There is no trading, no order placement, and
no external write capability anywhere in the tool set. Answers about markets are research
information, not financial advice.

Leo reads only the conversation it is speaking in. It has no access to other channels, other
DMs, or any Slack history beyond what it has itself recorded in the current scope.

## License

[Apache-2.0](LICENSE).
