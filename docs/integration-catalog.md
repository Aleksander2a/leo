# Research integration catalog

This catalog separates implemented contracts from candidates. An adapter being implemented does not
mean that its provider account, quota, or endpoint tier has been live-smoked. Leo exposes an optional
adapter only when its server-side credential is configured, and records provider availability as a
runtime fact rather than assuming it from documentation.

## Implemented read-only tools

| Tool ID | Provider endpoint | Bounds and freshness | Evidence semantics |
| --- | --- | --- | --- |
| `web.search_exa` | [Exa Search guide for coding agents](https://exa.ai/docs/reference/search-api-guide-for-coding-agents) | Fixed `POST https://api.exa.ai/search`; raw HTTP; fixed `type: auto`; bounded highlight content; first usable public result; at most three retained highlights; 10-minute TTL. Optional score metadata may be absent. | Each admitted canonical highlight is bound to the selected result's exact public URL and result digest. Tolerant parsing accepts documented response wrappers and missing optional scores; a result still needs usable URL/content. Exa Agent and third-party agent frameworks are not used. |
| `web.research_verified` | Exa Search with bounded Tavily + public-fetch fallback | One Exa attempt followed, on any typed or contained adapter failure, by at most one Tavily discovery and one public-fetch call with up to four candidate-local fallbacks | Returns only exact-URL Exa highlights or complete retained public text. The attempt ledger is bounded and secret-free; Tavily snippets never cross the family boundary as claim evidence. |
| `web.search_tavily` | [Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search) | Fixed `https://api.tavily.com/search` host; 1-5 results; bounded depth, topic, time/date, and include/exclude-domain filters; 10-minute TTL | Discovery only. Generated answers, raw content, and images are disabled. A result URL must be selected and fetched before its contents can support a source claim. |
| `market.get_company_profile` | [Finnhub Company Profile 2](https://finnhub.io/docs/api/company-profile2) | One symbol; bounded identity/listing fields; 24-hour TTL | Exact identity/listing statement may support a source claim. Numeric market-cap/share fields remain inference-only. |
| `market.get_company_news` | [Finnhub company news](https://finnhub.io/docs/api/company-news) | Server-derived 1-30 day window; 1-10 items; rejects invalid, future, and out-of-window items; 15-minute TTL | Each canonical headline statement is bound to the exact item source and public URL. |
| `market.get_earnings_surprises` | [Finnhub company earnings](https://finnhub.io/docs/api/company-earnings) | One symbol; at most four valid, non-future periods; 6-hour TTL | Exact period/actual/estimate/surprise statement may support a source claim. |
| `market.get_basic_financials` | [Finnhub basic financials](https://finnhub.io/docs/api/company-basic-financials) | One symbol; fixed metric set; 6-hour TTL | Only whitelisted finite metric/value statements may support a source claim. |
| `market.get_crypto_snapshot` | CoinGecko + CoinMarketCap provider family | One asset slug and quote currency; at most one call to each configured peer; up to 3-minute evidence TTL; partial success allowed | Retains each peer's exact identity, symbol, price, provider timestamp, expiry, and reference. Two peers produce agreement/divergence only inside the declared timestamp-skew window; otherwise the required statement says they are time-skewed and non-corroborating. |
| `market.get_crypto_snapshot_coingecko` | [CoinGecko `/coins/markets`](https://docs.coingecko.com/reference/coins-markets) | Exact Demo/Pro REST host and `/api/v3` base; one asset; 1 MiB response cap; shared cooldown/health gate | Exact provider-reported name, symbol, price, as-of time, and reference may support a canonical claim. A configured CoinGecko MCP URL is never reinterpreted as the REST base. |
| `market.get_crypto_snapshot_coinmarketcap` | [CoinMarketCap `/v2/simple/price`](https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency) | Exact Pro API origin; one slug/currency; 1 MiB response cap; shared cooldown/health gate | Exact provider-reported numeric ID, slug, name, symbol, price, timestamp, and credit telemetry are retained. Provider credits are observed, not treated as a known account entitlement. |
| `market.get_quote` | Finnhub alone, or the Finnhub + Massive + TickerLayer + Alpha Vantage provider family | One normalized symbol. With peers configured: deterministic failover, at most four calls and two successes, up to 15-minute evidence TTL, selected result freshest then provider order | Exact canonical price statement. The redundant shape preserves every successful provider reference/as-of/expiry plus failures and health skips. Agreement requires both price proximity and timestamp alignment; divergence and time-skew each require an exact caveat. |
| `market.search_equity_symbols` | Massive, TickerLayer, and Alpha Vantage reference/search endpoints | Deterministic sequential failover; up to ten bounded matches | Exact provider-attributed symbol/name statements; a search match never silently becomes a live quote. |
| `market.get_equity_profile` | Finnhub, Massive, TickerLayer, and Alpha Vantage company metadata | Deterministic sequential failover; one normalized symbol; 24-hour family TTL | Exact provider-attributed name/listing/industry statement. Provider-specific permission failures remain local and allow an admitted alternate. |
| `market.get_quote_alpha_vantage`, `market.search_symbols_alpha_vantage`, `market.get_company_profile_alpha_vantage` | [Alpha Vantage `GLOBAL_QUOTE`, `SYMBOL_SEARCH`, `OVERVIEW`](https://www.alphavantage.co/documentation/) | Exact `https://www.alphavantage.co/query`; shared conservative 5/minute and 25/UTC-day gate | Quote timestamps and the provider's default end-of-day semantics are explicit. Search/profile claims are provider-attributed and reference-bound. |
| `market.get_quote_massive`, `market.search_symbols_massive`, `market.get_company_profile_massive` | [Massive unified snapshot and ticker reference](https://massive.com/docs/rest/stocks) | Exact `https://api.massive.com`; Bearer auth; reference endpoints are catalogued separately from plan-dependent snapshots | A snapshot entitlement failure is typed and can fail over. Ticker reference/search remains usable without pretending the configured plan includes live snapshots. |
| `market.get_quote_ticker_layer`, `market.search_symbols_ticker_layer`, `market.get_company_profile_ticker_layer` | [TickerLayer Stocks REST](https://tickerlayer.com/docs/rest/stocks) and [Fundamentals](https://tickerlayer.com/docs/rest/fundamentals) | Exact `https://api.tickerlayer.com`; market-qualified `CC:SYMBOL`; shared local 3,000/UTC-month gate; fundamentals permission is separate | Quote statements explicitly label derived, indicative, non-exchange data. Market and permission failures are typed and provider-local. |
| `sec.get_recent_filings` | [SEC submissions data](https://www.sec.gov/edgar/sec-api-documentation) | One supported ticker; bounded recent filing tuples; primary-source freshness policy | Exact normalized filing metadata and derived SEC document URL may support a source claim. |
| `web.search_public` / `web.fetch_public_text` | Wikipedia OpenSearch plus bounded public HTTPS fetch | Small result set; public-address validation; byte/time limits | Search is discovery only. A fresh, complete selected page fetch is untrusted retrieval that must satisfy its grounding rule. |

All endpoints backed by the same configured provider credential share one runtime-owned call gate for
bounded concurrency, local call/credit windows, health, and cooldown across Slack turns. Direct tools
become ineligible while their provider is locally exhausted; a provider-family tool stays available
while an admitted alternate remains. These counters are process-lifetime safeguards and reset when the
listener restarts; they are not represented as a durable billing ledger.

The public `market.get_quote` identity is always the provider-neutral route, including its truthful
one-provider shape, and configured Finnhub also exposes `market.get_quote_finnhub`. For profile
compatibility, a Finnhub-only runtime keeps `market.get_company_profile`; when another profile
provider is configured, the family becomes `market.get_equity_profile` and Finnhub is exposed as
`market.get_company_profile_finnhub`.

Historical Finnhub candles are not
currently catalogued: Finnhub's official API documentation marks stock candles as requiring premium
access, and the configured demo account/plan has not been live-smoked for that endpoint. This is an
explicit capability absence, not a silent fallback.

## Ranked read-only backlog

1. **OpenAlex and Crossref scholarly metadata.** OpenAlex exposes REST entity endpoints and currently
   recommends an API key for a larger usage budget; [current pricing and anonymous/key budgets are
   mutable](https://help.openalex.org/access/pricing/). Crossref's public REST API asks clients to use a
   descriptive `User-Agent` and `mailto` parameter in its [access guidance](https://www.crossref.org/documentation/retrieve-metadata/rest-api/access-and-authentication/).
   Treat search results as discovery/provider metadata, then resolve a DOI or fetch an eligible primary
   work before making claims about a paper's contents.
2. **FRED and World Bank macro data.** [FRED requires an API key](https://fred.stlouisfed.org/docs/api/fred/v2/api_key.html)
   and exposes REST series/observation data. The [World Bank Indicators API v2](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)
   requires no authentication. Preserve series/indicator IDs, period, units, vintage/update metadata,
   and source organization; these are provider-reported aggregations, not automatically primary data.
3. **GitHub public repositories and releases.** Public REST reads can be unauthenticated but have lower
   limits; [authentication changes endpoint access and rate limits](https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api).
   Start with bounded repository, release, commit, issue, and raw-file reads. Exact API metadata can be
   provider-reported; file contents require a content digest and repository/ref binding.
4. **Open-Meteo current weather and forecasts.** The [forecast API documentation](https://open-meteo.com/en/docs)
   describes model-derived time series. Treat all values as provider-reported forecasts, carry model,
   coordinates, units, issue/valid time, and a short TTL. Confirm acceptable-use and live endpoint
   limits before enabling; do not infer commercial entitlement from the public documentation.
5. **Optional MCP servers.** MCP standardizes capability negotiation and tools, but transport does not
   establish evidence trust. The [MCP tool specification](https://modelcontextprotocol.io/specification/draft/server/tools)
   requires clients to treat annotations from untrusted servers as untrusted. Each admitted server/tool
   still needs server-bound authorization, an effect classification, a bounded schema, normalizer,
   freshness policy, provenance mapping, verifier rule, and failure taxonomy before catalog exposure.

## Admission rule

A candidate moves into the implemented catalog only after deterministic adapter, normalization,
grounding, malformed-response, rate-limit, freshness, and catalog-selection tests pass. Live provider
availability/tier is recorded separately by a credential-safe smoke; absence or quota failure removes or
fails that capability explicitly and never weakens the verifier.

Provider-wire schemas are deliberately tolerant. Adapters may recognize documented aliases/wrappers,
omit unknown bounded fields, accept missing optional metadata, and safely coerce primitive values.
They then emit Leo's canonical internal observation. A provider-local unusable response, quota, or
entitlement failure is isolated and does not abort a redundant family or unrelated capabilities.

See [the provider extension guide](provider-extension-guide.md) for the adapter, provider-family,
catalog, verification, MCP, and adversarial-test contract used to add another integration without
changing Leo's custom loop.
