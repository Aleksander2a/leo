# Native cryptocurrency provider integration

Leo calls CoinGecko and CoinMarketCap through small `httpx` adapters. No agentic
framework or provider SDK participates in planning, routing, or evidence verification.

## Official contracts used

- CoinGecko `GET /coins/markets`, with the Demo key in the
  `x-cg-demo-api-key` header (or `x-cg-pro-api-key` for the Pro host):
  <https://docs.coingecko.com/reference/coins-markets>
- CoinGecko REST/Demo delivery and authentication:
  <https://docs.coingecko.com/docs/data-delivery-methods>
- CoinMarketCap `GET /v2/simple/price`, with the key in the
  `X-CMC_PRO_API_KEY` header:
  <https://coinmarketcap.com/api/documentation/pro-api-reference/cryptocurrency>
- CoinMarketCap authentication, response envelope, credits, and rate-limit behavior:
  <https://coinmarketcap.com/api/documentation/guides/authentication>
  <https://coinmarketcap.com/api/documentation/guides/standards-and-conventions>
  <https://coinmarketcap.com/api/documentation/guides/errors-and-rate-limits>

Each call requests exactly one provider-common asset slug and one quote currency. The
CoinMarketCap response's numeric ID is retained in provenance even though the shared
input uses a slug; the adapter rejects a response whose returned slug does not exactly
match the requested asset.

## Resilience and evidence

`ProviderCallGate` is shared per provider credential. It bounds concurrency, calls per
minute, optional calls per UTC day/month, and rate-limit cooldowns. It also exposes
CoinMarketCap's provider-reported `credit_count` in health on both successful and failed
responses. That counter is observability, not a claim that Leo knows or enforces the
account's monthly credit entitlement; no CMC credit ceiling is assumed from an unknown
account tier. A 429 starts a bounded fail-fast cooldown, so Leo does not sleep inside the
agent loop or spend more provider quota retrying an already-limited service.

`market.get_crypto_snapshot` calls all configured crypto providers concurrently and:

- succeeds when at least one provider returns a fresh, schema-valid result;
- retains sanitized typed failure codes for unavailable providers;
- computes an exact midpoint-relative price spread in basis points when two succeed;
- labels an aligned pair `agreement` or `divergence` against a configured threshold;
- labels observations beyond the maximum provider-timestamp skew `time_skewed`, marks them
  non-corroborating, and emits an exact verifier-enforced caveat instead of comparing unlike times;
- binds provider references, timestamps, values, failure accounting, and the comparison
  into a deterministic SHA-256 provenance digest;
- exposes only canonical statements accepted by the deterministic verifier.

An unexpected adapter exception is converted into a provider-local content-free failure,
so one provider can never abort the aggregate call. If every provider fails, the aggregate
returns one typed `CRYPTO_PROVIDERS_UNAVAILABLE` result and leaves all other Leo tools
available.

Oversized responses are rejected before JSON decoding. A 429 is classified first so its bounded
`Retry-After` cooldown cannot be hidden by an oversized error body; provider credits from an
unparsed body remain zero rather than being guessed.

## Extension seam

To add a crypto provider:

1. Implement `CryptoSnapshotProvider` and emit `CryptoProviderPayload`.
2. Reuse a runtime-owned `ProviderCallGate` from `ProviderGateRegistry`.
3. Add the provider to `build_crypto_market_tools` and to the deterministic provider order.
4. Export discovery metadata through a `ProviderCapabilityDescriptor`.
5. Add success, schema drift, stale timestamp, 429/cooldown, exception containment,
   partial-success, all-failed, provenance-tamper, and verifier tests.

The custom harness continues to own tool selection, budgets, context, subagents, and
verification; provider adapters only perform bounded reads and normalization.
