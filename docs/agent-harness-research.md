# Leo elastic harness: research basis and design translation

This note records which outside ideas inform Leo and, more importantly, how they are translated into
bounded product contracts. Leo remains a custom harness. The references are design inputs, not runtime
dependencies and not claims that a paper's benchmark result transfers to this project.

## What Leo adopts

| Source pattern | Useful idea | Leo translation |
|---|---|---|
| [ReAct](https://arxiv.org/abs/2210.03629) | Interleave reasoning, actions, and observations so new evidence can change the next step. | A typed model decision chooses direct completion, clarification, or advertised tool calls; every result becomes a normalized observation before the next turn. Private chain-of-thought is neither requested nor persisted. |
| [Plan-and-Solve](https://arxiv.org/abs/2305.04091) | Decompose genuinely multi-step problems before executing them. | The depth policy creates a durable dependency plan only when decomposition, dependency, parallelism, uncertainty, or risk justifies it. Users do not need to say “make a plan.” |
| [Toolformer](https://arxiv.org/abs/2302.04761) | A model should decide whether, when, and how to call tools. | The model proposes calls, while the harness constrains eligible tools, exact schemas, phase, effect, authority, health, cost, and call budgets. |
| [Reflexion](https://arxiv.org/abs/2303.11366) and [Self-Refine](https://arxiv.org/abs/2303.17651) | Feedback can improve a subsequent attempt without retraining. | Deterministic verifier feedback can trigger a bounded correction or plan revision. Failed checks and progress deltas are durable; unconstrained self-critique loops are not. |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) and [LATS](https://arxiv.org/abs/2310.04406) | Hard tasks may benefit from branching, evaluation, feedback, and backtracking. | Leo may compare a small bounded set of candidate subplans or replan after evidence failure. It does not run open-ended tree search; branch, depth, elapsed-time, cost, and no-progress caps remain harness-owned. |
| [OpenAI Agents SDK orchestration](https://openai.github.io/openai-agents-python/multi_agent/) | A manager can retain the final conversation while specialists run as tools; independent work can run concurrently. | Leo's parent owns synthesis, verification, rendering, and terminal truth. Child agents inherit exact conversation authority, receive least-needed context/tools and separate budgets, and cannot post to Slack. Leo does not depend on the SDK's orchestration loop. |
| [OpenAI Agents SDK tracing](https://openai.github.io/openai-agents-python/tracing/) | Model turns, tools, handoffs, guardrails, and custom events should be observable. | Leo persists typed content-bounded events, context manifests, observations, claims, plan revisions, child links, budget usage, and delivery receipts with redacted replay/export. |
| [LangGraph durable execution](https://docs.langchain.com/oss/python/langgraph/functional-api) and [subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs) | Checkpoint task results, isolate per-invocation subgraphs, and resume without repeating completed work. | Leo keeps its own Postgres event/plan/lease model, but adopts the same explicit checkpoint, idempotency, child-namespace, and restart tests. It does not add LangGraph as a runtime dependency. |
| [AutoGen termination conditions](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/termination.html) | Multi-agent conversations need stateful, composable stop conditions instead of relying on an agent to decide when to stop. | Leo's harness owns terminal truth, no-progress, deadline, budget, dependency, and verifier gates. Specialist dialogue cannot extend itself or complete the parent. |
| [Google ADK workflow agents](https://google.github.io/adk-docs/api-reference/java/com/google/adk/agents/package-use.html) | Sequential, parallel, and loop agents are distinct orchestration primitives. | Leo exposes those execution shapes through one typed durable plan: dependencies imply sequence, ready independent reads may run in parallel, and correction loops are bounded by progress and verifier deltas. |
| [Model Context Protocol](https://modelcontextprotocol.io/specification/2025-06-18) | Standard capability negotiation and structured tools/resources make integrations composable. | Official MCP v2 is a narrow adapter behind the same catalog, authority, schema validation, normalization, timeout, and verifier boundary as native tools. Tool annotations from untrusted servers cannot grant authority. |

## Elastic depth policy

Prompt length and trigger words are poor complexity measures. Leo evaluates a typed set of factors:

- Is the authorized thread/context sufficient, or is one material fact missing?
- Is the request ambiguous enough that one clarification has higher value than guessing?
- Does the answer depend on current/external facts, and how fresh must they be?
- How many independent claims and distinct sources are required?
- Can work be decomposed, and are any nodes dependent or safely parallel?
- What is the uncertainty or contradiction level after each observation?
- Does the next step read, prepare, or cause an external effect?
- Which eligible capabilities are healthy and affordable within the remaining run budget?
- Did the last step reduce an explicit open question, satisfy evidence, or complete a dependency?

The model proposes a mode and next action. Deterministic policy can raise a minimum (for example,
fresh source evidence for a current quote), lower a maximum (for example, no tools after an explicit
tool-free request), or reject a call. The runtime may escalate from context to retrieval to research or
from a single call to a plan only when observations justify it. It de-escalates and synthesizes as soon
as completion requirements are satisfied.

The intended modes are:

1. direct contextual answer;
2. one concise clarification;
3. scoped thread/context or memory retrieval;
4. one read tool;
5. multi-source research with independent parallel reads;
6. durable dependency plan and bounded specialist children;
7. verifier correction or bounded replan;
8. verified answer or conversational non-success outcome.

## Context and memory boundary

The complete authorized Slack thread is the conversational source set. Finite context windows are
handled with source-linked hierarchical compaction, not silent recent-message truncation. Root,
decisions, corrections, unresolved questions, recent turns, and Leo's latest material or final outcomes
are pinned. Generic progress chatter may be compacted; summarized ranges retain provenance and can be
reopened through run-bound handles.

Shared channel, private-channel, shared-external, and group-DM turns can read only their exact
conversation (and current thread). A 1:1 DM may additionally retrieve relevant background from the
current user's exact membership intersection with conversations where Leo is also present in the same
workspace. Authorization is applied before ranking and every item keeps its source conversation.

## Search and tool trust

[Tavily Search](https://docs.tavily.com/documentation/api-reference/endpoint/search) supports bounded
depth, topic, date, domain, result, and chunk controls. Leo disables Tavily's generated answer and
normalizes individual results. Titles and snippets remain untrusted discovery metadata; Leo must fetch
and validate a selected public page before any external source claim can rely on its contents. Tavily
is optional—missing credentials remove the capability, not Leo.

[Exa Search](https://exa.ai/docs/reference/search) is an independent optional search capability.
Leo uses its raw ZDR Search request rather than Exa Agent or a third-party agent framework, fixes the
mode to `auto`, and selects highlights as the sole content mode. A highlight becomes eligible only
when its normalized public URL and content are usable and carry query/result digests. Title and score
metadata are optional. Tolerant wire parsing accepts documented wrappers and missing optional fields,
then emits a stable canonical provider-attributed result; materially incomplete results fail locally.

Provider selection is objective-sensitive rather than globally Exa-first. Versioned software changes
keep an explicit Tavily discovery-to-public-fetch chain. Other admitted web-research objectives can
use the bounded provider family, which tries Exa and then Tavily plus fetch without another semantic
model turn. Search-provider errors and unexpected adapter exceptions become safe provider-local
attempt codes; discovery snippets remain ineligible even during fallback.

The [official Finnhub Python client](https://github.com/Finnhub-Stock-API/finnhub-python) documents
quote, profile, company news, basic financials, earnings, peers, candles, and other endpoints. Leo adds
only endpoints with bounded inputs, stable typed outputs, explicit provider time/freshness, and a
deterministic grounding rule. Tool breadth is exposed through policy-first progressive discovery rather
than placing every JSON schema in every prompt.

MCP tools follow the protocol's requirements for input validation, access control, rate limits,
sanitized output, timeouts, and audit. Provider content is untrusted data even when the transport or
server is trusted.

## Deliberate non-goals through M5

- No hidden chain-of-thought persistence or user-facing “reasoning trace.” Leo records concise typed
  decisions, observations, policy factors, and verifier outcomes instead.
- No unbounded autonomous loops, branch search, recursive delegation, or self-granted permissions.
- No phrase grammar as the general dispatcher. Narrow deterministic renderers may optimize a route
  only after intent/depth selection.
- No child agent completion authority or direct Slack delivery.
- No search-provider generated answer treated as evidence.
- No optional provider or strategy mapping used as a conversation-availability gate.

## Current platform notes

The database remains at the Supabase/Postgres boundary described in the project migrations. Before
schema work, check the current [Supabase changelog](https://supabase.com/changelog?types=breaking-change)
and documentation. Relevant current guidance includes indexing columns used by authorization filters
and measuring RLS/query plans rather than assuming an index helps:
[RLS performance](https://supabase.com/docs/guides/database/postgres/row-level-security-performance)
and [Postgres indexes](https://supabase.com/docs/guides/database/postgres/indexes).
