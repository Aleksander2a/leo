// Mirrors the FastAPI response shapes in src/leo/api/dashboard.py. Kept in sync
// by hand — the backend is small and hand-typed too, so a generated client
// would be more machinery than the surface warrants.

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface ActivityPoint {
  day: string | null;
  runs: number;
  answered: number;
  cost: number;
}

export interface ToolUsage {
  name: string;
  calls: number;
  succeeded: number;
  failed: number;
  avg_ms: number | null;
}

export interface ToolErrorCount {
  code: string;
  count: number;
}

export interface OverviewResponse {
  run_status_counts: Record<string, number>;
  total_runs: number;
  answered_runs: number;
  answer_rate: number | null;
  total_tokens: number;
  total_cost: number;
  total_tool_calls: number;
  total_model_turns: number;
  avg_run_seconds: number | null;
  p50_run_seconds: number | null;
  p95_run_seconds: number | null;
  active_memories: number;
  conversations: number;
  messages: number;
  activity: ActivityPoint[];
  tool_usage: ToolUsage[];
  tool_errors: ToolErrorCount[];
}

/** One user request, from question to answer. */
export interface RunSummary {
  id: string;
  scope_key: string;
  conversation_id: string;
  actor_id: string | null;
  thread_key: string | null;
  question: string;
  status: string;
  model: string | null;
  turns: number;
  tool_calls: number;
  total_tokens: number;
  cost: number;
  duration_seconds: number | null;
  started_at: string | null;
  finished_at: string | null;
}

/**
 * One entry in the ReAct trace. `kind: "model"` is a turn the model took —
 * `name` is its finish reason, and `arguments.tools_offered` lists the tools it
 * asked for. `kind: "tool"` is one call, with the arguments sent and the payload
 * that came back.
 */
export interface Step {
  seq: number;
  kind: "model" | "tool" | string;
  name: string;
  ok: boolean;
  duration_ms: number;
  arguments: Record<string, unknown> | null;
  result: Record<string, unknown> | null;
  created_at: string | null;
}

export interface RunDetail extends RunSummary {
  answer: string | null;
  error: string | null;
  prompt_tokens: number;
  completion_tokens: number;
  memories_written: number;
  conversation: ConversationSummary | null;
  steps: Step[];
}

export interface FailedToolCall {
  name: string;
  error: string | null;
  message: string | null;
}

export interface FailureItem extends RunSummary {
  error: string | null;
  failed_tool_calls: FailedToolCall[];
}

export interface ConversationSummary {
  id: string;
  scope_key: string;
  provider: string;
  kind: string;
  title: string | null;
  team_id: string | null;
  channel_id: string | null;
  runs: number | null;
  memories: number | null;
  messages: number | null;
  created_at: string | null;
  last_active_at: string | null;
}

export interface ConversationMessage {
  id: number;
  role: string;
  content: string;
  author_id: string | null;
  run_id: string | null;
  thread_key: string | null;
  created_at: string | null;
}

export interface ConversationDetail extends ConversationSummary {
  /** The summary's `messages`/`runs`/`memories` are counts; these are the rows. */
  recent_messages: ConversationMessage[];
  recent_runs: RunSummary[];
  recent_memories: MemorySummary[];
}

export interface MemorySummary {
  id: string;
  scope_key: string;
  kind: string;
  subject: string;
  content: string;
  importance: number;
  active: boolean;
  superseded_by: string | null;
  source_run_id: string | null;
  author_id: string | null;
  embedded: boolean;
  created_at: string | null;
  updated_at: string | null;
}

export interface MemoryDetail extends MemorySummary {
  /** Rows this one replaced. */
  supersedes: MemorySummary[];
  /** Rows that replaced this one, oldest first. */
  superseded_chain: MemorySummary[];
  source_run: RunSummary | null;
}

export interface MemoryKindCount {
  kind: string;
  count: number;
}

export interface ToolInfo {
  name: string;
  domain: string;
  description: string | null;
  indexed: boolean;
  calls: number;
  succeeded: number;
  failed: number;
  avg_ms: number | null;
  last_used_at: string | null;
  errors: ToolErrorCount[];
  updated_at: string | null;
}

export interface ScopeOption {
  scope_key: string;
  label: string;
  kind: string;
}

export interface HealthResponse {
  status: string;
  environment: string;
  configured: Record<string, boolean>;
  database_reachable?: boolean;
  database_error?: string;
}
