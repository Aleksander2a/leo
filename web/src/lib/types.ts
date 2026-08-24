// Types mirror the FastAPI dashboard response shapes in
// D:\leo\src\leo\api\dashboard\routers\*.py -- keep them in sync by hand, there is no
// generated client (the backend is intentionally small and hand-typed too).

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface KeyCount {
  key: string;
  count: number;
}

export interface OverviewResponse {
  run_status_counts: Record<string, number>;
  task_status_counts: Record<string, number>;
  tool_calls: { started: number; completed: number; failed: number };
  tool_call_success_rate: number | null;
  total_cost: number | null;
  total_tokens: number | null;
  total_model_calls: number;
  total_tool_calls: number;
  memory_writes_total: number;
  memory_pages_referenced_total: number;
  delivery_state_counts: Record<string, number>;
  failure_reasons: KeyCount[];
  avg_run_latency_seconds: number | null;
}

export interface RunSummary {
  id: string;
  task_id: string;
  status: string;
  phase: string;
  iteration: number;
  task_objective: string;
  task_status: string;
  started_at: string | null;
  terminal_reason: string | null;
  total_tokens: number | null;
  cost: number | null;
  created_at: string;
}

export interface RunFields {
  id: string;
  task_id: string;
  organization_id: string;
  strategy_id: string;
  status: string;
  phase: string;
  iteration: number;
  limits: Record<string, unknown>;
  usage: Record<string, unknown>;
  started_at: string | null;
  deadline_at: string | null;
  final_output: string | null;
  terminal_reason: string | null;
  created_at: string;
  updated_at: string;
}

export interface TaskFields {
  id: string;
  thread_id: string;
  objective: string;
  parent_task_id: string | null;
  continuation_kind: string;
  status: string;
  final_output: string | null;
  verifier_feedback: string[];
  attempt_count: number;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface ThreadFields {
  id: string;
  origin_provider: string;
  external_thread_id: string;
  external_channel_id: string | null;
  conversation_id: string | null;
  created_at: string;
}

export type CallKind =
  | "mcp"
  | "rest_api"
  | "internal_memory"
  | "internal_agent"
  | "internal_context"
  | "unknown";

export interface ObservationSummary {
  id: string;
  tool_call_id: string;
  kind: string;
  data: Record<string, unknown>;
  source: { provider?: string; reference?: string; url?: string | null };
  status: string;
  quality: string;
  observed_at: string;
  expires_at: string | null;
  rejection_code: string | null;
  call_kind: CallKind | null;
  integration: string | null;
}

export interface ClaimSummary {
  id: string;
  kind: string;
  statement: string;
  observation_ids: string[];
}

export interface DeliverySummary {
  id: string;
  kind: string;
  state: string;
  destination_channel_id: string;
  destination_thread_ts: string;
  attempt_count: number;
  receipt_message_ts: string | null;
  last_error: string | null;
  created_at: string;
  updated_at: string;
}

export interface RunDetail {
  run: RunFields;
  task: TaskFields | null;
  thread: ThreadFields | null;
  observations: ObservationSummary[];
  claims: ClaimSummary[];
  deliveries: DeliverySummary[];
  event_count: number;
}

export interface EventEnvelope {
  event_id: string;
  run_id: string;
  task_id: string;
  scope: { organization_id: string; strategy_id: string };
  sequence: number;
  occurred_at: string;
  kind: string;
  schema_version: string;
  correlation_id: string;
  causation_id: string | null;
  payload: Record<string, unknown>;
}

export interface ModelCallTranscript {
  request: {
    model?: string;
    messages?: { role: string; content: unknown }[];
    tools?: { type: string; function: { name: string; description: string; parameters: unknown } }[];
    tool_choice?: unknown;
    [key: string]: unknown;
  };
  response: Record<string, unknown>;
}

export interface TimelineEntry {
  sequence: number;
  kind: string;
  occurred_at: string;
  envelope: EventEnvelope | null;
  raw_payload: Record<string, unknown>;
  normalization_error: boolean;
  call_kind: CallKind | null;
  integration: string | null;
  transcript: ModelCallTranscript | null;
}

export interface PlanRevisionSummary {
  id: string;
  number: number;
  goal: string;
  reason: string;
  digest: string;
  created_at: string;
}

export interface ChildRunSummary {
  id: string;
  status: string;
  phase: string;
  terminal_reason: string | null;
}

export interface DelegationEntry {
  id: string;
  attempt: number;
  status: string;
  output: string | null;
  error: string | null;
  child_task_id: string | null;
  child_run: ChildRunSummary | null;
  child_plans: PlanTreeNode[];
  created_at: string;
  finished_at: string | null;
}

export interface PlanNodeEntry {
  id: string;
  node_key: string;
  objective: string;
  depends_on: string[];
  status: string;
  attempt: number;
  max_attempts: number;
  output: string | null;
  error: string | null;
  delegations: DelegationEntry[];
}

export interface PlanTreeNode {
  id: string;
  status: string;
  current_revision: number;
  max_revisions: number;
  output: string | null;
  error: string | null;
  revisions: PlanRevisionSummary[];
  nodes: PlanNodeEntry[];
  created_at: string;
  updated_at: string;
}

export interface PlanTreeResponse {
  run_id: string;
  plans: PlanTreeNode[];
}

export interface MemoryRecordSummary {
  id: string;
  kind: string;
  visibility: string;
  namespace_id: string;
  current_revision: number;
  generation: number;
  status: string;
  created_at: string;
  content_preview: string | null;
  last_recorded_at: string | null;
  source_type: string | null;
  scope_label: string;
}

export interface MemoryRecordFields {
  id: string;
  kind: string;
  visibility: string;
  namespace_id: string;
  current_revision: number;
  generation: number;
  status: string;
  created_at: string;
  scope_label: string;
}

export interface MemorySourceFields {
  id: string;
  source_kind: string;
  reference: string;
  visibility: string;
  namespace_id: string;
}

export interface MemoryRevisionFields {
  number: number;
  content: string;
  content_hash: string;
  source_ids: string[];
  visibility: string;
  sensitivity: number;
  valid_from: string;
  valid_until: string | null;
  recorded_at: string;
  expires_at: string | null;
  status: string;
  actor_id: string;
  reason: string;
  supersedes_revision: number | null;
  source_type: string;
}

export interface MemoryRecordDetail {
  record: MemoryRecordFields;
  sources: MemorySourceFields[];
  revisions: MemoryRevisionFields[];
}

export interface ProviderStat {
  provider: string;
  display_name: string;
  call_kind: CallKind;
  total: number;
  retrieved: number;
  stale: number;
  rejected: number;
  success_rate: number | null;
}

export interface ToolFailureItem extends KeyCount {
  call_kind: CallKind;
  integration: string;
}

export interface IntegrationsResponse {
  providers: ProviderStat[];
  tool_failures: ToolFailureItem[];
}

export interface FailureItem {
  run_id: string;
  task_id: string;
  status: string;
  phase: string;
  terminal_reason: string | null;
  task_objective: string;
  task_last_error: string | null;
  attempt_count: number;
  updated_at: string;
}

export interface ConversationItem {
  id: string;
  provider: string;
  team_id: string;
  kind: string;
  bot_presence: string;
  lifecycle: string;
  external_provenance: string;
  thread_count: number;
  created_at: string;
  updated_at: string;
}

/** One iteration of Leo's plan/act/observe trace. */
export interface ReasoningStep {
  iteration: number;
  /** Model-authored intent for this turn. */
  plan: string;
  /** What the model actually did (a tool call, or answering). */
  action: string;
  /** Harness-written result, so a hallucinated success cannot enter the trace. */
  outcome: string;
}

export interface RunReasoning {
  run_id: string;
  task_id: string;
  objective: string | null;
  steps: ReasoningStep[];
  step_count: number;
}
