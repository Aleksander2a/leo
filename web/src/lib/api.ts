import type {
  ConversationDetail,
  ConversationSummary,
  FailureItem,
  HealthResponse,
  MemoryDetail,
  MemoryKindCount,
  MemorySummary,
  OverviewResponse,
  Page,
  RunDetail,
  RunSummary,
  ScopeOption,
  ToolInfo,
} from "@/lib/types";

const BASE_URL = process.env.NEXT_PUBLIC_DASHBOARD_API_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

type Params = Record<string, string | number | boolean | undefined>;

async function apiFetch<T>(path: string, params?: Params): Promise<T> {
  const url = new URL(path, BASE_URL);
  if (params) {
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") {
        url.searchParams.set(key, String(value));
      }
    }
  }
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new ApiError(response.status, body || response.statusText);
  }
  return (await response.json()) as T;
}

function dashboard<T>(path: string, params?: Params): Promise<T> {
  return apiFetch<T>(`/dashboard${path}`, params);
}

/** Scope keys contain colons, so they must be encoded into the path. */
const scopePath = (scopeKey: string) => encodeURIComponent(scopeKey);

export function getHealth(deep = false): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health", { deep });
}

export function getOverview(days = 14): Promise<OverviewResponse> {
  return dashboard<OverviewResponse>("/overview", { days });
}

export interface RunFilters {
  status?: string;
  scope_key?: string;
  q?: string;
  limit?: number;
  offset?: number;
}

export function listRuns(filters: RunFilters = {}): Promise<Page<RunSummary>> {
  return dashboard<Page<RunSummary>>("/runs", { ...filters });
}

export function getRun(runId: string): Promise<RunDetail> {
  return dashboard<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

export function listFailures(
  filters: { limit?: number; offset?: number } = {},
): Promise<Page<FailureItem>> {
  return dashboard<Page<FailureItem>>("/failures", { ...filters });
}

export function listConversations(
  filters: { kind?: string; limit?: number; offset?: number } = {},
): Promise<Page<ConversationSummary>> {
  return dashboard<Page<ConversationSummary>>("/conversations", { ...filters });
}

export function getConversation(scopeKey: string): Promise<ConversationDetail> {
  return dashboard<ConversationDetail>(`/conversations/${scopePath(scopeKey)}`);
}

export interface MemoryFilters {
  scope_key?: string;
  kind?: string;
  q?: string;
  include_inactive?: boolean;
  limit?: number;
  offset?: number;
}

export function listMemory(filters: MemoryFilters = {}): Promise<Page<MemorySummary>> {
  return dashboard<Page<MemorySummary>>("/memory", { ...filters });
}

export function getMemory(memoryId: string): Promise<MemoryDetail> {
  return dashboard<MemoryDetail>(`/memory/${encodeURIComponent(memoryId)}`);
}

export function listMemoryKinds(): Promise<{ items: MemoryKindCount[] }> {
  return dashboard<{ items: MemoryKindCount[] }>("/memory-kinds");
}

export function listTools(): Promise<{ items: ToolInfo[] }> {
  return dashboard<{ items: ToolInfo[] }>("/tools");
}

export function listScopes(): Promise<{ items: ScopeOption[] }> {
  return dashboard<{ items: ScopeOption[] }>("/scopes");
}
