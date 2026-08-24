import type {
  ConversationItem,
  FailureItem,
  IntegrationsResponse,
  MemoryRecordDetail,
  MemoryRecordSummary,
  OverviewResponse,
  Page,
  PlanTreeResponse,
  RunDetail,
  RunReasoning,
  RunSummary,
  TimelineEntry,
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

async function apiFetch<T>(path: string, params?: Record<string, string | number | undefined>): Promise<T> {
  const url = new URL(`/dashboard${path}`, BASE_URL);
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

export function getOverview(): Promise<OverviewResponse> {
  return apiFetch<OverviewResponse>("/overview");
}

export interface RunListFilters {
  status?: string;
  phase?: string;
  task_status?: string;
  limit?: number;
  offset?: number;
}

export function listRuns(filters: RunListFilters = {}): Promise<Page<RunSummary>> {
  return apiFetch<Page<RunSummary>>("/runs", { ...filters });
}

export function getRunDetail(runId: string): Promise<RunDetail> {
  return apiFetch<RunDetail>(`/runs/${encodeURIComponent(runId)}`);
}

export function getRunTimeline(runId: string): Promise<TimelineEntry[]> {
  return apiFetch<TimelineEntry[]>(`/runs/${encodeURIComponent(runId)}/timeline`);
}

export function getRunPlanTree(runId: string): Promise<PlanTreeResponse> {
  return apiFetch<PlanTreeResponse>(`/runs/${encodeURIComponent(runId)}/plan-tree`);
}

export function getRunReasoning(runId: string): Promise<RunReasoning> {
  return apiFetch<RunReasoning>(`/runs/${encodeURIComponent(runId)}/reasoning`);
}

export interface MemoryListFilters {
  kind?: string;
  visibility?: string;
  namespace_id?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export function listMemoryRecords(
  filters: MemoryListFilters = {},
): Promise<Page<MemoryRecordSummary>> {
  return apiFetch<Page<MemoryRecordSummary>>("/memory/records", { ...filters });
}

export function getMemoryRecord(recordId: string): Promise<MemoryRecordDetail> {
  return apiFetch<MemoryRecordDetail>(`/memory/records/${encodeURIComponent(recordId)}`);
}

export function getIntegrations(): Promise<IntegrationsResponse> {
  return apiFetch<IntegrationsResponse>("/integrations");
}

export interface FailureListFilters {
  terminal_reason?: string;
  status?: string;
  limit?: number;
  offset?: number;
}

export function listFailures(filters: FailureListFilters = {}): Promise<Page<FailureItem>> {
  return apiFetch<Page<FailureItem>>("/failures", { ...filters });
}

export function listConversations(
  filters: { limit?: number; offset?: number } = {},
): Promise<Page<ConversationItem>> {
  return apiFetch<Page<ConversationItem>>("/conversations", { ...filters });
}
