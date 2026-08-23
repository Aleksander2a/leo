import { cn } from "@/lib/utils";
import type { CallKind } from "@/lib/types";

const CALL_KIND_LABELS: Record<CallKind, string> = {
  mcp: "MCP",
  rest_api: "REST API",
  internal_memory: "Internal memory",
  internal_agent: "Internal subagent",
  internal_context: "Internal context",
  unknown: "Unknown",
};

const CALL_KIND_CLASSES: Record<CallKind, string> = {
  mcp: "bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-500/10 dark:text-violet-400 dark:ring-violet-500/30",
  rest_api:
    "bg-blue-50 text-blue-700 ring-blue-600/20 dark:bg-blue-500/10 dark:text-blue-400 dark:ring-blue-500/30",
  internal_memory:
    "bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-400 dark:ring-amber-500/30",
  internal_agent:
    "bg-slate-100 text-slate-700 ring-slate-500/20 dark:bg-slate-500/10 dark:text-slate-300 dark:ring-slate-500/30",
  internal_context:
    "bg-slate-100 text-slate-700 ring-slate-500/20 dark:bg-slate-500/10 dark:text-slate-300 dark:ring-slate-500/30",
  unknown: "bg-gray-100 text-gray-500 ring-gray-500/20 dark:bg-gray-500/10 dark:text-gray-400 dark:ring-gray-500/30",
};

/** Badge distinguishing an MCP call, a native REST integration call, or a harness-internal
 * capability (memory/subagent/thread-context) -- see leo.api.dashboard.provenance. */
export function ProvenanceBadge({
  callKind,
  integration,
  className,
}: {
  callKind: CallKind | string | null | undefined;
  integration?: string | null;
  className?: string;
}) {
  const key = (callKind as CallKind) ?? "unknown";
  const label = CALL_KIND_LABELS[key] ?? callKind ?? "Unknown";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset whitespace-nowrap",
        CALL_KIND_CLASSES[key] ?? CALL_KIND_CLASSES.unknown,
        className,
      )}
      title={integration ?? undefined}
    >
      {label}
      {integration ? <span className="opacity-70">· {integration}</span> : null}
    </span>
  );
}
