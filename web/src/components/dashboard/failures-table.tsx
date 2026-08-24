"use client";

import { StatusPill } from "@/components/ui/status-pill";
import type { FailureItem } from "@/lib/types";
import { formatRelative, scopeLabel } from "@/lib/utils";
import Link from "next/link";

/**
 * A failure here means the run ended without an answer — a provider outage, or
 * an empty final completion. Failed *tool calls* are listed alongside it as
 * context, because those are recoveries the loop handled rather than causes.
 */
export function FailuresTable({ failures }: { failures: FailureItem[] }) {
  if (failures.length === 0) {
    return (
      <div className="rounded-lg border border-gray-200 py-12 text-center dark:border-gray-800">
        <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
          No run has ended without an answer.
        </p>
        <p className="mt-1 text-xs text-gray-400">
          Tool failures are handled inside the loop and do not appear here.
        </p>
      </div>
    );
  }

  return (
    <ul className="space-y-3">
      {failures.map((failure) => (
        <li
          key={failure.id}
          className="rounded-lg border border-red-200 bg-red-50/40 p-4 dark:border-red-900/50 dark:bg-red-950/20"
        >
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={failure.status} />
            <Link
              href={`/runs/${failure.id}`}
              className="min-w-0 flex-1 truncate text-sm font-medium text-gray-800 hover:underline dark:text-gray-200"
            >
              {failure.question || "(no question recorded)"}
            </Link>
            <span className="shrink-0 font-mono text-xs text-gray-400" title={failure.scope_key}>
              {scopeLabel(failure.scope_key)}
            </span>
            <span className="shrink-0 text-xs text-gray-400">
              {formatRelative(failure.started_at)}
            </span>
          </div>

          {failure.error ? (
            <p className="mt-2 font-mono text-xs break-all text-red-700 dark:text-red-400">
              {failure.error}
            </p>
          ) : null}

          {failure.failed_tool_calls.length > 0 ? (
            <div className="mt-3">
              <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">
                Tool calls that failed during this run
              </p>
              <ul className="space-y-1">
                {failure.failed_tool_calls.map((call, index) => (
                  <li key={`${call.name}-${index}`} className="text-xs text-gray-600 dark:text-gray-400">
                    <span className="font-mono">{call.name}</span>
                    {call.error ? <span className="text-red-600 dark:text-red-400"> · {call.error}</span> : null}
                    {call.message ? <span className="text-gray-500"> — {call.message}</span> : null}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          <p className="mt-3 text-xs text-gray-400">
            {failure.turns} turns · {failure.tool_calls} tool calls
          </p>
        </li>
      ))}
    </ul>
  );
}
