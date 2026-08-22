"use client";

import { JsonTree } from "@/components/ui/json-tree";
import { StatusPill } from "@/components/ui/status-pill";
import { formatDateTime } from "@/lib/utils";
import type { TimelineEntry } from "@/lib/types";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { cn } from "@/lib/utils";

const KIND_TONES: Record<string, string> = {
  lifecycle: "bg-blue-500",
  terminal: "bg-gray-700 dark:bg-gray-300",
  context_built: "bg-purple-500",
  memory_retrieved: "bg-purple-500",
  memory_committed: "bg-purple-500",
  model_called: "bg-indigo-500",
  tool_call: "bg-teal-500",
  evidence_normalized: "bg-teal-500",
  plan_revision: "bg-orange-500",
  plan_node: "bg-orange-500",
  delegation: "bg-orange-500",
  verification: "bg-amber-500",
  delivery: "bg-emerald-500",
  usage: "bg-gray-400",
  conflict_detected: "bg-red-500",
  synthesis: "bg-indigo-400",
};

export function EventTimeline({ entries }: { entries: TimelineEntry[] }) {
  if (entries.length === 0) {
    return <p className="text-sm text-gray-400">No events recorded for this run.</p>;
  }

  return (
    <ol className="relative space-y-1 border-l border-gray-200 pl-4 dark:border-gray-800">
      {entries.map((entry) => (
        <TimelineRow key={entry.sequence} entry={entry} />
      ))}
    </ol>
  );
}

function TimelineRow({ entry }: { entry: TimelineEntry }) {
  const [open, setOpen] = useState(false);
  const status = (entry.envelope?.payload.status as string | undefined) ?? undefined;

  return (
    <li className="relative">
      <span
        className={cn(
          "absolute top-2 -left-[21px] h-2.5 w-2.5 rounded-full",
          KIND_TONES[entry.kind] ?? "bg-gray-400",
        )}
      />
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 rounded-md py-1.5 pl-2 text-left hover:bg-gray-50 dark:hover:bg-gray-900"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="w-8 shrink-0 text-xs text-gray-400 tabular-nums">#{entry.sequence}</span>
        <span className="text-sm font-medium text-gray-800 dark:text-gray-200">
          {entry.kind.replaceAll("_", " ")}
        </span>
        {status ? <StatusPill status={status} /> : null}
        {entry.normalization_error ? <StatusPill status="failed" className="ml-1" /> : null}
        <span className="ml-auto shrink-0 text-xs text-gray-400">{formatDateTime(entry.occurred_at)}</span>
      </button>
      {open ? (
        <div className="ml-8 space-y-3 border-l border-gray-100 py-2 pl-3 dark:border-gray-800">
          {entry.envelope ? (
            <section>
              <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">
                Normalized payload
              </p>
              <JsonTree data={entry.envelope.payload} />
            </section>
          ) : (
            <p className="text-xs text-amber-600 dark:text-amber-400">
              Could not normalize this event (non-contiguous sequence); showing raw payload only.
            </p>
          )}
          <section>
            <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">
              Raw payload (full fidelity)
            </p>
            <JsonTree data={entry.raw_payload} />
          </section>
        </div>
      ) : null}
    </li>
  );
}
