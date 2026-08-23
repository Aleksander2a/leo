"use client";

import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { ModelCallPanel } from "@/components/dashboard/model-call-panel";
import { JsonTree } from "@/components/ui/json-tree";
import { ProvenanceBadge } from "@/components/ui/provenance-badge";
import { StatusPill } from "@/components/ui/status-pill";
import { cn, formatDateTime } from "@/lib/utils";
import type { ObservationSummary, TimelineEntry } from "@/lib/types";

const KIND_TONES: Record<string, string> = {
  lifecycle: "bg-blue-500",
  terminal: "bg-gray-700 dark:bg-gray-300",
  context_built: "bg-purple-500",
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

interface ToolCallGroup {
  type: "tool";
  key: string;
  sequence: number;
  toolCallId: string;
  tool: string | null;
  started: TimelineEntry | null;
  completed: TimelineEntry | null;
  failed: TimelineEntry | null;
  observationCreated: TimelineEntry | null;
  callKind: TimelineEntry["call_kind"];
  integration: TimelineEntry["integration"];
}

interface SingleGroup {
  type: "single";
  key: string;
  sequence: number;
  entry: TimelineEntry;
}

type Group = ToolCallGroup | SingleGroup;

/**
 * A tool call is 2-4 separate durable events (tool_started, then tool_completed OR
 * tool_failed, then observation_created on success) -- correct for the append-only event
 * log, but confusing to read one at a time. This groups them back into one row per
 * tool_call_id so "the request and response from a tool call" reads as a single unit.
 * Every other event kind (model calls, context assembly, verification, delivery, plan
 * activity) stays its own row.
 */
function groupTimeline(entries: TimelineEntry[]): Group[] {
  const toolGroups = new Map<string, ToolCallGroup>();
  const ordered: Group[] = [];

  for (const entry of entries) {
    const toolCallId = entry.raw_payload.tool_call_id;
    if ((entry.kind === "tool_call" || entry.kind === "evidence_normalized") && typeof toolCallId === "string") {
      let group = toolGroups.get(toolCallId);
      if (!group) {
        group = {
          type: "tool",
          key: `tool-${toolCallId}`,
          sequence: entry.sequence,
          toolCallId,
          tool: null,
          started: null,
          completed: null,
          failed: null,
          observationCreated: null,
          callKind: null,
          integration: null,
        };
        toolGroups.set(toolCallId, group);
        ordered.push(group);
      }
      const tool = entry.raw_payload.tool;
      if (typeof tool === "string") group.tool = tool;
      if (entry.call_kind) {
        group.callKind = entry.call_kind;
        group.integration = entry.integration;
      }
      if (entry.kind === "evidence_normalized") {
        group.observationCreated = entry;
      } else if ("parallel_batch" in entry.raw_payload) {
        group.started = entry;
      } else if ("code" in entry.raw_payload && "retryable" in entry.raw_payload) {
        group.failed = entry;
      } else {
        group.completed = entry;
      }
      continue;
    }
    ordered.push({ type: "single", key: `single-${entry.sequence}`, sequence: entry.sequence, entry });
  }

  return ordered.sort((a, b) => a.sequence - b.sequence);
}

export function EventTimeline({
  entries,
  observations = [],
}: {
  entries: TimelineEntry[];
  observations?: ObservationSummary[];
}) {
  if (entries.length === 0) {
    return <p className="text-sm text-gray-400">No events recorded for this run.</p>;
  }
  const groups = groupTimeline(entries);

  return (
    <ol className="relative space-y-1 border-l border-gray-200 pl-4 dark:border-gray-800">
      {groups.map((group) =>
        group.type === "tool" ? (
          <ToolCallRow key={group.key} group={group} observations={observations} />
        ) : (
          <SingleRow key={group.key} entry={group.entry} timeline={entries} observations={observations} />
        ),
      )}
    </ol>
  );
}

function RowShell({
  dotClassName,
  sequence,
  title,
  badges,
  occurredAt,
  open,
  onToggle,
  children,
}: {
  dotClassName: string;
  sequence: number;
  title: string;
  badges: React.ReactNode;
  occurredAt: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}) {
  return (
    <li className="relative">
      <span className={cn("absolute top-2 -left-[21px] h-2.5 w-2.5 rounded-full", dotClassName)} />
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-2 rounded-md py-1.5 pl-2 text-left hover:bg-gray-50 dark:hover:bg-gray-900"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <span className="w-8 shrink-0 text-xs text-gray-400 tabular-nums">#{sequence}</span>
        <span className="text-sm font-medium text-gray-800 dark:text-gray-200">{title}</span>
        {badges}
        <span className="ml-auto shrink-0 text-xs text-gray-400">{formatDateTime(occurredAt)}</span>
      </button>
      {open ? <div className="ml-8 space-y-3 border-l border-gray-100 py-2 pl-3 dark:border-gray-800">{children}</div> : null}
    </li>
  );
}

function ToolCallRow({ group, observations }: { group: ToolCallGroup; observations: ObservationSummary[] }) {
  const [open, setOpen] = useState(false);
  const status = group.failed ? "failed" : group.observationCreated || group.completed ? "completed" : "started";
  const observationId = group.observationCreated?.raw_payload.observation_id;
  const observation =
    typeof observationId === "string" ? observations.find((item) => item.id === observationId) : undefined;
  const args = group.started?.raw_payload.arguments ?? group.failed?.raw_payload.arguments;
  const domain = group.tool?.split(".", 1)[0];

  return (
    <RowShell
      dotClassName={KIND_TONES.tool_call}
      sequence={group.sequence}
      title={group.tool ?? "tool call"}
      badges={
        <>
          {group.callKind ? <ProvenanceBadge callKind={group.callKind} integration={group.integration} /> : null}
          <StatusPill status={status} />
        </>
      }
      occurredAt={(group.started ?? group.completed ?? group.failed ?? group.observationCreated)!.occurred_at}
      open={open}
      onToggle={() => setOpen((value) => !value)}
    >
      {domain === "memory" ? (
        <MemoryCallDetail group={group} observation={observation} args={args} />
      ) : domain === "agent" ? (
        <AgentCallDetail group={group} observation={observation} args={args} />
      ) : (
        <GenericToolDetail group={group} observation={observation} args={args} />
      )}
    </RowShell>
  );
}

function ArgumentsSection({ args }: { args: unknown }) {
  return (
    <section>
      <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Request (arguments)</p>
      {args ? <JsonTree data={args} /> : <p className="text-xs text-gray-400">No arguments recorded.</p>}
    </section>
  );
}

function ResponseSection({
  group,
  observation,
}: {
  group: ToolCallGroup;
  observation: ObservationSummary | undefined;
}) {
  if (group.failed) {
    return (
      <section>
        <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Response (failure)</p>
        <p className="rounded-md bg-red-50 px-2 py-1.5 text-xs text-red-700 dark:bg-red-500/10 dark:text-red-400">
          {String(group.failed.raw_payload.code)}
          {group.failed.raw_payload.retryable ? " (retryable)" : ""}
        </p>
      </section>
    );
  }
  return (
    <section>
      <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Response (result)</p>
      {observation ? <JsonTree data={observation.data} /> : <p className="text-xs text-gray-400">Not yet observed.</p>}
    </section>
  );
}

function GenericToolDetail({
  group,
  observation,
  args,
}: {
  group: ToolCallGroup;
  observation: ObservationSummary | undefined;
  args: unknown;
}) {
  return (
    <div className="space-y-3">
      <ArgumentsSection args={args} />
      <ResponseSection group={group} observation={observation} />
    </div>
  );
}

function MemoryCallDetail({
  group,
  observation,
  args,
}: {
  group: ToolCallGroup;
  observation: ObservationSummary | undefined;
  args: unknown;
}) {
  const isWrite = group.tool === "memory.note" || group.tool?.startsWith("memory.remember") || group.tool?.startsWith("memory.correct");
  const label = isWrite ? "Memory written" : "Memory retrieved";
  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold tracking-wide text-purple-600 uppercase dark:text-purple-400">{label}</p>
      <ArgumentsSection args={args} />
      <ResponseSection group={group} observation={observation} />
    </div>
  );
}

function AgentCallDetail({
  group,
  observation,
  args,
}: {
  group: ToolCallGroup;
  observation: ObservationSummary | undefined;
  args: unknown;
}) {
  return (
    <div className="space-y-3">
      <p className="text-xs font-semibold tracking-wide text-orange-600 uppercase dark:text-orange-400">
        Delegation to subagent
      </p>
      <ArgumentsSection args={args} />
      <ResponseSection group={group} observation={observation} />
    </div>
  );
}

function SingleRow({
  entry,
  timeline,
  observations,
}: {
  entry: TimelineEntry;
  timeline: TimelineEntry[];
  observations: ObservationSummary[];
}) {
  const [open, setOpen] = useState(false);
  const status = (entry.envelope?.payload.status as string | undefined) ?? undefined;

  return (
    <RowShell
      dotClassName={KIND_TONES[entry.kind] ?? "bg-gray-400"}
      sequence={entry.sequence}
      title={entry.kind.replaceAll("_", " ")}
      badges={
        <>
          {entry.call_kind ? <ProvenanceBadge callKind={entry.call_kind} integration={entry.integration} /> : null}
          {status ? <StatusPill status={status} /> : null}
          {entry.normalization_error ? <StatusPill status="failed" className="ml-1" /> : null}
        </>
      }
      occurredAt={entry.occurred_at}
      open={open}
      onToggle={() => setOpen((value) => !value)}
    >
      {entry.kind === "model_called" ? (
        <ModelCallPanel entry={entry} timeline={timeline} observations={observations} />
      ) : (
        <>
          {entry.envelope ? (
            <section>
              <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Normalized payload</p>
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
        </>
      )}
    </RowShell>
  );
}
