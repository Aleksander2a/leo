"use client";

import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import type { RunSummary } from "@/lib/types";
import { formatCost, formatNumber, formatRelative, formatSeconds, scopeLabel, truncate } from "@/lib/utils";
import type { ColumnDef } from "@tanstack/react-table";
import { useRouter } from "next/navigation";

const columns: ColumnDef<RunSummary, unknown>[] = [
  {
    accessorKey: "status",
    header: "Status",
    cell: (info) => <StatusPill status={info.getValue() as string} />,
  },
  {
    accessorKey: "question",
    header: "Question",
    cell: (info) => (
      <span className="block max-w-md truncate">{truncate(info.getValue() as string, 90)}</span>
    ),
  },
  {
    accessorKey: "scope_key",
    header: "Conversation",
    cell: (info) => (
      <span className="font-mono text-xs text-gray-500" title={info.getValue() as string}>
        {scopeLabel(info.getValue() as string)}
      </span>
    ),
  },
  { accessorKey: "turns", header: "Turns" },
  { accessorKey: "tool_calls", header: "Tools" },
  {
    accessorKey: "duration_seconds",
    header: "Duration",
    cell: (info) => formatSeconds(info.getValue() as number | null),
  },
  {
    accessorKey: "total_tokens",
    header: "Tokens",
    cell: (info) => formatNumber(info.getValue() as number),
  },
  {
    accessorKey: "cost",
    header: "Cost",
    cell: (info) => formatCost(info.getValue() as number),
  },
  {
    accessorKey: "started_at",
    header: "Started",
    cell: (info) => formatRelative(info.getValue() as string | null),
  },
];

export function RunsTable({ runs }: { runs: RunSummary[] }) {
  const router = useRouter();
  return (
    <DataTable
      columns={columns}
      data={runs}
      onRowClick={(run) => router.push(`/runs/${run.id}`)}
      emptyMessage="No runs match these filters."
    />
  );
}
