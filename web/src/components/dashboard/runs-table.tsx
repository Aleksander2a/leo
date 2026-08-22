"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import { formatCost, formatDateTime, formatNumber, truncate } from "@/lib/utils";
import type { RunSummary } from "@/lib/types";
import { useRouter } from "next/navigation";

const columns: ColumnDef<RunSummary, unknown>[] = [
  {
    accessorKey: "status",
    header: "Status",
    cell: (info) => <StatusPill status={info.getValue() as string} />,
  },
  {
    accessorKey: "task_objective",
    header: "Objective",
    cell: (info) => <span className="block max-w-md truncate">{truncate(info.getValue() as string, 90)}</span>,
  },
  { accessorKey: "phase", header: "Phase" },
  { accessorKey: "iteration", header: "Iter" },
  {
    accessorKey: "terminal_reason",
    header: "Terminal reason",
    cell: (info) => (info.getValue() as string | null) ?? "—",
  },
  {
    accessorKey: "total_tokens",
    header: "Tokens",
    cell: (info) => formatNumber(info.getValue() as number | null),
  },
  {
    accessorKey: "cost",
    header: "Cost",
    cell: (info) => formatCost(info.getValue() as number | null),
  },
  {
    accessorKey: "created_at",
    header: "Created",
    cell: (info) => formatDateTime(info.getValue() as string),
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
