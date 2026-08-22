"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import { formatDateTime, truncate } from "@/lib/utils";
import type { FailureItem } from "@/lib/types";
import { useRouter } from "next/navigation";

const columns: ColumnDef<FailureItem, unknown>[] = [
  {
    accessorKey: "status",
    header: "Status",
    cell: (info) => <StatusPill status={info.getValue() as string} />,
  },
  {
    accessorKey: "terminal_reason",
    header: "Reason",
    cell: (info) => <span className="font-mono text-xs">{(info.getValue() as string | null) ?? "—"}</span>,
  },
  {
    accessorKey: "task_objective",
    header: "Objective",
    cell: (info) => <span className="block max-w-md truncate">{truncate(info.getValue() as string, 90)}</span>,
  },
  { accessorKey: "attempt_count", header: "Attempts" },
  {
    accessorKey: "task_last_error",
    header: "Last error",
    cell: (info) => <span className="text-xs text-red-500">{(info.getValue() as string | null) ?? "—"}</span>,
  },
  {
    accessorKey: "updated_at",
    header: "Updated",
    cell: (info) => formatDateTime(info.getValue() as string),
  },
];

export function FailuresTable({ failures }: { failures: FailureItem[] }) {
  const router = useRouter();
  return (
    <DataTable
      columns={columns}
      data={failures}
      onRowClick={(failure) => router.push(`/runs/${failure.run_id}`)}
      emptyMessage="No failures match these filters."
    />
  );
}
