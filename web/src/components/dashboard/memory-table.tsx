"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import { formatDateTime, truncate } from "@/lib/utils";
import type { MemoryRecordSummary } from "@/lib/types";
import { useRouter } from "next/navigation";

const columns: ColumnDef<MemoryRecordSummary, unknown>[] = [
  { accessorKey: "kind", header: "Kind" },
  {
    accessorKey: "visibility",
    header: "Visibility",
    cell: (info) => <span className="text-xs">{(info.getValue() as string).replaceAll("_", " ")}</span>,
  },
  { accessorKey: "namespace_id", header: "Namespace" },
  {
    accessorKey: "status",
    header: "Status",
    cell: (info) => <StatusPill status={info.getValue() as string} />,
  },
  {
    accessorKey: "content_preview",
    header: "Latest content",
    cell: (info) => (
      <span className="block max-w-md truncate">{truncate((info.getValue() as string | null) ?? "", 100)}</span>
    ),
  },
  { accessorKey: "current_revision", header: "Rev" },
  {
    accessorKey: "last_recorded_at",
    header: "Last written",
    cell: (info) => formatDateTime(info.getValue() as string | null),
  },
];

export function MemoryTable({ records }: { records: MemoryRecordSummary[] }) {
  const router = useRouter();
  return (
    <DataTable
      columns={columns}
      data={records}
      onRowClick={(record) => router.push(`/memory/${record.id}`)}
      emptyMessage="No memory records match these filters."
    />
  );
}
