"use client";

import type { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import { SourceTypeBadge } from "@/components/ui/source-type-badge";
import { StatusPill } from "@/components/ui/status-pill";
import { formatDateTime, truncate } from "@/lib/utils";
import type { MemoryRecordSummary } from "@/lib/types";
import { useRouter } from "next/navigation";

const columns: ColumnDef<MemoryRecordSummary, unknown>[] = [
  { accessorKey: "kind", header: "Kind" },
  {
    accessorKey: "visibility",
    header: "Isolation boundary",
    cell: (info) => {
      const record = info.row.original;
      return (
        <div>
          <span className="text-xs">{record.visibility.replaceAll("_", " ")}</span>
          <p className="text-xs text-gray-400">{record.scope_label}</p>
        </div>
      );
    },
  },
  {
    accessorKey: "source_type",
    header: "Added",
    cell: (info) => {
      const value = info.getValue() as string | null;
      return value ? <SourceTypeBadge sourceType={value} /> : <span className="text-xs text-gray-400">—</span>;
    },
  },
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
