"use client";

import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import type { MemorySummary } from "@/lib/types";
import { formatRelative, scopeLabel, truncate } from "@/lib/utils";
import type { ColumnDef } from "@tanstack/react-table";
import { useRouter } from "next/navigation";

const columns: ColumnDef<MemorySummary, unknown>[] = [
  {
    accessorKey: "kind",
    header: "Kind",
    cell: (info) => <StatusPill status={info.getValue() as string} />,
  },
  {
    accessorKey: "subject",
    header: "Subject",
    cell: (info) => (info.getValue() as string) || "—",
  },
  {
    accessorKey: "content",
    header: "Content",
    cell: (info) => (
      <span className="block max-w-lg truncate">{truncate(info.getValue() as string, 110)}</span>
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
  {
    accessorKey: "importance",
    header: "Importance",
    cell: (info) => "★".repeat(info.getValue() as number),
  },
  {
    accessorKey: "active",
    header: "State",
    cell: (info) => (
      <StatusPill status={(info.getValue() as boolean) ? "active" : "superseded"} />
    ),
  },
  {
    accessorKey: "updated_at",
    header: "Updated",
    cell: (info) => formatRelative(info.getValue() as string | null),
  },
];

export function MemoryTable({ memories }: { memories: MemorySummary[] }) {
  const router = useRouter();
  return (
    <DataTable
      columns={columns}
      data={memories}
      onRowClick={(memory) => router.push(`/memory/${encodeURIComponent(memory.id)}`)}
      emptyMessage="No memories match these filters."
    />
  );
}
