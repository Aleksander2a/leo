"use client";

import { DataTable } from "@/components/ui/data-table";
import { StatusPill } from "@/components/ui/status-pill";
import type { ConversationSummary } from "@/lib/types";
import { formatNumber, formatRelative, scopeKind } from "@/lib/utils";
import type { ColumnDef } from "@tanstack/react-table";
import { useRouter } from "next/navigation";

const columns: ColumnDef<ConversationSummary, unknown>[] = [
  {
    accessorKey: "kind",
    header: "Kind",
    cell: (info) => <StatusPill status={scopeKind(info.getValue() as string)} />,
  },
  {
    accessorKey: "scope_key",
    header: "Scope key",
    cell: (info) => <span className="font-mono text-xs">{info.getValue() as string}</span>,
  },
  {
    accessorKey: "title",
    header: "Description",
    cell: (info) => (info.getValue() as string | null) ?? "—",
  },
  {
    accessorKey: "messages",
    header: "Messages",
    cell: (info) => formatNumber(info.getValue() as number | null),
  },
  {
    accessorKey: "runs",
    header: "Runs",
    cell: (info) => formatNumber(info.getValue() as number | null),
  },
  {
    accessorKey: "memories",
    header: "Memories",
    cell: (info) => formatNumber(info.getValue() as number | null),
  },
  {
    accessorKey: "last_active_at",
    header: "Last active",
    cell: (info) => formatRelative(info.getValue() as string | null),
  },
];

export function ConversationsTable({
  conversations,
}: {
  conversations: ConversationSummary[];
}) {
  const router = useRouter();
  return (
    <DataTable
      columns={columns}
      data={conversations}
      onRowClick={(conversation) =>
        router.push(`/conversations/${encodeURIComponent(conversation.scope_key)}`)
      }
      emptyMessage="No conversations recorded yet."
    />
  );
}
