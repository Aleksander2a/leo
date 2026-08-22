import { FilterBar } from "@/components/dashboard/filter-bar";
import { MemoryTable } from "@/components/dashboard/memory-table";
import { Pager } from "@/components/ui/pager";
import { listMemoryRecords } from "@/lib/api";

const VISIBILITIES = [
  "thread_local",
  "conversation_local",
  "channel_local",
  "actor_private",
  "strategy_shared",
  "organization_shared",
];
const STATUSES = ["active", "superseded", "contested", "retracted"];

const LIMIT = 25;

export default async function MemoryPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const visibility = firstValue(params.visibility);
  const status = firstValue(params.status);
  const kind = firstValue(params.kind);
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const page = await listMemoryRecords({ visibility, status, kind, limit: LIMIT, offset });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Memory</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Every durable memory record Leo has written, with its append-only revision log.
        </p>
      </div>

      <FilterBar
        filters={[
          {
            param: "visibility",
            label: "Visibility",
            options: VISIBILITIES.map((value) => ({ value, label: value.replaceAll("_", " ") })),
          },
          {
            param: "status",
            label: "Status",
            options: STATUSES.map((value) => ({ value, label: value })),
          },
        ]}
      />

      <MemoryTable records={page.items} />

      <Pager total={page.total} limit={page.limit} offset={page.offset} basePath="/memory" searchParams={{ visibility, status, kind }} />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
