import { FilterBar } from "@/components/dashboard/filter-bar";
import { MemoryTable } from "@/components/dashboard/memory-table";
import { SearchBox } from "@/components/dashboard/search-box";
import { Pager } from "@/components/ui/pager";
import { listMemory, listMemoryKinds, listScopes } from "@/lib/api";

const LIMIT = 25;

export default async function MemoryPage(props: PageProps<"/memory">) {
  const params = await props.searchParams;
  const kind = firstValue(params.kind);
  const scopeKey = firstValue(params.scope_key);
  const q = firstValue(params.q);
  const includeInactive = firstValue(params.include_inactive) === "true";
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const [page, kinds, scopes] = await Promise.all([
    listMemory({
      kind,
      scope_key: scopeKey,
      q,
      include_inactive: includeInactive,
      limit: LIMIT,
      offset,
    }),
    listMemoryKinds(),
    listScopes(),
  ]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Memory</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Durable facts, one conversation at a time. Updates supersede rather than overwrite,
          so an entry&apos;s history stays readable.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <FilterBar
          filters={[
            {
              param: "kind",
              label: "Kind",
              options: kinds.items.map((item) => ({
                value: item.kind,
                label: `${item.kind} (${item.count})`,
              })),
            },
            {
              param: "scope_key",
              label: "Conversation",
              options: scopes.items.map((scope) => ({
                value: scope.scope_key,
                label: `${scope.label} (${scope.kind})`,
              })),
            },
            {
              param: "include_inactive",
              label: "Superseded",
              options: [{ value: "true", label: "include" }],
            },
          ]}
        />
        <SearchBox placeholder="Search memory content…" />
      </div>

      <MemoryTable memories={page.items} />

      <Pager
        total={page.total}
        limit={page.limit}
        offset={page.offset}
        basePath="/memory"
        searchParams={{
          kind,
          scope_key: scopeKey,
          q,
          include_inactive: includeInactive ? "true" : undefined,
        }}
      />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
