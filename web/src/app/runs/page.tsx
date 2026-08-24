import { FilterBar } from "@/components/dashboard/filter-bar";
import { RunsTable } from "@/components/dashboard/runs-table";
import { SearchBox } from "@/components/dashboard/search-box";
import { Pager } from "@/components/ui/pager";
import { listRuns, listScopes } from "@/lib/api";

const RUN_STATUSES = ["answered", "failed", "running"];
const LIMIT = 25;

export default async function RunsPage(props: PageProps<"/runs">) {
  const params = await props.searchParams;
  const status = firstValue(params.status);
  const scopeKey = firstValue(params.scope_key);
  const q = firstValue(params.q);
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const [page, scopes] = await Promise.all([
    listRuns({ status, scope_key: scopeKey, q, limit: LIMIT, offset }),
    listScopes(),
  ]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Runs</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Every question Leo has handled, most recent first. Open one to see the full
          reason-act trace.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <FilterBar
          filters={[
            {
              param: "status",
              label: "Status",
              options: RUN_STATUSES.map((value) => ({ value, label: value })),
            },
            {
              param: "scope_key",
              label: "Conversation",
              options: scopes.items.map((scope) => ({
                value: scope.scope_key,
                label: `${scope.label} (${scope.kind})`,
              })),
            },
          ]}
        />
        <SearchBox placeholder="Search questions and answers…" />
      </div>

      <RunsTable runs={page.items} />

      <Pager
        total={page.total}
        limit={page.limit}
        offset={page.offset}
        basePath="/runs"
        searchParams={{ status, scope_key: scopeKey, q }}
      />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
