import { FilterBar } from "@/components/dashboard/filter-bar";
import { FailuresTable } from "@/components/dashboard/failures-table";
import { Pager } from "@/components/ui/pager";
import { listFailures } from "@/lib/api";

const STATUSES = ["failed", "cancelled", "timed_out", "budget_exhausted"];

const LIMIT = 25;

export default async function FailuresPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const status = firstValue(params.status);
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const page = await listFailures({ status, limit: LIMIT, offset });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Failures</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Every run that did not complete successfully, most recently updated first.
        </p>
      </div>

      <FilterBar
        filters={[
          {
            param: "status",
            label: "Status",
            options: STATUSES.map((value) => ({ value, label: value.replaceAll("_", " ") })),
          },
        ]}
      />

      <FailuresTable failures={page.items} />

      <Pager total={page.total} limit={page.limit} offset={page.offset} basePath="/failures" searchParams={{ status }} />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
