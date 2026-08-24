import { FailuresTable } from "@/components/dashboard/failures-table";
import { Pager } from "@/components/ui/pager";
import { listFailures } from "@/lib/api";

const LIMIT = 25;

export default async function FailuresPage(props: PageProps<"/failures">) {
  const params = await props.searchParams;
  const raw = Array.isArray(params.offset) ? params.offset[0] : params.offset;
  const offset = Number(raw ?? "0") || 0;

  const page = await listFailures({ limit: LIMIT, offset });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Failures</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Runs that ended without an answer. A failing tool is not one of these — the loop
          hands the error back to the model and carries on.
        </p>
      </div>

      <FailuresTable failures={page.items} />

      <Pager
        total={page.total}
        limit={page.limit}
        offset={page.offset}
        basePath="/failures"
        searchParams={{}}
      />
    </div>
  );
}
