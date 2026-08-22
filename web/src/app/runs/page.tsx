import { FilterBar } from "@/components/dashboard/filter-bar";
import { RunsTable } from "@/components/dashboard/runs-table";
import { Pager } from "@/components/ui/pager";
import { listRuns } from "@/lib/api";

const RUN_STATUSES = [
  "queued",
  "running",
  "requires_action",
  "completed",
  "failed",
  "cancelled",
  "timed_out",
  "budget_exhausted",
];
const RUN_PHASES = ["research", "proposal", "policy", "approval", "execution", "verification"];
const TASK_STATUSES = ["queued", "active", "requires_action", "completed", "failed", "cancelled"];

const LIMIT = 25;

export default async function RunsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const status = firstValue(params.status);
  const phase = firstValue(params.phase);
  const taskStatus = firstValue(params.task_status);
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const page = await listRuns({
    status,
    phase,
    task_status: taskStatus,
    limit: LIMIT,
    offset,
  });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Runs</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Every task attempt Leo&apos;s harness has executed, most recent first.
        </p>
      </div>

      <FilterBar
        filters={[
          {
            param: "status",
            label: "Run status",
            options: RUN_STATUSES.map((value) => ({ value, label: value.replaceAll("_", " ") })),
          },
          {
            param: "phase",
            label: "Phase",
            options: RUN_PHASES.map((value) => ({ value, label: value })),
          },
          {
            param: "task_status",
            label: "Task status",
            options: TASK_STATUSES.map((value) => ({ value, label: value.replaceAll("_", " ") })),
          },
        ]}
      />

      <RunsTable runs={page.items} />

      <Pager
        total={page.total}
        limit={page.limit}
        offset={page.offset}
        basePath="/runs"
        searchParams={{ status, phase, task_status: taskStatus }}
      />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
