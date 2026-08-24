import { OverviewClient } from "@/components/dashboard/overview-client";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/ui/status-pill";
import { getOverview, listRuns } from "@/lib/api";
import { formatCost, formatNumber, formatRelative, truncate } from "@/lib/utils";
import Link from "next/link";

export default async function OverviewPage() {
  const [overview, recentRuns] = await Promise.all([getOverview(), listRuns({ limit: 8 })]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Overview</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Every question Leo has been asked, what it did about it, and what that cost.
        </p>
      </div>

      <OverviewClient initial={overview} />

      <Card>
        <CardHeader className="flex items-center justify-between">
          <CardTitle>Recent runs</CardTitle>
          <Link
            href="/runs"
            className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
          >
            View all
          </Link>
        </CardHeader>
        <CardContent className="p-0">
          {recentRuns.items.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-gray-400">
              Nothing yet. Ask Leo something in Slack, or run{" "}
              <code className="font-mono">leo ask &quot;…&quot;</code>.
            </p>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-gray-800">
              {recentRuns.items.map((run) => (
                <li key={run.id}>
                  <Link
                    href={`/runs/${run.id}`}
                    className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-900"
                  >
                    <StatusPill status={run.status} />
                    <span className="min-w-0 flex-1 truncate text-sm text-gray-700 dark:text-gray-300">
                      {truncate(run.question, 90)}
                    </span>
                    <span className="shrink-0 text-xs text-gray-400">
                      {run.turns} turns · {run.tool_calls} tools
                    </span>
                    <span className="shrink-0 text-xs text-gray-400">
                      {formatNumber(run.total_tokens)} tok
                    </span>
                    <span className="shrink-0 text-xs text-gray-400">{formatCost(run.cost)}</span>
                    <span className="shrink-0 text-xs text-gray-400">
                      {formatRelative(run.started_at)}
                    </span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
