"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { KpiCard } from "@/components/ui/kpi-card";
import { getOverview } from "@/lib/api";
import type { OverviewResponse } from "@/lib/types";
import { formatCost, formatMs, formatNumber, formatPercent, formatSeconds } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const STATUS_COLORS: Record<string, string> = {
  answered: "#10b981",
  failed: "#ef4444",
  running: "#3b82f6",
};

export function OverviewClient({ initial }: { initial: OverviewResponse }) {
  const { data: overview } = useQuery({
    queryKey: ["overview"],
    queryFn: () => getOverview(),
    initialData: initial,
    refetchInterval: 15_000,
  });

  const statusData = Object.entries(overview.run_status_counts).map(([key, value]) => ({
    key,
    name: key.replaceAll("_", " "),
    value,
  }));

  const activity = overview.activity.map((point) => ({
    day: point.day ? new Date(point.day).toLocaleDateString("en-US", { month: "short", day: "numeric" }) : "",
    runs: point.runs,
    answered: point.answered,
  }));

  const tools = overview.tool_usage.slice(0, 12).map((tool) => ({
    name: tool.name,
    calls: tool.calls,
    failed: tool.failed,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
        <KpiCard
          label="Answer rate"
          value={formatPercent(overview.answer_rate)}
          hint={`${formatNumber(overview.answered_runs)} of ${formatNumber(overview.total_runs)} runs`}
          tone={
            overview.answer_rate === null
              ? "default"
              : overview.answer_rate >= 0.95
                ? "good"
                : overview.answer_rate < 0.8
                  ? "bad"
                  : "default"
          }
        />
        <KpiCard
          label="Tool calls"
          value={formatNumber(overview.total_tool_calls)}
          hint={`${formatNumber(overview.total_model_turns)} model turns`}
        />
        <KpiCard
          label="Median run"
          value={formatSeconds(overview.p50_run_seconds)}
          hint={`p95 ${formatSeconds(overview.p95_run_seconds)}`}
        />
        <KpiCard
          label="Total cost"
          value={formatCost(overview.total_cost)}
          hint={`${formatNumber(overview.total_tokens)} tokens`}
        />
        <KpiCard
          label="Conversations"
          value={formatNumber(overview.conversations)}
          hint={`${formatNumber(overview.messages)} messages`}
        />
        <KpiCard
          label="Memories"
          value={formatNumber(overview.active_memories)}
          hint="active, across all scopes"
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Runs per day</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            {activity.length === 0 ? (
              <Empty>No runs in this window.</Empty>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={activity} margin={{ left: -16, right: 8, top: 8 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    className="stroke-gray-200 dark:stroke-gray-800"
                  />
                  <XAxis dataKey="day" tick={{ fontSize: 12 }} />
                  <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
                  <Tooltip />
                  <Legend />
                  <Area
                    type="monotone"
                    dataKey="runs"
                    name="runs"
                    stroke="#6366f1"
                    fill="#6366f1"
                    fillOpacity={0.15}
                  />
                  <Area
                    type="monotone"
                    dataKey="answered"
                    name="answered"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Run outcomes</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            {statusData.length === 0 ? (
              <Empty>No runs recorded.</Empty>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={statusData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={45}
                    outerRadius={75}
                    paddingAngle={2}
                  >
                    {statusData.map((entry) => (
                      <Cell key={entry.key} fill={STATUS_COLORS[entry.key] ?? "#9ca3af"} />
                    ))}
                  </Pie>
                  <Legend />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Most-used tools</CardTitle>
          </CardHeader>
          <CardContent className="h-72">
            {tools.length === 0 ? (
              <Empty>No tool calls yet.</Empty>
            ) : (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={tools} layout="vertical" margin={{ left: 60 }}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    horizontal={false}
                    className="stroke-gray-200 dark:stroke-gray-800"
                  />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
                  <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 11 }} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="calls" name="calls" fill="#6366f1" radius={[0, 4, 4, 0]} />
                  <Bar dataKey="failed" name="failed" fill="#ef4444" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Tool failures by code</CardTitle>
          </CardHeader>
          <CardContent>
            {overview.tool_errors.length === 0 ? (
              <p className="py-8 text-center text-sm text-gray-400">
                No tool call has failed.
              </p>
            ) : (
              <>
                <p className="mb-3 text-xs text-gray-400">
                  A failed tool call is returned to the model as a message, not treated as a
                  run failure — these are recoveries, not outages.
                </p>
                <ul className="divide-y divide-gray-100 dark:divide-gray-800">
                  {overview.tool_errors.map((error) => (
                    <li
                      key={error.code}
                      className="flex items-center justify-between py-2 text-sm"
                    >
                      <span className="font-mono text-xs text-gray-700 dark:text-gray-300">
                        {error.code}
                      </span>
                      <span className="font-medium text-gray-900 tabular-nums dark:text-gray-100">
                        {error.count}
                      </span>
                    </li>
                  ))}
                </ul>
              </>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Tool reliability</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 text-sm dark:divide-gray-800">
              <thead className="bg-gray-50 dark:bg-gray-900">
                <tr>
                  <Th>Tool</Th>
                  <Th>Calls</Th>
                  <Th>Failed</Th>
                  <Th>Success</Th>
                  <Th>Avg latency</Th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                {overview.tool_usage.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-gray-400">
                      No tool calls recorded.
                    </td>
                  </tr>
                ) : (
                  overview.tool_usage.map((tool) => (
                    <tr key={tool.name}>
                      <td className="px-4 py-2 font-mono text-xs text-gray-700 dark:text-gray-300">
                        {tool.name}
                      </td>
                      <td className="px-4 py-2 tabular-nums">{formatNumber(tool.calls)}</td>
                      <td className="px-4 py-2 tabular-nums">
                        {tool.failed > 0 ? (
                          <span className="text-red-600 dark:text-red-400">{tool.failed}</span>
                        ) : (
                          "—"
                        )}
                      </td>
                      <td className="px-4 py-2 tabular-nums">
                        {formatPercent(tool.calls ? tool.succeeded / tool.calls : null)}
                      </td>
                      <td className="px-4 py-2 tabular-nums">{formatMs(tool.avg_ms)}</td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function Th({ children }: { children: React.ReactNode }) {
  return (
    <th className="px-4 py-2 text-left text-xs font-semibold tracking-wide text-gray-500 uppercase dark:text-gray-400">
      {children}
    </th>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center text-sm text-gray-400">{children}</div>
  );
}
