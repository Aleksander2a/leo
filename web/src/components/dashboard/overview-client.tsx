"use client";

import { KpiCard } from "@/components/ui/kpi-card";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getIntegrations, getOverview } from "@/lib/api";
import { formatCost, formatNumber, formatPercent, formatSeconds } from "@/lib/utils";
import type { IntegrationsResponse, OverviewResponse } from "@/lib/types";
import { useQuery } from "@tanstack/react-query";
import {
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
  completed: "#10b981",
  failed: "#ef4444",
  cancelled: "#f59e0b",
  timed_out: "#f59e0b",
  budget_exhausted: "#f59e0b",
  running: "#3b82f6",
  queued: "#94a3b8",
  requires_action: "#8b5cf6",
};

function toChartData(counts: Record<string, number>) {
  return Object.entries(counts).map(([key, value]) => ({ name: key.replaceAll("_", " "), key, value }));
}

export function OverviewClient({
  initialOverview,
  initialIntegrations,
}: {
  initialOverview: OverviewResponse;
  initialIntegrations: IntegrationsResponse;
}) {
  const overviewQuery = useQuery({
    queryKey: ["overview"],
    queryFn: getOverview,
    initialData: initialOverview,
    refetchInterval: 15_000,
  });
  const integrationsQuery = useQuery({
    queryKey: ["integrations"],
    queryFn: getIntegrations,
    initialData: initialIntegrations,
    refetchInterval: 30_000,
  });

  const overview = overviewQuery.data;
  const integrations = integrationsQuery.data;

  const runStatusData = toChartData(overview.run_status_counts);
  const providerData = integrations.providers.map((provider) => ({
    name: provider.display_name,
    total: provider.total,
    successRate: provider.success_rate,
  }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
        <KpiCard
          label="Tool call success rate"
          value={formatPercent(overview.tool_call_success_rate)}
          hint={`${formatNumber(overview.tool_calls.completed)} of ${formatNumber(
            overview.tool_calls.completed + overview.tool_calls.failed,
          )} finished calls`}
          tone={
            overview.tool_call_success_rate === null
              ? "default"
              : overview.tool_call_success_rate >= 0.9
                ? "good"
                : overview.tool_call_success_rate < 0.7
                  ? "bad"
                  : "default"
          }
        />
        <KpiCard
          label="Model calls"
          value={formatNumber(overview.total_model_calls)}
          hint={`${formatNumber(overview.total_tokens)} tokens`}
        />
        <KpiCard label="Total cost" value={formatCost(overview.total_cost)} hint="cumulative, all runs" />
        <KpiCard
          label="Avg run latency"
          value={formatSeconds(overview.avg_run_latency_seconds)}
          hint="terminal runs only"
        />
        <KpiCard
          label="Memory writes"
          value={formatNumber(overview.memory_writes_total)}
          hint={`${formatNumber(overview.memory_pages_referenced_total)} pages referenced`}
        />
      </div>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Run outcomes</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={runStatusData}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={50}
                  outerRadius={80}
                  paddingAngle={2}
                >
                  {runStatusData.map((entry) => (
                    <Cell key={entry.key} fill={STATUS_COLORS[entry.key] ?? "#9ca3af"} />
                  ))}
                </Pie>
                <Legend layout="vertical" verticalAlign="middle" align="right" />
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Calls per integration</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={providerData} layout="vertical" margin={{ left: 24 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={false} className="stroke-gray-200 dark:stroke-gray-800" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="name" width={110} tick={{ fontSize: 12 }} />
                <Tooltip />
                <Bar dataKey="total" fill="#6366f1" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Top failure reasons</CardTitle>
        </CardHeader>
        <CardContent>
          {overview.failure_reasons.length === 0 ? (
            <p className="text-sm text-gray-400">No failed runs recorded.</p>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-gray-800">
              {overview.failure_reasons.map((reason) => (
                <li key={reason.key} className="flex items-center justify-between py-2 text-sm">
                  <span className="text-gray-700 dark:text-gray-300">{reason.key}</span>
                  <span className="font-medium tabular-nums text-gray-900 dark:text-gray-100">
                    {reason.count}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
