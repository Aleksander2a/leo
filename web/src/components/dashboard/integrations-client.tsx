"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ProvenanceBadge } from "@/components/ui/provenance-badge";
import { formatNumber, formatPercent } from "@/lib/utils";
import type { IntegrationsResponse } from "@/lib/types";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function IntegrationsClient({ data }: { data: IntegrationsResponse }) {
  const chartData = data.providers.map((provider) => ({
    name: provider.display_name,
    total: provider.total,
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Call volume by integration</CardTitle>
        </CardHeader>
        <CardContent className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} className="stroke-gray-200 dark:stroke-gray-800" />
              <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="name" width={140} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="total" fill="#6366f1" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              {["Provider", "Kind", "Total calls", "Retrieved", "Stale", "Rejected", "Success rate"].map((header) => (
                <th
                  key={header}
                  className="px-4 py-2 text-left text-xs font-semibold tracking-wide text-gray-500 uppercase dark:text-gray-400"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
            {data.providers.map((provider) => (
              <tr key={provider.provider} className="text-sm">
                <td className="px-4 py-2 font-medium text-gray-800 dark:text-gray-200">{provider.display_name}</td>
                <td className="px-4 py-2">
                  <ProvenanceBadge callKind={provider.call_kind} />
                </td>
                <td className="px-4 py-2 tabular-nums text-gray-600 dark:text-gray-400">
                  {formatNumber(provider.total)}
                </td>
                <td className="px-4 py-2 tabular-nums text-emerald-600 dark:text-emerald-400">
                  {formatNumber(provider.retrieved)}
                </td>
                <td className="px-4 py-2 tabular-nums text-amber-600 dark:text-amber-400">
                  {formatNumber(provider.stale)}
                </td>
                <td className="px-4 py-2 tabular-nums text-red-600 dark:text-red-400">
                  {formatNumber(provider.rejected)}
                </td>
                <td className="px-4 py-2 tabular-nums text-gray-600 dark:text-gray-400">
                  {formatPercent(provider.success_rate)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Tool calls that never produced an observation</CardTitle>
        </CardHeader>
        <CardContent>
          {data.tool_failures.length === 0 ? (
            <p className="text-sm text-gray-400">No tool-level failures recorded.</p>
          ) : (
            <ul className="divide-y divide-gray-100 dark:divide-gray-800">
              {data.tool_failures.map((failure) => (
                <li key={failure.key} className="flex items-center justify-between py-2 text-sm">
                  <span className="flex items-center gap-2">
                    <span className="font-mono text-gray-700 dark:text-gray-300">{failure.key}</span>
                    <ProvenanceBadge callKind={failure.call_kind} integration={failure.integration} />
                  </span>
                  <span className="font-medium tabular-nums text-gray-900 dark:text-gray-100">
                    {failure.count}
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
