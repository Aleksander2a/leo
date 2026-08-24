"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { KpiCard } from "@/components/ui/kpi-card";
import { listTools } from "@/lib/api";
import type { ToolInfo } from "@/lib/types";
import { cn, formatMs, formatNumber, formatPercent, formatRelative } from "@/lib/utils";
import { useQuery } from "@tanstack/react-query";
import { useMemo, useState } from "react";

/**
 * Every tool Leo can reach, grouped by domain. "Indexed" means its description
 * has an embedding, which is what makes it discoverable — a tool without one is
 * only reachable if it happens to be in the always-available core set.
 */
export function ToolsClient({ initial }: { initial: ToolInfo[] }) {
  const { data } = useQuery({
    queryKey: ["tools"],
    queryFn: listTools,
    initialData: { items: initial },
    refetchInterval: 30_000,
  });
  const tools = data.items;
  const [domain, setDomain] = useState<string>("");

  const domains = useMemo(
    () => Array.from(new Set(tools.map((tool) => tool.domain))).sort(),
    [tools],
  );
  const visible = domain ? tools.filter((tool) => tool.domain === domain) : tools;

  const indexed = tools.filter((tool) => tool.indexed).length;
  const used = tools.filter((tool) => tool.calls > 0).length;
  const totalCalls = tools.reduce((sum, tool) => sum + tool.calls, 0);
  const totalFailed = tools.reduce((sum, tool) => sum + tool.failed, 0);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <KpiCard label="Tools available" value={formatNumber(tools.length)} hint={`${domains.length} domains`} />
        <KpiCard
          label="Semantically indexed"
          value={formatNumber(indexed)}
          hint="have a description embedding"
          tone={indexed === tools.length ? "good" : "default"}
        />
        <KpiCard label="Used at least once" value={formatNumber(used)} hint="since records began" />
        <KpiCard
          label="Call success rate"
          value={formatPercent(totalCalls ? (totalCalls - totalFailed) / totalCalls : null)}
          hint={`${formatNumber(totalCalls)} calls`}
          tone={totalCalls && totalFailed / totalCalls > 0.2 ? "bad" : "good"}
        />
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <FilterChip active={domain === ""} onClick={() => setDomain("")}>
          all
        </FilterChip>
        {domains.map((name) => (
          <FilterChip key={name} active={domain === name} onClick={() => setDomain(name)}>
            {name}
          </FilterChip>
        ))}
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        {visible.map((tool) => (
          <ToolCard key={tool.name} tool={tool} />
        ))}
      </div>
    </div>
  );
}

function ToolCard({ tool }: { tool: ToolInfo }) {
  const rate = tool.calls ? tool.succeeded / tool.calls : null;
  return (
    <Card>
      <CardHeader className="flex flex-wrap items-center justify-between gap-2">
        <CardTitle className="font-mono">{tool.name}</CardTitle>
        <div className="flex items-center gap-1.5">
          {tool.indexed ? (
            <span className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 dark:bg-emerald-500/10 dark:text-emerald-400">
              indexed
            </span>
          ) : (
            <span className="rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
              not indexed
            </span>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {tool.description ?? (
            <span className="text-gray-400 italic">
              Not in the embedding index — seen only in the run trace.
            </span>
          )}
        </p>
        <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Stat label="Calls" value={formatNumber(tool.calls)} />
          <Stat
            label="Success"
            value={formatPercent(rate)}
            tone={rate !== null && rate < 0.8 ? "bad" : undefined}
          />
          <Stat label="Avg latency" value={formatMs(tool.avg_ms)} />
          <Stat label="Last used" value={formatRelative(tool.last_used_at)} />
        </dl>
        {tool.errors.length > 0 ? (
          <div>
            <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">
              Failure codes
            </p>
            <div className="flex flex-wrap gap-1.5">
              {tool.errors.map((error) => (
                <span
                  key={error.code}
                  className="rounded-md bg-red-50 px-2 py-0.5 font-mono text-xs text-red-700 dark:bg-red-500/10 dark:text-red-400"
                >
                  {error.code} × {error.count}
                </span>
              ))}
            </div>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: "bad" }) {
  return (
    <div>
      <dt className="text-xs text-gray-400">{label}</dt>
      <dd
        className={cn(
          "text-sm font-medium tabular-nums",
          tone === "bad" ? "text-red-600 dark:text-red-400" : "text-gray-800 dark:text-gray-200",
        )}
      >
        {value}
      </dd>
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded-full px-3 py-1 text-xs font-medium transition-colors",
        active
          ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
          : "bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400 dark:hover:bg-gray-700",
      )}
    >
      {children}
    </button>
  );
}
