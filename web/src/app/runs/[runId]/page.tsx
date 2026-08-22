import { RunDetailTabs } from "@/components/dashboard/run-detail-tabs";
import { Card, CardContent } from "@/components/ui/card";
import { StatusPill } from "@/components/ui/status-pill";
import { ApiError, getRunDetail, getRunPlanTree, getRunTimeline } from "@/lib/api";
import { formatCost, formatDateTime, formatNumber } from "@/lib/utils";
import { notFound } from "next/navigation";

export default async function RunDetailPage({
  params,
}: {
  params: Promise<{ runId: string }>;
}) {
  const { runId } = await params;

  let detail;
  let timeline;
  let planTree;
  try {
    [detail, timeline, planTree] = await Promise.all([
      getRunDetail(runId),
      getRunTimeline(runId),
      getRunPlanTree(runId),
    ]);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  const usage = detail.run.usage as {
    total_tokens?: number;
    cost?: number;
    model_calls?: number;
    tool_calls?: number;
  };

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-xs text-gray-400">{detail.run.id}</p>
        <h1 className="mt-1 text-xl font-semibold text-gray-900 dark:text-gray-100">
          {detail.task?.objective ?? "Untitled task"}
        </h1>
        <div className="mt-2 flex flex-wrap items-center gap-2">
          <StatusPill status={detail.run.status} />
          <span className="text-xs text-gray-400">phase: {detail.run.phase}</span>
          <span className="text-xs text-gray-400">iteration {detail.run.iteration}</span>
          {detail.run.terminal_reason ? (
            <span className="text-xs text-gray-400">reason: {detail.run.terminal_reason}</span>
          ) : null}
        </div>
      </div>

      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-6">
          <Field label="Started" value={formatDateTime(detail.run.started_at)} />
          <Field label="Updated" value={formatDateTime(detail.run.updated_at)} />
          <Field label="Model calls" value={formatNumber(usage?.model_calls ?? null)} />
          <Field label="Tool calls" value={formatNumber(usage?.tool_calls ?? null)} />
          <Field label="Tokens" value={formatNumber(usage?.total_tokens ?? null)} />
          <Field label="Cost" value={formatCost(usage?.cost ?? null)} />
        </CardContent>
      </Card>

      {detail.run.final_output ? (
        <Card>
          <CardContent>
            <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">Final output</p>
            <p className="text-sm whitespace-pre-wrap text-gray-700 dark:text-gray-300">
              {detail.run.final_output}
            </p>
          </CardContent>
        </Card>
      ) : null}

      <RunDetailTabs detail={detail} timeline={timeline} planTree={planTree} />
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-sm font-medium text-gray-800 dark:text-gray-200">{value}</p>
    </div>
  );
}
