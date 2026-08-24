import { StepTrace } from "@/components/dashboard/step-trace";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/ui/status-pill";
import { ApiError, getRun } from "@/lib/api";
import {
  formatCost,
  formatDateTime,
  formatNumber,
  formatSeconds,
  scopeKind,
} from "@/lib/utils";
import Link from "next/link";
import { notFound } from "next/navigation";

export default async function RunDetailPage(props: PageProps<"/runs/[runId]">) {
  const { runId } = await props.params;

  let run;
  try {
    run = await getRun(runId);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  const toolSteps = run.steps.filter((step) => step.kind === "tool");
  const failedTools = toolSteps.filter((step) => !step.ok).length;

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-xs text-gray-400">{run.id}</p>
        <h1 className="mt-1 text-xl font-semibold text-gray-900 dark:text-gray-100">
          {run.question || "(no question recorded)"}
        </h1>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-gray-400">
          <StatusPill status={run.status} />
          {run.conversation ? (
            <Link
              href={`/conversations/${encodeURIComponent(run.scope_key)}`}
              className="hover:underline"
            >
              {scopeKind(run.conversation.kind)} · {run.scope_key}
            </Link>
          ) : (
            <span className="font-mono">{run.scope_key}</span>
          )}
          {run.actor_id ? <span>asked by {run.actor_id}</span> : null}
          <span>{run.model}</span>
        </div>
      </div>

      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4 lg:grid-cols-7">
          <Field label="Started" value={formatDateTime(run.started_at)} />
          <Field label="Duration" value={formatSeconds(run.duration_seconds)} />
          <Field label="Model turns" value={formatNumber(run.turns)} />
          <Field
            label="Tool calls"
            value={failedTools ? `${run.tool_calls} (${failedTools} failed)` : formatNumber(run.tool_calls)}
          />
          <Field
            label="Tokens"
            value={`${formatNumber(run.prompt_tokens)} in / ${formatNumber(run.completion_tokens)} out`}
          />
          <Field label="Cost" value={formatCost(run.cost)} />
          <Field label="Memories written" value={formatNumber(run.memories_written)} />
        </CardContent>
      </Card>

      {run.error ? (
        <Card className="border-red-200 dark:border-red-900/50">
          <CardContent>
            <p className="mb-1 text-xs font-semibold tracking-wide text-red-500 uppercase">
              Why this run produced no answer
            </p>
            <p className="font-mono text-sm break-all text-red-700 dark:text-red-400">
              {run.error}
            </p>
          </CardContent>
        </Card>
      ) : null}

      {run.answer ? (
        <Card>
          <CardHeader>
            <CardTitle>Answer</CardTitle>
          </CardHeader>
          <CardContent>
            <p className="text-sm leading-relaxed whitespace-pre-wrap text-gray-700 dark:text-gray-300">
              {run.answer}
            </p>
          </CardContent>
        </Card>
      ) : null}

      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            Reason–act trace
          </h2>
          <p className="text-xs text-gray-400">
            {run.steps.length} steps · {toolSteps.length} tool calls
          </p>
        </div>
        <StepTrace steps={run.steps} />
      </div>
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
