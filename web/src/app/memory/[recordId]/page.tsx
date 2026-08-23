import { Card, CardContent } from "@/components/ui/card";
import { SourceTypeBadge } from "@/components/ui/source-type-badge";
import { StatusPill } from "@/components/ui/status-pill";
import { ApiError, getMemoryRecord } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";
import { notFound } from "next/navigation";
import type { ReactNode } from "react";

export default async function MemoryRecordPage({
  params,
}: {
  params: Promise<{ recordId: string }>;
}) {
  const { recordId } = await params;

  let detail;
  try {
    detail = await getMemoryRecord(recordId);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-xs text-gray-400">{detail.record.id}</p>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{detail.record.kind}</h1>
          <StatusPill status={detail.record.status} />
        </div>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {detail.record.visibility.replaceAll("_", " ")} · namespace {detail.record.namespace_id}
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Tag>{detail.record.kind}</Tag>
          <Tag>{detail.record.visibility.replaceAll("_", " ")}</Tag>
          <Tag>{detail.record.scope_label}</Tag>
        </div>
      </div>

      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label="Current revision" value={String(detail.record.current_revision)} />
          <Field label="Generation" value={String(detail.record.generation)} />
          <Field label="Created" value={formatDateTime(detail.record.created_at)} />
          <Field label="Sources" value={String(detail.sources.length)} />
        </CardContent>
      </Card>

      {detail.sources.length > 0 ? (
        <div>
          <h2 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">Sources</h2>
          <ul className="flex flex-wrap gap-2">
            {detail.sources.map((source) => (
              <li
                key={source.id}
                className="rounded-md border border-gray-200 px-2 py-1 text-xs text-gray-600 dark:border-gray-800 dark:text-gray-400"
              >
                {source.source_kind}: {source.reference}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      <div>
        <h2 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Write history ({detail.revisions.length} revisions)
        </h2>
        <ol className="space-y-3 border-l border-gray-200 pl-4 dark:border-gray-800">
          {detail.revisions.map((revision) => (
            <li key={revision.number} className="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
              <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium text-gray-700 dark:text-gray-300">rev {revision.number}</span>
                <StatusPill status={revision.status} />
                <SourceTypeBadge sourceType={revision.source_type} />
                {revision.supersedes_revision ? (
                  <span className="text-gray-400">supersedes rev {revision.supersedes_revision}</span>
                ) : null}
                <span className="text-gray-400">by {revision.actor_id}</span>
                <span className="ml-auto text-gray-400">{formatDateTime(revision.recorded_at)}</span>
              </div>
              <p className="text-sm whitespace-pre-wrap text-gray-800 dark:text-gray-200">{revision.content}</p>
              <p className="mt-2 text-xs text-gray-400">reason: {revision.reason}</p>
              <p className="text-xs text-gray-400">
                sensitivity {revision.sensitivity.toFixed(2)} · valid from {formatDateTime(revision.valid_from)}
                {revision.valid_until ? ` to ${formatDateTime(revision.valid_until)}` : ""}
                {revision.expires_at ? ` · expires ${formatDateTime(revision.expires_at)}` : ""}
              </p>
            </li>
          ))}
        </ol>
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

function Tag({ children }: { children: ReactNode }) {
  return (
    <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-400">
      {children}
    </span>
  );
}
