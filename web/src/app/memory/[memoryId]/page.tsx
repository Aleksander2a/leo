import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/ui/status-pill";
import { ApiError, getMemory } from "@/lib/api";
import type { MemorySummary } from "@/lib/types";
import { formatDateTime, truncate } from "@/lib/utils";
import Link from "next/link";
import { notFound } from "next/navigation";

export default async function MemoryDetailPage(props: PageProps<"/memory/[memoryId]">) {
  const { memoryId } = await props.params;

  let memory;
  try {
    memory = await getMemory(decodeURIComponent(memoryId));
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  // Oldest first: what this replaced, then this, then what replaced it.
  const timeline = [
    ...memory.supersedes.map((row) => ({ row, current: false })),
    { row: memory as MemorySummary, current: true },
    ...memory.superseded_chain.map((row) => ({ row, current: false })),
  ];

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-xs text-gray-400">{memory.id}</p>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            {memory.subject || memory.kind}
          </h1>
          <StatusPill status={memory.kind} />
          <StatusPill status={memory.active ? "active" : "superseded"} />
        </div>
        <Link
          href={`/conversations/${encodeURIComponent(memory.scope_key)}`}
          className="mt-1 inline-block font-mono text-xs text-blue-600 hover:underline dark:text-blue-400"
        >
          {memory.scope_key}
        </Link>
      </div>

      <Card>
        <CardContent>
          <p className="text-base leading-relaxed text-gray-800 dark:text-gray-200">
            {memory.content}
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label="Importance" value={"★".repeat(memory.importance)} />
          <Field label="Written by" value={memory.author_id ?? "—"} />
          <Field label="Created" value={formatDateTime(memory.created_at)} />
          <Field
            label="Embedded"
            value={memory.embedded ? "yes — findable by meaning" : "no — recency only"}
          />
        </CardContent>
      </Card>

      {memory.source_run ? (
        <Card>
          <CardHeader>
            <CardTitle>Written during</CardTitle>
          </CardHeader>
          <CardContent className="p-0">
            <Link
              href={`/runs/${memory.source_run.id}`}
              className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-900"
            >
              <StatusPill status={memory.source_run.status} />
              <span className="min-w-0 flex-1 truncate text-sm text-gray-700 dark:text-gray-300">
                {truncate(memory.source_run.question, 90)}
              </span>
              <span className="shrink-0 text-xs text-gray-400">
                {formatDateTime(memory.source_run.started_at)}
              </span>
            </Link>
          </CardContent>
        </Card>
      ) : null}

      <div>
        <h2 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Revision history ({timeline.length})
        </h2>
        {timeline.length === 1 ? (
          <p className="text-sm text-gray-400">
            This entry has never been revised.
          </p>
        ) : (
          <ol className="space-y-3 border-l border-gray-200 pl-4 dark:border-gray-800">
            {timeline.map(({ row, current }) => (
              <li
                key={row.id}
                className={
                  current
                    ? "rounded-lg border border-blue-300 bg-blue-50/40 p-3 dark:border-blue-800 dark:bg-blue-950/20"
                    : "rounded-lg border border-gray-200 p-3 dark:border-gray-800"
                }
              >
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <StatusPill status={row.active ? "active" : "superseded"} />
                  {current ? (
                    <span className="font-medium text-blue-700 dark:text-blue-400">
                      viewing
                    </span>
                  ) : (
                    <Link
                      href={`/memory/${encodeURIComponent(row.id)}`}
                      className="text-blue-600 hover:underline dark:text-blue-400"
                    >
                      open
                    </Link>
                  )}
                  <span className="ml-auto text-gray-400">
                    {formatDateTime(row.updated_at)}
                  </span>
                </div>
                <p className="text-sm text-gray-800 dark:text-gray-200">{row.content}</p>
              </li>
            ))}
          </ol>
        )}
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
