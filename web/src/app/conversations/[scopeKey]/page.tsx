import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { StatusPill } from "@/components/ui/status-pill";
import { Tabs } from "@/components/ui/tabs";
import { ApiError, getConversation } from "@/lib/api";
import type { ConversationMessage, MemorySummary, RunSummary } from "@/lib/types";
import {
  cn,
  formatCost,
  formatDateTime,
  formatNumber,
  formatRelative,
  scopeKind,
  truncate,
} from "@/lib/utils";
import Link from "next/link";
import { notFound } from "next/navigation";

export default async function ConversationPage(props: PageProps<"/conversations/[scopeKey]">) {
  const { scopeKey } = await props.params;
  const decoded = decodeURIComponent(scopeKey);

  let conversation;
  try {
    conversation = await getConversation(decoded);
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      notFound();
    }
    throw error;
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="font-mono text-xs text-gray-400">{conversation.scope_key}</p>
        <div className="mt-1 flex flex-wrap items-center gap-2">
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            {conversation.title ?? conversation.channel_id ?? conversation.scope_key}
          </h1>
          <StatusPill status={scopeKind(conversation.kind)} />
        </div>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {conversation.provider}
          {conversation.team_id ? ` · team ${conversation.team_id}` : ""} · first seen{" "}
          {formatDateTime(conversation.created_at)}
        </p>
      </div>

      <Card>
        <CardContent className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Field label="Messages" value={formatNumber(conversation.messages)} />
          <Field label="Runs" value={formatNumber(conversation.runs)} />
          <Field label="Active memories" value={formatNumber(conversation.memories)} />
          <Field label="Last active" value={formatRelative(conversation.last_active_at)} />
        </CardContent>
      </Card>

      <Tabs
        tabs={[
          {
            id: "transcript",
            label: `Transcript (${conversation.recent_messages.length})`,
            content: <Transcript messages={conversation.recent_messages} />,
          },
          {
            id: "runs",
            label: `Runs (${conversation.recent_runs.length})`,
            content: <RunList runs={conversation.recent_runs} />,
          },
          {
            id: "memory",
            label: `Memory (${conversation.recent_memories.length})`,
            content: <MemoryList memories={conversation.recent_memories} />,
          },
        ]}
      />
    </div>
  );
}

function Transcript({ messages }: { messages: ConversationMessage[] }) {
  if (messages.length === 0) {
    return <p className="py-8 text-center text-sm text-gray-400">No messages recorded.</p>;
  }
  return (
    <ol className="space-y-3">
      {messages.map((message) => (
        <li
          key={message.id}
          className={cn(
            "rounded-lg border p-3",
            message.role === "assistant"
              ? "border-blue-200 bg-blue-50/40 dark:border-blue-900/50 dark:bg-blue-950/20"
              : "border-gray-200 dark:border-gray-800",
          )}
        >
          <div className="mb-1.5 flex flex-wrap items-center gap-2 text-xs text-gray-400">
            <StatusPill status={message.role} />
            {message.author_id ? <span>{message.author_id}</span> : null}
            {message.run_id ? (
              <Link href={`/runs/${message.run_id}`} className="hover:underline">
                view run
              </Link>
            ) : null}
            <span className="ml-auto">{formatDateTime(message.created_at)}</span>
          </div>
          <p className="text-sm leading-relaxed whitespace-pre-wrap text-gray-700 dark:text-gray-300">
            {message.content}
          </p>
        </li>
      ))}
    </ol>
  );
}

function RunList({ runs }: { runs: RunSummary[] }) {
  if (runs.length === 0) {
    return <p className="py-8 text-center text-sm text-gray-400">No runs in this conversation.</p>;
  }
  return (
    <Card>
      <CardContent className="p-0">
        <ul className="divide-y divide-gray-100 dark:divide-gray-800">
          {runs.map((run) => (
            <li key={run.id}>
              <Link
                href={`/runs/${run.id}`}
                className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-900"
              >
                <StatusPill status={run.status} />
                <span className="min-w-0 flex-1 truncate text-sm text-gray-700 dark:text-gray-300">
                  {truncate(run.question, 80)}
                </span>
                <span className="shrink-0 text-xs text-gray-400">
                  {run.turns} turns · {run.tool_calls} tools
                </span>
                <span className="shrink-0 text-xs text-gray-400">{formatCost(run.cost)}</span>
                <span className="shrink-0 text-xs text-gray-400">
                  {formatRelative(run.started_at)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
  );
}

function MemoryList({ memories }: { memories: MemorySummary[] }) {
  if (memories.length === 0) {
    return (
      <div className="py-8 text-center">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Leo remembers nothing about this conversation yet.
        </p>
        <p className="mt-1 text-xs text-gray-400">
          Memories are written when someone states something durable — a preference, a
          holding, a constraint.
        </p>
      </div>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle>What Leo remembers here — and nowhere else</CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <ul className="divide-y divide-gray-100 dark:divide-gray-800">
          {memories.map((memory) => (
            <li key={memory.id}>
              <Link
                href={`/memory/${encodeURIComponent(memory.id)}`}
                className="flex flex-wrap items-start gap-3 px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-900"
              >
                <StatusPill status={memory.kind} />
                <span className="min-w-0 flex-1">
                  {memory.subject ? (
                    <span className="block text-xs font-medium text-gray-500">
                      {memory.subject}
                    </span>
                  ) : null}
                  <span className="block text-sm text-gray-700 dark:text-gray-300">
                    {memory.content}
                  </span>
                </span>
                <span className="shrink-0 text-xs text-gray-400">
                  {formatRelative(memory.updated_at)}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      </CardContent>
    </Card>
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
