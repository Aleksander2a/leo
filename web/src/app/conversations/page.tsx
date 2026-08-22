import { Pager } from "@/components/ui/pager";
import { StatusPill } from "@/components/ui/status-pill";
import { listConversations } from "@/lib/api";
import { formatDateTime } from "@/lib/utils";

const LIMIT = 25;

export default async function ConversationsPage({
  searchParams,
}: {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}) {
  const params = await searchParams;
  const offset = Number(firstValue(params.offset) ?? "0") || 0;
  const page = await listConversations({ limit: LIMIT, offset });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Conversations</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Slack conversations Leo is present in, and how many threads it has tracked in each.
        </p>
      </div>

      <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-800">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-800">
          <thead className="bg-gray-50 dark:bg-gray-900">
            <tr>
              {["Kind", "Bot presence", "Lifecycle", "Provenance", "Threads", "Updated"].map((header) => (
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
            {page.items.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-sm text-gray-400">
                  No conversations recorded.
                </td>
              </tr>
            ) : (
              page.items.map((conversation) => (
                <tr key={conversation.id} className="text-sm">
                  <td className="px-4 py-2 text-gray-800 dark:text-gray-200">{conversation.kind}</td>
                  <td className="px-4 py-2">
                    <StatusPill status={conversation.bot_presence} />
                  </td>
                  <td className="px-4 py-2">
                    <StatusPill status={conversation.lifecycle} />
                  </td>
                  <td className="px-4 py-2 text-gray-500 dark:text-gray-400">{conversation.external_provenance}</td>
                  <td className="px-4 py-2 tabular-nums text-gray-600 dark:text-gray-400">
                    {conversation.thread_count}
                  </td>
                  <td className="px-4 py-2 text-gray-500 dark:text-gray-400">
                    {formatDateTime(conversation.updated_at)}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      <Pager total={page.total} limit={page.limit} offset={page.offset} basePath="/conversations" searchParams={{}} />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
