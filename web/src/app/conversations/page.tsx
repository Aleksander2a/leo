import { ConversationsTable } from "@/components/dashboard/conversations-table";
import { FilterBar } from "@/components/dashboard/filter-bar";
import { Pager } from "@/components/ui/pager";
import { listConversations } from "@/lib/api";

const KINDS = ["channel", "dm", "cli"];
const LIMIT = 25;

export default async function ConversationsPage(props: PageProps<"/conversations">) {
  const params = await props.searchParams;
  const kind = firstValue(params.kind);
  const offset = Number(firstValue(params.offset) ?? "0") || 0;

  const page = await listConversations({ kind, limit: LIMIT, offset });

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Conversations</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Each channel and DM is a separate scope. History and memory are read with that scope
          in the WHERE clause, so nothing crosses between them.
        </p>
      </div>

      <FilterBar
        filters={[
          {
            param: "kind",
            label: "Kind",
            options: KINDS.map((value) => ({ value, label: value })),
          },
        ]}
      />

      <ConversationsTable conversations={page.items} />

      <Pager
        total={page.total}
        limit={page.limit}
        offset={page.offset}
        basePath="/conversations"
        searchParams={{ kind }}
      />
    </div>
  );
}

function firstValue(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}
