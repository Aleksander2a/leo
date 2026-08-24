import { ToolsClient } from "@/components/dashboard/tools-client";
import { listTools } from "@/lib/api";

export default async function ToolsPage() {
  const tools = await listTools();

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Tools</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Leo picks the tools for a turn by embedding the question and ranking these
          descriptions against it — so a description is not documentation, it is the routing.
        </p>
      </div>

      <ToolsClient initial={tools.items} />
    </div>
  );
}
