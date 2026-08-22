import { IntegrationsClient } from "@/components/dashboard/integrations-client";
import { getIntegrations } from "@/lib/api";

export default async function IntegrationsPage() {
  const data = await getIntegrations();

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Integrations</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Provider identity comes from each observation&apos;s source, so newly wired providers show up
          automatically.
        </p>
      </div>
      <IntegrationsClient data={data} />
    </div>
  );
}
