"use client";

import { EventTimeline } from "@/components/dashboard/event-timeline";
import { ModelCallPanel } from "@/components/dashboard/model-call-panel";
import { PlanTree } from "@/components/dashboard/plan-tree";
import { JsonTree } from "@/components/ui/json-tree";
import { ProvenanceBadge } from "@/components/ui/provenance-badge";
import { StatusPill } from "@/components/ui/status-pill";
import { Tabs } from "@/components/ui/tabs";
import { formatDateTime } from "@/lib/utils";
import type { PlanTreeResponse, RunDetail, TimelineEntry } from "@/lib/types";

export function RunDetailTabs({
  detail,
  timeline,
  planTree,
}: {
  detail: RunDetail;
  timeline: TimelineEntry[];
  planTree: PlanTreeResponse;
}) {
  const contextEntries = timeline.filter((entry) => entry.kind === "context_built");
  const toolEntries = timeline.filter((entry) => entry.kind === "tool_call" || entry.kind === "evidence_normalized");
  const verificationEntries = timeline.filter((entry) => entry.kind === "verification");
  const modelCallEntries = timeline.filter((entry) => entry.kind === "model_called");

  return (
    <Tabs
      tabs={[
        {
          id: "timeline",
          label: "Timeline",
          content: <EventTimeline entries={timeline} observations={detail.observations} />,
        },
        {
          id: "model-calls",
          label: `Model Calls (${modelCallEntries.length})`,
          content: <ModelCallsTab entries={modelCallEntries} timeline={timeline} detail={detail} />,
        },
        {
          id: "context",
          label: "Context",
          content: <ContextTab entries={contextEntries} />,
        },
        {
          id: "tools",
          label: "Tools & Observations",
          content: <ToolsTab detail={detail} toolEvents={toolEntries} />,
        },
        {
          id: "plan",
          label: "Plan & Delegations",
          content: <PlanTree plans={planTree.plans} />,
        },
        {
          id: "claims",
          label: "Claims & Verification",
          content: <ClaimsTab detail={detail} verificationEntries={verificationEntries} />,
        },
        {
          id: "delivery",
          label: "Delivery",
          content: <DeliveryTab detail={detail} />,
        },
      ]}
    />
  );
}

function ModelCallsTab({
  entries,
  timeline,
  detail,
}: {
  entries: TimelineEntry[];
  timeline: TimelineEntry[];
  detail: RunDetail;
}) {
  if (entries.length === 0) {
    return <p className="text-sm text-gray-400">No model calls were recorded for this run.</p>;
  }
  return (
    <div className="space-y-4">
      {entries.map((entry) => (
        <div key={entry.sequence} className="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
          <div className="mb-3 flex items-center gap-2 text-xs text-gray-400">
            <span>turn #{entry.sequence}</span>
            <span>{formatDateTime(entry.occurred_at)}</span>
          </div>
          <ModelCallPanel entry={entry} timeline={timeline} observations={detail.observations} />
        </div>
      ))}
    </div>
  );
}

function ContextTab({ entries }: { entries: TimelineEntry[] }) {
  if (entries.length === 0) {
    return <p className="text-sm text-gray-400">No context-assembly events recorded for this run.</p>;
  }
  return (
    <div className="space-y-4">
      {entries.map((entry) => {
        const manifest = entry.raw_payload.source_manifest as Record<string, unknown> | undefined;
        return (
          <div key={entry.sequence} className="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
            <div className="mb-2 flex items-center gap-2 text-xs text-gray-400">
              <span>event #{entry.sequence}</span>
              <span>{formatDateTime(entry.occurred_at)}</span>
            </div>
            {manifest ? (
              <div className="mb-3 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
                <Metric label="Included tokens" value={manifest.included_estimated_tokens} />
                <Metric label="Excluded tokens" value={manifest.excluded_estimated_tokens} />
                <Metric label="Included sources" value={(manifest.included_source_ids as unknown[])?.length} />
                <Metric label="Excluded sources" value={(manifest.excluded_source_ids as unknown[])?.length} />
              </div>
            ) : null}
            <JsonTree data={entry.raw_payload} />
          </div>
        );
      })}
    </div>
  );
}

function Metric({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="rounded-md bg-gray-50 p-2 dark:bg-gray-900">
      <p className="text-gray-400">{label}</p>
      <p className="font-medium text-gray-800 dark:text-gray-200">{value == null ? "—" : String(value)}</p>
    </div>
  );
}

function ToolsTab({ detail, toolEvents }: { detail: RunDetail; toolEvents: TimelineEntry[] }) {
  return (
    <div className="space-y-6">
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Observations ({detail.observations.length})
        </h3>
        {detail.observations.length === 0 ? (
          <p className="text-sm text-gray-400">No observations were recorded for this run.</p>
        ) : (
          <div className="space-y-3">
            {detail.observations.map((observation) => (
              <div key={observation.id} className="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
                <div className="mb-2 flex flex-wrap items-center gap-2 text-xs">
                  <StatusPill status={observation.status} />
                  <span className="font-medium text-gray-700 dark:text-gray-300">{observation.kind}</span>
                  <ProvenanceBadge callKind={observation.call_kind} integration={observation.integration} />
                  {observation.source?.provider ? (
                    <span className="rounded bg-gray-100 px-1.5 py-0.5 text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                      {observation.source.provider}
                    </span>
                  ) : null}
                  <span className="text-gray-400">{observation.quality}</span>
                  <span className="ml-auto text-gray-400">{formatDateTime(observation.observed_at)}</span>
                </div>
                {observation.rejection_code ? (
                  <p className="mb-2 text-xs text-red-500">rejected: {observation.rejection_code}</p>
                ) : null}
                <JsonTree data={observation.data} />
              </div>
            ))}
          </div>
        )}
      </section>
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Tool call events ({toolEvents.length})
        </h3>
        {toolEvents.length === 0 ? (
          <p className="text-sm text-gray-400">No tool-call events recorded.</p>
        ) : (
          <EventList entries={toolEvents} />
        )}
      </section>
    </div>
  );
}

function ClaimsTab({ detail, verificationEntries }: { detail: RunDetail; verificationEntries: TimelineEntry[] }) {
  return (
    <div className="space-y-6">
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Claims ({detail.claims.length})
        </h3>
        {detail.claims.length === 0 ? (
          <p className="text-sm text-gray-400">No claims were recorded for this run.</p>
        ) : (
          <ul className="space-y-2">
            {detail.claims.map((claim) => (
              <li key={claim.id} className="rounded-lg border border-gray-200 p-3 text-sm dark:border-gray-800">
                <span className="mr-2 rounded bg-gray-100 px-1.5 py-0.5 text-xs text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                  {claim.kind}
                </span>
                {claim.statement}
              </li>
            ))}
          </ul>
        )}
      </section>
      <section>
        <h3 className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Verification events ({verificationEntries.length})
        </h3>
        {verificationEntries.length === 0 ? (
          <p className="text-sm text-gray-400">No verification events recorded.</p>
        ) : (
          <EventList entries={verificationEntries} />
        )}
      </section>
    </div>
  );
}

function DeliveryTab({ detail }: { detail: RunDetail }) {
  if (detail.deliveries.length === 0) {
    return <p className="text-sm text-gray-400">No Slack delivery attempts were recorded for this run.</p>;
  }
  return (
    <ul className="space-y-2">
      {detail.deliveries.map((delivery) => (
        <li key={delivery.id} className="rounded-lg border border-gray-200 p-3 text-sm dark:border-gray-800">
          <div className="flex flex-wrap items-center gap-2">
            <StatusPill status={delivery.state} />
            <span className="text-xs text-gray-500 dark:text-gray-400">{delivery.kind}</span>
            <span className="font-mono text-xs text-gray-400">{delivery.destination_channel_id}</span>
            <span className="ml-auto text-xs text-gray-400">{formatDateTime(delivery.updated_at)}</span>
          </div>
          <p className="mt-1 text-xs text-gray-400">
            attempt {delivery.attempt_count}
            {delivery.receipt_message_ts ? ` · receipt ${delivery.receipt_message_ts}` : ""}
          </p>
          {delivery.last_error ? <p className="mt-1 text-xs text-red-500">{delivery.last_error}</p> : null}
        </li>
      ))}
    </ul>
  );
}

function EventList({ entries }: { entries: TimelineEntry[] }) {
  return (
    <ul className="space-y-2">
      {entries.map((entry) => (
        <li key={entry.sequence} className="rounded-lg border border-gray-200 p-3 dark:border-gray-800">
          <div className="mb-2 flex items-center gap-2 text-xs text-gray-400">
            <span>#{entry.sequence}</span>
            <span>{formatDateTime(entry.occurred_at)}</span>
            {entry.call_kind ? <ProvenanceBadge callKind={entry.call_kind} integration={entry.integration} /> : null}
          </div>
          <JsonTree data={entry.raw_payload} />
        </li>
      ))}
    </ul>
  );
}
