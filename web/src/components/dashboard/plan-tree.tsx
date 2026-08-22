"use client";

import { StatusPill } from "@/components/ui/status-pill";
import { formatDateTime, truncate } from "@/lib/utils";
import type { DelegationEntry, PlanNodeEntry, PlanTreeNode } from "@/lib/types";
import { ChevronDown, ChevronRight, GitBranch } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export function PlanTree({ plans }: { plans: PlanTreeNode[] }) {
  if (plans.length === 0) {
    return <p className="text-sm text-gray-400">This run did not create any delegated plan.</p>;
  }
  return (
    <div className="space-y-4">
      {plans.map((plan) => (
        <PlanCard key={plan.id} plan={plan} />
      ))}
    </div>
  );
}

function PlanCard({ plan }: { plan: PlanTreeNode }) {
  const currentRevision = plan.revisions.find((r) => r.number === plan.current_revision);
  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-800">
      <div className="flex flex-wrap items-center gap-2 border-b border-gray-100 bg-gray-50 px-3 py-2 dark:border-gray-800 dark:bg-gray-900">
        <GitBranch size={14} className="text-gray-400" />
        <StatusPill status={plan.status} />
        <span className="text-xs text-gray-400">
          revision {plan.current_revision}/{plan.max_revisions}
        </span>
        <span className="ml-auto text-xs text-gray-400">{formatDateTime(plan.updated_at)}</span>
      </div>
      <div className="space-y-2 px-3 py-2">
        {currentRevision ? (
          <p className="text-sm text-gray-700 dark:text-gray-300">{truncate(currentRevision.goal, 240)}</p>
        ) : null}
        {plan.output ? (
          <p className="text-xs text-gray-500 dark:text-gray-400">
            <span className="font-medium">Output:</span> {truncate(plan.output, 240)}
          </p>
        ) : null}
        {plan.error ? (
          <p className="text-xs text-red-500">
            <span className="font-medium">Error:</span> {plan.error}
          </p>
        ) : null}
        <ol className="space-y-2 pt-1">
          {plan.nodes.map((node) => (
            <PlanNodeRow key={node.id} node={node} />
          ))}
        </ol>
      </div>
    </div>
  );
}

function PlanNodeRow({ node }: { node: PlanNodeEntry }) {
  const [open, setOpen] = useState(node.delegations.length > 0);
  return (
    <li className="rounded-md border border-gray-100 dark:border-gray-800">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-gray-50 dark:hover:bg-gray-900"
      >
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <StatusPill status={node.status} />
        <span className="text-xs font-medium text-gray-700 dark:text-gray-300">{node.node_key}</span>
        <span className="truncate text-xs text-gray-400">{truncate(node.objective, 100)}</span>
        <span className="ml-auto shrink-0 text-xs text-gray-400">
          attempt {node.attempt}/{node.max_attempts}
        </span>
      </button>
      {open && node.delegations.length > 0 ? (
        <ul className="space-y-2 border-t border-gray-100 px-3 py-2 pl-8 dark:border-gray-800">
          {node.delegations.map((delegation) => (
            <DelegationRow key={delegation.id} delegation={delegation} />
          ))}
        </ul>
      ) : null}
    </li>
  );
}

function DelegationRow({ delegation }: { delegation: DelegationEntry }) {
  return (
    <li className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <StatusPill status={delegation.status} />
        <span className="text-gray-400">attempt {delegation.attempt}</span>
        {delegation.child_run ? (
          <Link
            href={`/runs/${delegation.child_run.id}`}
            className="font-mono text-blue-600 hover:underline dark:text-blue-400"
          >
            {delegation.child_run.id}
          </Link>
        ) : (
          <span className="text-gray-400">no child run</span>
        )}
        <span className="text-gray-400">{formatDateTime(delegation.finished_at ?? delegation.created_at)}</span>
      </div>
      {delegation.output ? (
        <p className="text-xs text-gray-500 dark:text-gray-400">{truncate(delegation.output, 220)}</p>
      ) : null}
      {delegation.error ? <p className="text-xs text-red-500">{delegation.error}</p> : null}
      {delegation.child_plans.length > 0 ? (
        <div className="border-l border-gray-200 pl-3 dark:border-gray-800">
          <PlanTree plans={delegation.child_plans} />
        </div>
      ) : null}
    </li>
  );
}
