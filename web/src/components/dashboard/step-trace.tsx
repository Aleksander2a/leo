"use client";

import { JsonTree } from "@/components/ui/json-tree";
import { StatusPill } from "@/components/ui/status-pill";
import type { Step } from "@/lib/types";
import { cn, formatMs, truncate } from "@/lib/utils";
import { Brain, ChevronDown, ChevronRight, Wrench } from "lucide-react";
import { useState } from "react";

/**
 * The ReAct loop, rendered as it ran: alternating model turns and the tool
 * calls each one asked for. This is the whole trace — there is no plan tree or
 * verifier verdict layered on top, because there is no planner or verifier.
 */
export function StepTrace({ steps }: { steps: Step[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  if (steps.length === 0) {
    return (
      <p className="py-8 text-center text-sm text-gray-400">
        No trace recorded for this run.
      </p>
    );
  }

  const toggle = (seq: number) =>
    setExpanded((current) => {
      const next = new Set(current);
      if (next.has(seq)) next.delete(seq);
      else next.add(seq);
      return next;
    });

  const allOpen = expanded.size === steps.length;

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <button
          type="button"
          onClick={() =>
            setExpanded(allOpen ? new Set() : new Set(steps.map((step) => step.seq)))
          }
          className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          {allOpen ? "Collapse all" : "Expand all"}
        </button>
      </div>
      <ol className="space-y-2">
        {steps.map((step) => (
          <StepRow
            key={step.seq}
            step={step}
            open={expanded.has(step.seq)}
            onToggle={() => toggle(step.seq)}
          />
        ))}
      </ol>
    </div>
  );
}

function StepRow({
  step,
  open,
  onToggle,
}: {
  step: Step;
  open: boolean;
  onToggle: () => void;
}) {
  const isModel = step.kind === "model";
  const Icon = isModel ? Brain : Wrench;

  return (
    <li
      className={cn(
        "rounded-lg border",
        step.ok
          ? "border-gray-200 dark:border-gray-800"
          : "border-red-200 bg-red-50/40 dark:border-red-900/50 dark:bg-red-950/20",
      )}
    >
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left"
      >
        {open ? (
          <ChevronDown size={14} className="shrink-0 text-gray-400" />
        ) : (
          <ChevronRight size={14} className="shrink-0 text-gray-400" />
        )}
        <span className="w-6 shrink-0 text-right font-mono text-xs text-gray-400">
          {step.seq}
        </span>
        <Icon
          size={14}
          className={cn(
            "shrink-0",
            isModel ? "text-blue-500" : step.ok ? "text-violet-500" : "text-red-500",
          )}
        />
        <span className="min-w-0 flex-1">
          <span className="block truncate text-sm font-medium text-gray-800 dark:text-gray-200">
            {isModel ? modelHeadline(step) : step.name}
          </span>
          <span className="block truncate text-xs text-gray-400">{subtitle(step)}</span>
        </span>
        {!step.ok ? <StatusPill status="error" /> : null}
        <span className="shrink-0 font-mono text-xs text-gray-400">
          {formatMs(step.duration_ms)}
        </span>
      </button>

      {open ? (
        <div className="space-y-3 border-t border-gray-100 px-3 py-3 dark:border-gray-800">
          {isModel ? <ModelDetail step={step} /> : <ToolDetail step={step} />}
        </div>
      ) : null}
    </li>
  );
}

function ModelDetail({ step }: { step: Step }) {
  const result = (step.result ?? {}) as {
    content_preview?: string;
    prompt_tokens?: number;
    completion_tokens?: number;
    cost?: number;
  };
  const offered = ((step.arguments ?? {}).tools_offered ?? []) as string[];

  return (
    <>
      <Facts
        items={[
          ["Finish reason", step.name],
          ["Prompt tokens", String(result.prompt_tokens ?? 0)],
          ["Completion tokens", String(result.completion_tokens ?? 0)],
          ["Cost", `$${(result.cost ?? 0).toFixed(5)}`],
        ]}
      />
      {offered.length > 0 ? (
        <Section label={`Tools requested (${offered.length})`}>
          <div className="flex flex-wrap gap-1.5">
            {offered.map((name, index) => (
              <span
                key={`${name}-${index}`}
                className="rounded-md bg-violet-50 px-2 py-0.5 font-mono text-xs text-violet-700 dark:bg-violet-500/10 dark:text-violet-300"
              >
                {name}
              </span>
            ))}
          </div>
        </Section>
      ) : null}
      {result.content_preview ? (
        <Section label="What the model wrote">
          <p className="text-sm whitespace-pre-wrap text-gray-700 dark:text-gray-300">
            {result.content_preview}
          </p>
        </Section>
      ) : null}
    </>
  );
}

function ToolDetail({ step }: { step: Step }) {
  const result = (step.result ?? {}) as Record<string, unknown>;
  return (
    <>
      <Section label="Arguments">
        <JsonTree data={step.arguments} />
      </Section>
      {step.ok ? (
        <>
          {typeof result.source === "string" ? (
            <Facts
              items={[
                ["Source", String(result.source)],
                ["Reference", String(result.reference ?? "—")],
                ["Observed at", String(result.observed_at ?? "—")],
              ]}
            />
          ) : null}
          {typeof result.url === "string" ? (
            <Section label="URL">
              <a
                href={result.url}
                target="_blank"
                rel="noreferrer noopener"
                className="text-xs break-all text-blue-600 hover:underline dark:text-blue-400"
              >
                {result.url}
              </a>
            </Section>
          ) : null}
          <Section label="Result">
            <JsonTree data={result.data ?? result} />
          </Section>
        </>
      ) : (
        <Section label="Failure">
          <p className="text-sm font-medium text-red-700 dark:text-red-400">
            {String(result.error ?? "error")}
          </p>
          <p className="mt-1 text-sm text-gray-700 dark:text-gray-300">
            {String(result.message ?? "")}
          </p>
          <p className="mt-2 text-xs text-gray-400">
            The loop returns this to the model as a tool message; it is never a run failure
            on its own.
          </p>
        </Section>
      )}
    </>
  );
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <p className="mb-1 text-xs font-semibold tracking-wide text-gray-400 uppercase">{label}</p>
      {children}
    </div>
  );
}

function Facts({ items }: { items: [string, string][] }) {
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {items.map(([label, value]) => (
        <div key={label}>
          <dt className="text-xs text-gray-400">{label}</dt>
          <dd className="truncate text-sm text-gray-800 dark:text-gray-200" title={value}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function modelHeadline(step: Step): string {
  const offered = ((step.arguments ?? {}).tools_offered ?? []) as string[];
  if (offered.length > 0) {
    return `Model turn → ${offered.join(", ")}`;
  }
  return "Model turn → answer";
}

function subtitle(step: Step): string {
  if (step.kind === "model") {
    const preview = (step.result ?? {}) as { content_preview?: string };
    return preview.content_preview ? truncate(preview.content_preview, 110) : `finish: ${step.name}`;
  }
  const result = (step.result ?? {}) as Record<string, unknown>;
  if (!step.ok) return `${result.error ?? "error"}: ${truncate(String(result.message ?? ""), 90)}`;
  const args = step.arguments ?? {};
  const summary = Object.entries(args)
    .map(([key, value]) => `${key}=${truncate(String(value), 40)}`)
    .join(", ");
  return summary || String(result.reference ?? "");
}
