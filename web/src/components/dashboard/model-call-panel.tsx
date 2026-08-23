"use client";

import { JsonTree } from "@/components/ui/json-tree";
import { ProvenanceBadge } from "@/components/ui/provenance-badge";
import { classifyToolName } from "@/lib/provenance";
import { formatCost, formatDateTime, formatNumber } from "@/lib/utils";
import type { ObservationSummary, TimelineEntry } from "@/lib/types";

/**
 * Full "what did the model see this turn" view for one model_called timeline entry.
 *
 * Primary source: the exact request/response transcript captured by
 * leo.persistence.model_call_transcripts.PostgresModelCallTranscriptSink at call time --
 * the real message array sent to OpenRouter and the real completion returned. This is
 * stored in its own table, separate from the run-event log (which is deliberately capped
 * at 8KB and field-allowlisted for replay determinism, see leo.harness.persistence_rules),
 * so it carries full fidelity with no reconstruction involved.
 *
 * Older runs recorded before this feature existed (or a run where the best-effort sink
 * failed) have no transcript row; those fall back to a reconstruction from the paired
 * context_built event's bounded source-ID manifest, cross-referenced against the run's
 * actual Observation rows.
 */
export function ModelCallPanel({
  entry,
  timeline,
  observations,
}: {
  entry: TimelineEntry;
  timeline: TimelineEntry[];
  observations: ObservationSummary[];
}) {
  const payload = entry.raw_payload;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
        <Field label="Provider" value={String(payload.provider ?? "—")} />
        <Field label="Model" value={String(payload.model ?? "—")} />
        <Field label="Decision" value={String(payload.decision ?? "—")} />
        <Field label="Finish reason" value={String(payload.finish_reason ?? "—")} />
        <Field label="Prompt tokens" value={formatNumber(payload.prompt_tokens as number | null)} />
        <Field label="Completion tokens" value={formatNumber(payload.completion_tokens as number | null)} />
        <Field label="Total tokens" value={formatNumber(payload.total_tokens as number | null)} />
        <Field label="Cost" value={formatCost(payload.cost as number | null)} />
      </div>

      {entry.transcript ? (
        <TranscriptView transcript={entry.transcript} />
      ) : (
        <ReconstructedView entry={entry} timeline={timeline} observations={observations} />
      )}
    </div>
  );
}

function TranscriptView({ transcript }: { transcript: NonNullable<TimelineEntry["transcript"]> }) {
  const messages = transcript.request.messages ?? [];
  const tools = transcript.request.tools ?? [];
  const responseMessage = firstResponseMessage(transcript.response);

  return (
    <div className="space-y-4">
      <section>
        <p className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Exact message sent to the model
        </p>
        <div className="space-y-2">
          {messages.map((message, index) => (
            <MessageBlock key={index} role={message.role} content={message.content} />
          ))}
        </div>
      </section>

      <section>
        <p className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Tools offered {tools.length ? `(${tools.length})` : ""}
        </p>
        {tools.length === 0 ? (
          <p className="text-xs text-gray-400">No tools were advertised this turn.</p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {tools.map((tool) => {
              const { callKind, integration } = classifyToolName(tool.function.name);
              return (
                <span
                  key={tool.function.name}
                  className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 px-2 py-1 text-xs dark:border-gray-800"
                  title={tool.function.description}
                >
                  <span className="font-mono text-gray-700 dark:text-gray-300">{tool.function.name}</span>
                  <ProvenanceBadge callKind={callKind} integration={integration} />
                </span>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <p className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Exact response from the model
        </p>
        {responseMessage ? (
          <MessageBlock role="assistant" content={responseMessage} />
        ) : (
          <p className="text-xs text-gray-400">No message content in the recorded response.</p>
        )}
        <details className="mt-2">
          <summary className="cursor-pointer text-xs text-gray-400 hover:text-gray-600 dark:hover:text-gray-300">
            Full raw response payload
          </summary>
          <div className="mt-2">
            <JsonTree data={transcript.response} />
          </div>
        </details>
      </section>
    </div>
  );
}

function MessageBlock({ role, content }: { role: string; content: unknown }) {
  const text = renderMessageContent(content);
  const parsedJson = tryParseJson(text);
  return (
    <div className="rounded-md border border-gray-200 dark:border-gray-800">
      <p className="border-b border-gray-200 bg-gray-50 px-2 py-1 text-xs font-medium tracking-wide text-gray-500 uppercase dark:border-gray-800 dark:bg-gray-900 dark:text-gray-400">
        {role}
      </p>
      <div className="max-h-96 overflow-y-auto p-2">
        {parsedJson !== undefined ? (
          <JsonTree data={parsedJson} />
        ) : (
          <pre className="font-sans text-xs whitespace-pre-wrap text-gray-800 dark:text-gray-200">{text}</pre>
        )}
      </div>
    </div>
  );
}

/** OpenRouter/OpenAI-shaped content is either a plain string or an array of typed
 * blocks ({type: "text", text: "..."}); Leo's system message uses the block form so a
 * prompt-caching breakpoint can mark the static prefix separately from the per-turn
 * suffix (see leo.integrations.openrouter). Render either shape as plain text. */
function renderMessageContent(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((block) => (block && typeof block === "object" && "text" in block ? String(block.text) : JSON.stringify(block)))
      .join("\n\n");
  }
  return JSON.stringify(content, null, 2);
}

function tryParseJson(text: string): unknown {
  const trimmed = text.trim();
  if (!trimmed.startsWith("{") && !trimmed.startsWith("[")) return undefined;
  try {
    return JSON.parse(trimmed);
  } catch {
    return undefined;
  }
}

function firstResponseMessage(response: Record<string, unknown>): unknown {
  const choices = response.choices;
  if (!Array.isArray(choices) || choices.length === 0) return null;
  const message = (choices[0] as Record<string, unknown> | undefined)?.message as
    | Record<string, unknown>
    | undefined;
  if (!message) return null;
  if (message.content) return message.content;
  if (message.tool_calls) return JSON.stringify(message.tool_calls, null, 2);
  return null;
}

function ReconstructedView({
  entry,
  timeline,
  observations,
}: {
  entry: TimelineEntry;
  timeline: TimelineEntry[];
  observations: ObservationSummary[];
}) {
  const contextEntry = findPrecedingContextBuilt(entry, timeline);
  const ctx = contextEntry?.raw_payload ?? {};
  const manifest = (ctx.source_manifest as Record<string, unknown> | undefined) ?? undefined;
  const includedIds = asStringArray(manifest?.included_source_ids);
  const excludedIds = asStringArray(manifest?.excluded_source_ids);
  const toolsSelected = asStringArray(ctx.capability_selected);
  const observationById = new Map(observations.map((observation) => [observation.id, observation]));

  return (
    <div className="space-y-4">
      <p className="rounded-md bg-amber-50 px-2 py-1.5 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
        No exact transcript was recorded for this turn (older run, or the transcript sink failed) --
        showing a reconstruction from context-assembly accounting instead.
      </p>

      <section>
        <p className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Tools offered {toolsSelected.length ? `(${toolsSelected.length})` : ""}
        </p>
        {toolsSelected.length === 0 ? (
          <p className="text-xs text-gray-400">
            {contextEntry
              ? "No tools were advertised this turn."
              : "No context_built event found for this turn (older/partial run data)."}
          </p>
        ) : (
          <div className="flex flex-wrap gap-1.5">
            {toolsSelected.map((name) => {
              const { callKind, integration } = classifyToolName(name);
              return (
                <span
                  key={name}
                  className="inline-flex items-center gap-1.5 rounded-md border border-gray-200 px-2 py-1 text-xs dark:border-gray-800"
                >
                  <span className="font-mono text-gray-700 dark:text-gray-300">{name}</span>
                  <ProvenanceBadge callKind={callKind} integration={integration} />
                </span>
              );
            })}
          </div>
        )}
      </section>

      <section>
        <p className="mb-2 text-xs font-semibold tracking-wide text-gray-400 uppercase">
          Context sources included {includedIds.length ? `(${includedIds.length})` : ""}
        </p>
        {includedIds.length === 0 ? (
          <p className="text-xs text-gray-400">No context sources were included this turn.</p>
        ) : (
          <ul className="space-y-1.5">
            {includedIds.map((id) => {
              const observation = observationById.get(id);
              return (
                <li
                  key={id}
                  className="flex flex-wrap items-center gap-2 rounded-md border border-gray-200 px-2 py-1.5 text-xs dark:border-gray-800"
                >
                  <span className="font-mono text-gray-500 dark:text-gray-400">{id}</span>
                  {observation ? (
                    <>
                      <span className="font-medium text-gray-700 dark:text-gray-300">{observation.kind}</span>
                      <ProvenanceBadge callKind={observation.call_kind} integration={observation.integration} />
                      <span className="text-gray-400">{formatDateTime(observation.observed_at)}</span>
                    </>
                  ) : (
                    <span className="text-gray-400">unresolved (context item, not an observation)</span>
                  )}
                </li>
              );
            })}
          </ul>
        )}
        {excludedIds.length > 0 ? (
          <p className="mt-2 text-xs text-gray-400">
            {excludedIds.length} source{excludedIds.length === 1 ? "" : "s"} were excluded by the context budget.
          </p>
        ) : null}
      </section>
    </div>
  );
}

function findPrecedingContextBuilt(entry: TimelineEntry, timeline: TimelineEntry[]): TimelineEntry | null {
  const index = timeline.findIndex((item) => item.sequence === entry.sequence);
  if (index === -1) return null;
  for (let i = index - 1; i >= 0; i -= 1) {
    if (timeline[i].kind === "context_built") return timeline[i];
    if (timeline[i].kind === "model_called") break;
  }
  return null;
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : [];
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-gray-50 p-2 dark:bg-gray-900">
      <p className="text-gray-400">{label}</p>
      <p className="font-medium text-gray-800 dark:text-gray-200">{value}</p>
    </div>
  );
}
