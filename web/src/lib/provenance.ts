import type { CallKind } from "@/lib/types";

const INTERNAL_DOMAINS: Record<string, { kind: CallKind; label: string }> = {
  memory: { kind: "internal_memory", label: "Internal memory" },
  agent: { kind: "internal_agent", label: "Internal subagent" },
  thread_context: { kind: "internal_context", label: "Internal context" },
};

/**
 * Client-side mirror of leo.api.dashboard.provenance.classify_call, for the one place
 * (tools offered in a model call, read from a raw context_built payload) that has only a
 * bare tool name string and no server-classified field to read. Keep the *rule* --
 * `_mcp` suffix, domain prefix -- in sync with the Python version; the full display-name
 * table stays server-side since it's cosmetic, not structural.
 */
export function classifyToolName(toolName: string): { callKind: CallKind; integration: string } {
  const domain = toolName.split(".", 1)[0];
  const internal = INTERNAL_DOMAINS[domain];
  if (internal) return { callKind: internal.kind, integration: internal.label };

  const isMcp = toolName.endsWith("_mcp");
  const remainder = toolName.includes(".") ? toolName.slice(toolName.indexOf(".") + 1) : toolName;
  const bare = isMcp ? remainder.slice(0, -4) : remainder;
  const integration = bare
    .split("_")
    .filter(Boolean)
    .map((word) => word[0]?.toUpperCase() + word.slice(1))
    .join(" ");
  return { callKind: isMcp ? "mcp" : "rest_api", integration: integration || toolName };
}
