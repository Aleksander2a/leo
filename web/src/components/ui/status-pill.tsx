import { cn } from "@/lib/utils";

type Tone = "green" | "blue" | "amber" | "red" | "violet" | "gray";

const STATUS_TONES: Record<string, Tone> = {
  // Run outcomes
  answered: "green",
  running: "blue",
  failed: "red",
  // Step outcomes
  ok: "green",
  error: "red",
  // Trace step kinds
  tool: "violet",
  model: "blue",
  // Message roles
  user: "gray",
  assistant: "blue",
  // Memory kinds
  fact: "blue",
  preference: "violet",
  decision: "green",
  context: "gray",
  task: "amber",
  // Memory lifecycle
  active: "green",
  superseded: "amber",
  inactive: "gray",
  // Conversation kinds
  dm: "violet",
  channel: "blue",
  cli: "gray",
};

const TONE_CLASSES: Record<Tone, string> = {
  green:
    "bg-emerald-50 text-emerald-700 ring-emerald-600/20 dark:bg-emerald-500/10 dark:text-emerald-400 dark:ring-emerald-500/30",
  blue: "bg-blue-50 text-blue-700 ring-blue-600/20 dark:bg-blue-500/10 dark:text-blue-400 dark:ring-blue-500/30",
  amber:
    "bg-amber-50 text-amber-700 ring-amber-600/20 dark:bg-amber-500/10 dark:text-amber-400 dark:ring-amber-500/30",
  red: "bg-red-50 text-red-700 ring-red-600/20 dark:bg-red-500/10 dark:text-red-400 dark:ring-red-500/30",
  violet:
    "bg-violet-50 text-violet-700 ring-violet-600/20 dark:bg-violet-500/10 dark:text-violet-400 dark:ring-violet-500/30",
  gray: "bg-gray-100 text-gray-600 ring-gray-500/20 dark:bg-gray-500/10 dark:text-gray-400 dark:ring-gray-500/30",
};

export function StatusPill({
  status,
  className,
}: {
  status: string | null | undefined;
  className?: string;
}) {
  const key = (status ?? "unknown").toLowerCase();
  const tone = STATUS_TONES[key] ?? "gray";
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium whitespace-nowrap ring-1 ring-inset",
        TONE_CLASSES[tone],
        className,
      )}
    >
      {(status ?? "unknown").replaceAll("_", " ")}
    </span>
  );
}
