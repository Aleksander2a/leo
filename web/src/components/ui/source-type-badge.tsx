import { cn } from "@/lib/utils";

const LABELS: Record<string, string> = {
  explicit: "Explicit (user command)",
  autonomous: "Autonomous (model-proposed)",
};

const CLASSES: Record<string, string> = {
  explicit: "bg-blue-50 text-blue-700 ring-blue-600/20 dark:bg-blue-500/10 dark:text-blue-400 dark:ring-blue-500/30",
  autonomous:
    "bg-fuchsia-50 text-fuchsia-700 ring-fuchsia-600/20 dark:bg-fuchsia-500/10 dark:text-fuchsia-400 dark:ring-fuchsia-500/30",
};

/** Distinguishes a memory revision the user explicitly asked Leo to remember/correct
 * from one Leo proposed on its own (leo.memory.service.propose_autonomous). */
export function SourceTypeBadge({ sourceType, className }: { sourceType: string; className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset whitespace-nowrap",
        CLASSES[sourceType] ?? "bg-gray-100 text-gray-500 ring-gray-500/20 dark:bg-gray-500/10 dark:text-gray-400",
        className,
      )}
    >
      {LABELS[sourceType] ?? sourceType}
    </span>
  );
}
