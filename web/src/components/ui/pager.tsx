import Link from "next/link";
import { cn } from "@/lib/utils";

export function Pager({
  total,
  limit,
  offset,
  basePath,
  searchParams,
}: {
  total: number;
  limit: number;
  offset: number;
  basePath: string;
  searchParams: Record<string, string | undefined>;
}) {
  const buildHref = (nextOffset: number) => {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(searchParams)) {
      if (value) params.set(key, value);
    }
    params.set("offset", String(nextOffset));
    return `${basePath}?${params.toString()}`;
  };

  const hasPrev = offset > 0;
  const hasNext = offset + limit < total;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);

  return (
    <div className="flex items-center justify-between border-t border-gray-200 px-1 py-3 text-sm text-gray-500 dark:border-gray-800 dark:text-gray-400">
      <span>
        {start}–{end} of {total}
      </span>
      <div className="flex gap-2">
        <Link
          href={buildHref(Math.max(0, offset - limit))}
          aria-disabled={!hasPrev}
          className={cn(
            "rounded-md border border-gray-200 px-3 py-1 dark:border-gray-800",
            hasPrev
              ? "hover:bg-gray-50 dark:hover:bg-gray-900"
              : "pointer-events-none opacity-40",
          )}
        >
          Previous
        </Link>
        <Link
          href={buildHref(offset + limit)}
          aria-disabled={!hasNext}
          className={cn(
            "rounded-md border border-gray-200 px-3 py-1 dark:border-gray-800",
            hasNext
              ? "hover:bg-gray-50 dark:hover:bg-gray-900"
              : "pointer-events-none opacity-40",
          )}
        >
          Next
        </Link>
      </div>
    </div>
  );
}
