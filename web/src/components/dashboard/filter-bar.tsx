"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";

export interface FilterOption {
  param: string;
  label: string;
  options: { value: string; label: string }[];
}

export function FilterBar({ filters }: { filters: FilterOption[] }) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const update = (param: string, value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    if (value) {
      params.set(param, value);
    } else {
      params.delete(param);
    }
    params.delete("offset");
    router.push(`${pathname}?${params.toString()}`);
  };

  const hasActiveFilter = filters.some((filter) => searchParams.get(filter.param));

  return (
    <div className="flex flex-wrap items-center gap-3">
      {filters.map((filter) => (
        <label key={filter.param} className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
          {filter.label}
          <select
            value={searchParams.get(filter.param) ?? ""}
            onChange={(event) => update(filter.param, event.target.value)}
            className="rounded-md border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-300"
          >
            <option value="">all</option>
            {filter.options.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
      ))}
      {hasActiveFilter ? (
        <button
          type="button"
          onClick={() => router.push(pathname)}
          className="text-xs font-medium text-blue-600 hover:underline dark:text-blue-400"
        >
          Clear filters
        </button>
      ) : null}
    </div>
  );
}
