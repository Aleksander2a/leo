"use client";

import { Search } from "lucide-react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";

const DEBOUNCE_MS = 350;

/** A debounced free-text filter that writes into the URL, so results are linkable. */
export function SearchBox({
  param = "q",
  placeholder = "Search…",
}: {
  param?: string;
  placeholder?: string;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const current = searchParams.get(param) ?? "";

  const [value, setValue] = useState(current);
  const [synced, setSynced] = useState(current);
  // React's sanctioned "adjust state during render": when the URL param changes
  // from outside (Clear filters, the back button), pull the box back in line.
  // Doing this in an effect would cascade an extra render, and remounting on a
  // key would steal focus mid-typing.
  if (synced !== current) {
    setSynced(current);
    setValue(current);
  }

  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => () => clearTimer(timer), []);

  const handleChange = (next: string) => {
    setValue(next);
    clearTimer(timer);
    timer.current = setTimeout(() => {
      const params = new URLSearchParams(searchParams.toString());
      if (next) params.set(param, next);
      else params.delete(param);
      params.delete("offset");
      router.push(`${pathname}?${params.toString()}`);
    }, DEBOUNCE_MS);
  };

  return (
    <div className="relative">
      <Search
        size={14}
        className="pointer-events-none absolute top-1/2 left-2.5 -translate-y-1/2 text-gray-400"
      />
      <input
        type="search"
        value={value}
        onChange={(event) => handleChange(event.target.value)}
        placeholder={placeholder}
        className="w-64 rounded-md border border-gray-200 bg-white py-1.5 pr-2 pl-8 text-xs text-gray-700 placeholder:text-gray-400 dark:border-gray-800 dark:bg-gray-900 dark:text-gray-300"
      />
    </div>
  );
}

function clearTimer(timer: React.RefObject<ReturnType<typeof setTimeout> | null>) {
  if (timer.current) {
    clearTimeout(timer.current);
    timer.current = null;
  }
}
