"use client";

import { JsonView, allExpanded, darkStyles, defaultStyles } from "react-json-view-lite";
import "react-json-view-lite/dist/index.css";
import { useTheme } from "next-themes";

export function JsonTree({ data }: { data: unknown }) {
  const { resolvedTheme } = useTheme();
  if (data === null || data === undefined || (typeof data === "object" && Object.keys(data).length === 0)) {
    return <p className="text-xs text-gray-400 dark:text-gray-500">empty</p>;
  }
  return (
    <div className="text-xs">
      <JsonView
        data={data as object}
        shouldExpandNode={allExpanded}
        style={resolvedTheme === "dark" ? darkStyles : defaultStyles}
      />
    </div>
  );
}
