"use client";

import { cn } from "@/lib/utils";
import {
  AlertTriangle,
  Brain,
  Gauge,
  MessagesSquare,
  Plug,
  PlayCircle,
} from "lucide-react";
import Image from "next/image";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ThemeToggle } from "@/components/layout/theme-toggle";

const NAV_ITEMS = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/runs", label: "Runs", icon: PlayCircle },
  { href: "/memory", label: "Memory", icon: Brain },
  { href: "/integrations", label: "Integrations", icon: Plug },
  { href: "/failures", label: "Failures", icon: AlertTriangle },
  { href: "/conversations", label: "Conversations", icon: MessagesSquare },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="flex h-full w-56 shrink-0 flex-col border-r border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950">
      <div className="flex items-center justify-between px-4 py-4">
        <div className="flex items-center gap-2">
          <Image src="/leo_logo.png" alt="" width={28} height={28} className="rounded-md" />
          <div>
            <p className="text-sm font-semibold text-gray-900 dark:text-gray-100">Leo</p>
            <p className="text-xs text-gray-400 dark:text-gray-500">Admin dashboard</p>
          </div>
        </div>
        <ThemeToggle />
      </div>
      <nav className="flex flex-1 flex-col gap-0.5 px-2">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-2 rounded-md px-2.5 py-2 text-sm font-medium transition-colors",
                active
                  ? "bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900"
                  : "text-gray-600 hover:bg-gray-100 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-800 dark:hover:text-gray-100",
              )}
            >
              <Icon size={16} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="px-4 py-4 text-xs text-gray-400 dark:text-gray-500">
        Read-only · local demo
      </div>
    </aside>
  );
}
