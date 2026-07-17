"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";


const links = [
  ["Overview", "/"],
  ["Live Session", "/live"],
  ["Video Analysis", "/video-analysis"],
  ["Sessions", "/sessions"],
  ["Session Results", "/sessions/results"]
];


export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden w-60 shrink-0 border-r border-white/10 p-5 lg:block">
      <p className="mb-4 text-xs font-bold uppercase tracking-[0.2em] text-white/35">Workspace</p>
      <nav className="space-y-1">
        {links.map(([label, href]) => (
          <Link
            key={href}
            href={href}
            aria-current={pathname === href ? "page" : undefined}
            className={`block rounded-xl py-3 text-sm font-semibold transition hover:bg-white/5 hover:text-white ${href === "/sessions/results" ? "pl-7 pr-4" : "px-4"} ${pathname === href ? "bg-white/10 text-white" : "text-white/65"}`}
          >
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
