"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";


const links = [
  ["Overview", "/"],
  ["Live Session", "/live"],
  ["Video Analysis", "/video-analysis"],
  ["Sessions", "/sessions"],
  ["Session Results", "/sessions/results"],
  ["Virtual Pitch Lab", "/virtual-pitch-lab"]
];


export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="hidden w-60 shrink-0 border-r border-white/10 p-5 lg:block">
      <p className="mb-4 text-xs font-bold uppercase tracking-[0.2em] text-white/35">Workspace</p>
      <nav className="space-y-1">
        {links.map(([label, href]) => {
          const developerLink = href === "/virtual-pitch-lab";
          return (
          <Link
            key={href}
            href={href}
            aria-current={pathname === href ? "page" : undefined}
            className={`mt-1 block rounded-xl py-3 text-sm font-semibold transition hover:bg-white/5 hover:text-white ${href === "/sessions/results" ? "pl-7 pr-4" : "px-4"} ${developerLink ? "border border-dashed border-[#ffe761]/20" : ""} ${pathname === href ? "bg-white/10 text-white" : "text-white/65"}`}
          >
            <span className="flex items-center justify-between gap-2">
              <span>{label}</span>
              {developerLink && <span className="text-[9px] font-black uppercase tracking-normal text-[#ffe761]">Dev</span>}
            </span>
          </Link>
          );
        })}
      </nav>
    </aside>
  );
}
