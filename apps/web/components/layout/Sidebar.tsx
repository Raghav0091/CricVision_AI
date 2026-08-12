"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";


// Only the upload -> detect -> track -> speed path and live capture. Lens
// calibration, Virtual Pitch Lab, Quick Test and Bat Calibration all stay on
// disk and keep working if visited directly; they are just not in the nav.
const links = [
  ["Overview", "/"],
  ["Live Session", "/live"],
  ["Video Analysis", "/video-analysis"]
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
            className={`mt-1 block rounded-xl px-4 py-3 text-sm font-semibold transition hover:bg-white/5 hover:text-white ${pathname === href ? "bg-white/10 text-white" : "text-white/65"}`}
          >
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
