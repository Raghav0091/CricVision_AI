import Link from "next/link";


const links = [
  ["Overview", "/"],
  ["Live Session", "/live"],
  ["Video Analysis", "/video-analysis"],
  ["Sessions", "/sessions"]
];


export function Sidebar() {
  return (
    <aside className="hidden w-60 shrink-0 border-r border-white/10 p-5 lg:block">
      <p className="mb-4 text-xs font-bold uppercase tracking-[0.2em] text-white/35">Workspace</p>
      <nav className="space-y-1">
        {links.map(([label, href]) => (
          <Link key={href} href={href} className="block rounded-xl px-4 py-3 text-sm font-semibold text-white/65 transition hover:bg-white/5 hover:text-white">
            {label}
          </Link>
        ))}
      </nav>
    </aside>
  );
}
