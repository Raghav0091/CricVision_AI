export function Header() {
  return (
    <header className="flex h-16 items-center justify-between border-b border-white/10 px-5 lg:px-8">
      <div>
        <span className="text-lg font-black tracking-tight">CricVision</span>
        <span className="ml-2 rounded bg-lime px-1.5 py-0.5 text-[10px] font-black uppercase text-ink">Pro</span>
      </div>
      <div className="h-2 w-2 rounded-full bg-lime shadow-[0_0_14px_#b7f34b]" aria-label="System online" />
    </header>
  );
}
