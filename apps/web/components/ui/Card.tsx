import type { PropsWithChildren } from "react";


export function Card({ children, className = "" }: PropsWithChildren<{ className?: string }>) {
  return <section className={`rounded-2xl border border-white/10 bg-panel/90 p-5 shadow-glow ${className}`}>{children}</section>;
}
