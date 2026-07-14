export function StatusBadge({ label, tone = "neutral" }: { label: string; tone?: "neutral" | "good" | "warn" }) {
  const colors = {
    neutral: "border-white/15 bg-white/5 text-white/70",
    good: "border-lime/30 bg-lime/10 text-lime",
    warn: "border-signal/30 bg-signal/10 text-[#ffaaa6]"
  };
  return <span className={`inline-flex rounded-full border px-3 py-1 text-xs font-bold uppercase tracking-[0.12em] ${colors[tone]}`}>{label}</span>;
}
