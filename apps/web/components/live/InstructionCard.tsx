import type { ReactNode } from "react";


// The single instruction surface for every camera stage: a white card pinned
// near the top of the frame with a notched flag on its bottom-left corner. The
// left edge runs flush to the viewport so the notch reads as a folded tab
// rather than a floating badge.
export function InstructionCard({ tone = "info", children }: {
  tone?: "info" | "warning";
  children: ReactNode;
}) {
  const text = tone === "warning"
    ? "text-right text-sm font-bold leading-5 text-[#D32029]"
    : "text-center text-sm font-semibold leading-5 text-[#111]";
  return (
    <div className="pointer-events-none absolute left-0 right-4 top-16">
      <div className={`rounded-r-lg bg-white px-4 py-3 shadow-md ${text}`}>{children}</div>
      <div
        className="h-3.5 w-3.5 bg-white"
        style={{ clipPath: "polygon(0 0, 100% 0, 0 100%)" }}
        aria-hidden
      />
    </div>
  );
}
