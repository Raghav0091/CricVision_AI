import type { BoxLayout } from "@/lib/types";


export const ALIGNMENT_BOXES: BoxLayout = {
  striker: { x: 0.43, y: 0.22, width: 0.14, height: 0.24 },
  non_striker: { x: 0.35, y: 0.66, width: 0.30, height: 0.28 }
};


function Box({ label, box }: { label: string; box: { x: number; y: number; width: number; height: number } }) {
  return (
    <div
      className="absolute rounded-md border-2 border-dashed border-signal shadow-[0_0_18px_rgba(255,85,79,0.32)]"
      style={{ left: `${box.x * 100}%`, top: `${box.y * 100}%`, width: `${box.width * 100}%`, height: `${box.height * 100}%` }}
    >
      <span className="absolute -top-7 left-0 whitespace-nowrap rounded bg-signal px-2 py-1 text-[10px] font-black uppercase tracking-wide text-white">{label}</span>
    </div>
  );
}


export function StumpAlignmentOverlay({ showEnvironment = false }: { showEnvironment?: boolean }) {
  if (showEnvironment) {
    return <div className="pointer-events-none absolute left-[44%] top-[25%] h-[62%] w-[12%] border-x-2 border-lime/70 bg-lime/10" aria-label="Calibrated pitch corridor" />;
  }
  return (
    <div className="pointer-events-none absolute inset-0">
      <Box label="Striker Stumps" box={ALIGNMENT_BOXES.striker} />
      <Box label="Non-Striker Stumps" box={ALIGNMENT_BOXES.non_striker} />
    </div>
  );
}
