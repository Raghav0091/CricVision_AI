const STAGES = [
  "Uploading video", "Reading first frame", "Detecting wickets in first frame",
  "Checking nearby early frames", "Fitting virtual pitch", "Tracking ball",
  "Mapping ball into pitch space", "Finding bounce", "Estimating speed",
  "Measuring lateral movement", "Preparing replay"
];

export function PitchSpaceProgress({ activeIndex }: { activeIndex: number }) {
  return (
    <div className="border-y border-white/10 py-4" role="status" aria-live="polite">
      <div className="mb-2 flex items-center justify-between text-xs font-bold uppercase text-white/45">
        <span>{STAGES[Math.min(activeIndex, STAGES.length - 1)]}...</span>
        <span>{Math.round(((activeIndex + 1) / STAGES.length) * 100)}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-white/10">
        <div className="h-full bg-lime transition-all duration-500" style={{ width: `${((activeIndex + 1) / STAGES.length) * 100}%` }} />
      </div>
    </div>
  );
}
