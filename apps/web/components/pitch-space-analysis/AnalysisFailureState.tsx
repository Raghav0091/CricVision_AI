import { StatusBadge } from "@/components/ui/StatusBadge";
import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";

export function AnalysisFailureState({ analysis }: { analysis: PitchSpaceAnalysis }) {
  const partial = analysis.status !== "COMPLETE" || Boolean(analysis.warnings?.length || analysis.unavailable_metrics?.length);
  if (!partial) return null;
  return (
    <section className="border-l-2 border-[#ffc568] bg-[#ffc568]/[0.06] px-4 py-3">
      <div className="flex flex-wrap items-center gap-2">
        <StatusBadge label={analysis.status.replaceAll("_", " ")} tone="warn" />
        <p className="text-sm text-white/70">Available evidence is shown; unsupported metrics remain unavailable.</p>
      </div>
      {Boolean(analysis.warnings?.length) && <ul className="mt-2 space-y-1 text-xs text-white/50">{analysis.warnings?.map((warning) => <li key={warning}>{warning}</li>)}</ul>}
    </section>
  );
}
