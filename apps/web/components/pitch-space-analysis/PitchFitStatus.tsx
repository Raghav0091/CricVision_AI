import { StatusBadge } from "@/components/ui/StatusBadge";
import { confidenceLabel } from "@/lib/pitch-space-analysis/replay";
import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";

function statusTone(value?: string): "good" | "warn" | "neutral" {
  if (value === "FIXED_CAMERA" || value === "READY" || value === "COMPLETE") return "good";
  if (!value) return "neutral";
  return "warn";
}

export function PitchFitStatus({ analysis }: { analysis: PitchSpaceAnalysis }) {
  const setup = analysis.setup_frame_decision;
  const fit = analysis.pitch_fit;
  const camera = analysis.camera_stability;
  const frameStatus = setup?.preferred_frame_passed
    ? { value: "Passed", badge: "Selected", tone: "good" as const }
    : setup?.fallback_used
      ? { value: "Frame 0 unsuitable", badge: `Frame ${setup.selected_frame_index ?? "-"}`, tone: "warn" as const }
      : { value: "No usable setup", badge: "Unavailable", tone: "warn" as const };
  return (
    <section className="grid gap-px overflow-hidden rounded-md border border-white/10 bg-white/10 sm:grid-cols-3">
      <div className="bg-panel px-4 py-3">
        <p className="text-[11px] font-bold uppercase text-white/40">Frame 0</p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <strong className="text-sm">{frameStatus.value}</strong>
          <StatusBadge label={frameStatus.badge} tone={frameStatus.tone} />
        </div>
      </div>
      <div className="bg-panel px-4 py-3">
        <p className="text-[11px] font-bold uppercase text-white/40">Pitch fit</p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <strong className="text-sm">{confidenceLabel(fit?.confidence)}</strong>
          <StatusBadge label={fit?.status ?? "Unavailable"} tone={statusTone(fit?.status)} />
        </div>
      </div>
      <div className="bg-panel px-4 py-3">
        <p className="text-[11px] font-bold uppercase text-white/40">Camera</p>
        <div className="mt-2 flex items-center justify-between gap-2">
          <strong className="text-sm">{confidenceLabel(camera?.confidence)}</strong>
          <StatusBadge label={(camera?.status ?? "Unknown").replaceAll("_", " ")} tone={statusTone(camera?.status)} />
        </div>
      </div>
    </section>
  );
}
