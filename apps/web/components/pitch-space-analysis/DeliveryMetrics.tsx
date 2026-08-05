import { confidenceLabel } from "@/lib/pitch-space-analysis/replay";
import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";

function metric(value: string, label: string, confidence?: number | null, detail?: string | null) {
  return (
    <article className="min-w-0 border-t border-white/10 py-4">
      <p className="text-[11px] font-bold uppercase text-white/40">{label}</p>
      <p className="mt-1 break-words text-xl font-semibold text-white">{value}</p>
      <p className="mt-1 text-xs text-white/45">{confidenceLabel(confidence)}{detail ? ` / ${detail}` : ""}</p>
    </article>
  );
}

function words(value?: string | null) {
  return value ? value.replaceAll("_", " ").toLowerCase().replace(/^./, (letter) => letter.toUpperCase()) : "Unavailable";
}

export function DeliveryMetrics({ analysis }: { analysis: PitchSpaceAnalysis }) {
  const speed = analysis.estimated_planar_speed;
  const bounce = analysis.bounce;
  const line = analysis.line;
  const length = analysis.length;
  const movement = analysis.estimated_lateral_movement;
  const bounceDistance = line?.distance_from_striker_wicket_m ?? length?.distance_from_striker_wicket_m;
  const bounceDetail = bounce?.pitch_x_m == null ? null : `${Math.abs(bounce.pitch_x_m).toFixed(2)} m pitch-${bounce.pitch_x_m < 0 ? "left" : "right"}`;
  return (
    <section>
      <h2 className="text-sm font-bold">Delivery result</h2>
      <div className="mt-1 grid gap-x-5 sm:grid-cols-2 xl:grid-cols-5">
        {metric(speed?.speed_kmh == null ? "Unavailable" : `${speed.speed_kmh.toFixed(0)} km/h`, "Estimated speed", speed?.confidence_score, "planar")}
        {metric(bounceDistance == null ? "Unavailable" : `${bounceDistance.toFixed(2)} m`, "Bounce from striker", bounce?.confidence, bounceDetail)}
        {metric(words(line?.line), "Line", line?.confidence, line?.lateral_offset_from_middle_m == null ? null : `${Math.abs(line.lateral_offset_from_middle_m).toFixed(2)} m`)}
        {metric(words(length?.length), "Length", length?.confidence, length?.distance_from_striker_wicket_m == null ? null : `${length.distance_from_striker_wicket_m.toFixed(2)} m from wicket`)}
        {metric(movement?.movement_m == null ? "Unavailable" : `${Math.abs(movement.movement_m).toFixed(2)} m`, "Estimated movement", movement?.confidence_score, words(movement?.direction))}
      </div>
      {Boolean(analysis.unavailable_metrics?.length) && <p className="border-t border-white/10 pt-3 text-xs text-white/45">Unavailable: {analysis.unavailable_metrics?.map(words).join(", ")}</p>}
    </section>
  );
}
