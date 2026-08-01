import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";

export function DeveloperDiagnostics({ analysis }: { analysis: PitchSpaceAnalysis }) {
  const payload = {
    setup_frame: analysis.setup_frame_decision,
    stable_near_wicket: analysis.stable_near_wicket ?? null,
    stable_far_wicket: analysis.stable_far_wicket ?? null,
    pitch_fit: analysis.pitch_fit,
    camera_stability: analysis.camera_stability,
    image_track_points: analysis.image_space_track?.length ?? 0,
    pitch_track_points: analysis.pitch_space_track?.length ?? 0,
    bounce_evidence: analysis.bounce?.evidence ?? null,
    line_evidence: analysis.line ?? null,
    speed_evidence: analysis.estimated_planar_speed ?? null,
    movement_evidence: analysis.estimated_lateral_movement ?? null,
    stage_timings: analysis.stage_timings ?? null,
    diagnostics: analysis.diagnostics ?? null
  };
  return (
    <details className="border-t border-white/10 pt-4">
      <summary className="cursor-pointer text-xs font-bold uppercase text-white/45">Developer diagnostics</summary>
      <pre className="mt-3 max-h-96 overflow-auto rounded-md border border-white/10 bg-black/35 p-3 text-[11px] leading-5 text-white/55">{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}
