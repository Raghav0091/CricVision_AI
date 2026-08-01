import type { PitchSpaceTrackPoint, TrackProvenance } from "./types";

export function frameAtTime(seconds: number, fps: number, frameCount: number): number {
  if (!Number.isFinite(seconds) || !Number.isFinite(fps) || fps <= 0 || frameCount <= 0) return 0;
  return Math.min(frameCount - 1, Math.max(0, Math.round(seconds * fps)));
}

export function timeAtFrame(frame: number, fps: number): number {
  return Number.isFinite(frame) && Number.isFinite(fps) && fps > 0 ? Math.max(0, frame) / fps : 0;
}

export function pointsThroughFrame(points: readonly PitchSpaceTrackPoint[], frame: number) {
  return points.filter((point) => point.frame_index <= frame);
}

export function activePoint(points: readonly PitchSpaceTrackPoint[], frame: number) {
  let nearest: PitchSpaceTrackPoint | null = null;
  for (const point of points) {
    if (point.frame_index > frame) continue;
    if (!nearest || point.frame_index > nearest.frame_index) nearest = point;
  }
  return nearest;
}

export function provenanceStyle(provenance: TrackProvenance) {
  switch (provenance.toUpperCase()) {
    case "OBSERVED": return { color: "#d7ff7b", dash: "", label: "Observed" };
    case "RECOVERED": return { color: "#63d6ff", dash: "5 4", label: "Recovered" };
    default: return { color: "#ffc568", dash: "2 4", label: "Projected" };
  }
}

export function confidenceLabel(confidence?: number | null): string {
  if (confidence === null || confidence === undefined || !Number.isFinite(confidence)) return "Unavailable";
  if (confidence >= 0.75) return "High confidence";
  if (confidence >= 0.45) return "Medium confidence";
  return "Low confidence";
}
