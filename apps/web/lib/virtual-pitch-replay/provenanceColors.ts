import type { TrajectoryProvenance } from "./types";

export const PROVENANCE_COLORS: Record<TrajectoryProvenance, string> = {
  OBSERVED: "#78e08f",
  RECOVERED: "#ffca68",
  PHYSICS_FITTED: "#ffe600"
};

export function provenanceColor(provenance: TrajectoryProvenance): string {
  return PROVENANCE_COLORS[provenance];
}
