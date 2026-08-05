import { toThreeVector } from "@/lib/virtual-pitch/coordinates";
import type { ThreeVector3 } from "@/lib/virtual-pitch/types";

import type { WorldPoint3D } from "./types";

/** CRICVISION_PITCH_V1: x_m=lateral, y_m=longitudinal, z_m=height → Three.js scene. */
export function worldPointToThree(point: WorldPoint3D): ThreeVector3 {
  return toThreeVector({ x: point.x_m, y: point.y_m, z: point.z_m });
}

export function worldPointToTuple(point: WorldPoint3D): readonly [number, number, number] {
  const converted = worldPointToThree(point);
  return [converted.x, converted.y, converted.z];
}
