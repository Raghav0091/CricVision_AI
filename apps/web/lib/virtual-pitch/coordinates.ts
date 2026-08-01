import type { CricketVector3, ThreeVector3 } from "./types";

/** CricVision (lateral, bowler-to-striker, up) -> Three.js (right, up, camera-depth). */
export function toThreeVector(point: CricketVector3): ThreeVector3 {
  return { x: point.x, y: point.z, z: -point.y };
}

/** Exact inverse of toThreeVector. */
export function toCricketVector(point: ThreeVector3): CricketVector3 {
  return { x: point.x, y: -point.z, z: point.y };
}

export function toThreeTuple(point: CricketVector3): readonly [number, number, number] {
  const converted = toThreeVector(point);
  return [converted.x, converted.y, converted.z];
}

export function toCricketTuple(point: ThreeVector3): readonly [number, number, number] {
  const converted = toCricketVector(point);
  return [converted.x, converted.y, converted.z];
}
