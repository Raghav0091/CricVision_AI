import { toThreeVector } from "./coordinates";
import type { CricketVector3, ThreeVector3, VirtualPitchModel } from "./types";

export type SegmentGeometry = Readonly<{
  midpoint: ThreeVector3;
  direction: ThreeVector3;
  length: number;
}>;

export type CricketBounds = Readonly<{
  min: CricketVector3;
  max: CricketVector3;
  centre: CricketVector3;
  size: CricketVector3;
  radius: number;
}>;

function length(vector: ThreeVector3): number {
  return Math.hypot(vector.x, vector.y, vector.z);
}

export function segmentGeometry(start: CricketVector3, end: CricketVector3): SegmentGeometry {
  const a = toThreeVector(start);
  const b = toThreeVector(end);
  const delta = { x: b.x - a.x, y: b.y - a.y, z: b.z - a.z };
  const segmentLength = length(delta);
  if (!(segmentLength > 0)) throw new Error("Cannot create geometry for a zero-length segment.");
  return {
    midpoint: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: (a.z + b.z) / 2 },
    direction: { x: delta.x / segmentLength, y: delta.y / segmentLength, z: delta.z / segmentLength },
    length: segmentLength
  };
}

export function polygonIsFinite(vertices: readonly CricketVector3[]): boolean {
  return vertices.length >= 3 && vertices.every((point) =>
    Number.isFinite(point.x) && Number.isFinite(point.y) && Number.isFinite(point.z)
  );
}

export function modelBounds(model: VirtualPitchModel): CricketBounds {
  const points: CricketVector3[] = [
    ...model.landmarks.map((item) => item.point),
    ...model.stumps.flatMap((item) => {
      const half = item.heightM / 2;
      return [
        {
          x: item.centre.x - item.orientation.x * half,
          y: item.centre.y - item.orientation.y * half,
          z: item.centre.z - item.orientation.z * half
        },
        {
          x: item.centre.x + item.orientation.x * half,
          y: item.centre.y + item.orientation.y * half,
          z: item.centre.z + item.orientation.z * half
        }
      ];
    }),
    ...model.bails.flatMap((item) => [item.start, item.endPoint]),
    ...model.lineSegments.flatMap((item) => [item.start, item.endPoint]),
    ...model.polygons.flatMap((item) => item.vertices)
  ];
  if (points.length === 0 || !points.every((point) => polygonIsFinite([point, point, point]))) {
    throw new Error("Virtual-pitch geometry has no finite points.");
  }
  const min = {
    x: Math.min(...points.map((point) => point.x)),
    y: Math.min(...points.map((point) => point.y)),
    z: Math.min(...points.map((point) => point.z))
  };
  const max = {
    x: Math.max(...points.map((point) => point.x)),
    y: Math.max(...points.map((point) => point.y)),
    z: Math.max(...points.map((point) => point.z))
  };
  const size = { x: max.x - min.x, y: max.y - min.y, z: max.z - min.z };
  const centre = { x: (min.x + max.x) / 2, y: (min.y + max.y) / 2, z: (min.z + max.z) / 2 };
  return { min, max, centre, size, radius: Math.hypot(size.x, size.y, size.z) / 2 };
}

export function renderableCounts(model: VirtualPitchModel) {
  return {
    stumps: model.stumps.length,
    bails: model.bails.length,
    lines: model.lineSegments.length,
    polygons: model.polygons.length
  } as const;
}
