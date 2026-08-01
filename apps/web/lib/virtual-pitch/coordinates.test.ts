import { toCricketVector, toThreeVector } from "./coordinates";

function equal(actual: unknown, expected: unknown, message: string) {
  if (JSON.stringify(actual) !== JSON.stringify(expected)) throw new Error(message);
}

equal(toThreeVector({ x: 0, y: 0, z: 0 }), { x: 0, y: 0, z: 0 }, "Origin changed.");
equal(toThreeVector({ x: 2, y: 0, z: 0 }), { x: 2, y: 0, z: 0 }, "Pitch-right must remain scene-right.");
equal(toThreeVector({ x: -2, y: 0, z: 0 }), { x: -2, y: 0, z: 0 }, "Pitch-left must remain scene-left.");
equal(toThreeVector({ x: 0, y: 20, z: 0 }), { x: 0, y: 0, z: -20 }, "Bowler-to-striker must map toward negative scene z.");
equal(toThreeVector({ x: 0, y: 0, z: 3 }), { x: 0, y: 3, z: 0 }, "Cricket height must map to scene y.");

for (const point of [
  { x: 0, y: 0, z: 0 },
  { x: -1.25, y: 20.75, z: 0.72 },
  { x: 4, y: -3, z: 9 }
]) {
  equal(toCricketVector(toThreeVector(point)), point, "Coordinate conversion must round-trip exactly.");
}
