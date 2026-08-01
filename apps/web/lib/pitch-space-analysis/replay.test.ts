import assert from "node:assert/strict";
// @ts-expect-error Node's type-stripping test runner requires an explicit TypeScript extension.
import { activePoint, frameAtTime, pointsThroughFrame, provenanceStyle, timeAtFrame } from "./replay.ts";
import type { PitchSpaceTrackPoint } from "./types.ts";

const point = (frame_index: number, provenance: string): PitchSpaceTrackPoint => ({
  frame_index,
  timestamp_seconds: frame_index / 25,
  image_x_px: frame_index,
  image_y_px: 20,
  pitch_x_m: 0,
  pitch_y_m: frame_index,
  provenance
});
const points = [point(2, "OBSERVED"), point(5, "RECOVERED"), point(9, "PROJECTED")];

assert.equal(frameAtTime(0.2, 25, 100), 5);
assert.equal(frameAtTime(99, 25, 100), 99);
assert.equal(timeAtFrame(50, 25), 2);
assert.deepEqual(pointsThroughFrame(points, 5).map((item) => item.frame_index), [2, 5]);
assert.equal(activePoint(points, 7)?.frame_index, 5);
assert.equal(activePoint(points, 1), null);
assert.equal(provenanceStyle("OBSERVED").dash, "");
assert.equal(provenanceStyle("RECOVERED").dash, "5 4");
assert.notEqual(provenanceStyle("PROJECTED").color, provenanceStyle("OBSERVED").color);

console.log("pitch-space replay tests passed");
