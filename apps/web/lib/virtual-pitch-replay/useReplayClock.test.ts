import { findSampleIndex, resolveWindow, sortedSamples } from "./useReplayClock";
import type { ReplayPlayback, ReplayTrajectorySample } from "./types";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function close(actual: number, expected: number, tolerance = 1e-6, label = "value") {
  assert(Math.abs(actual - expected) <= tolerance, `${label}: expected ${expected}, received ${actual}`);
}

const trajectory: ReplayTrajectorySample[] = [
  { frame_index: 10, timestamp_seconds: 0.0, provenance: "OBSERVED", confidence: 0.9, world_position: { x_m: 0, y_m: 0, z_m: 1 } },
  { frame_index: 11, timestamp_seconds: 0.04, provenance: "OBSERVED", confidence: 0.9, world_position: { x_m: 0, y_m: 0.5, z_m: 1 } },
  { frame_index: 12, timestamp_seconds: 0.08, provenance: "RECOVERED", confidence: 0.7, world_position: { x_m: 0, y_m: 1.0, z_m: 1 } },
  { frame_index: 13, timestamp_seconds: 0.12, provenance: "PHYSICS_FITTED", confidence: 0.6, world_position: { x_m: 0, y_m: 1.5, z_m: 1 } }
];

const playback: ReplayPlayback = {
  export_width: 1920,
  export_height: 1080,
  landscape: true,
  fps: 25,
  duration_seconds: 0.12,
  start_frame_index: 10,
  end_frame_index: 13
};

function testWindowClipping() {
  const samples = sortedSamples(trajectory);
  const window = resolveWindow(samples, playback);
  close(window.startSeconds, 0);
  close(window.endSeconds, 0.12);
}

function testSampleLookup() {
  const samples = sortedSamples(trajectory);
  assert(findSampleIndex(samples, 0.05) === 1, "Expected sample index 1 at 0.05s");
  assert(findSampleIndex(samples, 0.12) === 3, "Expected final sample at window end");
}

function testSortedSamples() {
  const reversed = [...trajectory].reverse();
  const samples = sortedSamples(reversed);
  close(samples[0].timestamp_seconds, 0);
  close(samples[samples.length - 1].timestamp_seconds, 0.12);
}

testWindowClipping();
testSampleLookup();
testSortedSamples();

console.log("useReplayClock.test.ts passed");
