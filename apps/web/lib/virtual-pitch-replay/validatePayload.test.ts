import {
  canRenderWorldTrajectory,
  formatReprojectionStats,
  isGeometryRejected,
  validateReplayPayload
} from "./validatePayload";
import type { ReplayPayloadV1 } from "./types";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message);
}

function basePayload(overrides: Partial<ReplayPayloadV1> = {}): ReplayPayloadV1 {
  const unavailable = {
    value: null,
    unit: "km/h",
    confidence: null,
    method: null,
    status: "UNAVAILABLE" as const,
    unavailable_reason: "test"
  };
  return {
    schema_version: "1.0",
    analysis_id: "analysis_test",
    coordinate_system: "CRICVISION_PITCH_V1",
    distance_unit: "metre",
    time_unit: "second",
    measurement_validity: "CALIBRATED",
    camera: {
      source: "CALIBRATED",
      visualization_only: false,
      image_width: 1280,
      image_height: 720,
      camera_matrix: [[900, 0, 640], [0, 900, 360], [0, 0, 1]],
      rotation_matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      translation_vector: [0, 0, 10],
      distortion_coefficients: [0, 0, 0, 0, 0]
    },
    playback: {
      export_width: 1920,
      export_height: 1080,
      landscape: true
    },
    trajectory: [],
    bounce: { status: "NOT_COMPUTED", unavailable_reason: "test" },
    metrics: {
      release_speed_kmh: unavailable,
      average_pre_bounce_speed_kmh: unavailable,
      speed_at_bounce_kmh: unavailable,
      delivery_length_m: { ...unavailable, unit: "m" },
      estimated_lateral_deviation_m: { ...unavailable, unit: "m" }
    },
    diagnostics: {
      status: "READY",
      measurement_validity: "CALIBRATED",
      warnings: []
    },
    ...overrides
  };
}

function testValidPayload() {
  const result = validateReplayPayload(basePayload());
  assert(result.supported, "Expected calibrated payload to validate");
}

function testUnsupportedSchema() {
  const result = validateReplayPayload(basePayload({ schema_version: "2.0" as "1.0" }));
  assert(!result.supported, "Expected unsupported schema to fail");
  assert(result.issues.some((issue) => issue.code === "unsupported_schema"), "Expected schema issue");
}

function testImageSpaceOnlyRules() {
  const payload = basePayload({
    measurement_validity: "IMAGE_SPACE_ONLY",
    trajectory: [{
      frame_index: 1,
      timestamp_seconds: 0,
      provenance: "OBSERVED",
      confidence: 0.5,
      world_position: { x_m: 0, y_m: 0, z_m: 0 }
    }]
  });
  const result = validateReplayPayload(payload);
  assert(!result.supported, "IMAGE_SPACE_ONLY with world position should fail");
  assert(!canRenderWorldTrajectory(payload), "World trajectory should be blocked");
}

function testGeometryRejectionBlocksWorldTrajectory() {
  const payload = basePayload({
    measurement_validity: "IMAGE_SPACE_ONLY",
    trajectory: [{
      frame_index: 1,
      timestamp_seconds: 0,
      provenance: "OBSERVED",
      confidence: 0.5,
      image_position: { x: 100, y: 200 }
    }],
    diagnostics: {
      status: "READY",
      measurement_validity: "IMAGE_SPACE_ONLY",
      geometry_validity: "INVALID_REPROJECTION",
      mean_reprojection_px: 143.2,
      median_reprojection_px: 140.1,
      p95_reprojection_px: 180.0,
      max_reprojection_px: 210.5,
      in_pitch_fraction: 0.0,
      warnings: [],
      unavailable_reason: "Geometry failed"
    }
  });
  assert(isGeometryRejected(payload), "Expected geometry rejection");
  assert(!canRenderWorldTrajectory(payload), "Rejected geometry should block world trajectory");
  const stats = formatReprojectionStats(payload);
  assert(stats.length === 5, "Expected reprojection stats lines");
}

function testVisualizationOnlyRules() {
  const payload = basePayload({
    measurement_validity: "VISUALIZATION_ONLY",
    camera: {
      source: "PRESET_VISUALIZATION",
      visualization_only: false
    },
    metrics: {
      ...basePayload().metrics,
      release_speed_kmh: {
        value: 120,
        unit: "km/h",
        confidence: 0.8,
        method: "test",
        status: "AVAILABLE",
        unavailable_reason: null
      }
    }
  });
  const result = validateReplayPayload(payload);
  assert(!result.supported, "VISUALIZATION_ONLY violations should fail");
}

testValidPayload();
testUnsupportedSchema();
testImageSpaceOnlyRules();
testGeometryRejectionBlocksWorldTrajectory();
testVisualizationOnlyRules();

console.log("validatePayload.test.ts passed");
