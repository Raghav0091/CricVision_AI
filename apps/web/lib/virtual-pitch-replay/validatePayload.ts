import {
  REPLAY_COORDINATE_SYSTEM,
  REPLAY_SCHEMA_VERSION,
  type GeometryValidity,
  type MeasurementValidity,
  type ReplayMetric,
  type ReplayPayloadV1
} from "./types";

export type PayloadValidationIssue = {
  code: string;
  message: string;
};

export type PayloadValidationResult = {
  supported: boolean;
  issues: PayloadValidationIssue[];
};

const METRIC_KEYS = [
  "release_speed_kmh",
  "average_pre_bounce_speed_kmh",
  "speed_at_bounce_kmh",
  "delivery_length_m",
  "estimated_lateral_deviation_m"
] as const;

function validateMetric(metric: ReplayMetric, label: string): PayloadValidationIssue[] {
  const issues: PayloadValidationIssue[] = [];
  if (metric.status === "AVAILABLE" && metric.value === null) {
    issues.push({
      code: "metric_missing_value",
      message: `${label} is marked available but has no value.`
    });
  }
  if (
    (metric.status === "UNAVAILABLE" || metric.status === "NOT_COMPUTED" || metric.status === "NOT_IMPLEMENTED")
    && metric.value !== null
  ) {
    issues.push({
      code: "metric_unavailable_has_value",
      message: `${label} must remain null when unavailable.`
    });
  }
  return issues;
}

function validateMeasurementValidity(payload: ReplayPayloadV1): PayloadValidationIssue[] {
  const issues: PayloadValidationIssue[] = [];
  const validity: MeasurementValidity = payload.measurement_validity;

  if (validity === "IMAGE_SPACE_ONLY") {
    for (const sample of payload.trajectory) {
      if (sample.world_position != null) {
        issues.push({
          code: "image_space_world_position",
          message: "IMAGE_SPACE_ONLY replay must not expose world positions."
        });
        break;
      }
    }
    if (payload.bounce.world_position != null) {
      issues.push({
        code: "image_space_bounce_world",
        message: "IMAGE_SPACE_ONLY replay must not expose bounce world position."
      });
    }
    for (const key of METRIC_KEYS) {
      if (payload.metrics[key].value !== null) {
        issues.push({
          code: "image_space_metric",
          message: "IMAGE_SPACE_ONLY replay must not expose world metrics."
        });
        break;
      }
    }
  }

  if (validity === "VISUALIZATION_ONLY") {
    if (!payload.camera.visualization_only) {
      issues.push({
        code: "visualization_camera_flag",
        message: "VISUALIZATION_ONLY replay requires camera.visualization_only."
      });
    }
    for (const key of METRIC_KEYS) {
      if (payload.metrics[key].status === "AVAILABLE") {
        issues.push({
          code: "visualization_calibrated_metric",
          message: "VISUALIZATION_ONLY replay cannot claim calibrated measurements."
        });
        break;
      }
    }
  }

  return issues;
}

export function validateReplayPayload(payload: unknown): PayloadValidationResult {
  const issues: PayloadValidationIssue[] = [];

  if (!payload || typeof payload !== "object") {
    return {
      supported: false,
      issues: [{ code: "invalid_payload", message: "Replay payload is not an object." }]
    };
  }

  const record = payload as Partial<ReplayPayloadV1>;

  if (record.schema_version !== REPLAY_SCHEMA_VERSION) {
    issues.push({
      code: "unsupported_schema",
      message: `Unsupported replay schema version: ${String(record.schema_version)}.`
    });
  }

  if (record.coordinate_system !== REPLAY_COORDINATE_SYSTEM) {
    issues.push({
      code: "unsupported_coordinate_system",
      message: `Unsupported coordinate system: ${String(record.coordinate_system)}.`
    });
  }

  if (!record.analysis_id) {
    issues.push({ code: "missing_analysis_id", message: "Replay payload is missing analysis_id." });
  }

  if (record.metrics) {
    for (const key of METRIC_KEYS) {
      issues.push(...validateMetric(record.metrics[key], key));
    }
  }

  if (record.measurement_validity) {
    issues.push(...validateMeasurementValidity(record as ReplayPayloadV1));
  }

  return {
    supported: issues.length === 0,
    issues
  };
}

export function isGeometryRejected(payload: ReplayPayloadV1): boolean {
  const geometry = payload.diagnostics.geometry_validity;
  return geometry != null && geometry !== "VALID_METRIC_3D";
}

export function geometryRejectionLabel(payload: ReplayPayloadV1): GeometryValidity | null {
  const geometry = payload.diagnostics.geometry_validity;
  if (geometry == null || geometry === "VALID_METRIC_3D") {
    return null;
  }
  return geometry;
}

export function formatReprojectionStats(payload: ReplayPayloadV1): string[] {
  const diagnostics = payload.diagnostics;
  const lines: string[] = [];
  const stats: Array<[string, number | null | undefined]> = [
    ["Mean reprojection", diagnostics.mean_reprojection_px],
    ["Median reprojection", diagnostics.median_reprojection_px],
    ["P95 reprojection", diagnostics.p95_reprojection_px],
    ["Max reprojection", diagnostics.max_reprojection_px]
  ];
  for (const [label, value] of stats) {
    if (value != null) {
      lines.push(`${label}: ${value.toFixed(2)} px`);
    }
  }
  if (diagnostics.in_pitch_fraction != null) {
    lines.push(`In-pitch fraction: ${(diagnostics.in_pitch_fraction * 100).toFixed(1)}%`);
  }
  return lines;
}

export function canRenderWorldTrajectory(payload: ReplayPayloadV1): boolean {
  if (payload.measurement_validity === "IMAGE_SPACE_ONLY") return false;
  if (payload.measurement_validity === "INSUFFICIENT_EVIDENCE") return false;
  if (isGeometryRejected(payload)) return false;
  return payload.trajectory.some((sample) => sample.world_position != null);
}
