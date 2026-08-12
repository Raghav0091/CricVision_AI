import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";

export const REPLAY_SCHEMA_VERSION = "1.0" as const;
export const REPLAY_COORDINATE_SYSTEM = "CRICVISION_PITCH_V1" as const;
export const REPLAY_DISTANCE_UNIT = "metre" as const;
export const REPLAY_TIME_UNIT = "second" as const;
export const REPLAY_EXPORT_WIDTH = 1920;
export const REPLAY_EXPORT_HEIGHT = 1080;

export type MeasurementValidity =
  | "CALIBRATED"
  | "IMAGE_SPACE_ONLY"
  | "VISUALIZATION_ONLY"
  | "INSUFFICIENT_EVIDENCE";

export type GeometryValidity =
  | "VALID_METRIC_3D"
  | "INVALID_REPROJECTION"
  | "OUTSIDE_PITCH_GEOMETRY"
  | "IMAGE_SPACE_ONLY";

export type MetricAvailabilityStatus =
  | "AVAILABLE"
  | "UNAVAILABLE"
  | "NOT_COMPUTED"
  | "NOT_IMPLEMENTED";

export type TrajectoryProvenance = "OBSERVED" | "RECOVERED" | "PHYSICS_FITTED";

export type ReplayPayloadStatus =
  | "READY"
  | "PENDING"
  | "NOT_IMPLEMENTED"
  | "INSUFFICIENT_EVIDENCE";

export type CameraSource =
  | "CALIBRATED"
  | "PRESET_VISUALIZATION"
  | "UNAVAILABLE";

export type ReplayCalibrationSource =
  | "ACCEPTED_SCENE_CALIBRATION"
  | "ACCEPTED_WICKET_BOX_CALIBRATION";

export type WorldPoint3D = {
  x_m: number;
  y_m: number;
  z_m: number;
};

export type ImagePoint2D = {
  x: number;
  y: number;
};

export type ReplayMetric = {
  value: number | null;
  unit: string;
  confidence: number | null;
  method: string | null;
  status: MetricAvailabilityStatus;
  unavailable_reason: string | null;
};

export type ReplayTrajectorySample = {
  frame_index: number;
  timestamp_seconds: number;
  world_position?: WorldPoint3D | null;
  image_position?: ImagePoint2D | null;
  provenance: TrajectoryProvenance;
  confidence: number;
};

export type ReplayCamera = {
  source: CameraSource;
  calibration_source?: ReplayCalibrationSource | null;
  preset_name?: string | null;
  image_width?: number | null;
  image_height?: number | null;
  camera_matrix?: number[][] | null;
  rotation_matrix?: number[][] | null;
  translation_vector?: number[] | null;
  distortion_coefficients?: number[] | null;
  visualization_only: boolean;
};

export type ReplayPlayback = {
  export_width: number;
  export_height: number;
  landscape: true;
  fps?: number | null;
  duration_seconds?: number | null;
  start_frame_index?: number | null;
  end_frame_index?: number | null;
};

export type ReplayBounce = {
  detected?: boolean | null;
  frame_index?: number | null;
  timestamp_seconds?: number | null;
  world_position?: WorldPoint3D | null;
  image_position?: ImagePoint2D | null;
  confidence?: number | null;
  status: MetricAvailabilityStatus;
  unavailable_reason?: string | null;
};

export type ReplayMetrics = {
  release_speed_kmh: ReplayMetric;
  average_pre_bounce_speed_kmh: ReplayMetric;
  speed_at_bounce_kmh: ReplayMetric;
  delivery_length_m: ReplayMetric;
  estimated_lateral_deviation_m: ReplayMetric;
};

export type ReplayDiagnostics = {
  status: ReplayPayloadStatus;
  measurement_validity: MeasurementValidity;
  geometry_validity?: GeometryValidity | null;
  source_track_id?: string | null;
  tracking_job_id?: string | null;
  calibration_source?: ReplayCalibrationSource | null;
  mean_reprojection_px?: number | null;
  median_reprojection_px?: number | null;
  p95_reprojection_px?: number | null;
  max_reprojection_px?: number | null;
  in_pitch_fraction?: number | null;
  warnings: string[];
  unavailable_reason?: string | null;
};

export type ReplayPayloadV1 = {
  schema_version: "1.0";
  analysis_id: string;
  coordinate_system: typeof REPLAY_COORDINATE_SYSTEM;
  distance_unit: typeof REPLAY_DISTANCE_UNIT;
  time_unit: typeof REPLAY_TIME_UNIT;
  measurement_validity: MeasurementValidity;
  /** The pitch the pose was solved against. Absent means regulation. */
  pitch_geometry?: CricketPitchGeometry | null;
  camera: ReplayCamera;
  playback: ReplayPlayback;
  trajectory: ReplayTrajectorySample[];
  bounce: ReplayBounce;
  metrics: ReplayMetrics;
  diagnostics: ReplayDiagnostics;
};
