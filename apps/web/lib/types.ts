export type LiveStage =
  | "setup"
  | "orientation-gate"
  | "align-stumps"
  | "mapping"
  | "solving-calibration"
  | "setup-complete"
  | "capturing"
  | "results";

export type NormalizedBox = { x: number; y: number; width: number; height: number };

export type BoxLayout = {
  striker: NormalizedBox;
  non_striker: NormalizedBox;
};

export type NormalizedPoint = { x: number; y: number };

export type StumpKeypointName =
  | "left_top"
  | "middle_top"
  | "right_top"
  | "left_base"
  | "middle_base"
  | "right_base";

export type StumpKeypointSet = Record<StumpKeypointName, NormalizedPoint>;

export type KeypointLayout = {
  striker: StumpKeypointSet;
  non_striker: StumpKeypointSet;
};

export type CapturedFrame = {
  dataUrl: string;
  width: number;
  height: number;
};

export type PixelPoint = { x: number; y: number };

export type DetectionBoundingBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type StumpDetection = {
  found: boolean;
  confidence: number;
  bbox: DetectionBoundingBox | null;
  source_box: "striker" | "non_striker";
  class_name?: string | null;
};

export type VirtualStump = {
  name: "left" | "middle" | "right";
  top: PixelPoint;
  base: PixelPoint;
};

export type VirtualStumpGeometry = {
  striker: VirtualStumpEnd | null;
  non_striker: VirtualStumpEnd | null;
};

export type VirtualStumpEnd = {
  geometry_type: "estimated_from_bbox" | "manual_keypoints";
  stumps: VirtualStump[];
  bails: Array<{
    name: "left_bail" | "right_bail";
    start: PixelPoint;
    end: PixelPoint;
  }>;
};

export type PitchOverlay = {
  geometry_type: "estimated_from_stump_bboxes";
  pitch_axis: { start: PixelPoint; end: PixelPoint };
  pitch_corridor: PixelPoint[];
  center_line: PixelPoint[];
  wickets: VirtualStumpGeometry;
  crease_guides: Record<"striker" | "non_striker", PixelPoint[]>;
};

export type BatDetection = {
  found: boolean;
  confidence: number;
  bbox: DetectionBoundingBox | null;
  source_box: "striker" | "non_striker";
  class_name?: string | null;
};

export type CameraCalibration = {
  mode: "METRIC_3D" | "METRIC_GROUND_PLANE" | "IMAGE_SPACE_ONLY";
  confidence: "HIGH" | "MEDIUM" | "LOW" | "UNAVAILABLE";
  world_coordinate_system: "CRICVISION_PITCH_V1" | "CALIBRATION_V2";
  image_width: number;
  image_height: number;
  camera_matrix?: number[][] | null;
  distortion_coefficients?: number[] | null;
  rotation_vector?: number[] | null;
  rotation_matrix?: number[][] | null;
  translation_vector?: number[] | null;
  projection_matrix?: number[][] | null;
  correspondences_used: number;
  reprojection_error_px?: number | null;
  calibration_confidence: number;
  failure_reason?: string | null;
  warnings: string[];
};

export type BatPoseCalibrationResponse = {
  success: boolean;
  status: "invalid_calibration_frame" | "bat_detector_missing" | "bat_detector_error" | "bats_not_found" | "solve_failed" | "calibrated";
  analysis_id: string;
  message: string;
  detections?: Record<"striker" | "non_striker", BatDetection> | null;
  calibration?: CameraCalibration | null;
};

export type CalibrationResponse = {
  success: boolean;
  status: "invalid_calibration_frame" | "stump_detector_missing" | "stump_detector_error" | "stumps_not_found" | "setup_complete";
  quality: "Unavailable" | "Poor" | "Partial" | "Good";
  reason: string;
  message: string;
  calibration_frame_path?: string | null;
  model_path?: string | null;
  detections?: Record<"striker" | "non_striker", StumpDetection> | null;
  virtual_stumps?: VirtualStumpGeometry | null;
  pitch_overlay?: PitchOverlay | null;
  calibration_quality?: { status: "good"; score: number } | null;
  environment_context?: Record<string, unknown> | null;
  debug_files?: { original: string; overlay?: string | null } | null;
};
