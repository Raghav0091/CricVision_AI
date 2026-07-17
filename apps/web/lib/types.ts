export type LiveStage = "setup" | "align-stumps" | "solving-calibration" | "setup-complete" | "capturing" | "results";

export type NormalizedBox = { x: number; y: number; width: number; height: number };

export type BoxLayout = {
  striker: NormalizedBox;
  non_striker: NormalizedBox;
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
  geometry_type: "estimated_from_bbox";
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
