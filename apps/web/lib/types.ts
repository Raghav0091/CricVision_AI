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
  geometry_type: "estimated_from_bbox";
  striker: VirtualStump[];
  non_striker: VirtualStump[];
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
  environment_context?: Record<string, unknown> | null;
  debug_files?: { original: string; overlay?: string | null } | null;
};
