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

export type CalibrationResponse = {
  success: boolean;
  quality: "Unavailable" | "Poor" | "Partial" | "Good";
  reason: string;
  message: string;
  calibration_frame_path?: string | null;
  environment_context?: Record<string, unknown> | null;
};
