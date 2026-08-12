/** Mirrors services/api/schemas/device_calibration.py. */

export type CalibrationBand = "GOOD" | "ACCEPTABLE" | "POOR";

export type DeviceCalibrationStatus =
  | "CALIBRATED"
  | "INSUFFICIENT_VIEWS"
  | "FAILED";

export type CheckerboardSpecInput = {
  /** Inner corners across, not squares. */
  columns: number;
  rows: number;
  square_size_mm: number;
};

export type CalibrationQuality = {
  rms_reprojection_px: number;
  band: CalibrationBand;
  views_used: number;
  views_submitted: number;
  diagonal_fov_degrees: number;
  fov_plausible: boolean;
  advice: string;
};

export type DeviceLensProfile = {
  schema_version: "device_calibration_v1";
  device_id: string;
  device_label: string | null;
  calibrated_at: string;
  image_width: number;
  image_height: number;
  focal_length_x_px: number;
  focal_length_y_px: number;
  principal_point_x_px: number;
  principal_point_y_px: number;
  distortion_coefficients: number[];
  checkerboard: CheckerboardSpecInput;
  quality: CalibrationQuality;
};

export type DeviceCalibrationResponse = {
  success: boolean;
  status: DeviceCalibrationStatus;
  profile: DeviceLensProfile | null;
  message: string;
};

export const MIN_VIEWS = 8;
export const RECOMMENDED_VIEWS = 20;
