export type WicketBoxRole = "NEAR" | "FAR";

export type WicketBoxValidationStatus =
  | "PENDING"
  | "VALID"
  | "INVALID_DIMENSIONS"
  | "OUT_OF_BOUNDS"
  | "OVERLAP"
  | "ROLE_ORDER_INVALID"
  | "MISSING_ROLE";

export type StumpIdentity = "LEFT" | "MIDDLE" | "RIGHT";

export type StumpLandmarkProvenance = "AUTOMATIC" | "USER_CORRECTED";

export type StumpLandmarkVisibility =
  | "VISIBLE"
  | "PARTIAL"
  | "OCCLUDED"
  | "UNAVAILABLE";

export type WicketBoxCalibrationStageStatus =
  | "NOT_IMPLEMENTED"
  | "PENDING"
  | "READY"
  | "FAILED";

export type WicketBoxCalibrationStatus =
  | "NOT_STARTED"
  | "DETECTED"
  | "REGISTERED"
  | "ACCEPTED"
  | "FAILED"
  | "NOT_IMPLEMENTED";

export type PixelPoint = {
  x: number;
  y: number;
};

export type WicketBox = {
  role: WicketBoxRole;
  x: number;
  y: number;
  width: number;
  height: number;
  source_image_width: number;
  source_image_height: number;
  calibration_frame_index: number;
  validation_status: WicketBoxValidationStatus;
};

export type StumpLandmark = {
  wicket_role: WicketBoxRole;
  stump_identity: StumpIdentity;
  base: PixelPoint;
  top: PixelPoint;
  centre: PixelPoint;
  visibility: StumpLandmarkVisibility;
  confidence: number;
  provenance: StumpLandmarkProvenance;
};

export type ReprojectionDiagnostic = {
  landmark_id: string;
  observed_pixel_x: number;
  observed_pixel_y: number;
  reprojected_pixel_x: number;
  reprojected_pixel_y: number;
  error_px: number;
};

export type WicketBoxCalibrationCandidateSummary = {
  candidate_id: string;
  assignment_hypothesis: "A" | "B";
  near_semantic_end: "bowler" | "striker";
  far_semantic_end: "bowler" | "striker";
  reprojection_rmse_px?: number | null;
  near_wicket_error_px?: number | null;
  far_wicket_error_px?: number | null;
  camera_height_m?: number | null;
  focal_length_px?: number | null;
  stability_score?: number | null;
  physically_valid: boolean;
  rejection_reasons: string[];
};

export type WicketBoxRegistrationSummary = {
  recommended?: WicketBoxCalibrationCandidateSummary | null;
  alternative?: WicketBoxCalibrationCandidateSummary | null;
  rejected: WicketBoxCalibrationCandidateSummary[];
  auto_selected: boolean;
  orientation_ambiguous: boolean;
  user_message: string;
};

export type CalibrationResult = {
  status: WicketBoxCalibrationStatus;
  analysis_id: string;
  accepted_at?: string | null;
  calibration_frame_index: number;
  source_image_width: number;
  source_image_height: number;
  near_wicket_box?: WicketBox | null;
  far_wicket_box?: WicketBox | null;
  stump_landmarks: StumpLandmark[];
  automatic_stump_landmarks?: StumpLandmark[];
  camera_matrix?: number[][] | null;
  rotation_matrix?: number[][] | null;
  translation_vector?: number[] | null;
  distortion_coefficients?: number[] | null;
  reprojection_rmse_px?: number | null;
  reprojection_diagnostics: ReprojectionDiagnostic[];
  validation_status: WicketBoxValidationStatus;
  warnings: string[];
  registration_summary?: WicketBoxRegistrationSummary | null;
  message: string;
};

export type WicketBoxPairValidationResult = {
  valid: boolean;
  near_box?: WicketBox | null;
  far_box?: WicketBox | null;
  validation_status: WicketBoxValidationStatus;
  overlap_fraction?: number | null;
  role_order_valid: boolean;
  messages: string[];
};

export type WicketBoxCalibrationDetectResponse = {
  success: boolean;
  status: WicketBoxCalibrationStageStatus;
  analysis_id: string;
  calibration_frame_index?: number | null;
  source_image_width?: number | null;
  source_image_height?: number | null;
  near_wicket_box?: WicketBox | null;
  far_wicket_box?: WicketBox | null;
  stump_landmarks: StumpLandmark[];
  message: string;
};

export type WicketBoxCalibrationRegisterRequest = {
  analysis_id: string;
  calibration_frame_index: number;
  source_image_width: number;
  source_image_height: number;
  near_wicket_box: WicketBox;
  far_wicket_box: WicketBox;
  stump_landmarks?: StumpLandmark[];
};

export type WicketBoxCalibrationRegisterResponse = {
  success: boolean;
  status: WicketBoxCalibrationStageStatus;
  analysis_id: string;
  validation: WicketBoxPairValidationResult;
  calibration?: CalibrationResult | null;
  message: string;
};

export type WicketBoxCalibrationAcceptRequest = {
  analysis_id: string;
  accept_registered_calibration?: boolean;
  user_note?: string | null;
};

export type WicketBoxCalibrationAcceptResponse = {
  success: boolean;
  status: WicketBoxCalibrationStageStatus;
  analysis_id: string;
  calibration?: CalibrationResult | null;
  message: string;
};
