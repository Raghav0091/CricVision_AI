import type { BoxLayout, CalibrationResponse, CapturedFrame } from "./types";


export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL
  ?? process.env.NEXT_PUBLIC_API_URL
  ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");


export type BallDetectorModelKey = "automatic" | "e2" | "e3" | "e4c";


export type BallDetectorModelOption = {
  key: BallDetectorModelKey;
  display_name: string;
  description: string;
  available: boolean;
};


export type BallDetectorModelsResponse = {
  models: BallDetectorModelOption[];
  default_key: "automatic";
};


export type BallDetectionClipResponse = {
  success: boolean;
  status: "processing" | "ready" | "failed" | "ball_detector_missing" | "invalid_upload" | "upload_too_large" | "video_processing_failed" | "model_inference_failed" | "video_writer_failed";
  delivery_index?: number | null;
  session_id?: string | null;
  job_id?: string | null;
  progress: number;
  model_path_used?: string | null;
  frame_count: number;
  processed_frames: number;
  frames_with_ball: number;
  best_confidence: number;
  average_confidence: number;
  processed_video_url?: string | null;
  message: string;
};


function withBrowserSafeVideoUrl(result: BallDetectionClipResponse): BallDetectionClipResponse {
  const url = result.processed_video_url;
  if (!url || !url.startsWith("/")) return result;
  return { ...result, processed_video_url: `${API_BASE_URL}${url}` };
}


export async function solveCalibration(frame: CapturedFrame, boxLayout: BoxLayout): Promise<CalibrationResponse> {
  const response = await fetch(`${API_BASE_URL}/calibration/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_data_url: frame.dataUrl,
      frame_width: frame.width,
      frame_height: frame.height,
      box_layout: boxLayout
    })
  });
  if (!response.ok) {
    throw new Error(`Calibration service returned ${response.status}.`);
  }
  return response.json() as Promise<CalibrationResponse>;
}


export async function detectBallInDeliveryClip(blob: Blob, deliveryIndex: number, sessionId: string): Promise<BallDetectionClipResponse> {
  const extension = blob.type.includes("mp4") ? "mp4" : "webm";
  const formData = new FormData();
  formData.append("video", new File([blob], `delivery-${deliveryIndex}.${extension}`, { type: blob.type || `video/${extension}` }));
  formData.append("delivery_index", String(deliveryIndex));
  formData.append("session_id", sessionId);
  formData.append("source_mode", "experimental_delivery_test");
  formData.append("processing_mode", "quality");
  const response = await fetch(`${API_BASE_URL}/analysis/ball-detection-clip`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    let message = `Ball detection service returned ${response.status}.`;
    try {
      const body = await response.json() as { detail?: string; message?: string };
      message = body.message ?? body.detail ?? message;
    } catch {
      // Keep the HTTP status message when the response is not JSON.
    }
    throw new Error(message);
  }
  return withBrowserSafeVideoUrl(await response.json() as BallDetectionClipResponse);
}


export type ExperimentalSessionDelivery = {
  delivery_index: number;
  raw_video_url?: string | null;
  job_id?: string | null;
  analysis_status: "queued" | "processing" | "ready" | "failed";
  progress: number;
  processed_video_url?: string | null;
  frames_processed: number;
  frames_with_ball: number;
  best_confidence: number;
  average_confidence: number;
  model_path_used?: string | null;
  error_message?: string | null;
};


export type ExperimentalSession = {
  id: string;
  name: string;
  source: "live" | "upload";
  session_type: "standard" | "experimental_delivery_test";
  created_at: string;
  updated_at: string;
  status: "created" | "capturing" | "complete";
  capture_status: "recording" | "capture_complete";
  analysis_status: "not_started" | "processing" | "partially_ready" | "ready" | "failed";
  delivery_count: number;
  deliveries: ExperimentalSessionDelivery[];
};


function withBrowserSafeSessionUrls(session: ExperimentalSession): ExperimentalSession {
  return {
    ...session,
    deliveries: session.deliveries.map((delivery) => ({
      ...delivery,
      raw_video_url: resolveApiUrl(delivery.raw_video_url),
      processed_video_url: resolveApiUrl(delivery.processed_video_url)
    }))
  };
}


function resolveApiUrl(url?: string | null): string | null | undefined {
  if (!url || !url.startsWith("/")) return url;
  return `${API_BASE_URL}${url}`;
}


async function sessionRequest(url: string, init?: RequestInit): Promise<ExperimentalSession> {
  const response = await fetch(`${API_BASE_URL}${url}`, { cache: "no-store", ...init });
  if (!response.ok) throw new Error(`Session service returned ${response.status}.`);
  return withBrowserSafeSessionUrls(await response.json() as ExperimentalSession);
}


export function createExperimentalSession(): Promise<ExperimentalSession> {
  return sessionRequest("/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      name: "Experimental Session",
      source: "live",
      session_type: "experimental_delivery_test"
    })
  });
}


export async function listExperimentalSessions(): Promise<ExperimentalSession[]> {
  const response = await fetch(`${API_BASE_URL}/sessions`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Session service returned ${response.status}.`);
  const sessions = await response.json() as ExperimentalSession[];
  return sessions
    .filter((session) => session.session_type === "experimental_delivery_test")
    .map(withBrowserSafeSessionUrls);
}


export function getExperimentalSession(sessionId: string): Promise<ExperimentalSession> {
  return sessionRequest(`/sessions/${encodeURIComponent(sessionId)}`);
}


export function completeExperimentalSession(sessionId: string): Promise<ExperimentalSession> {
  return sessionRequest(`/sessions/${encodeURIComponent(sessionId)}/capture-complete`, { method: "POST" });
}


export async function getBallDetectionJob(jobId: string): Promise<BallDetectionClipResponse> {
  const response = await fetch(`${API_BASE_URL}/analysis/jobs/${encodeURIComponent(jobId)}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`Ball detection job status returned ${response.status}.`);
  return withBrowserSafeVideoUrl(await response.json() as BallDetectionClipResponse);
}


export type VideoAnalysisPreparedResponse = {
  success: boolean;
  analysis_id: string;
  status: "prepared" | "calibrated";
  original_filename: string;
  stored_filename: string;
  file_size_bytes: number;
  created_at: string;
  updated_at?: string | null;
  duration_seconds: number;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  codec?: string | null;
  reference_frame_index: number;
  reference_frame_selection?: {
    strategy?: string;
    window_scanned?: number;
    window_limit?: number;
    selected_index?: number;
    score?: number;
    reason?: string;
  } | null;
  original_video_url: string;
  reference_frame_url: string;
  calibration_status?: "confirmed" | null;
  calibration_url?: string | null;
  calibration_overlay_url?: string | null;
  visual_calibration_quality?: "READY" | "WEAK" | "FAILED" | null;
  visual_calibration_mode?: "automatic_visual" | null;
  calibration_v2_status?:
    | "confirmed"
    | "ready"
    | "weak"
    | "unstable"
    | "insufficient_geometry"
    | null;
  calibration_v2_url?: string | null;
  calibration_v2_overlay_url?: string | null;
  calibration_v2_quality_grade?: CalibrationQualityGradeV2 | null;
  calibration_v2_reprojection_rmse_px?: number | null;
  camera_pose_status?: CameraPoseStatus | null;
  camera_pose_quality?: number | null;
  camera_pose_url?: string | null;
  camera_pose_overlay_url?: string | null;
  camera_intrinsics_source?: CameraIntrinsicsSource | null;
  camera_pose_reprojection_rmse_px?: number | null;
  calibration_mode_used?: "ground_plane" | "wicket_camera_pose" | null;
  ball_detection_status?: "detection_queued" | "detecting_ball" | "detection_complete" | "detection_failed" | null;
  ball_detection_job_id?: string | null;
  ball_detection_started_at?: string | null;
  ball_detection_completed_at?: string | null;
  detection_summary_url?: string | null;
  detection_overlay_url?: string | null;
  tracking_status?: "tracking_queued" | "tracking_ball" | "tracking_complete" | "tracking_failed" | "tracking_no_reliable_track" | null;
  tracking_job_id?: string | null;
  tracking_started_at?: string | null;
  tracking_completed_at?: string | null;
  tracking_summary_url?: string | null;
  tracking_video_url?: string | null;
  message: string;
};


function withBrowserSafeAnalysisUrls(record: VideoAnalysisPreparedResponse): VideoAnalysisPreparedResponse {
  return {
    ...record,
    original_video_url: resolveApiUrl(record.original_video_url) ?? record.original_video_url,
    reference_frame_url: resolveApiUrl(record.reference_frame_url) ?? record.reference_frame_url,
    calibration_url: resolveApiUrl(record.calibration_url),
    calibration_overlay_url: resolveApiUrl(record.calibration_overlay_url),
    calibration_v2_url: resolveApiUrl(record.calibration_v2_url),
    calibration_v2_overlay_url: resolveApiUrl(record.calibration_v2_overlay_url),
    camera_pose_url: resolveApiUrl(record.camera_pose_url),
    camera_pose_overlay_url: resolveApiUrl(record.camera_pose_overlay_url),
    detection_summary_url: resolveApiUrl(record.detection_summary_url),
    detection_overlay_url: resolveApiUrl(record.detection_overlay_url),
    tracking_summary_url: resolveApiUrl(record.tracking_summary_url),
    tracking_video_url: resolveApiUrl(record.tracking_video_url)
  };
}


async function videoAnalysisError(response: Response, fallback: string): Promise<Error> {
  let message = fallback;
  try {
    const body = await response.json() as { detail?: string; message?: string };
    message = body.detail ?? body.message ?? fallback;
  } catch {
    // Keep the HTTP status message when the response is not JSON.
  }
  return new Error(message);
}


export async function prepareVideoAnalysis(file: File): Promise<VideoAnalysisPreparedResponse> {
  const formData = new FormData();
  formData.append("video", file);
  const response = await fetch(`${API_BASE_URL}/video-analysis/prepare`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Video preparation returned ${response.status}.`);
  }
  return withBrowserSafeAnalysisUrls(await response.json() as VideoAnalysisPreparedResponse);
}


export async function getVideoAnalysis(analysisId: string): Promise<VideoAnalysisPreparedResponse> {
  const response = await fetch(`${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Video analysis lookup returned ${response.status}.`);
  }
  return withBrowserSafeAnalysisUrls(await response.json() as VideoAnalysisPreparedResponse);
}


export type NormalizedPoint = {
  x: number;
  y: number;
};


export type NormalizedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};


export type WicketDetectionPass = "full_frame" | "far_roi" | "near_roi" | "guide_roi";


export type WicketCandidate = {
  candidate_id: string;
  confidence: number;
  class_name: string;
  box: NormalizedBox;
  center: NormalizedPoint;
  bottom_center: NormalizedPoint;
  detection_pass?: WicketDetectionPass | null;
};


export type WicketCalibration = {
  label: "striker" | "non_striker";
  source: "detected" | "adjusted" | "manual";
  confidence?: number | null;
  box: NormalizedBox;
  center: NormalizedPoint;
  bottom_center: NormalizedPoint;
  approximate_wicket_base_reference?: NormalizedPoint | null;
  detection_pass?: WicketDetectionPass | null;
};


export type PitchGeometry = {
  axis_start: NormalizedPoint;
  axis_end: NormalizedPoint;
  corridor: [NormalizedPoint, NormalizedPoint, NormalizedPoint, NormalizedPoint];
  near_end_label: "striker" | "non_striker";
  far_end_label: "striker" | "non_striker";
  geometry_type: "approximate_2d";
  corridor_width_multiplier: number;
};


export type VisualCalibrationQuality = "READY" | "WEAK" | "FAILED";


export type VisualCalibrationDetectionDebug = {
  pass_count: number;
  passes: Array<Record<string, unknown>>;
  rejected: Array<Record<string, unknown>>;
  rois: Record<string, NormalizedBox>;
  selected?: Record<string, unknown> | null;
  debug_overlay_url?: string | null;
  debug_json_url?: string | null;
};


export type VideoCalibrationDetectionResponse = {
  success: boolean;
  status:
    | "candidates_ready"
    | "manual_required"
    | "detection_incomplete"
    | "stump_detector_missing"
    | "stump_detector_error";
  analysis_id: string;
  reference_frame_index: number;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  candidates: WicketCandidate[];
  provisional_striker_wicket?: WicketCalibration | null;
  provisional_non_striker_wicket?: WicketCalibration | null;
  pitch_geometry?: PitchGeometry | null;
  striker_guide?: NormalizedBox | null;
  non_striker_guide?: NormalizedBox | null;
  failed_ends?: Array<"striker" | "non_striker">;
  model_path_used: string;
  mode?: "automatic_visual";
  quality?: VisualCalibrationQuality;
  quality_reasons?: string[];
  assignment_warning?: string | null;
  warning?: string | null;
  message: string;
  detection_debug?: VisualCalibrationDetectionDebug | null;
};


export type VideoCalibrationConfirmationRequest = {
  analysis_id: string;
  striker_wicket: Pick<
    WicketCalibration,
    "label" | "source" | "confidence" | "box" | "detection_pass"
  >;
  non_striker_wicket: Pick<
    WicketCalibration,
    "label" | "source" | "confidence" | "box" | "detection_pass"
  >;
  corridor_width_multiplier: number;
  user_note?: string | null;
  striker_guide?: NormalizedBox | null;
  non_striker_guide?: NormalizedBox | null;
};


export type ConfirmedVideoCalibrationResponse = {
  success: boolean;
  status: "calibrated";
  analysis_id: string;
  created_at: string;
  updated_at: string;
  reference_frame_index: number;
  reference_frame_url: string;
  calibration_url: string;
  calibration_overlay_url: string;
  scene_overlay_url?: string | null;
  scene_overlay_status?: "ready" | "failed" | "skipped" | null;
  image_width: number;
  image_height: number;
  model_path_used?: string | null;
  mode?: "automatic_visual";
  quality?: VisualCalibrationQuality;
  quality_reasons?: string[];
  assignment_warning?: string | null;
  striker_wicket: WicketCalibration;
  non_striker_wicket: WicketCalibration;
  pitch_geometry: PitchGeometry;
  striker_guide?: NormalizedBox | null;
  non_striker_guide?: NormalizedBox | null;
  user_note?: string | null;
  message: string;
};


export type CalibrationLandmarkSource =
  | "detected"
  | "inferred"
  | "manually_adjusted"
  | "manual";


export type ImageLeftRightConvention =
  | "image_left_is_world_left"
  | "image_left_is_world_right";


export type CricketPitchGeometry = {
  pitch_length_m: number;
  wicket_width_m: number;
  wicket_height_m: number;
  stump_diameter_m: number;
  pitch_width_m: number;
  popping_crease_distance_m: number;
  stump_lateral_positions_m: {
    left: number;
    middle: number;
    right: number;
  };
};


export type CalibrationLandmarkInput = {
  id: string;
  label: string;
  wicket_end: "bowler" | "striker" | "ground";
  landmark_type: "stump_base" | "ground_control";
  normalized_x: number;
  normalized_y: number;
  source: CalibrationLandmarkSource;
  confidence?: number | null;
  world_x_m?: number | null;
  world_y_m?: number | null;
  world_z_m?: number | null;
};


export type CalibrationLandmark = CalibrationLandmarkInput & {
  pixel_x: number;
  pixel_y: number;
  world_x_m: number;
  world_y_m: number;
  world_z_m: number;
};


export type CalibrationV2InitialiseResponse = {
  success: true;
  status: "initialised";
  analysis_id: string;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  pitch_geometry: CricketPitchGeometry;
  landmarks: CalibrationLandmark[];
  image_left_right_convention: ImageLeftRightConvention;
  warnings: string[];
  message: string;
};


export type ReprojectionDiagnostic = {
  landmark_id: string;
  landmark_source: CalibrationLandmarkSource;
  used_for_homography: boolean;
  ransac_inlier?: boolean | null;
  observed_pixel_x: number;
  observed_pixel_y: number;
  reprojected_pixel_x: number;
  reprojected_pixel_y: number;
  error_px: number;
};


export type CalibrationQualityGradeV2 =
  | "excellent"
  | "good"
  | "usable"
  | "weak"
  | "poor"
  | "insufficient_geometry";


export type CalibrationQualityV2 = {
  landmark_coverage: number;
  usable_landmarks: number;
  metric_correspondence_count: number;
  additional_metric_ground_landmark_count: number;
  landmark_spread_score: number;
  world_coverage: number;
  reprojection_rmse_px?: number | null;
  max_reprojection_error_px?: number | null;
  median_reprojection_error_px?: number | null;
  normalized_reprojection_rmse?: number | null;
  geometry_condition: "well_conditioned" | "weak" | "unstable" | "insufficient";
  homography_condition_number?: number | null;
  image_coverage: number;
  wicket_order_valid: boolean;
  transform_available: boolean;
  full_pitch_projection_allowed: boolean;
  projection_outside_fraction?: number | null;
  manual_adjustment_count: number;
  used_landmark_ids: string[];
  ignored_landmark_ids: string[];
  landmark_sources: Record<string, number>;
  warnings: string[];
  quality_grade: CalibrationQualityGradeV2;
  overall_confidence: number;
  reprojection_diagnostics: ReprojectionDiagnostic[];
};


export type GroundHomographyResult = {
  transform_available: boolean;
  image_to_ground_homography?: number[][] | null;
  ground_to_image_homography?: number[][] | null;
  determinant?: number | null;
  condition_number?: number | null;
  estimation_method: "none" | "direct" | "ransac";
  ransac_reprojection_threshold_px?: number | null;
  ransac_inlier_count?: number | null;
  ransac_inlier_landmark_ids: string[];
  round_trip_image_rmse_px?: number | null;
  round_trip_ground_rmse_m?: number | null;
  image_convention: "pixel_uv";
  ground_convention: "pitch_xy_metres_z0";
};


export type ProjectedPitchLine = {
  id: string;
  label: string;
  ground_points: Array<{ x_m: number; y_m: number }>;
  image_points: Array<{ x: number; y: number }>;
};


export type CalibrationV2Result = {
  success: boolean;
  status:
    | "confirmed"
    | "ready"
    | "weak"
    | "unstable"
    | "insufficient_geometry";
  schema_version: "2.0" | "2.1";
  analysis_id: string;
  calibration_mode: "ground_plane";
  coordinate_system: {
    units: "metres";
    origin: "bowler_wicket_centre";
    x_axis: "toward_striker";
    y_axis: "lateral";
    z_axis: "up";
    left_right_convention: string;
    image_left_right_convention: ImageLeftRightConvention;
  };
  pitch_geometry: CricketPitchGeometry;
  landmark_set: {
    primary_stump_bases: CalibrationLandmark[];
    optional_ground_landmarks: CalibrationLandmark[];
  };
  homography: GroundHomographyResult;
  quality: CalibrationQualityV2;
  virtual_pitch_overlay_geometry: {
    projected_lines: ProjectedPitchLine[];
    projection_mode: "full_pitch" | "local_debug" | "landmarks_only";
  };
  calibration_v2_url: string;
  calibration_v2_overlay_url: string;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  landmark_semantics_confirmed: boolean;
  ground_reference_mode: "use" | "skip";
  ground_transform_reason?: string | null;
  created_at: string;
  updated_at: string;
  user_note?: string | null;
  message: string;
};


export type CalibrationV2ConfirmRequest = {
  analysis_id: string;
  landmarks: CalibrationLandmarkInput[];
  pitch_geometry: CricketPitchGeometry;
  image_left_right_convention: ImageLeftRightConvention;
  landmark_semantics_confirmed: boolean;
  ground_reference_mode: "use" | "skip";
  user_note?: string | null;
};


export type WicketLandmarkVisibility =
  | "visible"
  | "uncertain"
  | "occluded"
  | "unavailable";


export type CameraIntrinsicsSource =
  | "calibrated_device_profile"
  | "metadata_estimated"
  | "heuristic_estimated"
  | "manually_provided";


export type CameraPoseStatus =
  | "ready"
  | "usable"
  | "weak"
  | "unstable"
  | "insufficient_landmarks"
  | "solver_failed"
  | "implausible_pose";


export type WicketPoseLandmarkInput = {
  id: string;
  label: string;
  wicket_end: "bowler" | "striker";
  stump_position: "left" | "middle" | "right";
  point_type: "base" | "top";
  normalized_x: number;
  normalized_y: number;
  source: CalibrationLandmarkSource;
  confidence?: number | null;
  visibility: WicketLandmarkVisibility;
};


export type WicketPoseLandmark = WicketPoseLandmarkInput & {
  pixel_x: number;
  pixel_y: number;
  world_x_m: number;
  world_y_m: number;
  world_z_m: number;
};


export type CameraIntrinsics = {
  image_width: number;
  image_height: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  intrinsic_matrix: number[][];
  distortion_coefficients: number[];
  source: CameraIntrinsicsSource;
  quality: "calibrated" | "estimated" | "low";
  device_profile_id?: string | null;
  camera_model?: string | null;
  lens_mode?: string | null;
  resolution_label?: string | null;
  assumed_horizontal_fov_degrees?: number | null;
  distortion_model_source: "calibrated" | "not_calibrated";
  assumptions: string[];
};


export type CameraPoseReprojectionDiagnostic = {
  landmark_id: string;
  observed_pixel_x: number;
  observed_pixel_y: number;
  projected_pixel_x: number;
  projected_pixel_y: number;
  residual_px: number;
  camera_depth_m: number;
  ransac_inlier: boolean;
};


export type CameraPoseSolution = {
  solved: boolean;
  accepted: boolean;
  solver_method: string;
  refinement_method?: string | null;
  rotation_vector?: number[] | null;
  rotation_matrix?: number[][] | null;
  translation_vector?: number[] | null;
  camera_position_world?: number[] | null;
  camera_forward_direction_world?: number[] | null;
  camera_height_m?: number | null;
  landmark_count: number;
  used_landmark_ids: string[];
  unavailable_landmark_ids: string[];
  ransac_inlier_ids: string[];
  ransac_outlier_ids: string[];
  reprojection_rmse_px?: number | null;
  reprojection_median_px?: number | null;
  reprojection_max_px?: number | null;
  normalized_reprojection_rmse?: number | null;
  reprojection_diagnostics: CameraPoseReprojectionDiagnostic[];
  positive_depth_for_all_used_landmarks?: boolean | null;
  both_wickets_in_front?: boolean | null;
  camera_faces_pitch?: boolean | null;
  wicket_order_plausible?: boolean | null;
  warnings: string[];
  rejection_reasons: string[];
};


export type CameraPoseQualityComponents = {
  landmark_quality: number;
  landmark_coverage: number;
  reprojection_quality: number;
  intrinsics_quality: number;
  geometry_condition: number;
  pose_plausibility: number;
  overall_pose_quality: number;
};


export type WicketCameraPoseInitialiseResponse = {
  success: true;
  status: "initialised";
  analysis_id: string;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  pitch_geometry: CricketPitchGeometry;
  landmarks: WicketPoseLandmark[];
  camera_intrinsics: CameraIntrinsics;
  warnings: string[];
  message: string;
};


export type WicketCameraPoseSolveRequest = {
  analysis_id: string;
  landmarks: WicketPoseLandmarkInput[];
  pitch_geometry: CricketPitchGeometry;
  camera_intrinsics: CameraIntrinsics;
  landmark_semantics_confirmed: boolean;
  user_note?: string | null;
};


export type WicketCameraPoseResult = {
  success: boolean;
  status: CameraPoseStatus;
  schema_version: "2.2";
  analysis_id: string;
  calibration_mode: "wicket_camera_pose";
  coordinate_system: CalibrationV2Result["coordinate_system"];
  pitch_geometry: CricketPitchGeometry;
  stump_top_definition: "top_of_stump_body_excluding_bails";
  landmarks: WicketPoseLandmark[];
  camera_intrinsics: CameraIntrinsics;
  camera_pose: CameraPoseSolution;
  quality: CameraPoseQualityComponents;
  camera_pose_url: string;
  camera_pose_overlay_url: string;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  landmark_semantics_confirmed: boolean;
  created_at: string;
  updated_at: string;
  user_note?: string | null;
  message: string;
};


function withBrowserSafeDetectionUrls(
  result: VideoCalibrationDetectionResponse
): VideoCalibrationDetectionResponse {
  const debug = result.detection_debug;
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url) ?? result.reference_frame_url,
    detection_debug: debug
      ? {
          ...debug,
          debug_overlay_url: resolveApiUrl(debug.debug_overlay_url) ?? debug.debug_overlay_url,
          debug_json_url: resolveApiUrl(debug.debug_json_url) ?? debug.debug_json_url
        }
      : debug
  };
}


function withBrowserSafeCalibrationUrls(
  result: ConfirmedVideoCalibrationResponse
): ConfirmedVideoCalibrationResponse {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url) ?? result.reference_frame_url,
    calibration_url: resolveApiUrl(result.calibration_url) ?? result.calibration_url,
    calibration_overlay_url: resolveApiUrl(result.calibration_overlay_url) ?? result.calibration_overlay_url,
    scene_overlay_url: resolveApiUrl(result.scene_overlay_url) ?? result.scene_overlay_url
  };
}


function withBrowserSafeCalibrationV2InitialiseUrls(
  result: CalibrationV2InitialiseResponse
): CalibrationV2InitialiseResponse {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url)
      ?? result.reference_frame_url
  };
}


function withBrowserSafeCalibrationV2Urls(
  result: CalibrationV2Result
): CalibrationV2Result {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url)
      ?? result.reference_frame_url,
    calibration_v2_url: resolveApiUrl(result.calibration_v2_url)
      ?? result.calibration_v2_url,
    calibration_v2_overlay_url: resolveApiUrl(result.calibration_v2_overlay_url)
      ?? result.calibration_v2_overlay_url
  };
}


function withBrowserSafeCameraPoseInitialiseUrls(
  result: WicketCameraPoseInitialiseResponse
): WicketCameraPoseInitialiseResponse {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url)
      ?? result.reference_frame_url
  };
}


function withBrowserSafeCameraPoseUrls(
  result: WicketCameraPoseResult
): WicketCameraPoseResult {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url)
      ?? result.reference_frame_url,
    camera_pose_url: resolveApiUrl(result.camera_pose_url)
      ?? result.camera_pose_url,
    camera_pose_overlay_url: resolveApiUrl(result.camera_pose_overlay_url)
      ?? result.camera_pose_overlay_url
  };
}


export async function detectVideoAnalysisCalibration(
  analysisId: string,
  options?: {
    refreshEarlyReference?: boolean;
    strikerGuide?: NormalizedBox;
    nonStrikerGuide?: NormalizedBox;
  }
): Promise<VideoCalibrationDetectionResponse> {
  const params = options?.refreshEarlyReference
    ? "?refresh_early_reference=true"
    : "";
  const body = {
    striker_guide: options?.strikerGuide ?? null,
    non_striker_guide: options?.nonStrikerGuide ?? null
  };
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/detect${params}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Stump detection returned ${response.status}.`);
  }
  return withBrowserSafeDetectionUrls(
    await response.json() as VideoCalibrationDetectionResponse
  );
}


export async function confirmVideoAnalysisCalibration(
  analysisId: string,
  request: VideoCalibrationConfirmationRequest
): Promise<ConfirmedVideoCalibrationResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/confirm`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Calibration confirmation returned ${response.status}.`);
  }
  return withBrowserSafeCalibrationUrls(
    await response.json() as ConfirmedVideoCalibrationResponse
  );
}


export async function initialiseVideoAnalysisCalibrationV2(
  analysisId: string
): Promise<CalibrationV2InitialiseResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2/initialise`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Calibration v2 initialisation returned ${response.status}.`
    );
  }
  return withBrowserSafeCalibrationV2InitialiseUrls(
    await response.json() as CalibrationV2InitialiseResponse
  );
}


export async function confirmVideoAnalysisCalibrationV2(
  analysisId: string,
  request: CalibrationV2ConfirmRequest
): Promise<CalibrationV2Result> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2/confirm`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Calibration v2 confirmation returned ${response.status}.`
    );
  }
  return withBrowserSafeCalibrationV2Urls(
    await response.json() as CalibrationV2Result
  );
}


export async function getVideoAnalysisCalibrationV2(
  analysisId: string
): Promise<CalibrationV2Result | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Calibration v2 lookup returned ${response.status}.`
    );
  }
  return withBrowserSafeCalibrationV2Urls(
    await response.json() as CalibrationV2Result
  );
}


export async function initialiseWicketCameraPose(
  analysisId: string
): Promise<WicketCameraPoseInitialiseResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2/camera-pose/initialise`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Camera-pose initialisation returned ${response.status}.`
    );
  }
  return withBrowserSafeCameraPoseInitialiseUrls(
    await response.json() as WicketCameraPoseInitialiseResponse
  );
}


export async function solveWicketCameraPose(
  analysisId: string,
  request: WicketCameraPoseSolveRequest
): Promise<WicketCameraPoseResult> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2/camera-pose/solve`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Camera-pose solve returned ${response.status}.`
    );
  }
  return withBrowserSafeCameraPoseUrls(
    await response.json() as WicketCameraPoseResult
  );
}


export async function getWicketCameraPose(
  analysisId: string
): Promise<WicketCameraPoseResult | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/v2/camera-pose`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Camera-pose lookup returned ${response.status}.`
    );
  }
  return withBrowserSafeCameraPoseUrls(
    await response.json() as WicketCameraPoseResult
  );
}


export async function getVideoAnalysisCalibration(
  analysisId: string
): Promise<ConfirmedVideoCalibrationResponse | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(response, `Calibration lookup returned ${response.status}.`);
  }
  return withBrowserSafeCalibrationUrls(
    await response.json() as ConfirmedVideoCalibrationResponse
  );
}


export type VideoBallDetectionJobStatus =
  | "queued"
  | "loading_model"
  | "processing"
  | "writing_video"
  | "saving_results"
  | "ready"
  | "failed"
  | "ball_detector_missing";


export type VideoBallDetectionResultLinks = {
  processed_video_url: string;
  detections_json_url: string;
  detections_csv_url: string;
  detection_summary_url: string;
};


export type VideoBallDetectionStartResponse = {
  success: boolean;
  status: "queued";
  analysis_id: string;
  job_id: string;
  progress: number;
  current_frame: number;
  total_frames: number;
  ball_detector_model_key: string;
  ball_detector_model_name: string;
  message: string;
};


export type VideoBallDetectionJobResponse = {
  success: boolean;
  status: VideoBallDetectionJobStatus;
  analysis_id: string;
  job_id: string;
  progress: number;
  current_frame: number;
  total_frames: number;
  created_at: string;
  updated_at: string;
  model_path_used?: string | null;
  ball_detector_model_key: string;
  ball_detector_model_name: string;
  error_message?: string | null;
  result?: VideoBallDetectionResultLinks | null;
  message: string;
};


export type VideoBallDetectionSummary = {
  analysis_id: string;
  status: "ready";
  created_at: string;
  completed_at: string;
  original_video_url: string;
  processed_video_url: string;
  detections_json_url: string;
  detections_csv_url: string;
  detection_summary_url: string;
  detector?: {
    requested_key: string;
    selected_key: string;
    display_name: string;
    model_file: string;
  } | null;
  model_path_used: string;
  model_warning?: string | null;
  model_class_names: string[];
  device_used: string;
  imgsz: 960;
  confidence_threshold: 0.15;
  frame_stride: 1;
  max_det: 20;
  total_frames: number;
  frames_processed: number;
  frames_with_candidates: number;
  frames_without_candidates: number;
  total_candidates: number;
  frames_with_multiple_candidates: number;
  candidates_inside_pitch_corridor: number;
  candidates_outside_pitch_corridor: number;
  candidates_without_corridor_information: number;
  best_confidence: number;
  average_confidence: number;
  average_candidates_per_detected_frame: number;
  processing_duration_seconds: number;
  output_video_frame_count: number;
  input_fps: number;
  output_fps: number;
  input_duration_seconds: number;
  output_duration_seconds: number;
  message: string;
};


export type VideoBallDetectionResultResponse = {
  success: true;
  status: "ready";
  analysis_id: string;
  summary: VideoBallDetectionSummary;
  frame_candidate_counts: number[];
  message: string;
};


function withBrowserSafeVideoBallDetectionLinks(
  links?: VideoBallDetectionResultLinks | null
): VideoBallDetectionResultLinks | null | undefined {
  if (!links) return links;
  return {
    processed_video_url: resolveApiUrl(links.processed_video_url) ?? links.processed_video_url,
    detections_json_url: resolveApiUrl(links.detections_json_url) ?? links.detections_json_url,
    detections_csv_url: resolveApiUrl(links.detections_csv_url) ?? links.detections_csv_url,
    detection_summary_url: resolveApiUrl(links.detection_summary_url) ?? links.detection_summary_url
  };
}


function withBrowserSafeVideoBallDetectionResult(
  result: VideoBallDetectionResultResponse
): VideoBallDetectionResultResponse {
  return {
    ...result,
    summary: {
      ...result.summary,
      original_video_url: resolveApiUrl(result.summary.original_video_url) ?? result.summary.original_video_url,
      processed_video_url: resolveApiUrl(result.summary.processed_video_url) ?? result.summary.processed_video_url,
      detections_json_url: resolveApiUrl(result.summary.detections_json_url) ?? result.summary.detections_json_url,
      detections_csv_url: resolveApiUrl(result.summary.detections_csv_url) ?? result.summary.detections_csv_url,
      detection_summary_url: resolveApiUrl(result.summary.detection_summary_url) ?? result.summary.detection_summary_url
    }
  };
}


export async function startVideoBallDetection(
  analysisId: string,
  ballDetectorModelKey: BallDetectorModelKey = "automatic"
): Promise<VideoBallDetectionStartResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection/start`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ball_detector_model_key: ballDetectorModelKey })
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Ball detection start returned ${response.status}.`);
  }
  return response.json() as Promise<VideoBallDetectionStartResponse>;
}


export async function getBallDetectorModels(): Promise<BallDetectorModelsResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/detector-models`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Detector model lookup returned ${response.status}.`);
  }
  return response.json() as Promise<BallDetectorModelsResponse>;
}


export async function getVideoBallDetectionJob(
  analysisId: string,
  jobId: string
): Promise<VideoBallDetectionJobResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection/job/${encodeURIComponent(jobId)}`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Ball detection job lookup returned ${response.status}.`);
  }
  const job = await response.json() as VideoBallDetectionJobResponse;
  return {
    ...job,
    result: withBrowserSafeVideoBallDetectionLinks(job.result)
  };
}


export async function getVideoBallDetectionResult(
  analysisId: string
): Promise<VideoBallDetectionResultResponse | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(response, `Ball detection result lookup returned ${response.status}.`);
  }
  return withBrowserSafeVideoBallDetectionResult(
    await response.json() as VideoBallDetectionResultResponse
  );
}


export type VideoBallTrackingJobStatus =
  | "queued"
  | "loading_detections"
  | "analysing_candidates"
  | "building_track"
  | "recovering_gaps"
  | "fitting_physics"
  | "rendering_video"
  | "saving_results"
  | "ready"
  | "failed"
  | "no_reliable_track";


export type VideoBallTrackingResultLinks = {
  tracking_video_url: string;
  tracking_json_url: string;
  tracking_csv_url: string;
  tracking_summary_url: string;
  delivery_replay_url?: string | null;
  physics_result_url?: string | null;
};


export type VideoBallTrackingStartResponse = {
  success: true;
  status: "queued";
  analysis_id: string;
  job_id: string;
  progress: number;
  message: string;
};


export type VideoBallTrackingJobResponse = {
  success: boolean;
  status: VideoBallTrackingJobStatus;
  analysis_id: string;
  job_id: string;
  progress: number;
  created_at: string;
  updated_at: string;
  error_message?: string | null;
  result?: VideoBallTrackingResultLinks | null;
  message: string;
};


export type TrackingProvenance =
  | "OBSERVED"
  | "TRACKER_RECOVERED"
  | "PHYSICS_RECONSTRUCTED"
  | "PROJECTED";


export type VideoBallTrackingPoint = {
  frame_index: number;
  timestamp_seconds: number;
  source: "observed" | "predicted" | "recovered";
  provenance: TrackingProvenance;
  candidate_id?: string | null;
  x: number;
  y: number;
  normalized_x: number;
  normalized_y: number;
  confidence: number;
  uncertainty?: number;
  vx: number;
  vy: number;
  prediction_error?: number | null;
  inside_pitch_corridor?: boolean | null;
};


export type PrimaryBounceResult = {
  bounce_detected: boolean | "uncertain";
  bounce_frame?: number | null;
  bounce_timestamp_seconds?: number | null;
  bounce_x?: number | null;
  bounce_y?: number | null;
  bounce_normalized_x?: number | null;
  bounce_normalized_y?: number | null;
  confidence: number;
  evidence: string[];
  warnings: string[];
};


export type VideoBallTrackingSummary = {
  analysis_id: string;
  status: "ready" | "no_reliable_track";
  total_video_frames: number;
  raw_candidate_count: number;
  candidate_frames: number;
  track_start_frame?: number | null;
  track_end_frame?: number | null;
  first_supported_delivery_point?: number | null;
  track_start_label?: "track_start" | "unavailable";
  track_duration_frames: number;
  track_duration_seconds: number;
  observed_track_points: number;
  predicted_points: number;
  recovered_points: number;
  physics_reconstructed_points?: number;
  projected_points?: number;
  rejected_candidates: number;
  longest_gap_frames: number;
  observation_ratio?: number;
  average_observed_confidence: number;
  consistency_score?: number;
  track_confidence: number;
  track_quality: "high" | "medium" | "low" | "failed";
  approximate_direction: string;
  possible_bounce_transition_detected: boolean | "uncertain";
  bounce_detected?: boolean | "uncertain";
  bounce_frame?: number | null;
  bounce_confidence?: number;
  tracking_video_url: string;
  delivery_replay_url?: string | null;
  physics_result_url?: string | null;
  physics_engine_version?: "v1" | null;
  physics_status?: DeliveryPhysicsStatus | null;
  tracking_json_url: string;
  tracking_csv_url: string;
  tracking_summary_url: string;
  processing_duration_seconds: number;
  message: string;
};


export type PhysicsConfidence =
  | "HIGH"
  | "MEDIUM"
  | "LOW"
  | "INSUFFICIENT_EVIDENCE";


export type VirtualPitchWorldPoint = { x: number; y: number; z: number };
export type VirtualPitchPixelPoint = { x: number; y: number };


export type VirtualPitchSpecification = {
  virtual_pitch_model_version: "v1";
  coordinate_system: {
    units: "metres";
    handedness: "right_handed";
    origin: "bowler_end_middle_stump_base";
    x_axis: "lateral_camera_neutral_right";
    y_axis: "bowler_to_striker";
    z_axis: "up";
    description: string;
    off_leg_assignment: "not_assigned";
  };
  dimensions: {
    pitch_length_m: number;
    pitch_width_m: number;
    wicket_width_m: number;
    stump_height_m: number;
    stump_diameter_min_m: number;
    stump_diameter_max_m: number;
    bowling_crease_length_m: number;
    popping_crease_offset_m: number;
    return_crease_offset_m: number;
  };
  landmarks: Array<{
    semantic_id: string;
    point: VirtualPitchWorldPoint;
    geometry_category: string;
    geometry_class: "official" | "analytical" | "optional";
    end: "bowler" | "striker" | "both" | "none";
    calibration_anchor: boolean;
    description: string;
  }>;
  stumps: Array<{
    primitive_id: string;
    centre: VirtualPitchWorldPoint;
    radius_m: number;
    height_m: number;
    orientation: VirtualPitchWorldPoint;
    end: "bowler" | "striker";
    stump_index: "left" | "middle" | "right";
  }>;
  bails: Array<{
    primitive_id: string;
    start: VirtualPitchWorldPoint;
    end_point: VirtualPitchWorldPoint;
    radius_m: number;
    end: "bowler" | "striker";
    bail_index: string;
  }>;
  line_segments: Array<{
    primitive_id: string;
    start: VirtualPitchWorldPoint;
    end_point: VirtualPitchWorldPoint;
    line_category: string;
    geometry_class: "official" | "analytical" | "optional";
    line_width_m: number;
    end: "bowler" | "striker" | "both" | "none";
    profile_id?: string | null;
  }>;
  polygons: Array<{
    primitive_id: string;
    vertices: VirtualPitchWorldPoint[];
    polygon_category: string;
    geometry_class: "official" | "analytical" | "optional";
    display_opacity: number;
  }>;
  profiles: Array<{
    profile_id: string;
    label: string;
    geometry_class: "official" | "analytical" | "optional";
    description: string;
    universal_official_geometry: boolean;
  }>;
  synthetic_camera_names: string[];
};


export type VirtualPitchCamera = {
  name: string;
  description: string;
  image_width: number;
  image_height: number;
  camera_matrix: number[][];
  distortion_coefficients: number[];
  rotation_vector: number[];
  rotation_matrix: number[][];
  translation_vector: number[];
  camera_position_world: number[];
  target_world: number[];
  near_m: number;
  far_m: number;
  horizontal_fov_degrees: number;
  developer_only: true;
};


export type ProjectedPitchGeometry = {
  virtual_pitch_model_version: "v1";
  source_camera: VirtualPitchCamera;
  projected_landmarks: Array<{
    semantic_id: string;
    world_point: VirtualPitchWorldPoint;
    pixel_point?: VirtualPitchPixelPoint | null;
    visible: boolean;
    in_frame: boolean;
    depth_m: number;
    projection_valid: boolean;
  }>;
  projected_line_segments: Array<{
    primitive_id: string;
    line_category: string;
    geometry_class: "official" | "analytical" | "optional";
    pixel_start?: VirtualPitchPixelPoint | null;
    pixel_end?: VirtualPitchPixelPoint | null;
    projection_valid: boolean;
    partially_out_of_frame: boolean;
  }>;
  projected_stumps: Array<{
    primitive_id: string;
    end: "bowler" | "striker";
    stump_index: "left" | "middle" | "right";
    pixel_base?: VirtualPitchPixelPoint | null;
    pixel_top?: VirtualPitchPixelPoint | null;
    projected_height_px?: number | null;
    projected_radius_px?: number | null;
    projection_valid: boolean;
    in_frame: boolean;
  }>;
  projected_polygons: Array<{
    primitive_id: string;
    polygon_category: string;
    geometry_class: "official" | "analytical" | "optional";
    pixel_vertices: Array<VirtualPitchPixelPoint | null>;
    projection_valid: boolean;
    partially_out_of_frame: boolean;
  }>;
  projected_bails: Array<{
    primitive_id: string;
    pixel_start?: VirtualPitchPixelPoint | null;
    pixel_end?: VirtualPitchPixelPoint | null;
    projection_valid: boolean;
  }>;
  diagnostics: {
    valid_landmark_count: number;
    in_frame_landmark_count: number;
    behind_camera_count: number;
    out_of_frame_count: number;
    nearer_wicket: "bowler" | "striker" | "equal" | "unavailable";
    perspective_order_valid: boolean;
    warnings: string[];
  };
  synthetic_only: true;
};


export type SyntheticPitchPreviewResponse = {
  specification: VirtualPitchSpecification;
  projection: ProjectedPitchGeometry;
  selected_profile: string;
  developer_only: true;
  registration_status: "not_registered_to_video";
  message: string;
};


export type DeliveryPhysicsStatus =
  | "SUCCESS"
  | "PARTIAL"
  | "IMAGE_SPACE_ONLY"
  | "INSUFFICIENT_EVIDENCE"
  | "FAILED";

export type PhysicsTrajectorySample = {
  frame_index: number;
  timestamp_seconds: number;
  world_x_m?: number | null;
  world_y_m?: number | null;
  world_z_m?: number | null;
  pixel_x: number;
  pixel_y: number;
  speed_mps?: number | null;
  provenance: "OBSERVED" | "RECONSTRUCTED" | "PROJECTED";
  confidence: number;
};

export type DeliveryPhysicsResult = {
  physics_engine_version: "v1";
  status: DeliveryPhysicsStatus;
  analysis_id: string;
  coordinate_system: string;
  pitch_geometry: {
    pitch_length_m: number;
    pitch_width_m: number;
  };
  calibration: {
    mode: "METRIC_3D" | "METRIC_GROUND_PLANE" | "IMAGE_SPACE_ONLY";
    confidence: PhysicsConfidence | "UNAVAILABLE";
    reprojection_error_px?: number | null;
    failure_reason?: string | null;
  };
  trajectory_samples: PhysicsTrajectorySample[];
  bounce: {
    status: "DETECTED" | "ESTIMATED" | "INSUFFICIENT_EVIDENCE";
    frame_index?: number | null;
    distance_from_striker_wicket_m?: number | null;
    lateral_offset_m?: number | null;
    confidence: PhysicsConfidence;
  };
  speed: {
    earliest_measured_speed_kmh?: number | null;
    average_pre_bounce_speed_kmh?: number | null;
    speed_at_bounce_kmh?: number | null;
    average_post_bounce_speed_kmh?: number | null;
    confidence: PhysicsConfidence;
    unavailable_reason?: string | null;
  };
  pre_bounce_lateral_movement: {
    movement_m?: number | null;
    movement_cm?: number | null;
    direction: string;
    lateral_acceleration_mps2?: number | null;
    confidence: PhysicsConfidence;
    unavailable_reason?: string | null;
  };
  post_bounce_movement: {
    status: "MEASURED" | "PROJECTED" | "UNAVAILABLE";
    lateral_turn_cm_at_last_observation?: number | null;
    speed_loss_kmh?: number | null;
    confidence: PhysicsConfidence;
    unavailable_reason?: string | null;
  };
  line_and_length: {
    line: string;
    length: string;
    bounce_distance_from_striker_m?: number | null;
    lateral_offset_from_middle_m?: number | null;
  };
  fit_diagnostics: {
    selected_model: string;
    weighted_reprojection_rmse_px?: number | null;
    inlier_frames: number[];
    outlier_frames: number[];
    processing_duration_seconds: number;
  };
  confidence: PhysicsConfidence;
  confidence_score: number;
  exact_spin_rpm: null;
  exact_spin_rpm_unavailable_reason: string;
  warnings: string[];
  physics_result_url?: string | null;
};


export type VideoBallTrackingResultResponse = {
  success: boolean;
  status: "ready" | "no_reliable_track";
  analysis_id: string;
  summary: VideoBallTrackingSummary;
  primary_track: VideoBallTrackingPoint[];
  bounce?: PrimaryBounceResult | null;
  physics?: DeliveryPhysicsResult | null;
  message: string;
};


function withBrowserSafeTrackingLinks(
  links?: VideoBallTrackingResultLinks | null
): VideoBallTrackingResultLinks | null | undefined {
  if (!links) return links;
  return {
    tracking_video_url: resolveApiUrl(links.tracking_video_url) ?? links.tracking_video_url,
    tracking_json_url: resolveApiUrl(links.tracking_json_url) ?? links.tracking_json_url,
    tracking_csv_url: resolveApiUrl(links.tracking_csv_url) ?? links.tracking_csv_url,
    tracking_summary_url: resolveApiUrl(links.tracking_summary_url) ?? links.tracking_summary_url,
    delivery_replay_url: links.delivery_replay_url
      ? resolveApiUrl(links.delivery_replay_url) ?? links.delivery_replay_url
      : links.delivery_replay_url,
    physics_result_url: links.physics_result_url
      ? resolveApiUrl(links.physics_result_url) ?? links.physics_result_url
      : links.physics_result_url
  };
}


function withBrowserSafeTrackingResult(
  result: VideoBallTrackingResultResponse
): VideoBallTrackingResultResponse {
  return {
    ...result,
    summary: {
      ...result.summary,
      tracking_video_url: resolveApiUrl(result.summary.tracking_video_url) ?? result.summary.tracking_video_url,
      delivery_replay_url: result.summary.delivery_replay_url
        ? resolveApiUrl(result.summary.delivery_replay_url) ?? result.summary.delivery_replay_url
        : result.summary.delivery_replay_url,
      physics_result_url: result.summary.physics_result_url
        ? resolveApiUrl(result.summary.physics_result_url) ?? result.summary.physics_result_url
        : result.summary.physics_result_url,
      tracking_json_url: resolveApiUrl(result.summary.tracking_json_url) ?? result.summary.tracking_json_url,
      tracking_csv_url: resolveApiUrl(result.summary.tracking_csv_url) ?? result.summary.tracking_csv_url,
      tracking_summary_url: resolveApiUrl(result.summary.tracking_summary_url) ?? result.summary.tracking_summary_url
    },
    physics: result.physics
      ? {
          ...result.physics,
          physics_result_url: result.physics.physics_result_url
            ? resolveApiUrl(result.physics.physics_result_url) ?? result.physics.physics_result_url
            : result.physics.physics_result_url
        }
      : result.physics
  };
}


export type WicketObservationQuality = "HIGH" | "MEDIUM" | "LOW" | "UNAVAILABLE";
export type WicketObservationStatus =
  | "READY_FOR_REGISTRATION_EXPERIMENT"
  | "PARTIAL"
  | "INSUFFICIENT_WICKETS"
  | "INSUFFICIENT_LANDMARKS"
  | "UNSTABLE"
  | "FAILED";

export type WicketObservationLandmark = {
  semantic_id: string;
  geometry_type: "POINT" | "LINE";
  pixel_x?: number | null;
  pixel_y?: number | null;
  line?: {
    start: { x: number; y: number };
    end: { x: number; y: number };
  } | null;
  confidence: number;
  uncertainty_px: number;
  registration_role:
    | "PRIMARY_ANCHOR"
    | "SECONDARY_ANCHOR"
    | "VALIDATION_ONLY"
    | "DO_NOT_USE";
  quality: WicketObservationQuality;
  status: "AVAILABLE" | "UNAVAILABLE" | "REJECTED";
  rejection_reason?: string | null;
};

export type RealWicketObservation = {
  region: {
    bbox: { x: number; y: number; width: number; height: number };
    detector_confidence: number;
    temporal_support: number;
    supporting_frame_ids: number[];
    perspective_role:
      | "NEAR_WICKET_CANDIDATE"
      | "FAR_WICKET_CANDIDATE"
      | "UNRESOLVED_WICKET";
    stability: "STABLE" | "PARTIALLY_STABLE" | "UNSTABLE" | "NOT_FOUND";
    quality: WicketObservationQuality;
    uncertainty_px: number;
  };
  roi: {
    source_frame_width: number;
    source_frame_height: number;
    x: number;
    y: number;
    width: number;
    height: number;
  };
  coarse_landmarks: WicketObservationLandmark[];
  detailed_landmarks: WicketObservationLandmark[];
  detailed_landmarks_status: "AVAILABLE" | "PARTIAL" | "INSUFFICIENT_EVIDENCE";
  quality_score: number;
  warnings: string[];
};

export type WicketObservationResult = {
  version: "wicket_observations_v1";
  analysis_id: string;
  status: WicketObservationStatus;
  setup_frame?: {
    frame_index: number;
    timestamp_seconds: number;
    image_width: number;
    image_height: number;
    score: number;
    sharpness: number;
    brightness: number;
    wicket_detection_count: number;
    detection_stability: number;
  } | null;
  supporting_frames: Array<{
    frame_index: number;
    timestamp_seconds: number;
    score: number;
  }>;
  near_wicket?: RealWicketObservation | null;
  far_wicket?: RealWicketObservation | null;
  assignment_hypotheses: Array<{
    hypothesis_id: "A" | "B";
    near_semantic_end: "bowler" | "striker";
    far_semantic_end: "bowler" | "striker";
    finalised: false;
    confidence: number;
    evidence: string[];
  }>;
  warnings: string[];
  diagnostics: {
    detector_model_path: string;
    detector_class_labels: string[];
    sampled_frame_ids: number[];
    raw_detections: Array<{
      frame_index: number;
      bbox: { x: number; y: number; width: number; height: number };
      confidence: number;
      perspective_role:
        | "NEAR_WICKET_CANDIDATE"
        | "FAR_WICKET_CANDIDATE"
        | "UNRESOLVED_WICKET";
    }>;
    setup_frame_image_url?: string | null;
    raw_detection_overlay_url?: string | null;
    landmark_overlay_url?: string | null;
  };
  future_registration_readiness: WicketObservationStatus;
  message: string;
  developer_only: true;
};


function withBrowserSafeWicketObservation(
  result: WicketObservationResult
): WicketObservationResult {
  return {
    ...result,
    diagnostics: {
      ...result.diagnostics,
      setup_frame_image_url: resolveApiUrl(result.diagnostics.setup_frame_image_url),
      raw_detection_overlay_url: resolveApiUrl(result.diagnostics.raw_detection_overlay_url),
      landmark_overlay_url: resolveApiUrl(result.diagnostics.landmark_overlay_url)
    }
  };
}


export async function runWicketObservations(
  analysisId: string
): Promise<WicketObservationResult> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/wicket-observations/run`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket observation returned ${response.status}.`
    );
  }
  return withBrowserSafeWicketObservation(
    await response.json() as WicketObservationResult
  );
}


export async function getWicketObservations(
  analysisId: string
): Promise<WicketObservationResult | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/wicket-observations`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket observation lookup returned ${response.status}.`
    );
  }
  return withBrowserSafeWicketObservation(
    await response.json() as WicketObservationResult
  );
}


export async function getVirtualPitchSpecification(): Promise<VirtualPitchSpecification> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/virtual-pitch`,
    { cache: "force-cache" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Virtual pitch lookup returned ${response.status}.`
    );
  }
  return response.json() as Promise<VirtualPitchSpecification>;
}


export async function getSyntheticPitchPreview(
  cameraName: string,
  profile = "analytical"
): Promise<SyntheticPitchPreviewResponse> {
  const query = new URLSearchParams({
    camera_name: cameraName,
    profile
  });
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/virtual-pitch/synthetic-projection?${query}`,
    { cache: "force-cache" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Synthetic pitch projection returned ${response.status}.`
    );
  }
  return response.json() as Promise<SyntheticPitchPreviewResponse>;
}


export async function startVideoBallTracking(
  analysisId: string
): Promise<VideoBallTrackingStartResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/tracking/start`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Moving Ball Tracker start returned ${response.status}.`);
  }
  return response.json() as Promise<VideoBallTrackingStartResponse>;
}


export async function getVideoBallTrackingJob(
  analysisId: string,
  jobId: string
): Promise<VideoBallTrackingJobResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/tracking/job/${encodeURIComponent(jobId)}`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Moving Ball Tracker job lookup returned ${response.status}.`);
  }
  const job = await response.json() as VideoBallTrackingJobResponse;
  return {
    ...job,
    result: withBrowserSafeTrackingLinks(job.result)
  };
}


export async function getVideoBallTrackingResult(
  analysisId: string
): Promise<VideoBallTrackingResultResponse | null> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/tracking`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(response, `Moving Ball Tracker result lookup returned ${response.status}.`);
  }
  return withBrowserSafeTrackingResult(
    await response.json() as VideoBallTrackingResultResponse
  );
}
