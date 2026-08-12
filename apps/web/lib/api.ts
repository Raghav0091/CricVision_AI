import type { BatPoseCalibrationResponse, BoxLayout, CalibrationResponse, CapturedFrame } from "./types";
import type { CameraBridgeInput } from "./virtual-pitch/opencvCameraBridge";
import type {
  CalibrationResult,
  WicketBoxCalibrationAcceptRequest,
  WicketBoxCalibrationAcceptResponse,
  WicketBoxCalibrationDetectResponse,
  WicketBoxCalibrationRegisterRequest,
  WicketBoxCalibrationRegisterResponse,
} from "./wicketCalibration/types";
import type { ReplayPayloadV1 } from "./virtual-pitch-replay/types";
import type {
  CheckerboardSpecInput,
  DeviceCalibrationResponse,
  DeviceLensProfile,
} from "./deviceCalibration/types";


function resolveApiBaseUrl(): string {
  const configured = process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_URL;
  if (configured) {
    return configured.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    const { hostname, origin } = window.location;
    // ponytail: phone tunnel (trycloudflare) or any remote host uses Next proxy.
    if (hostname !== "localhost" && hostname !== "127.0.0.1") {
      return `${origin}/cricvision-api`.replace(/\/$/, "");
    }
  }
  if (process.env.NEXT_PUBLIC_USE_SAME_ORIGIN_API === "true" && typeof window !== "undefined") {
    return `${window.location.origin}/cricvision-api`.replace(/\/$/, "");
  }
  return "http://127.0.0.1:8000";
}


export function getApiBaseUrl(): string {
  return resolveApiBaseUrl();
}


/** @deprecated Use getApiBaseUrl() so phone tunnel URLs resolve at request time. */
export const API_BASE_URL = "http://127.0.0.1:8000";


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
  return { ...result, processed_video_url: `${getApiBaseUrl()}${url}` };
}


export async function solveCalibration(frame: CapturedFrame, boxLayout: BoxLayout): Promise<CalibrationResponse> {
  const response = await fetch(`${getApiBaseUrl()}/calibration/solve`, {
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


export type ManualBatBoxes = {
  striker: { x: number; y: number; width: number; height: number };
  non_striker: { x: number; y: number; width: number; height: number };
};

export type BatDetectorModelKey = "cricshot10k" | "bat_v2";

export async function runBatPoseCalibration(
  analysisId: string,
  frame: CapturedFrame,
  boxLayout: BoxLayout | null,
  batHeightM: number,
  manualBoxes: ManualBatBoxes | null = null,
  batDetectorModelKey: BatDetectorModelKey = "cricshot10k"
): Promise<BatPoseCalibrationResponse> {
  const response = await fetch(`${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/bat-pose-calibration/solve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_data_url: frame.dataUrl,
      frame_width: frame.width,
      frame_height: frame.height,
      box_layout: manualBoxes ? null : boxLayout,
      bat_height_m: batHeightM,
      manual_boxes: manualBoxes,
      bat_detector_model_key: batDetectorModelKey
    })
  });
  if (!response.ok) {
    throw new Error(`Bat-pose calibration service returned ${response.status}.`);
  }
  return response.json() as Promise<BatPoseCalibrationResponse>;
}


export async function getBatPoseCalibration(analysisId: string): Promise<BatPoseCalibrationResponse> {
  const response = await fetch(`${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/bat-pose-calibration`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw new Error(`Bat-pose calibration service returned ${response.status}.`);
  }
  return response.json() as Promise<BatPoseCalibrationResponse>;
}


export async function detectBallInDeliveryClip(blob: Blob, deliveryIndex: number, sessionId: string): Promise<BallDetectionClipResponse> {
  const extension = blob.type.includes("mp4") ? "mp4" : "webm";
  const formData = new FormData();
  formData.append("video", new File([blob], `delivery-${deliveryIndex}.${extension}`, { type: blob.type || `video/${extension}` }));
  formData.append("delivery_index", String(deliveryIndex));
  formData.append("session_id", sessionId);
  formData.append("source_mode", "experimental_delivery_test");
  formData.append("processing_mode", "quality");
  const response = await fetch(`${getApiBaseUrl()}/analysis/ball-detection-clip`, {
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
  return `${getApiBaseUrl()}${url}`;
}


async function sessionRequest(url: string, init?: RequestInit): Promise<ExperimentalSession> {
  const response = await fetch(`${getApiBaseUrl()}${url}`, { cache: "no-store", ...init });
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


export function completeExperimentalSession(sessionId: string): Promise<ExperimentalSession> {
  return sessionRequest(`/sessions/${encodeURIComponent(sessionId)}/capture-complete`, { method: "POST" });
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
  const response = await fetch(`${getApiBaseUrl()}/video-analysis/prepare`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Video preparation returned ${response.status}.`);
  }
  return withBrowserSafeAnalysisUrls(await response.json() as VideoAnalysisPreparedResponse);
}


export async function prepareQuickTestVideo(file: File): Promise<VideoAnalysisPreparedResponse> {
  const formData = new FormData();
  formData.append("video", file);
  const response = await fetch(`${getApiBaseUrl()}/quick-test/prepare`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Quick Test upload returned ${response.status}.`);
  }
  return withBrowserSafeAnalysisUrls(await response.json() as VideoAnalysisPreparedResponse);
}


export async function getVideoAnalysis(analysisId: string): Promise<VideoAnalysisPreparedResponse> {
  const response = await fetch(`${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}`, {
    cache: "no-store"
  });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Video analysis lookup returned ${response.status}.`);
  }
  return withBrowserSafeAnalysisUrls(await response.json() as VideoAnalysisPreparedResponse);
}


export type CalibrationQualityGradeV2 =
  | "excellent"
  | "good"
  | "usable"
  | "weak"
  | "poor"
  | "insufficient_geometry";


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
  frames?: VideoBallDetectionFrame[] | null;
  message: string;
};


export type VideoBallDetectionCandidate = {
  candidate_id: string;
  class_id: number;
  class_name: string;
  confidence: number;
  bbox_xyxy: [number, number, number, number];
  bbox_normalized: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  center: { x: number; y: number };
  center_normalized: { x: number; y: number };
  width_pixels: number;
  height_pixels: number;
  area_pixels: number;
  inside_pitch_corridor?: boolean | null;
};


export type VideoBallDetectionFrame = {
  frame_index: number;
  timestamp_seconds: number;
  processed: boolean;
  detections: VideoBallDetectionCandidate[];
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
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection/start`,
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
    `${getApiBaseUrl()}/video-analysis/detector-models`,
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
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection/job/${encodeURIComponent(jobId)}`,
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
  analysisId: string,
  includeFrames = false
): Promise<VideoBallDetectionResultResponse | null> {
  const query = includeFrames ? "?include_frames=true" : "";
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection${query}`,
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


export type TrackingCandidateScoreComponents = {
  detector_confidence: number;
  prediction_proximity: number;
  motion: number;
  direction: number;
  size_consistency: number;
  corridor: number;
  static_penalty: number;
  jump_penalty: number;
  total: number;
};


export type TrackingCandidateDiagnostic = {
  frame_index: number;
  candidate_id: string;
  selected: boolean;
  selection_reason: string;
  static_likelihood: number;
  score_components?: TrackingCandidateScoreComponents | null;
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
  tracking_job_id?: string | null;
  source_track_id?: string | null;
  track_source_consistent?: boolean | null;
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
  replay_payload_url?: string | null;
  finalized_track_url?: string | null;
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
  overall_stump_to_stump: {
    speed_mps?: number | null;
    speed_kph?: number | null;
    start_time_seconds?: number | null;
    end_time_seconds?: number | null;
    travelled_distance_m?: number | null;
    observed_fraction?: number;
    recovered_fraction?: number;
    projected_fraction?: number;
    confidence: PhysicsConfidence;
    status: "MEASURED" | "PARTIALLY_PROJECTED" | "UNAVAILABLE";
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
  render_track?: VideoBallTrackingPoint[];
  raw_primary_track?: VideoBallTrackingPoint[];
  candidate_diagnostics?: TrackingCandidateDiagnostic[];
  bounce?: PrimaryBounceResult | null;
  physics?: DeliveryPhysicsResult | null;
  track_source_consistency_errors?: string[];
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
      replay_payload_url: result.summary.replay_payload_url
        ? resolveApiUrl(result.summary.replay_payload_url) ?? result.summary.replay_payload_url
        : result.summary.replay_payload_url,
      finalized_track_url: result.summary.finalized_track_url
        ? resolveApiUrl(result.summary.finalized_track_url) ?? result.summary.finalized_track_url
        : result.summary.finalized_track_url,
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


export async function getVirtualPitchSpecification(): Promise<VirtualPitchSpecification> {
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/virtual-pitch`,
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


export type CameraBridgeApiResponse = {
  bridge_version: "opencv_three_camera_bridge_v1";
  status: "AVAILABLE" | "UNAVAILABLE";
  camera?: {
    source: string;
    source_version: string;
    analysis_id?: string | null;
    candidate_id: string;
    accepted: boolean;
    classification: string;
    image_width: number;
    image_height: number;
    camera_matrix: number[][];
    distortion: {
      mode: "ZERO_DISTORTION" | "PREUNDISTORTED_FRAME" | "NONZERO_DISTORTION_UNSUPPORTED";
      coefficients: number[];
      frame_preundistorted: boolean;
      exact_pinhole_rendering_supported: boolean;
      warning?: string | null;
    };
    rotation_vector: number[];
    rotation_matrix: number[][];
    translation_vector: number[];
    near_m: number;
    far_m: number;
    setup_frame?: {
      image_url: string;
    } | null;
    warnings: string[];
  } | null;
  projected_pitch_geometry?: ProjectedPitchGeometry | null;
  warnings: string[];
  message: string;
};


export type NormalizedCameraBridgeResponse = {
  camera: CameraBridgeInput & { setup_frame_url?: string | null };
  projection: ProjectedPitchGeometry | null;
};


function normalizeCameraBridgeResponse(response: CameraBridgeApiResponse): NormalizedCameraBridgeResponse {
  const camera = response.camera;
  if (response.status !== "AVAILABLE" || !camera) {
    throw new Error(response.message || "No selected camera candidate is available.");
  }
  const setupFrameUrl = camera.setup_frame?.image_url;
  return {
    camera: {
      source: camera.source,
      source_version: camera.source_version,
      analysis_id: camera.analysis_id ?? null,
      candidate_id: camera.candidate_id,
      accepted: camera.accepted,
      classification: camera.classification,
      image_width: camera.image_width,
      image_height: camera.image_height,
      camera_matrix: camera.camera_matrix as unknown as CameraBridgeInput["camera_matrix"],
      distortion_coefficients: camera.distortion.coefficients,
      rotation_representation: "matrix_and_rodrigues",
      rotation_vector: camera.rotation_vector as unknown as CameraBridgeInput["rotation_vector"],
      rotation_matrix: camera.rotation_matrix as unknown as CameraBridgeInput["rotation_matrix"],
      translation_vector: camera.translation_vector as unknown as CameraBridgeInput["translation_vector"],
      extrinsic_convention: "opencv_world_to_camera",
      world_coordinate_system: "cricvision_pitch_v1",
      setup_frame_url: setupFrameUrl
        ? `${getApiBaseUrl()}${setupFrameUrl.startsWith("/") ? "" : "/"}${setupFrameUrl}`
        : null,
      frame_preundistorted: camera.distortion.frame_preundistorted,
      near: camera.near_m,
      far: camera.far_m,
      warnings: [...camera.warnings, ...response.warnings]
    },
    projection: response.projected_pitch_geometry ?? null
  };
}


async function cameraBridgeRequest(url: string): Promise<NormalizedCameraBridgeResponse> {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw await videoAnalysisError(response, `Camera bridge lookup returned ${response.status}.`);
  }
  return normalizeCameraBridgeResponse(await response.json() as CameraBridgeApiResponse);
}


export function getSyntheticCameraBridge(cameraName: string): Promise<NormalizedCameraBridgeResponse> {
  const query = new URLSearchParams({ camera_name: cameraName });
  return cameraBridgeRequest(`${getApiBaseUrl()}/video-analysis/virtual-pitch/camera-bridge?${query}`);
}


export function getAnalysisCameraBridge(analysisId: string): Promise<NormalizedCameraBridgeResponse> {
  return cameraBridgeRequest(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/camera-bridge`
  );
}


export async function startVideoBallTracking(
  analysisId: string
): Promise<VideoBallTrackingStartResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/tracking/start`,
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
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/tracking/job/${encodeURIComponent(jobId)}`,
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
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/tracking`,
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


export type {
  CalibrationResult,
  WicketBoxCalibrationAcceptRequest,
  WicketBoxCalibrationAcceptResponse,
  WicketBoxCalibrationDetectResponse,
  WicketBoxCalibrationRegisterRequest,
  WicketBoxCalibrationRegisterResponse,
} from "./wicketCalibration/types";

export type { ReplayPayloadV1 } from "./virtual-pitch-replay/types";


export async function getWicketBoxCalibration(
  analysisId: string
): Promise<CalibrationResult | null> {
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/wicket-box-calibration`
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket box calibration returned ${response.status}.`
    );
  }
  return response.json() as Promise<CalibrationResult>;
}


export async function detectWicketBoxCalibration(
  analysisId: string,
  request: WicketBoxCalibrationRegisterRequest
): Promise<WicketBoxCalibrationDetectResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/wicket-box-calibration/detect`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket box calibration detect returned ${response.status}.`
    );
  }
  return response.json() as Promise<WicketBoxCalibrationDetectResponse>;
}


export async function registerWicketBoxCalibration(
  analysisId: string,
  request: WicketBoxCalibrationRegisterRequest,
  assignmentHypothesis?: "A" | "B" | null
): Promise<WicketBoxCalibrationRegisterResponse> {
  const query = assignmentHypothesis
    ? `?assignment_hypothesis=${encodeURIComponent(assignmentHypothesis)}`
    : "";
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/wicket-box-calibration/register${query}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket box calibration register returned ${response.status}.`
    );
  }
  return response.json() as Promise<WicketBoxCalibrationRegisterResponse>;
}


export async function acceptWicketBoxCalibration(
  analysisId: string,
  request: WicketBoxCalibrationAcceptRequest,
  assignmentHypothesis?: "A" | "B" | null
): Promise<WicketBoxCalibrationAcceptResponse> {
  const query = assignmentHypothesis
    ? `?assignment_hypothesis=${encodeURIComponent(assignmentHypothesis)}`
    : "";
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/wicket-box-calibration/accept${query}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(request)
    }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Wicket box calibration accept returned ${response.status}.`
    );
  }
  return response.json() as Promise<WicketBoxCalibrationAcceptResponse>;
}


export async function getReplayPayload(
  analysisId: string
): Promise<ReplayPayloadV1> {
  const response = await fetch(
    `${getApiBaseUrl()}/video-analysis/${encodeURIComponent(analysisId)}/replay-payload`,
    { cache: "no-store" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Replay payload lookup returned ${response.status}.`
    );
  }
  return response.json() as Promise<ReplayPayloadV1>;
}


export async function solveDeviceCalibration(
  video: Blob,
  deviceId: string,
  deviceLabel: string | null,
  spec: CheckerboardSpecInput
): Promise<DeviceCalibrationResponse> {
  const extension = video.type.includes("mp4") ? "mp4" : "webm";
  const formData = new FormData();
  formData.append(
    "video",
    new File([video], `calibration.${extension}`, { type: video.type || `video/${extension}` })
  );
  formData.append("device_id", deviceId);
  if (deviceLabel) formData.append("device_label", deviceLabel);
  formData.append("columns", String(spec.columns));
  formData.append("rows", String(spec.rows));
  formData.append("square_size_mm", String(spec.square_size_mm));
  const response = await fetch(`${getApiBaseUrl()}/device-calibration/solve`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Device calibration returned ${response.status}.`
    );
  }
  return response.json() as Promise<DeviceCalibrationResponse>;
}


/** Null means this device has no profile yet — not an error worth throwing on. */
export async function getDeviceCalibration(
  deviceId: string
): Promise<DeviceLensProfile | null> {
  const response = await fetch(
    `${getApiBaseUrl()}/device-calibration/${encodeURIComponent(deviceId)}`,
    { cache: "no-store" }
  );
  if (response.status === 404) return null;
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Device calibration lookup returned ${response.status}.`
    );
  }
  return response.json() as Promise<DeviceLensProfile>;
}


export async function deleteDeviceCalibration(deviceId: string): Promise<void> {
  const response = await fetch(
    `${getApiBaseUrl()}/device-calibration/${encodeURIComponent(deviceId)}`,
    { method: "DELETE" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(
      response,
      `Device calibration delete returned ${response.status}.`
    );
  }
}
