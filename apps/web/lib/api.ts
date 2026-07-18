import type { BoxLayout, CalibrationResponse, CapturedFrame } from "./types";


const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL
  ?? process.env.NEXT_PUBLIC_API_URL
  ?? "http://localhost:8000"
).replace(/\/$/, "");


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
  original_video_url: string;
  reference_frame_url: string;
  calibration_status?: "confirmed" | null;
  calibration_url?: string | null;
  calibration_overlay_url?: string | null;
  ball_detection_status?: "detection_queued" | "detecting_ball" | "detection_complete" | "detection_failed" | null;
  ball_detection_job_id?: string | null;
  ball_detection_started_at?: string | null;
  ball_detection_completed_at?: string | null;
  detection_summary_url?: string | null;
  detection_overlay_url?: string | null;
  message: string;
};


function withBrowserSafeAnalysisUrls(record: VideoAnalysisPreparedResponse): VideoAnalysisPreparedResponse {
  return {
    ...record,
    original_video_url: resolveApiUrl(record.original_video_url) ?? record.original_video_url,
    reference_frame_url: resolveApiUrl(record.reference_frame_url) ?? record.reference_frame_url,
    calibration_url: resolveApiUrl(record.calibration_url),
    calibration_overlay_url: resolveApiUrl(record.calibration_overlay_url),
    detection_summary_url: resolveApiUrl(record.detection_summary_url),
    detection_overlay_url: resolveApiUrl(record.detection_overlay_url)
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


export type WicketCandidate = {
  candidate_id: string;
  confidence: number;
  class_name: string;
  box: NormalizedBox;
  center: NormalizedPoint;
  bottom_center: NormalizedPoint;
};


export type WicketCalibration = {
  label: "striker" | "non_striker";
  source: "detected" | "adjusted" | "manual";
  confidence?: number | null;
  box: NormalizedBox;
  center: NormalizedPoint;
  bottom_center: NormalizedPoint;
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


export type VideoCalibrationDetectionResponse = {
  success: boolean;
  status: "candidates_ready" | "manual_required" | "stump_detector_missing" | "stump_detector_error";
  analysis_id: string;
  reference_frame_url: string;
  image_width: number;
  image_height: number;
  candidates: WicketCandidate[];
  provisional_striker_wicket?: WicketCalibration | null;
  provisional_non_striker_wicket?: WicketCalibration | null;
  pitch_geometry?: PitchGeometry | null;
  model_path_used: string;
  warning?: string | null;
  message: string;
};


export type VideoCalibrationConfirmationRequest = {
  analysis_id: string;
  striker_wicket: Pick<WicketCalibration, "label" | "source" | "confidence" | "box">;
  non_striker_wicket: Pick<WicketCalibration, "label" | "source" | "confidence" | "box">;
  corridor_width_multiplier: number;
  user_note?: string | null;
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
  image_width: number;
  image_height: number;
  model_path_used?: string | null;
  striker_wicket: WicketCalibration;
  non_striker_wicket: WicketCalibration;
  pitch_geometry: PitchGeometry;
  user_note?: string | null;
  message: string;
};


function withBrowserSafeDetectionUrls(
  result: VideoCalibrationDetectionResponse
): VideoCalibrationDetectionResponse {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url) ?? result.reference_frame_url
  };
}


function withBrowserSafeCalibrationUrls(
  result: ConfirmedVideoCalibrationResponse
): ConfirmedVideoCalibrationResponse {
  return {
    ...result,
    reference_frame_url: resolveApiUrl(result.reference_frame_url) ?? result.reference_frame_url,
    calibration_url: resolveApiUrl(result.calibration_url) ?? result.calibration_url,
    calibration_overlay_url: resolveApiUrl(result.calibration_overlay_url) ?? result.calibration_overlay_url
  };
}


export async function detectVideoAnalysisCalibration(
  analysisId: string
): Promise<VideoCalibrationDetectionResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/calibration/detect`,
    { method: "POST" }
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
  analysisId: string
): Promise<VideoBallDetectionStartResponse> {
  const response = await fetch(
    `${API_BASE_URL}/video-analysis/${encodeURIComponent(analysisId)}/ball-detection/start`,
    { method: "POST" }
  );
  if (!response.ok) {
    throw await videoAnalysisError(response, `Ball detection start returned ${response.status}.`);
  }
  return response.json() as Promise<VideoBallDetectionStartResponse>;
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
