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
