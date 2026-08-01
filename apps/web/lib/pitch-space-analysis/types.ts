import type { VirtualPitchModel } from "@/lib/virtual-pitch/types";

export type AnalysisStatus =
  | "NO_VIDEO"
  | "UPLOAD_FAILED"
  | "FRAME_ZERO_UNUSABLE"
  | "INSUFFICIENT_WICKETS"
  | "PITCH_FIT_FAILED"
  | "UNSTABLE_CAMERA"
  | "BALL_TRACK_UNAVAILABLE"
  | "BOUNCE_UNAVAILABLE"
  | "SPEED_UNAVAILABLE"
  | "MOVEMENT_UNAVAILABLE"
  | "PROCESSING"
  | "COMPLETE"
  | "PARTIAL";

export type TrackProvenance = "OBSERVED" | "RECOVERED" | "PROJECTED" | string;

export type PitchSpaceTrackPoint = {
  frame_index: number;
  timestamp_seconds: number;
  image_x_px: number;
  image_y_px: number;
  pitch_x_m: number;
  pitch_y_m: number;
  detection_confidence?: number | null;
  pitch_fit_confidence?: number | null;
  combined_confidence?: number | null;
  provenance: TrackProvenance;
  in_pitch_bounds?: boolean;
  bounce_phase?: string | null;
  warnings?: string[];
};

export type ConfidenceMetric = {
  confidence?: number | null;
  warnings?: string[];
};

export type PitchSpaceAnalysis = {
  version: string;
  analysis_id: string;
  status: AnalysisStatus | string;
  source_video?: { original_filename?: string; video_url?: string | null } | null;
  source_video_url?: string | null;
  source_filename?: string | null;
  native_width?: number | null;
  native_height?: number | null;
  fps?: number | null;
  frame_count?: number | null;
  setup_frame_decision?: {
    preferred_frame_attempted?: boolean;
    preferred_frame_index?: number;
    preferred_frame_passed?: boolean;
    selected_frame_index?: number | null;
    selected_timestamp_seconds?: number | null;
    fallback_used?: boolean;
    fallback_candidates?: unknown[];
    selection_reasons?: string[];
    quality_score?: number | null;
  } | null;
  stable_near_wicket?: Record<string, unknown> | null;
  stable_far_wicket?: Record<string, unknown> | null;
  pitch_fit?: {
    status?: string;
    confidence?: number | null;
    fit_score?: number | null;
    selected_hypothesis?: string | null;
    condition_number?: number | null;
    warnings?: string[];
    correspondences?: unknown[];
    projected_pitch?: Array<{
      primitive_id?: string;
      primitive_type?: "LINE" | "POLYGON" | "WICKET_BASE" | string;
      image_points?: Array<{ x: number; y: number }>;
    }>;
  } | null;
  camera_stability?: {
    status?: "FIXED_CAMERA" | "MINOR_DRIFT" | "UNSTABLE_CAMERA" | string;
    confidence?: number | null;
    drift_frame?: number | null;
    warnings?: string[];
  } | null;
  image_space_track?: PitchSpaceTrackPoint[];
  pitch_space_track?: PitchSpaceTrackPoint[];
  bounce?: ConfidenceMetric & {
    bounce_frame?: number | null;
    bounce_timestamp_seconds?: number | null;
    pitch_x_m?: number | null;
    pitch_y_m?: number | null;
    evidence?: unknown;
    alternative_candidates?: unknown[];
  } | null;
  line?: ConfidenceMetric & {
    line?: string | null;
    length?: string | null;
    lateral_offset_from_middle_m?: number | null;
    distance_from_striker_wicket_m?: number | null;
    distance_from_striker_popping_crease_m?: number | null;
  } | null;
  length?: ConfidenceMetric & {
    line?: string | null;
    length?: string | null;
    lateral_offset_from_middle_m?: number | null;
    distance_from_striker_wicket_m?: number | null;
    distance_from_striker_popping_crease_m?: number | null;
  } | null;
  estimated_planar_speed?: ConfidenceMetric & {
    speed_mps?: number | null;
    speed_kmh?: number | null;
    method?: string | null;
    frames_used?: number[] | number | null;
    confidence_score?: number | null;
  } | null;
  estimated_lateral_movement?: ConfidenceMetric & {
    direction?: string | null;
    movement_m?: number | null;
    method?: string | null;
    frames_used?: number[] | number | null;
    confidence_score?: number | null;
  } | null;
  overall_confidence?: number | null;
  warnings?: string[];
  unavailable_metrics?: string[];
  stage_timings?: Record<string, number | null>;
  diagnostics?: Record<string, unknown> | null;
  overlay_url?: string | null;
};

export type RecentAnalysis = {
  analysis_id: string;
  original_filename?: string | null;
  status?: string | null;
  updated_at?: string | null;
  source_filename?: string | null;
};

export type PitchSpaceLabData = {
  analysis: PitchSpaceAnalysis;
  pitch: VirtualPitchModel;
  videoUrl: string;
};
