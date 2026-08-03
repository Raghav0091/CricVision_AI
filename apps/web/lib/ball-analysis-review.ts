import type {
  TrackingCandidateDiagnostic,
  VideoBallDetectionCandidate,
  VideoBallDetectionFrame,
  VideoBallTrackingPoint
} from "@/lib/api";

export type BallReviewDisplayToggles = {
  primaryTrack: boolean;
  acceptedCandidates: boolean;
  rejectedCandidates: boolean;
  detectionBoxes: boolean;
  reconstructedPoints: boolean;
  completeTrail: boolean;
};

export const DEFAULT_BALL_REVIEW_TOGGLES: BallReviewDisplayToggles = {
  primaryTrack: true,
  acceptedCandidates: false,
  rejectedCandidates: false,
  detectionBoxes: false,
  reconstructedPoints: true,
  completeTrail: false
};

export type BallReviewCandidate = VideoBallDetectionCandidate & {
  frame_index: number;
  timestamp_seconds: number;
  selected: boolean;
  rejection_label: string;
  static_likelihood: number;
  motion_score: number | null;
  temporal_score: number | null;
  track_compatibility: number | null;
};

export function mapSelectionReason(
  diagnostic: TrackingCandidateDiagnostic,
  detection?: VideoBallDetectionCandidate | null
): string {
  if (diagnostic.selected) {
    return "accepted";
  }
  const reason = diagnostic.selection_reason.toLowerCase();
  if (reason.includes("stationary")) {
    return "static candidate";
  }
  const components = diagnostic.score_components;
  if (components) {
    if (components.jump_penalty >= 0.35) {
      return "excessive jump";
    }
    if (components.static_penalty >= 0.5 || diagnostic.static_likelihood >= 0.6) {
      return "static candidate";
    }
    if (components.corridor < -0.02 || detection?.inside_pitch_corridor === false) {
      return "outside expected region";
    }
    if (components.motion < 0.05) {
      return "insufficient movement";
    }
    if (components.detector_confidence < 0.12) {
      return "weak confidence";
    }
    if (components.prediction_proximity < 0.08) {
      return "failed track linking";
    }
    if (components.direction < 0.05) {
      return "temporal inconsistency";
    }
  }
  if (reason.includes("coherent track")) {
    return "not selected for primary track";
  }
  return "not selected for primary track";
}

export function buildReviewCandidates(
  frames: VideoBallDetectionFrame[] | null | undefined,
  diagnostics: TrackingCandidateDiagnostic[] | null | undefined
): BallReviewCandidate[] {
  if (!frames?.length || !diagnostics?.length) {
    return [];
  }
  const detectionById = new Map<string, VideoBallDetectionCandidate & {
    frame_index: number;
    timestamp_seconds: number;
  }>();
  for (const frame of frames) {
    for (const detection of frame.detections) {
      detectionById.set(detection.candidate_id, {
        ...detection,
        frame_index: frame.frame_index,
        timestamp_seconds: frame.timestamp_seconds
      });
    }
  }
  return diagnostics.flatMap((diagnostic) => {
    const detection = detectionById.get(diagnostic.candidate_id);
    if (!detection) {
      return [];
    }
    const components = diagnostic.score_components;
    return [{
      ...detection,
      frame_index: diagnostic.frame_index,
      timestamp_seconds: detection.timestamp_seconds,
      selected: diagnostic.selected,
      rejection_label: mapSelectionReason(diagnostic, detection),
      static_likelihood: diagnostic.static_likelihood,
      motion_score: components?.motion ?? null,
      temporal_score: components?.direction ?? null,
      track_compatibility: components?.prediction_proximity ?? null
    }];
  });
}

export function rejectionReasonCounts(candidates: BallReviewCandidate[]) {
  const counts = new Map<string, number>();
  for (const candidate of candidates) {
    if (candidate.selected) continue;
    counts.set(candidate.rejection_label, (counts.get(candidate.rejection_label) ?? 0) + 1);
  }
  return counts;
}

export function activeFrameCandidates(
  candidates: BallReviewCandidate[],
  frameIndex: number
) {
  return candidates.filter((candidate) => candidate.frame_index === frameIndex);
}

export function activeFramePrimaryPoint(
  track: VideoBallTrackingPoint[],
  frameIndex: number
) {
  return track.find((point) => point.frame_index === frameIndex) ?? null;
}

export function isReconstructedPoint(point: VideoBallTrackingPoint) {
  const provenance = point.provenance.toUpperCase();
  return provenance.includes("RECOVERED")
    || provenance.includes("RECONSTRUCTED")
    || provenance.includes("PROJECTED")
    || point.source === "predicted"
    || point.source === "recovered";
}

export function trackProvenanceLabel(point: VideoBallTrackingPoint) {
  const provenance = point.provenance.toUpperCase();
  if (provenance === "OBSERVED") return "observed";
  if (provenance.includes("RECOVERED")) return "reconstructed";
  if (provenance.includes("RECONSTRUCTED")) return "reconstructed";
  if (provenance.includes("PROJECTED")) return "projected";
  return point.source;
}
