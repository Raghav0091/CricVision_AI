"use client";

import { memo, useMemo } from "react";

import type { BallReviewCandidate, BallReviewDisplayToggles } from "@/lib/ball-analysis-review";

function isReconstructedProvenance(provenance: string, source?: string) {
  const normalized = provenance.toUpperCase();
  return normalized.includes("RECOVERED")
    || normalized.includes("RECONSTRUCTED")
    || normalized.includes("PROJECTED")
    || source === "predicted"
    || source === "recovered";
}

export type BallTrackProvenance =
  | "OBSERVED"
  | "TRACKER_RECOVERED"
  | "RECOVERED"
  | "PHYSICS_RECONSTRUCTED"
  | "RECONSTRUCTED"
  | "PROJECTED";

export type NativePixelBoundingBox =
  | readonly [x1: number, y1: number, x2: number, y2: number]
  | {
      x_min: number;
      y_min: number;
      x_max: number;
      y_max: number;
    };

type TrackPointCoordinates =
  | {
      image_x_px: number;
      image_y_px: number;
      x?: number;
      y?: number;
    }
  | {
      image_x_px?: number | null;
      image_y_px?: number | null;
      x: number;
      y: number;
    };

export type BallTrackOverlayPoint = TrackPointCoordinates & {
  frame_index: number;
  timestamp_seconds: number;
  provenance: BallTrackProvenance | string;
  source?: string;
  bounding_box?: NativePixelBoundingBox | null;
  detector_confidence?: number | null;
  tracking_confidence?: number | null;
  valid?: boolean;
};

export type BallTrackOverlayProps = {
  points: readonly BallTrackOverlayPoint[];
  candidates?: readonly BallReviewCandidate[];
  toggles?: BallReviewDisplayToggles;
  currentTimeSeconds: number;
  currentFrame?: number | null;
  nativeWidth: number;
  nativeHeight: number;
  showCompleteTrail?: boolean;
};

type RenderPoint = BallTrackOverlayPoint & {
  pixelX: number;
  pixelY: number;
};

const PROVENANCE_STYLES = {
  observed: { color: "#50e650", dash: undefined },
  recovered: { color: "#ff9600", dash: "7 5" },
  projected: { color: "#ffe600", dash: "3 5" },
} as const;

function pointStyle(provenance: string) {
  const normalized = provenance.toUpperCase();
  if (normalized === "OBSERVED") return PROVENANCE_STYLES.observed;
  if (normalized.includes("RECOVERED") || normalized.includes("RECONSTRUCTED")) {
    return PROVENANCE_STYLES.recovered;
  }
  return PROVENANCE_STYLES.projected;
}

function pixelCoordinates(point: BallTrackOverlayPoint): [number, number] | null {
  const x = point.image_x_px ?? point.x;
  const y = point.image_y_px ?? point.y;
  return typeof x === "number" && Number.isFinite(x)
    && typeof y === "number" && Number.isFinite(y)
    ? [x, y]
    : null;
}

function normalizedBox(box: NativePixelBoundingBox | null | undefined) {
  if (!box) return null;
  const values = "x_min" in box
    ? [box.x_min, box.y_min, box.x_max, box.y_max]
    : box;
  if (values.some((value) => !Number.isFinite(value))) return null;
  const [x1, y1, x2, y2] = values;
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  };
}

function candidateBox(candidate: BallReviewCandidate) {
  const [x1, y1, x2, y2] = candidate.bbox_xyxy;
  return {
    x: Math.min(x1, x2),
    y: Math.min(y1, y2),
    width: Math.abs(x2 - x1),
    height: Math.abs(y2 - y1),
  };
}

function isReached(
  frameIndex: number,
  currentFrame: number | null | undefined,
  currentTimeSeconds: number,
  timestampSeconds: number,
) {
  if (currentFrame !== null && currentFrame !== undefined && Number.isFinite(currentFrame)) {
    return frameIndex <= currentFrame;
  }
  return timestampSeconds <= currentTimeSeconds;
}

export const BallTrackOverlay = memo(function BallTrackOverlay({
  points,
  candidates = [],
  toggles,
  currentTimeSeconds,
  currentFrame,
  nativeWidth,
  nativeHeight,
  showCompleteTrail = false,
}: BallTrackOverlayProps) {
  const display = toggles ?? {
    primaryTrack: true,
    acceptedCandidates: false,
    rejectedCandidates: false,
    detectionBoxes: false,
    reconstructedPoints: true,
    completeTrail: false,
  };
  const trailEnabled = showCompleteTrail || display.completeTrail;

  const orderedPoints = useMemo<RenderPoint[]>(() => points
    .filter((point) => point.valid !== false)
    .flatMap((point) => {
      const coordinates = pixelCoordinates(point);
      return coordinates ? [{ ...point, pixelX: coordinates[0], pixelY: coordinates[1] }] : [];
    })
    .sort((left, right) => left.frame_index - right.frame_index), [points]);

  const visibleTrackPoints = useMemo(() => orderedPoints.filter((point) => {
    if (!display.primaryTrack) return false;
    if (!display.reconstructedPoints && isReconstructedProvenance(point.provenance, point.source)) {
      return false;
    }
    return true;
  }), [display.primaryTrack, display.reconstructedPoints, orderedPoints]);

  const reachedPoints = useMemo(
    () => visibleTrackPoints.filter((point) => isReached(
      point.frame_index,
      currentFrame,
      currentTimeSeconds,
      point.timestamp_seconds
    )),
    [currentFrame, currentTimeSeconds, visibleTrackPoints],
  );
  const trailPoints = trailEnabled ? visibleTrackPoints : reachedPoints;
  const activePoint = reachedPoints.at(-1) ?? null;
  const activeBox = useMemo(
    () => normalizedBox(activePoint?.bounding_box),
    [activePoint],
  );

  const frameCandidates = useMemo(
    () => candidates.filter((candidate) => candidate.frame_index === currentFrame),
    [candidates, currentFrame],
  );

  if (!(nativeWidth > 0) || !(nativeHeight > 0)) return null;

  const markerRadius = Math.max(5, Math.min(nativeWidth, nativeHeight) * 0.007);
  const strokeWidth = Math.max(2, Math.min(nativeWidth, nativeHeight) * 0.003);

  return (
    <svg
      aria-label="Synchronized ball analysis overlay"
      viewBox={`0 0 ${nativeWidth} ${nativeHeight}`}
      preserveAspectRatio="xMidYMid meet"
      className="pointer-events-none absolute inset-0 z-20 h-full w-full"
    >
      {trailPoints.slice(1).map((point, index) => {
        const previous = trailPoints[index];
        const style = pointStyle(point.provenance);
        return (
          <line
            key={`${previous.frame_index}-${point.frame_index}-${index}`}
            x1={previous.pixelX}
            y1={previous.pixelY}
            x2={point.pixelX}
            y2={point.pixelY}
            stroke={style.color}
            strokeWidth={strokeWidth}
            strokeDasharray={style.dash}
            strokeLinecap="round"
            opacity="0.82"
          />
        );
      })}

      {frameCandidates.flatMap((candidate) => {
        const box = candidateBox(candidate);
        const showAccepted = display.acceptedCandidates && candidate.selected;
        const showRejected = display.rejectedCandidates && !candidate.selected;
        if (!showAccepted && !showRejected && !display.detectionBoxes) {
          return [];
        }
        const stroke = candidate.selected ? "#50e650" : "#ff8a7a";
        const opacity = candidate.selected ? 0.85 : 0.55;
        return [(
          <rect
            key={`candidate-box-${candidate.candidate_id}`}
            x={box.x}
            y={box.y}
            width={box.width}
            height={box.height}
            fill="none"
            stroke={stroke}
            strokeWidth={Math.max(1.5, strokeWidth * 0.7)}
            opacity={opacity}
          />
        ), display.detectionBoxes ? (
          <circle
            key={`candidate-center-${candidate.candidate_id}`}
            cx={candidate.center.x}
            cy={candidate.center.y}
            r={markerRadius * 0.65}
            fill={stroke}
            opacity={opacity * 0.8}
          />
        ) : null].filter(Boolean);
      })}

      {display.detectionBoxes && activeBox && activePoint && (
        <rect
          x={activeBox.x}
          y={activeBox.y}
          width={activeBox.width}
          height={activeBox.height}
          fill="none"
          stroke={pointStyle(activePoint.provenance).color}
          strokeWidth={Math.max(1.5, strokeWidth * 0.65)}
          strokeDasharray="5 4"
          opacity="0.72"
        />
      )}

      {display.primaryTrack && activePoint && (
        <circle
          cx={activePoint.pixelX}
          cy={activePoint.pixelY}
          r={markerRadius}
          fill={pointStyle(activePoint.provenance).color}
          stroke="#080b09"
          strokeWidth={strokeWidth}
        />
      )}
    </svg>
  );
});
