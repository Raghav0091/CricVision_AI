"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

import type {
  NormalizedBox,
  NormalizedPoint,
  PitchGeometry,
  WicketCalibration
} from "@/lib/api";

import { AnalysisMediaStage } from "./AnalysisMediaStage";


type WicketLabel = "striker" | "non_striker";
type ResizeCorner = "nw" | "ne" | "sw" | "se";
type InteractionMode = "guides" | "review" | "locked";
type PointerOperation = {
  pointerId: number;
  label: WicketLabel;
  mode: "move" | "resize";
  corner?: ResizeCorner;
  startX: number;
  startY: number;
  original: NormalizedBox;
};


const MIN_BOX_SIZE = 0.035;

// ponytail: far/near search defaults match typical side-on fixed-camera framing.
export const DEFAULT_VIDEO_GUIDES = {
  striker: { x: 0.34, y: 0.16, width: 0.32, height: 0.36 },
  non_striker: { x: 0.22, y: 0.48, width: 0.56, height: 0.48 }
};


export function wicketFromBox(
  label: WicketLabel,
  source: WicketCalibration["source"],
  confidence: number | null,
  box: NormalizedBox,
  detectionPass: WicketCalibration["detection_pass"] = null
): WicketCalibration {
  const centerX = clamp(box.x + box.width / 2);
  const bottomCenter = {
    x: centerX,
    y: clamp(box.y + box.height)
  };
  return {
    label,
    source,
    confidence,
    box,
    center: {
      x: centerX,
      y: clamp(box.y + box.height / 2)
    },
    bottom_center: bottomCenter,
    approximate_wicket_base_reference: bottomCenter,
    detection_pass: detectionPass ?? null
  };
}


export function calculateApproximatePitchGeometry(
  striker: WicketCalibration | null,
  nonStriker: WicketCalibration | null,
  widthMultiplier: number
): PitchGeometry | null {
  if (!striker || !nonStriker) return null;
  const ends = { striker, non_striker: nonStriker };
  const nearLabel: WicketLabel = nearScore(striker) >= nearScore(nonStriker)
    ? "striker"
    : "non_striker";
  const farLabel: WicketLabel = nearLabel === "striker" ? "non_striker" : "striker";
  const near = ends[nearLabel];
  const far = ends[farLabel];
  const dx = near.bottom_center.x - far.bottom_center.x;
  const dy = near.bottom_center.y - far.bottom_center.y;
  const length = Math.hypot(dx, dy);
  if (length < 0.000001) return null;
  const perpendicularX = -dy / length;
  const perpendicularY = dx / length;
  const rawNearHalfWidth = Math.max(near.box.width * 1.35, 0.02);
  const rawFarHalfWidth = Math.max(far.box.width * 1.35, 0.014);
  const nearHalfWidth = Math.max(rawNearHalfWidth, rawFarHalfWidth * 1.15) * widthMultiplier;
  const farHalfWidth = Math.min(
    rawFarHalfWidth * widthMultiplier,
    nearHalfWidth * 0.86
  );
  const corridor: PitchGeometry["corridor"] = [
    offsetPoint(near.bottom_center, perpendicularX, perpendicularY, nearHalfWidth),
    offsetPoint(far.bottom_center, perpendicularX, perpendicularY, farHalfWidth),
    offsetPoint(far.bottom_center, perpendicularX, perpendicularY, -farHalfWidth),
    offsetPoint(near.bottom_center, perpendicularX, perpendicularY, -nearHalfWidth)
  ];
  return {
    axis_start: striker.bottom_center,
    axis_end: nonStriker.bottom_center,
    corridor,
    near_end_label: nearLabel,
    far_end_label: farLabel,
    geometry_type: "approximate_2d",
    corridor_width_multiplier: widthMultiplier
  };
}


export function wicketDistanceWarning(
  striker: WicketCalibration | null,
  nonStriker: WicketCalibration | null
): string | null {
  if (!striker || !nonStriker) return null;
  const distance = Math.hypot(
    striker.bottom_center.x - nonStriker.bottom_center.x,
    striker.bottom_center.y - nonStriker.bottom_center.y
  );
  if (distance < 0.055 || intersectionOverUnion(striker.box, nonStriker.box) > 0.3) {
    return "The two wicket locations appear too close together. Check the calibration.";
  }
  return null;
}


export function CalibrationCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  striker,
  nonStriker,
  strikerGuide = null,
  nonStrikerGuide = null,
  pitchGeometry,
  interactionMode = "locked",
  showGuides,
  onGuideChange
}: {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  striker: WicketCalibration | null;
  nonStriker: WicketCalibration | null;
  strikerGuide?: NormalizedBox | null;
  nonStrikerGuide?: NormalizedBox | null;
  pitchGeometry: PitchGeometry | null;
  interactionMode?: InteractionMode;
  showGuides?: boolean;
  onGuideChange?: (label: WicketLabel, box: NormalizedBox) => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const operationRef = useRef<PointerOperation | null>(null);
  const guidesEditable = interactionMode === "guides" && Boolean(onGuideChange);
  const shouldShowGuides = showGuides ?? Boolean(strikerGuide || nonStrikerGuide);

  function startGuideOperation(
    event: ReactPointerEvent<HTMLElement>,
    label: WicketLabel,
    box: NormalizedBox,
    mode: PointerOperation["mode"],
    corner?: ResizeCorner
  ) {
    if (!guidesEditable) return;
    event.preventDefault();
    event.stopPropagation();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    operationRef.current = {
      pointerId: event.pointerId,
      label,
      mode,
      corner,
      startX: (event.clientX - rect.left) / rect.width,
      startY: (event.clientY - rect.top) / rect.height,
      original: { ...box }
    };
    wrapper.setPointerCapture(event.pointerId);
  }

  function moveOperation(event: ReactPointerEvent<HTMLDivElement>) {
    if (!guidesEditable || !onGuideChange) return;
    const operation = operationRef.current;
    const wrapper = wrapperRef.current;
    if (!operation || !wrapper || event.pointerId !== operation.pointerId) return;
    event.preventDefault();
    const rect = wrapper.getBoundingClientRect();
    const currentX = (event.clientX - rect.left) / rect.width;
    const currentY = (event.clientY - rect.top) / rect.height;
    const dx = currentX - operation.startX;
    const dy = currentY - operation.startY;
    const box = operation.mode === "move"
      ? moveBox(operation.original, dx, dy)
      : resizeBox(operation.original, operation.corner ?? "se", dx, dy);
    onGuideChange(operation.label, box);
  }

  function endOperation(event: ReactPointerEvent<HTMLDivElement>) {
    if (operationRef.current?.pointerId !== event.pointerId) return;
    operationRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function renderGuide(label: WicketLabel, box: NormalizedBox | null) {
    if (!box || !shouldShowGuides) return null;
    const title = label === "striker" ? "Striker Guide" : "Non-Striker Guide";
    const dimmed = interactionMode !== "guides";
    return (
      <div
        className="absolute rounded-md border-2 border-dashed border-signal"
        style={{
          left: `${box.x * 100}%`,
          top: `${box.y * 100}%`,
          width: `${box.width * 100}%`,
          height: `${box.height * 100}%`,
          opacity: dimmed ? 0.45 : 1,
          boxShadow: dimmed ? undefined : "0 0 18px rgba(255,85,79,0.32)",
          cursor: guidesEditable ? "move" : "default",
          touchAction: "none",
          pointerEvents: guidesEditable ? "auto" : "none"
        }}
        onPointerDown={guidesEditable
          ? (event) => startGuideOperation(event, label, box, "move")
          : undefined}
      >
        <span className="absolute -top-7 left-0 whitespace-nowrap rounded bg-signal px-2 py-1 text-[10px] font-black uppercase tracking-wide text-white">
          {title}
        </span>
        {guidesEditable && (["nw", "ne", "sw", "se"] as ResizeCorner[]).map((corner) => (
          <button
            key={corner}
            type="button"
            aria-label={`Resize ${title} from ${corner}`}
            className={`absolute h-4 w-4 rounded-sm border-2 border-ink bg-signal ${
              corner.includes("n") ? "-top-2" : "-bottom-2"
            } ${
              corner.includes("w") ? "-left-2" : "-right-2"
            }`}
            style={{
              cursor: `${corner}-resize`,
              touchAction: "none"
            }}
            onPointerDown={(event) => startGuideOperation(
              event,
              label,
              box,
              "resize",
              corner
            )}
          />
        ))}
      </div>
    );
  }

  // ponytail: overlays use normalized 0–1 coords on the aspect box; letterbox is outside.
  return (
    <AnalysisMediaStage
      aspectWidth={imageWidth}
      aspectHeight={imageHeight}
      expandable
      label="Guided scene calibration reference"
    >
      <div
        ref={wrapperRef}
        className="absolute inset-0 select-none"
        style={{ touchAction: guidesEditable ? "none" : "auto" }}
        onPointerMove={moveOperation}
        onPointerUp={endOperation}
        onPointerCancel={endOperation}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          className="absolute inset-0 h-full w-full object-contain"
          src={imageUrl}
          alt="Guided scene calibration reference"
          draggable={false}
        />
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          aria-label="Scene calibration overlay"
        >
          {pitchGeometry && (
            <>
              <polygon
                points={pitchGeometry.corridor.map((point) => `${point.x},${point.y}`).join(" ")}
                fill="rgba(226,183,72,0.16)"
                stroke="rgba(255,225,132,0.75)"
                strokeWidth="0.004"
              />
              <line
                x1={pitchGeometry.axis_start.x}
                y1={pitchGeometry.axis_start.y}
                x2={pitchGeometry.axis_end.x}
                y2={pitchGeometry.axis_end.y}
                stroke="rgba(255,255,255,0.85)"
                strokeWidth="0.004"
                strokeDasharray="0.012 0.008"
              />
            </>
          )}
          {[striker, nonStriker].map((wicket) => wicket && (
            <g key={`${wicket.label}-detection`}>
              <rect
                x={wicket.box.x}
                y={wicket.box.y}
                width={wicket.box.width}
                height={wicket.box.height}
                fill="rgba(183,243,75,0.08)"
                stroke="#b7f34b"
                strokeWidth="0.004"
                strokeDasharray="0.012 0.008"
                rx="0.006"
              />
              <text
                x={wicket.box.x + 0.008}
                y={Math.max(0.03, wicket.box.y - 0.012)}
                fill="#b7f34b"
                fontSize="0.028"
                fontWeight="700"
              >
                {wicket.label}
                {wicket.confidence != null ? ` ${(wicket.confidence * 100).toFixed(0)}%` : ""}
              </text>
              {virtualStumpsFromBox(wicket.box).map((segment) => (
                <line
                  key={segment.key}
                  x1={segment.x1}
                  y1={segment.y1}
                  x2={segment.x2}
                  y2={segment.y2}
                  stroke={segment.kind === "bail" ? "#ffdf7e" : "#f6cf62"}
                  strokeWidth={segment.kind === "bail" ? "0.005" : "0.006"}
                  strokeLinecap="round"
                />
              ))}
            </g>
          ))}
        </svg>
        {renderGuide("striker", strikerGuide)}
        {renderGuide("non_striker", nonStrikerGuide)}
        {(striker || nonStriker) && (
          <span className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-ink/85 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.1em] text-white/80">
            Guided scene · approximate 2D corridor
          </span>
        )}
      </div>
    </AnalysisMediaStage>
  );
}


function virtualStumpsFromBox(box: NormalizedBox): Array<{
  key: string;
  kind: "stump" | "bail";
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}> {
  // ponytail: mirror upload `_virtual_stumps_from_bbox` in normalized space.
  const topY = box.y + box.height * 0.08;
  const baseY = box.y + box.height;
  const stumpFractions = [
    ["left", 0.2],
    ["middle", 0.5],
    ["right", 0.8]
  ] as const;
  return [
    ...stumpFractions.map(([name, fraction]) => {
      const x = box.x + box.width * fraction;
      return {
        key: `stump-${name}`,
        kind: "stump" as const,
        x1: x,
        y1: topY,
        x2: x,
        y2: baseY
      };
    }),
    {
      key: "bail-left",
      kind: "bail" as const,
      x1: box.x + box.width * 0.14,
      y1: topY,
      x2: box.x + box.width * 0.5,
      y2: topY
    },
    {
      key: "bail-right",
      kind: "bail" as const,
      x1: box.x + box.width * 0.5,
      y1: topY,
      x2: box.x + box.width * 0.86,
      y2: topY
    }
  ];
}


function moveBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  return {
    ...box,
    x: clampBetween(box.x + dx, 0, 1 - box.width),
    y: clampBetween(box.y + dy, 0, 1 - box.height)
  };
}


function resizeBox(
  box: NormalizedBox,
  corner: ResizeCorner,
  dx: number,
  dy: number
): NormalizedBox {
  let left = box.x;
  let top = box.y;
  let right = box.x + box.width;
  let bottom = box.y + box.height;
  if (corner.includes("w")) left = clampBetween(left + dx, 0, right - MIN_BOX_SIZE);
  if (corner.includes("e")) right = clampBetween(right + dx, left + MIN_BOX_SIZE, 1);
  if (corner.includes("n")) top = clampBetween(top + dy, 0, bottom - MIN_BOX_SIZE);
  if (corner.includes("s")) bottom = clampBetween(bottom + dy, top + MIN_BOX_SIZE, 1);
  return {
    x: left,
    y: top,
    width: right - left,
    height: bottom - top
  };
}


function offsetPoint(
  point: NormalizedPoint,
  perpendicularX: number,
  perpendicularY: number,
  distance: number
): NormalizedPoint {
  return {
    x: clamp(point.x + perpendicularX * distance),
    y: clamp(point.y + perpendicularY * distance)
  };
}


function nearScore(wicket: WicketCalibration): number {
  return wicket.bottom_center.y + wicket.box.width * 0.35 + wicket.box.height * 0.1;
}


function intersectionOverUnion(first: NormalizedBox, second: NormalizedBox): number {
  const left = Math.max(first.x, second.x);
  const top = Math.max(first.y, second.y);
  const right = Math.min(first.x + first.width, second.x + second.width);
  const bottom = Math.min(first.y + first.height, second.y + second.height);
  const intersection = Math.max(0, right - left) * Math.max(0, bottom - top);
  if (intersection <= 0) return 0;
  const firstArea = first.width * first.height;
  const secondArea = second.width * second.height;
  return intersection / Math.max(firstArea + secondArea - intersection, 0.000001);
}


function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}


function clampBetween(value: number, minimum: number, maximum: number): number {
  return Math.max(minimum, Math.min(maximum, value));
}
