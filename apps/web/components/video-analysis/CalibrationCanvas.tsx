"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

import type {
  NormalizedBox,
  NormalizedPoint,
  PitchGeometry,
  WicketCalibration
} from "@/lib/api";


type WicketLabel = "striker" | "non_striker";
type ResizeCorner = "nw" | "ne" | "sw" | "se";
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


export function wicketFromBox(
  label: WicketLabel,
  source: WicketCalibration["source"],
  confidence: number | null,
  box: NormalizedBox
): WicketCalibration {
  const centerX = clamp(box.x + box.width / 2);
  return {
    label,
    source,
    confidence,
    box,
    center: {
      x: centerX,
      y: clamp(box.y + box.height / 2)
    },
    bottom_center: {
      x: centerX,
      y: clamp(box.y + box.height)
    }
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
  pitchGeometry,
  disabled = false,
  onWicketChange
}: {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  striker: WicketCalibration | null;
  nonStriker: WicketCalibration | null;
  pitchGeometry: PitchGeometry | null;
  disabled?: boolean;
  onWicketChange: (label: WicketLabel, wicket: WicketCalibration) => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const operationRef = useRef<PointerOperation | null>(null);

  function startOperation(
    event: ReactPointerEvent<HTMLElement>,
    label: WicketLabel,
    wicket: WicketCalibration,
    mode: PointerOperation["mode"],
    corner?: ResizeCorner
  ) {
    if (disabled) return;
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
      original: { ...wicket.box }
    };
    wrapper.setPointerCapture(event.pointerId);
  }

  function moveOperation(event: ReactPointerEvent<HTMLDivElement>) {
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
    const current = operation.label === "striker" ? striker : nonStriker;
    if (!current) return;
    const source = current.source === "detected" ? "adjusted" : current.source;
    onWicketChange(
      operation.label,
      wicketFromBox(operation.label, source, current.confidence ?? null, box)
    );
  }

  function endOperation(event: ReactPointerEvent<HTMLDivElement>) {
    if (operationRef.current?.pointerId !== event.pointerId) return;
    operationRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function renderWicket(
    wicket: WicketCalibration | null,
    label: WicketLabel
  ) {
    if (!wicket) return null;
    const strikerBox = label === "striker";
    const color = strikerBox ? "#ffca68" : "#50dcff";
    const title = strikerBox ? "Striker Wicket" : "Non-Striker Wicket";
    return (
      <div
        className="absolute border-2"
        style={{
          left: `${wicket.box.x * 100}%`,
          top: `${wicket.box.y * 100}%`,
          width: `${wicket.box.width * 100}%`,
          height: `${wicket.box.height * 100}%`,
          borderColor: color,
          boxShadow: `0 0 0 1px rgba(0,0,0,.7), 0 0 18px ${color}44`
        }}
      >
        <button
          type="button"
          disabled={disabled}
          aria-label={`Move ${title}`}
          className="absolute -top-7 left-0 max-w-[12rem] cursor-move whitespace-nowrap rounded-t-md bg-ink/90 px-2 py-1 text-[10px] font-black uppercase tracking-[0.08em] disabled:cursor-default"
          style={{ color, touchAction: "none" }}
          onPointerDown={(event) => startOperation(event, label, wicket, "move")}
        >
          {title}
          {wicket.confidence != null && ` ${(wicket.confidence * 100).toFixed(0)}%`}
        </button>
        {(["nw", "ne", "sw", "se"] as ResizeCorner[]).map((corner) => (
          <button
            key={corner}
            type="button"
            disabled={disabled}
            aria-label={`Resize ${title} from ${corner}`}
            className={`absolute h-4 w-4 rounded-sm border-2 border-ink disabled:cursor-default ${
              corner.includes("n") ? "-top-2" : "-bottom-2"
            } ${
              corner.includes("w") ? "-left-2" : "-right-2"
            }`}
            style={{
              backgroundColor: color,
              cursor: `${corner}-resize`,
              touchAction: "none"
            }}
            onPointerDown={(event) => startOperation(
              event,
              label,
              wicket,
              "resize",
              corner
            )}
          />
        ))}
      </div>
    );
  }

  return (
    <div
      ref={wrapperRef}
      className="relative w-full select-none overflow-hidden rounded-xl bg-black"
      style={{
        aspectRatio: `${imageWidth} / ${imageHeight}`,
        touchAction: "none"
      }}
      onPointerMove={moveOperation}
      onPointerUp={endOperation}
      onPointerCancel={endOperation}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        className="absolute inset-0 h-full w-full object-contain"
        src={imageUrl}
        alt="Scene calibration reference"
        draggable={false}
      />
      {pitchGeometry && (
        <svg
          className="pointer-events-none absolute inset-0 h-full w-full"
          viewBox="0 0 1 1"
          preserveAspectRatio="none"
          aria-label="Approximate pitch axis and corridor"
        >
          <polygon
            points={pitchGeometry.corridor.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="rgba(213,255,107,.16)"
            stroke="rgba(213,255,107,.9)"
            strokeWidth="0.004"
          />
          <line
            x1={pitchGeometry.axis_start.x}
            y1={pitchGeometry.axis_start.y}
            x2={pitchGeometry.axis_end.x}
            y2={pitchGeometry.axis_end.y}
            stroke="white"
            strokeWidth="0.004"
            strokeDasharray="0.012 0.008"
          />
          {[striker, nonStriker].map((wicket) => wicket && (
            <circle
              key={wicket.label}
              cx={wicket.bottom_center.x}
              cy={wicket.bottom_center.y}
              r="0.009"
              fill={wicket.label === "striker" ? "#ffca68" : "#50dcff"}
              stroke="#080c10"
              strokeWidth="0.003"
            />
          ))}
        </svg>
      )}
      {renderWicket(striker, "striker")}
      {renderWicket(nonStriker, "non_striker")}
      {pitchGeometry && (
        <span className="pointer-events-none absolute bottom-3 left-3 rounded-md bg-ink/85 px-2 py-1 text-[10px] font-bold uppercase tracking-[0.1em] text-white/80">
          Approximate pitch axis · 2D corridor
        </span>
      )}
    </div>
  );
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
