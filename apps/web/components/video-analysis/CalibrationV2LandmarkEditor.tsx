"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

import type {
  CalibrationLandmarkInput,
  WicketPoseLandmarkInput
} from "@/lib/api";


const BASE_COLOR = "#50dcff";
const TOP_COLOR = "#ffca68";


type EditableLandmark =
  | CalibrationLandmarkInput
  | WicketPoseLandmarkInput;


function pointType(landmark: EditableLandmark): "base" | "top" | "ground" {
  if ("point_type" in landmark) return landmark.point_type;
  return landmark.landmark_type === "ground_control" ? "ground" : "base";
}


function clamp(value: number): number {
  return Math.max(0, Math.min(1, value));
}


export function CalibrationV2LandmarkEditor({
  imageUrl,
  imageWidth,
  imageHeight,
  landmarks,
  disabled = false,
  showLabels = true,
  onLandmarkChange
}: {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  landmarks: EditableLandmark[];
  disabled?: boolean;
  showLabels?: boolean;
  onLandmarkChange: (
    landmarkId: string,
    normalizedX: number,
    normalizedY: number
  ) => void;
}) {
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const activePointerRef = useRef<{
    pointerId: number;
    landmarkId: string;
  } | null>(null);

  function updateFromPointer(
    event: ReactPointerEvent<HTMLDivElement>,
    landmarkId: string
  ) {
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    const rect = wrapper.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    onLandmarkChange(
      landmarkId,
      clamp((event.clientX - rect.left) / rect.width),
      clamp((event.clientY - rect.top) / rect.height)
    );
  }

  function startDrag(
    event: ReactPointerEvent<HTMLButtonElement>,
    landmarkId: string
  ) {
    if (disabled) return;
    event.preventDefault();
    event.stopPropagation();
    const wrapper = wrapperRef.current;
    if (!wrapper) return;
    activePointerRef.current = {
      pointerId: event.pointerId,
      landmarkId
    };
    wrapper.setPointerCapture(event.pointerId);
    const rect = wrapper.getBoundingClientRect();
    onLandmarkChange(
      landmarkId,
      clamp((event.clientX - rect.left) / rect.width),
      clamp((event.clientY - rect.top) / rect.height)
    );
  }

  function moveDrag(event: ReactPointerEvent<HTMLDivElement>) {
    const active = activePointerRef.current;
    if (!active || event.pointerId !== active.pointerId) return;
    event.preventDefault();
    updateFromPointer(event, active.landmarkId);
  }

  function endDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (activePointerRef.current?.pointerId !== event.pointerId) return;
    activePointerRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  const primary = landmarks.filter((landmark) => pointType(landmark) !== "ground");

  return (
    <div
      ref={wrapperRef}
      className="relative w-full select-none overflow-hidden rounded-xl bg-black"
      style={{
        aspectRatio: `${imageWidth} / ${imageHeight}`,
        touchAction: "none"
      }}
      onPointerMove={moveDrag}
      onPointerUp={endDrag}
      onPointerCancel={endDrag}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        className="pointer-events-none absolute inset-0 h-full w-full object-fill"
        src={imageUrl}
        alt="Calibration v2 semantic wicket landmark editor"
        draggable={false}
      />

      <svg
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {(["bowler", "striker"] as const).flatMap((wicketEnd) => (
          (["base", "top"] as const).map((kind) => {
            const points = primary
              .filter((landmark) => (
                landmark.wicket_end === wicketEnd
                && pointType(landmark) === kind
              ))
              .map((landmark) => (
                `${landmark.normalized_x * 100},${landmark.normalized_y * 100}`
              ))
              .join(" ");
            return points ? (
              <polyline
                key={`${wicketEnd}-${kind}`}
                points={points}
                fill="none"
                stroke={kind === "base" ? BASE_COLOR : TOP_COLOR}
                strokeWidth="0.35"
                strokeDasharray="1 0.7"
                vectorEffect="non-scaling-stroke"
              />
            ) : null;
          })
        ))}
        {primary
          .filter((landmark): landmark is WicketPoseLandmarkInput => (
            "point_type" in landmark && landmark.point_type === "base"
          ))
          .map((base) => {
            const top = primary.find((candidate) => (
              "point_type" in candidate
              && candidate.point_type === "top"
              && candidate.wicket_end === base.wicket_end
              && candidate.stump_position === base.stump_position
            ));
            return top ? (
              <line
                key={`${base.id}-vertical`}
                x1={base.normalized_x * 100}
                y1={base.normalized_y * 100}
                x2={top.normalized_x * 100}
                y2={top.normalized_y * 100}
                stroke="#ffffff"
                strokeOpacity="0.65"
                strokeWidth="0.25"
                strokeDasharray="0.7 0.6"
                vectorEffect="non-scaling-stroke"
              />
            ) : null;
          })}
      </svg>

      {landmarks.map((landmark) => {
        const kind = pointType(landmark);
        const groundReference = kind === "ground";
        const color = groundReference
          ? "#ff5ebe"
          : kind === "top"
            ? TOP_COLOR
            : BASE_COLOR;
        const sideLabel = landmark.id.includes("_left_")
          ? "L"
          : landmark.id.includes("_middle_")
            ? "M"
            : "R";
        const shortLabel = groundReference
          ? `${landmark.id.startsWith("bowler_") ? "B" : "S"}${sideLabel}`
          : "point_type" in landmark
            ? `${kind === "top" ? "T" : "B"}-${sideLabel}`
            : sideLabel;
        const unavailable = (
          "visibility" in landmark
          && ["occluded", "unavailable"].includes(landmark.visibility)
        );
        return (
          <button
            key={landmark.id}
            type="button"
            disabled={disabled || unavailable}
            aria-label={`Move ${landmark.label}`}
            data-landmark-kind={kind}
            className={`group absolute h-8 w-8 -translate-x-1/2 -translate-y-1/2 cursor-grab border-2 border-black/80 text-[10px] font-black text-black shadow-[0_0_0_2px_rgba(255,255,255,.75)] active:cursor-grabbing disabled:cursor-default disabled:opacity-60 ${
              groundReference || kind === "top" ? "rounded-md" : "rounded-full"
            }`}
            style={{
              left: `${landmark.normalized_x * 100}%`,
              top: `${landmark.normalized_y * 100}%`,
              backgroundColor: color,
              opacity: unavailable ? 0.35 : 1,
              touchAction: "none"
            }}
            onPointerDown={(event) => startDrag(event, landmark.id)}
          >
            {shortLabel}
            {showLabels && (
              <span
                className="pointer-events-none absolute left-1/2 top-[-1.8rem] -translate-x-1/2 whitespace-nowrap rounded bg-black/90 px-2 py-1 text-[9px] uppercase tracking-[0.08em]"
                style={{ color }}
              >
                {groundReference ? "metric ground" : landmark.wicket_end}{" "}
                {shortLabel}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
