"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

import type { KeypointLayout, NormalizedPoint, StumpKeypointName } from "@/lib/types";


type End = keyof KeypointLayout;

type Operation = {
  pointerId: number;
  end: End;
  point: StumpKeypointName;
};

const POINT_ORDER: StumpKeypointName[] = ["left_top", "middle_top", "right_top", "left_base", "middle_base", "right_base"];
const STUMP_PAIRS: Array<[StumpKeypointName, StumpKeypointName]> = [
  ["left_top", "left_base"],
  ["middle_top", "middle_base"],
  ["right_top", "right_base"]
];
const END_COLORS: Record<End, string> = { striker: "#ffd35f", non_striker: "#ff554f" };
const END_LABELS: Record<End, string> = { striker: "Far end (striker)", non_striker: "Near end (non-striker)" };

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function isTop(point: StumpKeypointName): boolean {
  return point.endsWith("_top");
}

export function seedKeypointsFromBox(box: { x: number; y: number; width: number; height: number }): Record<StumpKeypointName, NormalizedPoint> {
  const topY = box.y + box.height * 0.08;
  const baseY = box.y + box.height;
  return {
    left_top: { x: box.x + box.width * 0.2, y: topY },
    middle_top: { x: box.x + box.width * 0.5, y: topY },
    right_top: { x: box.x + box.width * 0.8, y: topY },
    left_base: { x: box.x + box.width * 0.2, y: baseY },
    middle_base: { x: box.x + box.width * 0.5, y: baseY },
    right_base: { x: box.x + box.width * 0.8, y: baseY }
  };
}

export function DraggableStumpKeypoints({
  layout,
  onChange,
  editable = true
}: {
  layout: KeypointLayout;
  onChange: (next: KeypointLayout) => void;
  editable?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const operationRef = useRef<Operation | null>(null);

  function startOperation(event: ReactPointerEvent<HTMLElement>, end: End, point: StumpKeypointName) {
    if (!editable) return;
    event.preventDefault();
    event.stopPropagation();
    const container = containerRef.current;
    if (!container) return;
    operationRef.current = { pointerId: event.pointerId, end, point };
    container.setPointerCapture(event.pointerId);
  }

  function moveOperation(event: ReactPointerEvent<HTMLDivElement>) {
    const operation = operationRef.current;
    const container = containerRef.current;
    if (!operation || !container || event.pointerId !== operation.pointerId) return;
    event.preventDefault();
    const rect = container.getBoundingClientRect();
    const x = clamp01((event.clientX - rect.left) / rect.width);
    const y = clamp01((event.clientY - rect.top) / rect.height);
    const nextEnd = { ...layout[operation.end], [operation.point]: { x, y } };
    onChange({ ...layout, [operation.end]: nextEnd });
  }

  function endOperation(event: ReactPointerEvent<HTMLDivElement>) {
    const operation = operationRef.current;
    if (!operation || event.pointerId !== operation.pointerId) return;
    containerRef.current?.releasePointerCapture(event.pointerId);
    operationRef.current = null;
  }

  return (
    <div
      ref={containerRef}
      className="absolute inset-0"
      style={{ touchAction: editable ? "none" : "auto" }}
      onPointerMove={moveOperation}
      onPointerUp={endOperation}
      onPointerCancel={endOperation}
    >
      <svg className="absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none" aria-hidden>
        {(["striker", "non_striker"] as End[]).map((end) =>
          STUMP_PAIRS.map(([top, base]) => {
            const a = layout[end][top];
            const b = layout[end][base];
            return (
              <line
                key={`${end}-${top}-${base}`}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke={END_COLORS[end]}
                strokeWidth={0.004}
                vectorEffect="non-scaling-stroke"
                opacity={0.85}
              />
            );
          })
        )}
      </svg>
      {(["striker", "non_striker"] as End[]).map((end) => (
        <span
          key={`${end}-label`}
          className="absolute -translate-x-1/2 -translate-y-full whitespace-nowrap rounded px-2 py-0.5 text-[10px] font-black uppercase tracking-wide text-black"
          style={{
            left: `${layout[end].middle_top.x * 100}%`,
            top: `${Math.max(0, layout[end].middle_top.y * 100 - 2)}%`,
            backgroundColor: END_COLORS[end],
            pointerEvents: "none"
          }}
        >
          {END_LABELS[end]}
        </span>
      ))}
      {(["striker", "non_striker"] as End[]).flatMap((end) =>
        POINT_ORDER.map((point) => {
          const position = layout[end][point];
          const top = isTop(point);
          return (
            <button
              key={`${end}-${point}`}
              type="button"
              aria-label={`${END_LABELS[end]} ${point.replace("_", " ")}`}
              className="absolute -translate-x-1/2 -translate-y-1/2 rounded-full border-2 shadow-lg"
              style={{
                left: `${position.x * 100}%`,
                top: `${position.y * 100}%`,
                width: 22,
                height: 22,
                borderColor: "rgba(0,0,0,0.6)",
                backgroundColor: top ? "#8fd3ff" : "#ff6b62",
                touchAction: "none",
                cursor: editable ? "grab" : "default",
                pointerEvents: editable ? "auto" : "none"
              }}
              onPointerDown={(event) => startOperation(event, end, point)}
            />
          );
        })
      )}
    </div>
  );
}
