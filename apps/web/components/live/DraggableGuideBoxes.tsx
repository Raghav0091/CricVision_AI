"use client";

import { useRef, type PointerEvent as ReactPointerEvent } from "react";

import type { BoxLayout, NormalizedBox } from "@/lib/types";


type End = keyof BoxLayout;
type Corner = "nw" | "ne" | "sw" | "se";

type Operation = {
  pointerId: number;
  end: End;
  mode: "move" | "resize";
  corner?: Corner;
  startX: number;
  startY: number;
  original: NormalizedBox;
};

const MIN_SIZE = 0.04;
const COLORS: Record<End, string> = { striker: "#ffd35f", non_striker: "#ff554f" };
const LABELS: Record<End, string> = { striker: "Far end", non_striker: "Near end" };

function clamp(value: number, min = 0, max = 1): number {
  return Math.max(min, Math.min(max, value));
}

function moveBox(box: NormalizedBox, dx: number, dy: number): NormalizedBox {
  return { ...box, x: clamp(box.x + dx, 0, 1 - box.width), y: clamp(box.y + dy, 0, 1 - box.height) };
}

function resizeBox(box: NormalizedBox, corner: Corner, dx: number, dy: number): NormalizedBox {
  let { x, y, width, height } = box;
  if (corner.includes("e")) width = clamp(width + dx, MIN_SIZE, 1 - x);
  if (corner.includes("w")) {
    const nextX = clamp(x + dx, 0, x + width - MIN_SIZE);
    width = clamp(width - (nextX - x), MIN_SIZE, 1);
    x = nextX;
  }
  if (corner.includes("s")) height = clamp(height + dy, MIN_SIZE, 1 - y);
  if (corner.includes("n")) {
    const nextY = clamp(y + dy, 0, y + height - MIN_SIZE);
    height = clamp(height - (nextY - y), MIN_SIZE, 1);
    y = nextY;
  }
  return { x, y, width, height };
}

export function DraggableGuideBoxes({
  layout,
  onChange,
  editable = true
}: {
  layout: BoxLayout;
  onChange: (next: BoxLayout) => void;
  editable?: boolean;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const operationRef = useRef<Operation | null>(null);

  function startOperation(event: ReactPointerEvent<HTMLElement>, end: End, mode: "move" | "resize", corner?: Corner) {
    if (!editable) return;
    event.preventDefault();
    event.stopPropagation();
    const container = containerRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    operationRef.current = {
      pointerId: event.pointerId,
      end,
      mode,
      corner,
      startX: (event.clientX - rect.left) / rect.width,
      startY: (event.clientY - rect.top) / rect.height,
      original: { ...layout[end] }
    };
    container.setPointerCapture(event.pointerId);
  }

  function moveOperation(event: ReactPointerEvent<HTMLDivElement>) {
    const operation = operationRef.current;
    const container = containerRef.current;
    if (!operation || !container || event.pointerId !== operation.pointerId) return;
    event.preventDefault();
    const rect = container.getBoundingClientRect();
    const currentX = (event.clientX - rect.left) / rect.width;
    const currentY = (event.clientY - rect.top) / rect.height;
    const dx = currentX - operation.startX;
    const dy = currentY - operation.startY;
    const next = operation.mode === "move"
      ? moveBox(operation.original, dx, dy)
      : resizeBox(operation.original, operation.corner ?? "se", dx, dy);
    onChange({ ...layout, [operation.end]: next });
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
      {(["striker", "non_striker"] as End[]).map((end) => {
        const box = layout[end];
        const colour = COLORS[end];
        return (
          <div
            key={end}
            className="absolute rounded-md border-2 border-dashed"
            style={{
              left: `${box.x * 100}%`,
              top: `${box.y * 100}%`,
              width: `${box.width * 100}%`,
              height: `${box.height * 100}%`,
              borderColor: colour,
              boxShadow: `0 0 16px ${colour}55`,
              cursor: editable ? "move" : "default",
              touchAction: "none",
              pointerEvents: editable ? "auto" : "none"
            }}
            onPointerDown={(event) => startOperation(event, end, "move")}
          >
            <span
              className="absolute -top-6 left-0 whitespace-nowrap rounded px-2 py-0.5 text-[10px] font-black uppercase tracking-wide text-black"
              style={{ backgroundColor: colour, pointerEvents: "none" }}
            >
              {LABELS[end]}
            </span>
            {editable && (["nw", "ne", "sw", "se"] as Corner[]).map((corner) => (
              <button
                key={corner}
                type="button"
                aria-label={`Resize ${LABELS[end]} box`}
                className={`absolute h-4 w-4 rounded-sm border-2 border-black/70 ${corner.includes("n") ? "-top-2" : "-bottom-2"} ${corner.includes("w") ? "-left-2" : "-right-2"}`}
                style={{ backgroundColor: colour, cursor: `${corner}-resize`, touchAction: "none" }}
                onPointerDown={(event) => startOperation(event, end, "resize", corner)}
              />
            ))}
          </div>
        );
      })}
    </div>
  );
}
