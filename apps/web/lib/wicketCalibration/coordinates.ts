import type { WicketBox, WicketBoxRole } from "./types";


export type NormalizedBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};


export const DEFAULT_NORMALIZED_BOXES: Record<WicketBoxRole, NormalizedBox> = {
  FAR: { x: 0.34, y: 0.16, width: 0.32, height: 0.36 },
  NEAR: { x: 0.22, y: 0.48, width: 0.56, height: 0.48 }
};


export function defaultWicketBoxes(
  sourceWidth: number,
  sourceHeight: number,
  calibrationFrameIndex: number
): Record<WicketBoxRole, WicketBox> {
  return {
    FAR: normalizedToWicketBox(
      DEFAULT_NORMALIZED_BOXES.FAR,
      "FAR",
      sourceWidth,
      sourceHeight,
      calibrationFrameIndex
    ),
    NEAR: normalizedToWicketBox(
      DEFAULT_NORMALIZED_BOXES.NEAR,
      "NEAR",
      sourceWidth,
      sourceHeight,
      calibrationFrameIndex
    )
  };
}


export function normalizedToWicketBox(
  box: NormalizedBox,
  role: WicketBoxRole,
  sourceWidth: number,
  sourceHeight: number,
  calibrationFrameIndex: number
): WicketBox {
  return {
    role,
    x: roundNative(box.x * sourceWidth),
    y: roundNative(box.y * sourceHeight),
    width: roundNative(box.width * sourceWidth),
    height: roundNative(box.height * sourceHeight),
    source_image_width: sourceWidth,
    source_image_height: sourceHeight,
    calibration_frame_index: calibrationFrameIndex,
    validation_status: "PENDING"
  };
}


export function wicketBoxToNormalized(box: WicketBox): NormalizedBox {
  const width = Math.max(box.source_image_width, 1);
  const height = Math.max(box.source_image_height, 1);
  return {
    x: box.x / width,
    y: box.y / height,
    width: box.width / width,
    height: box.height / height
  };
}


export function normalizedPointToNative(
  point: { x: number; y: number },
  sourceWidth: number,
  sourceHeight: number
): { x: number; y: number } {
  return {
    x: roundNative(point.x * sourceWidth),
    y: roundNative(point.y * sourceHeight)
  };
}


export function clampNativePoint(
  point: { x: number; y: number },
  sourceWidth: number,
  sourceHeight: number
): { x: number; y: number } {
  return {
    x: roundNative(Math.max(0, Math.min(sourceWidth, point.x))),
    y: roundNative(Math.max(0, Math.min(sourceHeight, point.y)))
  };
}


function roundNative(value: number): number {
  return Math.round(value * 1000) / 1000;
}
