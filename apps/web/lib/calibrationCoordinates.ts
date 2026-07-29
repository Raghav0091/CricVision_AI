export type CalibrationPoint = { x: number; y: number };
export type CalibrationSize = { width: number; height: number };
export type CalibrationViewTransform = {
  zoom: number;
  panX: number;
  panY: number;
  devicePixelRatio?: number;
};


export type ContainedMediaRect = {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
};


export function containedMediaRect(
  native: CalibrationSize,
  container: CalibrationSize
): ContainedMediaRect {
  if (
    native.width <= 0
    || native.height <= 0
    || container.width <= 0
    || container.height <= 0
  ) {
    throw new Error("Calibration dimensions must be positive.");
  }
  const scale = Math.min(
    container.width / native.width,
    container.height / native.height
  );
  const width = native.width * scale;
  const height = native.height * scale;
  return {
    x: (container.width - width) / 2,
    y: (container.height - height) / 2,
    width,
    height,
    scale
  };
}


export function videoNativeToDisplay(
  point: CalibrationPoint,
  native: CalibrationSize,
  container: CalibrationSize,
  transform: CalibrationViewTransform = { zoom: 1, panX: 0, panY: 0 }
): CalibrationPoint {
  const rect = containedMediaRect(native, container);
  const zoom = Math.max(transform.zoom, 0.01);
  const centreX = container.width / 2;
  const centreY = container.height / 2;
  const baseX = rect.x + point.x * rect.scale;
  const baseY = rect.y + point.y * rect.scale;
  return {
    x: centreX + (baseX - centreX) * zoom + transform.panX,
    y: centreY + (baseY - centreY) * zoom + transform.panY
  };
}


export function displayToVideoNative(
  point: CalibrationPoint,
  native: CalibrationSize,
  container: CalibrationSize,
  transform: CalibrationViewTransform = { zoom: 1, panX: 0, panY: 0 }
): CalibrationPoint | null {
  const rect = containedMediaRect(native, container);
  const zoom = Math.max(transform.zoom, 0.01);
  const centreX = container.width / 2;
  const centreY = container.height / 2;
  const baseX = centreX + (point.x - transform.panX - centreX) / zoom;
  const baseY = centreY + (point.y - transform.panY - centreY) / zoom;
  const x = (baseX - rect.x) / rect.scale;
  const y = (baseY - rect.y) / rect.scale;
  if (x < 0 || y < 0 || x >= native.width || y >= native.height) {
    return null;
  }
  return { x, y };
}

