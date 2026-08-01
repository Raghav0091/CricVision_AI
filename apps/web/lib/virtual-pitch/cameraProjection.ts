import { Matrix4, Vector3, Vector4 } from "three";

export type Matrix3Rows = readonly [
  readonly [number, number, number],
  readonly [number, number, number],
  readonly [number, number, number]
];

export type Vector3Values = readonly [number, number, number];

export type CameraIntrinsics = {
  imageWidth: number;
  imageHeight: number;
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  skew: number;
};

export type PixelPoint = { x: number; y: number };
export type NdcPoint = { x: number; y: number };

export type ContainMapping = {
  nativeWidth: number;
  nativeHeight: number;
  displayWidth: number;
  displayHeight: number;
  renderedWidth: number;
  renderedHeight: number;
  scale: number;
  offsetX: number;
  offsetY: number;
};

const EPSILON = 1e-12;

function requirePositiveFinite(value: number, label: string): void {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`${label} must be positive and finite.`);
  }
}

function requireFinite(value: number, label: string): void {
  if (!Number.isFinite(value)) throw new Error(`${label} must be finite.`);
}

export function intrinsicsFromCameraMatrix(
  cameraMatrix: readonly (readonly number[])[],
  imageWidth: number,
  imageHeight: number
): CameraIntrinsics {
  requirePositiveFinite(imageWidth, "Image width");
  requirePositiveFinite(imageHeight, "Image height");
  if (
    cameraMatrix.length !== 3
    || cameraMatrix.some((row) => row.length !== 3)
    || cameraMatrix.flat().some((value) => !Number.isFinite(value))
  ) {
    throw new Error("Camera matrix must be a finite 3x3 matrix.");
  }
  if (
    Math.abs(cameraMatrix[2][0]) > EPSILON
    || Math.abs(cameraMatrix[2][1]) > EPSILON
    || Math.abs(cameraMatrix[2][2] - 1) > EPSILON
    || Math.abs(cameraMatrix[1][0]) > EPSILON
  ) {
    throw new Error("Camera matrix must use the OpenCV pinhole form [[fx, skew, cx], [0, fy, cy], [0, 0, 1]].");
  }

  const intrinsics = {
    imageWidth,
    imageHeight,
    fx: cameraMatrix[0][0],
    fy: cameraMatrix[1][1],
    cx: cameraMatrix[0][2],
    cy: cameraMatrix[1][2],
    skew: cameraMatrix[0][1]
  };
  requirePositiveFinite(intrinsics.fx, "fx");
  requirePositiveFinite(intrinsics.fy, "fy");
  requireFinite(intrinsics.cx, "cx");
  requireFinite(intrinsics.cy, "cy");
  requireFinite(intrinsics.skew, "skew");
  return intrinsics;
}

/**
 * OpenCV pixels use a top-left origin. Three/WebGL uses bottom-left NDC with a
 * camera looking down -Z, hence the principal-point and skew signs below.
 */
export function buildCalibratedProjectionMatrix(
  intrinsics: CameraIntrinsics,
  near: number,
  far: number
): Matrix4 {
  requirePositiveFinite(intrinsics.imageWidth, "Image width");
  requirePositiveFinite(intrinsics.imageHeight, "Image height");
  requirePositiveFinite(intrinsics.fx, "fx");
  requirePositiveFinite(intrinsics.fy, "fy");
  requirePositiveFinite(near, "Near plane");
  requirePositiveFinite(far, "Far plane");
  if (far <= near) throw new Error("Far plane must be greater than near plane.");

  const { imageWidth: width, imageHeight: height, fx, fy, cx, cy, skew } = intrinsics;
  return new Matrix4().set(
    2 * fx / width, -2 * skew / width, 1 - 2 * cx / width, 0,
    0, 2 * fy / height, 2 * cy / height - 1, 0,
    0, 0, -(far + near) / (far - near), -2 * far * near / (far - near),
    0, 0, -1, 0
  );
}

export function buildProjectionMatrixInverse(projectionMatrix: Matrix4): Matrix4 {
  const determinant = projectionMatrix.determinant();
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= EPSILON) {
    throw new Error("Projection matrix is not invertible.");
  }
  return projectionMatrix.clone().invert();
}

export function nativePixelToNdc(pixel: PixelPoint, imageWidth: number, imageHeight: number): NdcPoint {
  requirePositiveFinite(imageWidth, "Image width");
  requirePositiveFinite(imageHeight, "Image height");
  return {
    x: 2 * pixel.x / imageWidth - 1,
    y: 1 - 2 * pixel.y / imageHeight
  };
}

export function ndcToNativePixel(ndc: NdcPoint, imageWidth: number, imageHeight: number): PixelPoint {
  requirePositiveFinite(imageWidth, "Image width");
  requirePositiveFinite(imageHeight, "Image height");
  return {
    x: (ndc.x + 1) * imageWidth / 2,
    y: (1 - ndc.y) * imageHeight / 2
  };
}

export function projectThreeCameraPoint(
  cameraPoint: Vector3,
  projectionMatrix: Matrix4,
  imageWidth: number,
  imageHeight: number
): { pixel: PixelPoint | null; ndc: NdcPoint | null; clipW: number } {
  const clip = new Vector4(cameraPoint.x, cameraPoint.y, cameraPoint.z, 1).applyMatrix4(projectionMatrix);
  if (!Number.isFinite(clip.w) || Math.abs(clip.w) <= EPSILON) {
    return { pixel: null, ndc: null, clipW: clip.w };
  }
  const ndc = { x: clip.x / clip.w, y: clip.y / clip.w };
  if (!Number.isFinite(ndc.x) || !Number.isFinite(ndc.y)) {
    return { pixel: null, ndc: null, clipW: clip.w };
  }
  return { pixel: ndcToNativePixel(ndc, imageWidth, imageHeight), ndc, clipW: clip.w };
}

export function calculateContainMapping(
  nativeWidth: number,
  nativeHeight: number,
  displayWidth: number,
  displayHeight: number
): ContainMapping {
  requirePositiveFinite(nativeWidth, "Native width");
  requirePositiveFinite(nativeHeight, "Native height");
  requirePositiveFinite(displayWidth, "Display width");
  requirePositiveFinite(displayHeight, "Display height");
  const scale = Math.min(displayWidth / nativeWidth, displayHeight / nativeHeight);
  const renderedWidth = nativeWidth * scale;
  const renderedHeight = nativeHeight * scale;
  return {
    nativeWidth,
    nativeHeight,
    displayWidth,
    displayHeight,
    renderedWidth,
    renderedHeight,
    scale,
    offsetX: (displayWidth - renderedWidth) / 2,
    offsetY: (displayHeight - renderedHeight) / 2
  };
}

export function nativePixelToDisplay(pixel: PixelPoint, mapping: ContainMapping): PixelPoint {
  return {
    x: mapping.offsetX + pixel.x * mapping.scale,
    y: mapping.offsetY + pixel.y * mapping.scale
  };
}

export function displayPixelToNative(pixel: PixelPoint, mapping: ContainMapping): PixelPoint {
  return {
    x: (pixel.x - mapping.offsetX) / mapping.scale,
    y: (pixel.y - mapping.offsetY) / mapping.scale
  };
}
