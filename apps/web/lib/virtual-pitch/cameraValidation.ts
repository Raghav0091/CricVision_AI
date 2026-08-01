import { Vector3, Vector4 } from "three";
import { toThreeVector } from "./coordinates";
import { projectThreeCameraPoint, type PixelPoint } from "./cameraProjection";
import type { CameraBridgeInput, ThreeCameraBridge } from "./opencvCameraBridge";
import type { WorldPoint3D } from "./types";

export type CameraValidationLandmark = {
  semanticId: string;
  world: WorldPoint3D;
};

export type ProjectionResult = {
  pixel: PixelPoint | null;
  cameraDepth: number;
  positiveDepth: boolean;
  finite: boolean;
};

export type LandmarkProjectionComparison = {
  semanticId: string;
  world: WorldPoint3D;
  openCvPixel: PixelPoint | null;
  threePixel: PixelPoint | null;
  xResidual: number | null;
  yResidual: number | null;
  pixelError: number | null;
  cameraDepth: number;
  positiveCameraDepth: boolean;
  inFrame: boolean;
  clippingState: "inside_frustum" | "outside_image" | "before_near" | "after_far" | "behind_camera" | "invalid";
};

export type CameraValidationMetrics = {
  pointCount: number;
  validPointCount: number;
  pointsBehindCamera: number;
  invalidPointCount: number;
  meanError: number | null;
  medianError: number | null;
  rmse: number | null;
  maximumError: number | null;
  horizontalBias: number | null;
  verticalBias: number | null;
  mirroredAxisWarning: boolean;
  bowlerStrikerReversalWarning: boolean;
  finiteMatrixStatus: boolean;
};

export type CameraValidationReport = {
  points: LandmarkProjectionComparison[];
  metrics: CameraValidationMetrics;
};

function multiplyRotation(input: CameraBridgeInput, world: WorldPoint3D): [number, number, number] {
  const rotation = input.rotation_matrix;
  const translation = input.translation_vector;
  return [
    rotation[0][0] * world.x + rotation[0][1] * world.y + rotation[0][2] * world.z + translation[0],
    rotation[1][0] * world.x + rotation[1][1] * world.y + rotation[1][2] * world.z + translation[1],
    rotation[2][0] * world.x + rotation[2][1] * world.y + rotation[2][2] * world.z + translation[2]
  ];
}

export function projectOpenCvWorldPoint(input: CameraBridgeInput, world: WorldPoint3D): ProjectionResult {
  const [x, y, z] = multiplyRotation(input, world);
  const finite = [x, y, z].every(Number.isFinite);
  if (!finite || z <= 0) return { pixel: null, cameraDepth: z, positiveDepth: z > 0, finite };
  const matrix = input.camera_matrix;
  const pixel = {
    x: (matrix[0][0] * x + matrix[0][1] * y) / z + matrix[0][2],
    y: matrix[1][1] * y / z + matrix[1][2]
  };
  return {
    pixel: Number.isFinite(pixel.x) && Number.isFinite(pixel.y) ? pixel : null,
    cameraDepth: z,
    positiveDepth: true,
    finite: Number.isFinite(pixel.x) && Number.isFinite(pixel.y)
  };
}

export function projectThreeWorldPoint(bridge: ThreeCameraBridge, world: WorldPoint3D): ProjectionResult {
  const point = toThreeVector(world);
  const cameraPoint4 = new Vector4(point.x, point.y, point.z, 1).applyMatrix4(bridge.matrixWorldInverse);
  const cameraPoint = new Vector3(cameraPoint4.x, cameraPoint4.y, cameraPoint4.z);
  const cameraDepth = -cameraPoint.z;
  if (![cameraPoint.x, cameraPoint.y, cameraPoint.z].every(Number.isFinite) || cameraDepth <= 0) {
    return { pixel: null, cameraDepth, positiveDepth: cameraDepth > 0, finite: false };
  }
  const projected = projectThreeCameraPoint(
    cameraPoint,
    bridge.projectionMatrix,
    bridge.intrinsics.imageWidth,
    bridge.intrinsics.imageHeight
  );
  return {
    pixel: projected.pixel,
    cameraDepth,
    positiveDepth: true,
    finite: projected.pixel !== null
  };
}

function median(values: readonly number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function mean(values: readonly number[]): number | null {
  return values.length === 0 ? null : values.reduce((sum, value) => sum + value, 0) / values.length;
}

function signCorrelation(reference: readonly number[], comparison: readonly number[]): number {
  if (reference.length < 2 || comparison.length !== reference.length) return 1;
  const referenceMean = mean(reference) ?? 0;
  const comparisonMean = mean(comparison) ?? 0;
  let covariance = 0;
  let referenceVariance = 0;
  let comparisonVariance = 0;
  for (let index = 0; index < reference.length; index += 1) {
    const referenceDelta = reference[index] - referenceMean;
    const comparisonDelta = comparison[index] - comparisonMean;
    covariance += referenceDelta * comparisonDelta;
    referenceVariance += referenceDelta * referenceDelta;
    comparisonVariance += comparisonDelta * comparisonDelta;
  }
  const denominator = Math.sqrt(referenceVariance * comparisonVariance);
  return denominator > 1e-12 ? covariance / denominator : 1;
}

function detectsLongitudinalReversal(
  landmarks: readonly CameraValidationLandmark[],
  comparisons: readonly LandmarkProjectionComparison[]
): boolean {
  for (let left = 0; left < landmarks.length; left += 1) {
    for (let right = left + 1; right < landmarks.length; right += 1) {
      if (Math.abs(landmarks[left].world.y - landmarks[right].world.y) < 1) continue;
      const first = comparisons[left];
      const second = comparisons[right];
      if (!first.openCvPixel || !first.threePixel || !second.openCvPixel || !second.threePixel) continue;
      const openCvDelta = {
        x: second.openCvPixel.x - first.openCvPixel.x,
        y: second.openCvPixel.y - first.openCvPixel.y
      };
      const threeDelta = {
        x: second.threePixel.x - first.threePixel.x,
        y: second.threePixel.y - first.threePixel.y
      };
      const dot = openCvDelta.x * threeDelta.x + openCvDelta.y * threeDelta.y;
      if (dot < -1e-8) return true;
    }
  }
  return false;
}

export function validateCameraBridge(
  bridge: ThreeCameraBridge,
  landmarks: readonly CameraValidationLandmark[]
): CameraValidationReport {
  const width = bridge.intrinsics.imageWidth;
  const height = bridge.intrinsics.imageHeight;
  const points = landmarks.map(({ semanticId, world }): LandmarkProjectionComparison => {
    const openCv = projectOpenCvWorldPoint(bridge.input, world);
    const three = projectThreeWorldPoint(bridge, world);
    const comparable = openCv.pixel !== null && three.pixel !== null;
    const xResidual = comparable ? three.pixel!.x - openCv.pixel!.x : null;
    const yResidual = comparable ? three.pixel!.y - openCv.pixel!.y : null;
    const pixelError = xResidual === null || yResidual === null ? null : Math.hypot(xResidual, yResidual);
    const inFrame = openCv.pixel !== null
      && openCv.pixel.x >= 0 && openCv.pixel.x <= width
      && openCv.pixel.y >= 0 && openCv.pixel.y <= height;
    let clippingState: LandmarkProjectionComparison["clippingState"];
    if (!openCv.finite || !three.finite) clippingState = openCv.cameraDepth <= 0 ? "behind_camera" : "invalid";
    else if (openCv.cameraDepth < bridge.near) clippingState = "before_near";
    else if (openCv.cameraDepth > bridge.far) clippingState = "after_far";
    else clippingState = inFrame ? "inside_frustum" : "outside_image";
    return {
      semanticId,
      world,
      openCvPixel: openCv.pixel,
      threePixel: three.pixel,
      xResidual,
      yResidual,
      pixelError,
      cameraDepth: openCv.cameraDepth,
      positiveCameraDepth: openCv.positiveDepth,
      inFrame,
      clippingState
    };
  });

  const valid = points.filter((point) => point.pixelError !== null);
  const errors = valid.map((point) => point.pixelError!);
  const horizontalResiduals = valid.map((point) => point.xResidual!);
  const verticalResiduals = valid.map((point) => point.yResidual!);
  const openCvX = valid.map((point) => point.openCvPixel!.x);
  const threeX = valid.map((point) => point.threePixel!.x);
  const openCvY = valid.map((point) => point.openCvPixel!.y);
  const threeY = valid.map((point) => point.threePixel!.y);
  return {
    points,
    metrics: {
      pointCount: points.length,
      validPointCount: valid.length,
      pointsBehindCamera: points.filter((point) => !point.positiveCameraDepth).length,
      invalidPointCount: points.length - valid.length,
      meanError: mean(errors),
      medianError: median(errors),
      rmse: errors.length === 0 ? null : Math.sqrt(errors.reduce((sum, value) => sum + value * value, 0) / errors.length),
      maximumError: errors.length === 0 ? null : Math.max(...errors),
      horizontalBias: mean(horizontalResiduals),
      verticalBias: mean(verticalResiduals),
      mirroredAxisWarning: signCorrelation(openCvX, threeX) < -0.9 || signCorrelation(openCvY, threeY) < -0.9,
      bowlerStrikerReversalWarning: detectsLongitudinalReversal(landmarks, points),
      finiteMatrixStatus: bridge.diagnostics.finiteMatrices
    }
  };
}
