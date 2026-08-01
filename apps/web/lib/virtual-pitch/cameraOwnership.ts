import { PerspectiveCamera, Vector3, Vector4 } from "three";

import { toThreeVector } from "./coordinates";
import { projectThreeCameraPoint, type PixelPoint } from "./cameraProjection";
import {
  projectOpenCvWorldPoint,
  projectThreeWorldPoint,
  type CameraValidationLandmark
} from "./cameraValidation";
import type { ThreeCameraBridge } from "./opencvCameraBridge";
import type { WorldPoint3D } from "./types";


export type OwnedPerspectiveCamera = PerspectiveCamera & { manual?: boolean };
export type CameraFamily = "DEVELOPMENT_CAMERA" | "CALIBRATED_OPENCV_CAMERA";

export type ActiveRendererProjection = {
  semanticId: string;
  world: WorldPoint3D;
  openCvPixel: PixelPoint | null;
  bridgePixel: PixelPoint | null;
  activeCameraPixel: PixelPoint | null;
  openCvToBridgeError: number | null;
  bridgeToActiveCameraError: number | null;
};

export type ActiveRendererCameraValidation = {
  points: ActiveRendererProjection[];
  rmse: number | null;
  maximumError: number | null;
};


export function cameraFamilyForBridge(bridge: ThreeCameraBridge | null): CameraFamily {
  return bridge ? "CALIBRATED_OPENCV_CAMERA" : "DEVELOPMENT_CAMERA";
}


export function shouldMountOrbitControls(family: CameraFamily, requested: boolean): boolean {
  return family === "DEVELOPMENT_CAMERA" && requested;
}


export function matrixChecksum(elements: readonly number[]): string {
  let hash = 2166136261;
  for (const value of elements) {
    const text = Number.isFinite(value) ? value.toPrecision(15) : String(value);
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}


export function configureCalibratedCamera(
  camera: OwnedPerspectiveCamera,
  bridge: ThreeCameraBridge
): OwnedPerspectiveCamera {
  if (!bridge.renderable) {
    throw new Error(bridge.diagnostics.distortion.warning ?? "The calibrated camera cannot be rendered exactly.");
  }
  camera.manual = true;
  camera.matrixAutoUpdate = false;
  camera.near = bridge.near;
  camera.far = bridge.far;
  camera.projectionMatrix.copy(bridge.projectionMatrix);
  camera.projectionMatrixInverse.copy(bridge.projectionMatrixInverse);
  camera.matrix.copy(bridge.matrixWorld);
  camera.matrixWorld.copy(bridge.matrixWorld);
  camera.matrixWorldInverse.copy(bridge.matrixWorldInverse);
  camera.matrixWorld.decompose(camera.position, camera.quaternion, camera.scale);
  camera.matrixWorldNeedsUpdate = false;
  return camera;
}


export function projectActiveCameraWorldPoint(
  camera: PerspectiveCamera,
  bridge: ThreeCameraBridge,
  world: WorldPoint3D
): PixelPoint | null {
  const point = toThreeVector(world);
  const cameraPoint = new Vector4(point.x, point.y, point.z, 1).applyMatrix4(camera.matrixWorldInverse);
  if (![cameraPoint.x, cameraPoint.y, cameraPoint.z].every(Number.isFinite) || -cameraPoint.z <= 0) return null;
  return projectThreeCameraPoint(
    new Vector3(cameraPoint.x, cameraPoint.y, cameraPoint.z),
    camera.projectionMatrix,
    bridge.intrinsics.imageWidth,
    bridge.intrinsics.imageHeight
  ).pixel;
}


function pixelError(left: PixelPoint | null, right: PixelPoint | null): number | null {
  return left && right ? Math.hypot(right.x - left.x, right.y - left.y) : null;
}


export function validateActiveRendererCamera(
  camera: PerspectiveCamera,
  bridge: ThreeCameraBridge,
  landmarks: readonly CameraValidationLandmark[]
): ActiveRendererCameraValidation {
  const points = landmarks.map(({ semanticId, world }) => {
    const openCvPixel = projectOpenCvWorldPoint(bridge.input, world).pixel;
    const bridgePixel = projectThreeWorldPoint(bridge, world).pixel;
    const activeCameraPixel = projectActiveCameraWorldPoint(camera, bridge, world);
    return {
      semanticId,
      world,
      openCvPixel,
      bridgePixel,
      activeCameraPixel,
      openCvToBridgeError: pixelError(openCvPixel, bridgePixel),
      bridgeToActiveCameraError: pixelError(bridgePixel, activeCameraPixel)
    };
  });
  const errors = points
    .map((point) => point.bridgeToActiveCameraError)
    .filter((error): error is number => error !== null);
  return {
    points,
    rmse: errors.length
      ? Math.sqrt(errors.reduce((sum, error) => sum + error * error, 0) / errors.length)
      : null,
    maximumError: errors.length ? Math.max(...errors) : null
  };
}


export function validRenderBounds(width: number, height: number): boolean {
  return Number.isFinite(width) && Number.isFinite(height) && width > 0 && height > 0;
}
