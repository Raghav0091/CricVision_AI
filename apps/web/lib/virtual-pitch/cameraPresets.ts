import { toThreeVector } from "./coordinates";
import { modelBounds } from "./geometry";
import type {
  CameraAdjustments,
  CameraPreset,
  CameraPresetId,
  CricketVector3,
  ResolvedCameraAdjustments,
  ThreeVector3,
  VirtualPitchModel
} from "./types";

const REFERENCE_ASPECT = 16 / 9;
const DEFAULT_VERTICAL_FOV = 48;
const MIN_FOV = 20;
const MAX_FOV = 100;
const SETUP_CAMERA_HEIGHT_M = 1.25;
const SETUP_DISTANCE_PITCH_WIDTHS = 1.25;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.max(minimum, Math.min(maximum, value));
}

export function aspectAdjustedVerticalFov(baseVerticalFovDegrees: number, aspect: number): number {
  if (!(aspect > 0) || !Number.isFinite(aspect)) throw new Error("Camera aspect must be positive and finite.");
  const baseRadians = baseVerticalFovDegrees * Math.PI / 180;
  const referenceHorizontal = 2 * Math.atan(Math.tan(baseRadians / 2) * REFERENCE_ASPECT);
  const adjusted = 2 * Math.atan(Math.tan(referenceHorizontal / 2) / aspect) * 180 / Math.PI;
  return clamp(adjusted, MIN_FOV, MAX_FOV);
}

export function distanceToFrame(radius: number, verticalFovDegrees: number, aspect: number): number {
  if (!(radius > 0) || !Number.isFinite(radius)) throw new Error("Frame radius must be positive and finite.");
  if (!(aspect > 0) || !Number.isFinite(aspect)) throw new Error("Camera aspect must be positive and finite.");
  const vertical = verticalFovDegrees * Math.PI / 180;
  const horizontal = 2 * Math.atan(Math.tan(vertical / 2) * aspect);
  return radius / Math.sin(Math.min(vertical, horizontal) / 2);
}

function rotateView(position: CricketVector3, target: CricketVector3, yawDegrees = 0, pitchDegrees = 0) {
  const yaw = yawDegrees * Math.PI / 180;
  const pitch = pitchDegrees * Math.PI / 180;
  const delta = { x: position.x - target.x, y: position.y - target.y, z: position.z - target.z };
  const yawed = {
    x: delta.x * Math.cos(yaw) - delta.y * Math.sin(yaw),
    y: delta.x * Math.sin(yaw) + delta.y * Math.cos(yaw),
    z: delta.z
  };
  const horizontal = Math.hypot(yawed.x, yawed.y);
  const currentPitch = Math.atan2(yawed.z, horizontal);
  const azimuth = Math.atan2(yawed.y, yawed.x);
  const radius = Math.hypot(horizontal, yawed.z);
  const nextPitch = currentPitch + pitch;
  const nextHorizontal = radius * Math.cos(nextPitch);
  return {
    x: target.x + nextHorizontal * Math.cos(azimuth),
    y: target.y + nextHorizontal * Math.sin(azimuth),
    z: target.z + radius * Math.sin(nextPitch)
  };
}

function rollUp(rollDegrees = 0): ThreeVector3 {
  const roll = rollDegrees * Math.PI / 180;
  return { x: Math.sin(roll), y: Math.cos(roll), z: 0 };
}

export function calculateCameraPreset(
  id: CameraPresetId,
  model: VirtualPitchModel,
  aspect: number,
  adjustments: CameraAdjustments = {}
): CameraPreset {
  const bounds = modelBounds(model);
  const resolved = resolveCameraAdjustments(model, aspect, adjustments, id);
  const fov = resolved.verticalFovDegrees;
  const fitDistance = distanceToFrame(Math.max(bounds.radius, model.dimensions.pitchWidthM), fov, aspect);
  const height = resolved.heightM;
  const behind = resolved.distanceBehindM;
  const lateral = resolved.lateralOffsetM;
  const targetHeight = resolved.targetHeightM;
  const centreY = bounds.centre.y;
  let position: CricketVector3;
  let target: CricketVector3;
  let label: string;
  let orbitEnabled = false;

  switch (id) {
    case "setup":
      label = "Synthetic Setup Camera";
      position = { x: lateral, y: bounds.min.y - behind, z: height };
      target = { x: 0, y: centreY, z: targetHeight };
      break;
    case "bowler-end":
      label = "Bowler-End View";
      position = { x: lateral, y: bounds.min.y - behind, z: Math.max(height, bounds.size.z * 2) };
      target = { x: 0, y: centreY, z: targetHeight };
      break;
    case "striker-end":
      label = "Striker-End View";
      position = { x: -lateral, y: bounds.max.y + behind, z: Math.max(height, bounds.size.z * 2) };
      target = { x: 0, y: centreY, z: targetHeight };
      break;
    case "side":
      label = "Side View";
      position = { x: bounds.max.x + fitDistance, y: centreY, z: Math.max(height, bounds.size.z * 2) };
      target = { x: 0, y: centreY, z: targetHeight };
      break;
    case "top-down":
      label = "Top-Down View";
      position = { x: 0, y: centreY, z: bounds.max.z + fitDistance };
      target = { x: 0, y: centreY, z: 0 };
      break;
    case "free-orbit":
      label = "Free Orbit";
      position = { x: bounds.max.x + fitDistance * 0.55, y: bounds.min.y - fitDistance * 0.35, z: bounds.max.z + fitDistance * 0.4 };
      target = bounds.centre;
      orbitEnabled = true;
      break;
  }

  position = rotateView(position, target, resolved.yawDegrees, resolved.pitchDegrees);
  const sceneRadius = Math.max(bounds.radius, 1);
  return {
    id,
    label,
    synthetic: true,
    position: toThreeVector(position),
    target: toThreeVector(target),
    up: id === "top-down"
      ? { x: 0, y: 0, z: -1 }
      : rollUp(resolved.rollDegrees),
    verticalFovDegrees: clamp(fov, MIN_FOV, MAX_FOV),
    near: Math.max(sceneRadius / 1000, Number.EPSILON),
    far: sceneRadius * 20 + fitDistance,
    orbitEnabled
  };
}

export function resolveCameraAdjustments(
  model: VirtualPitchModel,
  aspect: number,
  adjustments: CameraAdjustments = {},
  preset: CameraPresetId = "setup"
): ResolvedCameraAdjustments {
  const bounds = modelBounds(model);
  const defaultHeight = preset === "setup"
    ? SETUP_CAMERA_HEIGHT_M
    : Math.max(model.dimensions.stumpHeightM * 1.75, bounds.size.z * 1.5);
  const defaultDistanceBehind = preset === "setup"
    ? model.dimensions.pitchWidthM * SETUP_DISTANCE_PITCH_WIDTHS
    : Math.max(model.dimensions.pitchWidthM * 1.5, bounds.size.z * 4);
  return {
    heightM: adjustments.heightM ?? defaultHeight,
    distanceBehindM: adjustments.distanceBehindM ?? defaultDistanceBehind,
    lateralOffsetM: adjustments.lateralOffsetM ?? 0,
    verticalFovDegrees: adjustments.verticalFovDegrees ?? aspectAdjustedVerticalFov(DEFAULT_VERTICAL_FOV, aspect),
    targetHeightM: adjustments.targetHeightM ?? model.dimensions.stumpHeightM * 0.45,
    yawDegrees: adjustments.yawDegrees ?? 0,
    pitchDegrees: adjustments.pitchDegrees ?? 0,
    rollDegrees: adjustments.rollDegrees ?? 0
  };
}

export function allCameraPresets(model: VirtualPitchModel, aspect: number): readonly CameraPreset[] {
  const ids: readonly CameraPresetId[] = ["setup", "bowler-end", "striker-end", "side", "top-down", "free-orbit"];
  return ids.map((id) => calculateCameraPreset(id, model, aspect));
}
