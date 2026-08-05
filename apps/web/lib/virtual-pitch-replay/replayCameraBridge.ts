import type { CameraBridgeInput } from "@/lib/virtual-pitch/opencvCameraBridge";
import type { Matrix3Rows, Vector3Values } from "@/lib/virtual-pitch/cameraProjection";

import type { ReplayCamera } from "./types";

function asMatrix3Rows(matrix: number[][]): Matrix3Rows {
  return matrix as unknown as Matrix3Rows;
}

function asVector3(values: number[]): Vector3Values {
  return values as unknown as Vector3Values;
}

export function replayCameraToBridgeInput(
  camera: ReplayCamera,
  analysisId: string
): CameraBridgeInput | null {
  if (
    !camera.camera_matrix
    || !camera.rotation_matrix
    || !camera.translation_vector
    || !camera.image_width
    || !camera.image_height
  ) {
    return null;
  }

  const bridgeSource =
    camera.source === "CALIBRATED"
      ? (camera.calibration_source ?? "ACCEPTED_SCENE_CALIBRATION")
      : "replay_payload";

  return {
    source: bridgeSource,
    source_version: "1.0",
    analysis_id: analysisId,
    candidate_id: camera.preset_name,
    accepted: camera.source === "CALIBRATED",
    classification: camera.visualization_only ? "VISUALIZATION" : "CALIBRATED",
    image_width: camera.image_width,
    image_height: camera.image_height,
    camera_matrix: asMatrix3Rows(camera.camera_matrix),
    distortion_coefficients: camera.distortion_coefficients ?? [0, 0, 0, 0, 0],
    rotation_representation: "matrix",
    rotation_matrix: asMatrix3Rows(camera.rotation_matrix),
    translation_vector: asVector3(camera.translation_vector),
    extrinsic_convention: "opencv_world_to_camera",
    world_coordinate_system: "cricvision_pitch_v1",
    frame_preundistorted: false
  };
}
