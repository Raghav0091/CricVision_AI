import { Matrix4, Vector3 } from "three";
import {
  buildCalibratedProjectionMatrix,
  buildProjectionMatrixInverse,
  intrinsicsFromCameraMatrix,
  type CameraIntrinsics,
  type Matrix3Rows,
  type Vector3Values
} from "./cameraProjection";

export type DistortionMode =
  | "ZERO_DISTORTION"
  | "PREUNDISTORTED_FRAME"
  | "NONZERO_DISTORTION_UNSUPPORTED";

export type OpenCvExtrinsicConvention = "opencv_world_to_camera";
export type CricVisionWorldCoordinateSystem = "cricvision_pitch_v1";

/** Normalized backend-to-renderer camera contract. Matrices are row-major. */
export type CameraBridgeInput = {
  source: string;
  source_version: string;
  analysis_id?: string | null;
  candidate_id?: string | null;
  accepted: boolean;
  classification?: string | null;
  image_width: number;
  image_height: number;
  camera_matrix: Matrix3Rows;
  distortion_coefficients: readonly number[];
  rotation_representation: "matrix" | "matrix_and_rodrigues";
  rotation_vector?: Vector3Values | null;
  rotation_matrix: Matrix3Rows;
  translation_vector: Vector3Values;
  extrinsic_convention: OpenCvExtrinsicConvention;
  world_coordinate_system: CricVisionWorldCoordinateSystem;
  setup_frame_url?: string | null;
  frame_preundistorted?: boolean;
  near?: number;
  far?: number;
  warnings?: readonly string[];
};

export type DistortionAssessment = {
  mode: DistortionMode;
  coefficientMagnitude: number;
  coefficientL2Norm: number;
  framePreundistorted: boolean;
  exact: boolean;
  supported: boolean;
  warning: string | null;
};

export type CameraBridgeDiagnostics = {
  finiteMatrices: boolean;
  rotationDeterminant: number;
  worldAxisDeterminant: number;
  cameraAxisDeterminant: number;
  worldToCameraDeterminant: number;
  handednessPreserved: boolean;
  distortion: DistortionAssessment;
  warnings: string[];
};

export type ThreeCameraBridge = {
  input: CameraBridgeInput;
  intrinsics: CameraIntrinsics;
  projectionMatrix: Matrix4;
  projectionMatrixInverse: Matrix4;
  matrixWorld: Matrix4;
  matrixWorldInverse: Matrix4;
  cameraWorldPosition: Vector3;
  cameraForwardDirection: Vector3;
  near: number;
  far: number;
  renderable: boolean;
  exact: boolean;
  diagnostics: CameraBridgeDiagnostics;
};

const ZERO_DISTORTION_TOLERANCE = 1e-12;

/** CricVision world (x, y, z) -> Three world (x, z, -y). */
export const CRICVISION_WORLD_TO_THREE = new Matrix4().set(
  1, 0, 0, 0,
  0, 0, 1, 0,
  0, -1, 0, 0,
  0, 0, 0, 1
);

/** OpenCV camera (x right, y down, z forward) -> Three camera (x right, y up, z back). */
export const OPENCV_CAMERA_TO_THREE_CAMERA = new Matrix4().set(
  1, 0, 0, 0,
  0, -1, 0, 0,
  0, 0, -1, 0,
  0, 0, 0, 1
);

function matrixFromRotationTranslation(rotation: Matrix3Rows, translation: Vector3Values): Matrix4 {
  const values = [...rotation.flat(), ...translation];
  if (values.some((value) => !Number.isFinite(value))) {
    throw new Error("OpenCV extrinsics must contain only finite values.");
  }
  return new Matrix4().set(
    rotation[0][0], rotation[0][1], rotation[0][2], translation[0],
    rotation[1][0], rotation[1][1], rotation[1][2], translation[1],
    rotation[2][0], rotation[2][1], rotation[2][2], translation[2],
    0, 0, 0, 1
  );
}

export function assessDistortion(
  coefficients: readonly number[],
  framePreundistorted = false
): DistortionAssessment {
  if (coefficients.some((value) => !Number.isFinite(value))) {
    throw new Error("Distortion coefficients must be finite.");
  }
  const coefficientMagnitude = coefficients.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0);
  const coefficientL2Norm = Math.sqrt(coefficients.reduce((sum, value) => sum + value * value, 0));
  if (coefficientMagnitude <= ZERO_DISTORTION_TOLERANCE) {
    return {
      mode: "ZERO_DISTORTION",
      coefficientMagnitude,
      coefficientL2Norm,
      framePreundistorted,
      exact: true,
      supported: true,
      warning: null
    };
  }
  if (framePreundistorted) {
    return {
      mode: "PREUNDISTORTED_FRAME",
      coefficientMagnitude,
      coefficientL2Norm,
      framePreundistorted: true,
      exact: true,
      supported: true,
      warning: "The bridge assumes the background frame was pre-undistorted with this camera calibration."
    };
  }
  return {
    mode: "NONZERO_DISTORTION_UNSUPPORTED",
    coefficientMagnitude,
    coefficientL2Norm,
    framePreundistorted: false,
    exact: false,
    supported: false,
    warning: "Camera bridge requires a zero-distortion camera model or a pre-undistorted background frame."
  };
}

function allFinite(matrix: Matrix4): boolean {
  return matrix.elements.every(Number.isFinite);
}

/**
 * Builds a Three camera from OpenCV's X_camera = R * X_cricvision + t.
 *
 * W2C_three = C_camera * W2C_opencv * inverse(S_world), where S_world is
 * CricVision-world to Three-world and C_camera flips OpenCV camera Y and Z.
 */
export function buildThreeCameraFromOpenCv(input: CameraBridgeInput): ThreeCameraBridge {
  if (input.extrinsic_convention !== "opencv_world_to_camera") {
    throw new Error(`Unsupported extrinsic convention: ${input.extrinsic_convention}`);
  }
  if (input.world_coordinate_system !== "cricvision_pitch_v1") {
    throw new Error(`Unsupported world coordinate system: ${input.world_coordinate_system}`);
  }
  const near = input.near ?? 0.01;
  const far = input.far ?? 1000;
  const intrinsics = intrinsicsFromCameraMatrix(input.camera_matrix, input.image_width, input.image_height);
  const projectionMatrix = buildCalibratedProjectionMatrix(intrinsics, near, far);
  const projectionMatrixInverse = buildProjectionMatrixInverse(projectionMatrix);
  const openCvWorldToCamera = matrixFromRotationTranslation(input.rotation_matrix, input.translation_vector);
  const threeToCricVisionWorld = CRICVISION_WORLD_TO_THREE.clone().invert();
  const matrixWorldInverse = OPENCV_CAMERA_TO_THREE_CAMERA.clone()
    .multiply(openCvWorldToCamera)
    .multiply(threeToCricVisionWorld);
  const determinant = matrixWorldInverse.determinant();
  if (!Number.isFinite(determinant) || Math.abs(determinant) <= ZERO_DISTORTION_TOLERANCE) {
    throw new Error("Converted world-to-camera matrix is not invertible.");
  }
  const matrixWorld = matrixWorldInverse.clone().invert();
  const cameraWorldPosition = new Vector3().setFromMatrixPosition(matrixWorld);
  const cameraForwardDirection = new Vector3(0, 0, -1).transformDirection(matrixWorld);
  const distortion = assessDistortion(input.distortion_coefficients, input.frame_preundistorted ?? false);
  const rotationMatrix = matrixFromRotationTranslation(input.rotation_matrix, [0, 0, 0]);
  const rotationDeterminant = rotationMatrix.determinant();
  const finiteMatrices = [projectionMatrix, projectionMatrixInverse, matrixWorld, matrixWorldInverse].every(allFinite);
  const warnings = [...(input.warnings ?? [])];
  if (distortion.warning) warnings.push(distortion.warning);
  if (Math.abs(rotationDeterminant - 1) > 1e-4) warnings.push("OpenCV rotation matrix is not a proper unit rotation.");
  if (!finiteMatrices) warnings.push("Camera conversion produced a non-finite matrix.");

  const worldAxisDeterminant = CRICVISION_WORLD_TO_THREE.determinant();
  const cameraAxisDeterminant = OPENCV_CAMERA_TO_THREE_CAMERA.determinant();
  return {
    input,
    intrinsics,
    projectionMatrix,
    projectionMatrixInverse,
    matrixWorld,
    matrixWorldInverse,
    cameraWorldPosition,
    cameraForwardDirection,
    near,
    far,
    renderable: distortion.supported && finiteMatrices,
    exact: distortion.exact && finiteMatrices,
    diagnostics: {
      finiteMatrices,
      rotationDeterminant,
      worldAxisDeterminant,
      cameraAxisDeterminant,
      worldToCameraDeterminant: determinant,
      handednessPreserved: worldAxisDeterminant > 0 && cameraAxisDeterminant > 0 && rotationDeterminant > 0,
      distortion,
      warnings
    }
  };
}
