import type {
  PitchProjectionGeometry,
  VirtualPitchCamera,
  VirtualPitchPixelPoint,
  VirtualPitchWorldPoint
} from "@/lib/api";
import type { CameraBridgeInput } from "@/lib/virtual-pitch/opencvCameraBridge";


export type CameraSourceMode =
  | "auto-registration"
  | "development"
  | "synthetic-opencv"
  | "real-analysis";

export type OverlayComparisonMode =
  | "three"
  | "svg"
  | "both"
  | "side-by-side";

export type DistortionMode =
  | "ZERO_DISTORTION"
  | "PREUNDISTORTED_FRAME"
  | "NONZERO_DISTORTION_UNSUPPORTED";

export type LabCameraBridgeInput = CameraBridgeInput & {
  setup_frame_url?: string | null;
};

export type LandmarkProjectionComparison = {
  semantic_id: string;
  world_point: VirtualPitchWorldPoint;
  opencv_pixel?: VirtualPitchPixelPoint | null;
  three_pixel?: VirtualPitchPixelPoint | null;
  residual_x_px?: number | null;
  residual_y_px?: number | null;
  error_px?: number | null;
  camera_depth?: number | null;
  in_frame?: boolean;
  clipped?: boolean;
};

export type CameraBridgeDiagnostics = {
  point_count: number;
  valid_point_count: number;
  invalid_point_count: number;
  points_behind_camera: number;
  rmse_px?: number | null;
  maximum_error_px?: number | null;
  mean_error_px?: number | null;
  median_error_px?: number | null;
  horizontal_bias_px?: number | null;
  vertical_bias_px?: number | null;
  finite_matrices: boolean;
  mirrored_axis_warning: boolean;
  bowler_striker_reversal_warning: boolean;
  distortion_mode: DistortionMode;
  exact: boolean;
  warnings: string[];
};

export type CameraBridgePayload = {
  camera: LabCameraBridgeInput;
  projection?: PitchProjectionGeometry | null;
  comparisons?: LandmarkProjectionComparison[];
  diagnostics?: CameraBridgeDiagnostics | null;
};


export function syntheticCameraBridgeInput(camera: VirtualPitchCamera): LabCameraBridgeInput {
  const coefficientMagnitude = Math.max(
    0,
    ...camera.distortion_coefficients.map((value) => Math.abs(value))
  );
  const zeroDistortion = coefficientMagnitude <= 1e-12;
  return {
    source: "synthetic_virtual_pitch_camera",
    source_version: "virtual_pitch_v1",
    analysis_id: null,
    candidate_id: camera.name,
    accepted: false,
    classification: "SYNTHETIC_EXACT",
    image_width: camera.image_width,
    image_height: camera.image_height,
    camera_matrix: camera.camera_matrix as unknown as CameraBridgeInput["camera_matrix"],
    distortion_coefficients: camera.distortion_coefficients,
    rotation_representation: "matrix_and_rodrigues",
    rotation_vector: camera.rotation_vector as unknown as CameraBridgeInput["rotation_vector"],
    rotation_matrix: camera.rotation_matrix as unknown as CameraBridgeInput["rotation_matrix"],
    translation_vector: camera.translation_vector as unknown as CameraBridgeInput["translation_vector"],
    extrinsic_convention: "opencv_world_to_camera",
    world_coordinate_system: "cricvision_pitch_v1",
    setup_frame_url: null,
    warnings: zeroDistortion
      ? []
      : ["Camera bridge requires a zero-distortion camera model or a pre-undistorted background frame."],
    frame_preundistorted: false,
    near: camera.near_m,
    far: camera.far_m
  };
}
