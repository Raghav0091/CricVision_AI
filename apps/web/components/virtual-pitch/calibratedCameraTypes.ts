import type { ThreeCameraBridge } from "@/lib/virtual-pitch/opencvCameraBridge";


/** Structural transport shape accepted at the renderer boundary. */
export interface CalibratedCameraBridgeInput {
  source: string;
  source_version: string;
  accepted: boolean;
  image_width: number;
  image_height: number;
  camera_matrix: readonly (readonly number[])[];
  distortion_coefficients: readonly number[];
  rotation_representation: string;
  rotation_matrix: readonly (readonly number[])[];
  translation_vector: readonly number[];
  extrinsic_convention: string;
  world_coordinate_system: string;
}

/** The renderer accepts a normalized transport input or a prebuilt bridge. */
export type CalibratedThreeCameraConfiguration = CalibratedCameraBridgeInput | ThreeCameraBridge;
