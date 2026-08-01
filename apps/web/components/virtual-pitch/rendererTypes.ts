import type {
  CameraPreset,
  VirtualPitchMaterialPreset,
  VirtualPitchModel
} from "@/lib/virtual-pitch";


export type VirtualPitchRendererMode =
  | "development"
  | "video-overlay"
  | "live-overlay"
  | "interactive-replay";

export type ThreeCoordinate = readonly [number, number, number];

export type VirtualPitchCameraConfiguration = CameraPreset;

export interface VirtualPitchVisualOptions {
  showPitch?: boolean;
  showStumps?: boolean;
  showBails?: boolean;
  showLines?: boolean;
  showCorridor?: boolean;
  showAxes?: boolean;
  showGrid?: boolean;
  showLandmarkLabels?: boolean;
  showBounds?: boolean;
  enableOrbitControls?: boolean;
  corridorOpacity?: number;
  lowPerformance?: boolean;
  dprCap?: number;
  materialPreset: VirtualPitchMaterialPreset;
}

export interface VirtualPitchSceneProps {
  model: VirtualPitchModel;
  camera: VirtualPitchCameraConfiguration;
  visualOptions: VirtualPitchVisualOptions;
  mode?: VirtualPitchRendererMode;
}
