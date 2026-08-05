import type {
  CameraPreset,
  VirtualPitchMaterialPreset,
  VirtualPitchModel
} from "@/lib/virtual-pitch";

import type { CalibratedThreeCameraConfiguration } from "./calibratedCameraTypes";
import type { ActiveCameraDiagnostics } from "./VirtualPitchCameraController";
import type { ReactNode } from "react";


export type VirtualPitchRendererMode =
  | "development"
  | "camera-validation"
  | "real-frame-overlay"
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
  overlayOpacity?: number;
  materialPreset: VirtualPitchMaterialPreset;
}

export interface VirtualPitchSceneProps {
  model: VirtualPitchModel;
  camera: VirtualPitchCameraConfiguration;
  calibratedCamera?: CalibratedThreeCameraConfiguration;
  visualOptions: VirtualPitchVisualOptions;
  mode?: VirtualPitchRendererMode;
  onCameraDiagnostics?: (diagnostics: ActiveCameraDiagnostics) => void;
  frameloop?: "always" | "demand";
  exportWidth?: number;
  exportHeight?: number;
  onCanvasReady?: (canvas: HTMLCanvasElement, invalidate: () => void) => void;
  replayContent?: ReactNode;
}
