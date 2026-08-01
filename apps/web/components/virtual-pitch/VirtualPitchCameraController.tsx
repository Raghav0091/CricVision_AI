import { OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useState, type ReactNode } from "react";
import { PerspectiveCamera } from "three";

import {
  configureCalibratedCamera,
  cameraFamilyForBridge,
  matrixChecksum,
  shouldMountOrbitControls,
  validateActiveRendererCamera,
  type ActiveRendererProjection,
  type CameraFamily,
  type OwnedPerspectiveCamera
} from "@/lib/virtual-pitch/cameraOwnership";
import type { CameraValidationLandmark } from "@/lib/virtual-pitch/cameraValidation";
import type { ThreeCameraBridge } from "@/lib/virtual-pitch/opencvCameraBridge";

import { sceneVector } from "./geometry";
import type { VirtualPitchCameraConfiguration, VirtualPitchRendererMode } from "./rendererTypes";


const ACTIVE_CAMERA_ERROR_TOLERANCE_PX = 1e-6;

export type ActiveCameraDiagnostics = {
  requestedMode: VirtualPitchRendererMode;
  activeCameraMode: "development" | "calibrated-opencv";
  cameraSource: string;
  cameraUuid: string;
  cameraInstanceCount: 1;
  cameraReady: boolean;
  orbitControlsMounted: boolean;
  customProjectionActive: boolean;
  matrixWorldChecksum: string;
  projectionChecksum: string;
  activeCameraRmse: number | null;
  activeCameraMaximumError: number | null;
  nativeImageWidth: number | null;
  nativeImageHeight: number | null;
  displayedMediaWidth: number;
  displayedMediaHeight: number;
  activeCameraMatchesBridge: boolean;
  points: ActiveRendererProjection[];
};

export type OwnedPitchCamera = {
  camera: OwnedPerspectiveCamera;
  family: CameraFamily;
  bridge: ThreeCameraBridge | null;
};


export function createDevelopmentCamera(configuration: VirtualPitchCameraConfiguration): OwnedPerspectiveCamera {
  const camera = new PerspectiveCamera(
    configuration.verticalFovDegrees,
    1,
    configuration.near,
    configuration.far
  ) as OwnedPerspectiveCamera;
  camera.manual = false;
  camera.position.copy(sceneVector(configuration.position));
  camera.up.copy(sceneVector(configuration.up));
  camera.lookAt(sceneVector(configuration.target));
  camera.updateProjectionMatrix();
  return camera;
}


export function createCalibratedCamera(bridge: ThreeCameraBridge): OwnedPerspectiveCamera {
  return configureCalibratedCamera(new PerspectiveCamera() as OwnedPerspectiveCamera, bridge);
}


export function useOwnedPitchCamera(
  configuration: VirtualPitchCameraConfiguration,
  bridge: ThreeCameraBridge | null
): OwnedPitchCamera {
  return useMemo(() => bridge
    ? {
        camera: createCalibratedCamera(bridge),
        family: cameraFamilyForBridge(bridge),
        bridge
      }
    : {
        camera: createDevelopmentCamera(configuration),
        family: cameraFamilyForBridge(null),
        bridge: null
      }, [bridge, configuration]);
}


export function VirtualPitchCameraController({
  ownedCamera,
  configuration,
  mode,
  landmarks,
  enableOrbitControls,
  onReadyChange,
  onDiagnostics,
  children
}: {
  ownedCamera: OwnedPitchCamera;
  configuration: VirtualPitchCameraConfiguration;
  mode: VirtualPitchRendererMode;
  landmarks: readonly CameraValidationLandmark[];
  enableOrbitControls: boolean;
  onReadyChange?: (cameraUuid: string, ready: boolean) => void;
  onDiagnostics?: (diagnostics: ActiveCameraDiagnostics) => void;
  children: ReactNode;
}) {
  const activeCamera = useThree((state) => state.camera);
  const invalidate = useThree((state) => state.invalidate);
  const size = useThree((state) => state.size);
  const [ready, setReady] = useState(false);
  const [diagnostics, setDiagnostics] = useState<ActiveCameraDiagnostics | null>(null);
  const target = useMemo(() => sceneVector(configuration.target), [configuration.target]);
  const orbitControlsMounted = shouldMountOrbitControls(ownedCamera.family, enableOrbitControls);

  useLayoutEffect(() => {
    setReady(false);
    setDiagnostics(null);
    if (activeCamera !== ownedCamera.camera) {
      throw new Error("Active renderer camera does not match the owned camera instance.");
    }

    let validation = null;
    if (ownedCamera.bridge) {
      const expectedProjection = matrixChecksum(ownedCamera.bridge.projectionMatrix.elements);
      const expectedPose = matrixChecksum(ownedCamera.bridge.matrixWorld.elements);
      if (
        matrixChecksum(activeCamera.projectionMatrix.elements) !== expectedProjection
        || matrixChecksum(activeCamera.matrixWorld.elements) !== expectedPose
      ) {
        throw new Error("Active renderer camera does not match calibrated camera.");
      }
      validation = validateActiveRendererCamera(activeCamera, ownedCamera.bridge, landmarks);
      if (validation.rmse === null || validation.rmse > ACTIVE_CAMERA_ERROR_TOLERANCE_PX) {
        throw new Error("Active renderer camera does not match calibrated camera.");
      }
    }

    const nextDiagnostics: ActiveCameraDiagnostics = {
      requestedMode: mode,
      activeCameraMode: ownedCamera.bridge ? "calibrated-opencv" : "development",
      cameraSource: ownedCamera.bridge?.input.source ?? "development-preset",
      cameraUuid: activeCamera.uuid,
      cameraInstanceCount: 1,
      cameraReady: true,
      orbitControlsMounted,
      customProjectionActive: Boolean(ownedCamera.bridge),
      matrixWorldChecksum: matrixChecksum(activeCamera.matrixWorld.elements),
      projectionChecksum: matrixChecksum(activeCamera.projectionMatrix.elements),
      activeCameraRmse: validation?.rmse ?? null,
      activeCameraMaximumError: validation?.maximumError ?? null,
      nativeImageWidth: ownedCamera.bridge?.intrinsics.imageWidth ?? null,
      nativeImageHeight: ownedCamera.bridge?.intrinsics.imageHeight ?? null,
      displayedMediaWidth: size.width,
      displayedMediaHeight: size.height,
      activeCameraMatchesBridge: ownedCamera.bridge ? true : false,
      points: validation?.points ?? []
    };
    setDiagnostics(nextDiagnostics);
    setReady(true);
    invalidate();
  }, [activeCamera, invalidate, landmarks, mode, orbitControlsMounted, ownedCamera, size.height, size.width]);

  useEffect(() => {
    if (!ready || !diagnostics) return;
    onReadyChange?.(activeCamera.uuid, true);
    onDiagnostics?.(diagnostics);
  }, [activeCamera.uuid, diagnostics, onDiagnostics, onReadyChange, ready]);

  return (
    <>
      {ready ? children : null}
      {orbitControlsMounted ? (
        <OrbitControls
          makeDefault
          camera={ownedCamera.camera}
          target={target}
          enableDamping={false}
          enablePan
          minDistance={1.5}
          maxDistance={80}
          minPolarAngle={0.08}
          maxPolarAngle={Math.PI * 0.49}
          onChange={() => invalidate()}
        />
      ) : null}
    </>
  );
}
