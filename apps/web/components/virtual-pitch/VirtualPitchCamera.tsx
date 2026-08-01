import { useLayoutEffect, useMemo } from "react";
import { OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import type { PerspectiveCamera } from "three";

import { sceneVector } from "./geometry";
import {
  buildThreeCameraFromOpenCv,
  type CameraBridgeInput,
  type ThreeCameraBridge
} from "@/lib/virtual-pitch/opencvCameraBridge";

import type { CalibratedThreeCameraConfiguration } from "./calibratedCameraTypes";
import type { VirtualPitchCameraConfiguration } from "./rendererTypes";


export function VirtualPitchCamera({
  configuration,
  calibratedCamera,
  enableOrbitControls = false
}: {
  configuration: VirtualPitchCameraConfiguration;
  calibratedCamera?: CalibratedThreeCameraConfiguration;
  enableOrbitControls?: boolean;
}) {
  const { camera, invalidate } = useThree();
  const position = useMemo(() => sceneVector(configuration.position), [configuration.position]);
  const target = useMemo(() => sceneVector(configuration.target), [configuration.target]);
  const up = useMemo(() => sceneVector(configuration.up), [configuration.up]);
  const bridge = useMemo<ThreeCameraBridge | null>(() => {
    if (!calibratedCamera) return null;
    return "projectionMatrix" in calibratedCamera
      ? calibratedCamera
      : buildThreeCameraFromOpenCv(calibratedCamera as CameraBridgeInput);
  }, [calibratedCamera]);

  if (bridge && !bridge.renderable) {
    throw new Error(bridge.diagnostics.distortion.warning ?? "The calibrated camera cannot be rendered exactly.");
  }

  useLayoutEffect(() => {
    const perspective = camera as PerspectiveCamera;
    if (bridge) {
      perspective.matrixAutoUpdate = false;
      perspective.projectionMatrix.copy(bridge.projectionMatrix);
      perspective.projectionMatrixInverse.copy(bridge.projectionMatrixInverse);
      perspective.matrixWorld.copy(bridge.matrixWorld);
      perspective.matrixWorldInverse.copy(bridge.matrixWorldInverse);
      perspective.position.setFromMatrixPosition(perspective.matrixWorld);
      perspective.near = bridge.near;
      perspective.far = bridge.far;
      perspective.matrixWorldNeedsUpdate = false;
      invalidate();

      return () => {
        perspective.matrixAutoUpdate = true;
      };
    }

    perspective.matrixAutoUpdate = true;
    perspective.position.copy(position);
    perspective.up.copy(up);
    perspective.fov = configuration.verticalFovDegrees;
    perspective.near = configuration.near;
    perspective.far = configuration.far;
    perspective.lookAt(target);
    perspective.updateProjectionMatrix();
    invalidate();
  }, [bridge, camera, configuration.far, configuration.near, configuration.verticalFovDegrees, invalidate, position, target, up]);

  return enableOrbitControls && !bridge ? (
    <OrbitControls
      makeDefault
      target={target}
      enableDamping={false}
      enablePan
      onChange={() => invalidate()}
    />
  ) : null;
}
