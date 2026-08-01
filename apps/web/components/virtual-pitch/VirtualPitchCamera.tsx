import { useLayoutEffect, useMemo } from "react";
import { OrbitControls } from "@react-three/drei";
import { useThree } from "@react-three/fiber";
import type { PerspectiveCamera } from "three";

import { sceneVector } from "./geometry";
import type { VirtualPitchCameraConfiguration } from "./rendererTypes";


export function VirtualPitchCamera({
  configuration,
  enableOrbitControls = false
}: {
  configuration: VirtualPitchCameraConfiguration;
  enableOrbitControls?: boolean;
}) {
  const { camera, invalidate } = useThree();
  const position = useMemo(() => sceneVector(configuration.position), [configuration.position]);
  const target = useMemo(() => sceneVector(configuration.target), [configuration.target]);
  const up = useMemo(() => sceneVector(configuration.up), [configuration.up]);

  useLayoutEffect(() => {
    const perspective = camera as PerspectiveCamera;
    perspective.position.copy(position);
    perspective.up.copy(up);
    perspective.fov = configuration.verticalFovDegrees;
    perspective.near = configuration.near;
    perspective.far = configuration.far;
    perspective.lookAt(target);
    perspective.updateProjectionMatrix();
    invalidate();
  }, [camera, configuration.far, configuration.near, configuration.verticalFovDegrees, invalidate, position, target, up]);

  return enableOrbitControls ? (
    <OrbitControls
      makeDefault
      target={target}
      enableDamping={false}
      enablePan
      onChange={() => invalidate()}
    />
  ) : null;
}
