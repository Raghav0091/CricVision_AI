import { Html } from "@react-three/drei";
import { useMemo } from "react";
import * as THREE from "three";

import type { VirtualPitchModel } from "@/lib/virtual-pitch";

import { asThreeVector } from "./geometry";


export function VirtualPitchDebugHelpers({
  model,
  showAxes = false,
  showGrid = false,
  showLandmarkLabels = false,
  showBounds = false,
  gridColour = "#6b7280",
  labelColour = "#f8fafc"
}: {
  model: VirtualPitchModel;
  showAxes?: boolean;
  showGrid?: boolean;
  showLandmarkLabels?: boolean;
  showBounds?: boolean;
  gridColour?: string;
  labelColour?: string;
}) {
  const gridSize = Math.max(model.dimensions.pitchLengthM, model.dimensions.pitchWidthM);
  const bounds = useMemo(() => {
    const points = [
      ...model.polygons.flatMap((polygon) => polygon.vertices),
      ...model.stumps.map((stump) => stump.centre),
      ...model.bails.flatMap((bail) => [bail.start, bail.endPoint])
    ].map(asThreeVector);
    return new THREE.Box3().setFromPoints(points);
  }, [model]);
  const boundsHelper = useMemo(
    () => new THREE.Box3Helper(bounds, gridColour),
    [bounds, gridColour]
  );

  return (
    <group name="virtual-pitch-debug-helpers">
      {showAxes && <axesHelper args={[gridSize * 0.2]} />}
      {showGrid && (
        <gridHelper
          args={[gridSize, 20, gridColour, gridColour]}
          position={[0, -0.004, -model.dimensions.pitchLengthM / 2]}
        />
      )}
      {showBounds && <primitive object={boundsHelper} />}
      {showLandmarkLabels && model.landmarks.map((landmark) => (
        <Html
          key={landmark.semanticId}
          position={asThreeVector(landmark.point)}
          center
          distanceFactor={8}
          style={{
            color: labelColour,
            fontSize: "10px",
            lineHeight: 1,
            pointerEvents: "none",
            whiteSpace: "nowrap"
          }}
        >
          {landmark.semanticId}
        </Html>
      ))}
    </group>
  );
}
