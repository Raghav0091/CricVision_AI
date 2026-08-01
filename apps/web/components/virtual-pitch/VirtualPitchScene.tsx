import { useMemo } from "react";

import type { PitchPolygon } from "@/lib/virtual-pitch";

import {
  VirtualPitchCameraController,
  type OwnedPitchCamera
} from "./VirtualPitchCameraController";
import { VirtualPitchCorridor } from "./VirtualPitchCorridor";
import { VirtualPitchDebugHelpers } from "./VirtualPitchDebugHelpers";
import { VirtualPitchLighting } from "./VirtualPitchLighting";
import { VirtualPitchLines } from "./VirtualPitchLines";
import { VirtualPitchSurface } from "./VirtualPitchSurface";
import { VirtualWicket } from "./VirtualWicket";
import type { VirtualPitchSceneProps } from "./rendererTypes";


export function VirtualPitchScene({
  model,
  camera,
  visualOptions,
  mode = "development",
  onCameraDiagnostics,
  ownedCamera,
  onCameraReadyChange
}: VirtualPitchSceneProps & {
  ownedCamera: OwnedPitchCamera;
  onCameraReadyChange: (cameraUuid: string, ready: boolean) => void;
}) {
  const materials = visualOptions.materialPreset;
  const surface = useMemo(
    () => model.polygons.find((polygon) => polygon.polygonCategory === "pitch_surface"),
    [model.polygons]
  );
  const corridors = useMemo(
    () => model.polygons.filter((polygon) => polygon.polygonCategory === "lbw_corridor"),
    [model.polygons]
  );
  const landmarks = useMemo(
    () => model.landmarks.map((landmark) => ({ semanticId: landmark.semanticId, world: landmark.point })),
    [model.landmarks]
  );
  const debugEnabled = mode === "development";
  const transparentBackground = mode === "real-frame-overlay";

  return (
    <VirtualPitchCameraController
      ownedCamera={ownedCamera}
      configuration={camera}
      mode={mode}
      landmarks={landmarks}
      enableOrbitControls={debugEnabled && (visualOptions.enableOrbitControls ?? false)}
      onReadyChange={onCameraReadyChange}
      onDiagnostics={onCameraDiagnostics}
    >
      {!transparentBackground && <color attach="background" args={[materials.background]} />}
      <VirtualPitchLighting
        ambientColour={materials.ambientLight}
        keyColour={materials.keyLight}
        lowPerformance={visualOptions.lowPerformance}
      />
      {visualOptions.showPitch !== false && surface && (
        <VirtualPitchSurface
          polygon={surface}
          material={materials.pitch}
        />
      )}
      {(["bowler", "striker"] as const).map((wicketEnd) => (
        <VirtualWicket
          key={wicketEnd}
          wicketEnd={wicketEnd}
          stumps={model.stumps}
          bails={model.bails}
          stumpMaterial={materials.stump}
          bailMaterial={materials.bail}
          showStumps={visualOptions.showStumps !== false}
          showBails={visualOptions.showBails !== false}
        />
      ))}
      {visualOptions.showLines !== false && (
        <VirtualPitchLines
          lines={model.lineSegments}
          officialMaterial={materials.officialLine}
          analyticalMaterial={materials.analyticalLine}
        />
      )}
      {visualOptions.showCorridor !== false && corridors.map((polygon: PitchPolygon) => (
        <VirtualPitchCorridor
          key={polygon.primitiveId}
          polygon={polygon}
          material={materials.corridor}
          opacity={visualOptions.corridorOpacity ?? polygon.displayOpacity}
        />
      ))}
      {debugEnabled && (
        <VirtualPitchDebugHelpers
          model={model}
          showAxes={visualOptions.showAxes}
          showGrid={visualOptions.showGrid}
          showLandmarkLabels={visualOptions.showLandmarkLabels}
          showBounds={visualOptions.showBounds}
          gridColour={materials.analyticalLine.color}
          labelColour={materials.officialLine.color}
        />
      )}
    </VirtualPitchCameraController>
  );
}
