import { useEffect, useMemo } from "react";

import type { MaterialStyle, PitchPolygon } from "@/lib/virtual-pitch";

import { polygonGeometry } from "./geometry";


export function VirtualPitchCorridor({
  polygon,
  material,
  opacity,
}: {
  polygon: PitchPolygon;
  material: MaterialStyle;
  opacity: number;
}) {
  const geometry = useMemo(() => polygonGeometry(polygon.vertices), [polygon.vertices]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh
      name={polygon.primitiveId}
      geometry={geometry}
      position={[0, 0.001, 0]}
      renderOrder={2}
    >
      <meshBasicMaterial
        color={material.color}
        depthWrite={material.depthWrite}
        opacity={Math.max(0, Math.min(1, opacity))}
        transparent={material.transparent || opacity < 1}
        side={2}
        wireframe={material.wireframe}
      />
    </mesh>
  );
}
