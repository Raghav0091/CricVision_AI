import { useEffect, useMemo } from "react";

import type { MaterialStyle, PitchPolygon } from "@/lib/virtual-pitch";

import { polygonGeometry } from "./geometry";


export function VirtualPitchSurface({
  polygon,
  material
}: {
  polygon: PitchPolygon;
  material: MaterialStyle;
}) {
  const geometry = useMemo(() => polygonGeometry(polygon.vertices), [polygon.vertices]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh name={polygon.primitiveId} geometry={geometry} receiveShadow={false}>
      <meshStandardMaterial
        color={material.color}
        depthWrite={material.depthWrite}
        roughness={material.roughness}
        metalness={material.metalness}
        opacity={material.opacity}
        transparent={material.transparent}
        side={2}
        wireframe={material.wireframe}
      />
    </mesh>
  );
}
