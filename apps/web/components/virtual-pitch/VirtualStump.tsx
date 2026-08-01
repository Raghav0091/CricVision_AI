import { useMemo } from "react";
import * as THREE from "three";

import type { MaterialStyle, StumpModel } from "@/lib/virtual-pitch";

import { asThreeVector, cylinderTransform } from "./geometry";


interface VirtualStumpProps {
  stump: StumpModel;
  material: MaterialStyle;
}


export function VirtualStump({ stump, material }: VirtualStumpProps) {
  const transform = useMemo(() => {
    const centre = asThreeVector(stump.centre);
    const orientation = asThreeVector(stump.orientation).normalize();
    const halfAxis = orientation.multiplyScalar(stump.heightM / 2);
    return cylinderTransform(
      centre.clone().sub(halfAxis),
      centre.clone().add(halfAxis)
    );
  }, [stump]);

  if (transform.length <= Number.EPSILON) return null;
  return (
    <mesh
      name={stump.primitiveId}
      position={transform.midpoint}
      quaternion={transform.quaternion}
    >
      <cylinderGeometry args={[stump.radiusM, stump.radiusM, transform.length, 12]} />
      <meshStandardMaterial
        color={material.color}
        metalness={material.metalness}
        opacity={material.opacity}
        roughness={material.roughness}
        transparent={material.transparent}
        wireframe={material.wireframe}
      />
    </mesh>
  );
}
