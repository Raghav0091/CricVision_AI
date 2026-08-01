import { useMemo } from "react";

import type { BailModel, MaterialStyle } from "@/lib/virtual-pitch";

import { asThreeVector, cylinderTransform } from "./geometry";


export function VirtualBail({
  bail,
  material
}: {
  bail: BailModel;
  material: MaterialStyle;
}) {
  const transform = useMemo(
    () => cylinderTransform(asThreeVector(bail.start), asThreeVector(bail.endPoint)),
    [bail]
  );
  if (transform.length <= Number.EPSILON) return null;
  return (
    <mesh
      name={bail.primitiveId}
      position={transform.midpoint}
      quaternion={transform.quaternion}
    >
      <cylinderGeometry args={[bail.radiusM, bail.radiusM, transform.length, 10]} />
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


export function VirtualBails({
  bails,
  material
}: {
  bails: readonly BailModel[];
  material: MaterialStyle;
}) {
  return (
    <group name="virtual-pitch-bails">
      {bails.map((bail) => (
        <VirtualBail key={bail.primitiveId} bail={bail} material={material} />
      ))}
    </group>
  );
}
