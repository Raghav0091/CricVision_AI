import { useMemo } from "react";

import type { MaterialStyle, PitchLineSegment } from "@/lib/virtual-pitch";

import { asThreeVector, stripTransform } from "./geometry";


const LINE_LIFT_M = 0.003;
const LINE_HEIGHT_M = 0.002;


function VirtualPitchLine({
  line,
  material
}: {
  line: PitchLineSegment;
  material: MaterialStyle;
}) {
  const transform = useMemo(() => {
    const start = asThreeVector(line.start);
    const end = asThreeVector(line.endPoint);
    start.y += LINE_LIFT_M;
    end.y += LINE_LIFT_M;
    return stripTransform(start, end);
  }, [line]);
  if (transform.length <= Number.EPSILON) return null;
  return (
    <mesh
      name={line.primitiveId}
      position={transform.midpoint}
      quaternion={transform.quaternion}
    >
      <boxGeometry args={[transform.length, LINE_HEIGHT_M, line.lineWidthM]} />
      <meshStandardMaterial
        color={material.color}
        depthWrite={material.depthWrite}
        metalness={material.metalness}
        opacity={material.opacity}
        roughness={material.roughness}
        transparent={material.transparent}
        wireframe={material.wireframe}
      />
    </mesh>
  );
}


export function VirtualPitchLines({
  lines,
  officialMaterial,
  analyticalMaterial
}: {
  lines: readonly PitchLineSegment[];
  officialMaterial: MaterialStyle;
  analyticalMaterial: MaterialStyle;
}) {
  return (
    <group name="virtual-pitch-lines">
      {lines.map((line) => (
        <VirtualPitchLine
          key={line.primitiveId}
          line={line}
          material={line.geometryClass === "official" ? officialMaterial : analyticalMaterial}
        />
      ))}
    </group>
  );
}
