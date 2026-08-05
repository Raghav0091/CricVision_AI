import { useMemo } from "react";
import * as THREE from "three";

import { worldPointToTuple } from "@/lib/virtual-pitch-replay/replayCoordinates";
import type { WorldPoint3D } from "@/lib/virtual-pitch-replay/types";

export function VirtualBall({ position }: { position: WorldPoint3D }) {
  const [x, y, z] = useMemo(() => worldPointToTuple(position), [position]);

  return (
    <mesh position={[x, y, z]} castShadow={false} receiveShadow={false}>
      <sphereGeometry args={[0.036, 24, 24]} />
      <meshStandardMaterial color="#ff4d4d" emissive="#661111" emissiveIntensity={0.35} roughness={0.35} />
    </mesh>
  );
}

export function VirtualBounceMarker({ position }: { position: WorldPoint3D }) {
  const [x, y, z] = useMemo(() => worldPointToTuple(position), [position]);

  return (
    <group position={[x, y, z + 0.01]}>
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.08, 0.12, 32]} />
        <meshBasicMaterial color="#ffe761" transparent opacity={0.95} side={THREE.DoubleSide} />
      </mesh>
      <mesh position={[0, 0, 0.02]}>
        <sphereGeometry args={[0.018, 16, 16]} />
        <meshStandardMaterial color="#ffe761" emissive="#665500" emissiveIntensity={0.4} />
      </mesh>
    </group>
  );
}
