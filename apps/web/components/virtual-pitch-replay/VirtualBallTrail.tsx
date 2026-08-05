import { Line } from "@react-three/drei";
import { useMemo } from "react";

import { provenanceColor } from "@/lib/virtual-pitch-replay/provenanceColors";
import { worldPointToTuple } from "@/lib/virtual-pitch-replay/replayCoordinates";
import type { ReplayTrajectorySample } from "@/lib/virtual-pitch-replay/types";

type Segment = {
  key: string;
  points: [readonly [number, number, number], readonly [number, number, number]];
  color: string;
};

function buildSegments(
  samples: ReplayTrajectorySample[],
  visibleCount: number
): Segment[] {
  const segments: Segment[] = [];
  const limit = Math.min(visibleCount, samples.length);
  if (limit < 2) return segments;

  for (let index = 1; index < limit; index += 1) {
    const previous = samples[index - 1];
    const current = samples[index];
    if (!previous.world_position || !current.world_position) continue;
    segments.push({
      key: `${previous.frame_index}-${current.frame_index}`,
      points: [
        worldPointToTuple(previous.world_position),
        worldPointToTuple(current.world_position)
      ] as Segment["points"],
      color: provenanceColor(current.provenance)
    });
  }
  return segments;
}

export function VirtualBallTrail({
  samples,
  visibleCount
}: {
  samples: ReplayTrajectorySample[];
  visibleCount: number;
}) {
  const segments = useMemo(
    () => buildSegments(samples, visibleCount),
    [samples, visibleCount]
  );

  return (
    <group>
      {segments.map((segment) => (
        <Line
          key={segment.key}
          points={segment.points}
          color={segment.color}
          lineWidth={2.5}
          transparent
          opacity={0.92}
        />
      ))}
    </group>
  );
}
