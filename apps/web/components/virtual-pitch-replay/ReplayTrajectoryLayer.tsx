import { useMemo } from "react";

import { canRenderWorldTrajectory } from "@/lib/virtual-pitch-replay/validatePayload";
import type { ReplayPayloadV1 } from "@/lib/virtual-pitch-replay/types";

import { VirtualBall, VirtualBounceMarker } from "./VirtualBall";
import { VirtualBallTrail } from "./VirtualBallTrail";

export function ReplayTrajectoryLayer({
  payload,
  currentSampleIndex
}: {
  payload: ReplayPayloadV1;
  currentSampleIndex: number | null;
  invalidate?: () => void;
}) {
  const samples = useMemo(
    () => [...payload.trajectory].sort((left, right) => left.timestamp_seconds - right.timestamp_seconds),
    [payload.trajectory]
  );

  if (!canRenderWorldTrajectory(payload)) {
    return null;
  }

  const visibleCount = currentSampleIndex == null ? 0 : currentSampleIndex + 1;
  const currentSample = currentSampleIndex == null ? null : samples[currentSampleIndex];
  const bouncePosition = payload.bounce.world_position;

  return (
    <group>
      <VirtualBallTrail samples={samples} visibleCount={visibleCount} />
      {currentSample?.world_position ? (
        <VirtualBall position={currentSample.world_position} />
      ) : null}
      {bouncePosition && payload.bounce.detected ? (
        <VirtualBounceMarker position={bouncePosition} />
      ) : null}
    </group>
  );
}
