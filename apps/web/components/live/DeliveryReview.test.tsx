import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

import type { ReplayPayloadV1 } from "@/lib/virtual-pitch-replay/types";
import { REGULATION_PITCH, normalizePitchGeometry } from "@/lib/pitchGeometry";
import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";

import { DeliveryReview } from "./DeliveryReview";


function assert(condition: boolean, message: string) {
  if (!condition) throw new Error(message);
}


function metric(value: number | null) {
  return {
    value,
    unit: "km/h",
    status: value === null ? ("UNAVAILABLE" as const) : ("AVAILABLE" as const),
    method: null,
    unavailable_reason: value === null ? "not measured" : null
  };
}


// A calibrated replay reporting a real speed is the case that matters: a rig
// number and a net number look identical without the label.
function replayPayload(
  pitchGeometry: CricketPitchGeometry | null
): ReplayPayloadV1 {
  return {
    schema_version: "1.0",
    analysis_id: "analysis_20260811_000000_aaaaaa",
    coordinate_system: "CRICVISION_PITCH_V1",
    distance_unit: "metre",
    time_unit: "second",
    measurement_validity: "CALIBRATED",
    pitch_geometry: pitchGeometry,
    camera: { source: "UNAVAILABLE", visualization_only: false },
    playback: {},
    trajectory: [],
    bounce: { detected: false, timestamp_seconds: null },
    metrics: {
      release_speed_kmh: metric(14),
      average_pre_bounce_speed_kmh: metric(null),
      speed_at_bounce_kmh: metric(null),
      overall_stump_to_stump_speed_kmh: metric(null),
      delivery_length_m: metric(null),
      estimated_lateral_deviation_m: metric(null)
    },
    diagnostics: {}
  } as unknown as ReplayPayloadV1;
}


function render(pitchGeometry: CricketPitchGeometry | null): string {
  return renderToStaticMarkup(
    <DeliveryReview
      clipUrl="blob:delivery"
      replay={replayPayload(pitchGeometry)}
      onRetake={() => {}}
      onClose={() => {}}
    />
  );
}


// A declared 3.2 m rig reporting 14 km/h must be obviously a rig reading.
const rigMarkup = render(
  normalizePitchGeometry({ pitch_length_m: 3.2, wicket_height_m: 0.4 })
);
assert(
  rigMarkup.includes("3.20 m pitch"),
  "DeliveryReview must render the declared pitch length."
);
assert(
  rigMarkup.includes("40 cm stumps"),
  "DeliveryReview must render declared stump height when it is not regulation."
);
assert(
  rigMarkup.includes("Custom rig"),
  "DeliveryReview must mark a declared rig as a rig."
);

// Regulation must look exactly as it always has: no rig banner at all.
for (const regulation of [null, REGULATION_PITCH]) {
  const markup = render(regulation);
  assert(
    !markup.includes("Custom rig"),
    "A regulation pitch must not be labelled as a custom rig."
  );
  assert(
    !markup.includes("m pitch"),
    "A regulation pitch must not render a declared-length banner."
  );
}

console.log("DeliveryReview.test.tsx passed");
