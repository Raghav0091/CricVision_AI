"use client";

import { useEffect, useRef, useState } from "react";

import type { ReplayPayloadV1 } from "@/lib/api";
import type { MeasurementValidity, ReplayMetric } from "@/lib/virtual-pitch-replay/types";
import { describePitchGeometry } from "@/lib/pitchGeometry";
import type { CricketPitchGeometry } from "@/lib/wicketCalibration/types";
import { Button } from "@/components/ui/Button";


type ChipSpec = {
  label: string;
  metric: ReplayMetric;
  /** Metric units are SI; the chip shows what a bowler expects to read. */
  format: (value: number) => string;
};


// ReplayPayloadV1 deliberately keeps `value` null unless a metric was really
// measured, so the chip renders an em dash and carries the backend's own
// reason rather than inventing a zero.
function MetricChip({ label, metric, format }: ChipSpec) {
  const available = metric.status === "AVAILABLE" && metric.value !== null;
  return (
    <div
      className="rounded-lg border border-white/15 bg-black/55 px-3 py-2 backdrop-blur-sm"
      title={available ? (metric.method ?? undefined) : (metric.unavailable_reason ?? "Not measured for this delivery.")}
    >
      <p className="text-[10px] font-bold uppercase tracking-wide text-white/60">{label}</p>
      <p className={`mt-0.5 text-lg font-black tabular-nums ${available ? "text-white" : "text-white/35"}`}>
        {available ? format(metric.value as number) : "—"}
      </p>
    </div>
  );
}


const VALIDITY_NOTE: Record<MeasurementValidity, string | null> = {
  CALIBRATED: null,
  IMAGE_SPACE_ONLY: "Camera pose was not metric for this delivery, so distances and speeds are unavailable.",
  VISUALIZATION_ONLY: "This replay is illustrative only — no calibrated measurements were produced.",
  INSUFFICIENT_EVIDENCE: "Not enough of the delivery was tracked to measure it."
};


export function DeliveryReview({
  clipUrl,
  replay,
  onRetake,
  onClose,
  pitchGeometry
}: {
  clipUrl: string;
  replay: ReplayPayloadV1;
  onRetake: () => void;
  onClose: () => void;
  /** The pitch this delivery was measured on. Null means regulation. */
  pitchGeometry?: CricketPitchGeometry | null;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(true);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    video.addEventListener("play", onPlay);
    video.addEventListener("pause", onPause);
    return () => {
      video.removeEventListener("play", onPlay);
      video.removeEventListener("pause", onPause);
    };
  }, []);

  function replayClip() {
    const video = videoRef.current;
    if (!video) return;
    video.currentTime = 0;
    void video.play();
  }

  const { metrics, bounce, trajectory, measurement_validity: validity } = replay;
  const note = VALIDITY_NOTE[validity];
  const trackedSamples = trajectory.length;
  // Prefer what the payload says the pose was solved against; fall back to
  // what this screen declared. A short rig reporting 14 km/h has to read as a
  // rig number, never as a net number.
  const declaredPitch = describePitchGeometry(
    (replay.pitch_geometry as CricketPitchGeometry | null | undefined) ?? pitchGeometry ?? null
  );

  return (
    <div className="relative h-full w-full overflow-hidden bg-black">
      <video
        ref={videoRef}
        src={clipUrl}
        className="h-full w-full object-contain"
        autoPlay
        loop
        muted
        playsInline
        controls={false}
      />

      <button
        type="button"
        onClick={onClose}
        aria-label="Close delivery"
        className="absolute right-4 top-4 grid h-8 w-8 place-items-center rounded-full bg-white/85 text-lg font-bold leading-none text-ink"
      >
        ×
      </button>

      <div className="absolute left-4 top-4 flex w-32 flex-col gap-2">
        <MetricChip label="Speed" metric={metrics.release_speed_kmh} format={(v) => `${Math.round(v)} km/h`} />
        <MetricChip
          label="Swing"
          metric={metrics.estimated_lateral_deviation_m}
          format={(v) => `${(v * 100).toFixed(1)} cm`}
        />
        <MetricChip label="Length" metric={metrics.delivery_length_m} format={(v) => `${v.toFixed(2)} m`} />
      </div>

      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/85 to-transparent px-4 pb-6 pt-10">
        {declaredPitch && (
          <p className="mb-3 inline-flex rounded-md border border-[#ffcf4d]/40 bg-[#ffcf4d]/10 px-2 py-1 text-[11px] font-semibold text-[#ffcf4d]">
            Custom rig · {declaredPitch} — not a full-length pitch
          </p>
        )}
        {note && <p className="mb-3 text-xs leading-5 text-[#ffcf4d]">{note}</p>}
        <p className="text-[11px] text-white/45">
          {trackedSamples} tracked {trackedSamples === 1 ? "sample" : "samples"}
          {bounce.detected === true && bounce.timestamp_seconds !== null && bounce.timestamp_seconds !== undefined
            ? ` · bounce at ${bounce.timestamp_seconds.toFixed(2)}s`
            : bounce.detected === false
              ? " · no bounce detected"
              : ""}
        </p>
        <div className="mt-4 flex items-center gap-3">
          <Button variant="secondary" onClick={replayClip}>
            {playing ? "Restart" : "Play"}
          </Button>
          <Button onClick={onRetake}>Bowl again</Button>
        </div>
      </div>
    </div>
  );
}
