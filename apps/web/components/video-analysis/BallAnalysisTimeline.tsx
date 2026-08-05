"use client";

import type { VideoBallTrackingPoint } from "@/lib/api";

type TimelineMarker = {
  frame_index: number;
  kind: "candidate" | "accepted" | "rejected" | "track_start" | "track_end" | "gap";
};

export function BallAnalysisTimeline({
  totalFrames,
  fps,
  currentFrame,
  currentTimeSeconds,
  markers,
  onSeekFrame
}: {
  totalFrames: number;
  fps: number;
  currentFrame: number;
  currentTimeSeconds: number;
  markers: TimelineMarker[];
  onSeekFrame: (frameIndex: number) => void;
}) {
  if (totalFrames <= 0) return null;
  const playhead = Math.min(100, Math.max(0, (currentFrame / Math.max(1, totalFrames - 1)) * 100));

  return (
    <div className="border-t border-white/10 bg-[#070d0a]/95 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs text-white/55">
        <span>Frame {currentFrame} / {Math.max(0, totalFrames - 1)}</span>
        <span>{currentTimeSeconds.toFixed(2)}s @ {fps.toFixed(2)} FPS</span>
      </div>
      <div className="relative mt-2 h-8 rounded bg-black/40">
        {markers.map((marker) => {
          const left = (marker.frame_index / Math.max(1, totalFrames - 1)) * 100;
          const color = markerColor(marker.kind);
          return (
            <button
              key={`${marker.kind}-${marker.frame_index}`}
              type="button"
              aria-label={`${marker.kind} at frame ${marker.frame_index}`}
              className="absolute top-1/2 h-2.5 w-1 -translate-x-1/2 -translate-y-1/2 rounded-sm"
              style={{ left: `${left}%`, backgroundColor: color }}
              onClick={() => onSeekFrame(marker.frame_index)}
            />
          );
        })}
        <div
          className="pointer-events-none absolute inset-y-0 w-0.5 bg-lime"
          style={{ left: `${playhead}%` }}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-3 text-[10px] text-white/40">
        <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-lime" />Track span</span>
        <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-[#78e08f]" />Accepted</span>
        <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-[#ff8a7a]" />Rejected</span>
        <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-[#ffc568]" />Candidate</span>
        <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-white/25" />Gap</span>
      </div>
    </div>
  );
}

function markerColor(kind: TimelineMarker["kind"]) {
  switch (kind) {
    case "track_start":
    case "track_end":
      return "#d7ff7b";
    case "accepted":
      return "#78e08f";
    case "rejected":
      return "#ff8a7a";
    case "gap":
      return "rgba(255,255,255,0.25)";
    default:
      return "#ffc568";
  }
}

export function buildTimelineMarkers(
  track: VideoBallTrackingPoint[],
  candidateFrames: number[],
  acceptedFrames: number[],
  rejectedFrames: number[]
): TimelineMarker[] {
  const markers: TimelineMarker[] = [];
  for (const frame of candidateFrames) {
    markers.push({ frame_index: frame, kind: "candidate" });
  }
  for (const frame of acceptedFrames) {
    markers.push({ frame_index: frame, kind: "accepted" });
  }
  for (const frame of rejectedFrames) {
    markers.push({ frame_index: frame, kind: "rejected" });
  }
  if (track.length > 0) {
    markers.push({ frame_index: track[0].frame_index, kind: "track_start" });
    markers.push({ frame_index: track[track.length - 1].frame_index, kind: "track_end" });
    for (let index = 1; index < track.length; index += 1) {
      const gap = track[index].frame_index - track[index - 1].frame_index;
      if (gap > 1) {
        markers.push({
          frame_index: track[index - 1].frame_index + 1,
          kind: "gap"
        });
      }
    }
  }
  return markers;
}
