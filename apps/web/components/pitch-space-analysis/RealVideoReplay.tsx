"use client";

import { forwardRef } from "react";

import { activePoint, pointsThroughFrame, provenanceStyle } from "@/lib/pitch-space-analysis/replay";
import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";

export const RealVideoReplay = forwardRef<HTMLVideoElement, {
  analysis: PitchSpaceAnalysis;
  videoUrl: string;
  currentFrame: number;
  fullTrail: boolean;
  onTime: (seconds: number) => void;
  onEnded: () => void;
}>(function RealVideoReplay({ analysis, videoUrl, currentFrame, fullTrail, onTime, onEnded }, ref) {
  const width = analysis.native_width ?? 1280;
  const height = analysis.native_height ?? 720;
  const all = analysis.pitch_space_track ?? [];
  const points = fullTrail ? all : pointsThroughFrame(all, currentFrame);
  const active = activePoint(all, currentFrame);
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold">Original video</h2>
        <span className="text-[11px] uppercase text-white/40">Image-space overlay</span>
      </div>
      <div className="relative aspect-video overflow-hidden rounded-md border border-white/10 bg-black">
        <video
          ref={ref}
          src={videoUrl}
          preload="metadata"
          playsInline
          onTimeUpdate={(event) => onTime(event.currentTarget.currentTime)}
          onEnded={onEnded}
          className="h-full w-full object-contain"
        />
        <svg aria-label="Ball trail over original video" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid meet" className="pointer-events-none absolute inset-0 h-full w-full">
          {analysis.pitch_fit?.projected_pitch?.map((primitive, index) => {
            const coordinates = primitive.image_points?.map((point) => `${point.x},${point.y}`).join(" ") ?? "";
            if (!coordinates) return null;
            return primitive.primitive_type === "POLYGON"
              ? <polygon key={primitive.primitive_id ?? index} points={coordinates} fill="#d7ff7b12" stroke="#d7ff7b" strokeWidth="1.5" opacity=".6" />
              : <polyline key={primitive.primitive_id ?? index} points={coordinates} fill="none" stroke="#d7ff7b" strokeWidth="1.5" opacity=".62" />;
          })}
          {points.slice(1).map((point, index) => {
            const previous = points[index];
            const style = provenanceStyle(point.provenance);
            return <line key={`${point.frame_index}-${index}`} x1={previous.image_x_px} y1={previous.image_y_px} x2={point.image_x_px} y2={point.image_y_px} stroke={style.color} strokeWidth="3" strokeDasharray={style.dash} opacity=".78" />;
          })}
          {analysis.bounce?.pitch_x_m != null && analysis.bounce.bounce_frame != null && (() => {
            const bounce = all.find((point) => point.frame_index === analysis.bounce?.bounce_frame);
            return bounce ? <circle cx={bounce.image_x_px} cy={bounce.image_y_px} r="11" fill="none" stroke="#ff6b6b" strokeWidth="4" /> : null;
          })()}
          {active && <circle cx={active.image_x_px} cy={active.image_y_px} r="7" fill={provenanceStyle(active.provenance).color} stroke="#0a0d0b" strokeWidth="3" />}
        </svg>
      </div>
    </section>
  );
});
