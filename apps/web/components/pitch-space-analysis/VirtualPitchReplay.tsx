import { activePoint, pointsThroughFrame, provenanceStyle } from "@/lib/pitch-space-analysis/replay";
import type { PitchSpaceAnalysis } from "@/lib/pitch-space-analysis/types";
import type { VirtualPitchModel } from "@/lib/virtual-pitch/types";

export function VirtualPitchReplay({ analysis, pitch, currentFrame, fullTrail }: {
  analysis: PitchSpaceAnalysis;
  pitch: VirtualPitchModel;
  currentFrame: number;
  fullTrail: boolean;
}) {
  const width = pitch.dimensions.pitchWidthM;
  const length = pitch.dimensions.pitchLengthM;
  const pad = Math.max(0.7, width * 0.55);
  const all = analysis.pitch_space_track ?? [];
  const points = fullTrail ? all : pointsThroughFrame(all, currentFrame);
  const active = activePoint(all, currentFrame);
  const x = (value: number) => ((value + width / 2 + pad) / (width + pad * 2)) * 300;
  const y = (value: number) => 540 - ((value + pad) / (length + pad * 2)) * 500;
  const pop = pitch.dimensions.poppingCreaseOffsetM;
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-bold">Canonical pitch</h2>
        <span className="text-[11px] uppercase text-white/40">Top-down metres</span>
      </div>
      <div className="aspect-video overflow-hidden rounded-md border border-white/10 bg-[#111713]">
        <svg aria-label="Top-down pitch-space replay" viewBox="0 0 300 560" className="h-full w-full">
          <rect x={x(-width / 2)} y={y(length)} width={x(width / 2) - x(-width / 2)} height={y(0) - y(length)} fill="#788061" stroke="#dfe8c6" strokeWidth="1.5" />
          {[0, pop, length - pop, length].map((value) => <line key={value} x1={x(-width / 2 - .35)} x2={x(width / 2 + .35)} y1={y(value)} y2={y(value)} stroke="#f4f3df" strokeWidth={value === 0 || value === length ? 2 : 1.2} />)}
          <line x1={x(0)} x2={x(0)} y1={y(0)} y2={y(length)} stroke="#dfe8c6" strokeDasharray="5 5" opacity=".45" />
          {points.slice(1).map((point, index) => {
            const previous = points[index];
            const style = provenanceStyle(point.provenance);
            return <line key={`${point.frame_index}-${index}`} x1={x(previous.pitch_x_m)} y1={y(previous.pitch_y_m)} x2={x(point.pitch_x_m)} y2={y(point.pitch_y_m)} stroke={style.color} strokeWidth="3" strokeDasharray={style.dash} opacity=".82" />;
          })}
          {analysis.bounce?.pitch_x_m != null && analysis.bounce.pitch_y_m != null && <circle cx={x(analysis.bounce.pitch_x_m)} cy={y(analysis.bounce.pitch_y_m)} r="7" fill="none" stroke="#ff6b6b" strokeWidth="3" />}
          {active && <circle cx={x(active.pitch_x_m)} cy={y(active.pitch_y_m)} r="5" fill={provenanceStyle(active.provenance).color} stroke="#0a0d0b" strokeWidth="2" />}
          <text x="12" y="25" fill="#ffffff80" fontSize="10">Striker end</text>
          <text x="12" y="548" fill="#ffffff80" fontSize="10">Bowler end</text>
        </svg>
      </div>
    </section>
  );
}
