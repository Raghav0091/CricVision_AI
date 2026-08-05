"use client";

export function PitchSpaceTimeline({ frame, frameCount, fps, playing, speed, fullTrail, onPlay, onFrame, onSpeed, onTrail }: {
  frame: number;
  frameCount: number;
  fps: number;
  playing: boolean;
  speed: number;
  fullTrail: boolean;
  onPlay: () => void;
  onFrame: (frame: number) => void;
  onSpeed: (speed: number) => void;
  onTrail: (full: boolean) => void;
}) {
  const duration = frameCount > 0 && fps > 0 ? frameCount / fps : 0;
  return (
    <div className="border-y border-white/10 py-3">
      <input aria-label="Replay frame" type="range" min={0} max={Math.max(0, frameCount - 1)} value={Math.min(frame, Math.max(0, frameCount - 1))} onChange={(event) => onFrame(Number(event.target.value))} className="w-full accent-lime" />
      <div className="mt-2 flex flex-wrap items-center gap-2">
        <button type="button" aria-label={playing ? "Pause replay" : "Play replay"} onClick={onPlay} className="grid h-9 w-9 place-items-center rounded-md border border-white/15 bg-white/5 text-xs font-bold hover:bg-white/10">{playing ? "II" : ">"}</button>
        <span className="min-w-[9rem] text-xs tabular-nums text-white/65">Frame {frame.toLocaleString()} / {Math.max(0, frameCount - 1).toLocaleString()} / {(frame / Math.max(fps, 1)).toFixed(2)}s / {duration.toFixed(2)}s</span>
        <div className="ml-auto flex items-center gap-1 rounded-md border border-white/10 p-1" aria-label="Playback speed">
          {[0.5, 1, 1.5, 2].map((value) => <button key={value} type="button" onClick={() => onSpeed(value)} className={`rounded px-2 py-1 text-xs font-bold ${speed === value ? "bg-lime text-ink" : "text-white/55 hover:text-white"}`}>{value}x</button>)}
        </div>
        <label className="flex items-center gap-2 text-xs text-white/60">
          <input type="checkbox" checked={fullTrail} onChange={(event) => onTrail(event.target.checked)} className="accent-lime" /> Complete trail
        </label>
      </div>
    </div>
  );
}
