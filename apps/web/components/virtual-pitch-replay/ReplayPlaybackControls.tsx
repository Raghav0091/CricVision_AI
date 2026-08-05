"use client";

import { Button } from "@/components/ui/Button";

export function ReplayPlaybackControls({
  isPlaying,
  currentTimeSeconds,
  durationSeconds,
  windowStartSeconds,
  disabled,
  exporting,
  onToggle,
  onRestart,
  onScrub,
  onExport,
  exportStatus,
  exportProgress
}: {
  isPlaying: boolean;
  currentTimeSeconds: number;
  durationSeconds: number;
  windowStartSeconds: number;
  disabled?: boolean;
  exporting?: boolean;
  onToggle: () => void;
  onRestart: () => void;
  onScrub: (timeSeconds: number) => void;
  onExport?: () => void;
  exportStatus?: string;
  exportProgress?: number | null;
}) {
  const relativeTime = Math.max(currentTimeSeconds - windowStartSeconds, 0);
  const sliderMax = Math.max(durationSeconds, 0.001);

  return (
    <div className="space-y-3 rounded-xl border border-white/10 bg-black/30 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="secondary"
          className="px-3 py-2 text-xs"
          disabled={disabled || exporting}
          onClick={onToggle}
        >
          {isPlaying ? "Pause" : "Play"}
        </Button>
        <Button
          type="button"
          variant="secondary"
          className="px-3 py-2 text-xs"
          disabled={disabled || exporting}
          onClick={onRestart}
        >
          Restart
        </Button>
        {onExport ? (
          <Button
            type="button"
            className="px-3 py-2 text-xs"
            disabled={disabled || exporting}
            onClick={onExport}
          >
            {exporting ? "Exporting…" : "Download WebM"}
          </Button>
        ) : null}
        <span className="ml-auto tabular-nums text-xs text-white/55">
          {relativeTime.toFixed(2)}s / {durationSeconds.toFixed(2)}s
        </span>
      </div>

      <label className="block text-xs text-white/55">
        Scrub
        <input
          className="mt-2 h-1.5 w-full accent-lime disabled:opacity-40"
          type="range"
          min={0}
          max={sliderMax}
          step={0.001}
          value={relativeTime}
          disabled={disabled || exporting}
          onChange={(event) => onScrub(windowStartSeconds + Number(event.target.value))}
        />
      </label>

      {exportStatus ? (
        <p className="text-xs text-white/55">
          {exportStatus}
          {exportProgress != null ? ` (${Math.round(exportProgress * 100)}%)` : ""}
        </p>
      ) : null}
    </div>
  );
}
