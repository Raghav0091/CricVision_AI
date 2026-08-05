import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { ReplayPlayback, ReplayTrajectorySample } from "./types";

export type ReplayClockSnapshot = {
  currentTimeSeconds: number;
  currentSampleIndex: number | null;
  durationSeconds: number;
  windowStartSeconds: number;
  windowEndSeconds: number;
  isPlaying: boolean;
  isAtEnd: boolean;
};

export type ReplayClockControls = {
  play: () => void;
  pause: () => void;
  restart: () => void;
  scrub: (timeSeconds: number) => void;
  toggle: () => void;
};

export function sortedSamples(trajectory: ReplayTrajectorySample[]): ReplayTrajectorySample[] {
  return [...trajectory].sort((left, right) => {
    if (left.timestamp_seconds !== right.timestamp_seconds) {
      return left.timestamp_seconds - right.timestamp_seconds;
    }
    return left.frame_index - right.frame_index;
  });
}

export function resolveWindow(
  trajectory: ReplayTrajectorySample[],
  playback: ReplayPlayback
): { startSeconds: number; endSeconds: number } {
  const samples = sortedSamples(trajectory);
  if (samples.length === 0) {
    const fallbackDuration = playback.duration_seconds ?? 0;
    return { startSeconds: 0, endSeconds: Math.max(fallbackDuration, 0) };
  }

  let startIndex = 0;
  let endIndex = samples.length - 1;

  if (playback.start_frame_index != null) {
    const index = samples.findIndex((sample) => sample.frame_index >= playback.start_frame_index!);
    if (index >= 0) startIndex = index;
  }
  if (playback.end_frame_index != null) {
    for (let index = samples.length - 1; index >= 0; index -= 1) {
      if (samples[index].frame_index <= playback.end_frame_index!) {
        endIndex = index;
        break;
      }
    }
  }

  if (endIndex < startIndex) {
    endIndex = startIndex;
  }

  return {
    startSeconds: samples[startIndex]?.timestamp_seconds ?? 0,
    endSeconds: samples[endIndex]?.timestamp_seconds ?? samples[startIndex]?.timestamp_seconds ?? 0
  };
}

export function findSampleIndex(samples: ReplayTrajectorySample[], timeSeconds: number): number | null {
  if (samples.length === 0) return null;
  if (timeSeconds <= samples[0].timestamp_seconds) return 0;
  const lastIndex = samples.length - 1;
  if (timeSeconds >= samples[lastIndex].timestamp_seconds) return lastIndex;

  let low = 0;
  let high = lastIndex;
  while (low <= high) {
    const mid = Math.floor((low + high) / 2);
    const current = samples[mid].timestamp_seconds;
    const next = samples[mid + 1]?.timestamp_seconds ?? Number.POSITIVE_INFINITY;
    if (timeSeconds >= current && timeSeconds < next) {
      return mid;
    }
    if (timeSeconds < current) {
      high = mid - 1;
    } else {
      low = mid + 1;
    }
  }
  return lastIndex;
}

/** Single deterministic playback loop for preview export — one rAF driver. */
export async function runDeterministicPlayback(options: {
  startSeconds: number;
  endSeconds: number;
  onTime: (timeSeconds: number, progressRatio: number) => void;
  signal?: AbortSignal;
}): Promise<void> {
  const { startSeconds, endSeconds, onTime, signal } = options;
  const duration = Math.max(endSeconds - startSeconds, 0.001);

  await new Promise<void>((resolve) => {
    const wallStart = performance.now();
    const tick = () => {
      if (signal?.aborted) {
        resolve();
        return;
      }
      const elapsed = (performance.now() - wallStart) / 1000;
      const progress = Math.min(elapsed / duration, 1);
      const timeSeconds = Math.min(startSeconds + elapsed, endSeconds);
      onTime(timeSeconds, progress);
      if (progress >= 1) {
        resolve();
        return;
      }
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  });
}

export function useReplayClock(
  trajectory: ReplayTrajectorySample[],
  playback: ReplayPlayback,
  playing: boolean,
  onPlayingChange?: (playing: boolean) => void
): ReplayClockSnapshot & ReplayClockControls {
  const samples = useMemo(() => sortedSamples(trajectory), [trajectory]);
  const window = useMemo(() => resolveWindow(samples, playback), [samples, playback]);
  const [currentTimeSeconds, setCurrentTimeSeconds] = useState(window.startSeconds);
  const anchorRef = useRef<{ wallTime: number; replayTime: number } | null>(null);
  const currentTimeRef = useRef(currentTimeSeconds);

  useEffect(() => {
    currentTimeRef.current = currentTimeSeconds;
  }, [currentTimeSeconds]);

  useEffect(() => {
    setCurrentTimeSeconds(window.startSeconds);
    anchorRef.current = null;
  }, [window.startSeconds, window.endSeconds, samples.length]);

  useEffect(() => {
    if (!playing) {
      anchorRef.current = null;
      return undefined;
    }

    anchorRef.current = {
      wallTime: performance.now(),
      replayTime: currentTimeRef.current
    };

    let frame = 0;
    const tick = () => {
      const anchor = anchorRef.current;
      if (!anchor) return;
      const elapsed = (performance.now() - anchor.wallTime) / 1000;
      const nextTime = Math.min(window.endSeconds, anchor.replayTime + elapsed);
      setCurrentTimeSeconds(nextTime);
      if (nextTime >= window.endSeconds) {
        onPlayingChange?.(false);
        anchorRef.current = null;
        return;
      }
      frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, window.endSeconds, onPlayingChange]);

  const clampTime = useCallback(
    (timeSeconds: number) => Math.max(window.startSeconds, Math.min(window.endSeconds, timeSeconds)),
    [window.endSeconds, window.startSeconds]
  );

  const scrub = useCallback(
    (timeSeconds: number) => {
      anchorRef.current = null;
      setCurrentTimeSeconds(clampTime(timeSeconds));
    },
    [clampTime]
  );

  const play = useCallback(() => {
    if (currentTimeSeconds >= window.endSeconds) {
      setCurrentTimeSeconds(window.startSeconds);
    }
    onPlayingChange?.(true);
  }, [currentTimeSeconds, onPlayingChange, window.endSeconds, window.startSeconds]);

  const pause = useCallback(() => {
    onPlayingChange?.(false);
  }, [onPlayingChange]);

  const restart = useCallback(() => {
    anchorRef.current = null;
    setCurrentTimeSeconds(window.startSeconds);
    onPlayingChange?.(false);
  }, [onPlayingChange, window.startSeconds]);

  const toggle = useCallback(() => {
    if (playing) {
      pause();
    } else {
      play();
    }
  }, [pause, play, playing]);

  const currentSampleIndex = findSampleIndex(samples, currentTimeSeconds);
  const durationSeconds = Math.max(window.endSeconds - window.startSeconds, 0);

  return {
    currentTimeSeconds,
    currentSampleIndex,
    durationSeconds,
    windowStartSeconds: window.startSeconds,
    windowEndSeconds: window.endSeconds,
    isPlaying: playing,
    isAtEnd: currentTimeSeconds >= window.endSeconds,
    play,
    pause,
    restart,
    scrub,
    toggle
  };
}
