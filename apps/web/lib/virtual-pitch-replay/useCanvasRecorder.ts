"use client";

import { useCallback, useRef, useState } from "react";

import { REPLAY_EXPORT_HEIGHT, REPLAY_EXPORT_WIDTH } from "./types";

export type CanvasRecorderStatus =
  | "idle"
  | "preparing"
  | "recording"
  | "finalizing"
  | "complete"
  | "error";

export type CanvasRecorderState = {
  status: CanvasRecorderStatus;
  progressRatio: number | null;
  error: string | null;
  blob: Blob | null;
  isRecording: boolean;
};

export type CanvasRecorderControls = {
  startRecording: (options: {
    canvas: HTMLCanvasElement;
    durationSeconds: number;
    onFrame?: () => void;
    drivePlayback?: (controls: {
      onFrame: (progressRatio: number) => void;
    }) => Promise<void>;
  }) => Promise<Blob | null>;
  cancelRecording: () => void;
  reset: () => void;
  download: (filename?: string) => void;
};

function pickMimeType(): string | undefined {
  const candidates = [
    "video/webm;codecs=vp9",
    "video/webm;codecs=vp8",
    "video/webm"
  ];
  for (const candidate of candidates) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(candidate)) {
      return candidate;
    }
  }
  return undefined;
}

export function useCanvasRecorder(): CanvasRecorderState & CanvasRecorderControls {
  const [status, setStatus] = useState<CanvasRecorderStatus>("idle");
  const [progressRatio, setProgressRatio] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [blob, setBlob] = useState<Blob | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const cancelRef = useRef(false);

  const reset = useCallback(() => {
    cancelRef.current = true;
    recorderRef.current?.stop();
    recorderRef.current = null;
    chunksRef.current = [];
    setStatus("idle");
    setProgressRatio(null);
    setError(null);
    setBlob(null);
  }, []);

  const cancelRecording = useCallback(() => {
    cancelRef.current = true;
    recorderRef.current?.stop();
    recorderRef.current = null;
    chunksRef.current = [];
    setStatus("idle");
    setProgressRatio(null);
    setError(null);
  }, []);

  const startRecording = useCallback(async ({
    canvas,
    durationSeconds,
    onFrame,
    drivePlayback
  }: {
    canvas: HTMLCanvasElement;
    durationSeconds: number;
    onFrame?: () => void;
    drivePlayback?: (controls: {
      onFrame: (progressRatio: number) => void;
    }) => Promise<void>;
  }): Promise<Blob | null> => {
    if (typeof MediaRecorder === "undefined") {
      setStatus("error");
      setError("MediaRecorder is unavailable in this browser.");
      return null;
    }

    if (recorderRef.current) {
      setStatus("error");
      setError("An export is already in progress.");
      return null;
    }

    cancelRef.current = false;
    setStatus("preparing");
    setProgressRatio(0);
    setError(null);
    setBlob(null);
    chunksRef.current = [];

    const mimeType = pickMimeType();
    const stream = canvas.captureStream(30);
    const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    recorderRef.current = recorder;

    return new Promise((resolve) => {
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = () => {
        setStatus("error");
        setError("Canvas recording failed.");
        recorderRef.current = null;
        resolve(null);
      };

      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        recorderRef.current = null;
        if (cancelRef.current) {
          resolve(null);
          return;
        }
        const output = new Blob(chunksRef.current, { type: mimeType ?? "video/webm" });
        setBlob(output);
        setStatus("complete");
        setProgressRatio(1);
        resolve(output);
      };

      try {
        recorder.start(250);
        setStatus("recording");
      } catch (caught) {
        setStatus("error");
        setError(caught instanceof Error ? caught.message : "Could not start canvas recording.");
        resolve(null);
        return;
      }

      if (drivePlayback) {
        void drivePlayback({
          onFrame: (progressRatio) => {
            onFrame?.();
            setProgressRatio(progressRatio);
          }
        }).then(() => {
          if (cancelRef.current || !recorderRef.current) {
            resolve(null);
            return;
          }
          setStatus("finalizing");
          recorder.stop();
        });
        return;
      }

      const startedAt = performance.now();
      const totalMs = Math.max(durationSeconds, 0.1) * 1000;

      const tick = () => {
        if (cancelRef.current || !recorderRef.current) return;
        onFrame?.();
        const elapsed = performance.now() - startedAt;
        setProgressRatio(Math.min(elapsed / totalMs, 1));
        if (elapsed >= totalMs) {
          setStatus("finalizing");
          recorder.stop();
          return;
        }
        requestAnimationFrame(tick);
      };

      requestAnimationFrame(tick);
    });
  }, []);

  const download = useCallback((filename = "virtual-pitch-replay.webm") => {
    if (!blob) return;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [blob]);

  return {
    status,
    progressRatio,
    error,
    blob,
    isRecording: status === "preparing" || status === "recording" || status === "finalizing",
    startRecording,
    cancelRecording,
    reset,
    download
  };
}

export const EXPORT_WIDTH = REPLAY_EXPORT_WIDTH;
export const EXPORT_HEIGHT = REPLAY_EXPORT_HEIGHT;
