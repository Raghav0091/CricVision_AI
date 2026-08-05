"use client";

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  completeExperimentalSession,
  createExperimentalSession,
  detectBallInDeliveryClip
} from "@/lib/api";


type DetectionState =
  | "WAITING_FOR_DELIVERY"
  | "MOTION_CANDIDATE"
  | "DELIVERY_DETECTED"
  | "RECORDING_DELIVERY"
  | "DELIVERY_SAVED"
  | "COOLDOWN"
  | "OVER_COMPLETE";
type Sensitivity = "Low" | "Medium" | "High";
type AnalysisStatus = "not_started" | "uploading" | "processing" | "ready" | "failed";


const SAMPLE_WIDTH = 160;
const SAMPLE_HEIGHT = 90;
const SAMPLE_INTERVAL_MS = 100;
const MOTION_CONFIRMATION_MS = 500;
const DELIVERY_SAVED_DISPLAY_MS = 800;
const COOLDOWN_MS = 5500;
const CLIP_DURATION_SECONDS = 2.5;
const PIXEL_CHANGE_THRESHOLD = 18;
const MAX_DELIVERIES = 6;
const SENSITIVITY_THRESHOLDS: Record<Sensitivity, number> = {
  Low: 0.18,
  Medium: 0.12,
  High: 0.075
};


type MotionMetrics = {
  motionScore: number;
  changedPixelRatio: number;
  motionEnergy: number;
};


type DeliveryClip = {
  id: number;
  deliveryIndex: number;
  blob: Blob;
  sessionId: string;
  analysisStatus: AnalysisStatus;
  jobId?: string;
  analysisError?: string;
};


const EMPTY_METRICS: MotionMetrics = { motionScore: 0, changedPixelRatio: 0, motionEnergy: 0 };


function calculateMotion(current: ImageData, previous: ImageData): MotionMetrics {
  const startX = Math.floor(current.width * 0.15);
  const endX = Math.ceil(current.width * 0.85);
  const startY = Math.floor(current.height * 0.10);
  const endY = Math.ceil(current.height * 0.90);
  let changedPixels = 0;
  let totalDifference = 0;
  let sampledPixels = 0;

  for (let y = startY; y < endY; y += 1) {
    for (let x = startX; x < endX; x += 1) {
      const index = (y * current.width + x) * 4;
      const currentGray = (current.data[index] * 299 + current.data[index + 1] * 587 + current.data[index + 2] * 114) / 1000;
      const previousGray = (previous.data[index] * 299 + previous.data[index + 1] * 587 + previous.data[index + 2] * 114) / 1000;
      const difference = Math.abs(currentGray - previousGray);
      if (difference >= PIXEL_CHANGE_THRESHOLD) changedPixels += 1;
      totalDifference += difference;
      sampledPixels += 1;
    }
  }

  if (!sampledPixels) return EMPTY_METRICS;
  const changedPixelRatio = changedPixels / sampledPixels;
  const motionEnergy = totalDifference / (sampledPixels * 255);
  return {
    changedPixelRatio,
    motionEnergy,
    motionScore: changedPixelRatio * 0.7 + motionEnergy * 0.3
  };
}


function statusLabel(state: DetectionState, running: boolean, readyUntil: number): string {
  if (state === "OVER_COMPLETE") return "Over Complete";
  if (state === "MOTION_CANDIDATE") return "Confirming motion";
  if (state === "DELIVERY_DETECTED") return "Delivery detected";
  if (state === "RECORDING_DELIVERY") return "Recording delivery";
  if (state === "DELIVERY_SAVED") return "Delivery saved";
  if (state === "COOLDOWN") return "Cooldown";
  if (!running) return "Waiting to start";
  if (readyUntil > Date.now()) return "Ready for next delivery";
  return "Waiting for delivery";
}


function supportedMimeType(): string | undefined {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}


export function ExperimentalDeliveryTest({ onSelectCalibration }: { onSelectCalibration: (mode: "camera" | "upload") => void }) {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const previousFrameRef = useRef<ImageData | null>(null);
  const stateRef = useRef<DetectionState>("WAITING_FOR_DELIVERY");
  const candidateStartedAtRef = useRef(0);
  const deliverySavedUntilRef = useRef(0);
  const cooldownUntilRef = useRef(0);
  const readyUntilRef = useRef(0);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const recorderStopTimerRef = useRef<number | null>(null);
  const recordingCancelledRef = useRef(false);
  const recordingFailedRef = useRef(false);
  const recordedChunksRef = useRef<Blob[]>([]);
  const clipsRef = useRef<DeliveryClip[]>([]);
  const sessionIdRef = useRef<string | null>(null);
  const mountedRef = useRef(true);

  const [running, setRunning] = useState(false);
  const [state, setState] = useState<DetectionState>("WAITING_FOR_DELIVERY");
  const [sensitivity, setSensitivity] = useState<Sensitivity>("Medium");
  const [clips, setClips] = useState<DeliveryClip[]>([]);
  const [lastDetected, setLastDetected] = useState<Date | null>(null);
  const [recordingError, setRecordingError] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionCreating, setSessionCreating] = useState(false);

  const transition = useCallback((nextState: DetectionState) => {
    stateRef.current = nextState;
    setState(nextState);
  }, []);

  const enterCooldown = useCallback(() => {
    cooldownUntilRef.current = Date.now() + COOLDOWN_MS;
    transition("COOLDOWN");
  }, [transition]);

  const updateClip = useCallback((clipId: number, updates: Partial<DeliveryClip>) => {
    if (!mountedRef.current) return;
    setClips((currentClips) => {
      const nextClips = currentClips.map((clip) => clip.id === clipId ? { ...clip, ...updates } : clip);
      clipsRef.current = nextClips;
      return nextClips;
    });
  }, []);

  const submitClip = useCallback(async (clip: DeliveryClip) => {
    updateClip(clip.id, { analysisStatus: "uploading", analysisError: undefined });
    try {
      const submitted = await detectBallInDeliveryClip(clip.blob, clip.deliveryIndex, clip.sessionId);
      if (process.env.NODE_ENV !== "production") {
        console.info("[Delivery ball detection] upload response", { clipId: clip.id, sessionId: clip.sessionId, response: submitted });
      }
      if (!submitted.success) {
        updateClip(clip.id, { analysisStatus: "failed", analysisError: submitted.message });
        return;
      }
      updateClip(clip.id, {
        analysisStatus: submitted.status === "ready" ? "ready" : "processing",
        jobId: submitted.job_id ?? undefined,
        analysisError: undefined
      });
    } catch (error) {
      updateClip(clip.id, {
        analysisStatus: "failed",
        analysisError: error instanceof Error ? error.message : "Ball detection upload failed."
      });
    } finally {
      if (clip.deliveryIndex === MAX_DELIVERIES) {
        void completeExperimentalSession(clip.sessionId).catch((error) => {
          setRecordingError(error instanceof Error ? error.message : "Could not mark the session complete.");
        });
      }
    }
  }, [updateClip]);

  const beginRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state !== "inactive") return;
    const stream = cameraRef.current?.getStream();
    if (!stream || !stream.getVideoTracks().some((track) => track.readyState === "live")) {
      setRecordingError("Camera stream is unavailable. Check camera permission and try again.");
      enterCooldown();
      return;
    }
    if (!sessionIdRef.current) {
      setRecordingError("Session is not ready. Stop and start detection again.");
      return;
    }
    if (typeof window.MediaRecorder === "undefined") {
      setRecordingError("This browser does not support MediaRecorder. Try a current Chrome, Edge, or Safari browser.");
      setRunning(false);
      transition("WAITING_FOR_DELIVERY");
      return;
    }

    try {
      const mimeType = supportedMimeType();
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      recordedChunksRef.current = [];
      recordingCancelledRef.current = false;
      recordingFailedRef.current = false;
      recorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) recordedChunksRef.current.push(event.data);
      };
      recorder.onerror = () => {
        recordingFailedRef.current = true;
        setRecordingError("The delivery recording failed. Detection will resume after cooldown.");
        if (recorder.state !== "inactive") recorder.stop();
      };
      recorder.onstop = () => {
        if (recorderStopTimerRef.current !== null) window.clearTimeout(recorderStopTimerRef.current);
        recorderStopTimerRef.current = null;
        recorderRef.current = null;
        if (recordingCancelledRef.current) {
          recordedChunksRef.current = [];
          return;
        }
        if (recordingFailedRef.current || recordedChunksRef.current.length === 0) {
          recordedChunksRef.current = [];
          setRecordingError("No playable delivery clip was produced. Detection will resume after cooldown.");
          enterCooldown();
          return;
        }

        const actualMimeType = recorder.mimeType || recordedChunksRef.current[0].type || "video/webm";
        const clip: DeliveryClip = {
          id: Date.now(),
          deliveryIndex: clipsRef.current.length + 1,
          blob: new Blob(recordedChunksRef.current, { type: actualMimeType }),
          sessionId: sessionIdRef.current as string,
          analysisStatus: "not_started"
        };
        recordedChunksRef.current = [];
        const nextClips = [...clipsRef.current, clip].slice(0, MAX_DELIVERIES);
        clipsRef.current = nextClips;
        setClips(nextClips);
        setLastDetected(new Date());
        setRecordingError(null);
        window.setTimeout(() => void submitClip(clip), 0);

        if (nextClips.length >= MAX_DELIVERIES) {
          setRunning(false);
          transition("OVER_COMPLETE");
          return;
        }
        deliverySavedUntilRef.current = Date.now() + DELIVERY_SAVED_DISPLAY_MS;
        transition("DELIVERY_SAVED");
      };

      recorder.start(250);
      transition("RECORDING_DELIVERY");
      recorderStopTimerRef.current = window.setTimeout(() => {
        if (recorder.state !== "inactive") recorder.stop();
      }, CLIP_DURATION_SECONDS * 1000);
    } catch {
      recorderRef.current = null;
      setRecordingError("Recording could not start. Check browser camera and recording permissions.");
      enterCooldown();
    }
  }, [enterCooldown, submitClip, transition]);

  async function startDetection() {
    if (clipsRef.current.length >= MAX_DELIVERIES) {
      transition("OVER_COMPLETE");
      return;
    }
    if (typeof window.MediaRecorder === "undefined") {
      setRecordingError("This browser does not support MediaRecorder. Try a current Chrome, Edge, or Safari browser.");
      return;
    }
    const stream = cameraRef.current?.getStream();
    if (!stream || !stream.getVideoTracks().some((track) => track.readyState === "live")) {
      setRecordingError("Camera is not ready. Allow camera access, wait a moment, and try again.");
      return;
    }

    if (!sessionIdRef.current) {
      setSessionCreating(true);
      try {
        const session = await createExperimentalSession();
        sessionIdRef.current = session.id;
        setSessionId(session.id);
      } catch (error) {
        setRecordingError(error instanceof Error ? error.message : "Could not create the experimental session.");
        setSessionCreating(false);
        return;
      }
      setSessionCreating(false);
    }

    previousFrameRef.current = null;
    deliverySavedUntilRef.current = 0;
    cooldownUntilRef.current = 0;
    readyUntilRef.current = 0;
    setRecordingError(null);
    transition("WAITING_FOR_DELIVERY");
    setRunning(true);
  }

  function stopDetection() {
    setRunning(false);
    previousFrameRef.current = null;
    if (recorderStopTimerRef.current !== null) window.clearTimeout(recorderStopTimerRef.current);
    recorderStopTimerRef.current = null;
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recordingCancelledRef.current = true;
      recorder.stop();
    }
    transition("WAITING_FOR_DELIVERY");
  }

  function resetSession() {
    stopDetection();
    if (sessionIdRef.current && clipsRef.current.length > 0) {
      void completeExperimentalSession(sessionIdRef.current);
    }
    clipsRef.current = [];
    setClips([]);
    sessionIdRef.current = null;
    setSessionId(null);
    setLastDetected(null);
    setRecordingError(null);
  }

  useEffect(() => {
    if (!running) return;
    const threshold = SENSITIVITY_THRESHOLDS[sensitivity];
    const interval = window.setInterval(() => {
      const currentFrame = cameraRef.current?.sampleFrame(SAMPLE_WIDTH, SAMPLE_HEIGHT);
      if (!currentFrame) return;
      const previousFrame = previousFrameRef.current;
      previousFrameRef.current = currentFrame;
      if (!previousFrame) return;

      const nextMetrics = calculateMotion(currentFrame, previousFrame);
      const now = Date.now();

      if (stateRef.current === "WAITING_FOR_DELIVERY") {
        if (nextMetrics.motionScore >= threshold) {
          candidateStartedAtRef.current = now;
          transition("MOTION_CANDIDATE");
        }
        return;
      }

      if (stateRef.current === "MOTION_CANDIDATE") {
        if (nextMetrics.motionScore < threshold) {
          transition("WAITING_FOR_DELIVERY");
          return;
        }
        if (now - candidateStartedAtRef.current >= MOTION_CONFIRMATION_MS) {
          transition("DELIVERY_DETECTED");
          beginRecording();
        }
        return;
      }

      if (stateRef.current === "DELIVERY_SAVED" && now >= deliverySavedUntilRef.current) {
        enterCooldown();
        return;
      }

      if (stateRef.current === "COOLDOWN" && now >= cooldownUntilRef.current) {
        readyUntilRef.current = now + 1200;
        previousFrameRef.current = null;
        transition("WAITING_FOR_DELIVERY");
      }
    }, SAMPLE_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [beginRecording, enterCooldown, running, sensitivity, transition]);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (recorderStopTimerRef.current !== null) window.clearTimeout(recorderStopTimerRef.current);
      const recorder = recorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recordingCancelledRef.current = true;
        recorder.stop();
      }
    };
  }, []);

  const displayStatus = statusLabel(state, running, readyUntilRef.current);
  const zoneActive = state === "MOTION_CANDIDATE" || state === "DELIVERY_DETECTED" || state === "RECORDING_DELIVERY";
  const overComplete = clips.length >= MAX_DELIVERIES || state === "OVER_COMPLETE";
  const nextBallNumber = overComplete ? MAX_DELIVERIES : clips.length + 1;
  const submittedCount = clips.filter((clip) => clip.jobId || clip.analysisStatus === "ready").length;
  const failedUploads = clips.filter((clip) => clip.analysisStatus === "failed").length;

  return (
    <div className="mx-auto max-w-7xl py-2">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-4">
        <div>
          <p className="text-xs font-bold uppercase tracking-[0.2em] text-[#ffca68]">Experimental sandbox</p>
          <h1 className="mt-1 text-2xl font-black">Live delivery clip test</h1>
        </div>
        <div className="flex flex-wrap gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1.5">
          <Button className="px-3 py-2" variant="secondary" onClick={() => onSelectCalibration("camera")}>Camera Calibration</Button>
          <Button className="px-3 py-2" variant="secondary" onClick={() => onSelectCalibration("upload")}>Upload Calibration Image</Button>
          <Button className="px-3 py-2">Experimental Delivery Test</Button>
        </div>
      </div>

      <div className="grid gap-5 xl:grid-cols-[1fr_22rem]">
        <div className="relative">
          <CameraPreview ref={cameraRef} />
          <div className={`pointer-events-none absolute left-[15%] top-[10%] h-[80%] w-[70%] rounded-xl border-2 transition duration-200 ${zoneActive ? "border-[#ffca68] bg-[#ffca68]/10 shadow-[0_0_28px_rgba(255,202,104,0.32)]" : "border-white/25 bg-white/[0.025]"}`}>
            <span className="absolute left-2 top-2 rounded bg-ink/75 px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-white/65">Experimental detection zone</span>
          </div>
          {(state === "DELIVERY_DETECTED" || state === "RECORDING_DELIVERY") && (
            <div className="pointer-events-none absolute inset-0 grid place-items-center bg-lime/10">
              <div className="rounded-2xl border border-lime/50 bg-ink/90 px-7 py-5 text-center shadow-[0_0_45px_rgba(183,243,75,0.35)]">
                <p className="text-2xl font-black text-lime sm:text-4xl">{state === "DELIVERY_DETECTED" ? "Delivery Detected!" : "Recording Delivery"}</p>
                <p className="mt-2 text-xs font-bold uppercase tracking-[0.16em] text-white/55">Capturing {CLIP_DURATION_SECONDS} second clip</p>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-4">
          <Card>
            <div className="flex items-start justify-between gap-4">
              <div>
                <StatusBadge label={overComplete ? "Over complete" : running ? "Detection active" : "Detection stopped"} tone={overComplete || running ? "good" : "neutral"} />
                <p className={`mt-4 text-lg font-black ${overComplete ? "text-lime" : "text-white"}`}>{displayStatus}</p>
                <p className="mt-2 text-sm font-bold text-white/65">Delivery {nextBallNumber} of {MAX_DELIVERIES}</p>
              </div>
              <div className="text-right">
                <p className="text-xs uppercase tracking-wide text-white/40">Captured</p>
                <p className="text-4xl font-black text-lime">{clips.length}/{MAX_DELIVERIES}</p>
              </div>
            </div>

            <div className="mt-5 space-y-2 rounded-xl bg-black/20 p-4 text-xs">
              <div className="flex justify-between gap-3"><span className="text-white/45">Session</span><span className="max-w-44 truncate font-mono">{sessionId ?? "Not started"}</span></div>
              <div className="flex justify-between gap-3"><span className="text-white/45">Capture state</span><span className="font-bold">{state.replaceAll("_", " ")}</span></div>
              <div className="flex justify-between gap-3"><span className="text-white/45">Analysis submitted</span><span className="font-bold">{submittedCount}/{clips.length}</span></div>
              {lastDetected && <div className="flex justify-between gap-3"><span className="text-white/45">Last saved</span><span>{lastDetected.toLocaleTimeString()}</span></div>}
            </div>

            {recordingError && <p className="mt-4 rounded-lg border border-signal/30 bg-signal/10 p-3 text-xs leading-5 text-[#ffaaa6]">{recordingError}</p>}
            {failedUploads > 0 && <p className="mt-3 text-xs text-[#ffaaa6]">{failedUploads} delivery upload{failedUploads === 1 ? "" : "s"} failed.</p>}

            {overComplete && sessionId && (
              <div className="mt-5 rounded-xl border border-lime/20 bg-lime/[0.05] p-4">
                <p className="font-black text-lime">Over Complete</p>
                <p className="mt-2 text-sm leading-6 text-white/60">Six delivery clips were saved. Ball-detection jobs were submitted for each clip.</p>
              </div>
            )}

            <label className="mt-5 block text-xs font-bold uppercase tracking-[0.14em] text-white/45" htmlFor="motion-sensitivity">
              Sensitivity
              <select
                id="motion-sensitivity"
                className="mt-2 w-full rounded-xl border border-white/15 bg-ink px-3 py-3 text-sm font-bold normal-case tracking-normal text-white"
                value={sensitivity}
                disabled={running}
                onChange={(event) => setSensitivity(event.target.value as Sensitivity)}
              >
                <option>Low</option>
                <option>Medium</option>
                <option>High</option>
              </select>
            </label>

            <div className="mt-5 grid grid-cols-2 gap-3">
              <Button disabled={running || overComplete || sessionCreating} onClick={() => void startDetection()}>{sessionCreating ? "Creating Session..." : "Start Detection"}</Button>
              <Button disabled={!running} variant="secondary" onClick={stopDetection}>Stop Detection</Button>
              <Button className="col-span-2" variant="secondary" onClick={resetSession}>Reset Session</Button>
            </div>
          </Card>

          <p className="px-1 text-xs leading-5 text-white/35">Each captured clip is uploaded independently under this session. Processed results live in Session Results.</p>
        </div>
      </div>
    </div>
  );
}
