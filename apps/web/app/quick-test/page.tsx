"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { BallDetectionPanel } from "@/components/video-analysis/BallDetectionPanel";
import { BallTrackingPanel } from "@/components/video-analysis/BallTrackingPanel";
import { CameraPreview, type CameraPreviewHandle } from "@/components/live/CameraPreview";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  prepareQuickTestVideo,
  type VideoAnalysisPreparedResponse,
  type VideoBallDetectionResultResponse
} from "@/lib/api";


type Stage = "idle" | "recording" | "uploading" | "ready" | "failed";

const MAX_CLIP_SECONDS = 8;

function supportedMimeType(): string | undefined {
  const candidates = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType));
}

export default function QuickTestPage() {
  const cameraRef = useRef<CameraPreviewHandle>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<number | null>(null);

  const [stage, setStage] = useState<Stage>("idle");
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<VideoAnalysisPreparedResponse | null>(null);
  const [detectionResult, setDetectionResult] = useState<VideoBallDetectionResultResponse | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  useEffect(() => {
    if (stage !== "recording") return;
    const start = Date.now();
    const interval = window.setInterval(() => setElapsedSeconds((Date.now() - start) / 1000), 100);
    return () => window.clearInterval(interval);
  }, [stage]);

  const uploadClip = useCallback(async (blob: Blob) => {
    try {
      setStage("uploading");
      const file = new File([blob], `quick-test-${Date.now()}.webm`, { type: blob.type || "video/webm" });
      const prepared = await prepareQuickTestVideo(file);
      setAnalysis(prepared);
      setDetectionResult(null);
      setStage("ready");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed.");
      setStage("failed");
    }
  }, []);

  const stopRecording = useCallback(() => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current);
      stopTimerRef.current = null;
    }
    const recorder = recorderRef.current;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }, []);

  const startRecording = useCallback(() => {
    setError(null);
    setAnalysis(null);
    setDetectionResult(null);
    setElapsedSeconds(0);
    const stream = cameraRef.current?.getStream();
    if (!stream || !stream.getVideoTracks().some((track) => track.readyState === "live")) {
      setError("Camera is not ready. Allow camera access and try again.");
      return;
    }
    if (typeof window.MediaRecorder === "undefined") {
      setError("This browser does not support MediaRecorder. Try current Chrome, Edge, or Safari.");
      return;
    }
    try {
      const mimeType = supportedMimeType();
      // ponytail: MediaRecorder's default bitrate is far below a native camera
      // app's, and blurs exactly the fast-moving small ball the detector needs.
      const recorderOptions: MediaRecorderOptions = { videoBitsPerSecond: 8_000_000 };
      if (mimeType) recorderOptions.mimeType = mimeType;
      const recorder = new MediaRecorder(stream, recorderOptions);
      chunksRef.current = [];
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        recorderRef.current = null;
        const clipBlob = new Blob(chunksRef.current, { type: recorder.mimeType || "video/webm" });
        chunksRef.current = [];
        if (clipBlob.size === 0) {
          setError("No video was captured. Try recording again.");
          setStage("idle");
          return;
        }
        void uploadClip(clipBlob);
      };
      recorder.start(250);
      setStage("recording");
      stopTimerRef.current = window.setTimeout(stopRecording, MAX_CLIP_SECONDS * 1000);
    } catch {
      setError("Recording could not start. Check browser camera and recording permissions.");
    }
  }, [stopRecording, uploadClip]);

  const reset = useCallback(() => {
    setStage("idle");
    setError(null);
    setAnalysis(null);
    setDetectionResult(null);
    setElapsedSeconds(0);
  }, []);

  const statusMessage =
    stage === "recording" ? "Recording... press Stop after the delivery."
    : stage === "uploading" ? "Uploading clip..."
    : stage === "ready" ? "Uploaded — run detection below."
    : "Record a short clip of a delivery.";

  return (
    <div className="mx-auto flex max-w-3xl flex-col gap-5 p-6">
      <div>
        <h1 className="text-2xl font-black text-white">Quick Test</h1>
        <p className="mt-1 text-sm text-white/60">
          Record a delivery on any camera, no stumps or calibration required. Runs the same
          ball detection and Moving Ball Tracker pipeline as Video Analysis, including gap
          recovery for frames where the ball wasn&apos;t detected.
        </p>
      </div>

      <Card>
        <CameraPreview ref={cameraRef} />
        <div className="mt-4 flex items-center justify-between gap-3">
          <StatusBadge
            label={stage === "failed" ? "Failed" : statusMessage}
            tone={stage === "failed" ? "warn" : "neutral"}
          />
          {stage === "recording" && <span className="text-sm text-white/60">{elapsedSeconds.toFixed(1)}s</span>}
        </div>
        <div className="mt-4 flex gap-3">
          {(stage === "idle" || stage === "failed" || stage === "ready") && (
            <Button onClick={startRecording}>
              {stage === "ready" ? "Record Another" : "Start Recording"}
            </Button>
          )}
          {stage === "recording" && (
            <Button variant="danger" onClick={stopRecording}>
              Stop Recording
            </Button>
          )}
          {stage === "ready" && (
            <Button variant="secondary" onClick={reset}>
              Clear
            </Button>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-[#ffaaa6]">{error}</p>}
      </Card>

      {analysis && (
        <BallDetectionPanel
          key={analysis.analysis_id}
          analysis={analysis}
          initialResult={null}
          onResult={setDetectionResult}
        />
      )}

      {analysis && detectionResult && (
        <BallTrackingPanel
          key={analysis.analysis_id}
          analysis={analysis}
          detectionResult={detectionResult}
          initialResult={null}
          onResult={() => undefined}
        />
      )}
    </div>
  );
}
