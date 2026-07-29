"use client";

import { type ChangeEvent, useEffect, useRef, useState } from "react";

import { BallDetectionPanel } from "@/components/video-analysis/BallDetectionPanel";
import { BallTrackingPanel } from "@/components/video-analysis/BallTrackingPanel";
import { AssistedSceneCalibrationPanel } from "@/components/video-analysis/AssistedSceneCalibrationPanel";
import { MEDIA_FIT_CLASS } from "@/components/video-analysis/AnalysisMediaStage";
import { RealPitchRegistrationPanel } from "@/components/video-analysis/RealPitchRegistrationPanel";
import { SceneCalibrationPanel } from "@/components/video-analysis/SceneCalibrationPanel";
import { VirtualPitchOverlay } from "@/components/video-analysis/VirtualPitchOverlay";
import { WicketObservationPanel } from "@/components/video-analysis/WicketObservationPanel";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getVideoAnalysis,
  getVideoAnalysisCalibration,
  getSceneCalibration,
  getVideoBallDetectionResult,
  getVideoBallTrackingResult,
  prepareVideoAnalysis,
  type ConfirmedVideoCalibrationResponse,
  type SceneCalibrationResult,
  type VideoAnalysisPreparedResponse,
  type VideoBallDetectionResultResponse,
  type VideoBallTrackingResultResponse
} from "@/lib/api";


type WorkspaceState = "idle" | "file_selected" | "uploading" | "prepared" | "failed";
type ActiveStage = "upload" | "calibration" | "ball_detection" | "ball_tracking";


const MAX_FILE_BYTES = 500 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = new Set(["mp4", "mov", "webm", "avi", "mkv"]);
const ACCEPT_VALUE = "video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska";


function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB"];
  let value = bytes / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && value >= 1024; index += 1) {
    value /= 1024;
    unit = units[index];
  }
  return `${value.toFixed(value >= 100 ? 0 : 1)} ${unit}`;
}


function formatDuration(seconds: number): string {
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds - minutes * 60;
  return minutes ? `${minutes}m ${remainder.toFixed(1)}s` : `${remainder.toFixed(2)}s`;
}


function validateVideo(file: File): string | null {
  const extension = file.name.split(".").pop()?.toLowerCase();
  if (!extension || !ACCEPTED_EXTENSIONS.has(extension)) {
    return "Choose an MP4, MOV, WebM, AVI, or MKV video.";
  }
  if (file.size === 0) return "The selected video is empty.";
  if (file.size > MAX_FILE_BYTES) return "The selected video exceeds the 500 MB limit.";
  return null;
}


type StepState = "complete" | "current" | "future" | "failed";


function WorkflowStepper({
  steps,
  onSelect
}: {
  steps: Array<{ key: string; label: string; state: StepState; stage?: ActiveStage }>;
  onSelect?: (stage: ActiveStage) => void;
}) {
  return (
    <nav aria-label="Analysis workflow" className="overflow-x-auto">
      <ol className="flex min-w-0 items-center gap-1 sm:gap-2">
        {steps.map((step, index) => {
          const styles: Record<StepState, string> = {
            complete: "border-lime/35 bg-lime/[0.08] text-lime",
            current: "border-lime/50 bg-lime/[0.12] text-white",
            future: "border-white/10 bg-white/[0.03] text-white/40",
            failed: "border-signal/40 bg-signal/10 text-[#ffaaa6]"
          };
          const clickable =
            Boolean(onSelect && step.stage)
            && (step.state === "complete" || step.state === "current" || (
              step.key === "detect" && steps.some((s) => s.key === "calibration" && s.state === "complete")
            ) || (
              step.key === "track" && steps.some((s) => s.key === "detect" && s.state === "complete")
            ) || (
              step.key === "calibration" && steps.some((s) => s.key === "upload" && s.state === "complete")
            ));
          const content = (
            <>
              <span className="tabular-nums opacity-70">{index + 1}</span>
              {step.label}
            </>
          );
          return (
            <li key={step.key} className="flex min-w-0 items-center gap-1 sm:gap-2">
              {index > 0 && (
                <span
                  className={`hidden h-px w-3 shrink-0 sm:block sm:w-5 ${
                    step.state === "future" ? "bg-white/10" : "bg-lime/35"
                  }`}
                  aria-hidden
                />
              )}
              {clickable && step.stage ? (
                <button
                  type="button"
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] sm:px-3 ${styles[step.state]}`}
                  aria-current={step.state === "current" ? "step" : undefined}
                  onClick={() => onSelect?.(step.stage!)}
                >
                  {content}
                </button>
              ) : (
                <span
                  className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-bold uppercase tracking-[0.08em] sm:px-3 ${styles[step.state]}`}
                  aria-current={step.state === "current" ? "step" : undefined}
                >
                  {content}
                </span>
              )}
            </li>
          );
        })}
      </ol>
    </nav>
  );
}


function CompactMeta({ analysis }: { analysis: VideoAnalysisPreparedResponse }) {
  const items = [
    `${analysis.width}×${analysis.height}`,
    `${analysis.fps.toFixed(1).replace(/\.0$/, "")} fps`,
    `${analysis.frame_count.toLocaleString()} frames`,
    formatDuration(analysis.duration_seconds)
  ];
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <span
          key={item}
          className="rounded-md border border-white/10 bg-black/25 px-2 py-0.5 text-[11px] font-semibold text-white/55"
        >
          {item}
        </span>
      ))}
    </div>
  );
}


export default function VideoAnalysisPage() {
  const previewUrlRef = useRef<string | null>(null);
  const [workspaceState, setWorkspaceState] = useState<WorkspaceState>("idle");
  const [activeStage, setActiveStage] = useState<ActiveStage>("upload");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<VideoAnalysisPreparedResponse | null>(null);
  const [confirmedCalibration, setConfirmedCalibration] = useState<
    ConfirmedVideoCalibrationResponse | null
  >(null);
  const [sceneCalibration, setSceneCalibration] = useState<
    SceneCalibrationResult | null
  >(null);
  const [ballDetectionResult, setBallDetectionResult] = useState<
    VideoBallDetectionResultResponse | null
  >(null);
  const [ballTrackingResult, setBallTrackingResult] = useState<
    VideoBallTrackingResultResponse | null
  >(null);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);

  function releasePreview() {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreviewUrl(null);
  }

  function clearPreparedUrl() {
    window.history.replaceState(null, "", "/video-analysis");
  }

  function selectFile(file: File) {
    releasePreview();
    const objectUrl = URL.createObjectURL(file);
    previewUrlRef.current = objectUrl;
    setPreviewUrl(objectUrl);
    setSelectedFile(file);
    setAnalysis(null);
    setConfirmedCalibration(null);
    setSceneCalibration(null);
    setBallDetectionResult(null);
    setBallTrackingResult(null);
    setActiveStage("upload");
    setWorkspaceState("file_selected");
    setError(null);
    clearPreparedUrl();
  }

  function handleFileSelection(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const validationError = validateVideo(file);
    if (validationError) {
      setError(validationError);
      setWorkspaceState(selectedFile ? "file_selected" : "failed");
      return;
    }
    selectFile(file);
  }

  function removeSelectedFile() {
    releasePreview();
    setSelectedFile(null);
    setAnalysis(null);
    setConfirmedCalibration(null);
    setSceneCalibration(null);
    setBallDetectionResult(null);
    setBallTrackingResult(null);
    setError(null);
    setActiveStage("upload");
    setWorkspaceState("idle");
    clearPreparedUrl();
  }

  async function uploadAndPrepare() {
    if (!selectedFile) {
      setError("Select a video before preparing the analysis.");
      setWorkspaceState("failed");
      return;
    }
    const validationError = validateVideo(selectedFile);
    if (validationError) {
      setError(validationError);
      setWorkspaceState("failed");
      return;
    }

    setWorkspaceState("uploading");
    setError(null);
    try {
      const prepared = await prepareVideoAnalysis(selectedFile);
      setAnalysis(prepared);
      setConfirmedCalibration(null);
      setSceneCalibration(null);
      setBallDetectionResult(null);
      setBallTrackingResult(null);
      setWorkspaceState("prepared");
      setActiveStage("calibration");
      window.history.replaceState(
        null,
        "",
        `/video-analysis?analysis_id=${encodeURIComponent(prepared.analysis_id)}`
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Video preparation failed.");
      setWorkspaceState("failed");
    }
  }

  useEffect(() => {
    const analysisId = new URLSearchParams(window.location.search).get("analysis_id");
    if (!analysisId) return;
    let cancelled = false;
    setRestoring(true);
    setWorkspaceState("uploading");
    void Promise.all([
      getVideoAnalysis(analysisId),
      getVideoAnalysisCalibration(analysisId),
      getSceneCalibration(analysisId),
      getVideoBallDetectionResult(analysisId),
      getVideoBallTrackingResult(analysisId)
    ])
      .then(([
        restored,
        calibration,
        assistedCalibration,
        detectionResult,
        trackingResult
      ]) => {
        if (cancelled) return;
        setAnalysis(restored);
        setConfirmedCalibration(calibration);
        setSceneCalibration(assistedCalibration);
        setBallDetectionResult(detectionResult);
        setBallTrackingResult(trackingResult);
        setWorkspaceState("prepared");
        setActiveStage(
          trackingResult
          || restored.tracking_status === "tracking_queued"
          || restored.tracking_status === "tracking_ball"
            ? "ball_tracking"
            : detectionResult
              || restored.ball_detection_status === "detection_queued"
              || restored.ball_detection_status === "detecting_ball"
              || restored.ball_detection_status === "detection_complete"
                ? "ball_detection"
                : "calibration"
        );
        setError(null);
      })
      .catch((caught) => {
        if (cancelled) return;
        setError(caught instanceof Error ? caught.message : "The saved analysis could not be restored.");
        setWorkspaceState("failed");
      })
      .finally(() => {
        if (!cancelled) setRestoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
  }, []);

  const uploadComplete = workspaceState === "prepared" && analysis !== null;
  const calibrationComplete = Boolean(
    confirmedCalibration
    || (
      sceneCalibration
      && ![
        "NOT_STARTED",
        "DETECTING_WICKETS",
        "OBSERVING_WICKETS",
        "GENERATING_POSE"
      ].includes(sceneCalibration.stage)
    )
  );
  const calibrationActive = uploadComplete && activeStage === "calibration";
  const ballDetectionActive = (
    uploadComplete
    && activeStage === "ball_detection"
    && calibrationComplete
  );
  const ballTrackingActive = (
    uploadComplete
    && activeStage === "ball_tracking"
    && ballDetectionResult !== null
  );
  const replayReady = Boolean(ballTrackingResult?.summary?.delivery_replay_url);

  const workspaceTone: "neutral" | "good" | "warn" =
    workspaceState === "failed"
      ? "warn"
      : uploadComplete
        ? "good"
        : workspaceState === "uploading"
          ? "neutral"
          : "neutral";

  const workspaceLabel =
    workspaceState === "uploading"
      ? "Processing"
      : workspaceState === "failed"
        ? "Failed"
        : uploadComplete
          ? "Ready"
          : selectedFile
            ? "Ready"
            : "Ready";

  const workflowSteps: Array<{ key: string; label: string; state: StepState; stage?: ActiveStage }> = [
    {
      key: "upload",
      label: "Upload",
      stage: "upload",
      state: workspaceState === "failed" && !uploadComplete
        ? "failed"
        : uploadComplete
          ? "complete"
          : "current"
    },
    {
      key: "calibration",
      label: "Scene Setup",
      stage: "calibration",
      state: !uploadComplete
        ? "future"
        : calibrationComplete && !calibrationActive
          ? "complete"
          : calibrationActive
            ? "current"
            : uploadComplete
              ? "current"
              : "future"
    },
    {
      key: "detect",
      label: "Detect",
      stage: "ball_detection",
      state: !calibrationComplete
        ? "future"
        : ballDetectionResult && !ballDetectionActive
          ? "complete"
          : ballDetectionActive
            ? "current"
            : "future"
    },
    {
      key: "track",
      label: "Track",
      stage: "ball_tracking",
      state: !ballDetectionResult
        ? "future"
        : ballTrackingResult && !ballTrackingActive
          ? "complete"
          : ballTrackingActive
            ? "current"
            : "future"
    },
    {
      key: "replay",
      label: "Replay",
      state: replayReady
        ? "complete"
        : "future"
    }
  ];

  function selectWorkflowStage(stage: ActiveStage) {
    if (stage === "upload") return;
    if (stage === "calibration" && uploadComplete) {
      setActiveStage("calibration");
      return;
    }
    if (stage === "ball_detection" && calibrationComplete) {
      setActiveStage("ball_detection");
      return;
    }
    if (stage === "ball_tracking" && ballDetectionResult) {
      setActiveStage("ball_tracking");
    }
  }
  return (
    <div className="mx-auto max-w-7xl py-3 sm:py-4">
      <header className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 max-w-2xl">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="text-2xl font-black tracking-tight sm:text-3xl">Video Analysis</h1>
            <StatusBadge label={workspaceLabel} tone={workspaceTone} />
          </div>
          <p className="mt-1.5 text-sm leading-6 text-white/45">
            Upload a cricket clip, calibrate the scene, then detect and track the ball.
          </p>
          {analysis && uploadComplete && (
            <div className="mt-2.5">
              <CompactMeta analysis={analysis} />
              <p className="mt-1.5 truncate text-[11px] text-white/30" title={analysis.original_filename}>
                {analysis.original_filename}
              </p>
            </div>
          )}
        </div>
        {uploadComplete && (
          <label
            htmlFor="analysis-video"
            className="cursor-pointer rounded-xl border border-white/15 bg-white/5 px-3 py-2 text-xs font-bold hover:bg-white/10"
          >
            Prepare another
          </label>
        )}
      </header>

      <div className="mt-4">
        <WorkflowStepper steps={workflowSteps} onSelect={selectWorkflowStage} />
      </div>

      <input
        id="analysis-video"
        className="sr-only"
        type="file"
        accept={ACCEPT_VALUE}
        onChange={handleFileSelection}
      />

      {!uploadComplete && (
        <Card className="mt-4 p-4 sm:p-5">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <h2 className="text-base font-black">Upload Video</h2>
              <p className="mt-1 text-sm text-white/45">MP4, MOV, WebM, AVI, or MKV · max 500 MB</p>
            </div>
          </div>

          {error && (
            <p className="mt-3 rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]">
              {error}
            </p>
          )}

          {!selectedFile && !restoring && (
            <label
              htmlFor="analysis-video"
              className="mt-4 block cursor-pointer rounded-xl border border-dashed border-white/20 bg-white/[0.03] px-5 py-8 text-center transition hover:border-lime/40 hover:bg-lime/[0.03]"
            >
              <span className="block text-base font-black">Choose one cricket video</span>
              <span className="mt-1 block text-sm text-white/40">
                File stays local until you prepare analysis.
              </span>
            </label>
          )}
          {selectedFile && (
            <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(14rem,17rem)]">
              <div className="flex max-h-[min(42dvh,calc(100dvh-16rem))] min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08] sm:max-h-[min(52dvh,calc(100dvh-14rem))]">
                <video
                  className={MEDIA_FIT_CLASS}
                  controls
                  preload="metadata"
                  src={previewUrl ?? undefined}
                />
              </div>
              <aside className="rounded-xl border border-white/10 bg-black/20 p-3 lg:sticky lg:top-4 lg:self-start">
                <p className="break-all text-sm font-bold">{selectedFile.name}</p>
                <p className="mt-1 text-sm text-white/45">{formatBytes(selectedFile.size)}</p>
                <div className="mt-4 space-y-2">
                  <label
                    htmlFor="analysis-video"
                    className="block cursor-pointer rounded-xl border border-white/15 bg-white/5 px-4 py-2.5 text-center text-sm font-bold hover:bg-white/10"
                  >
                    Change file
                  </label>
                  <Button
                    className="w-full"
                    variant="danger"
                    disabled={workspaceState === "uploading"}
                    onClick={removeSelectedFile}
                  >
                    Remove
                  </Button>
                  <Button
                    className="w-full"
                    disabled={workspaceState === "uploading"}
                    onClick={() => void uploadAndPrepare()}
                  >
                    {workspaceState === "uploading"
                      ? "Preparing…"
                      : "Upload and Prepare"}
                  </Button>
                </div>
              </aside>
            </div>
          )}

          {workspaceState === "uploading" && (
            <div className="mt-4 rounded-xl border border-lime/20 bg-lime/[0.04] p-3">
              <p className="text-sm font-bold text-lime">
                {restoring ? "Restoring prepared analysis…" : "Uploading and preparing video…"}
              </p>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
                <div className="h-full w-1/2 animate-pulse rounded-full bg-lime" />
              </div>
            </div>
          )}
        </Card>
      )}

      {uploadComplete && analysis && (
        <section className="mt-4 space-y-4">
          {error && (
            <p className="rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]">
              {error}
            </p>
          )}

          <details className="rounded-xl border border-white/10 bg-panel/60 px-4 py-3">
            <summary className="cursor-pointer text-sm font-bold text-white/60">
              Source media &amp; reference frame
            </summary>
            <div className="mt-3 grid gap-4 md:grid-cols-2">
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-white/35">
                  Original video
                </p>
                <div className="mt-2 flex max-h-[min(42dvh,calc(100dvh-16rem))] min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08] sm:max-h-[min(52dvh,calc(100dvh-14rem))]">
                  <video
                    className={MEDIA_FIT_CLASS}
                    controls
                    preload="metadata"
                    src={analysis.original_video_url}
                  />
                </div>
              </div>
              <div>
                <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-white/35">
                  Reference frame {analysis.reference_frame_index}
                </p>
                <div className="mt-2 flex max-h-[min(42dvh,calc(100dvh-16rem))] min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08] sm:max-h-[min(52dvh,calc(100dvh-14rem))]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    className={MEDIA_FIT_CLASS}
                    src={analysis.reference_frame_url}
                    alt={`Calibration reference frame ${analysis.reference_frame_index}`}
                  />
                </div>
              </div>
            </div>
          </details>

          {calibrationActive && (
            <Card>
              <AssistedSceneCalibrationPanel
                analysisId={analysis.analysis_id}
                initialResult={sceneCalibration}
                onResult={setSceneCalibration}
                onContinue={() => setActiveStage("ball_detection")}
              />
            </Card>
          )}

          {sceneCalibration && sceneCalibration.stage !== "NOT_STARTED" && (
            <details className="rounded-xl border border-white/10 bg-panel/60 px-4 py-3">
              <summary className="cursor-pointer text-sm font-bold text-white/60">
                Developer Diagnostics
              </summary>
              <div className="mt-4 space-y-4">
                <details className="border border-white/10 p-3">
                  <summary className="cursor-pointer text-sm font-bold text-white/55">
                    Legacy Scene Calibration
                  </summary>
                  <div className="mt-3">
                    <SceneCalibrationPanel
              key={analysis.analysis_id}
              analysis={analysis}
              initialCalibration={confirmedCalibration}
              onDirty={() => setConfirmedCalibration(null)}
              onReferenceFrameUpdated={(referenceFrameIndex, referenceFrameUrl) => {
                setAnalysis((current) => current ? {
                  ...current,
                  reference_frame_index: referenceFrameIndex,
                  reference_frame_url: referenceFrameUrl
                } : current);
              }}
              onCalibrated={(calibration) => {
                setConfirmedCalibration(calibration);
                setAnalysis((current) => current ? {
                  ...current,
                  status: "calibrated",
                  calibration_status: "confirmed",
                  calibration_url: calibration.calibration_url,
                  calibration_overlay_url: calibration.calibration_overlay_url,
                  visual_calibration_mode: calibration.mode ?? "automatic_visual",
                  visual_calibration_quality: calibration.quality ?? null,
                  reference_frame_index: calibration.reference_frame_index,
                  updated_at: calibration.updated_at
                } : current);
              }}
            />
                  </div>
                </details>
                <details className="border border-white/10 p-3">
                  <summary className="cursor-pointer text-sm font-bold text-white/55">
                    Wicket Observations V1
                  </summary>
                  <div className="mt-3">
                    <WicketObservationPanel analysisId={analysis.analysis_id} />
                  </div>
                </details>
                <details className="border border-white/10 p-3">
                  <summary className="cursor-pointer text-sm font-bold text-white/55">
                    Real Camera Pose Candidates
                  </summary>
                  <div className="mt-3">
                    <RealPitchRegistrationPanel analysisId={analysis.analysis_id} />
                  </div>
                </details>
                <details className="border border-white/10 p-3">
                  <summary className="cursor-pointer text-sm font-bold text-white/55">
                    Synthetic Virtual Pitch
                  </summary>
                  <div className="mt-3">
                    <VirtualPitchOverlay />
                  </div>
                </details>
              </div>
            </details>
          )}

          {ballDetectionActive && (
            <>
              <BallDetectionPanel
                key={analysis.analysis_id}
                analysis={analysis}
                initialResult={ballDetectionResult}
                initialJobId={analysis.ball_detection_job_id}
                onResult={(result) => {
                  setBallDetectionResult(result);
                  if (!result) setBallTrackingResult(null);
                }}
              />
              <div className="flex flex-wrap gap-2">
                {ballDetectionResult && (
                  <Button onClick={() => setActiveStage("ball_tracking")}>
                    Continue to Ball Tracking
                  </Button>
                )}
                <Button variant="secondary" onClick={() => setActiveStage("calibration")}>
                  Review Calibration
                </Button>
              </div>
            </>
          )}

          {ballTrackingActive && ballDetectionResult && (
            <>
              <BallTrackingPanel
                key={analysis.analysis_id}
                analysis={analysis}
                detectionResult={ballDetectionResult}
                initialResult={ballTrackingResult}
                initialJobId={analysis.tracking_job_id}
                onResult={setBallTrackingResult}
              />
              <Button variant="secondary" onClick={() => setActiveStage("ball_detection")}>
                Back to Ball Detection
              </Button>
            </>
          )}
        </section>
      )}
    </div>
  );
}
