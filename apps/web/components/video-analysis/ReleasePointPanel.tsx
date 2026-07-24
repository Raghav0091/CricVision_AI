"use client";

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getReleasePointJob,
  getReleasePointResult,
  startReleasePointAnalysis,
  type ReleaseEvidenceMode,
  type ReleaseJobStatus,
  type ReleaseResult,
  type ReleaseResultResponse,
  type VideoAnalysisPreparedResponse,
  type VideoBallTrackingResultResponse
} from "@/lib/api";

import { MEDIA_FIT_CLASS } from "./AnalysisMediaStage";


type JobView = {
  jobId: string;
  status: ReleaseJobStatus;
  progress: number;
  message: string;
};


type ConfidenceBand = "high" | "medium" | "low";


type DeliveryEventType = "release" | "bounce" | "impact";


type DeliveryTimelineEvent = {
  type: DeliveryEventType;
  label: string;
  timeSeconds: number;
  confidenceBand?: ConfidenceBand;
};


const ACTIVE_JOB_STATUSES = new Set<ReleaseJobStatus>([
  "queued",
  "loading_inputs",
  "generating_candidates",
  "scoring_candidates",
  "saving_results"
]);


const PROCESSING_STEPS = [
  "Analysing bowler action",
  "Evaluating ball trajectory",
  "Estimating release event"
];


function statusLabel(status: ReleaseJobStatus): string {
  const labels: Record<ReleaseJobStatus, string> = {
    queued: "Queued",
    loading_inputs: "Loading inputs",
    generating_candidates: "Generating candidates",
    scoring_candidates: "Scoring candidates",
    saving_results: "Saving results",
    ready: "Ready",
    unresolved: "Unresolved",
    failed: "Failed"
  };
  return labels[status];
}


function evidenceModeLabel(mode: ReleaseEvidenceMode): string {
  const labels: Record<ReleaseEvidenceMode, string> = {
    observed_pose_ball_separation: "Observed Release",
    trajectory_pose_inferred: "Estimated Release",
    fallback_trajectory_only: "Trajectory Estimate",
    unresolved: "Unresolved"
  };
  return labels[mode];
}


function evidenceModeDescription(mode: ReleaseEvidenceMode): string {
  const descriptions: Record<ReleaseEvidenceMode, string> = {
    observed_pose_ball_separation: "Ball and bowler motion support the release moment directly.",
    trajectory_pose_inferred: "The release is estimated from pose timing and ball trajectory.",
    fallback_trajectory_only: "The release is estimated from the tracked ball path only.",
    unresolved: "The available evidence was not reliable enough to identify a release point."
  };
  return descriptions[mode];
}


function confidenceBand(confidence: number): ConfidenceBand {
  if (confidence >= 0.85) return "high";
  if (confidence >= 0.6) return "medium";
  return "low";
}


function confidenceLabel(confidence: number): string {
  return confidenceBand(confidence).toUpperCase();
}


function confidenceTone(band: ConfidenceBand): "good" | "neutral" | "warn" {
  if (band === "high") return "good";
  if (band === "medium") return "neutral";
  return "warn";
}


function formatConfidence(confidence: number): string {
  return `${Math.round(Math.max(0, Math.min(1, confidence)) * 100)}%`;
}


function formatTime(seconds?: number | null): string {
  return seconds == null ? "-" : `${seconds.toFixed(3)} s`;
}


function formatFrame(frame?: number | null): string {
  return frame == null ? "-" : frame.toLocaleString();
}


function frameInterval(result: ReleaseResult): string | null {
  const interval = result.frame_uncertainty;
  if (!interval) return null;
  if (interval.start === interval.end) return `Frame ${interval.start}`;
  return `Frames ${interval.start}-${interval.end}`;
}


function getString(value: unknown): string | null {
  return typeof value === "string" && value.trim() ? value : null;
}


function readableMethod(method: string): string {
  return method
    .replace(/^release_point_v1_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}


function DeliveryEventTimeline({
  durationSeconds,
  events
}: {
  durationSeconds: number;
  events: DeliveryTimelineEvent[];
}) {
  const safeDuration = Math.max(durationSeconds, ...events.map((event) => event.timeSeconds), 0.1);

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-3">
      <div className="flex items-center justify-between gap-3 text-[11px] font-bold text-white/40">
        <span>0.0s</span>
        <span>{safeDuration.toFixed(1)}s</span>
      </div>
      <div className="relative mt-8 h-14">
        <div className="absolute left-0 right-0 top-3 h-px rounded-full bg-white/15" />
        {events.map((event) => {
          const left = Math.min(100, Math.max(0, (event.timeSeconds / safeDuration) * 100));
          const color =
            event.confidenceBand === "low"
              ? "border-[#ffca68] bg-[#ffca68]"
              : event.confidenceBand === "medium"
                ? "border-white bg-white"
                : "border-lime bg-lime";
          return (
            <div
              key={`${event.type}-${event.timeSeconds}`}
              className="absolute top-0 -translate-x-1/2"
              style={{ left: `${left}%` }}
            >
              <div className={`mx-auto h-6 w-6 rounded-full border-4 border-black ${color}`} />
              <div className="mt-2 whitespace-nowrap text-center">
                <p className="text-[10px] font-black uppercase tracking-[0.12em] text-white">
                  {event.label}
                </p>
                <p className="text-[11px] font-semibold text-white/45">{event.timeSeconds.toFixed(3)}s</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}


function ReleaseVideo({
  analysis,
  result
}: {
  analysis: VideoAnalysisPreparedResponse;
  result: ReleaseResult;
}) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const hasMarker = (
    result.status === "ready"
    && result.evidence_mode !== "unresolved"
    && result.release_point_px != null
    && result.release_time_seconds != null
  );
  const xPercent = hasMarker
    ? Math.min(100, Math.max(0, (result.release_point_px!.x / analysis.width) * 100))
    : 0;
  const yPercent = hasMarker
    ? Math.min(100, Math.max(0, (result.release_point_px!.y / analysis.height) * 100))
    : 0;

  function viewReleaseMoment() {
    if (!videoRef.current || result.release_time_seconds == null) return;
    videoRef.current.currentTime = Math.max(0, result.release_time_seconds - 0.08);
    videoRef.current.pause();
    videoRef.current.focus();
  }

  return (
    <div className="rounded-xl border border-lime/20 bg-lime/[0.03] p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-lime">
          Release Visualization
        </p>
        {hasMarker && (
          <Button className="px-3 py-2 text-xs" variant="secondary" onClick={viewReleaseMoment}>
            View Release Moment
          </Button>
        )}
      </div>
      <div
        className="relative mt-2 overflow-hidden rounded-xl bg-[#050a08]"
        style={{ aspectRatio: `${analysis.width} / ${analysis.height}` }}
      >
        <video
          ref={videoRef}
          className={MEDIA_FIT_CLASS}
          controls
          preload="metadata"
          src={analysis.original_video_url}
        />
        {hasMarker && (
          <div
            className="pointer-events-none absolute -translate-x-1/2 -translate-y-1/2"
            style={{ left: `${xPercent}%`, top: `${yPercent}%` }}
            aria-hidden
          >
            <div className="relative">
              <span className="absolute left-1/2 top-1/2 h-12 w-12 -translate-x-1/2 -translate-y-1/2 rounded-full border border-lime/40" />
              <span className="block h-4 w-4 rounded-full border-2 border-black bg-lime shadow-[0_0_22px_rgba(195,255,83,0.65)]" />
              <span className="absolute left-5 top-1/2 -translate-y-1/2 rounded-md border border-lime/30 bg-black/80 px-2 py-1 text-[10px] font-black uppercase tracking-[0.12em] text-lime">
                Release
              </span>
            </div>
          </div>
        )}
      </div>
      <p className="mt-2 text-xs leading-5 text-white/40">
        Browser seeking is positioned close to the release timestamp; the source video is unchanged.
      </p>
    </div>
  );
}


function ReleaseResultView({
  analysis,
  trackingResult,
  resultResponse
}: {
  analysis: VideoAnalysisPreparedResponse;
  trackingResult: VideoBallTrackingResultResponse;
  resultResponse: ReleaseResultResponse;
}) {
  const result = resultResponse.result;
  const band = confidenceBand(result.confidence);
  const interval = frameInterval(result);
  const unresolved = result.status === "unresolved" || result.evidence_mode === "unresolved";
  const timelineEvents: DeliveryTimelineEvent[] = unresolved || result.release_time_seconds == null
    ? []
    : [{
      type: "release",
      label: "Release",
      timeSeconds: result.release_time_seconds,
      confidenceBand: band
    }];
  const detector = getString(result.provenance.ball_detector_model_key)
    ?? getString(result.provenance.ball_detector_model)
    ?? getString(result.provenance.detector);
  const trackingVersion = getString(result.provenance.tracking_version)
    ?? "Complete Delivery Tracking v2";
  const poseProvider = getString(result.provenance.pose_provider);

  if (unresolved) {
    return (
      <div className="mt-4 space-y-4">
        <div className="rounded-xl border border-[#ffca68]/30 bg-[#ffca68]/[0.05] p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-[11px] font-black uppercase tracking-[0.14em] text-[#ffdc9a]">
                Release Point
              </p>
              <h3 className="mt-1 text-xl font-black">Could not determine release reliably</h3>
              <p className="mt-2 max-w-2xl text-sm leading-6 text-white/55">
                Tracking remains available, but the release event was not supported strongly enough for this delivery.
              </p>
            </div>
            <StatusBadge label="Unresolved" tone="warn" />
          </div>
        </div>
        <DeliveryEventTimeline durationSeconds={analysis.duration_seconds} events={timelineEvents} />
        <AnalysisDetails
          detector={detector}
          trackingVersion={trackingVersion}
          poseProvider={poseProvider}
          result={result}
          releaseJsonUrl={resultResponse.release_json_url}
        />
      </div>
    );
  }

  return (
    <div className="mt-4 space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(16rem,22rem)]">
        <div className="rounded-xl border border-lime/25 bg-lime/[0.04] p-4">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <p className="text-[11px] font-black uppercase tracking-[0.14em] text-lime">
                Release Point
              </p>
              <div className="mt-3 grid grid-cols-2 gap-4 sm:grid-cols-[8rem_8rem_minmax(10rem,1fr)]">
                <div>
                  <span className="block text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
                    Frame
                  </span>
                  <strong className="mt-1 block text-3xl font-black tabular-nums">
                    {formatFrame(result.release_frame)}
                  </strong>
                </div>
                <div>
                  <span className="block text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
                    Time
                  </span>
                  <strong className="mt-1 block text-3xl font-black tabular-nums">
                    {formatTime(result.release_time_seconds)}
                  </strong>
                </div>
                <div className="col-span-2 sm:col-span-1">
                  <span className="block text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
                    Evidence
                  </span>
                  <strong className="mt-1 block text-lg font-black">
                    {evidenceModeLabel(result.evidence_mode)}
                  </strong>
                  <p className="mt-1 text-sm leading-5 text-white/45">
                    {evidenceModeDescription(result.evidence_mode)}
                  </p>
                </div>
              </div>
              {interval && (
                <p className="mt-4 rounded-lg border border-white/10 bg-black/20 px-3 py-2 text-sm text-white/60">
                  Likely interval: {interval}
                </p>
              )}
            </div>
            <div className="min-w-[9rem] rounded-xl border border-white/10 bg-black/25 p-3 text-right">
              <span className="block text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
                Confidence
              </span>
              <strong className="mt-1 block text-3xl font-black tabular-nums">
                {formatConfidence(result.confidence)}
              </strong>
              <StatusBadge label={confidenceLabel(result.confidence)} tone={confidenceTone(band)} />
            </div>
          </div>
        </div>
        <aside className="grid gap-2 sm:grid-cols-3 xl:grid-cols-1">
          <div className="rounded-lg bg-black/25 p-2.5">
            <span className="block text-[10px] text-white/35">Release type</span>
            <strong className="mt-0.5 block text-sm">{evidenceModeLabel(result.evidence_mode)}</strong>
          </div>
          <div className="rounded-lg bg-black/25 p-2.5">
            <span className="block text-[10px] text-white/35">Track quality</span>
            <strong className="mt-0.5 block text-sm capitalize">{trackingResult.summary.track_quality}</strong>
          </div>
          <div className="rounded-lg bg-black/25 p-2.5">
            <span className="block text-[10px] text-white/35">Pixel position</span>
            <strong className="mt-0.5 block text-sm tabular-nums">
              {result.release_point_px
                ? `${result.release_point_px.x.toFixed(1)}, ${result.release_point_px.y.toFixed(1)}`
                : "-"}
            </strong>
          </div>
        </aside>
      </div>

      <ReleaseVideo analysis={analysis} result={result} />
      <DeliveryEventTimeline durationSeconds={analysis.duration_seconds} events={timelineEvents} />
      <AnalysisDetails
        detector={detector}
        trackingVersion={trackingVersion}
        poseProvider={poseProvider}
        result={result}
        releaseJsonUrl={resultResponse.release_json_url}
      />
    </div>
  );
}


function AnalysisDetails({
  detector,
  trackingVersion,
  poseProvider,
  result,
  releaseJsonUrl
}: {
  detector: string | null;
  trackingVersion: string;
  poseProvider: string | null;
  result: ReleaseResult;
  releaseJsonUrl: string;
}) {
  const qualityFlags = result.quality_flags.length
    ? result.quality_flags
    : ["No quality warnings reported."];

  return (
    <details className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
      <summary className="cursor-pointer text-sm font-bold text-white/55">
        Analysis Details
      </summary>
      <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg bg-black/25 p-2.5">
          <span className="block text-[10px] text-white/35">Method</span>
          <strong className="mt-0.5 block text-sm">{readableMethod(result.method)}</strong>
        </div>
        <div className="rounded-lg bg-black/25 p-2.5">
          <span className="block text-[10px] text-white/35">Detector</span>
          <strong className="mt-0.5 block text-sm">{detector ?? "Not reported"}</strong>
        </div>
        <div className="rounded-lg bg-black/25 p-2.5">
          <span className="block text-[10px] text-white/35">Tracking</span>
          <strong className="mt-0.5 block text-sm">{trackingVersion}</strong>
        </div>
        <div className="rounded-lg bg-black/25 p-2.5">
          <span className="block text-[10px] text-white/35">Pose provider</span>
          <strong className="mt-0.5 block text-sm">{poseProvider ?? "Not available"}</strong>
        </div>
      </div>
      <div className="mt-3 rounded-lg border border-white/10 bg-black/20 p-3">
        <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">
          Quality warnings
        </p>
        <ul className="mt-2 space-y-1 text-sm leading-6 text-white/55">
          {qualityFlags.map((flag) => (
            <li key={flag}>{flag.replace(/_/g, " ")}</li>
          ))}
        </ul>
      </div>
      <a
        className="mt-3 inline-flex rounded-lg border border-white/15 bg-white/5 px-2.5 py-1.5 text-[11px] font-bold text-lime hover:bg-white/10"
        href={releaseJsonUrl}
        target="_blank"
        rel="noreferrer"
      >
        Open release report JSON
      </a>
    </details>
  );
}


export function ReleasePointPanel({
  analysis,
  trackingResult,
  initialResult,
  onResult
}: {
  analysis: VideoAnalysisPreparedResponse;
  trackingResult: VideoBallTrackingResultResponse;
  initialResult: ReleaseResultResponse | null;
  onResult?: (result: ReleaseResultResponse | null) => void;
}) {
  const pollGeneration = useRef(0);
  const [job, setJob] = useState<JobView | null>(null);
  const [result, setResult] = useState(initialResult);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  function stopPolling() {
    pollGeneration.current += 1;
  }

  async function restoreResult(): Promise<boolean> {
    const completed = await getReleasePointResult(analysis.analysis_id);
    if (!completed) return false;
    setResult(completed);
    onResult?.(completed);
    setJob(null);
    setError(completed.status === "unresolved" ? completed.message : null);
    return true;
  }

  function pollJob(jobId: string) {
    const generation = pollGeneration.current + 1;
    pollGeneration.current = generation;

    const poll = async () => {
      try {
        const current = await getReleasePointJob(analysis.analysis_id, jobId);
        if (pollGeneration.current !== generation) return;
        setJob({
          jobId: current.job_id,
          status: current.status,
          progress: current.progress,
          message: current.message
        });
        if (current.status === "ready" || current.status === "unresolved") {
          await restoreResult();
          return;
        }
        if (current.status === "failed") {
          setError(current.error_message ?? current.message);
          return;
        }
        window.setTimeout(poll, 1000);
      } catch (caught) {
        if (pollGeneration.current !== generation) return;
        try {
          if (await restoreResult()) return;
        } catch {
          // Report the original polling error below.
        }
        setJob(null);
        setError(
          caught instanceof Error
            ? caught.message
            : "Release Point Analysis status could not be restored."
        );
      }
    };
    void poll();
  }

  async function runReleasePointAnalysis() {
    stopPolling();
    setStarting(true);
    setError(null);
    try {
      const started = await startReleasePointAnalysis(analysis.analysis_id);
      setResult(null);
      onResult?.(null);
      setJob({
        jobId: started.job_id,
        status: started.status,
        progress: started.progress,
        message: started.message
      });
      pollJob(started.job_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Release Point Analysis could not be started.");
    } finally {
      setStarting(false);
    }
  }

  const active = starting || (job !== null && ACTIVE_JOB_STATUSES.has(job.status));
  const unresolved = result?.status === "unresolved";
  const ready = result?.status === "ready";
  const lowConfidence = ready && result.result.confidence < 0.6;

  useEffect(() => {
    return () => {
      pollGeneration.current += 1;
    };
  }, []);

  return (
    <Card className="border-lime/20 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-black tracking-tight sm:text-xl">Release Point Analysis</h2>
            <StatusBadge
              label={
                ready
                  ? lowConfidence
                    ? "Low Confidence"
                    : "Ready"
                  : unresolved
                    ? "Unresolved"
                    : active
                      ? "Processing"
                      : error
                        ? "Failed"
                        : "Ready"
              }
              tone={error || unresolved || lowConfidence ? "warn" : ready ? "good" : "neutral"}
            />
          </div>
          <p className="mt-1 text-sm text-white/45">
            Release frame, timestamp, and pixel location from completed tracking
            {job ? ` - ${statusLabel(job.status)}` : ""}
          </p>
        </div>
        <Button disabled={active} onClick={() => void runReleasePointAnalysis()}>
          {starting
            ? "Starting..."
            : active
              ? "Analysing..."
              : result
                ? "Run Again"
                : "Analyse Release Point"}
        </Button>
      </div>

      {!result && !job && !error && (
        <div className="mt-4 rounded-xl border border-white/10 bg-black/20 p-4">
          <p className="text-sm font-bold text-white">Tracking complete</p>
          <p className="mt-1 max-w-2xl text-sm leading-6 text-white/45">
            Start Release Point V1 to identify the likely ball release moment for this delivery.
          </p>
        </div>
      )}

      {job && (
        <div className="mt-3 rounded-xl border border-lime/20 bg-lime/[0.04] p-3">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <p className="text-sm font-bold text-lime">{job.message}</p>
            {Number.isFinite(job.progress) && (
              <span className="text-sm font-black">{job.progress}%</span>
            )}
          </div>
          {Number.isFinite(job.progress) && (
            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full bg-lime transition-[width]" style={{ width: `${job.progress}%` }} />
            </div>
          )}
          <div className="mt-3 grid gap-2 sm:grid-cols-3">
            {PROCESSING_STEPS.map((step) => (
              <div key={step} className="rounded-lg bg-black/25 px-3 py-2 text-xs font-bold text-white/55">
                {step}
              </div>
            ))}
          </div>
        </div>
      )}

      {error && (
        <p className="mt-3 rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]">{error}</p>
      )}

      {result && (
        <ReleaseResultView
          analysis={analysis}
          trackingResult={trackingResult}
          resultResponse={result}
        />
      )}
    </Card>
  );
}
