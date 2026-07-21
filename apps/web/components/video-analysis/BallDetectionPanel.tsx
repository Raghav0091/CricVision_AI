"use client";

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getVideoBallDetectionJob,
  getVideoBallDetectionResult,
  startVideoBallDetection,
  type VideoAnalysisPreparedResponse,
  type VideoBallDetectionJobStatus,
  type VideoBallDetectionResultResponse
} from "@/lib/api";

import { MEDIA_FIT_CLASS } from "./AnalysisMediaStage";


type JobView = {
  jobId: string;
  status: VideoBallDetectionJobStatus;
  progress: number;
  currentFrame: number;
  totalFrames: number;
  message: string;
};


const ACTIVE_JOB_STATUSES = new Set<VideoBallDetectionJobStatus>([
  "queued",
  "loading_model",
  "processing",
  "writing_video",
  "saving_results"
]);


function statusLabel(status: VideoBallDetectionJobStatus): string {
  const labels: Record<VideoBallDetectionJobStatus, string> = {
    queued: "Queued",
    loading_model: "Loading model",
    processing: "Processing",
    writing_video: "Generating video",
    saving_results: "Saving results",
    ready: "Ready",
    failed: "Failed",
    ball_detector_missing: "Model missing"
  };
  return labels[status];
}


function confidence(value: number): string {
  return value > 0 ? value.toFixed(3) : "—";
}


function DetectionCoverageTimeline({ counts }: { counts: number[] }) {
  const groupSize = Math.max(1, Math.ceil(counts.length / 120));
  const groups: number[] = [];
  for (let index = 0; index < counts.length; index += groupSize) {
    groups.push(Math.max(...counts.slice(index, index + groupSize)));
  }

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h3 className="font-black">Detection coverage by frame</h3>
        <div className="flex gap-3 text-[11px] text-white/45">
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-white/15" />Zero</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-lime" />One</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-sm bg-[#ffca68]" />Multiple</span>
        </div>
      </div>
      <div
        className="mt-3 grid h-10 gap-px overflow-hidden rounded-lg bg-black/30 p-1"
        style={{ gridTemplateColumns: `repeat(${groups.length}, minmax(2px, 1fr))` }}
        aria-label={`Detection coverage for ${counts.length} frames`}
      >
        {groups.map((count, index) => (
          <span
            key={index}
            className={
              count === 0
                ? "rounded-sm bg-white/15"
                : count === 1
                  ? "rounded-sm bg-lime"
                  : "rounded-sm bg-[#ffca68]"
            }
            title={`Frames ${index * groupSize + 1}–${Math.min((index + 1) * groupSize, counts.length)}: ${count === 0 ? "no candidates" : count === 1 ? "one candidate" : "multiple candidates"}`}
          />
        ))}
      </div>
      {groupSize > 1 && (
        <p className="mt-2 text-xs text-white/35">
          Long videos are grouped into blocks of {groupSize} frames; each block shows its highest candidate count.
        </p>
      )}
    </div>
  );
}


function DetectionResult({
  analysis,
  result
}: {
  analysis: VideoAnalysisPreparedResponse;
  result: VideoBallDetectionResultResponse;
}) {
  const summary = result.summary;
  const stats = [
    ["Total frames", summary.total_frames.toLocaleString()],
    ["Frames processed", summary.frames_processed.toLocaleString()],
    ["With candidates", summary.frames_with_candidates.toLocaleString()],
    ["Without candidates", summary.frames_without_candidates.toLocaleString()],
    ["Total candidates", summary.total_candidates.toLocaleString()],
    ["Multiple-candidate frames", summary.frames_with_multiple_candidates.toLocaleString()],
    ["Best confidence", confidence(summary.best_confidence)],
    ["Average confidence", confidence(summary.average_confidence)],
    ["Inside pitch corridor", summary.candidates_inside_pitch_corridor.toLocaleString()],
    ["Outside pitch corridor", summary.candidates_outside_pitch_corridor.toLocaleString()]
  ];
  const links = [
    ["Download detections JSON", summary.detections_json_url],
    ["Download detections CSV", summary.detections_csv_url],
    ["Download summary JSON", summary.detection_summary_url],
    ["Open processed video", summary.processed_video_url]
  ];

  return (
    <div className="mt-4 space-y-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(15rem,20rem)]">
        <div className="min-w-0 rounded-xl border border-lime/20 bg-lime/[0.03] p-3">
          <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-lime">Raw Detection Video</p>
          <div className="mt-2 flex max-h-[min(42dvh,calc(100dvh-16rem))] min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08] sm:max-h-[min(52dvh,calc(100dvh-14rem))]">
            <video className={MEDIA_FIT_CLASS} controls preload="metadata" src={summary.processed_video_url} />
          </div>
          <p className="mt-2 text-xs text-white/40">Every-frame overlay · raw candidates only</p>
        </div>
        <aside className="flex min-w-0 flex-col gap-3 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-5rem)] xl:self-start xl:overflow-y-auto">
          <div className="grid grid-cols-2 gap-2">
            {stats.slice(0, 6).map(([label, value]) => (
              <div key={label} className="rounded-lg bg-black/25 p-2.5">
                <span className="block text-[10px] text-white/35">{label}</span>
                <strong className="mt-0.5 block text-sm">{value}</strong>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap gap-2">
            {links.map(([label, url]) => (
              <a
                key={label}
                className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1.5 text-[11px] font-bold text-lime hover:bg-white/10"
                href={url}
                target="_blank"
                rel="noreferrer"
              >
                {label}
              </a>
            ))}
          </div>
        </aside>
      </div>

      <details className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
        <summary className="cursor-pointer text-sm font-bold text-white/55">Original video &amp; details</summary>
        <div className="mt-3 space-y-3">
          <div className="flex min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08]">
            <video className={MEDIA_FIT_CLASS} controls preload="metadata" src={analysis.original_video_url} />
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {stats.slice(6).map(([label, value]) => (
              <div key={label} className="rounded-lg bg-black/25 p-2.5">
                <span className="block text-[10px] text-white/35">{label}</span>
                <strong className="mt-0.5 block text-sm">{value}</strong>
              </div>
            ))}
          </div>
          <div className="grid gap-3 rounded-lg border border-white/10 bg-black/20 p-3 sm:grid-cols-2 xl:grid-cols-4">
            <div><span className="block text-[10px] text-white/35">Model</span><strong className="mt-0.5 block break-all text-xs">{summary.model_path_used}</strong></div>
            <div><span className="block text-[10px] text-white/35">Device</span><strong className="mt-0.5 block text-xs">{summary.device_used}</strong></div>
            <div><span className="block text-[10px] text-white/35">Settings</span><strong className="mt-0.5 block text-xs">{summary.imgsz}px · conf {summary.confidence_threshold} · stride {summary.frame_stride}</strong></div>
            <div><span className="block text-[10px] text-white/35">Duration</span><strong className="mt-0.5 block text-xs">{summary.processing_duration_seconds.toFixed(2)}s</strong></div>
          </div>
        </div>
      </details>

      <DetectionCoverageTimeline counts={result.frame_candidate_counts} />

      {summary.model_warning && (
        <p className="rounded-lg border border-[#ffca68]/30 bg-[#ffca68]/[0.05] px-3 py-2 text-sm text-[#ffdc9a]">{summary.model_warning}</p>
      )}
    </div>
  );
}


export function BallDetectionPanel({
  analysis,
  initialResult,
  initialJobId,
  onResult
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialResult: VideoBallDetectionResultResponse | null;
  initialJobId?: string | null;
  onResult?: (result: VideoBallDetectionResultResponse | null) => void;
}) {
  const pollGeneration = useRef(0);
  const [job, setJob] = useState<JobView | null>(null);
  const [result, setResult] = useState(initialResult);
  const [error, setError] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);

  function stopPolling() {
    pollGeneration.current += 1;
  }

  async function restoreCompletedResult(): Promise<boolean> {
    const completed = await getVideoBallDetectionResult(analysis.analysis_id);
    if (!completed) return false;
    setResult(completed);
    onResult?.(completed);
    setJob(null);
    setError(null);
    return true;
  }

  function pollJob(jobId: string) {
    const generation = pollGeneration.current + 1;
    pollGeneration.current = generation;

    const poll = async () => {
      try {
        const current = await getVideoBallDetectionJob(analysis.analysis_id, jobId);
        if (pollGeneration.current !== generation) return;
        setJob({
          jobId: current.job_id,
          status: current.status,
          progress: current.progress,
          currentFrame: current.current_frame,
          totalFrames: current.total_frames,
          message: current.message
        });
        if (current.status === "ready") {
          await restoreCompletedResult();
          return;
        }
        if (current.status === "failed" || current.status === "ball_detector_missing") {
          setError(current.error_message ?? current.message);
          return;
        }
        window.setTimeout(poll, 1000);
      } catch (caught) {
        if (pollGeneration.current !== generation) return;
        try {
          if (await restoreCompletedResult()) return;
        } catch {
          // Report the original job polling error below.
        }
        setJob(null);
        setError(
          caught instanceof Error
            ? caught.message
            : "Ball detection status could not be restored."
        );
      }
    };
    void poll();
  }

  async function runDetection() {
    stopPolling();
    setStarting(true);
    setError(null);
    try {
      const started = await startVideoBallDetection(analysis.analysis_id);
      setResult(null);
      onResult?.(null);
      setJob({
        jobId: started.job_id,
        status: started.status,
        progress: started.progress,
        currentFrame: started.current_frame,
        totalFrames: started.total_frames,
        message: started.message
      });
      pollJob(started.job_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Ball detection could not be started.");
    } finally {
      setStarting(false);
    }
  }

  useEffect(() => {
    if (!initialResult && initialJobId) pollJob(initialJobId);
    return () => {
      pollGeneration.current += 1;
    };
    // The panel is keyed by analysis ID, so these initial values are intentionally read once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isActive = starting || (job !== null && ACTIVE_JOB_STATUSES.has(job.status));
  const progress = job?.progress ?? (result ? 100 : 0);

  return (
    <Card className="border-lime/20 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-black tracking-tight sm:text-xl">Ball Detection</h2>
            <StatusBadge
              label={error ? "Failed" : result ? "Ready" : isActive ? "Processing" : "Ready"}
              tone={error ? "warn" : result ? "good" : "neutral"}
            />
          </div>
          <p className="mt-1 text-sm text-white/45">
            Every-frame raw candidates · no tracking yet
            {job ? ` · ${statusLabel(job.status)}` : ""}
          </p>
        </div>
        <Button disabled={isActive} onClick={() => void runDetection()}>
          {starting ? "Starting…" : isActive ? "Detection running…" : result ? "Run Again" : "Run Detection"}
        </Button>
      </div>

      {job && (
        <div className="mt-3 rounded-xl border border-lime/20 bg-lime/[0.04] p-3">
          <div className="flex items-center justify-between gap-3">
            <p className="text-sm font-bold text-lime">{job.message}</p>
            <span className="text-sm font-black">{progress}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full bg-lime transition-[width]" style={{ width: `${progress}%` }} />
          </div>
          <p className="mt-1.5 text-[11px] text-white/40">
            Frame {job.currentFrame.toLocaleString()} / {job.totalFrames.toLocaleString()}
          </p>
        </div>
      )}

      {error && (
        <p className="mt-3 rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]">{error}</p>
      )}

      {result && <DetectionResult analysis={analysis} result={result} />}
    </Card>
  );
}
