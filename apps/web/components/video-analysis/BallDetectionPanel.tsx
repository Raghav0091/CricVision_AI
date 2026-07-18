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
    <div className="mt-6 space-y-6">
      <div className="grid gap-5 xl:grid-cols-2">
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Original Video</p>
          <video className="mt-3 aspect-video w-full rounded-lg bg-black object-contain" controls preload="metadata" src={analysis.original_video_url} />
        </div>
        <div className="rounded-xl border border-lime/20 bg-lime/[0.03] p-4">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-lime">Raw Detection Video</p>
          <video className="mt-3 aspect-video w-full rounded-lg bg-black object-contain" controls preload="metadata" src={summary.processed_video_url} />
          <p className="mt-2 text-xs text-white/40">Complete every-frame overlay. Raw candidates only; no tracking.</p>
        </div>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {stats.map(([label, value]) => (
          <div key={label} className="rounded-xl bg-black/20 p-3">
            <span className="block text-xs text-white/35">{label}</span>
            <strong className="mt-1 block text-lg">{value}</strong>
          </div>
        ))}
      </div>

      <DetectionCoverageTimeline counts={result.frame_candidate_counts} />

      <div className="grid gap-4 rounded-xl border border-white/10 bg-black/20 p-4 sm:grid-cols-2 xl:grid-cols-4">
        <div><span className="block text-xs text-white/35">Model used</span><strong className="mt-1 block break-all text-sm">{summary.model_path_used}</strong></div>
        <div><span className="block text-xs text-white/35">Processing device</span><strong className="mt-1 block text-sm">{summary.device_used}</strong></div>
        <div><span className="block text-xs text-white/35">Inference settings</span><strong className="mt-1 block text-sm">{summary.imgsz}px · conf {summary.confidence_threshold} · stride {summary.frame_stride}</strong></div>
        <div><span className="block text-xs text-white/35">Processing duration</span><strong className="mt-1 block text-sm">{summary.processing_duration_seconds.toFixed(2)}s</strong></div>
      </div>

      {summary.model_warning && (
        <p className="rounded-xl border border-[#ffca68]/30 bg-[#ffca68]/[0.05] p-4 text-sm text-[#ffdc9a]">{summary.model_warning}</p>
      )}

      <div className="flex flex-wrap gap-2">
        {links.map(([label, url]) => (
          <a key={label} className="rounded-lg border border-white/15 bg-white/5 px-3 py-2 text-xs font-bold text-lime hover:bg-white/10" href={url} target="_blank" rel="noreferrer">
            {label}
          </a>
        ))}
      </div>
    </div>
  );
}


export function BallDetectionPanel({
  analysis,
  initialResult,
  initialJobId
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialResult: VideoBallDetectionResultResponse | null;
  initialJobId?: string | null;
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
    <Card className="border-lime/20">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={result ? "Ball Detection — Ready" : job ? `Ball Detection — ${statusLabel(job.status)}` : "Ball Detection — Available"}
            tone={error ? "warn" : result ? "good" : "neutral"}
          />
          <h2 className="mt-4 text-2xl font-black">Every-Frame Ball Detection</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/50">
            Run the trained ball detector on every original frame and preserve every raw ball candidate. This stage does not track, connect, or interpolate detections.
          </p>
        </div>
        <Button disabled={isActive} onClick={() => void runDetection()}>
          {starting ? "Starting…" : isActive ? "Detection running…" : result ? "Run Ball Detection Again" : "Run Ball Detection"}
        </Button>
      </div>

      <div className="mt-5 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
        {[
          ["Model", "ball_only_E2_1280_baseline.pt"],
          ["Image size", "960"],
          ["Confidence", "0.15"],
          ["Frame stride", "1"],
          ["Coverage", "Every frame"]
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl bg-black/20 p-3">
            <span className="block text-xs text-white/35">{label}</span>
            <strong className="mt-1 block break-all text-sm">{value}</strong>
          </div>
        ))}
      </div>

      {job && (
        <div className="mt-5 rounded-xl border border-lime/20 bg-lime/[0.04] p-4">
          <div className="flex items-center justify-between gap-4">
            <p className="font-bold text-lime">{job.message}</p>
            <span className="text-sm font-black">{progress}%</span>
          </div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/10">
            <div className="h-full rounded-full bg-lime transition-[width]" style={{ width: `${progress}%` }} />
          </div>
          <div className="mt-2 flex flex-wrap justify-between gap-2 text-xs text-white/40">
            <span>{statusLabel(job.status)}</span>
            <span>Frame {job.currentFrame.toLocaleString()} of {job.totalFrames.toLocaleString()}</span>
          </div>
        </div>
      )}

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm leading-6 text-[#ffaaa6]">{error}</p>
      )}

      {result && <DetectionResult analysis={analysis} result={result} />}
    </Card>
  );
}
