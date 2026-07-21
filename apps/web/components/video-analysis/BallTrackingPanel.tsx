"use client";

import { useEffect, useRef, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  getVideoBallTrackingJob,
  getVideoBallTrackingResult,
  startVideoBallTracking,
  type VideoAnalysisPreparedResponse,
  type VideoBallDetectionResultResponse,
  type VideoBallTrackingJobStatus,
  type VideoBallTrackingResultResponse
} from "@/lib/api";


type JobView = {
  jobId: string;
  status: VideoBallTrackingJobStatus;
  progress: number;
  message: string;
};


const ACTIVE_STATUSES = new Set<VideoBallTrackingJobStatus>([
  "queued",
  "loading_detections",
  "analysing_candidates",
  "building_track",
  "recovering_gaps",
  "rendering_video",
  "saving_results"
]);


function statusLabel(status: VideoBallTrackingJobStatus): string {
  const labels: Record<VideoBallTrackingJobStatus, string> = {
    queued: "Queued",
    loading_detections: "Loading detections",
    analysing_candidates: "Analysing candidates",
    building_track: "Building track",
    recovering_gaps: "Recovering gaps",
    rendering_video: "Generating video",
    saving_results: "Saving results",
    ready: "Ready",
    failed: "Failed",
    no_reliable_track: "No reliable track"
  };
  return labels[status];
}


function frameRange(start?: number | null, end?: number | null): string {
  return start == null || end == null ? "—" : `${start}–${end}`;
}


function TrackingResult({
  analysis,
  detection,
  result
}: {
  analysis: VideoAnalysisPreparedResponse;
  detection: VideoBallDetectionResultResponse;
  result: VideoBallTrackingResultResponse;
}) {
  const summary = result.summary;
  const bounce = result.bounce;
  const replayRef = useRef<HTMLVideoElement | null>(null);
  const bounceLabel =
    summary.bounce_detected === true
      ? `Yes (frame ${summary.bounce_frame ?? "—"})`
      : summary.bounce_detected === false
        ? "No"
        : "Uncertain";
  const stats = [
    ["Raw candidates", summary.raw_candidate_count.toLocaleString()],
    ["Observed points", summary.observed_track_points.toLocaleString()],
    ["Recovered points", summary.recovered_points.toLocaleString()],
    ["Physics reconstructed", String(summary.physics_reconstructed_points ?? 0)],
    ["Track frames", frameRange(summary.track_start_frame, summary.track_end_frame)],
    ["Longest gap", `${summary.longest_gap_frames} frames`],
    ["Observation ratio", `${((summary.observation_ratio ?? 0) * 100).toFixed(0)}%`],
    ["Track confidence", summary.track_confidence.toFixed(2)],
    ["Track quality", summary.track_quality]
  ];
  const links = [
    ["Download tracking JSON", summary.tracking_json_url],
    ["Download tracking CSV", summary.tracking_csv_url],
    ["Download tracking summary", summary.tracking_summary_url],
    ["Open debug tracking video", summary.tracking_video_url],
    ...(summary.delivery_replay_url
      ? [["Open delivery replay", summary.delivery_replay_url] as const]
      : [])
  ];

  function setReplayRate(rate: number) {
    if (replayRef.current) {
      replayRef.current.playbackRate = rate;
    }
  }

  return (
    <div className="mt-6 space-y-6">
      {result.status === "no_reliable_track" && (
        <p className="rounded-xl border border-[#ffca68]/30 bg-[#ffca68]/[0.05] p-4 text-sm leading-6 text-[#ffdc9a]">
          No coherent moving-ball track met the reliability threshold. Raw detection results remain available and unchanged.
        </p>
      )}

      {summary.delivery_replay_url && (
        <div className="rounded-xl border border-lime/25 bg-lime/[0.04] p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-xs font-bold uppercase tracking-[0.15em] text-lime">Delivery Replay</p>
              <p className="mt-1 text-sm text-white/55">
                Clean original frames with ball highlight, short trail, and bounce marker. Track quality:{" "}
                <strong className="capitalize text-white/80">{summary.track_quality}</strong>
                {" · "}
                Bounce: <strong className="text-white/80">{bounceLabel}</strong>
              </p>
            </div>
            <div className="flex gap-2">
              {[0.25, 0.5, 1].map((rate) => (
                <button
                  key={rate}
                  type="button"
                  className="rounded-lg border border-white/15 bg-white/5 px-2.5 py-1 text-xs font-bold text-white/70 hover:bg-white/10"
                  onClick={() => setReplayRate(rate)}
                >
                  {rate}x
                </button>
              ))}
            </div>
          </div>
          <video
            ref={replayRef}
            className="mt-3 aspect-video w-full rounded-lg bg-black object-contain"
            controls
            preload="metadata"
            src={summary.delivery_replay_url}
          />
          <div className="mt-3 grid gap-3 sm:grid-cols-3 text-sm">
            <div className="rounded-lg bg-black/25 p-3">
              <span className="block text-xs text-white/35">Observed / recovered</span>
              <strong className="mt-1 block">
                {summary.observed_track_points} / {summary.recovered_points}
              </strong>
            </div>
            <div className="rounded-lg bg-black/25 p-3">
              <span className="block text-xs text-white/35">Bounce confidence</span>
              <strong className="mt-1 block">
                {(summary.bounce_confidence ?? bounce?.confidence ?? 0).toFixed(2)}
              </strong>
            </div>
            <div className="rounded-lg bg-black/25 p-3">
              <span className="block text-xs text-white/35">Track start (not true release)</span>
              <strong className="mt-1 block">
                {summary.first_supported_delivery_point ?? "—"}
              </strong>
            </div>
          </div>
        </div>
      )}

      <div className="grid gap-5 xl:grid-cols-3">
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Original Video</p>
          <video className="mt-3 aspect-video w-full rounded-lg bg-black object-contain" controls preload="metadata" src={analysis.original_video_url} />
        </div>
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Raw Detection Video</p>
          <video className="mt-3 aspect-video w-full rounded-lg bg-black object-contain" controls preload="metadata" src={detection.summary.processed_video_url} />
        </div>
        <div className="rounded-xl border border-white/10 bg-black/20 p-4">
          <p className="text-xs font-bold uppercase tracking-[0.15em] text-white/40">Tracking Debug Video</p>
          <video className="mt-3 aspect-video w-full rounded-lg bg-black object-contain" controls preload="metadata" src={summary.tracking_video_url} />
        </div>
      </div>

      <div className="flex flex-wrap gap-4 rounded-xl border border-white/10 bg-black/20 p-4 text-xs text-white/55">
        <span><i className="mr-2 inline-block h-3 w-3 rounded-full border-2 border-[#50e650]" />Green = OBSERVED</span>
        <span><i className="mr-2 inline-block h-3 w-3 rounded-full border-2 border-[#ff9600]" />Orange = TRACKER_RECOVERED</span>
        <span><i className="mr-2 inline-block h-3 w-3 rounded-full border-2 border-[#ffe600]" />Yellow = PROJECTED / physics</span>
        <strong className="text-white/70">Debug video shows provenance clearly; user replay keeps trail subtle.</strong>
      </div>

      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
        {stats.map(([label, value]) => (
          <div key={label} className="rounded-xl bg-black/20 p-3">
            <span className="block text-xs text-white/35">{label}</span>
            <strong className="mt-1 block text-lg capitalize">{value}</strong>
          </div>
        ))}
      </div>

      <div className="grid gap-4 rounded-xl border border-white/10 bg-black/20 p-4 sm:grid-cols-2 xl:grid-cols-4">
        <div><span className="block text-xs text-white/35">Approximate direction</span><strong className="mt-1 block text-sm capitalize">{summary.approximate_direction}</strong></div>
        <div><span className="block text-xs text-white/35">Primary bounce</span><strong className="mt-1 block text-sm">{bounceLabel}</strong></div>
        <div><span className="block text-xs text-white/35">Observed confidence</span><strong className="mt-1 block text-sm">{summary.average_observed_confidence.toFixed(3)}</strong></div>
        <div><span className="block text-xs text-white/35">Processing duration</span><strong className="mt-1 block text-sm">{summary.processing_duration_seconds.toFixed(2)}s</strong></div>
      </div>

      {bounce && bounce.evidence.length > 0 && (
        <p className="text-xs text-white/45">
          Bounce evidence: {bounce.evidence.join(", ")}
          {bounce.warnings.length ? ` · Warnings: ${bounce.warnings.join("; ")}` : ""}
        </p>
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


export function BallTrackingPanel({
  analysis,
  detectionResult,
  initialResult,
  initialJobId
}: {
  analysis: VideoAnalysisPreparedResponse;
  detectionResult: VideoBallDetectionResultResponse;
  initialResult: VideoBallTrackingResultResponse | null;
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

  async function restoreResult(): Promise<boolean> {
    const completed = await getVideoBallTrackingResult(analysis.analysis_id);
    if (!completed) return false;
    setResult(completed);
    setJob(null);
    setError(completed.status === "no_reliable_track" ? completed.message : null);
    return true;
  }

  function pollJob(jobId: string) {
    const generation = pollGeneration.current + 1;
    pollGeneration.current = generation;

    const poll = async () => {
      try {
        const current = await getVideoBallTrackingJob(analysis.analysis_id, jobId);
        if (pollGeneration.current !== generation) return;
        setJob({
          jobId: current.job_id,
          status: current.status,
          progress: current.progress,
          message: current.message
        });
        if (current.status === "ready" || current.status === "no_reliable_track") {
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
            : "Moving Ball Tracker status could not be restored."
        );
      }
    };
    void poll();
  }

  async function runTracking() {
    stopPolling();
    setStarting(true);
    setError(null);
    try {
      const started = await startVideoBallTracking(analysis.analysis_id);
      setResult(null);
      setJob({
        jobId: started.job_id,
        status: started.status,
        progress: started.progress,
        message: started.message
      });
      pollJob(started.job_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Moving Ball Tracker could not be started.");
    } finally {
      setStarting(false);
    }
  }

  useEffect(() => {
    if (!initialResult && initialJobId) pollJob(initialJobId);
    return () => {
      pollGeneration.current += 1;
    };
    // The panel is keyed by analysis ID, so initial restoration is intentionally read once.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const active = starting || (job !== null && ACTIVE_STATUSES.has(job.status));
  const progress = job?.progress ?? (result ? 100 : 0);
  const ready = result?.status === "ready";
  const noReliableTrack = result?.status === "no_reliable_track";

  return (
    <Card className="border-lime/20">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={ready ? "Ball Tracking — Ready" : noReliableTrack ? "Ball Tracking — No reliable track" : job ? `Ball Tracking — ${statusLabel(job.status)}` : "Ball Tracking — Available"}
            tone={error || noReliableTrack ? "warn" : ready ? "good" : "neutral"}
          />
          <h2 className="mt-4 text-2xl font-black">Complete Delivery Tracking</h2>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-white/50">
            Build one primary moving-ball track from the saved raw detections. YOLO is not rerun; stationary likelihood, motion continuity, adaptive gating, size, direction, and the pitch corridor soft score are evaluated together.
          </p>
        </div>
        <Button disabled={active} onClick={() => void runTracking()}>
          {starting ? "Starting…" : active ? "Tracker running…" : result ? "Run Delivery Tracking Again" : "Run Delivery Tracking"}
        </Button>
      </div>

      <div className="mt-5 grid gap-2 sm:grid-cols-2 xl:grid-cols-5">
        {[
          ["Input", "Saved detections.json"],
          ["Motion model", "Constant velocity"],
          ["Maximum gap", "6 frames"],
          ["Corridor", "Soft score only"],
          ["Output", "One primary track"]
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl bg-black/20 p-3">
            <span className="block text-xs text-white/35">{label}</span>
            <strong className="mt-1 block text-sm">{value}</strong>
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
          <p className="mt-2 text-xs text-white/40">{statusLabel(job.status)}</p>
        </div>
      )}

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm leading-6 text-[#ffaaa6]">{error}</p>
      )}

      {result && (
        <TrackingResult
          analysis={analysis}
          detection={detectionResult}
          result={result}
        />
      )}
    </Card>
  );
}
