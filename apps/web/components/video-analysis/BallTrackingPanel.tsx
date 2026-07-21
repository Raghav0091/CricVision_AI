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

import { MEDIA_FIT_CLASS } from "./AnalysisMediaStage";


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
    <div className="mt-4 space-y-4">
      {result.status === "no_reliable_track" && (
        <p className="rounded-lg border border-[#ffca68]/30 bg-[#ffca68]/[0.05] px-3 py-2 text-sm leading-6 text-[#ffdc9a]">
          No coherent moving-ball track met the reliability threshold. Raw detection results remain available.
        </p>
      )}

      {summary.delivery_replay_url && (
        <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(15rem,20rem)]">
          <div className="min-w-0 rounded-xl border border-lime/25 bg-lime/[0.04] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-[11px] font-bold uppercase tracking-[0.12em] text-lime">Delivery Replay</p>
              <div className="flex gap-1.5">
                {[0.25, 0.5, 1].map((rate) => (
                  <button
                    key={rate}
                    type="button"
                    className="rounded-lg border border-white/15 bg-white/5 px-2 py-1 text-[11px] font-bold text-white/70 hover:bg-white/10"
                    onClick={() => setReplayRate(rate)}
                  >
                    {rate}x
                  </button>
                ))}
              </div>
            </div>
            <div className="mt-2 flex max-h-[min(42dvh,calc(100dvh-16rem))] min-h-[8rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08] sm:max-h-[min(52dvh,calc(100dvh-14rem))]">
              <video
                ref={replayRef}
                className={MEDIA_FIT_CLASS}
                controls
                preload="metadata"
                src={summary.delivery_replay_url}
              />
            </div>
          </div>
          <aside className="flex min-w-0 flex-col gap-2 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-5rem)] xl:self-start xl:overflow-y-auto">
            <div className="rounded-lg bg-black/25 p-2.5 text-sm">
              <span className="block text-[10px] text-white/35">Track quality</span>
              <strong className="mt-0.5 block capitalize">{summary.track_quality}</strong>
            </div>
            <div className="rounded-lg bg-black/25 p-2.5 text-sm">
              <span className="block text-[10px] text-white/35">Bounce</span>
              <strong className="mt-0.5 block">{bounceLabel}</strong>
            </div>
            <div className="rounded-lg bg-black/25 p-2.5 text-sm">
              <span className="block text-[10px] text-white/35">Observed / recovered</span>
              <strong className="mt-0.5 block">
                {summary.observed_track_points} / {summary.recovered_points}
              </strong>
            </div>
            <div className="rounded-lg bg-black/25 p-2.5 text-sm">
              <span className="block text-[10px] text-white/35">Bounce confidence</span>
              <strong className="mt-0.5 block">
                {(summary.bounce_confidence ?? bounce?.confidence ?? 0).toFixed(2)}
              </strong>
            </div>
            <div className="flex flex-wrap gap-2 pt-1">
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
      )}

      <details className="rounded-xl border border-white/10 bg-black/20 px-3 py-2">
        <summary className="cursor-pointer text-sm font-bold text-white/55">
          Debug videos &amp; stats
        </summary>
        <div className="mt-3 space-y-3">
          <div className="grid gap-3 md:grid-cols-3">
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">Original</p>
              <div className="mt-1.5 flex min-h-[6rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08]">
                <video className={MEDIA_FIT_CLASS} controls preload="metadata" src={analysis.original_video_url} />
              </div>
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">Detection</p>
              <div className="mt-1.5 flex min-h-[6rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08]">
                <video className={MEDIA_FIT_CLASS} controls preload="metadata" src={detection.summary.processed_video_url} />
              </div>
            </div>
            <div>
              <p className="text-[10px] font-bold uppercase tracking-[0.12em] text-white/35">Tracking debug</p>
              <div className="mt-1.5 flex min-h-[6rem] items-center justify-center overflow-hidden rounded-xl bg-[#050a08]">
                <video className={MEDIA_FIT_CLASS} controls preload="metadata" src={summary.tracking_video_url} />
              </div>
            </div>
          </div>
          <div className="flex flex-wrap gap-3 text-[11px] text-white/55">
            <span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full border-2 border-[#50e650]" />Observed</span>
            <span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full border-2 border-[#ff9600]" />Recovered</span>
            <span><i className="mr-1.5 inline-block h-2.5 w-2.5 rounded-full border-2 border-[#ffe600]" />Projected</span>
          </div>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {stats.map(([label, value]) => (
              <div key={label} className="rounded-lg bg-black/25 p-2.5">
                <span className="block text-[10px] text-white/35">{label}</span>
                <strong className="mt-0.5 block text-sm capitalize">{value}</strong>
              </div>
            ))}
          </div>
          {bounce && bounce.evidence.length > 0 && (
            <p className="text-[11px] text-white/45">
              Bounce evidence: {bounce.evidence.join(", ")}
              {bounce.warnings.length ? ` · Warnings: ${bounce.warnings.join("; ")}` : ""}
            </p>
          )}
          {!summary.delivery_replay_url && (
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
          )}
        </div>
      </details>
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
    <Card className="border-lime/20 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-black tracking-tight sm:text-xl">Ball Tracking</h2>
            <StatusBadge
              label={
                ready
                  ? "Ready"
                  : noReliableTrack
                    ? "Needs Attention"
                    : active
                      ? "Processing"
                      : error
                        ? "Failed"
                        : "Ready"
              }
              tone={error || noReliableTrack ? "warn" : ready ? "good" : "neutral"}
            />
          </div>
          <p className="mt-1 text-sm text-white/45">
            Primary delivery track from saved detections
            {job ? ` · ${statusLabel(job.status)}` : ""}
          </p>
        </div>
        <Button disabled={active} onClick={() => void runTracking()}>
          {starting ? "Starting…" : active ? "Tracker running…" : result ? "Run Again" : "Run Tracking"}
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
        </div>
      )}

      {error && (
        <p className="mt-3 rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm leading-6 text-[#ffaaa6]">{error}</p>
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
