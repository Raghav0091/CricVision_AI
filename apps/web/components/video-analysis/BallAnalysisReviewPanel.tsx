"use client";

import { Button } from "@/components/ui/Button";
import type { BallDetectorModelKey, BallDetectorModelOption } from "@/lib/api";
import type {
  BallReviewCandidate,
  BallReviewDisplayToggles
} from "@/lib/ball-analysis-review";
import type { VideoBallDetectionSummary, VideoBallTrackingResultResponse } from "@/lib/api";

type BallAnalysisPhase =
  | "idle"
  | "preparing_detector"
  | "detecting"
  | "filtering_static"
  | "selecting_candidates"
  | "tracking"
  | "recovering_gaps"
  | "scoring_primary"
  | "preparing_review"
  | "ready"
  | "failed";

const PHASE_LABELS: Record<BallAnalysisPhase, string> = {
  idle: "",
  preparing_detector: "Preparing detector",
  detecting: "Detecting candidates",
  filtering_static: "Filtering static",
  selecting_candidates: "Selecting moving",
  tracking: "Building track",
  recovering_gaps: "Recovering gaps",
  scoring_primary: "Scoring primary",
  preparing_review: "Preparing review",
  ready: "Analysis ready",
  failed: "Ball tracking unavailable"
};

function metricCard(label: string, value: string | number) {
  return (
    <div className="rounded-lg border border-white/10 bg-black/30 px-2.5 py-2">
      <span className="block text-[10px] uppercase tracking-wide text-white/35">{label}</span>
      <strong className="mt-0.5 block text-sm text-white">{value}</strong>
    </div>
  );
}

export function BallAnalysisReviewPanel({
  ballDetectorKey,
  detectorModels,
  detectionSummary,
  trackingResult,
  reviewCandidates,
  activeFrameCandidates,
  activePrimaryPoint,
  ballPhase,
  ballProgress,
  ballBusy,
  ballError,
  toggles,
  onToggleChange,
  onDetectorChange,
  onRun,
  selectedDetectorLabel,
  resolvedDetector,
  pipelineVersion
}: {
  ballDetectorKey: BallDetectorModelKey;
  detectorModels: BallDetectorModelOption[];
  detectionSummary: VideoBallDetectionSummary | null;
  trackingResult: VideoBallTrackingResultResponse | null;
  reviewCandidates: BallReviewCandidate[];
  activeFrameCandidates: BallReviewCandidate[];
  activePrimaryPoint: ReturnType<typeof import("@/lib/ball-analysis-review").activeFramePrimaryPoint>;
  ballPhase: BallAnalysisPhase;
  ballProgress: number;
  ballBusy: boolean;
  ballError: string | null;
  toggles: BallReviewDisplayToggles;
  onToggleChange: (key: keyof BallReviewDisplayToggles, value: boolean) => void;
  onDetectorChange: (key: BallDetectorModelKey) => void;
  onRun: (rerun: boolean) => void;
  selectedDetectorLabel: string;
  resolvedDetector: string | null;
  pipelineVersion: string;
}) {
  const summary = trackingResult?.summary;
  const acceptedCount = reviewCandidates.filter((candidate) => candidate.selected).length;
  const rejectedCount = reviewCandidates.length - acceptedCount;
  const observedCount = summary?.observed_track_points ?? 0;
  const reconstructedCount = (summary?.recovered_points ?? 0)
    + (summary?.physics_reconstructed_points ?? 0)
    + (summary?.projected_points ?? 0);
  const hasResults = Boolean(detectionSummary || trackingResult || reviewCandidates.length > 0);

  return (
    <aside className="flex min-w-[17rem] max-w-[22rem] flex-col gap-4 border border-white/10 bg-[#070d0a]/90 p-4 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-6rem)] xl:overflow-y-auto">
      <div>
        <p className="text-[10px] font-bold uppercase tracking-[0.14em] text-lime">Ball Tracking</p>
        <p className="mt-1 text-[11px] text-white/40">Pipeline {pipelineVersion}</p>
        <label className="mt-3 block">
          <span className="text-[10px] font-bold uppercase text-white/45">Detector</span>
          <select
            className="mt-1 w-full border border-white/15 bg-black/40 px-2.5 py-2 text-sm text-white"
            value={ballDetectorKey}
            disabled={ballBusy}
            onChange={(event) => onDetectorChange(event.target.value as BallDetectorModelKey)}
          >
            {detectorModels.length > 0 ? detectorModels.map((model) => (
              <option key={model.key} value={model.key} disabled={!model.available}>
                {model.display_name}{model.available ? "" : " (unavailable)"}
              </option>
            )) : (
              <option value="automatic">Automatic</option>
            )}
          </select>
        </label>
        <p className="mt-2 text-xs text-white/45">
          Selected: {selectedDetectorLabel}
          {resolvedDetector ? ` · Resolved: ${resolvedDetector}` : ""}
        </p>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button disabled={ballBusy} onClick={() => onRun(Boolean(trackingResult || ballError))}>
          {ballBusy
            ? PHASE_LABELS[ballPhase]
            : trackingResult || ballError
              ? "Rerun Detection + Tracking"
              : "Run Ball Analysis"}
        </Button>
      </div>

      {ballBusy ? (
        <div>
          <div className="flex items-center justify-between text-xs text-white/55">
            <span>{PHASE_LABELS[ballPhase]}</span>
            <span>{Math.round(ballProgress)}%</span>
          </div>
          <div className="mt-2 h-1.5 overflow-hidden bg-white/10">
            <div className="h-full bg-lime transition-[width]" style={{ width: `${Math.max(2, ballProgress)}%` }} />
          </div>
        </div>
      ) : null}

      {ballError ? (
        <p className="border border-signal/30 bg-signal/10 px-2.5 py-2 text-xs text-[#ffaaa6]">{ballError}</p>
      ) : null}

      {trackingResult?.status === "no_reliable_track" && !ballError ? (
        <p className="rounded-lg border border-[#ffca68]/30 bg-[#ffca68]/[0.05] px-2.5 py-2 text-xs leading-5 text-[#ffdc9a]">
          No coherent moving-ball track met the reliability threshold.
        </p>
      ) : null}

      {hasResults ? (
        <div className="grid grid-cols-2 gap-2">
          {metricCard("Raw candidates", detectionSummary?.total_candidates ?? summary?.raw_candidate_count ?? 0)}
          {metricCard("Accepted", acceptedCount)}
          {metricCard("Rejected", rejectedCount)}
          {metricCard("Track points", trackingResult?.primary_track.length ?? 0)}
          {metricCard("Observed", observedCount)}
          {metricCard("Reconstructed", reconstructedCount)}
          {metricCard("Quality", summary?.track_quality ?? "—")}
          {metricCard("Status", trackingResult?.status ?? "—")}
        </div>
      ) : null}

      <details className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
        <summary className="cursor-pointer text-xs font-bold text-white/55">Overlay display</summary>
        <div className="mt-2 space-y-1.5 text-xs text-white/70">
          {(Object.keys(toggles) as Array<keyof BallReviewDisplayToggles>).map((key) => (
            <label key={key} className="flex items-center gap-2">
              <input
                type="checkbox"
                className="accent-lime"
                checked={toggles[key]}
                onChange={(event) => onToggleChange(key, event.target.checked)}
              />
              <span>{toggleLabel(key)}</span>
            </label>
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-2 text-[10px] text-white/45">
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#50e650]" />Observed</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#ff9600]" />Recovered</span>
          <span><i className="mr-1 inline-block h-2 w-2 rounded-full bg-[#ffe600]" />Projected</span>
        </div>
      </details>

      {(activeFrameCandidates.length > 0 || activePrimaryPoint) ? (
        <details className="rounded-lg border border-white/10 bg-black/20 px-2.5 py-2">
          <summary className="cursor-pointer text-xs font-bold text-white/55">Active frame evidence</summary>
          <ul className="mt-2 space-y-1.5 text-[11px] text-white/65">
            {activeFrameCandidates.map((candidate) => (
              <li key={candidate.candidate_id}>
                {candidate.selected ? "accepted" : "rejected"} · {candidate.confidence.toFixed(2)} · {candidate.rejection_label}
              </li>
            ))}
            {activePrimaryPoint ? (
              <li className="text-lime/90">
                primary · {(activePrimaryPoint.detector_confidence ?? activePrimaryPoint.confidence).toFixed(2)} · {activePrimaryPoint.provenance.toLowerCase()}
              </li>
            ) : null}
          </ul>
        </details>
      ) : null}
    </aside>
  );
}

function toggleLabel(key: keyof BallReviewDisplayToggles) {
  const labels: Record<keyof BallReviewDisplayToggles, string> = {
    primaryTrack: "Primary track",
    acceptedCandidates: "Accepted candidates",
    rejectedCandidates: "Rejected candidates",
    detectionBoxes: "Detection boxes",
    reconstructedPoints: "Reconstructed points",
    completeTrail: "Complete trail"
  };
  return labels[key];
}
