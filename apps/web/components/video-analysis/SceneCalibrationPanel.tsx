"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  confirmVideoAnalysisCalibration,
  detectVideoAnalysisCalibration,
  type ConfirmedVideoCalibrationResponse,
  type VideoAnalysisPreparedResponse,
  type VisualCalibrationDetectionDebug,
  type VisualCalibrationQuality,
  type WicketCalibration
} from "@/lib/api";

import {
  CalibrationCanvas,
  calculateApproximatePitchGeometry,
  wicketDistanceWarning,
  wicketFromBox
} from "./CalibrationCanvas";


type CalibrationPhase =
  | "detecting"
  | "review"
  | "saving"
  | "accepted"
  | "failed";


export function SceneCalibrationPanel({
  analysis,
  initialCalibration,
  onCalibrated,
  onDirty,
  onReferenceFrameUpdated
}: {
  analysis: VideoAnalysisPreparedResponse;
  initialCalibration: ConfirmedVideoCalibrationResponse | null;
  onCalibrated: (calibration: ConfirmedVideoCalibrationResponse) => void;
  onDirty: () => void;
  onReferenceFrameUpdated?: (referenceFrameIndex: number, referenceFrameUrl: string) => void;
}) {
  const [phase, setPhase] = useState<CalibrationPhase>(
    initialCalibration ? "accepted" : "detecting"
  );
  const [striker, setStriker] = useState<WicketCalibration | null>(
    initialCalibration?.striker_wicket ?? null
  );
  const [nonStriker, setNonStriker] = useState<WicketCalibration | null>(
    initialCalibration?.non_striker_wicket ?? null
  );
  const [message, setMessage] = useState(
    initialCalibration?.message
      ?? "Detecting wickets on the early reference frame…"
  );
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(
    initialCalibration?.assignment_warning ?? null
  );
  const [quality, setQuality] = useState<VisualCalibrationQuality | null>(
    initialCalibration?.quality ?? null
  );
  const [qualityReasons, setQualityReasons] = useState<string[]>(
    initialCalibration?.quality_reasons ?? []
  );
  const [candidateCount, setCandidateCount] = useState(0);
  const [modelPath, setModelPath] = useState<string | null>(
    initialCalibration?.model_path_used ?? null
  );
  const [imageWidth, setImageWidth] = useState(
    initialCalibration?.image_width ?? analysis.width
  );
  const [imageHeight, setImageHeight] = useState(
    initialCalibration?.image_height ?? analysis.height
  );
  const [referenceUrl, setReferenceUrl] = useState(
    initialCalibration?.reference_frame_url ?? analysis.reference_frame_url
  );
  const [referenceFrameIndex, setReferenceFrameIndex] = useState(
    initialCalibration?.reference_frame_index ?? analysis.reference_frame_index
  );
  const [savedCalibration, setSavedCalibration] = useState(initialCalibration);
  const [detectionDebug, setDetectionDebug] = useState<VisualCalibrationDetectionDebug | null>(null);

  const pitchGeometry = useMemo(
    () => calculateApproximatePitchGeometry(striker, nonStriker, 1),
    [striker, nonStriker]
  );
  const proximityWarning = wicketDistanceWarning(striker, nonStriker);
  const displayWarning = proximityWarning ?? warning;
  const bothWicketsReady = striker !== null && nonStriker !== null;
  const canAccept = bothWicketsReady && quality !== "FAILED" && !proximityWarning;

  useEffect(() => {
    if (initialCalibration) {
      setPhase("accepted");
      setStriker(initialCalibration.striker_wicket);
      setNonStriker(initialCalibration.non_striker_wicket);
      setMessage(initialCalibration.message);
      setQuality(initialCalibration.quality ?? "READY");
      setQualityReasons(initialCalibration.quality_reasons ?? []);
      setWarning(initialCalibration.assignment_warning ?? null);
      setModelPath(initialCalibration.model_path_used ?? null);
      setImageWidth(initialCalibration.image_width);
      setImageHeight(initialCalibration.image_height);
      setReferenceUrl(initialCalibration.reference_frame_url);
      setReferenceFrameIndex(initialCalibration.reference_frame_index);
      setSavedCalibration(initialCalibration);
      return;
    }
    // Auto-detect only for a fresh prepared analysis (no accepted calibration yet).
    void runDetection(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis.analysis_id]);

  useEffect(() => {
    if (!initialCalibration) return;
    setPhase("accepted");
    setStriker(initialCalibration.striker_wicket);
    setNonStriker(initialCalibration.non_striker_wicket);
    setMessage(initialCalibration.message);
    setQuality(initialCalibration.quality ?? "READY");
    setQualityReasons(initialCalibration.quality_reasons ?? []);
    setWarning(initialCalibration.assignment_warning ?? null);
    setModelPath(initialCalibration.model_path_used ?? null);
    setImageWidth(initialCalibration.image_width);
    setImageHeight(initialCalibration.image_height);
    setReferenceUrl(initialCalibration.reference_frame_url);
    setReferenceFrameIndex(initialCalibration.reference_frame_index);
    setSavedCalibration(initialCalibration);
  }, [initialCalibration]);

  async function runDetection(refreshEarlyReference: boolean) {
    setPhase("detecting");
    setError(null);
    setWarning(null);
    setSavedCalibration(null);
    onDirty();
    try {
      const result = await detectVideoAnalysisCalibration(
        analysis.analysis_id,
        { refreshEarlyReference }
      );
      const nextStriker = result.provisional_striker_wicket ?? null;
      const nextNonStriker = result.provisional_non_striker_wicket ?? null;
      setStriker(nextStriker);
      setNonStriker(nextNonStriker);
      setCandidateCount(result.candidates.length);
      setModelPath(result.model_path_used);
      setImageWidth(result.image_width);
      setImageHeight(result.image_height);
      setReferenceUrl(
        `${result.reference_frame_url}${result.reference_frame_url.includes("?") ? "&" : "?"}v=${result.reference_frame_index}-${Date.now()}`
      );
      setReferenceFrameIndex(result.reference_frame_index);
      onReferenceFrameUpdated?.(
        result.reference_frame_index,
        result.reference_frame_url
      );
      setMessage(result.message);
      setWarning(result.assignment_warning ?? result.warning ?? null);
      setQuality(result.quality ?? "FAILED");
      setQualityReasons(result.quality_reasons ?? []);
      setDetectionDebug(result.detection_debug ?? null);
      setPhase(
        nextStriker && nextNonStriker && result.quality !== "FAILED"
          ? "review"
          : "failed"
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Wicket detection failed.");
      setQuality("FAILED");
      setPhase("failed");
    }
  }

  function swapWicketEnds() {
    if (!striker || !nonStriker) return;
    setStriker(wicketFromBox(
      "striker",
      nonStriker.source,
      nonStriker.confidence ?? null,
      nonStriker.box,
      nonStriker.detection_pass
    ));
    setNonStriker(wicketFromBox(
      "non_striker",
      striker.source,
      striker.confidence ?? null,
      striker.box,
      striker.detection_pass
    ));
    setSavedCalibration(null);
    onDirty();
    setPhase("review");
    setMessage("Wicket ends swapped. Review the overlay, then Accept.");
  }

  async function acceptCalibration() {
    if (!striker || !nonStriker) {
      setError("Both wickets must be detected before accepting calibration.");
      return;
    }
    if (proximityWarning) {
      setError(proximityWarning);
      return;
    }
    if (quality === "FAILED") {
      setError("Calibration quality is FAILED. Press Redetect first.");
      return;
    }
    setPhase("saving");
    setError(null);
    try {
      const confirmed = await confirmVideoAnalysisCalibration(
        analysis.analysis_id,
        {
          analysis_id: analysis.analysis_id,
          striker_wicket: {
            label: "striker",
            source: striker.source,
            confidence: striker.confidence ?? null,
            box: striker.box,
            detection_pass: striker.detection_pass ?? null
          },
          non_striker_wicket: {
            label: "non_striker",
            source: nonStriker.source,
            confidence: nonStriker.confidence ?? null,
            box: nonStriker.box,
            detection_pass: nonStriker.detection_pass ?? null
          },
          corridor_width_multiplier: 1
        }
      );
      setStriker(confirmed.striker_wicket);
      setNonStriker(confirmed.non_striker_wicket);
      setMessage(confirmed.message);
      setQuality(confirmed.quality ?? "READY");
      setQualityReasons(confirmed.quality_reasons ?? []);
      setWarning(confirmed.assignment_warning ?? null);
      setSavedCalibration(confirmed);
      setPhase("accepted");
      onCalibrated(confirmed);
    } catch (caught) {
      setError(
        caught instanceof Error
          ? caught.message
          : "Calibration could not be accepted."
      );
      setPhase("review");
    }
  }

  const qualityTone = quality === "READY"
    ? "good"
    : quality === "WEAK"
      ? "warn"
      : "warn";

  return (
    <Card className="border-[#ffca68]/25">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <StatusBadge
            label={
              phase === "accepted"
                ? "Calibration Accepted"
                : phase === "detecting"
                  ? "Detecting Wickets"
                  : "Calibration Detected"
            }
            tone={phase === "accepted" ? "good" : "neutral"}
          />
          <h2 className="mt-4 text-2xl font-black">Automatic Visual Calibration</h2>
          <p className="mt-3 max-w-3xl text-sm leading-6 text-white/55">
            Approximate 2D scene calibration from an early reference frame.
            Soft pitch context only — not precise metric 3D / DRS / Hawk-Eye.
          </p>
        </div>
        {quality && (
          <StatusBadge label={`Quality ${quality}`} tone={qualityTone} />
        )}
      </div>

      {error && (
        <p className="mt-5 rounded-xl border border-signal/30 bg-signal/10 p-4 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}
      <div className="mt-5 rounded-xl border border-white/10 bg-black/20 p-4">
        <p className="text-sm leading-6 text-white/65">{message}</p>
        <div className="mt-3 flex flex-wrap gap-3 text-sm font-semibold">
          <span className={striker ? "text-lime" : "text-white/35"}>
            Striker {striker ? "✅" : phase === "detecting" ? "…" : "—"}
          </span>
          <span className={nonStriker ? "text-lime" : "text-white/35"}>
            Non-Striker {nonStriker ? "✅" : phase === "detecting" ? "…" : "—"}
          </span>
        </div>
        <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1 text-xs text-white/35">
          <span>Reference frame {referenceFrameIndex}</span>
          <span>{candidateCount} detector candidate{candidateCount === 1 ? "" : "s"}</span>
          {modelPath && <span>Model: {modelPath}</span>}
        </div>
        {qualityReasons.length > 0 && (
          <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-white/45">
            {qualityReasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        )}
        {detectionDebug && (
          <details className="mt-4 rounded-lg border border-white/10 bg-black/30 p-3 text-xs text-white/45">
            <summary className="cursor-pointer font-semibold text-white/55">
              Developer detection debug ({detectionDebug.pass_count} passes)
            </summary>
            <div className="mt-3 space-y-2">
              <p>
                Striker source: {striker?.detection_pass ?? "—"}
                {" · "}
                Non-striker source: {nonStriker?.detection_pass ?? "—"}
              </p>
              {detectionDebug.debug_overlay_url && (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  className="mt-2 h-auto w-full rounded-md bg-black object-contain"
                  src={detectionDebug.debug_overlay_url}
                  alt="Wicket detection debug overlay"
                />
              )}
              {detectionDebug.debug_json_url && (
                <a
                  className="inline-block text-lime underline"
                  href={detectionDebug.debug_json_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open detection debug JSON
                </a>
              )}
            </div>
          </details>
        )}
      </div>
      {displayWarning && (
        <p className="mt-4 rounded-xl border border-[#ffca68]/35 bg-[#ffca68]/10 p-4 text-sm text-[#ffe0a3]">
          {displayWarning}
        </p>
      )}

      <div className="mt-6">
        <CalibrationCanvas
          imageUrl={referenceUrl}
          imageWidth={imageWidth}
          imageHeight={imageHeight}
          striker={striker}
          nonStriker={nonStriker}
          pitchGeometry={pitchGeometry}
          readOnly
          disabled={phase === "detecting" || phase === "saving"}
        />
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        <Button
          disabled={!canAccept || phase === "detecting" || phase === "saving"}
          onClick={() => void acceptCalibration()}
        >
          {phase === "saving"
            ? "Accepting…"
            : phase === "accepted"
              ? "Accepted"
              : "Accept"}
        </Button>
        <Button
          variant="secondary"
          disabled={phase === "detecting" || phase === "saving"}
          onClick={() => void runDetection(true)}
        >
          {phase === "detecting" ? "Detecting…" : "Redetect"}
        </Button>
        <Button
          variant="secondary"
          disabled={!bothWicketsReady || phase === "detecting" || phase === "saving"}
          onClick={swapWicketEnds}
        >
          Swap Wicket Ends
        </Button>
      </div>

      {savedCalibration && (
        <div className="mt-7 rounded-xl border border-lime/20 bg-lime/[0.04] p-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-black text-lime">Automatic visual calibration accepted</p>
              <p className="mt-1 text-xs text-white/40">
                Approximate wicket base references use bbox bottom-centres.
                Soft scene context for later tracking — not a hard ball-rejection boundary.
              </p>
            </div>
            <a
              className="text-xs font-bold text-lime underline"
              href={savedCalibration.calibration_url}
              target="_blank"
              rel="noreferrer"
            >
              Open calibration JSON
            </a>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            className="mt-4 h-auto w-full rounded-lg bg-black object-contain"
            src={`${savedCalibration.calibration_overlay_url}?v=${encodeURIComponent(savedCalibration.updated_at)}`}
            alt="Accepted automatic visual calibration overlay"
          />
        </div>
      )}
    </Card>
  );
}
