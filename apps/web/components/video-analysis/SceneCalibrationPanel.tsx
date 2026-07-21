"use client";

import { useEffect, useMemo, useState } from "react";

import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { StatusBadge } from "@/components/ui/StatusBadge";
import {
  confirmVideoAnalysisCalibration,
  detectVideoAnalysisCalibration,
  type ConfirmedVideoCalibrationResponse,
  type NormalizedBox,
  type VideoAnalysisPreparedResponse,
  type VisualCalibrationDetectionDebug,
  type VisualCalibrationQuality,
  type WicketCalibration
} from "@/lib/api";

import { MEDIA_FIT_CLASS } from "./AnalysisMediaStage";
import {
  CalibrationCanvas,
  DEFAULT_VIDEO_GUIDES,
  calculateApproximatePitchGeometry,
  wicketDistanceWarning,
  wicketFromBox
} from "./CalibrationCanvas";


type CalibrationPhase =
  | "guides"
  | "detecting"
  | "review"
  | "saving"
  | "accepted"
  | "failed";


function guideOrDefault(
  guide: NormalizedBox | null | undefined,
  fallback: NormalizedBox
): NormalizedBox {
  return guide ?? fallback;
}


function phaseBadge(phase: CalibrationPhase): { label: string; tone: "neutral" | "good" | "warn" } {
  if (phase === "accepted") return { label: "Ready", tone: "good" };
  if (phase === "detecting" || phase === "saving") return { label: "Processing", tone: "neutral" };
  if (phase === "failed") return { label: "Needs Attention", tone: "warn" };
  if (phase === "review") return { label: "Needs Attention", tone: "warn" };
  return { label: "Ready", tone: "neutral" };
}


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
    initialCalibration ? "accepted" : "guides"
  );
  const [strikerGuide, setStrikerGuide] = useState<NormalizedBox>(
    guideOrDefault(initialCalibration?.striker_guide, DEFAULT_VIDEO_GUIDES.striker)
  );
  const [nonStrikerGuide, setNonStrikerGuide] = useState<NormalizedBox>(
    guideOrDefault(initialCalibration?.non_striker_guide, DEFAULT_VIDEO_GUIDES.non_striker)
  );
  const [striker, setStriker] = useState<WicketCalibration | null>(
    initialCalibration?.striker_wicket ?? null
  );
  const [nonStriker, setNonStriker] = useState<WicketCalibration | null>(
    initialCalibration?.non_striker_wicket ?? null
  );
  const [message, setMessage] = useState(
    initialCalibration?.message
      ?? "Drag guide boxes over far (striker) and near (non-striker) wickets, then Detect."
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
  const [failedEnds, setFailedEnds] = useState<Array<"striker" | "non_striker">>([]);
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
  // ponytail: keep guides movable until Accept so Redetect can reuse nudged ROIs.
  const canvasMode =
    phase === "accepted" || phase === "detecting" || phase === "saving"
      ? "locked"
      : "guides";
  const busy = phase === "detecting" || phase === "saving";
  const badge = phaseBadge(phase);

  useEffect(() => {
    if (!initialCalibration) {
      setPhase("guides");
      setStrikerGuide(DEFAULT_VIDEO_GUIDES.striker);
      setNonStrikerGuide(DEFAULT_VIDEO_GUIDES.non_striker);
      setStriker(null);
      setNonStriker(null);
      setMessage(
        "Drag guide boxes over far (striker) and near (non-striker) wickets, then Detect."
      );
      setQuality(null);
      setQualityReasons([]);
      setWarning(null);
      setFailedEnds([]);
      setSavedCalibration(null);
      setReferenceUrl(analysis.reference_frame_url);
      setReferenceFrameIndex(analysis.reference_frame_index);
      setImageWidth(analysis.width);
      setImageHeight(analysis.height);
      return;
    }
    setPhase("accepted");
    setStriker(initialCalibration.striker_wicket);
    setNonStriker(initialCalibration.non_striker_wicket);
    setStrikerGuide(
      guideOrDefault(initialCalibration.striker_guide, DEFAULT_VIDEO_GUIDES.striker)
    );
    setNonStrikerGuide(
      guideOrDefault(initialCalibration.non_striker_guide, DEFAULT_VIDEO_GUIDES.non_striker)
    );
    setMessage(initialCalibration.message);
    setQuality(initialCalibration.quality ?? "READY");
    setQualityReasons(initialCalibration.quality_reasons ?? []);
    setWarning(initialCalibration.assignment_warning ?? null);
    setFailedEnds([]);
    setModelPath(initialCalibration.model_path_used ?? null);
    setImageWidth(initialCalibration.image_width);
    setImageHeight(initialCalibration.image_height);
    setReferenceUrl(initialCalibration.reference_frame_url);
    setReferenceFrameIndex(initialCalibration.reference_frame_index);
    setSavedCalibration(initialCalibration);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysis.analysis_id, initialCalibration]);

  function handleGuideChange(label: "striker" | "non_striker", box: NormalizedBox) {
    if (label === "striker") setStrikerGuide(box);
    else setNonStrikerGuide(box);
    if (phase === "accepted") {
      setSavedCalibration(null);
      onDirty();
    }
  }

  async function runDetection() {
    setPhase("detecting");
    setError(null);
    setWarning(null);
    setFailedEnds([]);
    setSavedCalibration(null);
    onDirty();
    try {
      const result = await detectVideoAnalysisCalibration(
        analysis.analysis_id,
        {
          strikerGuide,
          nonStrikerGuide,
          refreshEarlyReference: false
        }
      );
      const nextStriker = result.provisional_striker_wicket ?? null;
      const nextNonStriker = result.provisional_non_striker_wicket ?? null;
      if (result.striker_guide) setStrikerGuide(result.striker_guide);
      if (result.non_striker_guide) setNonStrikerGuide(result.non_striker_guide);
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
      setFailedEnds(result.failed_ends ?? []);
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
    setStrikerGuide(nonStrikerGuide);
    setNonStrikerGuide(strikerGuide);
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
          corridor_width_multiplier: 1,
          striker_guide: strikerGuide,
          non_striker_guide: nonStrikerGuide
        }
      );
      setStriker(confirmed.striker_wicket);
      setNonStriker(confirmed.non_striker_wicket);
      setStrikerGuide(
        guideOrDefault(confirmed.striker_guide, strikerGuide)
      );
      setNonStrikerGuide(
        guideOrDefault(confirmed.non_striker_guide, nonStrikerGuide)
      );
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

  const failedEndsLabel = failedEnds.length === 0
    ? null
    : failedEnds.map((end) => (end === "striker" ? "striker" : "non-striker")).join(" and ");

  return (
    <Card className="border-[#ffca68]/25 p-4 sm:p-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-lg font-black tracking-tight sm:text-xl">Scene Calibration</h2>
            <StatusBadge label={badge.label} tone={badge.tone} />
            {quality && <StatusBadge label={quality} tone={qualityTone} />}
          </div>
          <p className="mt-1 text-sm text-white/45">
            Place guides · detect wickets · accept overlay for this video.
          </p>
        </div>
      </div>

      {error && (
        <p className="mt-3 rounded-lg border border-signal/30 bg-signal/10 px-3 py-2 text-sm text-[#ffaaa6]">
          {error}
        </p>
      )}
      {displayWarning && (
        <p className="mt-3 rounded-lg border border-[#ffca68]/35 bg-[#ffca68]/10 px-3 py-2 text-sm text-[#ffe0a3]">
          {displayWarning}
        </p>
      )}

      <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(15rem,20rem)]">
        <div className="min-w-0">
          <CalibrationCanvas
            imageUrl={referenceUrl}
            imageWidth={imageWidth}
            imageHeight={imageHeight}
            striker={striker}
            nonStriker={nonStriker}
            strikerGuide={strikerGuide}
            nonStrikerGuide={nonStrikerGuide}
            pitchGeometry={pitchGeometry}
            interactionMode={canvasMode}
            onGuideChange={handleGuideChange}
          />
        </div>

        <aside className="flex min-w-0 flex-col gap-3 xl:sticky xl:top-4 xl:max-h-[calc(100dvh-5rem)] xl:self-start xl:overflow-y-auto">
          <div className="rounded-xl border border-white/10 bg-black/25 p-3">
            <p className="text-sm leading-5 text-white/65">{message}</p>
            <div className="mt-3 flex flex-wrap gap-3 text-xs font-semibold">
              <span className={striker ? "text-lime" : "text-white/35"}>
                Striker {striker ? "✓" : phase === "detecting" ? "…" : failedEnds.includes("striker") ? "✗" : "—"}
              </span>
              <span className={nonStriker ? "text-lime" : "text-white/35"}>
                Non-Striker {nonStriker ? "✓" : phase === "detecting" ? "…" : failedEnds.includes("non_striker") ? "✗" : "—"}
              </span>
            </div>
            {failedEndsLabel && (
              <p className="mt-2 text-xs font-semibold text-[#ffaaa6]">
                Failed: {failedEndsLabel}. Adjust guides and Redetect.
              </p>
            )}
            <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-white/35">
              <span>Frame {referenceFrameIndex}</span>
              {(phase !== "guides" || candidateCount > 0) && (
                <span>{candidateCount} candidate{candidateCount === 1 ? "" : "s"}</span>
              )}
            </div>
            {qualityReasons.length > 0 && (
              <ul className="mt-2 list-disc space-y-0.5 pl-4 text-[11px] text-white/40">
                {qualityReasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            )}
          </div>

          <div className="flex flex-col gap-2">
            {phase === "guides" || phase === "detecting" ? (
              <Button
                className="w-full"
                disabled={busy}
                onClick={() => void runDetection()}
              >
                {phase === "detecting" ? "Detecting…" : "Detect Wickets"}
              </Button>
            ) : (
              <Button
                className="w-full"
                disabled={!canAccept || busy}
                onClick={() => void acceptCalibration()}
              >
                {phase === "saving"
                  ? "Accepting…"
                  : phase === "accepted"
                    ? "Accepted"
                    : "Accept"}
              </Button>
            )}
            {(phase === "review" || phase === "failed" || phase === "accepted" || phase === "saving") && (
              <>
                <Button
                  variant="secondary"
                  className="w-full"
                  disabled={busy}
                  onClick={() => void runDetection()}
                >
                  Redetect
                </Button>
                <Button
                  variant="secondary"
                  className="w-full"
                  disabled={!bothWicketsReady || busy}
                  onClick={swapWicketEnds}
                >
                  Swap Wicket Ends
                </Button>
              </>
            )}
          </div>

          {detectionDebug && (
            <details className="rounded-lg border border-white/10 bg-black/30 p-3 text-xs text-white/45">
              <summary className="cursor-pointer font-semibold text-white/55">
                Advanced diagnostics
              </summary>
              <div className="mt-3 space-y-2">
                <p>
                  Striker: {striker?.detection_pass ?? "—"}
                  {" · "}
                  Non-striker: {nonStriker?.detection_pass ?? "—"}
                  {modelPath ? ` · ${modelPath}` : ""}
                </p>
                {detectionDebug.debug_overlay_url && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    className={MEDIA_FIT_CLASS}
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

          {savedCalibration && phase === "accepted" && (
            <details className="rounded-xl border border-lime/20 bg-lime/[0.04] p-3">
              <summary className="cursor-pointer text-sm font-bold text-lime">
                Calibration accepted
              </summary>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-2">
                <a
                  className="text-[11px] font-bold text-lime underline"
                  href={savedCalibration.calibration_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open calibration JSON
                </a>
              </div>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                className={`${MEDIA_FIT_CLASS} mt-3`}
                src={`${savedCalibration.calibration_overlay_url}?v=${encodeURIComponent(savedCalibration.updated_at)}`}
                alt="Accepted guided scene calibration overlay"
              />
              {savedCalibration.scene_overlay_status === "ready" && savedCalibration.scene_overlay_url && (
                <video
                  className={`${MEDIA_FIT_CLASS} mt-3`}
                  src={`${savedCalibration.scene_overlay_url}?v=${encodeURIComponent(savedCalibration.updated_at)}`}
                  controls
                  playsInline
                  muted
                />
              )}
              {savedCalibration.scene_overlay_status === "failed" && (
                <p className="mt-2 text-[11px] text-white/40">
                  Scene overlay video unavailable; still overlay shown above.
                </p>
              )}
            </details>
          )}
        </aside>
      </div>
    </Card>
  );
}
